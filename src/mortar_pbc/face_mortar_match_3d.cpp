// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-B — broad-phase candidate-pair enumeration.
// See face_mortar_match_3d.hpp for the public API and rationale.

#include "face_mortar_match_3d.hpp"
#include "utilities/unified_logger.hpp"

#include "axom/core.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"

#include "mfem.hpp"
#include "utilities/mechanics_log.hpp"

#include <algorithm>
#include <cmath>

namespace mortar_pbc
{

namespace
{

using Point2D = axom::primal::Point<double, 2>;
using BBox2D  = axom::primal::BoundingBox<double, 2>;
using BVH2D   = axom::spin::BVH<2>;

/// Convert a perpendicular-axis name ("x" / "y" / "z") into the two
/// 2D-projection column indices (a_idx, b_idx) such that the 2D coords
/// are (coords[v, a_idx], coords[v, b_idx]). Cyclic ordering preserves
/// right-handedness:
///   "x" -> (1, 2) i.e. (y, z)
///   "y" -> (2, 0) i.e. (z, x)
///   "z" -> (0, 1) i.e. (x, y)
inline std::pair<int, int> ProjectionAxes(const std::string& perpendicular_axis)
{
    if (perpendicular_axis == "x") { return {1, 2}; }
    if (perpendicular_axis == "y") { return {2, 0}; }
    if (perpendicular_axis == "z") { return {0, 1}; }
    MFEM_ABORT("ProjectionAxes: unknown perpendicular_axis '"
               << perpendicular_axis << "'; expected one of {x, y, z}.");
    return {-1, -1};  // unreachable
}

/// Compute a per-element 2D AABB from the (n_nodes × 3) coords of a
/// face element. Returns a primal::BoundingBox<double, 2>.
template <typename ElementT>
BBox2D ComputeElement2DBBox(const ElementT& elem, int a_idx, int b_idx)
{
    BBox2D bb;
    const int n_nodes = ElementT::NumNodes();
    for (int v = 0; v < n_nodes; ++v)
    {
        bb.addPoint(Point2D{elem.coords(v, a_idx), elem.coords(v, b_idx)});
    }
    return bb;
}

/// Compute the maximum 2D edge length across all elements. Used to
/// scale the relative AABB pad into an absolute distance.
template <typename ElementT>
double MaxEdgeLength2D(const std::vector<ElementT>& elems, int a_idx, int b_idx)
{
    double max_len = 0.0;
    for (const auto& e : elems)
    {
        const int n_nodes = ElementT::NumNodes();
        for (int v = 0; v < n_nodes; ++v)
        {
            const int w = (v + 1) % n_nodes;
            const double da = e.coords(w, a_idx) - e.coords(v, a_idx);
            const double db = e.coords(w, b_idx) - e.coords(v, b_idx);
            const double len = std::sqrt(da * da + db * db);
            max_len = std::max(max_len, len);
        }
    }
    return max_len;
}

/// Templated implementation shared by quad and tri overloads. Builds
/// the 2D BVH on the mortar elements and queries it with each
/// nonmortar element's 2D AABB. Output is in CSR format that mirrors
/// Axom's `BVH::findBoundingBoxes` convention.
///
/// **Axom v0.14 API contract** (verified empirically — first attempt
/// got this wrong and Axom fired a SLIC error):
///   * `offsets` and `counts` are `ArrayView<IndexType>` and are
///     INPUT/OUTPUT — caller must pre-allocate them with size
///     `n_query`. Axom writes to them but does NOT resize them.
///   * `candidates` is `Array<IndexType>` and is purely OUTPUT —
///     Axom allocates and fills.
///   * `offsets` has size `n_query` (NOT `n_query+1`); there is no
///     sentinel. To get the total candidate count use
///     `candidates.size()` (or equivalently `offsets[n-1] +
///     counts[n-1]`).
///
/// We translate the Axom output into our `std::vector`-based
/// `ClippedPairCandidates` struct at the end so downstream code
/// doesn't have an Axom-owned dependency on the result. We also
/// add a sentinel `offsets[n_nonmortar] = candidates.size()` to our
/// std::vector form because the SciPy-style CSR convention is more
/// natural for the iteration patterns we'll use in Batch 4.4-C
/// (`for k in [offsets[s], offsets[s+1])`).
template <typename ElementT>
ClippedPairCandidates MatchClippedFacePairsImpl(
    const std::vector<ElementT>& nonmortar_elems,
    const std::vector<ElementT>& mortar_elems,
    const std::string& perpendicular_axis,
    double aabb_pad_rel)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::MatchClippedFacePairs");

    // ---- Sanity checks ----
    MFEM_VERIFY(!perpendicular_axis.empty(),
                "MatchClippedFacePairs: perpendicular_axis must be set.");
    for (const auto& e : nonmortar_elems)
    {
        MFEM_VERIFY(e.perpendicular_axis == perpendicular_axis,
                    "MatchClippedFacePairs: nonmortar element has "
                    "perpendicular_axis '" << e.perpendicular_axis
                    << "' but caller passed '" << perpendicular_axis << "'.");
    }
    for (const auto& e : mortar_elems)
    {
        MFEM_VERIFY(e.perpendicular_axis == perpendicular_axis,
                    "MatchClippedFacePairs: mortar element has "
                    "perpendicular_axis '" << e.perpendicular_axis
                    << "' but caller passed '" << perpendicular_axis << "'.");
    }

    const axom::IndexType n_nonmortar =
        static_cast<axom::IndexType>(nonmortar_elems.size());
    const axom::IndexType n_mortar =
        static_cast<axom::IndexType>(mortar_elems.size());

    // Empty edge cases — return all-zero CSR with single sentinel.
    ClippedPairCandidates result;
    result.offsets.assign(n_nonmortar + 1, 0);
    result.counts.assign(n_nonmortar, 0);
    if (n_nonmortar == 0 || n_mortar == 0) { return result; }

    // ---- Build 2D AABBs ----
    const auto axes = ProjectionAxes(perpendicular_axis);
    const int a_idx = axes.first;
    const int b_idx = axes.second;

    // Pad the mortar AABBs by aabb_pad_rel * max_mortar_edge_length to
    // tolerate exact-vertex-on-edge cases. The 1e-9 default matches
    // the architecture doc §3.6 vertex-matching tolerance.
    const double mortar_max_edge = MaxEdgeLength2D(mortar_elems, a_idx, b_idx);
    const double pad = aabb_pad_rel * mortar_max_edge;

    std::vector<BBox2D> mortar_bboxes(static_cast<std::size_t>(n_mortar));
    for (axom::IndexType m = 0; m < n_mortar; ++m)
    {
        mortar_bboxes[m] = ComputeElement2DBBox(mortar_elems[m], a_idx, b_idx);
        if (pad > 0.0) { mortar_bboxes[m].expand(pad); }
    }

    // ---- Build the BVH on mortar AABBs ----
    BVH2D bvh;
    {
        CALI_CXX_MARK_SCOPE("mortar_pbc::MatchClippedFacePairs::bvh_init");
        const int status = bvh.initialize(mortar_bboxes.data(), n_mortar);
        MFEM_VERIFY(status == 0,
                    "MatchClippedFacePairs: BVH::initialize returned non-zero "
                    "status: " << status);
    }

    // ---- Build nonmortar query AABBs ----
    std::vector<BBox2D> query_bboxes(static_cast<std::size_t>(n_nonmortar));
    for (axom::IndexType s = 0; s < n_nonmortar; ++s)
    {
        query_bboxes[s] = ComputeElement2DBBox(nonmortar_elems[s], a_idx, b_idx);
        // No pad on queries — the mortar pad already covers slop.
    }

    // ---- Query the BVH ----
    //
    // Per Axom v0.14 API (verified by SLIC error message in the first
    // attempt — "offsets length not equal to numObjs"):
    //   * `ax_offsets` and `ax_counts` are caller-allocated `Array<IndexType>`
    //     of size n_nonmortar (NOT n_nonmortar+1). Axom writes results into
    //     them but does NOT resize.
    //   * `ax_candidates` is purely output; Axom allocates+fills it.
    //   * The `findBoundingBoxes` overload takes `ArrayView<IndexType>`
    //     for offsets/counts (so caller controls allocation) and
    //     `Array<IndexType>&` for candidates.
    axom::Array<axom::IndexType> ax_offsets(n_nonmortar);
    axom::Array<axom::IndexType> ax_counts(n_nonmortar);
    axom::Array<axom::IndexType> ax_candidates;
    {
        CALI_CXX_MARK_SCOPE("mortar_pbc::MatchClippedFacePairs::bvh_query");
        bvh.findBoundingBoxes(ax_offsets.view(), ax_counts.view(),
                              ax_candidates,
                              n_nonmortar, query_bboxes.data());
    }

    // ---- Translate Axom output into our SciPy-style std::vector CSR ----
    //
    // Axom convention:    offsets[s] = start of candidates for query s
    //                     counts[s]  = number of candidates for query s
    //                     no sentinel
    // Our convention:     offsets[s] = start of candidates for query s
    //                     offsets[n] = total candidate count (sentinel)
    //                     counts[s]  = same as Axom
    // The sentinel makes `for k in [offsets[s], offsets[s+1])` work
    // uniformly across the whole array without special-casing the
    // last query, which is what Batches 4.4-C and 4.4-D will iterate
    // with.
    result.offsets.resize(static_cast<std::size_t>(n_nonmortar + 1));
    result.counts.resize(static_cast<std::size_t>(n_nonmortar));
    for (axom::IndexType s = 0; s < n_nonmortar; ++s)
    {
        result.offsets[s] = ax_offsets[s];
        result.counts[s]  = ax_counts[s];
    }
    result.offsets[n_nonmortar] =
        static_cast<axom::IndexType>(ax_candidates.size());

    const axom::IndexType n_total = result.offsets[n_nonmortar];
    result.candidates.resize(static_cast<std::size_t>(n_total));
    for (axom::IndexType k = 0; k < n_total; ++k)
    {
        result.candidates[k] = ax_candidates[k];
    }

    return result;
}

// ============================================================================
// Fine-phase clipping + fan-triangulation (Batch 4.4-C).
// ============================================================================

using Polygon2D = axom::primal::Polygon<double, 2>;

/// Build an Axom Polygon<double, 2> from a face element by 2D-projecting
/// its vertices via the (a_idx, b_idx) column selection. The polygon is
/// then **CCW-corrected**: Sutherland-Hodgman clipping (which Axom's
/// primal::clip implements) requires CCW orientation on both subject and
/// clipper to interpret the inside half-plane correctly. Two CW inputs
/// silently produce empty output.
///
/// Why we can't rely on the upstream face-element convention to give us
/// CCW:
///   1. The face-element docstring says "CCW from the outward normal of
///      the nonmortar face." But the mortar face's outward normal points
///      OPPOSITE to the nonmortar's (they're on opposite sides of the
///      periodic interface). After 2D projection into a single (a, b)
///      plane, the nonmortar comes out CCW and the mortar CW (or vice
///      versa) — even though both are CCW in their own 3D frame.
///   2. Test data (`MakeQuadOnY`) uses uniform vertex ordering for both
///      sides. After cyclic 2D projection that's CW — also a CW input.
///
/// So `BuildPolygon2D` always inspects the signed 2D area and calls
/// `reverseOrientation()` if it's negative. After this, both subject and
/// clipper are CCW, and clip works correctly. The fan-triangulation step
/// downstream then assumes CCW input (`sa > 0`) and asserts on it — that
/// assertion is the safety net catching any future regression here.
template <typename ElementT>
Polygon2D BuildPolygon2D(const ElementT& elem, int a_idx, int b_idx)
{
    Polygon2D poly;
    const int n_nodes = ElementT::NumNodes();
    for (int v = 0; v < n_nodes; ++v)
    {
        poly.addVertex(Point2D{elem.coords(v, a_idx), elem.coords(v, b_idx)});
    }

    // Compute signed 2D area via shoelace; reverse if CW.
    double sa = 0.0;
    for (int v = 0; v < n_nodes; ++v)
    {
        const int w = (v + 1) % n_nodes;
        sa += poly[v][0] * poly[w][1] - poly[w][0] * poly[v][1];
    }
    if (sa < 0.0) { poly.reverseOrientation(); }
    return poly;
}

/// Signed 2D area of a triangle (v0, v1, v2). Positive iff CCW.
inline double SignedArea2D(const Point2D& v0,
                           const Point2D& v1,
                           const Point2D& v2)
{
    const double ux = v1[0] - v0[0];
    const double uy = v1[1] - v0[1];
    const double vx = v2[0] - v0[0];
    const double vy = v2[1] - v0[1];
    return 0.5 * (ux * vy - uy * vx);
}

/// 2D area of an axis-aligned face element from its 4 (or 3) projected
/// vertices. Used as the reference scale for area_tol_rel.
template <typename ElementT>
double Element2DArea(const ElementT& elem, int a_idx, int b_idx)
{
    const int n_nodes = ElementT::NumNodes();
    // Shoelace formula:
    double area = 0.0;
    for (int v = 0; v < n_nodes; ++v)
    {
        const int w = (v + 1) % n_nodes;
        area += elem.coords(v, a_idx) * elem.coords(w, b_idx);
        area -= elem.coords(w, a_idx) * elem.coords(v, b_idx);
    }
    return 0.5 * std::abs(area);
}

/// Templated implementation of fine-phase clipping. Applies to both
/// quad-quad and tri-tri pairings (the templating is on the element
/// type only — the Axom Polygon construction handles arbitrary
/// vertex counts).
template <typename ElementT>
ClippedSubTriangulation ClipFacePairsImpl(
    const std::vector<ElementT>& nonmortar_elems,
    const std::vector<ElementT>& mortar_elems,
    const ClippedPairCandidates& candidates,
    const std::string& perpendicular_axis,
    double area_tol_rel)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::ClipFacePairs");

    // ---- Sanity checks ----
    MFEM_VERIFY(!perpendicular_axis.empty(),
                "ClipFacePairs: perpendicular_axis must be set.");
    const axom::IndexType n_nonmortar =
        static_cast<axom::IndexType>(nonmortar_elems.size());
    MFEM_VERIFY(static_cast<axom::IndexType>(candidates.counts.size()) == n_nonmortar,
                "ClipFacePairs: candidates.counts.size() != n_nonmortar.");
    MFEM_VERIFY(static_cast<axom::IndexType>(candidates.offsets.size())
                    == n_nonmortar + 1,
                "ClipFacePairs: candidates.offsets.size() != n_nonmortar + 1.");

    ClippedSubTriangulation result;
    result.offsets.assign(static_cast<std::size_t>(n_nonmortar + 1), 0);
    result.counts.assign(static_cast<std::size_t>(n_nonmortar), 0);

    if (n_nonmortar == 0) { return result; }

    const auto axes = ProjectionAxes(perpendicular_axis);
    const int a_idx = axes.first;
    const int b_idx = axes.second;

    // ---- Walk candidates, clip, fan-triangulate ----
    //
    // Outer loop: each nonmortar element s. Build its polygon once,
    // walk its candidate list, clip against each mortar partner.
    //
    // axom::primal::clip(subject, clipper) returns the intersection
    // polygon (CCW). For convex-on-convex the order of subject vs
    // clipper doesn't matter for the *set*, but we pass nonmortar as
    // subject to keep the convention "nonmortar is the one being
    // restricted to the mortar." The default eps tolerance (1e-12) is
    // fine for our use.
    for (axom::IndexType s = 0; s < n_nonmortar; ++s)
    {
        const ElementT& s_elem = nonmortar_elems[s];
        const Polygon2D s_poly = BuildPolygon2D(s_elem, a_idx, b_idx);

        const double s_area = Element2DArea(s_elem, a_idx, b_idx);
        const double area_tol_abs = area_tol_rel * s_area;

        const axom::IndexType k_lo = candidates.offsets[s];
        const axom::IndexType k_hi = candidates.offsets[s + 1];
        for (axom::IndexType k = k_lo; k < k_hi; ++k)
        {
            const axom::IndexType m = candidates.candidates[k];
            const ElementT& m_elem = mortar_elems[m];
            const Polygon2D m_poly = BuildPolygon2D(m_elem, a_idx, b_idx);

            const Polygon2D clip_poly = axom::primal::clip(s_poly, m_poly);
            const int n_verts = clip_poly.numVertices();
            if (n_verts < 3) { continue; }  // empty / shared-edge / degenerate

            // Fan-triangulate from vertex 0:
            //   tri_i = (v_0, v_{i+1}, v_{i+2}) for i in [0, n_verts-3].
            for (int i = 0; i + 2 < n_verts; ++i)
            {
                const Point2D& v0 = clip_poly[0];
                const Point2D& v1 = clip_poly[i + 1];
                const Point2D& v2 = clip_poly[i + 2];
                const double sa = SignedArea2D(v0, v1, v2);
                if (std::abs(sa) < area_tol_abs) { continue; }  // sliver
                MFEM_VERIFY(sa > 0.0,
                            "ClipFacePairs: fan triangle has negative signed "
                            "area — orientation invariant violated. CCW input "
                            "polygons should produce CCW intersections.");

                ClippedSubTriangle tri;
                tri.m_idx = m;
                tri.verts_ab[0][0] = v0[0]; tri.verts_ab[0][1] = v0[1];
                tri.verts_ab[1][0] = v1[0]; tri.verts_ab[1][1] = v1[1];
                tri.verts_ab[2][0] = v2[0]; tri.verts_ab[2][1] = v2[1];
                tri.area = sa;

                result.sub_tris.push_back(tri);
                ++result.counts[s];
            }
        }
        result.offsets[s + 1] = result.offsets[s] + result.counts[s];
    }

    for (axom::IndexType s = 0; s < n_nonmortar; ++s)
    {
        double covered = 0.0;
        for (axom::IndexType t = result.offsets[s]; t < result.offsets[s + 1]; ++t)
        {
            covered += result.sub_tris[t].area;
        }
        const double s_area = Element2DArea(nonmortar_elems[s], a_idx, b_idx);
        // Allow a generous relative slack; a genuine coverage gap is O(1),
        // not O(area_tol_rel).
        if (!(covered >= (1.0 - 1.0e-6) * s_area)) {
            std::stringstream output;
            output << " ClipFacePairs: nonmortar element " << s << " covered area "
                    << covered << " < element area " << s_area
                    << " — missing mortar partner (halo too small or ghosting "
                    "not run?)."
                    << std::endl;
            MFEM_WARNING_0(output.str());
        }
    }

    return result;
}

}  // anonymous namespace

ClippedPairCandidates MatchClippedQuadFacePairs(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const std::string& perpendicular_axis,
    double aabb_pad_rel)
{
    return MatchClippedFacePairsImpl(nonmortar_elems, mortar_elems,
                                     perpendicular_axis, aabb_pad_rel);
}

ClippedPairCandidates MatchClippedTriFacePairs(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const std::string& perpendicular_axis,
    double aabb_pad_rel)
{
    return MatchClippedFacePairsImpl(nonmortar_elems, mortar_elems,
                                     perpendicular_axis, aabb_pad_rel);
}

ClippedSubTriangulation ClipQuadFacePairs(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const ClippedPairCandidates& candidates,
    const std::string& perpendicular_axis,
    double area_tol_rel)
{
    return ClipFacePairsImpl(nonmortar_elems, mortar_elems, candidates,
                             perpendicular_axis, area_tol_rel);
}

ClippedSubTriangulation ClipTriFacePairs(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const ClippedPairCandidates& candidates,
    const std::string& perpendicular_axis,
    double area_tol_rel)
{
    return ClipFacePairsImpl(nonmortar_elems, mortar_elems, candidates,
                             perpendicular_axis, area_tol_rel);
}

}  // namespace mortar_pbc
