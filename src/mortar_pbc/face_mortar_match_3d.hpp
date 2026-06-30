// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-B — broad-phase candidate-pair enumeration for
// non-conforming face-mortar pairs.
//
// This header defines the broad-phase spatial-search step that enables
// non-conforming face mortar work. Given the nonmortar and mortar face-
// element lists for one periodic face pair (i.e., one axis-aligned
// face-pair on a cubic RVE), it returns a CSR-format list of candidate
// (s_idx, m_idx) pairs whose 2D-projected AABBs overlap. The 2D
// projection drops the perpendicular axis (normal to the periodic
// face) since the faces are flat and axis-aligned.
//
// The fine-phase clipping (Sutherland-Hodgman convex-on-convex) is
// Batch 4.4-C; the assembler that consumes the clipped sub-polygons
// is Batch 4.4-D. This file contains only the broad-phase.
//
// Implementation uses Axom's BVH<2> spatial index. The Phase 4.4
// architectural plan (§P4.4.6.10) and architecture doc §11.6 spell
// out the full pipeline.
//
// Cross-references:
//   * Phase 4 plan §P4.4.6.10 — overall plan
//   * Phase 4 plan §P4.8.18 — Axom dependency notes
//   * Architecture doc §3.5–3.7 — geometric matching algorithm
//   * Architecture doc §11.6 — face mortar matching pseudocode

#pragma once

#include "axom/core.hpp"
#include "types_3d.hpp"

#include <vector>

namespace mortar_pbc
{

/// Broad-phase output: CSR-format candidate (s_idx, m_idx) pair list.
///
/// For nonmortar element `s_idx ∈ [0, n_nonmortar)`, the mortar-element
/// candidate indices (in mortar_elems) are
///   `candidates[offsets[s_idx] : offsets[s_idx] + counts[s_idx]]`.
/// `offsets` has size `n_nonmortar + 1` so the final entry is a sentinel
/// equal to `candidates.size()` (mirrors Axom's CSR convention exactly).
///
/// `counts[s_idx]` is denormalized for convenience even though it equals
/// `offsets[s_idx + 1] - offsets[s_idx]`; matches Axom's BVH output.
struct ClippedPairCandidates
{
    std::vector<axom::IndexType> offsets;     ///< size n_nonmortar + 1
    std::vector<axom::IndexType> counts;      ///< size n_nonmortar
    std::vector<axom::IndexType> candidates;  ///< packed: total = offsets.back()
};

/// Fine-phase output: 2D-projected, fan-triangulated overlap polygon
/// per candidate (s_idx, m_idx) pair, in CSR format keyed by
/// nonmortar element index.
///
/// For nonmortar element `s_idx ∈ [0, n_nonmortar)`, the
/// sub-triangles owned by it are
///   `sub_tris[offsets[s_idx] : offsets[s_idx] + counts[s_idx]]`.
/// Each sub-triangle stores its mortar partner index `m_idx`, the
/// three 2D-projected vertices in (a, b) coords, and the signed
/// 2D area (always positive — guaranteed by the orientation
/// invariant; assertions catch bugs).
///
/// Pairs from `ClippedPairCandidates` whose `clip()` produced an
/// empty polygon, fewer than 3 vertices, or only degenerate
/// (sub-tolerance-area) sub-triangles are dropped here. A non-trivial
/// nonmortar element with no surviving sub-triangles is unusual but
/// not an error (e.g., touching only along an edge); `counts[s_idx]`
/// is then 0.
struct ClippedSubTriangle
{
    axom::IndexType m_idx;     ///< owning mortar element index
    double verts_ab[3][2];     ///< 3 vertices, each (a, b) 2D-projected
    double area;               ///< 2D signed area (positive by invariant)
};

struct ClippedSubTriangulation
{
    std::vector<axom::IndexType> offsets;        ///< size n_nonmortar + 1
    std::vector<axom::IndexType> counts;         ///< size n_nonmortar
    std::vector<ClippedSubTriangle> sub_tris;    ///< packed list

    /// Total 2D area summed across all sub-triangles. For full-coverage
    /// non-conforming pairs this equals the nonmortar face's total
    /// 2D-projected area to roundoff. Useful as a tile-cover invariant
    /// check.
    double TotalArea() const {
        double a = 0.0;
        for (const auto& t : sub_tris) { a += t.area; }
        return a;
    }
};

/// Enumerate candidate (s_idx, m_idx) pairs for a quad-quad face mortar
/// pair via 2D-projected AABB intersection.
///
/// @param[in] nonmortar_elems  nonmortar-side quad face elements (- side)
/// @param[in] mortar_elems     mortar-side quad face elements (+ side)
/// @param[in] perpendicular_axis  the axis normal to the periodic face;
///                                must be one of "x", "y", "z"; mortar
///                                and nonmortar elements must share this
///                                axis (assertion).
/// @param[in] aabb_pad_rel  relative padding applied to mortar AABBs to
///                          tolerate exact-vertex-on-edge cases. Default
///                          1e-9 (matches the architecture doc §3.6
///                          tolerance for vertex matching). Pad scales
///                          with the largest mortar-element edge length.
/// @return CSR candidate list (see ClippedPairCandidates).
///
/// @details
///   1. Drop the perpendicular axis to project both element sets into
///      2D parametric (a, b) coordinates: for perpendicular_axis = "x",
///      (a, b) = (y, z); for "y", (a, b) = (z, x); for "z", (a, b) =
///      (x, y). This convention preserves CCW orientation.
///   2. Build an axom::primal::BoundingBox<double, 2> per mortar element
///      from its 4 vertices, padded by aabb_pad_rel * max_edge_length.
///   3. Initialize axom::spin::BVH<2> on the mortar AABBs.
///   4. Build a query AABB per nonmortar element (no padding — the
///      mortar pad covers the slop).
///   5. Call BVH::findBoundingBoxes to populate offsets / counts /
///      candidates.
///
///   Used at setup time only (not in the hot path); host-only is fine.
ClippedPairCandidates MatchClippedQuadFacePairs(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const std::string& perpendicular_axis,
    double aabb_pad_rel = 1.0e-9);

/// Enumerate candidate (s_idx, m_idx) pairs for a tri-tri face mortar
/// pair via 2D-projected AABB intersection.
///
/// Identical contract to MatchClippedQuadFacePairs but for 3-node tri
/// face elements.
ClippedPairCandidates MatchClippedTriFacePairs(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const std::string& perpendicular_axis,
    double aabb_pad_rel = 1.0e-9);

/// Fine-phase polygon clipping + fan-triangulation for quad-quad face
/// mortar pairs.
///
/// @param[in] nonmortar_elems  nonmortar-side quad face elements (- side)
/// @param[in] mortar_elems     mortar-side quad face elements (+ side)
/// @param[in] candidates       broad-phase output from MatchClippedQuadFacePairs
/// @param[in] perpendicular_axis  same as MatchClippedQuadFacePairs
/// @param[in] area_tol_rel     drop sub-triangles whose area is below
///                             this fraction of the nonmortar element
///                             area (default 1e-12).
/// @return CSR-format sub-triangulation (see ClippedSubTriangulation).
///
/// @details
///   For each (s_idx, m_idx) candidate pair:
///     1. Build axom::primal::Polygon<double, 2> for nonmortar s_idx
///        (4 verts in CCW (a, b) order) and mortar m_idx (4 verts).
///     2. Compute their 2D intersection via axom::primal::clip.
///     3. If the result has < 3 vertices, skip (no overlap, or shared
///        edge only).
///     4. Fan-triangulate from vertex 0: triangles (v0, v1, v2),
///        (v0, v2, v3), …, (v0, v_{n-2}, v_{n-1}).
///     5. For each fan triangle, compute signed 2D area; drop if
///        |area| < area_tol_rel * nonmortar_area; assert area > 0
///        otherwise (CCW invariant).
///
///   Used at setup time only.
ClippedSubTriangulation ClipQuadFacePairs(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const ClippedPairCandidates& candidates,
    const std::string& perpendicular_axis,
    double area_tol_rel = 1.0e-12);

/// Fine-phase polygon clipping + fan-triangulation for tri-tri face
/// mortar pairs. Identical contract to ClipQuadFacePairs.
ClippedSubTriangulation ClipTriFacePairs(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const ClippedPairCandidates& candidates,
    const std::string& perpendicular_axis,
    double area_tol_rel = 1.0e-12);

}  // namespace mortar_pbc
