// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-2 — non-conforming Q1 quad-quad face mortar
// assembler. See face_mortar_assembler_clipped_3d.hpp for API and
// rationale.

#include "face_mortar_assembler_clipped_3d.hpp"

#include "face_mortar_assembler_3d.hpp"   // NQuad4, MQuad4DualModified,
                                          // GaussQuad3x3, DunavantTri6Pt
#include "face_mortar_inverse_map_3d.hpp"

#include "mfem.hpp"
#include "utilities/mechanics_log.hpp"   // CALI_CXX_MARK_SCOPE

#include <algorithm>
#include <map>
#include <set>

namespace mortar_pbc
{

namespace
{

// ----------------------------------------------------------------------------
// Helpers replicated from face_mortar_assembler_3d.cpp's anonymous
// namespace. These are pure functions; we duplicate rather than friend-
// export to keep the conforming class encapsulated.
// ----------------------------------------------------------------------------

/// Map "x"/"y"/"z" to the corresponding column index 0/1/2.
int AxisIndex(const std::string& axis)
{
    if (axis == "x") { return 0; }
    if (axis == "y") { return 1; }
    if (axis == "z") { return 2; }
    MFEM_ABORT("AxisIndex: unknown axis label '" << axis << "'");
    return -1;
}

/// Cyclic 2D-projection axes for a perpendicular direction (matches
/// face_mortar_match_3d.cpp's ProjectionAxes).
std::pair<int, int> ProjectionAxes(const std::string& perpendicular_axis)
{
    if (perpendicular_axis == "x") { return {1, 2}; }
    if (perpendicular_axis == "y") { return {2, 0}; }
    if (perpendicular_axis == "z") { return {0, 1}; }
    MFEM_ABORT("ProjectionAxes: unknown perpendicular_axis '"
               << perpendicular_axis << "'.");
    return {-1, -1};
}

/// Walk the elements, collecting the sorted list of unique kept
/// gtdofs. Sentinels (gtdof < 0) are dropped. Mirrors
/// face_mortar_assembler_3d.cpp's DiscoverKeptGtdofs.
template <typename FaceElemT>
void DiscoverKeptGtdofs(const std::vector<FaceElemT>& elems,
                                  mfem::Array<int>& sorted_kept,
                                  std::map<int, int>& idx_of)
{
    std::set<int> seen;
    std::vector<int> ordered;
    for (const auto& e : elems)
    {
        for (int g : e.gtdofs)
        {
            if (g < 0) { continue; }
            if (seen.insert(g).second) { ordered.push_back(g); }
        }
    }
    std::sort(ordered.begin(), ordered.end());
    sorted_kept.SetSize(static_cast<int>(ordered.size()));
    idx_of.clear();
    for (int i = 0; i < sorted_kept.Size(); ++i)
    {
        sorted_kept[i] = ordered[i];
        idx_of[ordered[i]] = i;
    }
}

/// Wohlmuth-modified dual-basis side selectors per boundary_tag for
/// QuadFaceElement. Mirrors QuadFaceMortarAssembler::BoundaryTagToSides.
std::pair<std::string, std::string>
BoundaryTagToSides(const std::string& boundary_tag)
{
    if (boundary_tag == "none")          { return {"none",  "none"};   }
    if (boundary_tag == "edge-xi-low")   { return {"left",  "none"};   }
    if (boundary_tag == "edge-xi-high")  { return {"right", "none"};   }
    if (boundary_tag == "edge-eta-low")  { return {"none",  "bottom"}; }
    if (boundary_tag == "edge-eta-high") { return {"none",  "top"};    }
    if (boundary_tag == "corner-LL")     { return {"left",  "bottom"}; }
    if (boundary_tag == "corner-LR")     { return {"right", "bottom"}; }
    if (boundary_tag == "corner-UL")     { return {"left",  "top"};    }
    if (boundary_tag == "corner-UR")     { return {"right", "top"};    }
    MFEM_ABORT("BoundaryTagToSides (clipped): unrecognised boundary_tag '"
               << boundary_tag << "'.");
    return {"none", "none"};
}

/// Axis-aligned-shortcut Jacobian for a Q1 quad face element. Returns
/// |J| = (Δa/2)(Δb/2) for axis-aligned quads. The clipped path's Phase
/// 4.4 scope is axis-aligned only, so we use the closed-form constant
/// here (matches QuadFaceMortarAssembler::NonmortarJacobian's
/// axis-aligned branch). For non-axis-aligned production data the
/// conforming code falls back to the bilinear point-by-point Jacobian
/// — we don't replicate that here because Phase 4.4 doesn't support it.
double NonmortarJacobianAxisAligned(const QuadFaceElement& elem)
{
    const int a_idx = AxisIndex(elem.parametric_axes[0]);
    const int b_idx = AxisIndex(elem.parametric_axes[1]);
    double a_lo = elem.coords(0, a_idx);
    double a_hi = a_lo;
    double b_lo = elem.coords(0, b_idx);
    double b_hi = b_lo;
    for (int n = 1; n < 4; ++n)
    {
        a_lo = std::min(a_lo, elem.coords(n, a_idx));
        a_hi = std::max(a_hi, elem.coords(n, a_idx));
        b_lo = std::min(b_lo, elem.coords(n, b_idx));
        b_hi = std::max(b_hi, elem.coords(n, b_idx));
    }
    return 0.25 * (a_hi - a_lo) * (b_hi - b_lo);
}

/// Wohlmuth-modified dual-basis drops per boundary_tag for
/// TriFaceElement. Mirrors TriFaceMortarAssembler::BoundaryTagToDrops.
/// Returns a 3-tuple of bool flags consumed by MTri3DualModified.
std::array<bool, 3> BoundaryTagToDropsTri(const std::string& boundary_tag)
{
    if (boundary_tag == "none")     { return {false, false, false}; }
    if (boundary_tag == "v0")       { return {true,  false, false}; }
    if (boundary_tag == "v1")       { return {false, true,  false}; }
    if (boundary_tag == "v2")       { return {false, false, true};  }
    if (boundary_tag == "v0-v1")    { return {true,  true,  false}; }
    if (boundary_tag == "v0-v2")    { return {true,  false, true};  }
    if (boundary_tag == "v1-v2")    { return {false, true,  true};  }
    if (boundary_tag == "v0-v1-v2") { return {true,  true,  true};  }
    MFEM_ABORT("BoundaryTagToDropsTri (clipped): unrecognised boundary_tag '"
               << boundary_tag << "'.");
    return {false, false, false};
}

/// Full-element Jacobian for a P1 tri face element on the reference
/// simplex |T_ref| = 1/2. Returns J = 2 * |T_phys|, where |T_phys|
/// is the 3D triangle area via cross-product magnitude. With weights
/// of GaussTri3Pt summing to 1/2, Σ phys_w = J · 1/2 = |T_phys| as
/// expected.
///
/// Mirrors the lambda in TriFaceMortarAssembler::AssemblePairConforming.
double TriFullJacobian(const TriFaceElement& elem)
{
    const auto& c = elem.coords;
    const double v01[3] = {c(1, 0) - c(0, 0),
                           c(1, 1) - c(0, 1),
                           c(1, 2) - c(0, 2)};
    const double v02[3] = {c(2, 0) - c(0, 0),
                           c(2, 1) - c(0, 1),
                           c(2, 2) - c(0, 2)};
    const double cx = v01[1] * v02[2] - v01[2] * v02[1];
    const double cy = v01[2] * v02[0] - v01[0] * v02[2];
    const double cz = v01[0] * v02[1] - v01[1] * v02[0];
    const double tri_area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    return 2.0 * tri_area;
}

}  // anonymous namespace

// ============================================================================
// AssembleQuadFacePairClipped
// ============================================================================

FaceMortarPairBlock AssembleQuadFacePairClipped(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const ClippedSubTriangulation& sub_tris,
    const std::string& perpendicular_axis,
    const std::string& nonmortar_face_name,
    const std::string& mortar_face_name)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::face_mortar::quad::integrate_pair_clipped");

    const axom::IndexType n_nonmortar =
        static_cast<axom::IndexType>(nonmortar_elems.size());
    MFEM_VERIFY(static_cast<axom::IndexType>(sub_tris.counts.size()) == n_nonmortar,
                "AssembleQuadFacePairClipped: sub_tris.counts.size() != "
                "n_nonmortar.");
    MFEM_VERIFY(static_cast<axom::IndexType>(sub_tris.offsets.size())
                    == n_nonmortar + 1,
                "AssembleQuadFacePairClipped: sub_tris.offsets.size() != "
                "n_nonmortar + 1.");

    FaceMortarPairBlock block;
    block.nonmortar_face_name = nonmortar_face_name;
    block.mortar_face_name    = mortar_face_name;

    // First pass: discover kept gtdof sets — same as the conforming path.
    std::map<int, int> nonmortar_row_of, mortar_col_of;
    DiscoverKeptGtdofs(nonmortar_elems,  block.nonmortar_gtdofs,  nonmortar_row_of);
    DiscoverKeptGtdofs(mortar_elems, block.mortar_gtdofs, mortar_col_of);
    const int n_rows = block.nonmortar_gtdofs.Size();
    const int n_cols = block.mortar_gtdofs.Size();
    block.D.SetSize(n_rows);
    block.D = 0.0;
    block.A_m = mfem::SparseMatrix(n_rows, n_cols);

    if (n_nonmortar == 0)
    {
        block.A_m.Finalize();
        return block;
    }

    // Quadrature rules: 9-point Gauss-Legendre on parent quad for D
    // (full-element integration), 6-point Dunavant on each clipped sub-
    // triangle for A^m (per-overlap integration).
    const auto rule_d = GaussQuad3x3();
    const auto rule_a = DunavantTri6Pt();

    // 2D-projection axes for the inverse maps and sub-triangle parameter
    // recovery.
    const auto axes = ProjectionAxes(perpendicular_axis);
    const int  a_idx = axes.first;
    const int  b_idx = axes.second;

    // Second pass: integrate per nonmortar element.
    for (axom::IndexType s_idx = 0; s_idx < n_nonmortar; ++s_idx)
    {
        const QuadFaceElement& s = nonmortar_elems[s_idx];
        const auto sides = BoundaryTagToSides(s.boundary_tag);
        const std::string& side_xi  = sides.first;
        const std::string& side_eta = sides.second;

        // -----------------------------------------------------------------
        // Pass 1: D contribution on the FULL nonmortar element. Same loop
        // as AssemblePairConforming's D accumulation. Wohlmuth biorthogonality
        // guarantees this lumps to a diagonal D when summed over all q-pts
        // in the parent reference quad.
        // -----------------------------------------------------------------
        std::array<double, 4> D_loc = {0.0, 0.0, 0.0, 0.0};
        const double J_full = NonmortarJacobianAxisAligned(s);
        for (int q = 0; q < 9; ++q)
        {
            const auto pt = rule_d.pts[q];
            const double w = rule_d.wts[q];
            const double phys_w = w * J_full;
            const auto N_nonmortar = NQuad4(pt[0], pt[1]);
            for (int k = 0; k < 4; ++k)
            {
                D_loc[k] += phys_w * N_nonmortar[k];
            }
        }

        // -----------------------------------------------------------------
        // Pass 2: A^m contribution on each clipped sub-triangle owned by
        // this nonmortar element. We accumulate A_loc[m_idx][k][l] keyed
        // by mortar element index because different sub-tris may have
        // different mortar partners. To avoid a hash-map allocation per
        // call, we accumulate directly into block.A_m by keeping a
        // running m_idx-keyed accumulator; the sparse Add() machinery
        // already handles cross-mortar accumulation correctly.
        //
        // Per-sub-triangle scaling: weights of DunavantTri6Pt sum to
        // |T_ref| = 1/2; physical sub-tri area is sub_tri.area; so
        // J_sub = 2 * sub_tri.area, which gives Σ phys_w = sub_tri.area
        // as expected.
        // -----------------------------------------------------------------
        const axom::IndexType k_lo = sub_tris.offsets[s_idx];
        const axom::IndexType k_hi = sub_tris.offsets[s_idx + 1];
        for (axom::IndexType k = k_lo; k < k_hi; ++k)
        {
            const ClippedSubTriangle& tri = sub_tris.sub_tris[k];
            const QuadFaceElement& m = mortar_elems[tri.m_idx];
            const double J_sub = 2.0 * tri.area;

            std::array<std::array<double, 4>, 4> A_loc = {};

            for (int q = 0; q < 6; ++q)
            {
                const auto& lam = rule_a.pts[q];
                const double w = rule_a.wts[q];
                const double sub_phys_w = w * J_sub;

                // Sub-triangle barycentric → 2D physical (a, b).
                const double a = lam[0] * tri.verts_ab[0][0]
                               + lam[1] * tri.verts_ab[1][0]
                               + lam[2] * tri.verts_ab[2][0];
                const double b = lam[0] * tri.verts_ab[0][1]
                               + lam[1] * tri.verts_ab[1][1]
                               + lam[2] * tri.verts_ab[2][1];

                // Inverse-iso-map: (a, b) → nonmortar (xi_nm, eta_nm).
                const auto pt_nm = InverseMapQuad2DAxisAligned(s, a_idx, b_idx,
                                                                            a, b);
                // Inverse-iso-map: (a, b) → mortar (xi_m, eta_m).
                const auto pt_m  = InverseMapQuad2DAxisAligned(m, a_idx, b_idx,
                                                                            a, b);

                const auto M_dual_nm = MQuad4DualModified(pt_nm[0], pt_nm[1],
                                                                       side_xi,
                                                                       side_eta);
                const auto N_mortar  = NQuad4(pt_m[0], pt_m[1]);

                for (int kk = 0; kk < 4; ++kk)
                {
                    for (int ll = 0; ll < 4; ++ll)
                    {
                        A_loc[kk][ll] += sub_phys_w * M_dual_nm[kk] * N_mortar[ll];
                    }
                }
            }

            // Scatter A_loc for this (s, m) sub-triangle into the global
            // block, dropping sentinel rows/cols. The Add() into the
            // SparseMatrix accumulates contributions across sub-triangles
            // sharing the same (s, m) pair OR the same row/col indices
            // from different (s, m) pairs.
            for (int kk_loc = 0; kk_loc < 4; ++kk_loc)
            {
                const int g_nm = s.gtdofs[kk_loc];
                if (g_nm < 0) { continue; }
                const int kk_global = nonmortar_row_of[g_nm];
                for (int ll_loc = 0; ll_loc < 4; ++ll_loc)
                {
                    const int g_m = m.gtdofs[ll_loc];
                    if (g_m < 0) { continue; }
                    const int ll_global = mortar_col_of[g_m];
                    block.A_m.Add(kk_global, ll_global, A_loc[kk_loc][ll_loc]);
                }
            }
        }

        // -----------------------------------------------------------------
        // Scatter D_loc for this nonmortar element into block.D, dropping
        // sentinels.
        // -----------------------------------------------------------------
        for (int k_loc = 0; k_loc < 4; ++k_loc)
        {
            const int g_nm = s.gtdofs[k_loc];
            if (g_nm < 0) { continue; }
            const int k_global = nonmortar_row_of[g_nm];
            block.D(k_global) += D_loc[k_loc];
        }
    }

    block.A_m.Finalize();
    return block;
}

// ============================================================================
// AssembleTriFacePairClipped
// ============================================================================

FaceMortarPairBlock AssembleTriFacePairClipped(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const ClippedSubTriangulation& sub_tris,
    const std::string& perpendicular_axis,
    const std::string& nonmortar_face_name,
    const std::string& mortar_face_name)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::face_mortar::tri::integrate_pair_clipped");

    const axom::IndexType n_nonmortar =
        static_cast<axom::IndexType>(nonmortar_elems.size());
    MFEM_VERIFY(static_cast<axom::IndexType>(sub_tris.counts.size()) == n_nonmortar,
                "AssembleTriFacePairClipped: sub_tris.counts.size() != "
                "n_nonmortar.");
    MFEM_VERIFY(static_cast<axom::IndexType>(sub_tris.offsets.size())
                    == n_nonmortar + 1,
                "AssembleTriFacePairClipped: sub_tris.offsets.size() != "
                "n_nonmortar + 1.");

    FaceMortarPairBlock block;
    block.nonmortar_face_name = nonmortar_face_name;
    block.mortar_face_name    = mortar_face_name;

    std::map<int, int> nonmortar_row_of, mortar_col_of;
    DiscoverKeptGtdofs(nonmortar_elems,  block.nonmortar_gtdofs,  nonmortar_row_of);
    DiscoverKeptGtdofs(mortar_elems, block.mortar_gtdofs, mortar_col_of);
    const int n_rows = block.nonmortar_gtdofs.Size();
    const int n_cols = block.mortar_gtdofs.Size();
    block.D.SetSize(n_rows);
    block.D = 0.0;
    block.A_m = mfem::SparseMatrix(n_rows, n_cols);

    if (n_nonmortar == 0)
    {
        block.A_m.Finalize();
        return block;
    }

    // Quadrature: 3-point Dunavant for D (full-tri integration) AND
    // for A^m (per-sub-tri integration). Both rules suffice — the
    // P1·P1 product is degree 2 in barycentric, exact on a degree-2
    // rule. (Quad case needed bumped 6-point Dunavant for sub-tris;
    // tri case doesn't.)
    const auto rule = GaussTri3Pt();

    // 2D-projection axes for the inverse maps and sub-triangle parameter
    // recovery.
    const auto axes = ProjectionAxes(perpendicular_axis);
    const int  a_idx = axes.first;
    const int  b_idx = axes.second;

    for (axom::IndexType s_idx = 0; s_idx < n_nonmortar; ++s_idx)
    {
        const TriFaceElement& s = nonmortar_elems[s_idx];
        const auto drops = BoundaryTagToDropsTri(s.boundary_tag);

        // -----------------------------------------------------------------
        // Pass 1: D contribution on the FULL nonmortar tri. Same loop as
        // the conforming tri assembler. J = 2 · |T_phys|; weights of
        // GaussTri3Pt sum to 1/2, so Σ phys_w = |T_phys|.
        // -----------------------------------------------------------------
        std::array<double, 3> D_loc = {0.0, 0.0, 0.0};
        const double J_full = TriFullJacobian(s);
        for (int q = 0; q < 3; ++q)
        {
            const auto& lam = rule.pts[q];
            const double w = rule.wts[q];
            const double phys_w = w * J_full;
            const auto N_nonmortar = NTri3(lam);
            for (int k = 0; k < 3; ++k)
            {
                D_loc[k] += phys_w * N_nonmortar[k];
            }
        }

        // -----------------------------------------------------------------
        // Pass 2: A^m contribution on each clipped sub-triangle.
        //
        // J_sub = 2 · sub_tri.area, same as the quad case (the sub-tri
        // is generic — element type doesn't change the per-sub-tri
        // Jacobian convention).
        // -----------------------------------------------------------------
        const axom::IndexType k_lo = sub_tris.offsets[s_idx];
        const axom::IndexType k_hi = sub_tris.offsets[s_idx + 1];
        for (axom::IndexType k = k_lo; k < k_hi; ++k)
        {
            const ClippedSubTriangle& tri = sub_tris.sub_tris[k];
            const TriFaceElement& m = mortar_elems[tri.m_idx];
            const double J_sub = 2.0 * tri.area;

            std::array<std::array<double, 3>, 3> A_loc = {};

            for (int q = 0; q < 3; ++q)
            {
                const auto& lam_sub = rule.pts[q];
                const double w = rule.wts[q];
                const double sub_phys_w = w * J_sub;

                // Sub-triangle barycentric → 2D physical (a, b).
                const double a = lam_sub[0] * tri.verts_ab[0][0]
                               + lam_sub[1] * tri.verts_ab[1][0]
                               + lam_sub[2] * tri.verts_ab[2][0];
                const double b = lam_sub[0] * tri.verts_ab[0][1]
                               + lam_sub[1] * tri.verts_ab[1][1]
                               + lam_sub[2] * tri.verts_ab[2][1];

                // Inverse-iso-map: (a, b) → nonmortar tri barycentric.
                const auto lam_nm = InverseMapTri2D(s, a_idx, b_idx, a, b);
                // Inverse-iso-map: (a, b) → mortar tri barycentric.
                const auto lam_m  = InverseMapTri2D(m, a_idx, b_idx, a, b);

                const auto M_dual_nm = MTri3DualModified(lam_nm, drops);
                const auto N_mortar  = NTri3(lam_m);

                for (int kk = 0; kk < 3; ++kk)
                {
                    for (int ll = 0; ll < 3; ++ll)
                    {
                        A_loc[kk][ll] += sub_phys_w * M_dual_nm[kk] * N_mortar[ll];
                    }
                }
            }

            // Scatter A_loc into the global block (sentinel-aware drop).
            for (int kk_loc = 0; kk_loc < 3; ++kk_loc)
            {
                const int g_nm = s.gtdofs[kk_loc];
                if (g_nm < 0) { continue; }
                const int kk_global = nonmortar_row_of[g_nm];
                for (int ll_loc = 0; ll_loc < 3; ++ll_loc)
                {
                    const int g_m = m.gtdofs[ll_loc];
                    if (g_m < 0) { continue; }
                    const int ll_global = mortar_col_of[g_m];
                    block.A_m.Add(kk_global, ll_global, A_loc[kk_loc][ll_loc]);
                }
            }
        }

        // Scatter D_loc into block.D (sentinel-aware drop).
        for (int k_loc = 0; k_loc < 3; ++k_loc)
        {
            const int g_nm = s.gtdofs[k_loc];
            if (g_nm < 0) { continue; }
            const int k_global = nonmortar_row_of[g_nm];
            block.D(k_global) += D_loc[k_loc];
        }
    }

    block.A_m.Finalize();
    return block;
}

}  // namespace mortar_pbc
