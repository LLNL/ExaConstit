// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_3d.py` (basis fns + quadrature)
// and `face_mortar_3d.py` (the two assembler classes + matching helper).

#include "face_mortar_assembler_3d.hpp"

#include "mortar_assembler_2d.hpp"  // MLine2DualModified

// Caliper instrumentation. We use ExaConstit's existing wrapper from
// `utilities/mechanics_log.hpp`, which dispatches to the real Caliper
// macros when `HAVE_CALIPER` is defined and to no-ops otherwise.
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mortar_pbc {

// ============================================================================
// Quad-4 dual basis (free function — tensor product of line-2 dual)
// ============================================================================

std::array<double, 4> MQuad4Dual(double xi, double eta) noexcept
{
    const auto Mxi  = MLine2Dual(xi);
    const auto Meta = MLine2Dual(eta);
    return {
        Mxi[0] * Meta[0],   // node 0: (-1, -1)
        Mxi[1] * Meta[0],   // node 1: (+1, -1)
        Mxi[1] * Meta[1],   // node 2: (+1, +1)
        Mxi[0] * Meta[1],   // node 3: (-1, +1)
    };
}

// ============================================================================
// Wohlmuth-modified tri-3 dual
// ============================================================================

std::array<double, 3>
MTri3DualModified(const std::array<double, 3>& lam,
                         const std::array<bool, 3>& boundary_nodes)
{
    int n_dropped = 0;
    for (bool b : boundary_nodes) { if (b) { ++n_dropped; } }

    if (n_dropped == 0) { return MTri3Dual(lam); }

    if (n_dropped == 3) { return {0.0, 0.0, 0.0}; }

    if (n_dropped == 2)
    {
        // Two corners dropped, one kept. Kept vertex's M is identically 1.
        std::array<double, 3> result = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i)
        {
            if (!boundary_nodes[i]) { result[i] = 1.0; break; }
        }
        return result;
    }

    // n_dropped == 1: edge-adjacent (eq. 5.5).
    //   For dropped vertex i and kept vertices j = (i+1)%3, k = (i+2)%3:
    //     M_i = 0
    //     M_j = 1/2 + 2 lam_j - 2 lam_k
    //     M_k = 1/2 - 2 lam_j + 2 lam_k
    int idx_dropped = -1;
    for (int i = 0; i < 3; ++i)
    {
        if (boundary_nodes[i]) { idx_dropped = i; break; }
    }
    const int idx_j = (idx_dropped + 1) % 3;
    const int idx_k = (idx_dropped + 2) % 3;
    const double lam_j = lam[idx_j];
    const double lam_k = lam[idx_k];

    std::array<double, 3> result = {0.0, 0.0, 0.0};
    result[idx_j] = 0.5 + 2.0 * lam_j - 2.0 * lam_k;
    result[idx_k] = 0.5 - 2.0 * lam_j + 2.0 * lam_k;
    // result[idx_dropped] stays 0.
    return result;
}

// ============================================================================
// Wohlmuth-modified quad-4 dual
// ============================================================================

std::array<double, 4>
MQuad4DualModified(double xi, double eta,
                          const std::string& side_xi,
                          const std::string& side_eta)
{
    // Map side_eta to line-2 left/right semantics so we can call
    // MLine2DualModified twice.
    std::string side_eta_mapped;
    if      (side_eta == "none")   { side_eta_mapped = "none";  }
    else if (side_eta == "bottom") { side_eta_mapped = "left";  }
    else if (side_eta == "top")    { side_eta_mapped = "right"; }
    else if (side_eta == "both")   { side_eta_mapped = "both";  }
    else
    {
        MFEM_ABORT("MQuad4DualModified: unknown side_eta '" << side_eta
                      << "'; expected one of "
                      << "{'none', 'bottom', 'top', 'both'}.");
    }

    const auto Mxi  = MLine2DualModified(xi,  side_xi);
    const auto Meta = MLine2DualModified(eta, side_eta_mapped);

    return {
        Mxi[0] * Meta[0],   // node 0: (-1, -1)
        Mxi[1] * Meta[0],   // node 1: (+1, -1)
        Mxi[1] * Meta[1],   // node 2: (+1, +1)
        Mxi[0] * Meta[1],   // node 3: (-1, +1)
    };
}

// ============================================================================
// Quadrature rules
// ============================================================================

namespace
{
    // 3-point GL on [-1, +1].
    constexpr int kGL3N = 3;
    const std::array<double, kGL3N> kGL3Pts1D = {
        -std::sqrt(0.6), 0.0, std::sqrt(0.6)
    };
    constexpr std::array<double, kGL3N> kGL3Wts1D = {
        5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0
    };
}  // namespace

QuadratureQuad3x3 GaussQuad3x3()
{
    QuadratureQuad3x3 rule;
    int k = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rule.pts[k] = {kGL3Pts1D[i], kGL3Pts1D[j]};
            rule.wts[k] = kGL3Wts1D[i] * kGL3Wts1D[j];
            ++k;
        }
    }
    return rule;
}

QuadratureTri3Pt GaussTri3Pt()
{
    QuadratureTri3Pt rule;
    // 3-point degree-2 Dunavant rule on the simplex; weights sum to 1/2.
    rule.pts[0] = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0};
    rule.pts[1] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
    rule.pts[2] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
    rule.wts[0] = rule.wts[1] = rule.wts[2] = 1.0 / 6.0;
    return rule;
}

QuadratureTri6Pt DunavantTri6Pt()
{
    QuadratureTri6Pt rule;
    // Dunavant 1985 degree-4 rule, 6 points, two symmetric orbits.
    // Barycentric coordinates and weights (standard tabulation uses
    // unit-area reference; multiply weights by |T_ref| = 1/2 to match
    // GaussTri3Pt's |T| = 1/2 convention).
    //
    // Orbit 1 (3 points):
    //   alpha_1 = 0.108103018168070
    //   beta_1  = 0.445948490915965
    //   weight (unit-area) = 0.223381589678011
    //   weight (|T|=1/2)   = 0.223381589678011 / 2 ≈ 0.111690794839006
    constexpr double a1 = 0.108103018168070;
    constexpr double b1 = 0.445948490915965;
    constexpr double w1 = 0.111690794839006;
    // Orbit 2 (3 points):
    //   alpha_2 = 0.816847572980459
    //   beta_2  = 0.091576213509771
    //   weight (unit-area) = 0.109951743655322
    //   weight (|T|=1/2)   = 0.109951743655322 / 2 ≈ 0.054975871827661
    constexpr double a2 = 0.816847572980459;
    constexpr double b2 = 0.091576213509771;
    constexpr double w2 = 0.054975871827661;

    rule.pts[0] = {a1, b1, b1};
    rule.pts[1] = {b1, a1, b1};
    rule.pts[2] = {b1, b1, a1};
    rule.pts[3] = {a2, b2, b2};
    rule.pts[4] = {b2, a2, b2};
    rule.pts[5] = {b2, b2, a2};
    rule.wts[0] = rule.wts[1] = rule.wts[2] = w1;
    rule.wts[3] = rule.wts[4] = rule.wts[5] = w2;
    return rule;
}

// ============================================================================
// Common helpers (shared between the two concrete assemblers)
// ============================================================================

namespace
{
    // Tolerance for the lumped-positivity check.
    constexpr double kLumpedPositivityTol = 1e-12;

    /// Walk the elements, collecting the sorted list of unique kept
    /// gtdofs. Sentinels (gtdof < 0) are dropped.
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

    /// Centroid of a face element along given axis indices.
    template <typename FaceElemT>
    std::array<double, 2>
    CentroidInPlane(const FaceElemT& e, int a_idx, int b_idx)
    {
        const int n = FaceElemT::NumNodes();
        double a = 0.0, b = 0.0;
        for (int v = 0; v < n; ++v)
        {
            a += e.coords(v, a_idx);
            b += e.coords(v, b_idx);
        }
        return {a / n, b / n};
    }

    /// Map "x"/"y"/"z" to the corresponding column index 0/1/2.
    int AxisIndex(const std::string& axis)
    {
        if (axis == "x") { return 0; }
        if (axis == "y") { return 1; }
        if (axis == "z") { return 2; }
        MFEM_ABORT("Unknown axis label '" << axis << "'");
        return -1;
    }
}  // namespace

// ============================================================================
// QuadFaceMortarAssembler
// ============================================================================

QuadFaceMortarAssembler::QuadFaceMortarAssembler()
{
    VerifyLumpedPositivity();
}

void QuadFaceMortarAssembler::VerifyLumpedPositivity()
{
    // s_j = ∫_{[-1,1]^2} N_j dA evaluated via 3x3 Gauss should equal 1
    // for all four nodes. (|E|=4, lumped distributes equally.)
    const auto rule = GaussQuad3x3();
    std::array<double, 4> s = {0, 0, 0, 0};
    for (int q = 0; q < 9; ++q)
    {
        const auto pt = rule.pts[q];
        const double w = rule.wts[q];
        const auto N = NQuad4(pt[0], pt[1]);
        for (int j = 0; j < 4; ++j) { s[j] += w * N[j]; }
    }
    for (int j = 0; j < 4; ++j)
    {
        MFEM_VERIFY(s[j] > kLumpedPositivityTol,
                        "QuadFaceMortarAssembler: lumped-positivity check failed "
                        "(s[" << j << "] = " << s[j] << "). "
                        "This indicates a bug in NQuad4 or GaussQuad3x3.");
    }
}

std::pair<std::string, std::string>
QuadFaceMortarAssembler::BoundaryTagToSides(const std::string& boundary_tag)
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
    MFEM_ABORT("QuadFaceMortarAssembler: unrecognised boundary_tag '"
                  << boundary_tag << "'.");
    return {"none", "none"};   // unreachable
}

double QuadFaceMortarAssembler::NonmortarJacobian(
     const QuadFaceElement& nonmortar_elem,
     std::array<double, 2> q_pt) const
{
    const int a_idx = AxisIndex(nonmortar_elem.parametric_axes[0]);
    const int b_idx = AxisIndex(nonmortar_elem.parametric_axes[1]);

    // Try the axis-aligned constant-J shortcut (the common case for
    // MakeCartesian3D meshes).
    constexpr double kAxisAlignedTol = 1e-12;
    double a_lo = nonmortar_elem.coords(0, a_idx);
    double a_hi = a_lo;
    double b_lo = nonmortar_elem.coords(0, b_idx);
    double b_hi = b_lo;
    for (int n = 1; n < 4; ++n)
    {
        a_lo = std::min(a_lo, nonmortar_elem.coords(n, a_idx));
        a_hi = std::max(a_hi, nonmortar_elem.coords(n, a_idx));
        b_lo = std::min(b_lo, nonmortar_elem.coords(n, b_idx));
        b_hi = std::max(b_hi, nonmortar_elem.coords(n, b_idx));
    }
    bool axis_aligned = true;
    for (int n = 0; n < 4 && axis_aligned; ++n)
    {
        const double a = nonmortar_elem.coords(n, a_idx);
        const double b = nonmortar_elem.coords(n, b_idx);
        const bool a_at_lo = std::abs(a - a_lo) < kAxisAlignedTol;
        const bool a_at_hi = std::abs(a - a_hi) < kAxisAlignedTol;
        const bool b_at_lo = std::abs(b - b_lo) < kAxisAlignedTol;
        const bool b_at_hi = std::abs(b - b_hi) < kAxisAlignedTol;
        if (!((a_at_lo || a_at_hi) && (b_at_lo || b_at_hi)))
        {
            axis_aligned = false;
        }
    }
    if (axis_aligned)
    {
        // Constant Jacobian: |J| = (Δa/2) * (Δb/2).
        return 0.25 * (a_hi - a_lo) * (b_hi - b_lo);
    }

    // Non-axis-aligned: bilinear quad Jacobian per point. Restrict to
    // the two parametric axes; the third is constant on the face.
    const double xi  = q_pt[0];
    const double eta = q_pt[1];
    const std::array<double, 4> dN_dxi = {
        -0.25 * (1.0 - eta),
        +0.25 * (1.0 - eta),
        +0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    };
    const std::array<double, 4> dN_deta = {
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        +0.25 * (1.0 + xi),
        +0.25 * (1.0 - xi),
    };
    double J11 = 0, J12 = 0, J21 = 0, J22 = 0;
    for (int n = 0; n < 4; ++n)
    {
        J11 += dN_dxi[n]  * nonmortar_elem.coords(n, a_idx);
        J12 += dN_dxi[n]  * nonmortar_elem.coords(n, b_idx);
        J21 += dN_deta[n] * nonmortar_elem.coords(n, a_idx);
        J22 += dN_deta[n] * nonmortar_elem.coords(n, b_idx);
    }
    return std::abs(J11 * J22 - J12 * J21);
}

std::array<double, 2>
QuadFaceMortarAssembler::MortarRefFromPermutation(
     const std::array<int, 4>& mortar_node_perm,
     std::array<double, 2> q_pt_nonmortar)
{
    // Identity short-circuit (the common case).
    if (mortar_node_perm[0] == 0 && mortar_node_perm[1] == 1 &&
         mortar_node_perm[2] == 2 && mortar_node_perm[3] == 3)
    {
        return q_pt_nonmortar;
    }

    // Map nonmortar (xi, eta) to mortar (xi, eta) via the affine map
    // determined by where the nonmortar's local nodes 0, 1, 3 land on the
    // mortar.
    constexpr std::array<std::array<double, 2>, 4> kRefQuad4 = {{
        {-1.0, -1.0}, {+1.0, -1.0}, {+1.0, +1.0}, {-1.0, +1.0},
    }};
    const auto& m0 = kRefQuad4[mortar_node_perm[0]];
    const auto& m1 = kRefQuad4[mortar_node_perm[1]];
    const auto& m3 = kRefQuad4[mortar_node_perm[3]];
    const std::array<double, 2> e_xi = {
        0.5 * (m1[0] - m0[0]), 0.5 * (m1[1] - m0[1])
    };
    const std::array<double, 2> e_eta = {
        0.5 * (m3[0] - m0[0]), 0.5 * (m3[1] - m0[1])
    };
    const double xi_s  = q_pt_nonmortar[0];
    const double eta_s = q_pt_nonmortar[1];
    return {
        m0[0] + (xi_s + 1.0) * e_xi[0] + (eta_s + 1.0) * e_eta[0],
        m0[1] + (xi_s + 1.0) * e_xi[1] + (eta_s + 1.0) * e_eta[1],
    };
}

FaceMortarPairBlock
QuadFaceMortarAssembler::AssemblePairConforming(
     const std::vector<QuadFaceElement>& nonmortar_elems,
     const std::vector<QuadFaceElement>& mortar_elems,
     const std::vector<QuadFacePairMatch>& pair_matches,
     const std::string& nonmortar_face_name,
     const std::string& mortar_face_name) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::face_mortar::quad::integrate_pair");

    FaceMortarPairBlock block;
    block.nonmortar_face_name  = nonmortar_face_name;
    block.mortar_face_name = mortar_face_name;

    // First pass: discover kept gtdof sets.
    std::map<int, int> nonmortar_row_of, mortar_col_of;
    DiscoverKeptGtdofs(nonmortar_elems,  block.nonmortar_gtdofs,  nonmortar_row_of);
    DiscoverKeptGtdofs(mortar_elems, block.mortar_gtdofs, mortar_col_of);
    const int n_rows = block.nonmortar_gtdofs.Size();
    const int n_cols = block.mortar_gtdofs.Size();
    block.D.SetSize(n_rows);
    block.D = 0.0;
    // Phase 4.2 / Batch L: A_m is now mfem::SparseMatrix. Construct
    // in build mode; Add() entries during integration; Finalize() to
    // CSR before returning.
    block.A_m = mfem::SparseMatrix(n_rows, n_cols);

    const auto rule = GaussQuad3x3();

    // Second pass: integrate per matched pair.
    for (const auto& match : pair_matches)
    {
        const QuadFaceElement& s = nonmortar_elems[match.nonmortar_idx];
        const QuadFaceElement& m = mortar_elems[match.mortar_idx];
        const auto sides = BoundaryTagToSides(s.boundary_tag);
        const std::string& side_xi  = sides.first;
        const std::string& side_eta = sides.second;

        // Per-element local D and A_m, before sentinel-aware accumulation.
        std::array<double, 4>                  D_loc = {0, 0, 0, 0};
        std::array<std::array<double, 4>, 4>   A_loc = {};
        // (Default-init is zero-init for std::array of trivially-default-
        //  constructible elements when value-init'd via {}.)

        for (int q = 0; q < 9; ++q)
        {
            const auto pt = rule.pts[q];
            const double w = rule.wts[q];
            const double J = NonmortarJacobian(s, pt);
            const double phys_w = w * J;

            const auto M_nonmortar = MQuad4DualModified(pt[0], pt[1],
                                                                  side_xi, side_eta);
            const auto N_nonmortar = NQuad4(pt[0], pt[1]);
            // pt_mortar lives in the mortar element's OWN reference
            // frame (MortarRefFromPermutation handles the nm→mortar
            // axis swap from the perm), so NQuad4(pt_mortar)[j] is
            // already mortar local node j's shape function value at the
            // current physical Gauss point. The scatter pairs N_mortar[l]
            // with m.gtdofs[l] directly, with no perm indirection on
            // the shape values themselves — same approach as
            // AssembleQuadFacePairClipped.
            const auto pt_mortar = MortarRefFromPermutation(match.mortar_node_perm,
                                                                             pt);
            const auto N_mortar = NQuad4(pt_mortar[0], pt_mortar[1]);

            for (int k = 0; k < 4; ++k)
            {
                D_loc[k] += phys_w * N_nonmortar[k];
                for (int l = 0; l < 4; ++l)
                {
                    A_loc[k][l] += phys_w * M_nonmortar[k] * N_mortar[l];
                }
            }
        }

        // Scatter into the global D and A_m, dropping sentinel rows/cols.
        // A_m is sparse; Add() accumulates into existing entries or
        // creates new ones (build mode, pre-Finalize).
        for (int k_loc = 0; k_loc < 4; ++k_loc)
        {
            const int g_nonmortar = s.gtdofs[k_loc];
            if (g_nonmortar < 0) { continue; }
            const int k_global = nonmortar_row_of[g_nonmortar];
            block.D(k_global) += D_loc[k_loc];
            for (int l_loc = 0; l_loc < 4; ++l_loc)
            {
                const int g_mortar = m.gtdofs[l_loc];
                if (g_mortar < 0) { continue; }
                const int l_global = mortar_col_of[g_mortar];
                block.A_m.Add(k_global, l_global, A_loc[k_loc][l_loc]);
            }
        }
    }

    // Finalize A_m: convert from build-mode (linked-list) to CSR.
    block.A_m.Finalize();
    return block;
}

// ============================================================================
// TriFaceMortarAssembler
// ============================================================================

TriFaceMortarAssembler::TriFaceMortarAssembler()
{
    VerifyLumpedPositivity();
}

void TriFaceMortarAssembler::VerifyLumpedPositivity()
{
    // s_j = ∫_T N_j dA on the reference simplex (|T| = 1/2). For tri-3,
    // s_j = |T|/3 = 1/6 for each j.
    const auto rule = GaussTri3Pt();
    std::array<double, 3> s = {0, 0, 0};
    for (int q = 0; q < 3; ++q)
    {
        const auto pt = rule.pts[q];
        const double w = rule.wts[q];
        const auto N = NTri3(pt);
        for (int j = 0; j < 3; ++j) { s[j] += w * N[j]; }
    }
    for (int j = 0; j < 3; ++j)
    {
        MFEM_VERIFY(s[j] > kLumpedPositivityTol,
                        "TriFaceMortarAssembler: lumped-positivity check failed "
                        "(s[" << j << "] = " << s[j] << ").");
    }
}

std::array<bool, 3>
TriFaceMortarAssembler::BoundaryTagToDrops(const std::string& boundary_tag)
{
    if (boundary_tag == "none")     { return {false, false, false}; }
    if (boundary_tag == "v0")       { return {true,  false, false}; }
    if (boundary_tag == "v1")       { return {false, true,  false}; }
    if (boundary_tag == "v2")       { return {false, false, true};  }
    if (boundary_tag == "v0-v1")    { return {true,  true,  false}; }
    if (boundary_tag == "v0-v2")    { return {true,  false, true};  }
    if (boundary_tag == "v1-v2")    { return {false, true,  true};  }
    if (boundary_tag == "v0-v1-v2") { return {true,  true,  true};  }
    MFEM_ABORT("TriFaceMortarAssembler: unrecognised boundary_tag '"
                  << boundary_tag << "'.");
    return {false, false, false};   // unreachable
}

std::array<double, 3>
TriFaceMortarAssembler::MortarBaryFromPermutation(
     const std::array<int, 3>& mortar_node_perm,
     const std::array<double, 3>& lam_nonmortar)
{
    if (mortar_node_perm[0] == 0 && mortar_node_perm[1] == 1 &&
         mortar_node_perm[2] == 2)
    {
        return lam_nonmortar;
    }
    // Permute components: mortar_q_pt[mortar_node_perm[i]] = nonmortar_q_pt[i].
    std::array<double, 3> result = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) { result[mortar_node_perm[i]] = lam_nonmortar[i]; }
    return result;
}

FaceMortarPairBlock
TriFaceMortarAssembler::AssemblePairConforming(
     const std::vector<TriFaceElement>& nonmortar_elems,
     const std::vector<TriFaceElement>& mortar_elems,
     const std::vector<TriFacePairMatch>& pair_matches,
     const std::string& nonmortar_face_name,
     const std::string& mortar_face_name) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::face_mortar::tri::integrate_pair");

    FaceMortarPairBlock block;
    block.nonmortar_face_name  = nonmortar_face_name;
    block.mortar_face_name = mortar_face_name;

    std::map<int, int> nonmortar_row_of, mortar_col_of;
    DiscoverKeptGtdofs(nonmortar_elems,  block.nonmortar_gtdofs,  nonmortar_row_of);
    DiscoverKeptGtdofs(mortar_elems, block.mortar_gtdofs, mortar_col_of);
    const int n_rows = block.nonmortar_gtdofs.Size();
    const int n_cols = block.mortar_gtdofs.Size();
    block.D.SetSize(n_rows);
    block.D = 0.0;
    // Phase 4.2 / Batch L: A_m is now mfem::SparseMatrix; same
    // pattern as the quad assembler.
    block.A_m = mfem::SparseMatrix(n_rows, n_cols);

    const auto rule = GaussTri3Pt();

    for (const auto& match : pair_matches)
    {
        const TriFaceElement& s = nonmortar_elems[match.nonmortar_idx];
        const TriFaceElement& m = mortar_elems[match.mortar_idx];
        const auto drops = BoundaryTagToDrops(s.boundary_tag);

        // Nonmortar Jacobian for tri-3: J = phys_area / ref_area = 2 * |T_phys|
        // (since |T_ref| = 1/2 and weights sum to 1/2). Multiplying weights
        // by J gives total physical area as expected.
        const double J_nonmortar = 2.0 * [&](){
            const auto& c = s.coords;
            // Cross product magnitude of two edge vectors.
            const double v01[3] = {c(1, 0) - c(0, 0), c(1, 1) - c(0, 1),
                                          c(1, 2) - c(0, 2)};
            const double v02[3] = {c(2, 0) - c(0, 0), c(2, 1) - c(0, 1),
                                          c(2, 2) - c(0, 2)};
            const double cx = v01[1] * v02[2] - v01[2] * v02[1];
            const double cy = v01[2] * v02[0] - v01[0] * v02[2];
            const double cz = v01[0] * v02[1] - v01[1] * v02[0];
            return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }();

        std::array<double, 3>                  D_loc = {0, 0, 0};
        std::array<std::array<double, 3>, 3>   A_loc = {};

        for (int q = 0; q < 3; ++q)
        {
            const auto lam = rule.pts[q];
            const double w = rule.wts[q];
            const double phys_w = w * J_nonmortar;

            const auto M_nonmortar = MTri3DualModified(lam, drops);
            const auto N_nonmortar = NTri3(lam);
            // lam_mortar lives in the mortar element's OWN barycentric
            // frame (MortarBaryFromPermutation handles the nm→mortar
            // vertex-relabel from the perm), so NTri3(lam_mortar)[j]
            // is already mortar local node j's shape function value at
            // the current physical Gauss point. Same fix and rationale
            // as the quad path.
            const auto lam_mortar = MortarBaryFromPermutation(match.mortar_node_perm,
                                                                                lam);
            const auto N_mortar = NTri3(lam_mortar);

            for (int k = 0; k < 3; ++k)
            {
                D_loc[k] += phys_w * N_nonmortar[k];
                for (int l = 0; l < 3; ++l)
                {
                    A_loc[k][l] += phys_w * M_nonmortar[k] * N_mortar[l];
                }
            }
        }

        for (int k_loc = 0; k_loc < 3; ++k_loc)
        {
            const int g_nonmortar = s.gtdofs[k_loc];
            if (g_nonmortar < 0) { continue; }
            const int k_global = nonmortar_row_of[g_nonmortar];
            block.D(k_global) += D_loc[k_loc];
            for (int l_loc = 0; l_loc < 3; ++l_loc)
            {
                const int g_mortar = m.gtdofs[l_loc];
                if (g_mortar < 0) { continue; }
                const int l_global = mortar_col_of[g_mortar];
                block.A_m.Add(k_global, l_global, A_loc[k_loc][l_loc]);
            }
        }
    }

    block.A_m.Finalize();
    return block;
}

// ============================================================================
// MatchConformingFacePairs — quad-4 overload
// ============================================================================

namespace
{
    template <typename FaceElemT>
    double CharacteristicLength(const FaceElemT& e)
    {
        const int n = FaceElemT::NumNodes();
        double lo[3] = { e.coords(0, 0), e.coords(0, 1), e.coords(0, 2) };
        double hi[3] = { lo[0], lo[1], lo[2] };
        for (int v = 1; v < n; ++v)
        {
            for (int d = 0; d < 3; ++d)
            {
                lo[d] = std::min(lo[d], e.coords(v, d));
                hi[d] = std::max(hi[d], e.coords(v, d));
            }
        }
        const double d0 = hi[0] - lo[0];
        const double d1 = hi[1] - lo[1];
        const double d2 = hi[2] - lo[2];
        return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }

    /// For each nonmortar local-node, find the mortar local-node at the same
    /// in-plane physical coords.
    template <typename FaceElemT, std::size_t NV>
    std::array<int, NV> NodePermByCoordMatch(
         const FaceElemT& s, const FaceElemT& m,
         int a_idx, int b_idx, double tol)
    {
        std::array<int, NV> perm{};
        for (std::size_t i = 0; i < NV; ++i) { perm[i] = -1; }

        for (int i = 0; i < static_cast<int>(NV); ++i)
        {
            const double s_a = s.coords(i, a_idx);
            const double s_b = s.coords(i, b_idx);
            int n_match = 0;
            int j_match = -1;
            for (int j = 0; j < static_cast<int>(NV); ++j)
            {
                const double dx = m.coords(j, a_idx) - s_a;
                const double dy = m.coords(j, b_idx) - s_b;
                const double d  = std::sqrt(dx * dx + dy * dy);
                if (d <= tol)
                {
                    ++n_match;
                    j_match = j;
                }
            }
            MFEM_VERIFY(n_match == 1,
                            "NodePermByCoordMatch: nonmortar node " << i << " at ("
                            << s_a << ", " << s_b << ") matched " << n_match
                            << " mortar nodes; expected exactly 1 within tol="
                            << tol << ".");
            perm[i] = j_match;
        }
        return perm;
    }
}  // namespace

std::vector<QuadFacePairMatch>
MatchConformingFacePairs(const std::vector<QuadFaceElement>& nonmortar_elems,
                                  const std::vector<QuadFaceElement>& mortar_elems,
                                  const std::string& perpendicular_axis,
                                  double /*period*/,
                                  double tol_rel)
{
    if (nonmortar_elems.empty() || mortar_elems.empty()) { return {}; }

    const int perp_idx = AxisIndex(perpendicular_axis);
    int a_idx = -1, b_idx = -1;
    {
        const std::array<int, 3> all = {0, 1, 2};
        std::vector<int> in_plane;
        for (int d : all) { if (d != perp_idx) { in_plane.push_back(d); } }
        a_idx = in_plane[0];
        b_idx = in_plane[1];
    }

    // Mortar centroids in-plane.
    const int n_mortar = static_cast<int>(mortar_elems.size());
    std::vector<std::array<double, 2>> mortar_centroids(n_mortar);
    for (int i = 0; i < n_mortar; ++i)
    {
        mortar_centroids[i] = CentroidInPlane(mortar_elems[i], a_idx, b_idx);
    }

    std::vector<QuadFacePairMatch> result;
    result.reserve(nonmortar_elems.size());
    for (int s_idx = 0; s_idx < static_cast<int>(nonmortar_elems.size()); ++s_idx)
    {
        const auto& s = nonmortar_elems[s_idx];
        const auto sc = CentroidInPlane(s, a_idx, b_idx);
        const double char_len = CharacteristicLength(s);
        const double tol = std::max(tol_rel * char_len, 1e-14);

        // Find mortar(s) within tol.
        int n_candidates = 0;
        int mortar_idx_match = -1;
        for (int j = 0; j < n_mortar; ++j)
        {
            const double dx = mortar_centroids[j][0] - sc[0];
            const double dy = mortar_centroids[j][1] - sc[1];
            const double d  = std::sqrt(dx * dx + dy * dy);
            if (d <= tol) { ++n_candidates; mortar_idx_match = j; }
        }
        MFEM_VERIFY(n_candidates >= 1,
                        "MatchConformingFacePairs(quad): nonmortar element " << s_idx
                        << " at centroid (" << sc[0] << ", " << sc[1]
                        << ") has no mortar partner within tol=" << tol);
        MFEM_VERIFY(n_candidates == 1,
                        "MatchConformingFacePairs(quad): nonmortar element " << s_idx
                        << " at centroid (" << sc[0] << ", " << sc[1]
                        << ") has " << n_candidates
                        << " mortar partners within tol=" << tol
                        << "; expected exactly 1.");

        const auto& m = mortar_elems[mortar_idx_match];
        QuadFacePairMatch match;
        match.nonmortar_idx  = s_idx;
        match.mortar_idx = mortar_idx_match;
        match.mortar_node_perm =
             NodePermByCoordMatch<QuadFaceElement, 4>(s, m, a_idx, b_idx, tol);
        result.push_back(match);
    }
    return result;
}

// ============================================================================
// MatchConformingFacePairs — tri-3 overload
// ============================================================================

std::vector<TriFacePairMatch>
MatchConformingFacePairs(const std::vector<TriFaceElement>& nonmortar_elems,
                                  const std::vector<TriFaceElement>& mortar_elems,
                                  const std::string& perpendicular_axis,
                                  double /*period*/,
                                  double tol_rel)
{
    if (nonmortar_elems.empty() || mortar_elems.empty()) { return {}; }

    const int perp_idx = AxisIndex(perpendicular_axis);
    int a_idx = -1, b_idx = -1;
    {
        const std::array<int, 3> all = {0, 1, 2};
        std::vector<int> in_plane;
        for (int d : all) { if (d != perp_idx) { in_plane.push_back(d); } }
        a_idx = in_plane[0];
        b_idx = in_plane[1];
    }

    const int n_mortar = static_cast<int>(mortar_elems.size());
    std::vector<std::array<double, 2>> mortar_centroids(n_mortar);
    for (int i = 0; i < n_mortar; ++i)
    {
        mortar_centroids[i] = CentroidInPlane(mortar_elems[i], a_idx, b_idx);
    }

    std::vector<TriFacePairMatch> result;
    result.reserve(nonmortar_elems.size());
    for (int s_idx = 0; s_idx < static_cast<int>(nonmortar_elems.size()); ++s_idx)
    {
        const auto& s = nonmortar_elems[s_idx];
        const auto sc = CentroidInPlane(s, a_idx, b_idx);
        const double char_len = CharacteristicLength(s);
        const double tol = std::max(tol_rel * char_len, 1e-14);

        int n_candidates = 0;
        int mortar_idx_match = -1;
        for (int j = 0; j < n_mortar; ++j)
        {
            const double dx = mortar_centroids[j][0] - sc[0];
            const double dy = mortar_centroids[j][1] - sc[1];
            const double d  = std::sqrt(dx * dx + dy * dy);
            if (d <= tol) { ++n_candidates; mortar_idx_match = j; }
        }
        MFEM_VERIFY(n_candidates >= 1,
                        "MatchConformingFacePairs(tri): nonmortar element " << s_idx
                        << " has no mortar partner within tol=" << tol);
        MFEM_VERIFY(n_candidates == 1,
                        "MatchConformingFacePairs(tri): nonmortar element " << s_idx
                        << " has " << n_candidates
                        << " mortar partners; expected exactly 1.");

        const auto& m = mortar_elems[mortar_idx_match];
        TriFacePairMatch match;
        match.nonmortar_idx  = s_idx;
        match.mortar_idx = mortar_idx_match;
        match.mortar_node_perm =
             NodePermByCoordMatch<TriFaceElement, 3>(s, m, a_idx, b_idx, tol);
        result.push_back(match);
    }
    return result;
}

// ============================================================================
// TryMatchConformingFacePairs (Phase 4.4 / Batch 4.4-E)
// ============================================================================
//
// Returns std::nullopt when the meshes are non-matching (zero or many
// candidates per nonmortar). Used by BuildLocalPairBlocks to detect
// non-conforming pairs and fall back to the clipped path. Algorithm
// is otherwise identical to MatchConformingFacePairs.

std::optional<std::vector<QuadFacePairMatch>>
TryMatchConformingFacePairs(const std::vector<QuadFaceElement>& nonmortar_elems,
                            const std::vector<QuadFaceElement>& mortar_elems,
                            const std::string& perpendicular_axis,
                            double /*period*/,
                            double tol_rel)
{
    if (nonmortar_elems.empty() || mortar_elems.empty())
    {
        return std::vector<QuadFacePairMatch>{};
    }

    const int perp_idx = AxisIndex(perpendicular_axis);
    int a_idx = -1, b_idx = -1;
    {
        const std::array<int, 3> all = {0, 1, 2};
        std::vector<int> in_plane;
        for (int d : all) { if (d != perp_idx) { in_plane.push_back(d); } }
        a_idx = in_plane[0];
        b_idx = in_plane[1];
    }

    const int n_mortar = static_cast<int>(mortar_elems.size());
    std::vector<std::array<double, 2>> mortar_centroids(n_mortar);
    for (int i = 0; i < n_mortar; ++i)
    {
        mortar_centroids[i] = CentroidInPlane(mortar_elems[i], a_idx, b_idx);
    }

    std::vector<QuadFacePairMatch> result;
    result.reserve(nonmortar_elems.size());
    for (int s_idx = 0; s_idx < static_cast<int>(nonmortar_elems.size()); ++s_idx)
    {
        const auto& s = nonmortar_elems[s_idx];
        const auto sc = CentroidInPlane(s, a_idx, b_idx);
        const double char_len = CharacteristicLength(s);
        const double tol = std::max(tol_rel * char_len, 1e-14);

        int n_candidates = 0;
        int mortar_idx_match = -1;
        for (int j = 0; j < n_mortar; ++j)
        {
            const double dx = mortar_centroids[j][0] - sc[0];
            const double dy = mortar_centroids[j][1] - sc[1];
            const double d  = std::sqrt(dx * dx + dy * dy);
            if (d <= tol) { ++n_candidates; mortar_idx_match = j; }
        }
        if (n_candidates != 1) { return std::nullopt; }

        const auto& m = mortar_elems[mortar_idx_match];
        QuadFacePairMatch match;
        match.nonmortar_idx  = s_idx;
        match.mortar_idx = mortar_idx_match;
        match.mortar_node_perm =
             NodePermByCoordMatch<QuadFaceElement, 4>(s, m, a_idx, b_idx, tol);
        result.push_back(match);
    }
    return result;
}

std::optional<std::vector<TriFacePairMatch>>
TryMatchConformingFacePairs(const std::vector<TriFaceElement>& nonmortar_elems,
                            const std::vector<TriFaceElement>& mortar_elems,
                            const std::string& perpendicular_axis,
                            double /*period*/,
                            double tol_rel)
{
    if (nonmortar_elems.empty() || mortar_elems.empty())
    {
        return std::vector<TriFacePairMatch>{};
    }

    const int perp_idx = AxisIndex(perpendicular_axis);
    int a_idx = -1, b_idx = -1;
    {
        const std::array<int, 3> all = {0, 1, 2};
        std::vector<int> in_plane;
        for (int d : all) { if (d != perp_idx) { in_plane.push_back(d); } }
        a_idx = in_plane[0];
        b_idx = in_plane[1];
    }

    const int n_mortar = static_cast<int>(mortar_elems.size());
    std::vector<std::array<double, 2>> mortar_centroids(n_mortar);
    for (int i = 0; i < n_mortar; ++i)
    {
        mortar_centroids[i] = CentroidInPlane(mortar_elems[i], a_idx, b_idx);
    }

    std::vector<TriFacePairMatch> result;
    result.reserve(nonmortar_elems.size());
    for (int s_idx = 0; s_idx < static_cast<int>(nonmortar_elems.size()); ++s_idx)
    {
        const auto& s = nonmortar_elems[s_idx];
        const auto sc = CentroidInPlane(s, a_idx, b_idx);
        const double char_len = CharacteristicLength(s);
        const double tol = std::max(tol_rel * char_len, 1e-14);

        int n_candidates = 0;
        int mortar_idx_match = -1;
        for (int j = 0; j < n_mortar; ++j)
        {
            const double dx = mortar_centroids[j][0] - sc[0];
            const double dy = mortar_centroids[j][1] - sc[1];
            const double d  = std::sqrt(dx * dx + dy * dy);
            if (d <= tol) { ++n_candidates; mortar_idx_match = j; }
        }
        if (n_candidates != 1) { return std::nullopt; }

        const auto& m = mortar_elems[mortar_idx_match];
        TriFacePairMatch match;
        match.nonmortar_idx  = s_idx;
        match.mortar_idx = mortar_idx_match;
        match.mortar_node_perm =
             NodePermByCoordMatch<TriFaceElement, 3>(s, m, a_idx, b_idx, tol);
        result.push_back(match);
    }
    return result;
}

}  // namespace mortar_pbc
