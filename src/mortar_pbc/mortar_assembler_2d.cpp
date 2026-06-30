// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/mortar_2d.py` (assembler logic)

#include "mortar_assembler_2d.hpp"

// Caliper instrumentation. We use ExaConstit's existing wrapper from
// `utilities/mechanics_log.hpp`, which dispatches to the real Caliper
// macros when `HAVE_CALIPER` is defined and to no-ops otherwise.
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace mortar_pbc {

// ============================================================================
// Free-function dual basis variants
// ============================================================================

std::array<double, 2> MLine2DualModified(double xi,
                                                        const std::string& corner_side)
{
    if (corner_side == "none")  { return MLine2Dual(xi); }
    if (corner_side == "left")  { return {0.0, 1.0}; }
    if (corner_side == "right") { return {1.0, 0.0}; }
    if (corner_side == "both")  { return {0.0, 0.0}; }
    MFEM_ABORT("MLine2DualModified: unknown corner_side '"
                  << corner_side << "'; expected one of "
                  << "{'none', 'left', 'right', 'both'}.");
    return {0.0, 0.0};   // unreachable; silence warnings
}

// ============================================================================
// Gauss-Legendre quadrature (3-point on [-1, 1])
// ============================================================================

namespace
{
    constexpr int kGL3NumPoints = 3;
    // sqrt(3/5) = 0.77459666924148340427791481488...
    const std::array<double, kGL3NumPoints> kGL3Pts = {
        -std::sqrt(0.6), 0.0, std::sqrt(0.6)
    };
    constexpr std::array<double, kGL3NumPoints> kGL3Wts = {
        5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0
    };

    // Tolerance for the overlap-segment "skip-if-empty" check. The Python
    // prototype uses `1e-14 * max(|+ element length|, 1.0)`; we mirror that
    // exactly to preserve bit-for-bit parity.
    constexpr double kOverlapRelTol = 1e-14;
}  // namespace

// ============================================================================
// MortarAssembler2D::AssemblePair
// ============================================================================

MortarBlock2D
MortarAssembler2D::AssemblePair(const EdgeInfo3D& plus_edge,
                                            const EdgeInfo3D& minus_edge) const
{
    // Caliper-mark the per-pair integration. Per-pair granularity matches
    // the §P4.6.4 instrumentation plan ("mortar_pbc::edge_mortar::integrate_pair").
    CALI_CXX_MARK_SCOPE("mortar_pbc::edge_mortar::integrate_pair");

    // ----- Preconditions -----
    MFEM_VERIFY(plus_edge.parametric_axis == minus_edge.parametric_axis,
                    "MortarAssembler2D::AssemblePair: parametric axes differ "
                    "between + edge ('" << plus_edge.parametric_axis
                    << "') and - edge ('" << minus_edge.parametric_axis << "')");
    {
        const double plus_extent  = plus_edge.edge_max  - plus_edge.edge_min;
        const double minus_extent = minus_edge.edge_max - minus_edge.edge_min;
        const double scale = std::max(std::abs(plus_extent), 1.0);
        MFEM_VERIFY(std::abs(plus_extent - minus_extent) <= 1e-12 * scale,
                        "MortarAssembler2D::AssemblePair: edge extents differ "
                        "(plus=" << plus_extent << ", minus=" << minus_extent
                        << "). Periodic translation requires identical extents.");
    }

    const int n_plus  = plus_edge.NumNodes();
    const int n_minus = minus_edge.NumNodes();

    MortarBlock2D block;
    block.A_m.SetSize(n_plus, n_minus);
    block.A_m = 0.0;
    block.D_nm.SetSize(n_plus);
    block.D_nm = 0.0;
    block.plus_edge_name  = plus_edge.label;
    block.minus_edge_name = minus_edge.label;

    // ---------------------------------------------- loop over + elements ---
    for (const auto& plus_elem : plus_edge.elements)
    {
        const int p_n0 = plus_elem.first;
        const int p_n1 = plus_elem.second;

        // Physical-edge-coord endpoints of this + element.
        const auto plus_phys = ParamEndpoints(plus_edge, p_n0, p_n1);
        const double plus_phys_lo = plus_phys.first;
        const double plus_phys_hi = plus_phys.second;
        if (plus_phys_hi <= plus_phys_lo) { continue; }

        // dphys / dxi on the + parent element (xi in [-1, 1]).
        const double plus_jacobian = 0.5 * (plus_phys_hi - plus_phys_lo);

        // Identify which side(s) (if any) of this element touch a Dirichlet
        // corner; selects the dual basis variant used on this element.
        const std::string corner_side = CornerSide(p_n0, p_n1);

        // ----- (1) D^{nm} contribution from this + element -----
        // D_kk = ∫ N^+_k dA, using STANDARD N (not modified M); this is
        // the *measure* the nonmortar node carries. For a line-2 element with
        // constant Jacobian J, ∫_-1^1 N_k(ξ) J dξ = J, i.e. each endpoint
        // receives J = (phys_hi - phys_lo)/2.
        for (int p_node_idx : {p_n0, p_n1})
        {
            if (p_node_idx < 0) { continue; }     // corner sentinel: row dropped
            block.D_nm(p_node_idx) += plus_jacobian;
        }

        // ----- (2) A^m contribution: integrate over each - element overlap ---
        for (const auto& minus_elem : minus_edge.elements)
        {
            const int m_n0 = minus_elem.first;
            const int m_n1 = minus_elem.second;

            const auto minus_phys = ParamEndpoints(minus_edge, m_n0, m_n1);
            const double minus_phys_lo = minus_phys.first;
            const double minus_phys_hi = minus_phys.second;
            if (minus_phys_hi <= minus_phys_lo) { continue; }

            // Interval intersection in physical edge coords.
            const double overlap_lo = std::max(plus_phys_lo, minus_phys_lo);
            const double overlap_hi = std::min(plus_phys_hi, minus_phys_hi);
            const double scale = std::max(std::abs(plus_phys_hi - plus_phys_lo), 1.0);
            if (overlap_hi - overlap_lo <= kOverlapRelTol * scale) { continue; }

            IntegrateOverlapSegment(
                 block.A_m,
                 {p_n0, p_n1},
                 {m_n0, m_n1},
                 {plus_phys_lo, plus_phys_hi},
                 {minus_phys_lo, minus_phys_hi},
                 {overlap_lo, overlap_hi},
                 corner_side);
        }
    }

    return block;
}

// ============================================================================
// MortarAssembler2D::IntegrateOverlapSegment
// ============================================================================

void MortarAssembler2D::IntegrateOverlapSegment(
     mfem::DenseMatrix& A_m,
     std::pair<int, int> plus_local_nodes,
     std::pair<int, int> minus_local_nodes,
     std::pair<double, double> plus_parent_phys,
     std::pair<double, double> minus_parent_phys,
     std::pair<double, double> overlap_phys,
     const std::string& corner_side) const
{
    const double overlap_lo = overlap_phys.first;
    const double overlap_hi = overlap_phys.second;

    // dphys / d(eta) on the overlap, where eta is the GL reference coord.
    const double overlap_jacobian = 0.5 * (overlap_hi - overlap_lo);
    const double overlap_phys_mid = 0.5 * (overlap_hi + overlap_lo);

    const double plus_phys_lo = plus_parent_phys.first;
    const double plus_phys_hi = plus_parent_phys.second;
    const double plus_parent_mid         = 0.5 * (plus_phys_hi + plus_phys_lo);
    const double plus_parent_half_length = 0.5 * (plus_phys_hi - plus_phys_lo);

    const double minus_phys_lo = minus_parent_phys.first;
    const double minus_phys_hi = minus_parent_phys.second;
    const double minus_parent_mid         = 0.5 * (minus_phys_hi + minus_phys_lo);
    const double minus_parent_half_length = 0.5 * (minus_phys_hi - minus_phys_lo);

    const int p_n0 = plus_local_nodes.first;
    const int p_n1 = plus_local_nodes.second;
    const int m_n0 = minus_local_nodes.first;
    const int m_n1 = minus_local_nodes.second;

    for (int gp = 0; gp < kGL3NumPoints; ++gp)
    {
        const double gp_eta    = kGL3Pts[gp];
        const double gp_weight = kGL3Wts[gp];

        // Physical edge coord at this Gauss point.
        const double phys_at_gp = overlap_phys_mid + overlap_jacobian * gp_eta;
        // Reference coord on each parent element.
        const double xi_on_plus  = (phys_at_gp - plus_parent_mid)  / plus_parent_half_length;
        const double xi_on_minus = (phys_at_gp - minus_parent_mid) / minus_parent_half_length;

        // Dual basis on + element (with corner modification if applicable).
        std::array<double, 2> M_at;
        if (corner_side == "none") {
            M_at = MLine2Dual(xi_on_plus);
        } else {
            M_at = MLine2DualModified(xi_on_plus, corner_side);
        }
        // Standard line-2 shape on - element.
        const std::array<double, 2> N_minus_at = NLine2(xi_on_minus);

        // Physical-coord weight: w_eta * (dphys / d eta).
        const double phys_weight = gp_weight * overlap_jacobian;

        // Accumulate into A^m. Drop rows for + corner sentinels (those
        // DOFs are Dirichlet) and cols for - corner sentinels (those
        // values are also prescribed = 0, so they don't need constraint
        // columns).
        const std::array<int, 2>    p_idx = {p_n0, p_n1};
        const std::array<double, 2> p_M   = {M_at[0], M_at[1]};
        const std::array<int, 2>    m_idx = {m_n0, m_n1};
        const std::array<double, 2> m_N   = {N_minus_at[0], N_minus_at[1]};

        for (int a = 0; a < 2; ++a)
        {
            if (p_idx[a] < 0) { continue; }
            for (int b = 0; b < 2; ++b)
            {
                if (m_idx[b] < 0) { continue; }
                A_m(p_idx[a], m_idx[b]) += phys_weight * p_M[a] * m_N[b];
            }
        }
    }
}

// ============================================================================
// MortarAssembler2D::ParamEndpoints
// ============================================================================

std::pair<double, double>
MortarAssembler2D::ParamEndpoints(const EdgeInfo3D& edge,
                                              int node_a_idx, int node_b_idx) const
{
    const int axis = edge.ParamAxisColumn();

    auto coord_or_sentinel = [&](int node_idx) -> double {
        if (node_idx == kEdgeNodeLeftCornerSentinel)  { return edge.edge_min; }
        if (node_idx == kEdgeNodeRightCornerSentinel) { return edge.edge_max; }
        MFEM_ASSERT(node_idx >= 0 && node_idx < edge.NumNodes(),
                        "ParamEndpoints: node_idx " << node_idx
                        << " out of range [0, " << edge.NumNodes() << ")");
        return edge.coords(node_idx, axis);
    };

    const double a_phys = coord_or_sentinel(node_a_idx);
    const double b_phys = coord_or_sentinel(node_b_idx);
    if (a_phys <= b_phys) { return {a_phys, b_phys}; }
    return {b_phys, a_phys};
}

// ============================================================================
// MortarAssembler2D::CornerSide
// ============================================================================

std::string MortarAssembler2D::CornerSide(int node1_idx,
                                                         int node2_idx) noexcept
{
    const bool n1_is_corner = (node1_idx == kEdgeNodeLeftCornerSentinel
                                        || node1_idx == kEdgeNodeRightCornerSentinel);
    const bool n2_is_corner = (node2_idx == kEdgeNodeLeftCornerSentinel
                                        || node2_idx == kEdgeNodeRightCornerSentinel);
    if (n1_is_corner && n2_is_corner) { return "both"; }
    if (n1_is_corner)                 { return "left"; }
    if (n2_is_corner)                 { return "right"; }
    return "none";
}

}  // namespace mortar_pbc
