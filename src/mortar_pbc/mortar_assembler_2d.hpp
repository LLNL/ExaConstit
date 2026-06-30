// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/mortar_2d.py`
//
// Build the 1D mortar coupling matrices A^m and D^{nm} for a single
// (+, -) edge pair of a 3D RVE. The output of this module feeds the
// global constraint matrix C built by ConstraintBuilder3D.
//
// In the C++ port, this assembler operates on `EdgeInfo3D` (the 3D
// types), not on a separate `EdgeInfo2D`. The "2d" suffix on the class
// name refers to the codimension of the integrand (1D mortar lives in
// codim-1 of a 2D ambient space, even though here the ambient space is
// 3D: each box edge is parametrised by one coordinate while the other
// two are constant). This matches the Python prototype's naming.
//
// References:
//   * MORTAR_PBC_ARCHITECTURE.md §3 (mortar method theory)
//   * MORTAR_PBC_ARCHITECTURE.md §4.2 (line-2 dual basis)
//   * MORTAR_PBC_ARCHITECTURE.md §5.1 (line-2 Wohlmuth modification)
//   * MORTAR_PBC_ARCHITECTURE.md §11.5 (3D edge mortar)
//   * Lopes et al. CMAME 384 (2021) 113930, Eqs. (C.1)/(C.2)

#pragma once
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <string>
#include <utility>

namespace mortar_pbc {

// ============================================================================
// Reference shape functions and dual basis (line-2 element, ξ ∈ [-1, 1])
// ============================================================================
//
// These are inline `constexpr`-compatible free functions (not constexpr
// because std::pair isn't constexpr-default in some toolchains we may
// support; behaviour-wise they ARE constexpr).
//
// All four pairs of routines below take a single reference coordinate
// `xi` ∈ [-1, +1] and return (value_at_node_0, value_at_node_1).

/// Standard line-2 (linear Lagrange) shape functions on [-1, 1].
///
///   N_0(ξ) = (1 - ξ)/2,  N_1(ξ) = (1 + ξ)/2.
///
/// Partition of unity: N_0 + N_1 = 1. Both non-negative on [-1, 1].
/// Used as the trial basis for displacement (nonmortar-side D^{nm} integrand
/// and mortar-side A^m integrand).
inline std::array<double, 2> NLine2(double xi) noexcept
{
    return { 0.5 * (1.0 - xi), 0.5 * (1.0 + xi) };
}

/// Line-2 dual basis (Lopes Eq. C.1) bi-orthogonal to the standard basis.
///
///   M_0(ξ) = (1 - 3ξ)/2,  M_1(ξ) = (1 + 3ξ)/2.
///
/// Bi-orthogonality on the reference element:
///   ∫_{-1}^{+1} M_k(ξ) N_l(ξ) dξ = δ_{kl}.
///
/// NOTE: M_0 is NEGATIVE for ξ > 1/3 and M_1 negative for ξ < -1/3.
/// This sign change is essential for bi-orthogonality and it means
/// individual entries of A^m can be negative — that's fine; only the
/// moment statements (constant and linear field reproduction) need to
/// hold globally.
inline std::array<double, 2> MLine2Dual(double xi) noexcept
{
    return { 0.5 * (1.0 - 3.0 * xi), 0.5 * (1.0 + 3.0 * xi) };
}

/// Wohlmuth-modified dual basis (Lopes Eq. C.2) for elements that touch a
/// Dirichlet corner.
///
/// `corner_side` selects WHICH local endpoint of the + element is the
/// corner:
///   "none"  : no corner; returns standard MLine2Dual(xi).
///   "left"  : node 0 (ξ=-1) is the corner -> M_0 = 0, M_1 = 1
///             (transfer everything to node 1)
///   "right" : node 1 (ξ=+1) is the corner -> M_0 = 1, M_1 = 0
///   "both"  : both endpoints are corners -> M_0 = M_1 = 0 (empty constraint)
///
/// The "none" branch is used by the quad-4 dual-modified tensor product
/// (face_mortar_assembler_3d) when only one parametric direction needs
/// modification; the edge mortar (this file) typically branches on
/// "none" before calling so it can use the simpler MLine2Dual directly.
///
/// These DELIBERATELY break bi-orthogonality on corner segments; they are
/// the price paid to avoid over-constraining the corner DOF. See
/// architecture §5.1 / §5.4 for the mathematical justification and
/// §11.5 for the 3D edge-mortar context.
std::array<double, 2> MLine2DualModified(double xi, const std::string& corner_side);

// ============================================================================
// Gauss-Legendre quadrature (3-point on [-1, 1])
// ============================================================================
//
// Integrates polynomials of degree ≤ 5 exactly. The integrand here is a
// product of two linears (degree 2) per Gauss-point loop, so 2-point
// would suffice; 3-point is used for robustness on the *segment* (which
// subdivides the parent element) where the effective polynomial degree
// can rise slightly due to compositions.
//
// Defined in the implementation as constexpr arrays.

/**
 * @brief Assembled mortar quantities for one (+, -) edge pair.
 *
 * @details Indexing of `A_m` and `D_nm` is by position along the edge
 * among interior (non-corner) nodes, ordered in increasing parametric
 * coord. Corner sentinels (-1, -2) are NOT present as indices: they
 * were dropped during assembly because corner DOFs are essential /
 * Dirichlet-pinned elsewhere.
 */
struct MortarBlock2D
{
    /// \f$(n_+, n_-)\f$ coupling matrix:
    /// \f$A^m[k, l] = \int_\Gamma M_k(\xi)\, N^-_l(\zeta(\xi))\, dA\f$.
    mfem::DenseMatrix A_m;
    /// \f$(n_+,)\f$ diagonal lumping:
    /// \f$D^{nm}[k] = \int_\Gamma N^+_k\, dA\f$.
    mfem::Vector D_nm;
    /// Name of the non-mortar (+) edge. For 3D edges, this is the edge label.
    std::string plus_edge_name;
    /// Name of the mortar (-) edge.
    std::string minus_edge_name;
};

/**
 * @brief Line-2 mortar coupling assembler for periodic edge pairs.
 *
 * @details Computes the per-pair coupling matrix \f$A^m\f$ and the
 * diagonal mass vector \f$D^{nm}\f$ that together encode one row-block
 * of the global periodic constraint matrix \f$C\f$ for a single pair
 * of opposite edges of a 3D box RVE.
 *
 * The class is **stateless** — no construction parameters, no internal
 * caches. Each call to AssemblePair() is independent; this is essential
 * for thread-safety in case the constraint builder ever needs to
 * assemble multiple pairs in parallel.
 *
 * **Usage:**
 * @code
 *    MortarAssembler2D assembler;          // stateless; no setup
 *    const auto& nm_edge = classifier.edges.at("x-bottom-front");
 *    const auto& m_edge  = classifier.edges.at("x-top-back");
 *    MortarBlock2D block = assembler.AssemblePair(nm_edge, m_edge);
 * @endcode
 *
 * **Algorithm (per pair):**
 *  1. Loop over + (nonmortar) elements (1D line-2 segments along the +
 *     edge).
 *  2. For each + element, accumulate \f$D^{nm}\f$ contributions: the
 *     standard \f$N^+_k\f$ integrates to the segment's Jacobian,
 *     distributed equally to both endpoints.
 *  3. Find each - element overlapping this + element's parametric range
 *     (interval intersection on the parametric axis).
 *  4. Integrate \f$M_k(\xi_+) N^-_l(\xi_-)\f$ over each overlap segment
 *     using 3-point Gauss quadrature; accumulate into \f$A^m\f$.
 *  5. Drop entries corresponding to corner sentinels (rows from + side,
 *     cols from - side).
 *
 * @see MortarBlock2D, EdgeInfo3D, MLine2Dual, MLine2DualModified
 */
class MortarAssembler2D
{
public:
    MortarAssembler2D() = default;
    // Non-copyable / non-movable — there's no state but we want
    // consistent behaviour.
    MortarAssembler2D(const MortarAssembler2D&) = delete;
    MortarAssembler2D& operator=(const MortarAssembler2D&) = delete;

    /**
     * @brief Assemble \f$A^m\f$ and \f$D^{nm}\f$ for one pair of opposite
     *        edges.
     *
     * @param plus_edge   The nonmortar edge (carries the constraint rows
     *                    / Lagrange-multiplier DOFs).
     * @param minus_edge  The mortar edge.
     * @return MortarBlock2D containing \f$A^m\f$, \f$D^{nm}\f$, and the
     *         edge labels.
     *
     * @details For 3D periodic edges this follows the convention in
     * BoundaryClassifier3D where one of every 4-edge group is the
     * mortar and the other 3 are nonmortar.
     *
     * MPI scope: **local** — no collective communication.
     *
     * @pre `plus_edge.parametric_axis == minus_edge.parametric_axis`
     * @pre `plus_edge.edge_max - plus_edge.edge_min ==
     *      minus_edge.edge_max - minus_edge.edge_min` (identical
     *      parametric extents).
     *
     * Failures throw via MFEM_VERIFY.
     */
    MortarBlock2D AssemblePair(const EdgeInfo3D& plus_edge,
                                        const EdgeInfo3D& minus_edge) const;

private:
    // ---------------------------------------------------------- internals ---

    /// Integrate M_k(ξ_+) · N^-_l(ξ_-) over one overlap segment using
    /// 3-point Gauss-Legendre quadrature, accumulating into `A_m`.
    ///
    /// `corner_side` selects between the standard dual basis and the
    /// Wohlmuth-modified variant:
    ///   "none"  -> standard dual (MLine2Dual)
    ///   "left"  -> Wohlmuth left  (MLine2DualModified, side="left")
    ///   "right" -> Wohlmuth right (MLine2DualModified, side="right")
    ///   "both"  -> Wohlmuth both  (M = 0; segment skipped)
    void IntegrateOverlapSegment(
         mfem::DenseMatrix& A_m,
         std::pair<int, int> plus_local_nodes,
         std::pair<int, int> minus_local_nodes,
         std::pair<double, double> plus_parent_phys,
         std::pair<double, double> minus_parent_phys,
         std::pair<double, double> overlap_phys,
         const std::string& corner_side) const;

    /// Resolve corner-sentinel indices to physical edge endpoints.
    /// Returns (lo, hi) with lo <= hi. See `EdgeInfo3D::elements` docs for
    /// the sentinel convention.
    std::pair<double, double> ParamEndpoints(
         const EdgeInfo3D& edge, int node_a_idx, int node_b_idx) const;

    /// Classify a + element by which local endpoint(s) are corner sentinels.
    /// Returns one of {"none", "left", "right", "both"}.
    ///
    /// Note on naming: "left"/"right" refer to LOCAL node ordering of the
    /// element (node 0 corresponds to local ξ=-1, node 1 to local ξ=+1).
    /// This is the convention the dual basis modifications in Eq. (C.2)
    /// are stated in (M_0 = 0 means "node 0 is corner").
    static std::string CornerSide(int node1_idx, int node2_idx) noexcept;
};

}  // namespace mortar_pbc
