// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/mortar_3d.py` (basis functions
// and quadrature) + `mortar_pbc/face_mortar_3d.py` (assembler classes
// and matching helper).
//
// This module provides the 3D face-mortar machinery: tri-3 and quad-4
// dual bases (with Wohlmuth modifications for elements that touch a
// face-boundary edge or corner), reference-element quadrature rules,
// and two concrete assembler classes that integrate D and A_m on
// matched nonmortar-mortar face-element pairs.
//
// The Phase 4 scope covers ONLY conforming pairs (1:1 matched nonmortar/
// mortar with same parametric extent). Non-conforming pairs require
// Sutherland-Hodgman polygon clipping, deferred to Phase 3.5 / Phase 5+.
//
// Higher-order element types (line-3, tri-6, quad-8, quad-9, hex-27,
// tet-10) are NOT ported. Their dual bases either don't exist as
// strict bi-orthogonal duals (lumped-positivity obstruction, §4.9.2 of
// the architecture doc) or require basis-transformation / LOR fallbacks
// that are out of scope. The Python prototype includes them for
// negative-result tests; the C++ port keeps the lumped-positivity
// runtime check on the supported types only.
//
// References:
//   * MORTAR_PBC_ARCHITECTURE.md §4 (dual basis derivations)
//   * MORTAR_PBC_ARCHITECTURE.md §4.9 (lumped-positivity obstruction)
//   * MORTAR_PBC_ARCHITECTURE.md §5.2, §5.3 (Wohlmuth modifications)
//   * MORTAR_PBC_ARCHITECTURE.md §11.4 (mixed-element faces)
//   * MORTAR_PBC_ARCHITECTURE.md §11.6 (3D face mortar)
//   * Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.

#pragma once
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mortar_pbc {

// ============================================================================
// Reference shape functions
// ============================================================================

/// Tri-3 (2D simplex, p=1) shape functions in barycentric coords.
/// Vertices at lam = (1,0,0), (0,1,0), (0,0,1). Returns {l1, l2, l3}.
inline std::array<double, 3> NTri3(const std::array<double, 3>& lam) noexcept
{
    return {lam[0], lam[1], lam[2]};
}

/// Quad-4 (bilinear) shape functions on (xi, eta) ∈ [-1, +1]^2.
/// Standard CCW node ordering: (-1,-1), (+1,-1), (+1,+1), (-1,+1).
inline std::array<double, 4> NQuad4(double xi, double eta) noexcept
{
    return {
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    };
}

// ============================================================================
// Reference dual bases
// ============================================================================

/// Tri-3 dual basis (architecture §4, eq. 4.19).
///   M_i(lam) = 4 lam_i - 1.
/// Bi-orthogonal on the reference triangle T (|T| = 1/2):
///   ∫_T M_i N_j dA = δ_ij * (|T|/3).
inline std::array<double, 3> MTri3Dual(const std::array<double, 3>& lam) noexcept
{
    return {
        4.0 * lam[0] - 1.0,
        4.0 * lam[1] - 1.0,
        4.0 * lam[2] - 1.0,
    };
}

/// Quad-4 dual basis (architecture §4, eq. 4.16).
/// Tensor product of the line-2 dual:
///   M_i(xi, eta) = M_line2_dual(xi)_i_xi · M_line2_dual(eta)_i_eta.
/// Node ordering matches NQuad4: (-1,-1), (+1,-1), (+1,+1), (-1,+1).
/// Bi-orthogonal on [-1,+1]^2 (|E| = 4): ∫_E M_i N_j dA = δ_ij.
std::array<double, 4> MQuad4Dual(double xi, double eta) noexcept;

// ============================================================================
// Wohlmuth-modified dual bases (architecture §5.2, §5.3)
// ============================================================================

/// Wohlmuth-modified tri-3 dual basis (eqs. 5.5, 5.6).
///
/// `boundary_nodes` is a 3-tuple of bool flags; b_i = true iff vertex i
/// is on a face-boundary feature (edge or corner) and so its row should
/// be dropped (M_i^mod = 0).
///
/// Cases:
///   0 dropped: standard tri-3 dual.
///   1 dropped: edge-adjacent (eq. 5.5). For dropped vertex i and kept
///              vertices j = (i+1)%3, k = (i+2)%3:
///                M_i = 0
///                M_j = 1/2 + 2 lam_j - 2 lam_k
///                M_k = 1/2 - 2 lam_j + 2 lam_k
///   2 dropped: corner-adjacent (eq. 5.6). The single kept vertex's M
///              is identically 1; the other two are 0.
///   3 dropped: all M_i = 0.
std::array<double, 3> MTri3DualModified(
     const std::array<double, 3>& lam,
     const std::array<bool, 3>& boundary_nodes);

/// Wohlmuth-modified quad-4 dual basis (eqs. 5.8, 5.10).
///
/// Constructed as the tensor product of two line-2 modified duals:
///   side_xi  ∈ {"none", "left", "right", "both"}
///   side_eta ∈ {"none", "bottom", "top", "both"}
///
/// "left"/"right" drop the xi=-1/+1 edge of the quad (nodes {0,3}/{1,2}
/// respectively). "bottom"/"top" drop the eta=-1/+1 edge (nodes {0,1}/
/// {2,3}). "both" drops the whole row of nodes along that direction.
///
/// Implementation maps side_eta to line-2 left/right semantics
/// ("bottom" -> "left", "top" -> "right") and calls
/// MLine2DualModified twice; the quad-4 modified dual is then the
/// outer product, mirroring the unmodified quad-4 dual derivation
/// (§4.16 of the architecture doc).
std::array<double, 4> MQuad4DualModified(
     double xi, double eta,
     const std::string& side_xi  = "none",
     const std::string& side_eta = "none");

// ============================================================================
// Reference-element quadrature rules
// ============================================================================

/// 2D 3x3 Gauss-Legendre tensor product on [-1, +1]^2 (degree 5 each
/// direction, 9 points total).
struct QuadratureQuad3x3
{
    std::array<std::array<double, 2>, 9> pts;   // (xi, eta)
    std::array<double, 9>                wts;
};
QuadratureQuad3x3 GaussQuad3x3();

/// 2D 3-point degree-2 Dunavant rule on the reference triangle T,
/// |T| = 1/2. Returns barycentric (lam_1, lam_2, lam_3) and weights
/// summing to |T| = 1/2.
struct QuadratureTri3Pt
{
    std::array<std::array<double, 3>, 3> pts;   // barycentric
    std::array<double, 3>                wts;
};
QuadratureTri3Pt GaussTri3Pt();

/// 2D 6-point degree-4 Dunavant rule on the reference triangle T,
/// |T| = 1/2. Required by the Phase 4.4 non-conforming face-mortar
/// integration on clipped quad-face sub-triangles: under the
/// barycentric-affine map, the Q1 dual basis × Q1 mortar shape
/// product is degree 4, so degree-2 Dunavant (3 points) underflows
/// for clipped quad sub-tris. Used by AssembleQuadFacePairClipped.
/// (Tri-face clipped sub-tris stay at degree 2, so they keep
/// GaussTri3Pt.)
///
/// Reference: Dunavant 1985, "High degree efficient symmetrical
/// Gaussian quadrature rules for the triangle." 6-point degree-4
/// rule, weights summing to |T| = 1/2.
struct QuadratureTri6Pt
{
    std::array<std::array<double, 3>, 6> pts;   // barycentric
    std::array<double, 6>                wts;
};
QuadratureTri6Pt DunavantTri6Pt();

// ============================================================================
// Pair-match record for conforming face pairs
// ============================================================================
//
// One record per nonmortar element: stores the nonmortar/mortar indices plus
// the mortar_node_perm describing how mortar local nodes correspond
// to nonmortar local nodes.
//
// `mortar_node_perm[i]` = local-node index in the mortar element of
// the mortar node geometrically at nonmortar-element local-node i.
//
// For axis-aligned MakeCartesian3D meshes (the validation cases in
// Phase 4.1), `mortar_node_perm` is always the identity (0, 1, 2, ...);
// the explicit storage exists for general conforming meshes where
// nonmortar/mortar orientations may differ.
//
// We use two separate structs (one for quads with a 4-element perm,
// one for tris with a 3-element perm) so the array sizes are fully
// type-safe — vs. a single dynamic-size struct that would re-introduce
// alloc overhead per pair.

struct QuadFacePairMatch
{
    int nonmortar_idx  = -1;
    int mortar_idx = -1;
    std::array<int, 4> mortar_node_perm = {0, 1, 2, 3};
};

struct TriFacePairMatch
{
    int nonmortar_idx  = -1;
    int mortar_idx = -1;
    std::array<int, 3> mortar_node_perm = {0, 1, 2};
};

/**
 * @brief Mortar assembler for conforming quad-4 face-element pairs.
 *
 * @details Computes per-pair \f$D\f$ (nonmortar diagonal) and \f$A^m\f$
 * (nonmortar-mortar coupling) for a conforming pair of quad-4 face
 * elements. The Wohlmuth-modified dual basis is selected per-element
 * via the `boundary_tag` field on the nonmortar element, so face
 * elements that touch face-boundary edges or corners use the
 * appropriate row-dropping modification.
 *
 * Construction performs a one-time lumped-positivity guard
 * (architecture §4.9.1) — the quad-4 dual basis IS lumped-positive,
 * so this just verifies the implementation. A failure here would
 * indicate a bug in the basis or quadrature.
 *
 * @see QuadFaceElement, QuadFacePairMatch, FaceMortarPairBlock,
 *      MQuad4DualModified, MatchConformingFacePairs
 */
class QuadFaceMortarAssembler
{
public:
    QuadFaceMortarAssembler();
    QuadFaceMortarAssembler(const QuadFaceMortarAssembler&) = delete;
    QuadFaceMortarAssembler& operator=(const QuadFaceMortarAssembler&) = delete;

    /**
     * @brief Assemble \f$(D, A^m)\f$ for a conforming face-element pair set.
     *
     * @param nonmortar_elems     Nonmortar-side face elements.
     * @param mortar_elems        Mortar-side face elements.
     * @param pair_matches        Output of MatchConformingFacePairs;
     *                            one entry per nonmortar element.
     * @param nonmortar_face_name Diagnostic label (e.g. "bottom") for
     *                            the resulting block; default
     *                            "nonmortar".
     * @param mortar_face_name    Diagnostic label for the mortar side;
     *                            default "mortar".
     *
     * @return FaceMortarPairBlock with row indexing by *kept* nonmortar
     *         gtdofs and column indexing by *kept* mortar gtdofs.
     *         Sentinel rows/cols (corner / edge sentinel values) are
     *         dropped during assembly.
     *
     * MPI scope: **local** — no collective communication.
     */
    FaceMortarPairBlock AssemblePairConforming(
         const std::vector<QuadFaceElement>& nonmortar_elems,
         const std::vector<QuadFaceElement>& mortar_elems,
         const std::vector<QuadFacePairMatch>& pair_matches,
         const std::string& nonmortar_face_name = "nonmortar",
         const std::string& mortar_face_name = "mortar") const;

private:
    /// Maps a quad-4 boundary_tag string to (side_xi, side_eta) for
    /// MQuad4DualModified.
    static std::pair<std::string, std::string>
         BoundaryTagToSides(const std::string& boundary_tag);

    /// Phase 3.2.B construction guard (architecture §4.9.1):
    /// computes s_j = ∫ N_j on the reference element via the 3x3 rule
    /// and verifies s_j > 0. Throws on failure.
    static void VerifyLumpedPositivity();

    /// Apply a 4-element node permutation to a nonmortar-side reference
    /// (xi, eta), giving the mortar-side reference (xi, eta).
    static std::array<double, 2> MortarRefFromPermutation(
         const std::array<int, 4>& mortar_node_perm,
         std::array<double, 2> q_pt_nonmortar);

    /// Reorder mortar shape values to match mortar-element local-node
    /// order. For identity permutation this is a no-op.
    static std::array<double, 4> ReorderMortarShape(
         const std::array<double, 4>& N_mortar_at_q,
         const std::array<int, 4>& mortar_node_perm);

    /// Compute per-point Jacobian for an axis-aligned (constant-J) or
    /// general bilinear quad face element.
    double NonmortarJacobian(const QuadFaceElement& nonmortar_elem,
                                std::array<double, 2> q_pt) const;
};

/**
 * @brief Mortar assembler for conforming tri-3 face-element pairs.
 *
 * @details Computes per-pair \f$D\f$ (nonmortar diagonal) and \f$A^m\f$
 * (nonmortar-mortar coupling) for a conforming pair of tri-3 face
 * elements. The Wohlmuth-modified dual basis is selected per-element
 * via the `boundary_tag` field on the nonmortar element.
 *
 * Construction performs a one-time lumped-positivity guard
 * (architecture §4.9.1).
 *
 * @see TriFaceElement, TriFacePairMatch, FaceMortarPairBlock,
 *      MTri3DualModified, MatchConformingFacePairs
 */
class TriFaceMortarAssembler
{
public:
    TriFaceMortarAssembler();
    TriFaceMortarAssembler(const TriFaceMortarAssembler&) = delete;
    TriFaceMortarAssembler& operator=(const TriFaceMortarAssembler&) = delete;

    /**
     * @brief Assemble \f$(D, A^m)\f$ for a conforming tri-3 face-element pair set.
     *
     * @param nonmortar_elems     Nonmortar-side face elements.
     * @param mortar_elems        Mortar-side face elements.
     * @param pair_matches        Output of MatchConformingFacePairs.
     * @param nonmortar_face_name Diagnostic label, default "nonmortar".
     * @param mortar_face_name    Diagnostic label, default "mortar".
     * @return FaceMortarPairBlock with sentinel rows/cols dropped.
     *
     * MPI scope: **local** — no collective communication.
     */
    FaceMortarPairBlock AssemblePairConforming(
         const std::vector<TriFaceElement>& nonmortar_elems,
         const std::vector<TriFaceElement>& mortar_elems,
         const std::vector<TriFacePairMatch>& pair_matches,
         const std::string& nonmortar_face_name = "nonmortar",
         const std::string& mortar_face_name = "mortar") const;

private:
    /// Map a tri-3 boundary_tag string to a 3-tuple of drop flags.
    static std::array<bool, 3>
         BoundaryTagToDrops(const std::string& boundary_tag);

    /// Phase 3.2.B construction guard for tri-3.
    static void VerifyLumpedPositivity();

    /// Apply a 3-element permutation to a nonmortar-side barycentric q_pt,
    /// giving the mortar-side barycentric q_pt.
    static std::array<double, 3> MortarBaryFromPermutation(
         const std::array<int, 3>& mortar_node_perm,
         const std::array<double, 3>& lam_nonmortar);

    /// Reorder mortar shape values to match mortar-element local-node
    /// order under a 3-element permutation.
    static std::array<double, 3> ReorderMortarShape(
         const std::array<double, 3>& N_mortar_at_q,
         const std::array<int, 3>& mortar_node_perm);
};

// ============================================================================
// Conforming-pair matching helpers
// ============================================================================

/**
 * @brief Match conforming quad-4 face pairs by parametric centroid.
 *
 * @param nonmortar_elems     Nonmortar-side face elements.
 * @param mortar_elems        Mortar-side face elements.
 * @param perpendicular_axis  "x", "y", or "z" — the periodic-pair axis.
 * @param period              The signed periodic translation along
 *                            `perpendicular_axis`
 *                            (`mortar_perp - nonmortar_perp`; can be
 *                            \f$\pm L\f$). Currently unused by the
 *                            matcher (in-plane centroid match only)
 *                            but reserved for future use.
 * @param tol_rel             Centroid-match tolerance, relative to the
 *                            nonmortar element's characteristic
 *                            in-plane size. Default 1e-9.
 *
 * @return One QuadFacePairMatch record per nonmortar element, packing
 *         the matched mortar element index and a node permutation
 *         describing how mortar local-node indices correspond to
 *         nonmortar local-node indices. For axis-aligned meshes this
 *         permutation is always the identity (0, 1, 2, 3).
 *
 * @details Throws via MFEM_ABORT if a nonmortar element has no mortar
 * partner within tolerance, or has multiple matches.
 *
 * MPI scope: **local** — no collective communication.
 */
std::vector<QuadFacePairMatch> MatchConformingFacePairs(
     const std::vector<QuadFaceElement>& nonmortar_elems,
     const std::vector<QuadFaceElement>& mortar_elems,
     const std::string& perpendicular_axis,
     double period,
     double tol_rel = 1e-9);

/**
 * @brief Match conforming tri-3 face pairs by parametric centroid.
 *
 * @copydetails MatchConformingFacePairs(const std::vector<QuadFaceElement>&,
 *              const std::vector<QuadFaceElement>&, const std::string&,
 *              double, double)
 */
std::vector<TriFacePairMatch> MatchConformingFacePairs(
     const std::vector<TriFaceElement>& nonmortar_elems,
     const std::vector<TriFaceElement>& mortar_elems,
     const std::string& perpendicular_axis,
     double period,
     double tol_rel = 1e-9);

/**
 * @brief Try to match conforming quad-4 face pairs by parametric centroid.
 *
 * Same algorithm as MatchConformingFacePairs but returns std::nullopt
 * instead of aborting when the meshes are non-matching (zero-candidate
 * or many-candidate nonmortar elements). Used by Phase 4.4
 * BoundaryClassifier3D::BuildLocalPairBlocks to detect non-matching
 * meshes and fall back to the clipped (Axom-based) assembler.
 *
 * @return If every nonmortar element has exactly one mortar partner
 *         within tolerance, returns the QuadFacePairMatch list (same
 *         as MatchConformingFacePairs would). Otherwise returns
 *         std::nullopt — caller should fall back to MatchClippedFacePairs.
 */
std::optional<std::vector<QuadFacePairMatch>> TryMatchConformingFacePairs(
     const std::vector<QuadFaceElement>& nonmortar_elems,
     const std::vector<QuadFaceElement>& mortar_elems,
     const std::string& perpendicular_axis,
     double period,
     double tol_rel = 1e-9);

/**
 * @brief Try to match conforming tri-3 face pairs by parametric centroid.
 *
 * @copydetails TryMatchConformingFacePairs(const std::vector<QuadFaceElement>&,
 *              const std::vector<QuadFaceElement>&, const std::string&,
 *              double, double)
 */
std::optional<std::vector<TriFacePairMatch>> TryMatchConformingFacePairs(
     const std::vector<TriFaceElement>& nonmortar_elems,
     const std::vector<TriFaceElement>& mortar_elems,
     const std::string& perpendicular_axis,
     double period,
     double tol_rel = 1e-9);

}  // namespace mortar_pbc
