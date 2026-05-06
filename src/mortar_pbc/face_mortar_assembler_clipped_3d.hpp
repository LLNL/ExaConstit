// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-2 — non-conforming face mortar assembler
// for Q1 quad-quad face-element pairs.
//
// This is the algorithmic core of Phase 4.4. The function
// AssembleQuadFacePairClipped consumes:
//   * the nonmortar and mortar Q1 quad face-element lists for one
//     periodic face pair,
//   * the per-nonmortar fan-triangulated overlap geometry produced
//     by ClipQuadFacePairs (Batch 4.4-C),
// and produces a FaceMortarPairBlock matching the AssemblePairConforming
// interface — same D vector, same A_m sparse matrix shape, same gtdof
// row/column indexing.
//
// The D-vs-A_m domain split (Phase 4 plan §P4.4.6.10, architecture
// doc §3.5):
//   * D entries are accumulated PER FULL NONMORTAR ELEMENT using the
//     existing conforming inner loop (9-point Gauss-Legendre on the
//     parent reference quad). This loop is shared with the conforming
//     assembler — same code, same result.
//   * A_m entries are accumulated PER CLIPPED SUB-TRIANGLE using the
//     6-point Dunavant rule (degree 4 — required because the bilinear
//     dual-modified basis × bilinear mortar shape product is degree 4
//     in the sub-triangle's barycentric parameterization).
//
// Wohlmuth corner/edge dual-basis modifications (architecture §5.3) are
// applied ONLY on the nonmortar side — same as the conforming case.
// The tag dispatch (BoundaryTagToSides) is replicated as a free function
// here.
//
// Mortar-side basis evaluation uses the NATURAL mortar local-node
// order — no MortarRefFromPermutation / ReorderMortarShape needed.
// In the clipped path, the inverse-iso-map gives us mortar (xi, eta)
// directly from physical (a, b), and we evaluate NQuad4 on the mortar's
// own reference frame. The scatter step pairs N_mortar[l_loc] with
// m.gtdofs[l_loc] directly — same shape as the conforming path's
// scatter, but no permutation indirection.

#pragma once

#include "face_mortar_match_3d.hpp"  // ClippedSubTriangulation
#include "types_3d.hpp"

#include <string>
#include <vector>

namespace mortar_pbc
{

/**
 * @brief Assemble the (D, A^m) block for a non-conforming Q1 quad-quad
 *        face-mortar pair set.
 *
 * @param nonmortar_elems         Nonmortar-side quad face elements (- side).
 * @param mortar_elems            Mortar-side quad face elements (+ side).
 * @param sub_tris                Per-nonmortar fan-triangulated overlap
 *                                geometry from ClipQuadFacePairs.
 * @param perpendicular_axis      Axis normal to the periodic face, one of
 *                                "x" / "y" / "z". Determines the (a, b)
 *                                projection axes used by the inverse-
 *                                isoparametric maps.
 * @param nonmortar_face_name     Diagnostic label (default "nonmortar").
 * @param mortar_face_name        Diagnostic label (default "mortar").
 * @return FaceMortarPairBlock with row indexing by *kept* nonmortar gtdofs
 *         and column indexing by *kept* mortar gtdofs (sentinel-aware
 *         drop, matching AssemblePairConforming).
 *
 * MPI scope: **local** — no collective communication.
 *
 * @details
 *   For each nonmortar element s:
 *     1. D contribution (Pass 1, full-element):
 *        Walk the canonical 9-point Gauss-Legendre rule on the parent
 *        reference quad. At each q-point evaluate the dual-modified
 *        nonmortar basis M_dual(xi_nm, eta_nm) with sides selected by
 *        s.boundary_tag, and the standard nonmortar shape N_nm. Accumulate
 *        D_loc[k] += phys_w * N_nm[k]. (Wohlmuth biorthogonality lumps
 *        D to its diagonal once integrated over the full element.)
 *     2. A^m contribution (Pass 2, per-sub-triangle):
 *        For each sub-triangle owned by s:
 *          * Mortar partner m = mortar_elems[sub_tri.m_idx].
 *          * Walk DunavantTri6Pt on the sub-triangle's reference simplex.
 *          * For each (lam_0, lam_1, lam_2) q-point:
 *              - Compute physical (a, b) = lam · sub_tri.verts_ab.
 *              - Inverse-iso-map: (xi_nm, eta_nm) =
 *                InverseMapQuad2DAxisAligned(s, ...).
 *              - Inverse-iso-map: (xi_m, eta_m) =
 *                InverseMapQuad2DAxisAligned(m, ...).
 *              - sub_phys_w = w_q * 2 * sub_tri.area.
 *              - M_dual_nm = MQuad4DualModified(xi_nm, eta_nm, sides on s).
 *              - N_mortar  = NQuad4(xi_m, eta_m).
 *              - A_loc[k][l] += sub_phys_w * M_dual_nm[k] * N_mortar[l].
 *     3. Scatter D_loc and A_loc into the global block (sentinel-aware
 *        drop).
 *
 *   On conforming meshes (where each nonmortar has exactly one mortar
 *   partner and the clipped sub-triangulation tile-covers each parent
 *   quad), this produces a FaceMortarPairBlock numerically equal (to FP
 *   roundoff) to AssemblePairConforming's output. That equivalence is
 *   the central correctness check in test_face_mortar_assembler_clipped_3d
 *   (Batch 4.4-D-2 sanity test).
 *
 * @see ClippedSubTriangulation, FaceMortarPairBlock, MQuad4DualModified,
 *      InverseMapQuad2DAxisAligned, DunavantTri6Pt
 */
FaceMortarPairBlock AssembleQuadFacePairClipped(
    const std::vector<QuadFaceElement>& nonmortar_elems,
    const std::vector<QuadFaceElement>& mortar_elems,
    const ClippedSubTriangulation& sub_tris,
    const std::string& perpendicular_axis,
    const std::string& nonmortar_face_name = "nonmortar",
    const std::string& mortar_face_name = "mortar");

/**
 * @brief Assemble the (D, A^m) block for a non-conforming P1 tri-tri
 *        face-mortar pair set.
 *
 * @copydetails AssembleQuadFacePairClipped(const std::vector<QuadFaceElement>&,
 *              const std::vector<QuadFaceElement>&, const ClippedSubTriangulation&,
 *              const std::string&, const std::string&, const std::string&)
 *
 * @details Mirrors AssembleQuadFacePairClipped with three element-type-
 * specific changes:
 *   1. Quadrature on clipped sub-triangles: `GaussTri3Pt` (degree 2)
 *      suffices because P1·P1 = degree 2 in barycentric, so the same
 *      rule used by the conforming tri path is correct here too.
 *      (Q1·Q1 needed the bumped-up DunavantTri6Pt rule; tri faces don't.)
 *   2. D-side Jacobian: `J = 2 * |T_phys|` via 3D cross-product
 *      magnitude, mirroring the conforming tri path. No axis-alignment
 *      assumption — works for arbitrary tri faces.
 *   3. Inverse-iso-map: `InverseMapTri2D` (Cramer's rule on the 2×2
 *      affine system) returns barycentrics directly. Both nonmortar
 *      and mortar tri parents use this map.
 *
 * Boundary-tag dispatch uses `BoundaryTagToDropsTri` (drops vector
 * for `MTri3DualModified`) instead of the quad's side-selector pair.
 *
 * @see ClippedSubTriangulation, FaceMortarPairBlock, MTri3DualModified,
 *      InverseMapTri2D, GaussTri3Pt, AssembleQuadFacePairClipped
 */
FaceMortarPairBlock AssembleTriFacePairClipped(
    const std::vector<TriFaceElement>& nonmortar_elems,
    const std::vector<TriFaceElement>& mortar_elems,
    const ClippedSubTriangulation& sub_tris,
    const std::string& perpendicular_axis,
    const std::string& nonmortar_face_name = "nonmortar",
    const std::string& mortar_face_name = "mortar");

}  // namespace mortar_pbc
