// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-1 — closed-form inverse-isoparametric maps
// for axis-aligned face elements.
//
// For non-conforming face mortar (Phase 4.4), each clipped sub-triangle
// quadrature point lives in 2D-projected (a, b) physical coords and
// must be mapped back into the *parent* element's reference frame:
//   * QuadFaceElement (Q1 axis-aligned) → (xi, eta) in [-1, +1]^2
//   * TriFaceElement  (P1)              → barycentric (lam_0, lam_1, lam_2)
//
// For axis-aligned grids (the Phase 4.4 scope) both inverse maps are
// closed-form:
//   * Q1 axis-aligned: bilinear collapses to affine; closed-form
//     pseudo-inverse via dot products with ξ / η edge vectors.
//   * P1: barycentric coords from Cramer's rule on the 2×2 affine system.
//
// These maps are needed by AssembleQuadFacePairClipped /
// AssembleTriFacePairClipped (Batch 4.4-D-2/3) and live in their own
// header so they can be tested independently of Axom (Batch 4.4-D-1).
//
// Architecture doc §11.6 spells out the same `locate_mortar` interface
// these functions provide (closed-form for axis-aligned; Newton in
// the general case which we do not implement here).

#pragma once

#include "types_3d.hpp"

#include <array>

namespace mortar_pbc
{

/// Closed-form inverse map for an axis-aligned Q1 quad face element.
///
/// Maps a 2D-projected physical point `(a, b)` (with `a_idx`, `b_idx`
/// the column indices in `coords` selecting the two non-perpendicular
/// 3D axes) to the element's reference (xi, eta) in [-1, +1]^2.
///
/// Assumptions:
///   * Element is a Q1 quad with 4 nodes ordered CCW from outward
///     normal: vertex 0, 1, 2, 3 → reference (-1, -1), (+1, -1),
///     (+1, +1), (-1, +1).
///   * Element is axis-aligned in the (a, b) projection plane —
///     i.e. each 3D edge of the quad aligns with exactly one
///     parametric direction (xi or eta). True for cubic-RVE meshes
///     with axis-aligned face elements; not for skewed quads.
///
/// Algorithm: vertex 0 → vertex 1 spans `+ξ` direction; vertex 0 →
/// vertex 3 spans `+η` direction. For axis-aligned quads these two
/// vectors are orthogonal in the (a, b) plane, so the inverse is a
/// pair of dot products (no matrix solve needed). Closed-form, no
/// Newton iteration.
///
/// @param[in] elem    the Q1 quad face element
/// @param[in] a_idx   column in coords for the "a" projection axis
/// @param[in] b_idx   column in coords for the "b" projection axis
/// @param[in] a, b    physical coordinates of the query point
/// @return {xi, eta} in [-1, +1]^2
std::array<double, 2> InverseMapQuad2DAxisAligned(
    const QuadFaceElement& elem, int a_idx, int b_idx,
    double a, double b);

/// Closed-form inverse map for a P1 tri face element.
///
/// Maps a 2D-projected physical point `(a, b)` to the element's
/// barycentric coordinates `(lam_0, lam_1, lam_2)`. For affine
/// (P1) triangles the inverse is exact via Cramer's rule on the
/// 2×2 system.
///
/// Assumptions:
///   * Element is a P1 tri with 3 nodes ordered CCW from outward
///     normal.
///   * Triangle is non-degenerate in the (a, b) projection (i.e.
///     2D area is non-zero).
///
/// @param[in] elem    the P1 tri face element
/// @param[in] a_idx   column in coords for the "a" projection axis
/// @param[in] b_idx   column in coords for the "b" projection axis
/// @param[in] a, b    physical coordinates of the query point
/// @return {lam_0, lam_1, lam_2} satisfying lam_0 + lam_1 + lam_2 = 1
std::array<double, 3> InverseMapTri2D(
    const TriFaceElement& elem, int a_idx, int b_idx,
    double a, double b);

}  // namespace mortar_pbc
