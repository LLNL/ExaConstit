// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-1 — inverse-isoparametric map implementations.
// See face_mortar_inverse_map_3d.hpp for API and rationale.

#include "face_mortar_inverse_map_3d.hpp"

#include "mfem.hpp"

namespace mortar_pbc
{

std::array<double, 2> InverseMapQuad2DAxisAligned(
    const QuadFaceElement& elem, int a_idx, int b_idx,
    double a, double b)
{
    // Reference convention (matches NQuad4 / MQuad4DualModified):
    //   vertex 0 → (xi, eta) = (-1, -1)
    //   vertex 1 → (xi, eta) = (+1, -1)
    //   vertex 2 → (xi, eta) = (+1, +1)
    //   vertex 3 → (xi, eta) = (-1, +1)
    //
    // For an axis-aligned quad in the (a, b) plane:
    //   v0 → v1 vector spans +xi direction at fixed eta = -1
    //   v0 → v3 vector spans +eta direction at fixed xi = -1
    //
    // The closed-form inverse for a parallelogram-shaped quad (which
    // axis-aligned always is) uses the dual basis of these edge
    // vectors. For axis-aligned quads the edge vectors are orthogonal
    // in (a, b), so the dual basis simplifies to division by the
    // squared edge length.
    const double a0 = elem.coords(0, a_idx);
    const double b0 = elem.coords(0, b_idx);

    const double da_xi  = elem.coords(1, a_idx) - a0;
    const double db_xi  = elem.coords(1, b_idx) - b0;
    const double da_eta = elem.coords(3, a_idx) - a0;
    const double db_eta = elem.coords(3, b_idx) - b0;

    const double len2_xi  = da_xi  * da_xi  + db_xi  * db_xi;
    const double len2_eta = da_eta * da_eta + db_eta * db_eta;

    MFEM_ASSERT(len2_xi  > 0.0,
                "InverseMapQuad2DAxisAligned: degenerate xi edge "
                "(vertices 0 and 1 coincide in projection).");
    MFEM_ASSERT(len2_eta > 0.0,
                "InverseMapQuad2DAxisAligned: degenerate eta edge "
                "(vertices 0 and 3 coincide in projection).");

    // Normalized parametric coordinates t_xi, t_eta in [0, 1] along the
    // two edge vectors. For axis-aligned quads, exactly one of (da, db)
    // is non-zero per direction; the dot product with the query
    // displacement yields t scaled by edge length squared, which is
    // recovered by dividing by len2.
    const double da = a - a0;
    const double db = b - b0;
    const double t_xi  = (da * da_xi  + db * db_xi)  / len2_xi;
    const double t_eta = (da * da_eta + db * db_eta) / len2_eta;

    // Map [0, 1] → [-1, +1].
    return {2.0 * t_xi  - 1.0,
            2.0 * t_eta - 1.0};
}

std::array<double, 3> InverseMapTri2D(
    const TriFaceElement& elem, int a_idx, int b_idx,
    double a, double b)
{
    // Reference convention (matches NTri3 / MTri3DualModified):
    //   vertex 0 → barycentric (1, 0, 0)
    //   vertex 1 → barycentric (0, 1, 0)
    //   vertex 2 → barycentric (0, 0, 1)
    //
    // Barycentric (lam_0, lam_1, lam_2) satisfy:
    //   a = lam_0 * a0 + lam_1 * a1 + lam_2 * a2
    //   b = lam_0 * b0 + lam_1 * b1 + lam_2 * b2
    //   lam_0 + lam_1 + lam_2 = 1
    //
    // Eliminate lam_0 = 1 - lam_1 - lam_2, then solve the 2×2:
    //   lam_1 * (a1 - a0) + lam_2 * (a2 - a0) = a - a0
    //   lam_1 * (b1 - b0) + lam_2 * (b2 - b0) = b - b0
    //
    // Cramer's rule with det = (a1-a0)(b2-b0) - (a2-a0)(b1-b0)
    // = 2 * signed_2D_area_of_triangle.
    const double a0 = elem.coords(0, a_idx);
    const double b0 = elem.coords(0, b_idx);
    const double a1 = elem.coords(1, a_idx);
    const double b1 = elem.coords(1, b_idx);
    const double a2 = elem.coords(2, a_idx);
    const double b2 = elem.coords(2, b_idx);

    const double da1 = a1 - a0;
    const double db1 = b1 - b0;
    const double da2 = a2 - a0;
    const double db2 = b2 - b0;

    const double det = da1 * db2 - da2 * db1;
    MFEM_ASSERT(std::abs(det) > 0.0,
                "InverseMapTri2D: triangle is degenerate in the (a, b) "
                "projection (zero 2D signed area).");

    const double da = a - a0;
    const double db = b - b0;
    // Cramer's rule:
    const double lam_1 = (da * db2 - da2 * db) / det;
    const double lam_2 = (da1 * db - da * db1) / det;
    const double lam_0 = 1.0 - lam_1 - lam_2;
    return {lam_0, lam_1, lam_2};
}

}  // namespace mortar_pbc
