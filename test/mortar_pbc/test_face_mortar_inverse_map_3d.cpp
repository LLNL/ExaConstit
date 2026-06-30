// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-1 — unit tests for the closed-form inverse
// isoparametric maps used by AssemblePairClipped (Batches 4.4-D-2/3).
//
// Test strategy: round-trip checks. For each element type, build a
// known element, evaluate forward iso-map at canonical reference
// points (vertex coords, face center, sub-points), then run the
// inverse map and check that we recover the original reference
// coords to roundoff. Also exercise the helpers at points NOT on
// vertices to catch the generic case.
//
// No Axom dependency — these tests run regardless of ENABLE_AXOM.

#include "face_mortar_inverse_map_3d.hpp"
#include "face_mortar_assembler_3d.hpp"  // NQuad4, NTri3
#include "types_3d.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>

namespace mortar_pbc
{
namespace
{

bool g_failures = false;

#define REQUIRE_NEAR(actual, expected, tol, msg)                              \
    do {                                                                      \
        const double err = std::abs((actual) - (expected));                   \
        if (err > (tol)) {                                                    \
            std::cerr << "  FAIL: " << msg << "  actual=" << actual           \
                      << "  expected=" << expected << "  err=" << err         \
                      << "  tol=" << tol << "  ("                             \
                      << __FILE__ << ":" << __LINE__ << ")\n";                \
            g_failures = true;                                                \
        }                                                                     \
    } while (0)

// ============================================================================
// Test 1 — InverseMapQuad2DAxisAligned: round-trip at vertices and interior
// ============================================================================
//
// Build an axis-aligned quad on the y = 0 plane:
//   vertex 0 at (x0, 0, z0) → reference (-1, -1)
//   vertex 1 at (x1, 0, z0) → reference (+1, -1)
//   vertex 2 at (x1, 0, z1) → reference (+1, +1)
//   vertex 3 at (x0, 0, z1) → reference (-1, +1)
// With perpendicular_axis = "y", projection axes (a, b) = (z, x) by
// the cyclic convention.
//
// For each test point (xi, eta) in reference space:
//   (a, b) = forward iso-map at (xi, eta)
//          = NQuad4(xi, eta) · {(z_v, x_v)}
//   (xi', eta') = InverseMapQuad2DAxisAligned(elem, a_idx=2, b_idx=0, a, b)
// Assert (xi', eta') ≈ (xi, eta) to 1e-14.
QuadFaceElement MakeTestQuad(double x0, double x1, double z0, double z1)
{
    QuadFaceElement e;
    e.coords.SetSize(4, 3);
    e.coords(0, 0) = x0; e.coords(0, 1) = 0.0; e.coords(0, 2) = z0;
    e.coords(1, 0) = x1; e.coords(1, 1) = 0.0; e.coords(1, 2) = z0;
    e.coords(2, 0) = x1; e.coords(2, 1) = 0.0; e.coords(2, 2) = z1;
    e.coords(3, 0) = x0; e.coords(3, 1) = 0.0; e.coords(3, 2) = z1;
    e.parametric_axes   = {"x", "z"};
    e.perpendicular_axis = "y";
    return e;
}

void test_inverse_map_quad_round_trip()
{
    std::cout << "  test_inverse_map_quad_round_trip\n";
    auto elem = MakeTestQuad(0.25, 0.75, 0.10, 0.40);

    // Projection axes for "y" are (a, b) = (z, x), i.e. a_idx = 2, b_idx = 0.
    const int a_idx = 2;
    const int b_idx = 0;

    // 9 reference points: vertices, mid-edges, and center.
    const double tests[][2] = {
        {-1.0, -1.0},  {1.0, -1.0},   {1.0, 1.0},   {-1.0, 1.0},  // vertices
        {0.0, -1.0},   {1.0, 0.0},    {0.0, 1.0},   {-1.0, 0.0},  // mid-edges
        {0.0, 0.0},                                                  // center
        {0.3, -0.7},   {-0.5, 0.4},                                  // generic
    };

    for (const auto& tp : tests)
    {
        const double xi  = tp[0];
        const double eta = tp[1];
        const auto N = NQuad4(xi, eta);

        // Forward: (a, b) = sum_k N_k * coords[k, {a_idx, b_idx}]
        double a = 0.0, b = 0.0;
        for (int k = 0; k < 4; ++k)
        {
            a += N[k] * elem.coords(k, a_idx);
            b += N[k] * elem.coords(k, b_idx);
        }

        // Inverse:
        const auto ref = InverseMapQuad2DAxisAligned(elem, a_idx, b_idx, a, b);
        REQUIRE_NEAR(ref[0], xi,  1.0e-14, "quad inverse: xi round-trip");
        REQUIRE_NEAR(ref[1], eta, 1.0e-14, "quad inverse: eta round-trip");
    }
}

// ============================================================================
// Test 2 — InverseMapTri2D: round-trip at vertices and interior
// ============================================================================
//
// Build a P1 tri on the y = 0 plane with vertices at known positions.
// Use barycentric coords from canonical sample points and round-trip.
TriFaceElement MakeTestTri(double xa, double za, double xb, double zb,
                           double xc, double zc)
{
    TriFaceElement e;
    e.coords.SetSize(3, 3);
    e.coords(0, 0) = xa; e.coords(0, 1) = 0.0; e.coords(0, 2) = za;
    e.coords(1, 0) = xb; e.coords(1, 1) = 0.0; e.coords(1, 2) = zb;
    e.coords(2, 0) = xc; e.coords(2, 1) = 0.0; e.coords(2, 2) = zc;
    e.parametric_axes   = {"x", "z"};
    e.perpendicular_axis = "y";
    return e;
}

void test_inverse_map_tri_round_trip()
{
    std::cout << "  test_inverse_map_tri_round_trip\n";
    // Right triangle: (0,0), (0.5, 0), (0.5, 0.3). Non-isosceles to
    // catch axis-swap bugs.
    auto elem = MakeTestTri(0.0, 0.0,  0.5, 0.0,  0.5, 0.3);

    const int a_idx = 2;
    const int b_idx = 0;

    // Test barycentric points: vertices, edge midpoints, centroid, generic.
    const double tests[][3] = {
        {1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},  {0.0, 0.0, 1.0},  // vertices
        {0.5, 0.5, 0.0},  {0.0, 0.5, 0.5},  {0.5, 0.0, 0.5},  // mid-edges
        {1.0/3, 1.0/3, 1.0/3},                                  // centroid
        {0.7, 0.2, 0.1},                                         // generic
    };

    for (const auto& tp : tests)
    {
        const double lam0 = tp[0];
        const double lam1 = tp[1];
        const double lam2 = tp[2];

        // Forward: (a, b) = sum_k lam_k * coords[k, {a_idx, b_idx}]
        const double a = lam0 * elem.coords(0, a_idx)
                       + lam1 * elem.coords(1, a_idx)
                       + lam2 * elem.coords(2, a_idx);
        const double b = lam0 * elem.coords(0, b_idx)
                       + lam1 * elem.coords(1, b_idx)
                       + lam2 * elem.coords(2, b_idx);

        const auto lam_inv = InverseMapTri2D(elem, a_idx, b_idx, a, b);
        REQUIRE_NEAR(lam_inv[0], lam0, 1.0e-14, "tri inverse: lam_0 round-trip");
        REQUIRE_NEAR(lam_inv[1], lam1, 1.0e-14, "tri inverse: lam_1 round-trip");
        REQUIRE_NEAR(lam_inv[2], lam2, 1.0e-14, "tri inverse: lam_2 round-trip");
    }
}

// ============================================================================
// Test 3 — DunavantTri6Pt: weights sum to |T| = 1/2; integrates monomials
// up to degree 4 exactly.
// ============================================================================
void test_dunavant_tri_6pt()
{
    std::cout << "  test_dunavant_tri_6pt\n";
    const auto rule = DunavantTri6Pt();

    double w_sum = 0.0;
    for (int q = 0; q < 6; ++q) { w_sum += rule.wts[q]; }
    REQUIRE_NEAR(w_sum, 0.5, 1.0e-14, "DunavantTri6Pt: weights sum to |T| = 1/2");

    // For a barycentric monomial lam_0^p lam_1^q lam_2^r on the
    // reference simplex, the exact integral is
    //   ∫ lam_0^p lam_1^q lam_2^r dA = p! q! r! / (p+q+r+2)!
    //                                      * |T_ref|
    // where |T_ref| = 1/2.
    //
    // We test all monomials with p+q+r ∈ {0, 1, 2, 3, 4} (degree-4 rule
    // should integrate these exactly).
    auto factorial = [](int n) {
        double f = 1.0;
        for (int i = 2; i <= n; ++i) { f *= i; }
        return f;
    };
    auto exact = [&](int p, int q, int r) {
        return factorial(p) * factorial(q) * factorial(r)
             / factorial(p + q + r + 2);  // already includes |T_ref| = 1/2
    };

    for (int total = 0; total <= 4; ++total)
    {
        for (int p = 0; p <= total; ++p)
        {
            for (int q = 0; q <= total - p; ++q)
            {
                const int r = total - p - q;
                double approx = 0.0;
                for (int qi = 0; qi < 6; ++qi)
                {
                    const auto& lam = rule.pts[qi];
                    approx += rule.wts[qi]
                            * std::pow(lam[0], p)
                            * std::pow(lam[1], q)
                            * std::pow(lam[2], r);
                }
                const double exa = exact(p, q, r);
                const std::string lbl = "DunavantTri6Pt: monomial ("
                    + std::to_string(p) + "," + std::to_string(q)
                    + "," + std::to_string(r) + ")";
                REQUIRE_NEAR(approx, exa, 1.0e-13, lbl);
            }
        }
    }
}

}  // anonymous namespace
}  // namespace mortar_pbc

int main()
{
    std::cout << "test_face_mortar_inverse_map_3d (Phase 4.4 / Batch 4.4-D-1)\n";
    mortar_pbc::test_inverse_map_quad_round_trip();
    mortar_pbc::test_inverse_map_tri_round_trip();
    mortar_pbc::test_dunavant_tri_6pt();

    if (mortar_pbc::g_failures)
    {
        std::cerr << "\nOne or more test_face_mortar_inverse_map_3d cases FAILED.\n";
        return 1;
    }
    std::cout << "\nAll test_face_mortar_inverse_map_3d cases passed.\n";
    return 0;
}
