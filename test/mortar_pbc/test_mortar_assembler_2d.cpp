// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `tests/test_mortar_2d_unit.py`
//
// Unit tests for the line-2 mortar machinery, mirroring the Python
// suite. Verifies:
//   1. Dual basis bi-orthogonality on the reference element.
//   2. Standard line-2 partition-of-unity.
//   3. Wohlmuth corner-modified dual basis behaviour:
//      (a) partition of unity preserved
//      (b) corner-side function is identically zero
//      (c) neighbor-side function integrates as constant 1
//   4. Conforming-pair recovers the lumped mass: A^m = diag(D^nm).
//   5. Non-conforming-pair linear-field reproduction (without corners).
//
// All tests are stand-alone with no MPI — `MortarAssembler2D` is
// stateless and stateless-pure for these inputs. The test harness uses
// MFEM's `MFEM_VERIFY` for assertions and prints PASS / FAIL lines.
//
// Run via:
//   cd build && ctest -V -R test_mortar_assembler_2d
//   ./tests/mortar_pbc/test_mortar_assembler_2d

#include "mortar_assembler_2d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using mortar_pbc::EdgeInfo3D;
using mortar_pbc::MortarAssembler2D;
using mortar_pbc::MortarBlock2D;
using mortar_pbc::MLine2Dual;
using mortar_pbc::MLine2DualModified;
using mortar_pbc::NLine2;

// 3-point Gauss-Legendre quadrature on [-1, 1] — match the assembler's
// internal rule. We re-derive locally so the test is independent of the
// implementation's anonymous-namespace constants (i.e. if those change
// shape, this test should still verify the math holds regardless).
namespace {
const double kSqrt3Over5 = std::sqrt(0.6);
const double kPts[3] = { -kSqrt3Over5, 0.0, kSqrt3Over5 };
const double kWts[3] = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };

int g_failures = 0;

void Pass(const std::string& msg) {
    std::cout << "  PASS  " << msg << "\n";
}

void Fail(const std::string& msg) {
    std::cout << "  FAIL  " << msg << "\n";
    ++g_failures;
}

double InfNorm(const mfem::Vector& v) {
    double m = 0.0;
    for (int i = 0; i < v.Size(); ++i) {
        m = std::max(m, std::abs(v(i)));
    }
    return m;
}
}  // namespace

// ---------------------------------------------------------------------------
// Test 1: dual basis bi-orthogonality
// ---------------------------------------------------------------------------
void TestDualBasisBiorthogonality()
{
    // ∫_{-1}^{1} M_i(ξ) N_j(ξ) dξ should equal δ_{ij}.
    double M_NN[2][2] = {{0, 0}, {0, 0}};
    for (int q = 0; q < 3; ++q) {
        const double x = kPts[q];
        const double w = kWts[q];
        const auto M = MLine2Dual(x);
        const auto N = NLine2(x);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                M_NN[i][j] += w * M[i] * N[j];
            }
        }
    }
    double err = 0.0;
    const double expected[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            err = std::max(err, std::abs(M_NN[i][j] - expected[i][j]));
        }
    }
    if (err < 1e-12) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                          "dual basis bi-orthogonality (max err %.2e)", err);
        Pass(msg);
    } else {
        Fail("dual basis bi-orthogonality");
        std::cout << "    M*N = [[" << M_NN[0][0] << "," << M_NN[0][1]
                     << "],[" << M_NN[1][0] << "," << M_NN[1][1] << "]]\n";
        std::cout << "    err = " << err << "\n";
    }
}

// ---------------------------------------------------------------------------
// Test 2: standard line-2 partition of unity
// ---------------------------------------------------------------------------
void TestPartitionOfUnity()
{
    // ∫_{-1}^{1} N_i(ξ) dξ should equal 1.
    double integrals[2] = {0, 0};
    for (int q = 0; q < 3; ++q) {
        const auto N = NLine2(kPts[q]);
        const double w = kWts[q];
        for (int i = 0; i < 2; ++i) { integrals[i] += w * N[i]; }
    }
    const double err = std::max(std::abs(integrals[0] - 1.0),
                                         std::abs(integrals[1] - 1.0));
    if (err < 1e-12) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                          "N partition of unity (max err %.2e)", err);
        Pass(msg);
    } else {
        Fail("N partition of unity");
        std::cout << "    integrals = [" << integrals[0] << "," << integrals[1]
                     << "]\n";
    }
}

// ---------------------------------------------------------------------------
// Test 3: Wohlmuth crosspoint modification (Lopes 2021 Eq. C.2)
// ---------------------------------------------------------------------------
void TestWohlmuthCrosspointModification()
{
    // (a) Partition of unity for both modifications
    for (const std::string& side : {std::string("left"), std::string("right")}) {
        double max_dev = 0.0;
        for (int q = 0; q < 3; ++q) {
            const auto M = MLine2DualModified(kPts[q], side);
            max_dev = std::max(max_dev, std::abs(M[0] + M[1] - 1.0));
        }
        if (max_dev > 1e-15) {
            Fail("Wohlmuth (a): partition of unity for side='" + side + "'");
            return;
        }
    }

    // (b) Corner-side function is identically zero
    for (int q = 0; q < 3; ++q) {
        const auto M_L = MLine2DualModified(kPts[q], "left");
        if (M_L[0] != 0.0) {
            Fail("Wohlmuth (b): side='left', M[0] should be 0");
            return;
        }
        const auto M_R = MLine2DualModified(kPts[q], "right");
        if (M_R[1] != 0.0) {
            Fail("Wohlmuth (b): side='right', M[1] should be 0");
            return;
        }
    }

    // (c) Neighbor-side function integrates as constant 1
    //   side='left' -> M[1] = 1 on [-1, 1]
    //   ∫ M[1] N[0] dξ = 1 (since ∫ N[0] dξ = 1)
    //   ∫ M[1] N[1] dξ = 1 (since ∫ N[1] dξ = 1)
    double int_M2_N1 = 0.0, int_M2_N2 = 0.0;
    double int_M1_N1 = 0.0, int_M1_N2 = 0.0;
    for (int q = 0; q < 3; ++q) {
        const double x = kPts[q];
        const double w = kWts[q];
        const auto N = NLine2(x);
        const auto M_left  = MLine2DualModified(x, "left");
        const auto M_right = MLine2DualModified(x, "right");
        int_M2_N1 += w * M_left[1]  * N[0];
        int_M2_N2 += w * M_left[1]  * N[1];
        int_M1_N1 += w * M_right[0] * N[0];
        int_M1_N2 += w * M_right[0] * N[1];
    }
    const double err = std::max({std::abs(int_M2_N1 - 1.0),
                                          std::abs(int_M2_N2 - 1.0),
                                          std::abs(int_M1_N1 - 1.0),
                                          std::abs(int_M1_N2 - 1.0)});
    if (err < 1e-12) {
        char msg[200];
        std::snprintf(msg, sizeof(msg),
                          "Wohlmuth crosspoint mod (Lopes 2021 Eq. C.2): "
                          "POU preserved, corner-func=0, neighbor-func "
                          "integrals=1 (max err %.2e)", err);
        Pass(msg);
    } else {
        Fail("Wohlmuth (c): neighbor-func integrals not 1");
        std::cout << "    int_M2_N1=" << int_M2_N1 << ", int_M2_N2=" << int_M2_N2
                     << ", int_M1_N1=" << int_M1_N1 << ", int_M1_N2=" << int_M1_N2
                     << "\n";
    }
}

// ---------------------------------------------------------------------------
// Helper: build a synthetic EdgeInfo3D with given node x-coords on a y=const
// edge, with corner sentinels at both ends.
// ---------------------------------------------------------------------------
EdgeInfo3D MakeSyntheticEdge(const std::string& label,
                                        const std::vector<double>& interior_xs,
                                        double y_const,
                                        double edge_min, double edge_max)
{
    EdgeInfo3D edge;
    edge.label = label;
    edge.is_mortar = false;
    edge.parametric_axis = "x";
    edge.edge_min = edge_min;
    edge.edge_max = edge_max;
    const int N = static_cast<int>(interior_xs.size());
    edge.coords.SetSize(N, 3);
    edge.coords = 0.0;
    for (int i = 0; i < N; ++i) {
        edge.coords(i, 0) = interior_xs[i];
        edge.coords(i, 1) = y_const;
        edge.coords(i, 2) = 0.0;  // unused
    }
    // Mock TDOFs.
    edge.gtdofs_x.SetSize(N);
    edge.gtdofs_y.SetSize(N);
    edge.gtdofs_z.SetSize(N);
    for (int i = 0; i < N; ++i) {
        edge.gtdofs_x[i] = i;
        edge.gtdofs_y[i] = i + 100;
        edge.gtdofs_z[i] = i + 200;
    }
    // Connectivity with corner sentinels at both ends.
    edge.elements.clear();
    edge.elements.emplace_back(-1, 0);
    for (int k = 0; k < N - 1; ++k) {
        edge.elements.emplace_back(k, k + 1);
    }
    edge.elements.emplace_back(N - 1, -2);
    return edge;
}

// ---------------------------------------------------------------------------
// Helper: build a synthetic EdgeInfo3D WITHOUT corner sentinels — the full
// edge interior is the domain, no Dirichlet boundary touched.
// ---------------------------------------------------------------------------
EdgeInfo3D MakeInteriorOnlyEdge(const std::string& label,
                                            const std::vector<double>& xs,
                                            double y_const,
                                            double edge_min, double edge_max)
{
    EdgeInfo3D edge;
    edge.label = label;
    edge.is_mortar = false;
    edge.parametric_axis = "x";
    edge.edge_min = edge_min;
    edge.edge_max = edge_max;
    const int N = static_cast<int>(xs.size());
    edge.coords.SetSize(N, 3);
    edge.coords = 0.0;
    for (int i = 0; i < N; ++i) {
        edge.coords(i, 0) = xs[i];
        edge.coords(i, 1) = y_const;
    }
    edge.gtdofs_x.SetSize(N);
    edge.gtdofs_y.SetSize(N);
    edge.gtdofs_z.SetSize(N);
    for (int i = 0; i < N; ++i) {
        edge.gtdofs_x[i] = i;
        edge.gtdofs_y[i] = i + 100;
        edge.gtdofs_z[i] = i + 200;
    }
    edge.elements.clear();
    for (int k = 0; k < N - 1; ++k) {
        edge.elements.emplace_back(k, k + 1);
    }
    return edge;
}

// ---------------------------------------------------------------------------
// Test 4: conforming pair recovers lumped mass
// ---------------------------------------------------------------------------
void TestConformingPairRecoversLumping()
{
    const double L = 1.0;
    // 5 nodes total: 2 corners + 3 interior — interior at x=0.25, 0.5, 0.75
    const std::vector<double> interior_xs = {0.25, 0.5, 0.75};
    auto plus_edge  = MakeSyntheticEdge("plus",  interior_xs, 0.0, 0.0, L);
    auto minus_edge = MakeSyntheticEdge("minus", interior_xs, L,   0.0, L);

    MortarAssembler2D assembler;
    const MortarBlock2D block = assembler.AssemblePair(plus_edge, minus_edge);

    // For a CONFORMING pair, A^m should equal diag(D^nm) for interior nodes.
    const int N = block.D_nm.Size();
    double diff_F = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double expected = (i == j) ? block.D_nm(i) : 0.0;
            const double dev = block.A_m(i, j) - expected;
            diff_F += dev * dev;
        }
    }
    diff_F = std::sqrt(diff_F);
    if (diff_F < 1e-12) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                          "conforming pair recovers lumped mass "
                          "(||A^m - diag(D^nm)||_F = %.2e)", diff_F);
        Pass(msg);
    } else {
        Fail("conforming pair recovers lumped mass");
        std::cout << "    D^nm = [";
        for (int i = 0; i < N; ++i) {
            std::cout << block.D_nm(i) << (i + 1 < N ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "    diag(A^m) = [";
        for (int i = 0; i < N; ++i) {
            std::cout << block.A_m(i, i) << (i + 1 < N ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "    ||A^m - diag(D^nm)||_F = " << diff_F << "\n";
    }
}

// ---------------------------------------------------------------------------
// Test 5: non-conforming linear-field reproduction (no corners)
// ---------------------------------------------------------------------------
void TestNonconformingLinearReproduction()
{
    // Use only the interior of [0, L] so no corner segments.
    const double Y0 = 0.1, Y1 = 0.9;
    const std::vector<double> plus_xs  = {0.10, 0.27, 0.41, 0.58, 0.73, 0.90};
    const std::vector<double> minus_xs = {0.10, 0.35, 0.62, 0.90};
    auto plus_edge  = MakeInteriorOnlyEdge("plus",  plus_xs,  0.0, Y0, Y1);
    auto minus_edge = MakeInteriorOnlyEdge("minus", minus_xs, 1.0, Y0, Y1);

    MortarAssembler2D assembler;
    const MortarBlock2D block = assembler.AssemblePair(plus_edge, minus_edge);

    // Sanity: D^nm[k] = (x_{k+1}-x_{k-1})/2 for interior, with appropriate
    // half-element values at endpoints.
    const int Np = static_cast<int>(plus_xs.size());
    mfem::Vector expected_Dnm(Np);
    expected_Dnm(0)      = (plus_xs[1] - plus_xs[0]) / 2.0;          // endpoint
    expected_Dnm(Np - 1) = (plus_xs[Np - 1] - plus_xs[Np - 2]) / 2.0;// endpoint
    for (int k = 1; k < Np - 1; ++k) {
        expected_Dnm(k) = (plus_xs[k + 1] - plus_xs[k - 1]) / 2.0;
    }
    mfem::Vector dD(block.D_nm);
    dD -= expected_Dnm;
    const double diff_D = InfNorm(dD);
    if (diff_D >= 1e-14) {
        Fail("non-conforming D^nm wrong");
        std::cout << "    ||D^nm - expected||_inf = " << diff_D << "\n";
        return;
    }

    // Linear-field reproduction:
    //   D^nm * u^+  -  A^m * u^-  =  0
    // for u(x) = a + b*x sampled at all + and - nodes.
    const double a = 0.3, b = 1.7;
    mfem::Vector u_plus(Np), u_minus(static_cast<int>(minus_xs.size()));
    for (int i = 0; i < Np; ++i) { u_plus(i) = a + b * plus_xs[i]; }
    for (int i = 0; i < static_cast<int>(minus_xs.size()); ++i) {
        u_minus(i) = a + b * minus_xs[i];
    }
    mfem::Vector Du(Np);
    for (int i = 0; i < Np; ++i) { Du(i) = block.D_nm(i) * u_plus(i); }
    mfem::Vector Au(Np);
    block.A_m.Mult(u_minus, Au);
    mfem::Vector residual(Np);
    for (int i = 0; i < Np; ++i) { residual(i) = Du(i) - Au(i); }
    const double res_inf = InfNorm(residual);

    if (res_inf < 1e-12) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                          "non-conforming pair reproduces linear field exactly "
                          "(||D^nm u^+ - A^m u^-||_inf = %.2e)", res_inf);
        Pass(msg);
    } else {
        Fail("non-conforming linear-field reproduction");
        std::cout << "    ||residual||_inf = " << res_inf << "\n";
        std::cout << "    ||D^nm - expected||_inf = " << diff_D << "\n";
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "=========================================================\n";
    std::cout << "   test_mortar_assembler_2d (Phase 4.1.A C++ port)\n";
    std::cout << "=========================================================\n";

    TestDualBasisBiorthogonality();
    TestPartitionOfUnity();
    TestWohlmuthCrosspointModification();
    TestConformingPairRecoversLumping();
    TestNonconformingLinearReproduction();

    std::cout << "=========================================================\n";
    if (g_failures == 0) {
        std::cout << "  All " << 5 << " tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cout << "  " << g_failures << " of " << 5 << " tests FAILED.\n";
    return EXIT_FAILURE;
}
