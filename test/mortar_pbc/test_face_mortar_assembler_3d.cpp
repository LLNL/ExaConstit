// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `tests/test_mortar_3d_unit.py`
// (subset: the active element types tri-3 and quad-4 only; higher-order
// tests are negative results and not ported).
//
// Verifies:
//   1. Quadrature rule weights & positivity (3x3 Gauss, tri-3pt).
//   2. Bi-orthogonality of MTri3Dual and MQuad4Dual on their reference
//      elements.
//   3. Partition of unity for dual bases.
//   4. Wohlmuth modifications:
//      (a) tri-3 with one vertex dropped (eq. 5.5).
//      (b) tri-3 with two vertices dropped (eq. 5.6).
//      (c) quad-4 edge-adjacent and corner-adjacent.
//   5. Conforming-pair recovery: A_m = diag(D) on identical nonmortar/mortar
//      meshes, for both quad-4 and tri-3.
//   6. MatchConformingFacePairs gives identity perm on aligned meshes.

#include "face_mortar_assembler_3d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace mortar_pbc;

namespace
{
    int g_failures = 0;
    int g_total    = 0;

    void Pass(const std::string& msg)
    {
        ++g_total;
        std::cout << "  PASS  " << msg << "\n";
    }
    void Fail(const std::string& msg)
    {
        ++g_total;
        ++g_failures;
        std::cout << "  FAIL  " << msg << "\n";
    }
}  // namespace

// ---------------------------------------------------------------------------
// Quadrature rule sanity
// ---------------------------------------------------------------------------
void TestQuadratureWeightsSum()
{
    const auto quad = GaussQuad3x3();
    double sum = 0.0;
    for (double w : quad.wts) { sum += w; }
    // |E| = 4 for [-1, +1]^2.
    if (std::abs(sum - 4.0) < 1e-13) {
        Pass("GaussQuad3x3: weights sum to |E| = 4");
    } else {
        Fail("GaussQuad3x3: weights sum incorrectly");
        std::cout << "    sum = " << sum << ", expected 4.0\n";
    }

    const auto tri = GaussTri3Pt();
    double tri_sum = 0.0;
    for (double w : tri.wts) { tri_sum += w; }
    // |T| = 1/2 for the reference simplex.
    if (std::abs(tri_sum - 0.5) < 1e-13) {
        Pass("GaussTri3Pt: weights sum to |T| = 1/2");
    } else {
        Fail("GaussTri3Pt: weights sum incorrectly");
        std::cout << "    sum = " << tri_sum << ", expected 0.5\n";
    }
}

// ---------------------------------------------------------------------------
// Bi-orthogonality of MTri3Dual on the reference simplex
// ---------------------------------------------------------------------------
//   ∫_T M_i N_j dA = δ_ij * (|T|/3) = δ_ij / 6
// ---------------------------------------------------------------------------
void TestBiorthogonalityTri3()
{
    const auto rule = GaussTri3Pt();
    double M_NN[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (int q = 0; q < 3; ++q) {
        const auto pt = rule.pts[q];
        const double w = rule.wts[q];
        const auto M = MTri3Dual(pt);
        const auto N = NTri3(pt);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                M_NN[i][j] += w * M[i] * N[j];
            }
        }
    }
    const double expected_diag = 1.0 / 6.0;
    double err = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double exp = (i == j) ? expected_diag : 0.0;
            err = std::max(err, std::abs(M_NN[i][j] - exp));
        }
    }
    if (err < 1e-13) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                          "tri-3 dual bi-orthogonality (delta_ij * |T|/3, "
                          "max err %.2e)", err);
        Pass(msg);
    } else {
        Fail("tri-3 dual bi-orthogonality");
        std::cout << "    err = " << err << "\n";
    }
}

// ---------------------------------------------------------------------------
// Bi-orthogonality of MQuad4Dual on the reference square
// ---------------------------------------------------------------------------
//   ∫_E M_i N_j dA = δ_ij * (|E|/4) = δ_ij
// ---------------------------------------------------------------------------
void TestBiorthogonalityQuad4()
{
    const auto rule = GaussQuad3x3();
    double M_NN[4][4] = {};
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const double w = rule.wts[q];
        const auto M = MQuad4Dual(pt[0], pt[1]);
        const auto N = NQuad4(pt[0], pt[1]);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                M_NN[i][j] += w * M[i] * N[j];
            }
        }
    }
    double err = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double exp = (i == j) ? 1.0 : 0.0;
            err = std::max(err, std::abs(M_NN[i][j] - exp));
        }
    }
    if (err < 1e-12) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                          "quad-4 dual bi-orthogonality (delta_ij, max err %.2e)",
                          err);
        Pass(msg);
    } else {
        Fail("quad-4 dual bi-orthogonality");
        std::cout << "    err = " << err << "\n";
    }
}

// ---------------------------------------------------------------------------
// Partition of unity for both N and M bases
// ---------------------------------------------------------------------------
void TestPartitionOfUnityDualBases()
{
    // tri-3: M_1 + M_2 + M_3 = (4 lam_1 - 1) + (4 lam_2 - 1) + (4 lam_3 - 1)
    //                       = 4*(lam_1 + lam_2 + lam_3) - 3 = 4 - 3 = 1.
    const auto tri_rule = GaussTri3Pt();
    double max_dev_tri_M = 0.0, max_dev_tri_N = 0.0;
    for (int q = 0; q < 3; ++q) {
        const auto pt = tri_rule.pts[q];
        const auto M = MTri3Dual(pt);
        const auto N = NTri3(pt);
        max_dev_tri_M = std::max(max_dev_tri_M,
                                          std::abs(M[0] + M[1] + M[2] - 1.0));
        max_dev_tri_N = std::max(max_dev_tri_N,
                                          std::abs(N[0] + N[1] + N[2] - 1.0));
    }
    if (max_dev_tri_M < 1e-13 && max_dev_tri_N < 1e-13) {
        Pass("tri-3 N + M partition of unity");
    } else {
        Fail("tri-3 partition of unity");
        std::cout << "    M dev = " << max_dev_tri_M
                     << ", N dev = " << max_dev_tri_N << "\n";
    }

    // quad-4 (similar)
    const auto quad_rule = GaussQuad3x3();
    double max_dev_quad_M = 0.0, max_dev_quad_N = 0.0;
    for (int q = 0; q < 9; ++q) {
        const auto pt = quad_rule.pts[q];
        const auto M = MQuad4Dual(pt[0], pt[1]);
        const auto N = NQuad4(pt[0], pt[1]);
        const double M_sum = M[0] + M[1] + M[2] + M[3];
        const double N_sum = N[0] + N[1] + N[2] + N[3];
        max_dev_quad_M = std::max(max_dev_quad_M, std::abs(M_sum - 1.0));
        max_dev_quad_N = std::max(max_dev_quad_N, std::abs(N_sum - 1.0));
    }
    if (max_dev_quad_M < 1e-13 && max_dev_quad_N < 1e-13) {
        Pass("quad-4 N + M partition of unity");
    } else {
        Fail("quad-4 partition of unity");
        std::cout << "    M dev = " << max_dev_quad_M
                     << ", N dev = " << max_dev_quad_N << "\n";
    }
}

// ---------------------------------------------------------------------------
// Wohlmuth tri-3: one vertex dropped (eq. 5.5)
// ---------------------------------------------------------------------------
//   For dropped vertex i and kept vertices j, k:
//      M_i = 0
//      M_j = 1/2 + 2 lam_j - 2 lam_k
//      M_k = 1/2 - 2 lam_j + 2 lam_k
//   Test: at the centroid (1/3, 1/3, 1/3), M_j = M_k = 1/2.
//         sum M = 1 (partition of unity restricted to kept).
// ---------------------------------------------------------------------------
void TestWohlmuthTri3OneDropped()
{
    const std::array<double, 3> lam = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    for (int dropped = 0; dropped < 3; ++dropped) {
        std::array<bool, 3> drops = {false, false, false};
        drops[dropped] = true;
        const auto M = MTri3DualModified(lam, drops);
        const int j = (dropped + 1) % 3;
        const int k = (dropped + 2) % 3;
        const bool drop_zero = std::abs(M[dropped]) < 1e-14;
        const bool kept_half_j = std::abs(M[j] - 0.5) < 1e-14;
        const bool kept_half_k = std::abs(M[k] - 0.5) < 1e-14;
        const bool sum_one = std::abs(M[0] + M[1] + M[2] - 1.0) < 1e-14;
        if (!(drop_zero && kept_half_j && kept_half_k && sum_one)) {
            Fail("tri-3 Wohlmuth 1-drop (vertex " + std::to_string(dropped)
                  + ") at centroid");
            std::cout << "    M = (" << M[0] << ", " << M[1] << ", " << M[2]
                         << "), sum = " << (M[0]+M[1]+M[2]) << "\n";
            return;
        }
    }
    Pass("tri-3 Wohlmuth 1-drop: M_dropped=0, M_kept=1/2 at centroid, "
          "POU preserved (eq. 5.5)");
}

// ---------------------------------------------------------------------------
// Wohlmuth tri-3: two vertices dropped (eq. 5.6)
// ---------------------------------------------------------------------------
//   The single kept vertex's M is identically 1.
// ---------------------------------------------------------------------------
void TestWohlmuthTri3TwoDropped()
{
    const std::array<std::array<double, 3>, 4> sample_lams = {{
        {1.0/3.0, 1.0/3.0, 1.0/3.0},  // centroid
        {0.6, 0.2, 0.2},
        {0.1, 0.7, 0.2},
        {0.1, 0.1, 0.8},
    }};
    for (const auto& lam : sample_lams) {
        for (int kept = 0; kept < 3; ++kept) {
            std::array<bool, 3> drops = {true, true, true};
            drops[kept] = false;
            const auto M = MTri3DualModified(lam, drops);
            double err = 0.0;
            for (int i = 0; i < 3; ++i) {
                const double exp = (i == kept) ? 1.0 : 0.0;
                err = std::max(err, std::abs(M[i] - exp));
            }
            if (err > 1e-14) {
                Fail("tri-3 Wohlmuth 2-drop (kept=" + std::to_string(kept) + ")");
                std::cout << "    M = (" << M[0] << "," << M[1] << "," << M[2]
                             << "), err = " << err << "\n";
                return;
            }
        }
    }
    Pass("tri-3 Wohlmuth 2-drop: kept vertex's M = 1, others = 0 (eq. 5.6)");
}

// ---------------------------------------------------------------------------
// Wohlmuth quad-4: edge-adjacent (one xi-side dropped, eta unmodified)
// ---------------------------------------------------------------------------
//   side_xi = "left" -> M_0 = M_3 = 0 (the xi=-1 nodes)
//   side_xi = "right" -> M_1 = M_2 = 0 (the xi=+1 nodes)
//   Partition of unity is preserved on the kept rows.
// ---------------------------------------------------------------------------
void TestWohlmuthQuad4EdgeAdjacent()
{
    const auto rule = GaussQuad3x3();

    // "left" — drops nodes 0 and 3.
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const auto M = MQuad4DualModified(pt[0], pt[1], "left", "none");
        if (std::abs(M[0]) > 1e-14 || std::abs(M[3]) > 1e-14) {
            Fail("quad-4 Wohlmuth edge-xi-low: dropped nodes not zero");
            std::cout << "    M = (" << M[0] << "," << M[1]
                         << "," << M[2] << "," << M[3] << ")\n";
            return;
        }
    }
    // "right" — drops nodes 1 and 2.
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const auto M = MQuad4DualModified(pt[0], pt[1], "right", "none");
        if (std::abs(M[1]) > 1e-14 || std::abs(M[2]) > 1e-14) {
            Fail("quad-4 Wohlmuth edge-xi-high: dropped nodes not zero");
            return;
        }
    }
    // "bottom" — drops nodes 0 and 1.
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const auto M = MQuad4DualModified(pt[0], pt[1], "none", "bottom");
        if (std::abs(M[0]) > 1e-14 || std::abs(M[1]) > 1e-14) {
            Fail("quad-4 Wohlmuth edge-eta-low: dropped nodes not zero");
            return;
        }
    }
    // "top" — drops nodes 2 and 3.
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const auto M = MQuad4DualModified(pt[0], pt[1], "none", "top");
        if (std::abs(M[2]) > 1e-14 || std::abs(M[3]) > 1e-14) {
            Fail("quad-4 Wohlmuth edge-eta-high: dropped nodes not zero");
            return;
        }
    }
    Pass("quad-4 Wohlmuth edge-adjacent: dropped nodes' M = 0 along all "
          "four edges");
}

// ---------------------------------------------------------------------------
// Wohlmuth quad-4: corner-adjacent (two sides dropped)
// ---------------------------------------------------------------------------
//   "corner-LL" = side_xi="left" + side_eta="bottom" -> drops {0, 1, 3}
//   keeping only node 2 (the corner_diagonally_opposite).
// ---------------------------------------------------------------------------
void TestWohlmuthQuad4CornerAdjacent()
{
    const auto rule = GaussQuad3x3();
    // corner-LL: xi=left + eta=bottom drops 0 (xi-low and eta-low both),
    //            1 (eta-low only), 3 (xi-low only). Keeps 2.
    //   But the tensor product of "left" (drops 0, 3) and "bottom"
    //   (drops 0, 1) means M = M_xi_modified * M_eta_modified. With
    //   modified line-2 producing constants:
    //     side_xi = "left"   -> Mxi = (0, 1)
    //     side_eta = "bottom" -> Meta = (0, 1)  (mapped to "left" semantics)
    //   So M = {0*0, 1*0, 1*1, 0*1} = {0, 0, 1, 0}.
    //   Node 2 (which is at xi=+1, eta=+1 — diagonally opposite the
    //   dropped corner LL at xi=-1, eta=-1) gets the full unit value.
    for (int q = 0; q < 9; ++q) {
        const auto pt = rule.pts[q];
        const auto M = MQuad4DualModified(pt[0], pt[1], "left", "bottom");
        const bool ok = std::abs(M[0]) < 1e-14
                              && std::abs(M[1]) < 1e-14
                              && std::abs(M[2] - 1.0) < 1e-14
                              && std::abs(M[3]) < 1e-14;
        if (!ok) {
            Fail("quad-4 Wohlmuth corner-LL: M != (0, 0, 1, 0)");
            std::cout << "    M = (" << M[0] << "," << M[1]
                         << "," << M[2] << "," << M[3] << ")\n";
            return;
        }
    }
    Pass("quad-4 Wohlmuth corner-LL: only opposite corner kept (M = (0,0,1,0))");
}

// ---------------------------------------------------------------------------
// Helper: build a single quad-4 face element on the y=plane_value plane,
// with given in-plane corner coords (x0, x1, z0, z1) and given gtdofs.
// ---------------------------------------------------------------------------
QuadFaceElement MakeQuad(double x0, double x1, double z0, double z1,
                                  double y, int g0, int g1, int g2, int g3,
                                  const std::string& boundary_tag = "none")
{
    QuadFaceElement e;
    e.coords.SetSize(4, 3);
    // Local node order: 0=(x0,z0), 1=(x1,z0), 2=(x1,z1), 3=(x0,z1)
    e.coords(0, 0) = x0; e.coords(0, 1) = y; e.coords(0, 2) = z0;
    e.coords(1, 0) = x1; e.coords(1, 1) = y; e.coords(1, 2) = z0;
    e.coords(2, 0) = x1; e.coords(2, 1) = y; e.coords(2, 2) = z1;
    e.coords(3, 0) = x0; e.coords(3, 1) = y; e.coords(3, 2) = z1;
    e.gtdofs = {g0, g1, g2, g3};
    e.parametric_axes = {"x", "z"};
    e.perpendicular_axis = "y";
    e.boundary_tag = boundary_tag;
    return e;
}

// ---------------------------------------------------------------------------
// Conforming-pair recovery for quad-4 face mortar
// ---------------------------------------------------------------------------
//   On a 1x1 single-quad face (nonmortar at y=0, mortar at y=1) with NO
//   sentinels (all gtdofs >= 0), A_m should equal diag(D) — the lumped
//   mass matrix. This is the 3D analog of test 4 in the 2D suite.
// ---------------------------------------------------------------------------
void TestConformingPairRecoversLumpingQuad4()
{
    QuadFaceMortarAssembler asm_q;

    // Nonmortar at y=0, mortar at y=1; identical 2x2 grid of unit-square quads.
    //   nodes laid out as
    //     (0,0)=0  (1,0)=1  (2,0)=2
    //     (0,1)=3  (1,1)=4  (2,1)=5
    //     (0,2)=6  (1,2)=7  (2,2)=8
    //   in (x, z) — 4 quads total.
    auto build_face = [](double y_const, int gtdof_offset)
         -> std::vector<QuadFaceElement> {
        std::vector<QuadFaceElement> elems;
        const double pts[3] = {0.0, 1.0, 2.0};
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                const int g00 = (j * 3 + i)         + gtdof_offset;
                const int g10 = (j * 3 + i + 1)     + gtdof_offset;
                const int g11 = ((j + 1) * 3 + i + 1) + gtdof_offset;
                const int g01 = ((j + 1) * 3 + i)   + gtdof_offset;
                elems.push_back(MakeQuad(pts[i], pts[i+1], pts[j], pts[j+1],
                                                    y_const, g00, g10, g11, g01));
            }
        }
        return elems;
    };
    auto nonmortar  = build_face(0.0, 0);
    auto mortar = build_face(1.0, 100);

    // Identity matching: i_th nonmortar maps to i_th mortar with identity perm.
    //   But the in-plane coords are (x, z) — the matching helper uses
    //   parametric centroid in the in-plane axes which here matches.
    const auto matches = MatchConformingFacePairs(nonmortar, mortar, "y", 1.0);
    if (static_cast<int>(matches.size()) != 4) {
        Fail("MatchConformingFacePairs(quad): expected 4 matches");
        std::cout << "    got " << matches.size() << "\n";
        return;
    }
    bool all_identity = true;
    for (const auto& m : matches) {
        for (int i = 0; i < 4; ++i) {
            if (m.mortar_node_perm[i] != i) { all_identity = false; }
        }
    }
    if (!all_identity) {
        Fail("MatchConformingFacePairs(quad): expected identity perms on "
              "axis-aligned mesh");
        return;
    }

    const auto block = asm_q.AssemblePairConforming(nonmortar, mortar, matches);

    // Expected: A_m == diag(D); all gtdofs are non-sentinel so n_rows=9, n_cols=9.
    const int N = block.D.Size();
    if (N != 9) {
        Fail("conforming quad-4 pair: expected 9 kept rows, got "
              + std::to_string(N));
        return;
    }
    double diff = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double exp = (i == j) ? block.D(i) : 0.0;
            diff += (block.A_m(i, j) - exp) * (block.A_m(i, j) - exp);
        }
    }
    diff = std::sqrt(diff);
    if (diff < 1e-12) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                          "conforming quad-4 pair recovers lumped mass "
                          "(||A^m - diag(D)||_F = %.2e)", diff);
        Pass(msg);
    } else {
        Fail("conforming quad-4 pair recovers lumped mass");
        std::cout << "    ||A^m - diag(D)||_F = " << diff << "\n";
        // Diagnostics
        double sum_D = 0.0;
        for (int i = 0; i < N; ++i) { sum_D += block.D(i); }
        std::cout << "    sum D = " << sum_D << " (expected total area = "
                     << 4.0 << ")\n";
    }
}

// ---------------------------------------------------------------------------
// Helper: build a single tri-3 face element
// ---------------------------------------------------------------------------
TriFaceElement MakeTri(double x0, double z0, double x1, double z1,
                                double x2, double z2, double y,
                                int g0, int g1, int g2,
                                const std::string& boundary_tag = "none")
{
    TriFaceElement e;
    e.coords.SetSize(3, 3);
    e.coords(0, 0) = x0; e.coords(0, 1) = y; e.coords(0, 2) = z0;
    e.coords(1, 0) = x1; e.coords(1, 1) = y; e.coords(1, 2) = z1;
    e.coords(2, 0) = x2; e.coords(2, 1) = y; e.coords(2, 2) = z2;
    e.gtdofs = {g0, g1, g2};
    e.parametric_axes = {"x", "z"};
    e.perpendicular_axis = "y";
    e.boundary_tag = boundary_tag;
    return e;
}

// ---------------------------------------------------------------------------
// Conforming-pair recovery for tri-3 face mortar
// ---------------------------------------------------------------------------
void TestConformingPairRecoversLumpingTri3()
{
    TriFaceMortarAssembler asm_t;

    // Nonmortar at y=0, mortar at y=1; both: a single 1x1 unit square split
    // into two triangles along the diagonal.
    //   nodes: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
    //   triangles: (0, 1, 2) and (0, 2, 3)  — CCW viewed from +y
    auto build_face = [](double y_const, int gtdof_offset)
         -> std::vector<TriFaceElement> {
        std::vector<TriFaceElement> elems;
        // Triangle 1: nodes 0, 1, 2
        elems.push_back(MakeTri(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, y_const,
                                          gtdof_offset + 0, gtdof_offset + 1,
                                          gtdof_offset + 2));
        // Triangle 2: nodes 0, 2, 3
        elems.push_back(MakeTri(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, y_const,
                                          gtdof_offset + 0, gtdof_offset + 2,
                                          gtdof_offset + 3));
        return elems;
    };
    auto nonmortar  = build_face(0.0, 0);
    auto mortar = build_face(1.0, 100);

    const auto matches = MatchConformingFacePairs(nonmortar, mortar, "y", 1.0);
    if (static_cast<int>(matches.size()) != 2) {
        Fail("MatchConformingFacePairs(tri): expected 2 matches, got "
              + std::to_string(matches.size()));
        return;
    }
    bool all_identity = true;
    for (const auto& m : matches) {
        for (int i = 0; i < 3; ++i) {
            if (m.mortar_node_perm[i] != i) { all_identity = false; }
        }
    }
    if (!all_identity) {
        Fail("MatchConformingFacePairs(tri): expected identity perms");
        return;
    }

    const auto block = asm_t.AssemblePairConforming(nonmortar, mortar, matches);
    const int N = block.D.Size();
    // 4 unique kept gtdofs (0, 1, 2, 3 from nonmortar; 100, 101, 102, 103 from
    // mortar are separate indexing).
    if (N != 4) {
        Fail("conforming tri-3 pair: expected 4 kept nonmortar rows, got "
              + std::to_string(N));
        return;
    }
    double diff = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double exp = (i == j) ? block.D(i) : 0.0;
            diff += (block.A_m(i, j) - exp) * (block.A_m(i, j) - exp);
        }
    }
    diff = std::sqrt(diff);
    if (diff < 1e-12) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                          "conforming tri-3 pair recovers lumped mass "
                          "(||A^m - diag(D)||_F = %.2e)", diff);
        Pass(msg);
    } else {
        Fail("conforming tri-3 pair recovers lumped mass");
        std::cout << "    ||A^m - diag(D)||_F = " << diff << "\n";
        double sum_D = 0.0;
        for (int i = 0; i < N; ++i) { sum_D += block.D(i); }
        std::cout << "    sum D = " << sum_D << " (expected = 1.0)\n";
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
    std::cout << "   test_face_mortar_assembler_3d (Phase 4.1.A C++ port)\n";
    std::cout << "=========================================================\n";

    TestQuadratureWeightsSum();
    TestBiorthogonalityTri3();
    TestBiorthogonalityQuad4();
    TestPartitionOfUnityDualBases();
    TestWohlmuthTri3OneDropped();
    TestWohlmuthTri3TwoDropped();
    TestWohlmuthQuad4EdgeAdjacent();
    TestWohlmuthQuad4CornerAdjacent();
    TestConformingPairRecoversLumpingQuad4();
    TestConformingPairRecoversLumpingTri3();

    std::cout << "=========================================================\n";
    if (g_failures == 0) {
        std::cout << "  All " << g_total << " tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cout << "  " << g_failures << " of " << g_total << " tests FAILED.\n";
    return EXIT_FAILURE;
}
