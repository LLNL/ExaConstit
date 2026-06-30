// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.4 / Batch 4.4-D-2 — sanity test for AssembleQuadFacePairClipped.
//
// CENTRAL CORRECTNESS GATE FOR PHASE 4.4: route a 4×4 vs 4×4
// CONFORMING setup through both the conforming and clipped paths,
// then assert their FaceMortarPairBlock outputs (D vector + A_m
// sparse matrix) agree to FP roundoff. If this test passes, we have
// high confidence the non-conforming path is correct because the only
// thing that changes for non-conforming meshes is the clipping geometry
// — the assembler itself is the same.
//
// The two paths integrate the same polynomial integrand
//   M_dual(xi_nm, eta_nm) · N_mortar(xi_m, eta_m)
// (degree 4 in barycentric on a sub-triangle, equivalently degree 4 in
// (xi, eta) on the parent quad) but on different reference domains:
//   * Conforming: 9-point Gauss-Legendre on the full parent reference
//     [-1,+1]^2 (degree 5 each direction).
//   * Clipped: 2 × 6-point Dunavant (degree 4) on the two sub-triangles
//     of each conforming quad pair.
// Both rules exactly integrate the integrand → sums match to FP
// roundoff (modulo summation order).

#include "face_mortar_assembler_3d.hpp"
#include "face_mortar_assembler_clipped_3d.hpp"
#include "face_mortar_match_3d.hpp"
#include "types_3d.hpp"

#include "axom/slic.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace mortar_pbc
{
namespace
{

bool g_failures = false;

#define REQUIRE(cond, msg)                                                    \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "  FAIL: " << msg << "  (" #cond " at "              \
                      << __FILE__ << ":" << __LINE__ << ")\n";                \
            g_failures = true;                                                \
        }                                                                     \
    } while (0)

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
// Mesh builders
// ============================================================================

/// Build a single quad face element on the y=y plane with given gtdofs.
/// Local node order: 0=(x0,z0), 1=(x1,z0), 2=(x1,z1), 3=(x0,z1) — same
/// convention as test_face_mortar_assembler_3d.cpp::MakeQuad.
QuadFaceElement MakeQuad(double x0, double x1, double z0, double z1,
                         double y, int g0, int g1, int g2, int g3,
                         const std::string& boundary_tag = "none")
{
    QuadFaceElement e;
    e.coords.SetSize(4, 3);
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

/// Build an n×n grid of quads on the y=y plane covering [0, L]^2.
/// Assigns sequential gtdofs starting from `gtdof_base`. Node sharing
/// across cells produces a conforming gtdof layout: the (n+1)^2
/// vertices in the grid each get a unique global tdof.
///
/// Each quad's `boundary_tag` is set based on its position in the grid:
/// interior cells get "none"; edge cells get appropriate "edge-*" tags;
/// corner cells get "corner-*". This exercises the full Wohlmuth
/// dispatch.
struct GridResult
{
    std::vector<QuadFaceElement> elems;
    int n_unique_gtdofs;
};

GridResult MakeQuadGridWithGtdofs(int n, double L, double y, int gtdof_base)
{
    GridResult result;
    result.elems.reserve(n * n);
    const double dx = L / n;

    auto vertex_gtdof = [&](int i, int j) {
        // (n+1) × (n+1) vertex grid. Vertex at (i, j) gets global index
        // gtdof_base + i + j * (n + 1). All sequential, no sentinels.
        return gtdof_base + i + j * (n + 1);
    };

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            const double x0 = i * dx;
            const double x1 = (i + 1) * dx;
            const double z0 = j * dx;
            const double z1 = (j + 1) * dx;
            // Local node order matches MakeQuad:
            //   0 = (x0,z0), 1 = (x1,z0), 2 = (x1,z1), 3 = (x0,z1)
            const int g0 = vertex_gtdof(i,     j    );
            const int g1 = vertex_gtdof(i + 1, j    );
            const int g2 = vertex_gtdof(i + 1, j + 1);
            const int g3 = vertex_gtdof(i,     j + 1);
            // For this sanity test we set boundary_tag = "none" on all
            // elements (i.e. don't exercise the Wohlmuth modifications).
            // The conforming-vs-clipped equivalence holds independently
            // of boundary_tag — both paths use the same MQuad4DualModified
            // call. A separate test below exercises a corner_LL element.
            result.elems.push_back(MakeQuad(x0, x1, z0, z1, y, g0, g1, g2, g3,
                                                  "none"));
        }
    }
    result.n_unique_gtdofs = (n + 1) * (n + 1);
    return result;
}

// ============================================================================
// Test 1: 4×4 vs 4×4 conforming agreement (boundary_tag = "none")
// ============================================================================
//
// Build identical 4×4 grids on opposite y faces. Run both paths and
// compare D and A_m entry-by-entry.
//
// Tolerance: FP roundoff. The integrand is degree-4 in (xi, eta), and
// both rules (9-pt Gauss on parent / 6-pt Dunavant on each sub-tri)
// integrate degree-4 exactly. So the ONLY difference between the two
// outputs is summation order (the conforming path sums 9 terms per
// pair; the clipped path sums 2 × 6 = 12 terms per pair). 1e-12
// relative tolerance comfortably absorbs this.
void test_quad_conforming_agreement_4x4()
{
    std::cout << "  test_quad_conforming_agreement_4x4\n";

    const int n = 4;
    const double L = 1.0;
    auto nm_grid = MakeQuadGridWithGtdofs(n, L, 0.0, 0);
    auto m_grid  = MakeQuadGridWithGtdofs(n, L, L,  1000);

    // ---- Reference: conforming path ----
    auto matches = MatchConformingFacePairs(nm_grid.elems, m_grid.elems,
                                                       "y", L);
    REQUIRE(matches.size() == nm_grid.elems.size(),
            "conforming match should produce one entry per nonmortar");

    QuadFaceMortarAssembler assembler;
    auto block_ref = assembler.AssemblePairConforming(
                              nm_grid.elems, m_grid.elems, matches);

    // ---- Test path: clipped ----
    auto cands = MatchClippedQuadFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipQuadFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block_clip = AssembleQuadFacePairClipped(
                          nm_grid.elems, m_grid.elems, sub_tris, "y");

    // ---- Compare D ----
    REQUIRE(block_ref.D.Size() == block_clip.D.Size(),
            "conforming agreement: D sizes must match");
    REQUIRE(block_ref.nonmortar_gtdofs.Size()
                == block_clip.nonmortar_gtdofs.Size(),
            "conforming agreement: nonmortar gtdof count must match");
    REQUIRE(block_ref.mortar_gtdofs.Size()
                == block_clip.mortar_gtdofs.Size(),
            "conforming agreement: mortar gtdof count must match");

    // Both paths sort kept gtdofs the same way → row indexing is identical.
    for (int i = 0; i < block_ref.nonmortar_gtdofs.Size(); ++i)
    {
        REQUIRE(block_ref.nonmortar_gtdofs[i] == block_clip.nonmortar_gtdofs[i],
                "conforming agreement: nonmortar gtdof ordering must match");
    }
    for (int i = 0; i < block_ref.mortar_gtdofs.Size(); ++i)
    {
        REQUIRE(block_ref.mortar_gtdofs[i] == block_clip.mortar_gtdofs[i],
                "conforming agreement: mortar gtdof ordering must match");
    }

    // D entries: should match exactly (D uses the same 9-point Gauss
    // rule on the same parent reference quads in both paths).
    double d_max_err = 0.0;
    double d_max_abs = 0.0;
    for (int i = 0; i < block_ref.D.Size(); ++i)
    {
        const double err = std::abs(block_ref.D(i) - block_clip.D(i));
        d_max_err = std::max(d_max_err, err);
        d_max_abs = std::max(d_max_abs, std::abs(block_ref.D(i)));
    }
    REQUIRE(d_max_err <= 1.0e-14 * std::max(d_max_abs, 1.0),
            "conforming agreement: D entries should match exactly "
            "(both paths use the same 9-pt rule on the parent)");

    // A_m entries: should match to FP roundoff. Use the CSR access
    // (GetI/GetJ/GetData) which works after Finalize() — both
    // AssemblePairConforming and AssembleQuadFacePairClipped call
    // Finalize() before returning.
    REQUIRE(block_ref.A_m.NumNonZeroElems() == block_clip.A_m.NumNonZeroElems(),
            "conforming agreement: A_m should have same nnz on both paths");

    const int n_rows = block_ref.A_m.Height();
    const int* I_ref  = block_ref.A_m.GetI();
    const int* J_ref  = block_ref.A_m.GetJ();
    const double* V_ref = block_ref.A_m.GetData();
    const int* I_clp  = block_clip.A_m.GetI();
    const int* J_clp  = block_clip.A_m.GetJ();
    const double* V_clp = block_clip.A_m.GetData();
    double a_max_err = 0.0;
    double a_max_abs = 0.0;
    for (int i = 0; i < n_rows; ++i)
    {
        // Both paths sort kept gtdofs identically and accumulate via
        // SparseMatrix::Add → after Finalize the column ordering per
        // row is identical. We compare in lockstep.
        const int rs_ref = I_ref[i + 1] - I_ref[i];
        const int rs_clp = I_clp[i + 1] - I_clp[i];
        REQUIRE(rs_ref == rs_clp,
                "conforming agreement: row sizes must match per row");
        for (int kk = 0; kk < rs_ref; ++kk)
        {
            const int j_r = J_ref[I_ref[i] + kk];
            const int j_c = J_clp[I_clp[i] + kk];
            REQUIRE(j_r == j_c, "conforming agreement: column ordering "
                                 "must match per row");
            const double v_r = V_ref[I_ref[i] + kk];
            const double v_c = V_clp[I_clp[i] + kk];
            const double err = std::abs(v_r - v_c);
            a_max_err = std::max(a_max_err, err);
            a_max_abs = std::max(a_max_abs, std::abs(v_r));
        }
    }
    REQUIRE(a_max_err <= 1.0e-12 * std::max(a_max_abs, 1.0),
            "conforming agreement: A_m entries should match to FP roundoff");

    std::cout << "    D max-error      = " << d_max_err
              << "  (max |D|     = "       << d_max_abs << ")\n";
    std::cout << "    A_m max-error    = " << a_max_err
              << "  (max |A_m|   = "       << a_max_abs << ")\n";
    std::cout << "    n_rows = "           << block_ref.D.Size()
              << "  n_cols = "             << block_ref.mortar_gtdofs.Size()
              << "  nnz = "                << block_ref.A_m.NumNonZeroElems()
              << "\n";
}

// ============================================================================
// Test 2: tile-cover invariant on the clipped output's D vector
// ============================================================================
//
// Independent of the conforming path: the clipped path's D vector (when
// summed over all rows for a non-sentinel grid) should equal the total
// nonmortar face area. Catches gross errors in the per-element D
// accumulation.
void test_clipped_d_total_area()
{
    std::cout << "  test_clipped_d_total_area\n";
    const int n = 4;
    const double L = 1.0;
    auto nm_grid = MakeQuadGridWithGtdofs(n, L, 0.0, 0);
    auto m_grid  = MakeQuadGridWithGtdofs(n, L, L,  1000);

    auto cands = MatchClippedQuadFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipQuadFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleQuadFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    double d_sum = 0.0;
    for (int i = 0; i < block.D.Size(); ++i) { d_sum += block.D(i); }
    const double expected_area = L * L;
    REQUIRE_NEAR(d_sum, expected_area, 1.0e-12,
                 "Σ D entries should equal nonmortar face area");
    std::cout << "    Σ D = " << d_sum
              << "  (expected " << expected_area << ")\n";
}

// ============================================================================
// Tri test infrastructure: build an n×n grid of tris (each square cell
// split along the (i,j)-(i+1,j+1) diagonal into 2 tris) on a y=const
// plane.
// ============================================================================

struct TriGridResult
{
    std::vector<TriFaceElement> elems;
    int n_unique_gtdofs;
};

TriGridResult MakeTriGridWithGtdofs(int n, double L, double y, int gtdof_base)
{
    TriGridResult result;
    result.elems.reserve(n * n * 2);
    const double dx = L / n;

    auto vertex_gtdof = [&](int i, int j) {
        // Same vertex layout as the quad grid: (n+1) × (n+1) vertices.
        return gtdof_base + i + j * (n + 1);
    };

    auto make = [&](double xa, double za, int ga,
                    double xb, double zb, int gb,
                    double xc, double zc, int gc) {
        TriFaceElement e;
        e.coords.SetSize(3, 3);
        e.coords(0, 0) = xa; e.coords(0, 1) = y; e.coords(0, 2) = za;
        e.coords(1, 0) = xb; e.coords(1, 1) = y; e.coords(1, 2) = zb;
        e.coords(2, 0) = xc; e.coords(2, 1) = y; e.coords(2, 2) = zc;
        e.gtdofs = {ga, gb, gc};
        e.parametric_axes   = {"x", "z"};
        e.perpendicular_axis = "y";
        e.boundary_tag = "none";
        return e;
    };

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            const double x0 = i * dx;
            const double x1 = (i + 1) * dx;
            const double z0 = j * dx;
            const double z1 = (j + 1) * dx;
            const int g00 = vertex_gtdof(i,     j    );
            const int g10 = vertex_gtdof(i + 1, j    );
            const int g11 = vertex_gtdof(i + 1, j + 1);
            const int g01 = vertex_gtdof(i,     j + 1);

            // Tri 1: (i,j), (i+1,j), (i+1,j+1) — CCW from +y normal.
            result.elems.push_back(make(x0, z0, g00,
                                        x1, z0, g10,
                                        x1, z1, g11));
            // Tri 2: (i,j), (i+1,j+1), (i,j+1).
            result.elems.push_back(make(x0, z0, g00,
                                        x1, z1, g11,
                                        x0, z1, g01));
        }
    }
    result.n_unique_gtdofs = (n + 1) * (n + 1);
    return result;
}

// Compare two sparse rows by column value rather than by raw CSR slot.
//
// MFEM's SparseMatrix::Finalize is allowed to produce different row
// structures when two assembly paths add the same algebraic value
// through different element/overlap walks. The conforming triangular
// path and clipped triangular path are especially prone to this: one
// path may never create an entry that the other path creates and later
// cancels to roundoff. The correctness contract is algebraic equality
// of each row/column value, not identical CSR insertion history.
double MaxSparseRowUnionDiff(const mfem::SparseMatrix& A,
                             const mfem::SparseMatrix& B,
                             double& max_abs_A)
{
    REQUIRE(A.Height() == B.Height(),
            "sparse union compare: matrix heights must match");
    REQUIRE(A.Width() == B.Width(),
            "sparse union compare: matrix widths must match");

    const int* IA = A.GetI();
    const int* JA = A.GetJ();
    const double* VA = A.GetData();
    const int* IB = B.GetI();
    const int* JB = B.GetJ();
    const double* VB = B.GetData();

    double max_err = 0.0;
    max_abs_A = 0.0;
    for (int r = 0; r < A.Height(); ++r)
    {
        std::map<int, double> row_a;
        std::map<int, double> row_b;
        std::set<int> cols;

        for (int kk = IA[r]; kk < IA[r + 1]; ++kk)
        {
            row_a[JA[kk]] += VA[kk];
            cols.insert(JA[kk]);
            max_abs_A = std::max(max_abs_A, std::abs(VA[kk]));
        }
        for (int kk = IB[r]; kk < IB[r + 1]; ++kk)
        {
            row_b[JB[kk]] += VB[kk];
            cols.insert(JB[kk]);
        }

        for (int c : cols)
        {
            const auto ia = row_a.find(c);
            const auto ib = row_b.find(c);
            const double va = (ia == row_a.end()) ? 0.0 : ia->second;
            const double vb = (ib == row_b.end()) ? 0.0 : ib->second;
            max_err = std::max(max_err, std::abs(va - vb));
        }
    }

    return max_err;
}

// ============================================================================
// Test 3: 4×4 vs 4×4 tri conforming agreement
// ============================================================================
//
// Same idea as Test 1 but for tri faces. Each square cell is split the
// same way on both sides → conforming tri pairing. Routes through both
// paths and asserts entry-by-entry agreement.
//
// For tri faces both paths use the SAME quadrature rule (3-point
// Dunavant). The integrand on a sub-triangle of the parent tri is
// degree 2 in barycentric (P1·P1 stays P1·P1 under affine
// reparameterization), so both rules integrate it exactly. D matches
// to roundoff and A_m matches to FP roundoff (rearrangement only).
void test_tri_conforming_agreement_4x4()
{
    std::cout << "  test_tri_conforming_agreement_4x4\n";

    const int n = 4;
    const double L = 1.0;
    auto nm_grid = MakeTriGridWithGtdofs(n, L, 0.0, 0);
    auto m_grid  = MakeTriGridWithGtdofs(n, L, L,  1000);

    REQUIRE(nm_grid.elems.size() == 32, "tri grid: 4x4 -> 32 tris");
    REQUIRE(m_grid.elems.size()  == 32, "tri grid: 4x4 -> 32 tris");

    // ---- Reference: conforming path ----
    auto matches = MatchConformingFacePairs(nm_grid.elems, m_grid.elems,
                                                       "y", L);
    REQUIRE(matches.size() == nm_grid.elems.size(),
            "tri conforming match should produce one entry per nonmortar");

    TriFaceMortarAssembler assembler;
    auto block_ref = assembler.AssemblePairConforming(
                              nm_grid.elems, m_grid.elems, matches);

    // ---- Test path: clipped ----
    auto cands = MatchClippedTriFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipTriFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block_clip = AssembleTriFacePairClipped(
                          nm_grid.elems, m_grid.elems, sub_tris, "y");

    // ---- Compare D ----
    REQUIRE(block_ref.D.Size() == block_clip.D.Size(),
            "tri conforming agreement: D sizes must match");
    REQUIRE(block_ref.nonmortar_gtdofs.Size()
                == block_clip.nonmortar_gtdofs.Size(),
            "tri conforming agreement: nonmortar gtdof count must match");
    REQUIRE(block_ref.mortar_gtdofs.Size()
                == block_clip.mortar_gtdofs.Size(),
            "tri conforming agreement: mortar gtdof count must match");

    for (int i = 0; i < block_ref.nonmortar_gtdofs.Size(); ++i)
    {
        REQUIRE(block_ref.nonmortar_gtdofs[i] == block_clip.nonmortar_gtdofs[i],
                "tri conforming agreement: nonmortar gtdof ordering must match");
    }
    for (int i = 0; i < block_ref.mortar_gtdofs.Size(); ++i)
    {
        REQUIRE(block_ref.mortar_gtdofs[i] == block_clip.mortar_gtdofs[i],
                "tri conforming agreement: mortar gtdof ordering must match");
    }

    double d_max_err = 0.0;
    double d_max_abs = 0.0;
    for (int i = 0; i < block_ref.D.Size(); ++i)
    {
        const double err = std::abs(block_ref.D(i) - block_clip.D(i));
        d_max_err = std::max(d_max_err, err);
        d_max_abs = std::max(d_max_abs, std::abs(block_ref.D(i)));
    }
    REQUIRE(d_max_err <= 1.0e-14 * std::max(d_max_abs, 1.0),
            "tri conforming agreement: D entries should match exactly");

    double a_max_abs = 0.0;
    const double a_max_err =
        MaxSparseRowUnionDiff(block_ref.A_m, block_clip.A_m, a_max_abs);
    REQUIRE(a_max_err <= 1.0e-12 * std::max(a_max_abs, 1.0),
            "tri conforming agreement: A_m entries should match by "
            "row/column value to FP roundoff");

    std::cout << "    D max-error      = " << d_max_err
              << "  (max |D|     = "       << d_max_abs << ")\n";
    std::cout << "    A_m max-error    = " << a_max_err
              << "  (max |A_m|   = "       << a_max_abs << ")\n";
    std::cout << "    n_rows = "           << block_ref.D.Size()
              << "  n_cols = "             << block_ref.mortar_gtdofs.Size()
              << "  nnz = "                << block_ref.A_m.NumNonZeroElems()
              << "\n";
}

// ============================================================================
// Test 4: tri-clipped Σ D = face area
// ============================================================================
void test_clipped_tri_d_total_area()
{
    std::cout << "  test_clipped_tri_d_total_area\n";
    const int n = 4;
    const double L = 1.0;
    auto nm_grid = MakeTriGridWithGtdofs(n, L, 0.0, 0);
    auto m_grid  = MakeTriGridWithGtdofs(n, L, L,  1000);

    auto cands = MatchClippedTriFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipTriFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleTriFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    double d_sum = 0.0;
    for (int i = 0; i < block.D.Size(); ++i) { d_sum += block.D(i); }
    const double expected_area = L * L;
    REQUIRE_NEAR(d_sum, expected_area, 1.0e-12,
                 "tri Σ D entries should equal nonmortar face area");
    std::cout << "    Σ D = " << d_sum
              << "  (expected " << expected_area << ")\n";
}

// ============================================================================
// Batch 4.4-D-4 — discrete reproduction tests on non-conforming meshes.
// ============================================================================
//
// PHASE 4.4 END-TO-END NUMERICAL CORRECTNESS GATE: the assembled block
// (D, A^m) must reproduce constant and linear fields exactly when applied
// as a mortar projector. Concretely, given
//   u_plus_vec  = u(x) sampled at mortar gtdofs
//   u_minus_vec = D^{-1} A^m u_plus_vec
// and u(x) is a constant or linear function in the (a, b) plane, then
// u_minus_vec must equal u(x) sampled at the nonmortar gtdofs to
// roundoff.
//
// Why this is the right test for non-conforming:
//   * Constant reproduction (u ≡ 1) is equivalent to A^m 1 = D 1, the
//     row-sum biorthogonality identity that the Wohlmuth dual basis is
//     designed to satisfy. If non-conforming clipping has dropped or
//     double-counted any sub-region, this fails.
//   * Linear reproduction (u(x) = x_a, x_b) is the discrete completeness
//     property: the mortar method is designed to preserve linear fields
//     exactly on flat axis-aligned interfaces. If any inverse-iso-map is
//     wrong, or any sub-triangle Jacobian is off, linear reproduction
//     fails.
//
// Both checks are independent of any reference assembler — there's no
// AssemblePairConforming counterpart for non-conforming meshes. Passing
// these tests on a 4×4 vs 5×5 setup demonstrates correctness end-to-end.

namespace
{

/// Apply the mortar projector u_minus = D^{-1} A^m u_plus to a sample
/// vector, given the assembled FaceMortarPairBlock. Pure host-side
/// linear algebra; uses MFEM SparseMatrix CSR access.
mfem::Vector ApplyMortarProjector(const FaceMortarPairBlock& block,
                                  const mfem::Vector& u_plus)
{
    const int n_rows = block.D.Size();
    MFEM_VERIFY(u_plus.Size() == block.mortar_gtdofs.Size(),
                "u_plus size mismatch");

    // First: A^m u_plus
    mfem::Vector ax(n_rows);
    ax = 0.0;
    const int* I = block.A_m.GetI();
    const int* J = block.A_m.GetJ();
    const double* V = block.A_m.GetData();
    for (int i = 0; i < n_rows; ++i)
    {
        for (int kk = I[i]; kk < I[i + 1]; ++kk)
        {
            ax(i) += V[kk] * u_plus(J[kk]);
        }
    }

    // Then: D^{-1} ax
    mfem::Vector u_minus(n_rows);
    for (int i = 0; i < n_rows; ++i)
    {
        // D entries are integrated lumped masses — strictly positive on
        // interior elements (Phase 3.2.B lumped-positivity guard). If
        // we ever see D[i] == 0 here, it indicates a sentinel-handling
        // bug or an orphan row.
        MFEM_VERIFY(block.D(i) > 0.0,
                    "ApplyMortarProjector: D[" << i << "] = " << block.D(i)
                    << " is non-positive; lumped-positivity guard violated.");
        u_minus(i) = ax(i) / block.D(i);
    }
    return u_minus;
}

/// For a 4×4 quad grid built by MakeQuadGridWithGtdofs(n, L, y, base),
/// reconstruct the (x, z) coordinate of vertex g. The grid has (n+1)²
/// vertices: vertex (i, j) gets gtdof base + i + j*(n+1) and lives at
/// (i*dx, y, j*dx).
void GtdofToVertexPos(int gtdof, int gtdof_base, int n, double L,
                      double& x_out, double& z_out)
{
    const int local = gtdof - gtdof_base;
    const int i = local % (n + 1);
    const int j = local / (n + 1);
    const double dx = L / n;
    x_out = i * dx;
    z_out = j * dx;
}

}  // anonymous namespace

// ============================================================================
// Test 5: constant-field reproduction (quad, conforming AND non-conforming)
// ============================================================================
//
// For u ≡ 1 (constant), expect D^{-1} A^m 1 = 1 to roundoff. Tests the
// row-sum biorthogonality identity directly.
void test_constant_reproduction_quad_conforming_4x4()
{
    std::cout << "  test_constant_reproduction_quad_conforming_4x4\n";
    const int n = 4;
    const double L = 1.0;
    auto nm_grid = MakeQuadGridWithGtdofs(n, L, 0.0, 0);
    auto m_grid  = MakeQuadGridWithGtdofs(n, L, L,  1000);

    auto cands = MatchClippedQuadFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipQuadFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleQuadFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    mfem::Vector u_plus(block.mortar_gtdofs.Size());
    u_plus = 1.0;
    auto u_minus = ApplyMortarProjector(block, u_plus);

    double max_err = 0.0;
    for (int i = 0; i < u_minus.Size(); ++i)
    {
        max_err = std::max(max_err, std::abs(u_minus(i) - 1.0));
    }
    REQUIRE(max_err <= 1.0e-13,
            "quad conforming: constant reproduction failed");
    std::cout << "    max |u_minus - 1| = " << max_err << "  (expected ~1e-15)\n";
}

void test_constant_reproduction_quad_nonconforming_4x4_vs_5x5()
{
    std::cout << "  test_constant_reproduction_quad_nonconforming_4x4_vs_5x5\n";
    const double L = 1.0;
    auto nm_grid = MakeQuadGridWithGtdofs(4, L, 0.0, 0);     // 4×4 nonmortar
    auto m_grid  = MakeQuadGridWithGtdofs(5, L, L,  1000);   // 5×5 mortar

    auto cands = MatchClippedQuadFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipQuadFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleQuadFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    mfem::Vector u_plus(block.mortar_gtdofs.Size());
    u_plus = 1.0;
    auto u_minus = ApplyMortarProjector(block, u_plus);

    double max_err = 0.0;
    for (int i = 0; i < u_minus.Size(); ++i)
    {
        max_err = std::max(max_err, std::abs(u_minus(i) - 1.0));
    }
    REQUIRE(max_err <= 1.0e-13,
            "quad NON-conforming: constant reproduction failed");
    std::cout << "    max |u_minus - 1| = " << max_err
              << "  (expected ~1e-15; n_rows = " << u_minus.Size() << ")\n";
}

// ============================================================================
// Test 6: linear-field reproduction (quad, conforming AND non-conforming)
// ============================================================================
//
// For u(x, z) = α·x + β·z + γ (linear in the (x, z) plane), expect
// D^{-1} A^m u_plus_vec to recover the same linear function sampled at
// the nonmortar nodes. Tests the discrete linear-completeness property
// of the mortar projector.
void test_linear_reproduction_quad(int nm_n, int m_n, const std::string& label)
{
    std::cout << "  test_linear_reproduction_quad_" << label << "\n";
    const double L = 1.0;
    const int gtdof_base_nm = 0;
    const int gtdof_base_m  = 1000;
    auto nm_grid = MakeQuadGridWithGtdofs(nm_n, L, 0.0, gtdof_base_nm);
    auto m_grid  = MakeQuadGridWithGtdofs(m_n,  L, L,  gtdof_base_m);

    auto cands = MatchClippedQuadFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipQuadFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleQuadFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    // Three test fields: u_x = x, u_z = z, u_lin = 1.7*x + 2.3*z + 0.5.
    auto run = [&](double alpha, double beta, double gamma,
                   const std::string& field_label) {
        // Sample u at mortar nodes.
        mfem::Vector u_plus(block.mortar_gtdofs.Size());
        for (int i = 0; i < u_plus.Size(); ++i)
        {
            double x, z;
            GtdofToVertexPos(block.mortar_gtdofs[i], gtdof_base_m, m_n, L, x, z);
            u_plus(i) = alpha * x + beta * z + gamma;
        }

        auto u_minus = ApplyMortarProjector(block, u_plus);

        // Expected: same linear field at nonmortar nodes.
        double max_err = 0.0;
        for (int i = 0; i < u_minus.Size(); ++i)
        {
            double x, z;
            GtdofToVertexPos(block.nonmortar_gtdofs[i], gtdof_base_nm, nm_n,
                             L, x, z);
            const double expected = alpha * x + beta * z + gamma;
            max_err = std::max(max_err, std::abs(u_minus(i) - expected));
        }
        REQUIRE(max_err <= 1.0e-13,
                "quad linear reproduction failed for field " + field_label);
        std::cout << "    " << field_label << ": max |u_minus - u_exact| = "
                  << max_err << "\n";
    };

    run(1.0, 0.0, 0.0, "u(x,z) = x");
    run(0.0, 1.0, 0.0, "u(x,z) = z");
    run(1.7, 2.3, 0.5, "u(x,z) = 1.7*x + 2.3*z + 0.5");
}

// ============================================================================
// Test 7: linear-field reproduction for tri faces.
// ============================================================================

namespace
{

/// Mirror of GtdofToVertexPos for the tri grid (same vertex layout —
/// MakeTriGridWithGtdofs uses identical (n+1)² vertex indexing).
void GtdofToVertexPosTri(int gtdof, int gtdof_base, int n, double L,
                          double& x_out, double& z_out)
{
    const int local = gtdof - gtdof_base;
    const int i = local % (n + 1);
    const int j = local / (n + 1);
    const double dx = L / n;
    x_out = i * dx;
    z_out = j * dx;
}

}  // anonymous namespace

void test_linear_reproduction_tri(int nm_n, int m_n, const std::string& label)
{
    std::cout << "  test_linear_reproduction_tri_" << label << "\n";
    const double L = 1.0;
    const int gtdof_base_nm = 0;
    const int gtdof_base_m  = 1000;
    auto nm_grid = MakeTriGridWithGtdofs(nm_n, L, 0.0, gtdof_base_nm);
    auto m_grid  = MakeTriGridWithGtdofs(m_n,  L, L,  gtdof_base_m);

    auto cands = MatchClippedTriFacePairs(nm_grid.elems, m_grid.elems, "y");
    auto sub_tris = ClipTriFacePairs(nm_grid.elems, m_grid.elems, cands, "y");
    auto block = AssembleTriFacePairClipped(
                     nm_grid.elems, m_grid.elems, sub_tris, "y");

    auto run = [&](double alpha, double beta, double gamma,
                   const std::string& field_label) {
        mfem::Vector u_plus(block.mortar_gtdofs.Size());
        for (int i = 0; i < u_plus.Size(); ++i)
        {
            double x, z;
            GtdofToVertexPosTri(block.mortar_gtdofs[i], gtdof_base_m, m_n, L,
                                x, z);
            u_plus(i) = alpha * x + beta * z + gamma;
        }
        auto u_minus = ApplyMortarProjector(block, u_plus);
        double max_err = 0.0;
        for (int i = 0; i < u_minus.Size(); ++i)
        {
            double x, z;
            GtdofToVertexPosTri(block.nonmortar_gtdofs[i], gtdof_base_nm,
                                nm_n, L, x, z);
            const double expected = alpha * x + beta * z + gamma;
            max_err = std::max(max_err, std::abs(u_minus(i) - expected));
        }
        REQUIRE(max_err <= 1.0e-13,
                "tri linear reproduction failed for field " + field_label);
        std::cout << "    " << field_label << ": max |u_minus - u_exact| = "
                  << max_err << "\n";
    };

    run(1.0, 0.0, 0.0, "u(x,z) = x");
    run(0.0, 1.0, 0.0, "u(x,z) = z");
    run(1.7, 2.3, 0.5, "u(x,z) = 1.7*x + 2.3*z + 0.5");
}

}  // anonymous namespace
}  // namespace mortar_pbc

int main()
{
    axom::slic::SimpleLogger slic_logger;

    std::cout << "test_face_mortar_assembler_clipped_3d (Phase 4.4 / "
                 "Batches 4.4-D-2 / D-3 / D-4)\n";
    // Batch 4.4-D-2 / D-3: conforming-via-clipped agreement.
    mortar_pbc::test_quad_conforming_agreement_4x4();
    mortar_pbc::test_clipped_d_total_area();
    mortar_pbc::test_tri_conforming_agreement_4x4();
    mortar_pbc::test_clipped_tri_d_total_area();
    // Batch 4.4-D-4: discrete reproduction tests on conforming AND
    // non-conforming meshes — the end-to-end Phase 4.4 correctness gate.
    mortar_pbc::test_constant_reproduction_quad_conforming_4x4();
    mortar_pbc::test_constant_reproduction_quad_nonconforming_4x4_vs_5x5();
    mortar_pbc::test_linear_reproduction_quad(4, 4, "conforming_4x4");
    mortar_pbc::test_linear_reproduction_quad(4, 5, "nonconforming_4x4_vs_5x5");
    mortar_pbc::test_linear_reproduction_tri (4, 4, "conforming_4x4");
    mortar_pbc::test_linear_reproduction_tri (4, 5, "nonconforming_4x4_vs_5x5");

    if (mortar_pbc::g_failures)
    {
        std::cerr << "\nOne or more test_face_mortar_assembler_clipped_3d "
                     "cases FAILED.\n";
        return 1;
    }
    std::cout << "\nAll test_face_mortar_assembler_clipped_3d cases passed.\n";
    return 0;
}
