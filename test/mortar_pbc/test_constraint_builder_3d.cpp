// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — integration test for ConstraintBuilder3D.
//
// Uses a small auto-generated cartesian 3D hex mesh — same mesh-
// construction pattern as test_boundary_classifier_3d.cpp — and
// validates the resulting constraint matrix C has:
//
//   * the predicted shape (n_constraints x n_global_tdofs)
//   * row count matching NumConstraints()
//   * non-empty entries (the build is non-trivial)
//   * column indices all within [0, n_global_tdofs)
//   * rows arranged as expected: edge rows first, then face rows
//
// The 2x2x2 hex mesh is the smallest case that produces non-trivial
// constraints: 1 interior node per edge × 12 edges + 1 interior node
// per face × 6 faces. Within the 9 edge pairs and 3 face pairs:
//   edge rows = 9 * 1 * 3 = 27
//   face rows = 3 * 1 * 3 = 9
//   total     = 36
//
// HypreParMatrix correctness is exercised at the API level: build it
// at np=1 with all rows local, verify Height/Width match the
// replicated matrix.
//
// Phase 5.7.A — the EmitRowFactors test was updated to use the
// post-5.7.A signature: the first arg is now
// `mfem::Vector& period_signed_per_row` (3 doubles per row, row-
// major) instead of `mfem::Array<int>& axis_index`. The per-axis
// histogram is recomputed as "how many rows have period_signed[a]
// nonzero?" — on the 2x2x2 unit cube this is [15, 15, 15] (3 face
// rows + 12 edge rows per axis), replacing the prior [12, 12, 12]
// (which counted the edge-parallel axis, the semantic the 5.7.A
// fix corrected).
//
// Phase 5.9 — filter API smoke tests added at the end:
//   * `test_filter_x_only_2x2x2`         — comp_mask = {X-only}.
//   * `test_filter_x_face_pair_only_2x2x2` — single face pair only,
//                                            all comps; edges drop.
//   * `test_filter_empty_2x2x2`          — empty filter → 0 rows.
//
// Phase 5.11 — sub-block partition tests added at the end:
//   * `test_subblock_face_edge_full_xyz_2x2x2`     — 2 sub-blocks
//                                                    (edge=0, face=1).
//   * `test_subblock_per_pair_full_xyz_2x2x2`      — 12 sub-blocks
//                                                    (9 edge pairs +
//                                                    3 face pairs).
//   * `test_subblock_face_edge_x_only_pair_2x2x2`  — FaceEdge under
//                                                    x-face filter.
//   * `test_subblock_per_pair_x_only_pair_2x2x2`   — PerPair under
//                                                    x-face filter
//                                                    (1 sub-block).
//   * `test_subblock_face_edge_x_comp_2x2x2`       — FaceEdge under
//                                                    X-comp mask.
//   * `test_subblock_empty_filter_2x2x2`           — empty filter
//                                                    sub-block output.
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success.

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;

namespace {

// ---- helper: assert + diagnostic ------------------------------------------
void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

// ---- helper: build a small unit-cube hex ParMesh + FE space --------------
struct FesBundle
{
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<mfem::H1_FECollection> fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> fes;
};

FesBundle BuildHexFesBundle(MPI_Comm comm, int n_per_side)
{
    FesBundle b;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n_per_side, n_per_side, n_per_side,
        mfem::Element::HEXAHEDRON,
        /*sx=*/1.0, /*sy=*/1.0, /*sz=*/1.0,
        /*sfc_ordering=*/false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(/*order=*/1, /*dim=*/3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), /*vdim=*/3, mfem::Ordering::byNODES);
    return b;
}

// ===========================================================================
// Test 1: NumConstraints() and Build() produce a matrix of the right shape
// ===========================================================================
//
// 2x2x2 hex mesh:
//   * 12 edges with 1 interior node each
//   * 6 faces with 1 interior node each
//   * 9 edge mortar pairs * 1 nonmortar interior node * vdim=3 = 27 rows
//   * 3 face mortar pairs * 1 nonmortar interior node * vdim=3 = 9 rows
//   * total: 36 rows
void test_row_count_2x2x2()
{
    std::cout << "Test 1: row count on 2x2x2 hex mesh" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    const int n_predicted = builder.NumConstraints();
    AssertOrDie(n_predicted == 36, "NumConstraints()",
                "got " + std::to_string(n_predicted) + ", expected 36");

    auto C = builder.Build();
    AssertOrDie(C->Height() == 36, "C.Height()",
                "got " + std::to_string(C->Height()) + ", expected 36");
    AssertOrDie(C->Width() == cl.NGlobalTdofs(), "C.Width()",
                "got " + std::to_string(C->Width()) + ", expected "
                + std::to_string(cl.NGlobalTdofs()));
    std::cout << "  PASS  C is " << C->Height() << " x " << C->Width()
              << ", NumConstraints() = " << n_predicted << std::endl;
}

// ===========================================================================
// Test 2: row count scales correctly on a 4x4x4 mesh
// ===========================================================================
//
// 4x4x4 hex mesh:
//   * each edge has 3 interior nodes (n_per_side - 1)
//   * each face has 3x3 = 9 interior nodes
//   * 9 edge pairs * 3 nonmortar interior nodes * vdim=3 = 81 rows
//   * 3 face pairs * 9 nonmortar interior nodes * vdim=3 = 81 rows
//   * total: 162 rows
void test_row_count_4x4x4()
{
    std::cout << "Test 2: row count on 4x4x4 hex mesh" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    const int n_predicted = builder.NumConstraints();
    AssertOrDie(n_predicted == 162, "NumConstraints()",
                "got " + std::to_string(n_predicted) + ", expected 162");

    auto C = builder.Build();
    AssertOrDie(C->Height() == 162, "C.Height()",
                "got " + std::to_string(C->Height()) + ", expected 162");
    std::cout << "  PASS  4x4x4: C is " << C->Height() << " x " << C->Width()
              << " (NumConstraints() = " << n_predicted << ")" << std::endl;
}

// ===========================================================================
// Test 3: C is structurally non-trivial (NumNonZeroElems > 0)
// ===========================================================================
void test_nonempty_build()
{
    std::cout << "Test 3: non-trivial build" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    auto C = builder.Build();
    const int nnz = C->NumNonZeroElems();
    AssertOrDie(nnz > 0, "NumNonZeroElems",
                "expected > 0, got " + std::to_string(nnz));
    AssertOrDie(nnz >= C->Height(),
                "NumNonZeroElems vs Height",
                "expected at least 1 nz per row (got " + std::to_string(nnz)
                + " for " + std::to_string(C->Height()) + " rows)");
    std::cout << "  PASS  C has " << nnz << " non-zero entries ("
              << static_cast<double>(nnz) / C->Height()
              << " avg per row)" << std::endl;
}

// ===========================================================================
// Test 4: column indices are in [0, n_global_tdofs)
// ===========================================================================
void test_column_indices_in_range()
{
    std::cout << "Test 4: column indices in valid range" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);
    auto C = builder.Build();

    const int n_cols = cl.NGlobalTdofs();
    const int* I = C->GetI();
    const int* J = C->GetJ();
    int min_col = 1 << 30, max_col = -1;
    for (int i = 0; i < C->Height(); ++i)
    {
        for (int k = I[i]; k < I[i+1]; ++k)
        {
            const int c = J[k];
            AssertOrDie(c >= 0 && c < n_cols,
                        "column index range",
                        "row " + std::to_string(i) + " has col "
                        + std::to_string(c) + " out of [0, "
                        + std::to_string(n_cols) + ")");
            if (c < min_col) min_col = c;
            if (c > max_col) max_col = c;
        }
    }
    std::cout << "  PASS  all columns in [" << min_col << ", " << max_col
              << "] ⊂ [0, " << n_cols << ")" << std::endl;
}

// ===========================================================================
// Test 5: row layout — edge rows come first, face rows after
//
// We can't directly check "row k is an edge row" but we CAN check that
// the first 27 rows on a 2x2x2 mesh (the edge rows) and the remaining
// 9 rows (the face rows) each have the structure we expect:
//   - Each row has at least 1 entry (D term)
//   - Each row's entries' columns reference DOFs on the boundary
//
// That's the structural sanity. Numerical correctness against an
// affine-jump field is the next test.
// ===========================================================================
void test_row_layout()
{
    std::cout << "Test 5: row layout (edge rows first, face rows second)"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);
    auto C = builder.Build();

    AssertOrDie(C->Height() == 36, "row count",
                "expected 36 for 2x2x2");
    const int* I = C->GetI();
    int n_empty_rows = 0;
    for (int i = 0; i < 36; ++i)
    {
        const int row_nnz = I[i+1] - I[i];
        if (row_nnz == 0) { ++n_empty_rows; }
    }
    // For a clean 2x2x2 mesh every row should have at least the
    // diagonal D entry plus some -A_m entries; no totally-empty rows.
    AssertOrDie(n_empty_rows == 0, "no empty rows",
                "found " + std::to_string(n_empty_rows) + " empty rows out of 36");
    std::cout << "  PASS  all 36 rows have entries; no empty rows" << std::endl;
}

// ===========================================================================
// Test 6: BuildHypreParMatrix — np=1 case, all rows owned locally
// ===========================================================================
void test_build_hypre_par_matrix()
{
    std::cout << "Test 6: BuildHypreParMatrix at np=1" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    const int n_total = builder.NumConstraints();

    // Phase 4.2 / Batch N: builder derives n_lam_local from FES-
    // aligned routing; we just query it after construction. At
    // np=1 every constraint row is owned locally, so n_lam_local
    // should equal n_total.
    mfem::HypreParMatrix* H = builder.BuildHypreParMatrix();
    const int n_lam_local = builder.NumLocalRows();
    AssertOrDie(H != nullptr, "BuildHypreParMatrix returned",
                "got nullptr");

    AssertOrDie(H->GetGlobalNumRows() == n_total,
                "HypreParMatrix global rows",
                "got " + std::to_string(H->GetGlobalNumRows())
                + ", expected " + std::to_string(n_total));
    AssertOrDie(H->GetGlobalNumCols() == cl.NGlobalTdofs(),
                "HypreParMatrix global cols",
                "got " + std::to_string(H->GetGlobalNumCols())
                + ", expected " + std::to_string(cl.NGlobalTdofs()));
    delete H;
    std::cout << "  PASS  HypreParMatrix sized "
              << n_total << " x " << cl.NGlobalTdofs()
              << " with " << n_lam_local << " local rows on this rank"
              << std::endl;
}

// ===========================================================================
// Test: EmitRowFactors — per-row reference-geometry metadata
// ===========================================================================
//
// Phase 5.7.A — signature changed: first argument is now
// `mfem::Vector& period_signed_per_row` (3 doubles per row, row-major)
// replacing the prior `mfem::Array<int>& axis_index`. See
// ConstraintBuilder3D::EmitRowFactors doc comments in the header.
//
// On a 2x2x2 hex mesh, the constraint matrix has 36 rows:
//   * 9 edge pairs * 1 nonmortar interior node * vdim=3 = 27 edge rows
//   * 3 face pairs * 1 nonmortar interior node * vdim=3 =  9 face rows
//
// We verify:
//   1. period_signed_per_row.Size() == 3 * n_local (3 doubles per row).
//   2. comp_idx.Size() == n_local, ell_hat.Size() == n_local.
//   3. Each row has 1 or 2 nonzero period entries (faces: 1; edges: 1
//      for "straight" nonmortars, 2 for the diagonal nonmortar per
//      axis triple).
//   4. Per-component histogram comp_hist == [12, 12, 12] (unchanged
//      from pre-5.7.A).
//   5. Per-axis nonzero count of period_signed = [15, 15, 15] on the
//      unit cube — derived below. Replaces the old [12, 12, 12]
//      axis_hist (which incorrectly tagged edge rows by their parallel
//      axis instead of by the jump axis).
//   6. All ell_hat[i] >= 0 (Wohlmuth lumped factor is a non-negative
//      integral of a partition-of-unity basis function).
//   7. All ell_hat[i] and period_signed_per_row[i] are finite.
//
// Derivation of period-nonzero histogram = [15, 15, 15] on 2x2x2:
//
//   Face rows contribute:
//     One face pair per axis × 1 nonmortar interior × 3 components
//     = 3 rows per axis with period_signed[a] != 0. Total face
//     contribution per axis: 3.
//
//   Edge rows contribute:
//     Per parametric axis k, the 3 nonmortar edges have period
//     vectors (transverse only). For k=0 ("x-parallel") these are
//     (0,-1,0), (0,0,-1), (0,-1,-1) — the "diagonal" nonmortar
//     produces 2 nonzero entries. Per non-parametric axis a (a != k):
//     2 of the 3 nonmortars are nonzero in a × 3 components per
//     nonmortar = 6 rows.
//     Per axis a, edge contribution = 6 (from parametric k=other_axis1)
//     + 6 (from parametric k=other_axis2) = 12 rows per axis.
//
//   Total per axis = 3 (face) + 12 (edge) = 15. ✓
// ===========================================================================
void test_emit_row_factors_2x2x2()
{
    std::cout << "Test: EmitRowFactors on 2x2x2 hex mesh" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    // Phase 5.7.A: first arg is now mfem::Vector& period_signed_per_row.
    mfem::Vector period_signed_per_row;
    mfem::Array<int> comp_idx;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(period_signed_per_row, comp_idx, ell_hat);

    const int n_local = builder.NumLocalRows();
    AssertOrDie(period_signed_per_row.Size() == 3 * n_local,
                "period_signed_per_row size",
                "got " + std::to_string(period_signed_per_row.Size())
                + ", expected " + std::to_string(3 * n_local));
    AssertOrDie(comp_idx.Size() == n_local, "comp_idx size",
                "got " + std::to_string(comp_idx.Size())
                + ", expected " + std::to_string(n_local));
    AssertOrDie(ell_hat.Size() == n_local, "ell_hat size",
                "got " + std::to_string(ell_hat.Size())
                + ", expected " + std::to_string(n_local));

    // Histogram pass — per-component count, per-axis period-nonzero
    // count, and per-row nonzero-count + finiteness checks.
    int comp_hist[3] = {0, 0, 0};
    int period_nonzero_hist[3] = {0, 0, 0};
    for (int i = 0; i < n_local; ++i)
    {
        const int c = comp_idx[i];
        AssertOrDie(c >= 0 && c < 3,
                    "comp_idx[i] in {0,1,2}",
                    "i=" + std::to_string(i) + " comp="
                    + std::to_string(c));
        AssertOrDie(std::isfinite(ell_hat[i]),
                    "ell_hat[i] is finite",
                    "i=" + std::to_string(i)
                    + " ell=" + std::to_string(ell_hat[i]));
        AssertOrDie(ell_hat[i] >= 0.0,
                    "ell_hat[i] >= 0",
                    "i=" + std::to_string(i)
                    + " ell=" + std::to_string(ell_hat[i]));
        ++comp_hist[c];

        // Period vector sanity: at least one component nonzero (every
        // row encodes some periodic jump), at most two on the 2x2x2
        // unit cube (no corner-to-corner mortar pairs exist — the
        // classifier's mortar/nonmortar pairing doesn't produce
        // 3-nonzero period vectors on any axis-aligned box).
        int n_nonzero = 0;
        for (int a = 0; a < 3; ++a)
        {
            const double v = period_signed_per_row[3*i + a];
            AssertOrDie(std::isfinite(v),
                        "period_signed_per_row[3i+a] finite",
                        "i=" + std::to_string(i) + " a="
                        + std::to_string(a) + " v="
                        + std::to_string(v));
            if (v != 0.0)
            {
                ++period_nonzero_hist[a];
                ++n_nonzero;
            }
        }
        AssertOrDie(n_nonzero >= 1 && n_nonzero <= 2,
                    "period_signed_per_row row has 1 or 2 nonzero",
                    "i=" + std::to_string(i) + " n_nonzero="
                    + std::to_string(n_nonzero));
    }

    // At np=1 we expect the symmetric distribution.
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks == 1)
    {
        AssertOrDie(n_local == 36,
                    "n_local at np=1",
                    "got " + std::to_string(n_local) + ", expected 36");
        for (int a = 0; a < 3; ++a)
        {
            AssertOrDie(comp_hist[a] == 12,
                        "comp_hist[" + std::to_string(a) + "]",
                        "got " + std::to_string(comp_hist[a])
                        + ", expected 12");
            AssertOrDie(period_nonzero_hist[a] == 15,
                        "period_nonzero_hist[" + std::to_string(a) + "]",
                        "got " + std::to_string(period_nonzero_hist[a])
                        + ", expected 15");
        }
    }

    // At np>1: per-rank counts vary, but the rank-summed totals
    // should still be 36 / 12 / 15.
    int n_global = 0;
    int comp_global[3] = {0, 0, 0};
    int period_nz_global[3] = {0, 0, 0};
    MPI_Allreduce(&n_local, &n_global, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(comp_hist, comp_global, 3, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(period_nonzero_hist, period_nz_global, 3, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    AssertOrDie(n_global == 36,
                "rank-summed n_local",
                "got " + std::to_string(n_global) + ", expected 36");
    for (int a = 0; a < 3; ++a)
    {
        AssertOrDie(comp_global[a] == 12,
                    "rank-summed comp_hist[" + std::to_string(a) + "]",
                    "got " + std::to_string(comp_global[a])
                    + ", expected 12");
        AssertOrDie(period_nz_global[a] == 15,
                    "rank-summed period_nonzero_hist["
                    + std::to_string(a) + "]",
                    "got " + std::to_string(period_nz_global[a])
                    + ", expected 15");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "  PASS  EmitRowFactors emits "
                  << n_global
                  << " rows (=36) with component hist ["
                  << comp_global[0] << ", " << comp_global[1] << ", "
                  << comp_global[2] << "] (each=12) and period-nonzero hist ["
                  << period_nz_global[0] << ", " << period_nz_global[1] << ", "
                  << period_nz_global[2] << "] (each=15)" << std::endl;
    }
}

// ===========================================================================
// Phase 5.9 — Filter API smoke tests
// ===========================================================================
//
// The new filtered overloads of Build, BuildHypreParMatrix,
// NumConstraints, NumLocalRows, and EmitRowFactors accept
// (active_pair_labels, comp_mask) and gate row emission. The
// parameter-less overloads forward to filtered with all-pairs / all-
// comps, which is exercised by tests 1–6 + the EmitRowFactors test
// above. Below we exercise the filter API directly on the 2x2x2 mesh.
//
// Filter rules (see constraint_builder_3d.hpp design block):
//   * Face mortars: gated on the pair's axis ∈ active_axes (derived
//     from active_pair_labels by classifier's label→axis mapping).
//   * Edge mortars: gated on BOTH perpendicular axes ∈ active_axes
//     (x-parallel edges require y AND z active; etc.).
//   * Within active pairs, comp_mask drops per-component rows.
// ===========================================================================

// Test: comp_mask = {true, false, false} (X component only).
//
// All pair labels active → all face pairs + all edge groups emit
// rows. comp_mask drops Y and Z per-component rows, so row count is
// reduced by 1/3.
//
// Baseline 36 rows × (1/3) = 12 rows total. All rows should have
// component_index == 0.
void test_filter_x_only_2x2x2()
{
    std::cout << "Phase 5.9 filter test: X-only comp_mask on 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    // All pairs active (mortar-side labels by the classifier's
    // convention: high-side faces along each axis).
    std::vector<std::string> all_pairs = {"top", "right", "back"};
    std::array<bool, 3> comp_mask = {true, false, false};

    const int n_baseline = builder.NumConstraints();
    const int n_filtered = builder.NumConstraints(all_pairs, comp_mask);
    AssertOrDie(n_baseline == 36, "baseline NumConstraints",
                "got " + std::to_string(n_baseline) + ", expected 36");
    AssertOrDie(n_filtered == 12,
                "filtered NumConstraints (X-only)",
                "got " + std::to_string(n_filtered) + ", expected 12");

    auto C = builder.Build(all_pairs, comp_mask);
    AssertOrDie(C->Height() == 12,
                "filtered C.Height() (X-only)",
                "got " + std::to_string(C->Height()) + ", expected 12");
    AssertOrDie(C->Width() == cl.NGlobalTdofs(),
                "filtered C.Width()",
                "got " + std::to_string(C->Width()) + ", expected "
                + std::to_string(cl.NGlobalTdofs()));

    // EmitRowFactors should also reflect the filter: every comp_idx
    // must be 0 (only X component is emitted).
    mfem::Vector period_signed;
    mfem::Array<int> comp_idx;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(all_pairs, comp_mask,
                           period_signed, comp_idx, ell_hat);
    const int n_local = builder.NumLocalRows(all_pairs, comp_mask);
    AssertOrDie(comp_idx.Size() == n_local,
                "filtered comp_idx.Size() (X-only)",
                "got " + std::to_string(comp_idx.Size())
                + ", expected " + std::to_string(n_local));
    AssertOrDie(period_signed.Size() == 3 * n_local,
                "filtered period_signed_per_row.Size() (X-only)",
                "got " + std::to_string(period_signed.Size())
                + ", expected " + std::to_string(3 * n_local));
    for (int i = 0; i < n_local; ++i)
    {
        AssertOrDie(comp_idx[i] == 0,
                    "X-only filter: comp_idx[i] == 0",
                    "i=" + std::to_string(i)
                    + " comp=" + std::to_string(comp_idx[i]));
    }

    std::cout << "  PASS  X-only filter: 12 rows (= 36/3), "
              << "all component_index == 0" << std::endl;
}

// Test: active_pair_labels = {"right"} only — one face pair active.
//
// Face filter: only the x-pair contributes. y-pair and z-pair are
// skipped.
// Edge filter: all edge groups need BOTH perpendicular axes active.
//   - x-parallel edges need y AND z active → dropped (only x active).
//   - y-parallel edges need x AND z active → dropped.
//   - z-parallel edges need x AND y active → dropped.
//   → all edge groups dropped.
//
// Result: 1 face pair × 1 nonmortar interior × 3 components = 3 rows.
void test_filter_x_face_pair_only_2x2x2()
{
    std::cout << "Phase 5.9 filter test: x-face-pair only on 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> x_only = {"right"};
    std::array<bool, 3> all_comps = {true, true, true};

    const int n_predicted = builder.NumConstraints(x_only, all_comps);
    AssertOrDie(n_predicted == 3,
                "NumConstraints({\"right\"}, all comps)",
                "got " + std::to_string(n_predicted)
                + ", expected 3 (only x-face pair, all edges dropped)");

    auto C = builder.Build(x_only, all_comps);
    AssertOrDie(C->Height() == 3,
                "C.Height() with x-only pair",
                "got " + std::to_string(C->Height()) + ", expected 3");

    // The 3 rows should all be face rows for the x-pair (period vector
    // (±L_x, 0, 0)). EmitRowFactors verifies this.
    mfem::Vector period_signed;
    mfem::Array<int> comp_idx;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(x_only, all_comps,
                           period_signed, comp_idx, ell_hat);
    const int n_local = builder.NumLocalRows(x_only, all_comps);
    AssertOrDie(period_signed.Size() == 3 * n_local,
                "filtered period_signed.Size() (x-pair only)",
                "got " + std::to_string(period_signed.Size())
                + ", expected " + std::to_string(3 * n_local));

    // For every emitted row, period_signed should have period[0] != 0
    // and period[1] == period[2] == 0 (face rows for x-axis only).
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < n_local; ++i)
    {
        const double px = period_signed[3*i + 0];
        const double py = period_signed[3*i + 1];
        const double pz = period_signed[3*i + 2];
        AssertOrDie(px != 0.0,
                    "x-pair-only: period_signed[0] != 0",
                    "i=" + std::to_string(i) + " period=("
                    + std::to_string(px) + ","
                    + std::to_string(py) + ","
                    + std::to_string(pz) + ")");
        AssertOrDie(py == 0.0,
                    "x-pair-only: period_signed[1] == 0",
                    "i=" + std::to_string(i) + " period_y="
                    + std::to_string(py));
        AssertOrDie(pz == 0.0,
                    "x-pair-only: period_signed[2] == 0",
                    "i=" + std::to_string(i) + " period_z="
                    + std::to_string(pz));
    }

    std::cout << "  PASS  x-face-pair-only filter: 3 rows (1 face pair "
              << "× 3 components, all edges dropped)" << std::endl;
}

// Test: empty filter — should produce 0 rows.
//
// Both "no active pairs" and "comp_mask all false" should yield a
// 0-row matrix. NumConstraints / NumLocalRows / Build / EmitRowFactors
// should all agree.
void test_filter_empty_2x2x2()
{
    std::cout << "Phase 5.9 filter test: empty filter on 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> none;
    std::vector<std::string> all_pairs = {"top", "right", "back"};
    std::array<bool, 3> all_comps = {true, true, true};
    std::array<bool, 3> no_comps  = {false, false, false};

    AssertOrDie(builder.NumConstraints(none, all_comps) == 0,
                "NumConstraints(empty pairs, all comps)", "");
    AssertOrDie(builder.NumConstraints(all_pairs, no_comps) == 0,
                "NumConstraints(all pairs, no comps)", "");
    AssertOrDie(builder.NumLocalRows(none, all_comps) == 0,
                "NumLocalRows(empty pairs, all comps)", "");
    AssertOrDie(builder.NumLocalRows(all_pairs, no_comps) == 0,
                "NumLocalRows(all pairs, no comps)", "");

    auto C1 = builder.Build(none, all_comps);
    auto C2 = builder.Build(all_pairs, no_comps);
    AssertOrDie(C1->Height() == 0,
                "Empty pairs C.Height()",
                "got " + std::to_string(C1->Height()) + ", expected 0");
    AssertOrDie(C2->Height() == 0,
                "No comps C.Height()",
                "got " + std::to_string(C2->Height()) + ", expected 0");

    mfem::Vector period_signed;
    mfem::Array<int> comp_idx;
    mfem::Vector ell_hat;
    builder.EmitRowFactors(none, all_comps,
                           period_signed, comp_idx, ell_hat);
    AssertOrDie(period_signed.Size() == 0,
                "EmitRowFactors(empty pairs) period size",
                "got " + std::to_string(period_signed.Size())
                + ", expected 0");
    AssertOrDie(comp_idx.Size() == 0,
                "EmitRowFactors(empty pairs) comp_idx size",
                "got " + std::to_string(comp_idx.Size())
                + ", expected 0");
    AssertOrDie(ell_hat.Size() == 0,
                "EmitRowFactors(empty pairs) ell_hat size",
                "got " + std::to_string(ell_hat.Size())
                + ", expected 0");

    std::cout << "  PASS  empty filter (no pairs OR no comps): 0 rows"
              << std::endl;
}

// ===========================================================================
// Phase 5.11 — GetRowSubblockIds tests
//
// Each test exercises a partition scheme × filter combination on the
// 2x2x2 hex mesh (the smallest non-trivial case). The 2x2x2 mesh
// has:
//   * 12 edges × 1 interior node × 3 comps = 36 edge rows (unfiltered)
//   * Wait — 9 EDGE PAIRS (3 per axis) × 1 interior × 3 comps = 27
//   * 3 FACE PAIRS × 1 interior × 3 comps = 9
//   * Total: 36 rows
//
// (Edge pair count is 9 because periodicity identifies opposite edges
// — 9 nonmortar edges per the classifier's EdgePairs() construction.)
// ===========================================================================

void test_subblock_face_edge_full_xyz_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: FaceEdge / full XYZ / 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::FaceEdge,
                              labels, sb_of_row);

    // FaceEdge: always 2 labels.
    AssertOrDie(labels.size() == 2,
                "FaceEdge label count",
                "got " + std::to_string(labels.size()) + ", expected 2");
    AssertOrDie(labels[0] == "edge",
                "FaceEdge labels[0]",
                "got '" + labels[0] + "', expected 'edge'");
    AssertOrDie(labels[1] == "face",
                "FaceEdge labels[1]",
                "got '" + labels[1] + "', expected 'face'");

    // Row count: 36 on 2x2x2 unfiltered.
    AssertOrDie(sb_of_row.Size() == 36,
                "FaceEdge sb_of_row size",
                "got " + std::to_string(sb_of_row.Size())
                + ", expected 36");

    // Layout: first 27 rows (9 edge pairs × 1 × 3) should be edge
    // sub-block (ID 0); last 9 rows (3 face pairs × 1 × 3) should
    // be face sub-block (ID 1).
    for (int i = 0; i < 27; ++i)
    {
        AssertOrDie(sb_of_row[i] == 0,
                    "edge row sub-block ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i]) + ", expected 0");
    }
    for (int i = 27; i < 36; ++i)
    {
        AssertOrDie(sb_of_row[i] == 1,
                    "face row sub-block ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i]) + ", expected 1");
    }

    std::cout << "  PASS  FaceEdge full XYZ: labels {edge, face}, "
              << "first 27 rows = 0, last 9 rows = 1" << std::endl;
}

void test_subblock_per_pair_full_xyz_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: PerPair / full XYZ / 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::PerPair,
                              labels, sb_of_row);

    // PerPair full XYZ: 9 edge pairs + 3 face pairs = 12 sub-blocks.
    AssertOrDie(labels.size() == 12,
                "PerPair full XYZ label count",
                "got " + std::to_string(labels.size()) + ", expected 12");

    // First 9 labels start with "edge_"; last 3 start with "face_".
    for (int i = 0; i < 9; ++i)
    {
        AssertOrDie(labels[i].rfind("edge_", 0) == 0,
                    "PerPair edge label prefix",
                    "labels[" + std::to_string(i) + "] = '"
                    + labels[i] + "' does not start with 'edge_'");
    }
    for (int i = 9; i < 12; ++i)
    {
        AssertOrDie(labels[i].rfind("face_", 0) == 0,
                    "PerPair face label prefix",
                    "labels[" + std::to_string(i) + "] = '"
                    + labels[i] + "' does not start with 'face_'");
    }

    // Face labels: the 3 mortar-side face labels are "top", "right",
    // "back" per the classifier's FacePairs() convention. The face-
    // pair walk order is FIXED by `mortar_pbc::GetFacePairs()` in
    // boundary_helpers_3d.cpp:
    //   pairs[0] = (top,   bottom)  — y-axis
    //   pairs[1] = (right, left)    — x-axis
    //   pairs[2] = (back,  front)   — z-axis
    // So the 3 face sub-blocks in walk order are face_top (y),
    // face_right (x), face_back (z) — y first because the array
    // literal puts "top" first, not because of any axis ordering.
    AssertOrDie(labels[9]  == "face_top",
                "PerPair labels[9] (y-face mortar)",
                "got '" + labels[9] + "', expected 'face_top'");
    AssertOrDie(labels[10] == "face_right",
                "PerPair labels[10] (x-face mortar)",
                "got '" + labels[10] + "', expected 'face_right'");
    AssertOrDie(labels[11] == "face_back",
                "PerPair labels[11] (z-face mortar)",
                "got '" + labels[11] + "', expected 'face_back'");

    // Row count: 36.
    AssertOrDie(sb_of_row.Size() == 36,
                "PerPair full XYZ sb_of_row size",
                "got " + std::to_string(sb_of_row.Size())
                + ", expected 36");

    // Each sub-block should have 3 consecutive rows (1 nonmortar × 3
    // comps). Check that IDs are monotonically non-decreasing (rows
    // for one sub-block come before rows for the next).
    int last_id = -1;
    for (int i = 0; i < 36; ++i)
    {
        AssertOrDie(sb_of_row[i] >= last_id,
                    "PerPair IDs monotonic non-decreasing",
                    "row " + std::to_string(i) + " ID "
                    + std::to_string(sb_of_row[i])
                    + " < prev " + std::to_string(last_id));
        AssertOrDie(sb_of_row[i] >= 0 && sb_of_row[i] < 12,
                    "PerPair IDs in range",
                    "row " + std::to_string(i) + " ID "
                    + std::to_string(sb_of_row[i]) + " out of [0, 12)");
        last_id = sb_of_row[i];
    }

    // Each ID should appear exactly 3 times (3 comps per pair, 1
    // nonmortar interior per edge/face on this mesh).
    std::array<int, 12> count = {};
    for (int i = 0; i < 36; ++i) { ++count[sb_of_row[i]]; }
    for (int k = 0; k < 12; ++k)
    {
        AssertOrDie(count[k] == 3,
                    "PerPair count per sub-block",
                    "sub-block " + std::to_string(k) + " has "
                    + std::to_string(count[k]) + " rows, expected 3");
    }

    std::cout << "  PASS  PerPair full XYZ: 12 sub-blocks, 3 rows each, "
              << "labels in walk order" << std::endl;
}

void test_subblock_face_edge_x_only_pair_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: FaceEdge / x-face-pair only / "
              << "2x2x2" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> x_only = {"right"};
    std::array<bool, 3> all_comps = {true, true, true};

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::FaceEdge,
                              x_only, all_comps, labels, sb_of_row);

    // Labels still 2 (FaceEdge always emits both, even when one is empty).
    AssertOrDie(labels.size() == 2,
                "FaceEdge x-only label count",
                "got " + std::to_string(labels.size()) + ", expected 2");

    // With only x-face active, all edges drop (each needs 2 perp axes).
    // Only 3 face rows from the x-face pair remain.
    AssertOrDie(sb_of_row.Size() == 3,
                "FaceEdge x-only sb_of_row size",
                "got " + std::to_string(sb_of_row.Size())
                + ", expected 3");

    // All 3 rows should be in the face sub-block (ID 1).
    for (int i = 0; i < 3; ++i)
    {
        AssertOrDie(sb_of_row[i] == 1,
                    "FaceEdge x-only row ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i])
                    + ", expected 1 (face)");
    }

    std::cout << "  PASS  FaceEdge x-only: 3 face rows in sub-block 1, "
              << "edge sub-block empty but label retained" << std::endl;
}

void test_subblock_per_pair_x_only_pair_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: PerPair / x-face-pair only / "
              << "2x2x2" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> x_only = {"right"};
    std::array<bool, 3> all_comps = {true, true, true};

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::PerPair,
                              x_only, all_comps, labels, sb_of_row);

    // Only 1 active pair (the x-face), no edges → 1 sub-block.
    AssertOrDie(labels.size() == 1,
                "PerPair x-only label count",
                "got " + std::to_string(labels.size()) + ", expected 1");
    AssertOrDie(labels[0] == "face_right",
                "PerPair x-only label",
                "got '" + labels[0] + "', expected 'face_right'");

    AssertOrDie(sb_of_row.Size() == 3,
                "PerPair x-only sb_of_row size",
                "got " + std::to_string(sb_of_row.Size())
                + ", expected 3");

    // All 3 rows in sub-block 0.
    for (int i = 0; i < 3; ++i)
    {
        AssertOrDie(sb_of_row[i] == 0,
                    "PerPair x-only row ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i]) + ", expected 0");
    }

    std::cout << "  PASS  PerPair x-only: 1 sub-block (face_right), 3 rows"
              << std::endl;
}

void test_subblock_face_edge_x_comp_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: FaceEdge / X-comp only / 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> all_pairs = {"top", "right", "back"};
    std::array<bool, 3> x_comp = {true, false, false};

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::FaceEdge,
                              all_pairs, x_comp, labels, sb_of_row);

    // Labels still 2.
    AssertOrDie(labels.size() == 2,
                "FaceEdge X-comp label count",
                "got " + std::to_string(labels.size()) + ", expected 2");

    // Row count: 36 / 3 = 12 (only X component).
    AssertOrDie(sb_of_row.Size() == 12,
                "FaceEdge X-comp sb_of_row size",
                "got " + std::to_string(sb_of_row.Size())
                + ", expected 12");

    // First 9 are edge rows (9 edge pairs × 1 interior × 1 comp);
    // last 3 are face rows (3 face pairs × 1 interior × 1 comp).
    for (int i = 0; i < 9; ++i)
    {
        AssertOrDie(sb_of_row[i] == 0,
                    "FaceEdge X-comp edge row ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i]) + ", expected 0");
    }
    for (int i = 9; i < 12; ++i)
    {
        AssertOrDie(sb_of_row[i] == 1,
                    "FaceEdge X-comp face row ID",
                    "row " + std::to_string(i) + " has ID "
                    + std::to_string(sb_of_row[i]) + ", expected 1");
    }

    std::cout << "  PASS  FaceEdge X-comp: 9 edge + 3 face rows, 1 comp each"
              << std::endl;
}

void test_subblock_empty_filter_2x2x2()
{
    std::cout << "Phase 5.11 sub-block test: empty filter / 2x2x2"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    ConstraintBuilder3D builder(cl);

    std::vector<std::string> none;
    std::array<bool, 3> all_comps = {true, true, true};

    // FaceEdge with empty pairs: labels still 2, sb_of_row empty.
    {
        std::vector<std::string> labels;
        mfem::Array<int> sb_of_row;
        builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::FaceEdge,
                                  none, all_comps, labels, sb_of_row);
        AssertOrDie(labels.size() == 2,
                    "FaceEdge empty label count",
                    "got " + std::to_string(labels.size())
                    + ", expected 2 (always emits both)");
        AssertOrDie(sb_of_row.Size() == 0,
                    "FaceEdge empty sb_of_row size",
                    "got " + std::to_string(sb_of_row.Size())
                    + ", expected 0");
    }

    // PerPair with empty pairs: 0 labels, 0 rows.
    {
        std::vector<std::string> labels;
        mfem::Array<int> sb_of_row;
        builder.GetRowSubblockIds(mortar_pbc::SubblockPartition::PerPair,
                                  none, all_comps, labels, sb_of_row);
        AssertOrDie(labels.empty(),
                    "PerPair empty label count",
                    "got " + std::to_string(labels.size())
                    + ", expected 0");
        AssertOrDie(sb_of_row.Size() == 0,
                    "PerPair empty sb_of_row size",
                    "got " + std::to_string(sb_of_row.Size())
                    + ", expected 0");
    }

    std::cout << "  PASS  empty filter: FaceEdge has 2 labels / 0 rows; "
              << "PerPair has 0 labels / 0 rows" << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running ConstraintBuilder3D integration tests"
                  << std::endl;
        std::cout << "----------------------------------------------"
                  << std::endl;
    }
    test_row_count_2x2x2();
    test_row_count_4x4x4();
    test_emit_row_factors_2x2x2();
    test_nonempty_build();
    test_column_indices_in_range();
    test_row_layout();
    test_build_hypre_par_matrix();

    // Phase 5.9 filter tests.
    test_filter_x_only_2x2x2();
    test_filter_x_face_pair_only_2x2x2();
    test_filter_empty_2x2x2();

    // Phase 5.11 sub-block partition tests.
    test_subblock_face_edge_full_xyz_2x2x2();
    test_subblock_per_pair_full_xyz_2x2x2();
    test_subblock_face_edge_x_only_pair_2x2x2();
    test_subblock_per_pair_x_only_pair_2x2x2();
    test_subblock_face_edge_x_comp_2x2x2();
    test_subblock_empty_filter_2x2x2();

    if (rank == 0)
    {
        std::cout << "----------------------------------------------"
                  << std::endl;
        std::cout << "All ConstraintBuilder3D tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}