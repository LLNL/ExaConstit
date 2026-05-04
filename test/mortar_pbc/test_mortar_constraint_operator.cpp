// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.3 / Batches O, P, Q — A/B validation harness for
// MortarConstraintOperator vs the HypreParMatrix path.
//
// Coverage progression:
//   - Batch O: construction + dimension match.
//   - Batch P: single-size (4³) Mult / MultTranspose match.
//   - Batch Q (this batch): multiple mesh sizes (4³, 6³, 8³),
//                            tightened tolerance, a negative test
//                            that confirms the harness catches a
//                            deliberately-perturbed result.
//
// Scope decision:
// All tests here run at np=1, matching the rest of the unit-test
// suite. Cross-rank A/B validation (the Alltoallv import/export
// path actually exchanging data) is exercised by the end-to-end
// patch tests at np=4 / np=7 with the --constraint-storage=ea
// flag (Phase 4.3 / Batch S). This file's purpose is the matvec-
// level contract: at fixed np, EA and HypreParMatrix paths
// produce identical y to FP-rearrangement precision.
//
// Tolerance contract (per §P4.4.6.3): the difference must be
// below 1e-12 * (||C||_F * ||u||_2) — for the small meshes here
//
// Phase 4.3.B / Batch X — GPU port note:
// Although this file runs serially on host, after the GPU port
// the matvec hot path goes through mfem::forall with full
// Read/Write memory-manager annotations. To exercise the
// memory-manager invariants in CI, build MFEM with DEVICE_DEBUG
// enabled and re-run this test — any host-stale or device-stale
// access pattern will trigger an MFEM_ASSERT failure rather than
// silently corrupting. (DEVICE_DEBUG works on host-only builds
// too; it's a memory-manager validation mode, not a device
// requirement.)
// (||C||_F ~ O(1), ||u||_2 ~ O(1)) this is 1e-12 absolute. Tests
// use 1e-12 with a max(1, ||y_hp||_2) safety floor.
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success.

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "mortar_constraint_operator.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;
using mortar_pbc::MortarConstraintOperator;

namespace {

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

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
// Test 1: Operator constructs successfully on the smallest non-trivial mesh.
// ===========================================================================
void test_constructs_on_2x2x2()
{
    std::cout << "Test 1: MortarConstraintOperator constructs on 2x2x2 hex"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    AssertOrDie(op.Height() > 0,
                "MortarConstraintOperator::Height()",
                "got 0, expected positive");
    AssertOrDie(op.Width() > 0,
                "MortarConstraintOperator::Width()",
                "got 0, expected positive");
    std::cout << "  PASS  Height=" << op.Height()
              << ", Width=" << op.Width() << std::endl;
}

// ===========================================================================
// Test 2: Height / Width match the HypreParMatrix path on np=1.
//
// At np=1 every constraint row is local (FES-aligned and fair-split
// degenerate to the same partition), so the HypreParMatrix's
// (Height, Width) and the EA operator's (Height, Width) must be
// identical. At np>1 they would also be identical because both paths
// use the same FES-aligned row partition (Batch N) and FES TDOF
// column partition (§P4.8.9), but this test runs at np=1 to keep
// it within the unit-test harness.
// ===========================================================================
void test_dimensions_match_hypre_path()
{
    std::cout << "Test 2: dimensions match HypreParMatrix path" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);

    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    // At np=1 the HypreParMatrix's local Height equals its global
    // Height; ditto for Width. We compare the EA operator's local
    // dimensions to those.
    AssertOrDie(op.Height() == H->Height(),
                "Height matches HypreParMatrix",
                "EA=" + std::to_string(op.Height())
                + ", Hypre=" + std::to_string(H->Height()));
    AssertOrDie(op.Width() == H->Width(),
                "Width matches HypreParMatrix",
                "EA=" + std::to_string(op.Width())
                + ", Hypre=" + std::to_string(H->Width()));
    std::cout << "  PASS  EA(Height,Width) = ("
              << op.Height() << ", " << op.Width()
              << ") matches HypreParMatrix" << std::endl;
}

// ===========================================================================
// A/B harness helper: at a given mesh size, builds both EA operator and
// HypreParMatrix, applies both to the same random u (and lambda for
// transpose), verifies the difference is below tolerance.
//
// Returns the absolute and relative error for diagnostic logging by
// the caller. Aborts on failure.
//
// `tag` shows up in PASS/FAIL diagnostics so multi-size runs can
// identify which size failed.
// ===========================================================================
struct AbDiff
{
    double mult_err_abs;
    double mult_norm;
    double mult_T_err_abs;
    double mult_T_norm;
};

AbDiff RunAbHarness(int n_per_side, double tol, const std::string& tag)
{
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, n_per_side);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    AssertOrDie(op.Width()  == H->Width(),
                tag + ": Width matches",
                "EA=" + std::to_string(op.Width())
                + ", H=" + std::to_string(H->Width()));
    AssertOrDie(op.Height() == H->Height(),
                tag + ": Height matches",
                "EA=" + std::to_string(op.Height())
                + ", H=" + std::to_string(H->Height()));

    // Deterministic LCG-generated u and lambda. Different seeds for
    // the two vectors so MultTranspose isn't accidentally exercising
    // the same data layout as Mult.
    auto fill_lcg = [](mfem::Vector& v, unsigned seed)
    {
        for (int i = 0; i < v.Size(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
        }
    };

    mfem::Vector u(op.Width());
    mfem::Vector lambda(op.Height());
    fill_lcg(u, 12345);
    fill_lcg(lambda, 67890);

    AbDiff result;

    // ----- Mult -----
    {
        mfem::Vector y_ea(op.Height());
        mfem::Vector y_hp(op.Height());
        op.Mult(u, y_ea);
        H->Mult(u, y_hp);

        mfem::Vector diff(op.Height());
        diff = y_ea;
        diff -= y_hp;
        result.mult_err_abs = diff.Norml2();
        result.mult_norm    = y_hp.Norml2();

        const double tol_abs = tol * std::max(1.0, result.mult_norm);
        if (result.mult_err_abs > tol_abs)
        {
            std::cerr << "  FAIL  " << tag
                      << ": ||C_ea u - C_hp u||_2 = "
                      << result.mult_err_abs
                      << " > tol*max(1, ||y_hp||) = " << tol_abs
                      << " (||y_hp||_2 = " << result.mult_norm << ")"
                      << std::endl;
            std::exit(1);
        }
    }

    // ----- MultTranspose -----
    {
        mfem::Vector y_ea(op.Width());
        mfem::Vector y_hp(op.Width());
        op.MultTranspose(lambda, y_ea);
        H->MultTranspose(lambda, y_hp);

        mfem::Vector diff(op.Width());
        diff = y_ea;
        diff -= y_hp;
        result.mult_T_err_abs = diff.Norml2();
        result.mult_T_norm    = y_hp.Norml2();

        const double tol_abs = tol * std::max(1.0, result.mult_T_norm);
        if (result.mult_T_err_abs > tol_abs)
        {
            std::cerr << "  FAIL  " << tag
                      << ": ||C^T_ea lambda - C^T_hp lambda||_2 = "
                      << result.mult_T_err_abs
                      << " > tol*max(1, ||y_hp||) = " << tol_abs
                      << " (||y_hp||_2 = " << result.mult_T_norm << ")"
                      << std::endl;
            std::exit(1);
        }
    }

    return result;
}

// ===========================================================================
// Test 3: A/B at multiple mesh sizes. Catches size-dependent bugs that
// might pass at one size but fail at another (e.g. an off-by-one in
// the per-pair scatter that only triggers when n_n > 1, or sparsity-
// pattern bugs that only show up when A_m has multiple nnz per row).
// ===========================================================================
void test_ab_multi_size()
{
    std::cout << "Test 3: A/B at multiple mesh sizes" << std::endl;
    // Phase 4.3 / Batch Q tolerance contract: 1e-12 abs (per
    // §P4.4.6.3). Headroom: typical FP-rearrangement error at these
    // sizes is ~1e-14, so 1e-12 catches real bugs while leaving 2
    // orders of magnitude for FP drift.
    constexpr double kTol = 1.0e-12;

    for (int n : {2, 4, 6, 8})
    {
        const std::string tag = "n=" + std::to_string(n);
        AbDiff d = RunAbHarness(n, kTol, tag);
        std::cout << "  PASS  " << tag
                  << ":  Mult err=" << d.mult_err_abs
                  << " (rel " << d.mult_err_abs / std::max(1.0, d.mult_norm)
                  << "),  MultT err=" << d.mult_T_err_abs
                  << " (rel " << d.mult_T_err_abs
                                / std::max(1.0, d.mult_T_norm)
                  << ")" << std::endl;
    }
}

// ===========================================================================
// Test 4: zero-input invariant. Both Mult(0, _) and MultTranspose(0, _)
// must produce zero output (Cu = 0 when u = 0; same for transpose).
// This is a basic linearity sanity check; if either path's
// initialization or accumulation is buggy it can leave residual
// noise in the output even on zero input.
// ===========================================================================
void test_zero_input()
{
    std::cout << "Test 4: zero-input produces zero output" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator op(cl);

    mfem::Vector u(op.Width());
    mfem::Vector lambda(op.Height());
    u = 0.0;
    lambda = 0.0;

    mfem::Vector y(op.Height());
    op.Mult(u, y);
    AssertOrDie(y.Norml2() < 1.0e-14,
                "Mult(0)",
                "||y||_2 = " + std::to_string(y.Norml2()));

    mfem::Vector z(op.Width());
    op.MultTranspose(lambda, z);
    AssertOrDie(z.Norml2() < 1.0e-14,
                "MultTranspose(0)",
                "||z||_2 = " + std::to_string(z.Norml2()));

    std::cout << "  PASS  Mult(0)=0 and MultTranspose(0)=0" << std::endl;
}

// ===========================================================================
// Test 5: harness self-check (negative test). Build the EA output,
// perturb one entry, and verify our A/B-comparison logic catches the
// difference. This guards against the harness being too lenient — if
// future tightening of tol breaks this check, the harness will alert
// us before silently accepting a real EA bug.
// ===========================================================================
void test_negative_harness_self_check()
{
    std::cout << "Test 5: harness catches a deliberately perturbed result"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    mfem::Vector u(op.Width());
    {
        unsigned seed = 12345;
        for (int i = 0; i < op.Width(); ++i)
        {
            seed = seed * 1103515245u + 12345u;
            u[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
        }
    }

    mfem::Vector y_ea(op.Height());
    mfem::Vector y_hp(op.Height());
    op.Mult(u, y_ea);
    H->Mult(u, y_hp);

    // Inject a 1e-3 perturbation — well above any tolerance we'd ever
    // realistically use. The harness comparison MUST flag this.
    constexpr double kPerturbation = 1.0e-3;
    if (y_ea.Size() > 0) { y_ea[0] += kPerturbation; }

    mfem::Vector diff(op.Height());
    diff = y_ea;
    diff -= y_hp;
    const double err  = diff.Norml2();
    const double norm = y_hp.Norml2();
    constexpr double kHarnessTol = 1.0e-12;
    const double tol_abs = kHarnessTol * std::max(1.0, norm);

    AssertOrDie(err > tol_abs,
                "harness catches perturbation",
                "perturbation " + std::to_string(kPerturbation)
                + " yielded ||diff||_2 = " + std::to_string(err)
                + " <= tol_abs " + std::to_string(tol_abs)
                + " (harness is too loose to catch real bugs)");
    std::cout << "  PASS  harness flags " << kPerturbation
              << "-magnitude perturbation: ||diff||_2 = " << err
              << " > " << tol_abs << std::endl;
}

// ===========================================================================
// Test 6 (Phase 4.3 / Batch R): ComputeInvDiagSchur agrees with the
// HypreParMatrix-path formula.
//
// The formula:
//   schur_diag[i] = sum_j C[i,j]^2 * inv_diag_K[j]
//
// We pick inv_diag_K = ones(global_size) so the formula simplifies to
//   schur_diag[i] = sum_j C[i,j]^2 = ||C[i,:]||_2^2.
//
// Then both:
//   - op.ComputeInvDiagSchur(ones).inv -> schur_diag (after element
//                                                     -wise reciprocal)
//   - HypreParMatrix C: walk CSR, sum squares per row -> schur_diag
//
// must match to FP precision. We compare the un-inverted Schur diagonals
// (not the inverses) to avoid 1/0 issues on Dirichlet-zeroed rows; the
// reciprocal logic is the same in both paths so we don't need to test
// it separately.
// ===========================================================================
void test_compute_inv_diag_schur_matches_hypre()
{
    std::cout << "Test 6: ComputeInvDiagSchur agrees with HypreParMatrix path"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator op(cl);
    ConstraintBuilder3D builder(cl);
    std::unique_ptr<mfem::HypreParMatrix> H(builder.BuildHypreParMatrix());

    // inv_diag_K = ones(local_size). At np=1 local_size = global_size.
    mfem::Vector inv_diag_K(op.Width());
    inv_diag_K = 1.0;

    // EA path: returns inv_schur. Invert back to schur for comparison.
    mfem::Vector inv_schur_ea = op.ComputeInvDiagSchur(inv_diag_K);
    mfem::Vector schur_ea(op.Height());
    for (int i = 0; i < op.Height(); ++i)
    {
        const double v = inv_schur_ea[i];
        schur_ea[i] = (std::abs(v) > 1.0e-300) ? (1.0 / v) : 0.0;
    }

    // HypreParMatrix path: sum-of-squares per row from CSR. At np=1
    // C's CSR is fully in the diag block; offd is empty.
    mfem::Vector schur_hp(op.Height());
    schur_hp = 0.0;
    {
        mfem::SparseMatrix C_diag;
        H->GetDiag(C_diag);
        const int* I    = C_diag.GetI();
        const double* A = C_diag.GetData();
        for (int i = 0; i < op.Height(); ++i)
        {
            double s = 0.0;
            for (int k = I[i]; k < I[i + 1]; ++k)
            {
                s += A[k] * A[k];
            }
            schur_hp[i] = s;
        }
    }

    mfem::Vector diff(op.Height());
    diff = schur_ea;
    diff -= schur_hp;
    const double err  = diff.Norml2();
    const double norm = schur_hp.Norml2();
    constexpr double kTol = 1.0e-12;
    const double tol_abs = kTol * std::max(1.0, norm);

    if (err > tol_abs)
    {
        std::cerr << "  FAIL  ||schur_ea - schur_hp||_2 = " << err
                  << " > " << tol_abs
                  << " (||schur_hp||_2 = " << norm << ")" << std::endl;
        // Diagnostic: print a few entries.
        std::cerr << "  First 5 entries (ea, hp, diff):" << std::endl;
        for (int i = 0; i < std::min(5, op.Height()); ++i)
        {
            std::cerr << "    [" << i << "] " << schur_ea[i] << ", "
                      << schur_hp[i] << ", "
                      << (schur_ea[i] - schur_hp[i]) << std::endl;
        }
        std::exit(1);
    }
    std::cout << "  PASS  ||schur_ea - schur_hp||_2 = " << err
              << " (rel " << err / std::max(1.0, norm) << ")" << std::endl;
}

}  // anonymous namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "==============================================="
                  << std::endl;
        std::cout << "test_mortar_constraint_operator (Phase 4.3/R)"
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }

    test_constructs_on_2x2x2();
    test_dimensions_match_hypre_path();
    test_ab_multi_size();
    test_zero_input();
    test_negative_harness_self_check();
    test_compute_inv_diag_schur_matches_hypre();

    if (rank == 0)
    {
        std::cout << "==============================================="
                  << std::endl;
        std::cout << "All MortarConstraintOperator tests passed."
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }
    MPI_Finalize();
    return 0;
}
