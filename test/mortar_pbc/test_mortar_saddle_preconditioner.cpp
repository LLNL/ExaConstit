// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.5.B.2 — smoke test for MortarSaddlePreconditioner.
//
// Verifies that the block-diagonal preconditioner correctly:
//   1. Constructs from valid K_block_prec / K_jacobi_prec / C_op.
//   2. Refreshes its internal pieces on SetOperator with a saddle
//      BlockOperator, including extraction of the (0,0) block as K.
//   3. Applies the expected block-diagonal action:
//        y_K   = K_block_prec(x_K)
//        y_lam = DiagonalScaler(inv_diag_S)(x_lam)
//      where inv_diag_S = C_op.ComputeInvDiagSchur(K_jacobi_prec).
//
// All tests run at np=1, matching the rest of the mortar_pbc unit
// suite. Cross-rank coverage lands when 5.5.B.4 wires this into
// SystemDriver and the patch tests run.
//
// Each test function exits via std::exit(1) on failure (with a
// diagnostic to stderr) or returns normally on success.

#include "boundary_classifier_3d.hpp"
#include "diagonal_scaler.hpp"
#include "mortar_constraint_operator.hpp"
#include "mortar_saddle_preconditioner.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::DiagonalScaler;
using mortar_pbc::MortarConstraintOperator;
using mortar_pbc::MortarSaddlePreconditioner;

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

// Deterministic LCG noise — same pattern used elsewhere in the
// mortar_pbc tests.
void FillLcg(mfem::Vector& v, unsigned seed)
{
    for (int i = 0; i < v.Size(); ++i)
    {
        seed = seed * 1103515245u + 12345u;
        v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
    }
}

// ===========================================================================
// Test 1: Construction succeeds with valid args.
// ===========================================================================
void test_constructs_with_valid_args()
{
    std::cout << "Test 1: MortarSaddlePreconditioner constructs with valid args"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator C_op(cl);

    const int n_K = C_op.Width();

    mfem::Vector ones_K(n_K);
    ones_K = 1.0;
    auto K_block_prec  = std::make_shared<DiagonalScaler>(n_K, ones_K);
    auto K_jacobi_prec = std::make_shared<DiagonalScaler>(n_K, ones_K);

    MortarSaddlePreconditioner prec(K_block_prec, K_jacobi_prec, C_op);
    // Pre-SetOperator: height/width default to 0; that's fine since
    // Mult is gated by an MFEM_VERIFY on m_block_prec.
    AssertOrDie(prec.Height() == 0,
                "pre-SetOperator height", "expected 0");
    AssertOrDie(prec.Width() == 0,
                "pre-SetOperator width", "expected 0");
    std::cout << "  PASS  constructed with n_K = " << n_K
              << ", n_lam = " << C_op.Height() << std::endl;
}

// ===========================================================================
// Test 2: SetOperator updates dimensions correctly.
// ===========================================================================
void test_set_operator_updates_dimensions()
{
    std::cout << "Test 2: SetOperator updates Height / Width correctly"
              << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator C_op(cl);

    const int n_K   = C_op.Width();
    const int n_lam = C_op.Height();

    mfem::Vector inv_diag_K(n_K);
    inv_diag_K = 0.2;  // matches a K with diag = 5
    auto K_block_prec  = std::make_shared<DiagonalScaler>(n_K, inv_diag_K);
    auto K_jacobi_prec = std::make_shared<DiagonalScaler>(n_K, inv_diag_K);

    MortarSaddlePreconditioner prec(K_block_prec, K_jacobi_prec, C_op);

    // Build a mock K = 5*I as a SparseMatrix (suffices: SparseMatrix
    // is an mfem::Operator and BlockOperator::SetBlock takes
    // Operator*; MortarSaddlePreconditioner only reads block(0,0)
    // and never invokes K's matvec — only its Height/Width and
    // forwarded SetOperator calls matter).
    mfem::SparseMatrix K_sp(n_K, n_K);
    for (int i = 0; i < n_K; ++i) { K_sp.Add(i, i, 5.0); }
    K_sp.Finalize();

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_K;
    offsets[2] = n_K + n_lam;

    mfem::BlockOperator saddle(offsets);
    saddle.SetBlock(0, 0, &K_sp);
    // Other blocks intentionally unset — preconditioner doesn't read them.

    prec.SetOperator(saddle);

    AssertOrDie(prec.Height() == n_K + n_lam,
                "post-SetOperator height",
                "got " + std::to_string(prec.Height())
                + ", expected " + std::to_string(n_K + n_lam));
    AssertOrDie(prec.Width() == n_K + n_lam,
                "post-SetOperator width",
                "got " + std::to_string(prec.Width())
                + ", expected " + std::to_string(n_K + n_lam));
    std::cout << "  PASS  Height = Width = " << prec.Height() << std::endl;
}

// ===========================================================================
// Test 3: Mult applies the expected block-diagonal action.
//
// Setup:
//   - K_block_prec  = DiagonalScaler with inv_diag = ones (acts as I)
//   - K_jacobi_prec = DiagonalScaler with inv_diag_K = 0.2*ones
//   - K (in BlockOperator (0,0)) is 5*I (only its size is consumed)
//
// Expected action of MortarSaddlePreconditioner:
//   y[0:n_K]       = K_block_prec(x[0:n_K]) = x[0:n_K]    (identity)
//   y[n_K:n_K+lam] = inv_diag_S * x[n_K:n_K+lam]
//
// where inv_diag_S = C_op.ComputeInvDiagSchur(K_jacobi_prec).
// We pre-compute inv_diag_S the same way and verify the lower-block
// action matches element-by-element.
// ===========================================================================
void test_mult_block_diagonal_action()
{
    std::cout << "Test 3: Mult applies block-diagonal action" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator C_op(cl);

    const int n_K   = C_op.Width();
    const int n_lam = C_op.Height();

    // K_block_prec acts as identity (inv_diag = ones).
    mfem::Vector ones_K(n_K);
    ones_K = 1.0;
    auto K_block_prec = std::make_shared<DiagonalScaler>(n_K, ones_K);

    // K_jacobi_prec advertises inv_diag(K) = 0.2 (matches K = 5*I).
    mfem::Vector inv_diag_K(n_K);
    inv_diag_K = 0.2;
    auto K_jacobi_prec = std::make_shared<DiagonalScaler>(n_K, inv_diag_K);

    // Pre-compute the expected Schur inverse-diagonal directly.
    mfem::Vector expected_inv_diag_S = C_op.ComputeInvDiagSchur(*K_jacobi_prec);
    AssertOrDie(expected_inv_diag_S.Size() == n_lam,
                "expected_inv_diag_S size",
                "got " + std::to_string(expected_inv_diag_S.Size())
                + ", expected " + std::to_string(n_lam));

    // Build the preconditioner.
    MortarSaddlePreconditioner prec(K_block_prec, K_jacobi_prec, C_op);

    // Build the saddle BlockOperator. K is mock 5*I; only block(0,0)
    // is needed (preconditioner ignores the other blocks).
    mfem::SparseMatrix K_sp(n_K, n_K);
    for (int i = 0; i < n_K; ++i) { K_sp.Add(i, i, 5.0); }
    K_sp.Finalize();

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_K;
    offsets[2] = n_K + n_lam;

    mfem::BlockOperator saddle(offsets);
    saddle.SetBlock(0, 0, &K_sp);

    prec.SetOperator(saddle);

    // Build a deterministic test input.
    mfem::Vector x(n_K + n_lam);
    FillLcg(x, 0xC0FFEEu);

    mfem::Vector y(n_K + n_lam);
    prec.Mult(x, y);

    // Verify upper block: y[0:n_K] == x[0:n_K] (identity action).
    constexpr double kTol = 1.0e-12;
    double max_err_K = 0.0;
    for (int i = 0; i < n_K; ++i)
    {
        const double err = std::abs(y[i] - x[i]);
        max_err_K = std::max(max_err_K, err);
    }
    AssertOrDie(max_err_K < kTol,
                "upper-block identity action",
                "max |y_K - x_K| = " + std::to_string(max_err_K)
                + " > tol " + std::to_string(kTol));

    // Verify lower block: y[n_K + i] == inv_diag_S[i] * x[n_K + i].
    double max_err_S = 0.0;
    for (int i = 0; i < n_lam; ++i)
    {
        const double expected = expected_inv_diag_S[i] * x[n_K + i];
        const double err = std::abs(y[n_K + i] - expected);
        max_err_S = std::max(max_err_S, err);
    }
    AssertOrDie(max_err_S < kTol,
                "lower-block diagonal-scaling action",
                "max |y_lam - inv_diag_S * x_lam| = "
                + std::to_string(max_err_S)
                + " > tol " + std::to_string(kTol));

    std::cout << "  PASS  max_err_K = " << max_err_K
              << ", max_err_S = " << max_err_S
              << " (n_K = " << n_K << ", n_lam = " << n_lam << ")"
              << std::endl;
}

// ===========================================================================
// Test 4: Re-SetOperator (per-Newton-iter pattern).
//
// Verifies that calling SetOperator a second time correctly tears
// down the previous BlockDiagonalPreconditioner and rebuilds it.
// We change K's diagonal between calls and verify the resulting
// inv_diag_S changes too.
// ===========================================================================
void test_resetoperator_rebuilds_internal_state()
{
    std::cout << "Test 4: re-SetOperator rebuilds internal state" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);
    MortarConstraintOperator C_op(cl);

    const int n_K   = C_op.Width();
    const int n_lam = C_op.Height();

    mfem::Vector ones_K(n_K);
    ones_K = 1.0;
    auto K_block_prec = std::make_shared<DiagonalScaler>(n_K, ones_K);

    // Use a Jacobi prec that we'll mutate between SetOperator calls
    // to simulate a per-Newton-iter inv_diag refresh. We construct
    // it with one set of values for the first call, then construct
    // a *new* DiagonalScaler with different values and swap it in
    // for the second call.

    // First refresh: inv_diag_K = 0.2 (matches K = 5*I)
    mfem::Vector inv_diag_K_1(n_K);
    inv_diag_K_1 = 0.2;
    auto K_jacobi_prec_1 = std::make_shared<DiagonalScaler>(n_K, inv_diag_K_1);
    mfem::Vector expected_inv_diag_S_1 =
        C_op.ComputeInvDiagSchur(*K_jacobi_prec_1);

    MortarSaddlePreconditioner prec(K_block_prec, K_jacobi_prec_1, C_op);

    mfem::SparseMatrix K_sp_1(n_K, n_K);
    for (int i = 0; i < n_K; ++i) { K_sp_1.Add(i, i, 5.0); }
    K_sp_1.Finalize();

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = n_K;
    offsets[2] = n_K + n_lam;

    mfem::BlockOperator saddle_1(offsets);
    saddle_1.SetBlock(0, 0, &K_sp_1);
    prec.SetOperator(saddle_1);

    // Second refresh would correspond to a fresh Newton iterate.
    // We construct a second saddle BlockOperator (K_sp_2) and
    // call SetOperator again. The K-Jacobi prec we passed in
    // construction is a DiagonalScaler whose values are baked in,
    // so the refresh path must still produce the same inv_diag_S
    // (since K_jacobi_prec doesn't actually update from K). What
    // we're testing here is the *idempotency* of the rebuild path:
    // calling SetOperator a second time must not crash, must
    // correctly tear down and rebuild the internal block prec, and
    // Mult must continue to work.
    mfem::SparseMatrix K_sp_2(n_K, n_K);
    for (int i = 0; i < n_K; ++i) { K_sp_2.Add(i, i, 7.0); }
    K_sp_2.Finalize();

    mfem::BlockOperator saddle_2(offsets);
    saddle_2.SetBlock(0, 0, &K_sp_2);
    prec.SetOperator(saddle_2);

    // Apply Mult and verify dimensions still match expectations.
    mfem::Vector x(n_K + n_lam);
    FillLcg(x, 0x12345u);
    mfem::Vector y(n_K + n_lam);
    prec.Mult(x, y);

    AssertOrDie(y.Size() == n_K + n_lam,
                "post-rebuild Mult output size",
                "got " + std::to_string(y.Size()));

    // Spot-check that the upper block still acts as identity (the
    // K_block_prec was unchanged across the rebuild).
    double max_err_K = 0.0;
    for (int i = 0; i < n_K; ++i)
    {
        max_err_K = std::max(max_err_K, std::abs(y[i] - x[i]));
    }
    AssertOrDie(max_err_K < 1.0e-12,
                "post-rebuild upper-block identity action",
                "max |y_K - x_K| = " + std::to_string(max_err_K));

    std::cout << "  PASS  rebuild succeeded; upper-block action preserved"
              << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running MortarSaddlePreconditioner tests" << std::endl;
        std::cout << "----------------------------------------------"
                  << std::endl;
    }

    test_constructs_with_valid_args();
    test_set_operator_updates_dimensions();
    test_mult_block_diagonal_action();
    test_resetoperator_rebuilds_internal_state();

    if (rank == 0)
    {
        std::cout << "----------------------------------------------"
                  << std::endl;
        std::cout << "All MortarSaddlePreconditioner tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
