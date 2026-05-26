// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.3 / Batch R — tests for MortarSaddlePointSystem.
//
// This file validates the saddle-point system adapter that composes
// a user-provided mechanical operator K (linear or nonlinear) with
// the EA constraint operator into a single mfem::Operator for use
// with mfem::Newton + mfem::BlockOperator-based Krylov methods.
//
// Coverage:
//   1. Construction succeeds; BlockOffsets / NumU / NumLambda are
//      correct.
//   2. Mult produces the correct block residual matching a
//      manually-assembled BlockOperator path.
//   3. GetGradient returns a BlockOperator whose action matches the
//      manually-assembled BlockOperator.
//   4. The KJacobianFn callback is invoked on each GetGradient call
//      (verified via a counter in the closure).
//   5. SetConstraintRHS / ClearConstraintRHS (Phase 5.0): when an
//      RHS is installed, Mult subtracts it from the constraint
//      block; ClearConstraintRHS restores the homogeneous default;
//      the constraint residual vanishes when u satisfies C * u = g.
//   6. Phase 6 shared-operator ownership: constructing from a
//      shared_ptr<MortarConstraintOperator> and then resetting the
//      operator filter is reflected by MortarSaddlePointSystem::
//      Refresh.
//
// All tests run at np=1, matching the rest of the unit suite. Cross-
// rank validation lands in Batch S via the patch-test integration.

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "elastic_3d_helpers.hpp"
#include "mortar_constraint_operator.hpp"
#include "mortar_saddle_point_system.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;
using mortar_pbc::MortarConstraintOperator;
using mortar_pbc::MortarSaddlePointSystem;

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
// Helper — fill a vector with deterministic LCG noise. Matches the
// pattern used in test_mortar_constraint_operator so the seeds /
// values produced are predictable.
// ===========================================================================
void FillLcg(mfem::Vector& v, unsigned seed)
{
    for (int i = 0; i < v.Size(); ++i)
    {
        seed = seed * 1103515245u + 12345u;
        v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
    }
}

// ===========================================================================
// Test 1: construction + block layout.
//
// MortarSaddlePointSystem takes the EA constraint operator + K's
// residual / Jacobian closures. Verify dimensions, offsets, and
// counts are consistent.
// ===========================================================================
void test_construction_and_layout()
{
    std::cout << "Test 1: construction + block layout" << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator C_op(cl);

    // Build K via the linear-elastic helper. Use this K in the
    // residual / Jacobian closures.
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    auto k_jacobian = [&K](const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);

    AssertOrDie(sys.NumU() == C_op.Width(),
                "NumU equals C_op.Width()",
                "NumU=" + std::to_string(sys.NumU())
                + ", C.Width()=" + std::to_string(C_op.Width()));
    AssertOrDie(sys.NumLambda() == C_op.Height(),
                "NumLambda equals C_op.Height()",
                "NumLambda=" + std::to_string(sys.NumLambda())
                + ", C.Height()=" + std::to_string(C_op.Height()));
    AssertOrDie(sys.Height() == sys.NumU() + sys.NumLambda(),
                "Height = NumU + NumLambda",
                "got Height=" + std::to_string(sys.Height()));
    AssertOrDie(sys.Width() == sys.Height(),
                "Width = Height (square saddle-point system)", "");

    const mfem::Array<int>& off = sys.BlockOffsets();
    AssertOrDie(off.Size() == 3, "BlockOffsets has 3 entries",
                "size=" + std::to_string(off.Size()));
    AssertOrDie(off[0] == 0,                "offsets[0] == 0", "");
    AssertOrDie(off[1] == sys.NumU(),       "offsets[1] == NumU", "");
    AssertOrDie(off[2] == sys.NumU() + sys.NumLambda(),
                "offsets[2] == NumU + NumLambda", "");

    std::cout << "  PASS  layout: NumU=" << sys.NumU()
              << ", NumLambda=" << sys.NumLambda()
              << ", Height=" << sys.Height() << std::endl;
}

// ===========================================================================
// Test 2: Mult produces the expected block residual.
//
// Ground truth: manually build the same residual using the K matvec
// and the EA C operator's Mult / MultTranspose, and compare.
//
//   Adapter Mult(x_block, r_block):
//     r_u   = K(u) + C^T lambda
//     r_lam = C u
//
// We tighten tolerance to 1e-12 — this is just an arithmetic
// rearrangement, no Krylov iteration involved.
// ===========================================================================
void test_mult_residual()
{
    std::cout << "Test 2: Mult residual matches manual block assembly"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator C_op(cl);
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    auto k_jacobian = [&K](const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);

    // Build a deterministic random block vector.
    mfem::Vector x_block(sys.Height());
    FillLcg(x_block, 24680);

    // Adapter path.
    mfem::Vector r_adapter(sys.Height());
    sys.Mult(x_block, r_adapter);

    // Manual path: extract u and lambda; compute r_u and r_lam
    // separately; concatenate.
    const int n_u   = sys.NumU();
    const int n_lam = sys.NumLambda();

    mfem::Vector u(n_u);
    mfem::Vector lambda(n_lam);
    for (int i = 0; i < n_u;   ++i) { u[i]      = x_block[i]; }
    for (int i = 0; i < n_lam; ++i) { lambda[i] = x_block[n_u + i]; }

    mfem::Vector r_u_manual(n_u);
    K->Mult(u, r_u_manual);  // r_u = K * u
    {
        mfem::Vector ct_lam(n_u);
        C_op.MultTranspose(lambda, ct_lam);
        r_u_manual += ct_lam;  // r_u += C^T * lambda
    }

    mfem::Vector r_lam_manual(n_lam);
    C_op.Mult(u, r_lam_manual);  // r_lam = C * u

    // Concatenate manual blocks and diff against adapter result.
    mfem::Vector r_manual(sys.Height());
    for (int i = 0; i < n_u;   ++i) { r_manual[i]       = r_u_manual[i]; }
    for (int i = 0; i < n_lam; ++i) { r_manual[n_u + i] = r_lam_manual[i]; }

    mfem::Vector diff(sys.Height());
    diff = r_adapter;
    diff -= r_manual;
    const double err  = diff.Norml2();
    const double norm = r_manual.Norml2();
    constexpr double kTol = 1.0e-12;
    const double tol_abs = kTol * std::max(1.0, norm);

    if (err > tol_abs)
    {
        std::cerr << "  FAIL  ||r_adapter - r_manual||_2 = " << err
                  << " > " << tol_abs
                  << " (||r_manual||_2 = " << norm << ")" << std::endl;
        std::exit(1);
    }
    std::cout << "  PASS  ||r_adapter - r_manual||_2 = " << err
              << " (rel " << err / std::max(1.0, norm) << ")" << std::endl;
}

// ===========================================================================
// Test 3: GetGradient returns a BlockOperator whose action matches
// a manually-assembled BlockOperator.
//
// Build the same block operator two ways:
//   (A) via sys.GetGradient(x) → BlockOperator
//   (B) manually:
//       block_offsets = [0, n_u, n_u + n_lam]
//       block(0,0) = K (HypreParMatrix*)
//       block(0,1) = TransposeOperator(C_op)
//       block(1,0) = C_op
//       (1,1) = zero
//
// Apply both to a random input vector; difference must be below
// FP-rearrangement tolerance.
// ===========================================================================
void test_get_gradient()
{
    std::cout << "Test 3: GetGradient action matches manual BlockOperator"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator C_op(cl);
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    auto k_jacobian = [&K](const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);

    // GetGradient takes a FULL block vector (size Height() = NumU +
    // NumLambda), not just the u-slice. The adapter extracts the
    // u-slice internally and forwards it to the K-Jacobian closure.
    // This matches mfem::Operator::GetGradient's API contract: same
    // input size as Mult.
    //
    // For linear K the closure ignores its input, so the value
    // doesn't matter — but the size has to be right.
    mfem::Vector x_block(sys.Height());
    mfem::Vector r_block(sys.Height());
    FillLcg(x_block, 22222);

    // Adapter path.
    mfem::Operator& J = sys.GetGradient(x_block);
    AssertOrDie(J.Height() == sys.Height(),
                "Gradient Height matches",
                "got " + std::to_string(J.Height()));
    AssertOrDie(J.Width()  == sys.Width(),
                "Gradient Width matches",
                "got " + std::to_string(J.Width()));

    mfem::Vector r_adapter(sys.Height());
    J.Mult(x_block, r_adapter);

    // Manual block-operator path.
    mfem::Array<int> off(3);
    off[0] = 0;
    off[1] = sys.NumU();
    off[2] = sys.NumU() + sys.NumLambda();

    mfem::TransposeOperator CT(&C_op);
    mfem::BlockOperator block_manual(off);
    block_manual.SetBlock(0, 0, K.get());
    block_manual.SetBlock(0, 1, &CT);
    block_manual.SetBlock(1, 0, &C_op);

    mfem::Vector r_manual(sys.Height());
    block_manual.Mult(x_block, r_manual);

    mfem::Vector diff(sys.Height());
    diff = r_adapter;
    diff -= r_manual;
    const double err  = diff.Norml2();
    const double norm = r_manual.Norml2();
    constexpr double kTol = 1.0e-12;
    const double tol_abs = kTol * std::max(1.0, norm);

    if (err > tol_abs)
    {
        std::cerr << "  FAIL  ||J_adapter x - J_manual x||_2 = " << err
                  << " > " << tol_abs
                  << " (||J_manual x||_2 = " << norm << ")" << std::endl;
        std::exit(1);
    }
    std::cout << "  PASS  ||J_adapter x - J_manual x||_2 = " << err
              << " (rel " << err / std::max(1.0, norm) << ")" << std::endl;
}

// ===========================================================================
// Test 4: KJacobianFn is invoked once per GetGradient call.
//
// This is a behavioral test, not a numerical one. The closure
// captures a mutable counter; we call GetGradient three times and
// verify the counter increments. This guards against a future
// optimization that might cache the Jacobian inappropriately
// (the production case has a per-Newton-iteration K that MUST be
// re-fetched each call, so caching would be a correctness bug).
// ===========================================================================
void test_jacobian_callback_invoked_per_call()
{
    std::cout << "Test 4: KJacobianFn is invoked on each GetGradient call"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator C_op(cl);
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    int call_count = 0;
    auto k_jacobian = [&K, &call_count]
        (const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        ++call_count;
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);

    // Block-sized input matching GetGradient's API contract (see
    // test 3). Value doesn't matter for linear K — only the size
    // gets checked.
    mfem::Vector x_block(sys.Height());
    x_block = 0.0;

    sys.GetGradient(x_block);
    sys.GetGradient(x_block);
    sys.GetGradient(x_block);

    AssertOrDie(call_count == 3,
                "KJacobianFn invoked 3 times for 3 GetGradient calls",
                "got call_count=" + std::to_string(call_count));
    std::cout << "  PASS  KJacobianFn was invoked exactly "
              << call_count << " times" << std::endl;
}

// ===========================================================================
// Test 5: SetConstraintRHS / ClearConstraintRHS (Phase 5.0).
//
// Validates the new constraint-RHS path that ExaConstit's
// MortarPbcManager (Phase 5.3) needs to support Method-D mortar
// PBC. Four sub-tests:
//
//   5.A — Default state has no RHS installed; HasConstraintRHS()
//         is false; Mult matches the homogeneous Phase 4.3
//         behavior verbatim (cross-checked against a recompute
//         with no RHS — should be bit-equal up to FP).
//
//   5.B — After SetConstraintRHS(g), the residual diff
//         (r_with_g - r_homogeneous) is exactly [0; -g]. The
//         u-block is unaffected (g doesn't enter r_u); the
//         lam-block shifts by -g.
//
//   5.C — Construct u_test arbitrarily, set g = C * u_test,
//         install g via SetConstraintRHS. Then Mult on the
//         block-vector [u_test; 0] returns r_lam = 0 to FP
//         precision. This is the Method-D "constraint satisfied"
//         demonstration: when u satisfies C * u = g, the
//         constraint residual vanishes.
//
//   5.D — ClearConstraintRHS restores HasConstraintRHS() to false
//         and Mult to the homogeneous behavior (bit-equal to the
//         5.A baseline).
//
// Tolerance is FP-rearrangement (1e-13) since these tests are
// arithmetic — no Krylov, no nontrivial summation reorderings.
// ===========================================================================
void test_constraint_rhs_path()
{
    std::cout << "Test 5: SetConstraintRHS / ClearConstraintRHS (Phase 5.0)"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    MortarConstraintOperator C_op(cl);
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    auto k_jacobian = [&K](const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);
    const int n_u   = sys.NumU();
    const int n_lam = sys.NumLambda();

    constexpr double kTol = 1.0e-13;

    // -----------------------------------------------------------------
    // 5.A — default: no RHS installed; baseline r_homogeneous.
    // -----------------------------------------------------------------
    AssertOrDie(!sys.HasConstraintRHS(),
                "5.A: default state has no constraint RHS installed",
                "HasConstraintRHS() returned true at construction");

    mfem::Vector x_block(sys.Height());
    FillLcg(x_block, 13579);

    mfem::Vector r_homogeneous(sys.Height());
    sys.Mult(x_block, r_homogeneous);

    // -----------------------------------------------------------------
    // 5.B — install non-zero g; verify r_block diff = [0; -g].
    // -----------------------------------------------------------------
    mfem::Vector g(n_lam);
    FillLcg(g, 24681);

    sys.SetConstraintRHS(g);
    AssertOrDie(sys.HasConstraintRHS(),
                "5.B: after SetConstraintRHS, HasConstraintRHS is true",
                "HasConstraintRHS() returned false post-install");

    mfem::Vector r_with_g(sys.Height());
    sys.Mult(x_block, r_with_g);

    mfem::Vector diff(sys.Height());
    diff = r_with_g;
    diff -= r_homogeneous;

    // u-side must be unchanged (g doesn't enter r_u).
    double u_diff_max = 0.0;
    for (int i = 0; i < n_u; ++i)
    {
        u_diff_max = std::max(u_diff_max, std::abs(diff[i]));
    }
    AssertOrDie(u_diff_max < kTol,
                "5.B: u-side residual unchanged by SetConstraintRHS",
                "max |diff_u| = " + std::to_string(u_diff_max));

    // lam-side diff must equal -g.
    double lam_diff_max = 0.0;
    for (int i = 0; i < n_lam; ++i)
    {
        const double expected = -g[i];
        lam_diff_max = std::max(lam_diff_max,
                                std::abs(diff[n_u + i] - expected));
    }
    AssertOrDie(lam_diff_max < kTol,
                "5.B: lam-side diff equals -g",
                "max |diff_lam - (-g)| = "
                + std::to_string(lam_diff_max));
    std::cout << "  PASS  5.B: diff = [0; -g] within tol "
              << "(|u|max=" << u_diff_max
              << ", |lam|max=" << lam_diff_max << ")" << std::endl;

    // -----------------------------------------------------------------
    // 5.C — Method-D demonstration: u satisfies C * u = g  =>  r_lam = 0.
    // -----------------------------------------------------------------
    mfem::Vector u_test(n_u);
    FillLcg(u_test, 99887);

    mfem::Vector g_satisfied(n_lam);
    C_op.Mult(u_test, g_satisfied);

    sys.SetConstraintRHS(g_satisfied);

    mfem::Vector x_satisfied(sys.Height());
    for (int i = 0; i < n_u;   ++i) { x_satisfied[i]       = u_test[i]; }
    for (int i = 0; i < n_lam; ++i) { x_satisfied[n_u + i] = 0.0; }

    mfem::Vector r_satisfied(sys.Height());
    sys.Mult(x_satisfied, r_satisfied);

    double r_lam_max = 0.0;
    for (int i = 0; i < n_lam; ++i)
    {
        r_lam_max = std::max(r_lam_max, std::abs(r_satisfied[n_u + i]));
    }
    AssertOrDie(r_lam_max < kTol,
                "5.C: constraint residual vanishes when C u = g",
                "max |r_lam| = " + std::to_string(r_lam_max));
    std::cout << "  PASS  5.C: r_lam = 0 when C u = g "
              << "(|r_lam|max=" << r_lam_max << ")" << std::endl;

    // -----------------------------------------------------------------
    // 5.D — ClearConstraintRHS restores homogeneous behavior.
    // -----------------------------------------------------------------
    sys.ClearConstraintRHS();
    AssertOrDie(!sys.HasConstraintRHS(),
                "5.D: after ClearConstraintRHS, HasConstraintRHS is false",
                "HasConstraintRHS() returned true post-clear");

    mfem::Vector r_after_clear(sys.Height());
    sys.Mult(x_block, r_after_clear);

    mfem::Vector diff_clear(sys.Height());
    diff_clear = r_after_clear;
    diff_clear -= r_homogeneous;
    const double clear_diff = diff_clear.Normlinf();
    AssertOrDie(clear_diff < kTol,
                "5.D: ClearConstraintRHS restores homogeneous Mult",
                "||r_after_clear - r_homogeneous||_inf = "
                + std::to_string(clear_diff));
    std::cout << "  PASS  5.D: ClearConstraintRHS restores default "
              << "(||diff||_inf=" << clear_diff << ")" << std::endl;
}

// ===========================================================================
// Test 6: Phase 6 shared-operator ownership + Refresh.
//
// The manager now owns MortarConstraintOperator behind shared_ptr and
// passes that handle to MortarSaddlePointSystem. This test verifies
// the saddle system observes a filter-induced C_op.Height() change
// through the shared handle after Refresh, without reconstructing the
// system.
// ===========================================================================
void test_shared_operator_refresh_after_reset()
{
    std::cout << "Test 6: shared C_op handle refreshes after Reset"
              << std::endl;

    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 4);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    auto C_op = std::make_shared<MortarConstraintOperator>(cl);
    std::unique_ptr<mfem::HypreParMatrix> K(
        mortar_pbc::AssembleLinearElasticKHypre(*b.pmesh, *b.fes,
                                                /*E=*/1.0, /*nu=*/0.3));

    auto k_residual = [&K](const mfem::Vector& u, mfem::Vector& r)
    {
        K->Mult(u, r);
    };
    auto k_jacobian = [&K](const mfem::Vector& /*u*/) -> mfem::Operator*
    {
        return K.get();
    };

    MortarSaddlePointSystem sys(k_residual, k_jacobian, C_op);
    const int initial_lambda = sys.NumLambda();

    C_op->Reset(std::vector<std::string>{"right"},
                std::array<bool, 3>{{true, false, false}});
    sys.Refresh();

    AssertOrDie(sys.NumU() == C_op->Width(),
                "shared refresh: NumU follows C width",
                "NumU=" + std::to_string(sys.NumU())
                + ", C.Width()=" + std::to_string(C_op->Width()));
    AssertOrDie(sys.NumLambda() == C_op->Height(),
                "shared refresh: NumLambda follows C height",
                "NumLambda=" + std::to_string(sys.NumLambda())
                + ", C.Height()=" + std::to_string(C_op->Height()));
    AssertOrDie(sys.NumLambda() < initial_lambda,
                "shared refresh: filtered C height decreased",
                "initial=" + std::to_string(initial_lambda)
                + ", filtered=" + std::to_string(sys.NumLambda()));
    AssertOrDie(sys.Height() == sys.NumU() + sys.NumLambda(),
                "shared refresh: Height is updated",
                "Height=" + std::to_string(sys.Height()));

    mfem::Vector x(sys.Height());
    x = 0.0;
    mfem::Operator& J = sys.GetGradient(x);
    AssertOrDie(J.Height() == sys.Height() && J.Width() == sys.Width(),
                "shared refresh: gradient dimensions follow refreshed size",
                "J is " + std::to_string(J.Height()) + " x "
                + std::to_string(J.Width()));

    std::cout << "  PASS  lambda rows " << initial_lambda << " -> "
              << sys.NumLambda() << " through shared C_op Refresh"
              << std::endl;
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
        std::cout << "test_mortar_saddle_point_system (Phase 4.3/R)"
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }

    test_construction_and_layout();
    test_mult_residual();
    test_get_gradient();
    test_jacobian_callback_invoked_per_call();
    test_constraint_rhs_path();
    test_shared_operator_refresh_after_reset();

    if (rank == 0)
    {
        std::cout << "==============================================="
                  << std::endl;
        std::cout << "All MortarSaddlePointSystem tests passed."
                  << std::endl;
        std::cout << "==============================================="
                  << std::endl;
    }
    MPI_Finalize();
    return 0;
}
