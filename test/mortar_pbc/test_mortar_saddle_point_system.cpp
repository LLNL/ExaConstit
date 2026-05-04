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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

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
