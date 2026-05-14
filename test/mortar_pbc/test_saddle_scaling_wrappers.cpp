// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11 — D=I identity tests using the production linear-elastic
// scaffolding (parallel hex FES + AssembleLinearElasticKHypre +
// MortarConstraintOperator + MortarSaddlePointSystem).
//
// Purpose: bug-isolation for the observed "scaling-with-factors-all-1.0
// behaves differently from no-scaling" pathology. With D = I, every
// wrapper layer must produce element-wise identical output to the
// corresponding direct call. Anything that diverges identifies the
// layer responsible.
//
// Tests 1-2: operator-action identity at `Mult` / `MultTranspose`.
// Test 3: MINRES iteration-count + final-norm identity (the
//         diagnostic test for the production divergence).
// Test 4: Post-wrapper Norm identity (flag-state coherence on the
//         BlockVector::Update path).
//
// Same harness style as test_mortar_saddle_point_system.cpp and the
// other mortar_pbc unit tests: helpers in an anonymous namespace,
// `AssertOrDie` for assertions, std::exit(1) on failure.

#include "boundary_classifier_3d.hpp"
#include "elastic_3d_helpers.hpp"
#include "mortar_constraint_operator.hpp"
#include "mortar_saddle_point_system.hpp"
#include "saddle_residual_scaler.hpp"
#include "saddle_scaling_wrappers.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::MortarConstraintOperator;
using mortar_pbc::MortarSaddlePointSystem;
using mortar_pbc::SaddleResidualScaler;
using mortar_pbc::SaddleResidualScalerConfig;
using mortar_pbc::ScaledJacobianOperator;
using mortar_pbc::ScaledSaddleOperator;
using mortar_pbc::SubblockPartition;

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

// ---- helper: deterministic LCG fill ---------------------------------------
void FillLcg(mfem::Vector& v, unsigned seed)
{
    for (int i = 0; i < v.Size(); ++i)
    {
        seed = seed * 1103515245u + 12345u;
        v[i] = (static_cast<int>(seed) % 1000) / 1000.0 - 0.5;
    }
}

// ---- helper: build a scaler in identity (D = I) state ---------------------
//
// Uses `SetPartitionDirect` to install a partition without going
// through `Choose`, so the factors stay at the construction-time
// 1.0 values. IsEnabled() is true so the wrappers go through their
// full code paths (the whole point).
std::shared_ptr<SaddleResidualScaler>
BuildIdentityScalerFor(const MortarConstraintOperator& C_op)
{
    SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;
    cfg.floor = 1.0e-12;
    cfg.range_cap = 1.0e12;
    cfg.partition = SubblockPartition::FaceEdge;

    auto scaler = std::make_shared<SaddleResidualScaler>(cfg);

    const int n_lam = C_op.Height();
    std::vector<std::string> labels = {"edge", "face"};
    mfem::Array<int> of_row(n_lam);
    const int mid = n_lam / 2;
    for (int i = 0; i < n_lam; ++i)
    {
        of_row[i] = (i < mid ? 0 : 1);
    }
    scaler->SetPartitionDirect(labels, of_row);

    // Sanity — factors must be exactly 1.0 after SetPartitionDirect,
    // and IsEnabled() must remain true.
    AssertOrDie(scaler->GetDu() == 1.0,
                "identity scaler: d_u",
                "got " + std::to_string(scaler->GetDu())
                + ", expected exactly 1.0");
    AssertOrDie(scaler->GetDLambda().Size() == n_lam,
                "identity scaler: d_lambda size",
                "got " + std::to_string(scaler->GetDLambda().Size())
                + ", expected " + std::to_string(n_lam));
    {
        const double* dl = scaler->GetDLambda().HostRead();
        for (int i = 0; i < n_lam; ++i)
        {
            if (dl[i] != 1.0)
            {
                AssertOrDie(false, "identity scaler: d_lambda[i]",
                            "row " + std::to_string(i)
                            + " has value " + std::to_string(dl[i])
                            + ", expected exactly 1.0");
            }
        }
    }
    AssertOrDie(scaler->IsEnabled() == true,
                "identity scaler: IsEnabled",
                "got false");
    return scaler;
}

// ---- helper: saddle block offsets [0, n_u, n_u + n_lam] -------------------
mfem::Array<int> SaddleOffsetsOf(const MortarSaddlePointSystem& sys)
{
    mfem::Array<int> off(3);
    off[0] = 0;
    off[1] = sys.NumU();
    off[2] = sys.NumU() + sys.NumLambda();
    return off;
}

// ---- helper: element-wise max abs difference, MPI-reduced ----------------
double GlobalMaxAbsDiff(const mfem::Vector& a, const mfem::Vector& b,
                        MPI_Comm comm)
{
    AssertOrDie(a.Size() == b.Size(),
                "GlobalMaxAbsDiff: size mismatch",
                "a.Size = " + std::to_string(a.Size())
                + ", b.Size = " + std::to_string(b.Size()));
    const double* ad = a.HostRead();
    const double* bd = b.HostRead();
    double local_max = 0.0;
    for (int i = 0; i < a.Size(); ++i)
    {
        const double d = std::abs(ad[i] - bd[i]);
        if (d > local_max) { local_max = d; }
    }
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max;
}

// ===========================================================================
// Test 1 — ScaledSaddleOperator::Mult identity
//
// With D = I, the wrapper's Mult must produce element-wise identical
// output to the direct sys.Mult on every random input.
// ===========================================================================
void test_scaled_saddle_op_mult_identity()
{
    std::cout << "Test 1: ScaledSaddleOperator::Mult identity"
              << " (parallel LE)" << std::endl;

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

    // ScaledSaddleOperator takes a shared_ptr<Operator>. Use a
    // non-owning shared_ptr so the underlying sys is destroyed by
    // the unique_ptr lifetime (it's a stack-equivalent local here).
    auto sys = std::shared_ptr<MortarSaddlePointSystem>(
        new MortarSaddlePointSystem(k_residual, k_jacobian, C_op));

    const auto offsets = SaddleOffsetsOf(*sys);
    auto scaler = BuildIdentityScalerFor(C_op);

    ScaledSaddleOperator scaled_op(sys, scaler, offsets);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr int N_TRIALS = 5;
    double worst_diff = 0.0;
    for (int trial = 0; trial < N_TRIALS; ++trial)
    {
        mfem::Vector x_block(sys->Height());
        FillLcg(x_block, 1000 + 13 * trial);

        mfem::Vector r_direct(sys->Height());
        mfem::Vector r_wrapped(sys->Height());

        sys->Mult(x_block, r_direct);
        scaled_op.Mult(x_block, r_wrapped);

        const double diff = GlobalMaxAbsDiff(r_direct, r_wrapped,
                                              MPI_COMM_WORLD);
        if (diff > worst_diff) { worst_diff = diff; }
        if (rank == 0)
        {
            std::cout << "  trial " << trial
                      << ": max |r_direct - r_wrapped| = " << diff
                      << std::endl;
        }
    }

    AssertOrDie(worst_diff == 0.0,
                "ScaledSaddleOperator::Mult identity",
                "worst global diff = " + std::to_string(worst_diff)
                + " (must be exactly 0.0)");
    if (rank == 0) { std::cout << "  PASS" << std::endl; }
}

// ===========================================================================
// Test 2 — ScaledJacobianOperator::Mult / MultTranspose identity
//
// Wraps the real BlockOperator returned by sys.GetGradient(x0) and
// verifies Jacobian-vector products match the direct path. This is
// the highest-impact test because ScaledJacobianOperator is what
// MINRES iterates against.
// ===========================================================================
void test_scaled_jacobian_op_identity()
{
    std::cout << "Test 2: ScaledJacobianOperator::Mult / MultTranspose identity"
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
    const auto offsets = SaddleOffsetsOf(sys);
    auto scaler = BuildIdentityScalerFor(C_op);

    mfem::Vector x0(sys.Height());
    FillLcg(x0, 9876);
    mfem::Operator& inner_jac = sys.GetGradient(x0);

    ScaledJacobianOperator scaled_jac(inner_jac, scaler, offsets);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr int N_TRIALS = 5;
    double worst_mult_diff = 0.0;
    double worst_mt_diff   = 0.0;

    for (int trial = 0; trial < N_TRIALS; ++trial)
    {
        mfem::Vector v(sys.Height());
        FillLcg(v, 2000 + 17 * trial);

        // --- Mult ---
        {
            mfem::Vector Jv_direct(sys.Height());
            mfem::Vector Jv_wrapped(sys.Height());
            inner_jac.Mult(v, Jv_direct);
            scaled_jac.Mult(v, Jv_wrapped);
            const double diff = GlobalMaxAbsDiff(Jv_direct, Jv_wrapped,
                                                  MPI_COMM_WORLD);
            if (diff > worst_mult_diff) { worst_mult_diff = diff; }
            if (rank == 0)
            {
                std::cout << "  trial " << trial
                          << " Mult:          max diff = "
                          << diff << std::endl;
            }
        }

        // --- MultTranspose ---
        {
            mfem::Vector JTv_direct(sys.Height());
            mfem::Vector JTv_wrapped(sys.Height());
            inner_jac.MultTranspose(v, JTv_direct);
            scaled_jac.MultTranspose(v, JTv_wrapped);
            const double diff = GlobalMaxAbsDiff(JTv_direct, JTv_wrapped,
                                                  MPI_COMM_WORLD);
            if (diff > worst_mt_diff) { worst_mt_diff = diff; }
            if (rank == 0)
            {
                std::cout << "  trial " << trial
                          << " MultTranspose: max diff = "
                          << diff << std::endl;
            }
        }
    }

    AssertOrDie(worst_mult_diff == 0.0,
                "ScaledJacobianOperator::Mult identity",
                "worst global diff = " + std::to_string(worst_mult_diff));
    AssertOrDie(worst_mt_diff == 0.0,
                "ScaledJacobianOperator::MultTranspose identity",
                "worst global diff = " + std::to_string(worst_mt_diff));
    if (rank == 0) { std::cout << "  PASS" << std::endl; }
}

// ===========================================================================
// Test 3 — MINRES iteration-count and final-norm identity
//
// The most diagnostic test for the production pathology. Runs MINRES
// twice on the same RHS — once with the raw inner Jacobian, once
// with ScaledJacobianOperator(scaler=identity) wrapping it. Same
// tolerances, same max-iter, same zero initial guess. The two runs
// MUST converge in the same iter count, to the same final norm, and
// produce element-wise close solutions.
//
// If iter counts or final norms differ, the inner Krylov is
// converging differently against the wrapped operator — exactly the
// symptom in the production data (26 iters with D=I scaling, 2 iters
// without).
// ===========================================================================
void test_minres_trajectory_identity()
{
    std::cout << "Test 3: MINRES against wrapped(D=I) vs direct operator"
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
    const auto offsets = SaddleOffsetsOf(sys);
    auto scaler = BuildIdentityScalerFor(C_op);

    mfem::Vector x0(sys.Height());
    FillLcg(x0, 31415);
    mfem::Operator& inner_jac = sys.GetGradient(x0);
    ScaledJacobianOperator scaled_jac(inner_jac, scaler, offsets);

    mfem::Vector rhs(sys.Height());
    FillLcg(rhs, 27182);

    auto run_minres = [&](mfem::Operator& op, mfem::Vector& x_out,
                           int& n_iter_out, double& final_norm_out)
    {
        mfem::MINRESSolver minres(MPI_COMM_WORLD);
        minres.SetOperator(op);
        minres.SetMaxIter(200);
        minres.SetRelTol(1.0e-10);
        minres.SetAbsTol(1.0e-14);
        minres.SetPrintLevel(0);
        minres.iterative_mode = false;

        x_out.SetSize(op.Height());
        x_out = 0.0;
        minres.Mult(rhs, x_out);
        n_iter_out     = minres.GetNumIterations();
        final_norm_out = minres.GetFinalNorm();
    };

    mfem::Vector sol_direct, sol_wrapped;
    int n_iter_direct = 0, n_iter_wrapped = 0;
    double fn_direct = 0.0, fn_wrapped = 0.0;
    run_minres(inner_jac,  sol_direct,  n_iter_direct,  fn_direct);
    run_minres(scaled_jac, sol_wrapped, n_iter_wrapped, fn_wrapped);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "  direct  MINRES: iter=" << n_iter_direct
                  << "  final_norm=" << fn_direct << std::endl;
        std::cout << "  wrapped MINRES: iter=" << n_iter_wrapped
                  << "  final_norm=" << fn_wrapped << std::endl;
    }

    AssertOrDie(n_iter_direct == n_iter_wrapped,
                "MINRES iter count identity",
                "direct = " + std::to_string(n_iter_direct)
                + ", wrapped = " + std::to_string(n_iter_wrapped));

    AssertOrDie(std::abs(fn_direct - fn_wrapped) < 1.0e-14,
                "MINRES final norm identity",
                "direct = " + std::to_string(fn_direct)
                + ", wrapped = " + std::to_string(fn_wrapped));

    const double diff = GlobalMaxAbsDiff(sol_direct, sol_wrapped,
                                          MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "  max |sol_direct - sol_wrapped| = "
                  << diff << std::endl;
    }
    AssertOrDie(diff < 1.0e-12,
                "MINRES solution identity",
                "global diff = " + std::to_string(diff));
    if (rank == 0) { std::cout << "  PASS" << std::endl; }
}

// ===========================================================================
// Test 4 — Post-wrapper Norm identity (BV-view flag-state coherence)
//
// Verifies that after `scaled_op.Mult(x, r)` the parent Vector `r`
// reads back data and Norm bit-equal to the direct path. Targets
// the "sub-vector writes through BlockVector::Update don't refresh
// parent flag state" hypothesis.
// ===========================================================================
void test_post_wrapper_norm_identity()
{
    std::cout << "Test 4: post-wrapper Norm identity (BV-view flag state)"
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

    auto sys = std::shared_ptr<MortarSaddlePointSystem>(
        new MortarSaddlePointSystem(k_residual, k_jacobian, C_op));
    const auto offsets = SaddleOffsetsOf(*sys);
    auto scaler = BuildIdentityScalerFor(C_op);

    ScaledSaddleOperator scaled_op(sys, scaler, offsets);

    mfem::Vector x(sys->Height());
    FillLcg(x, 555);

    mfem::Vector r_direct(sys->Height());
    r_direct.UseDevice(true);
    sys->Mult(x, r_direct);
    mfem::Vector r_snapshot(r_direct);   // deep copy

    mfem::Vector r_via_wrapper(sys->Height());
    r_via_wrapper.UseDevice(true);
    scaled_op.Mult(x, r_via_wrapper);

    const double diff = GlobalMaxAbsDiff(r_snapshot, r_via_wrapper,
                                          MPI_COMM_WORLD);

    // Norm computed exactly the way Newton does it: parallel
    // Vector::operator* (which Allreduces internally).
    const double norm_direct  = std::sqrt(r_snapshot    * r_snapshot);
    const double norm_wrapped = std::sqrt(r_via_wrapper * r_via_wrapper);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "  max |r_direct - r_wrapped|   = " << diff << std::endl;
        std::cout << "  ||r_direct||                 = " << norm_direct
                  << std::endl;
        std::cout << "  ||r_wrapped||                = " << norm_wrapped
                  << std::endl;
    }

    AssertOrDie(diff == 0.0,
                "post-wrapper r data identity",
                "global diff = " + std::to_string(diff));
    AssertOrDie(norm_direct == norm_wrapped,
                "post-wrapper Norm identity",
                "direct = " + std::to_string(norm_direct)
                + ", wrapped = " + std::to_string(norm_wrapped));
    if (rank == 0) { std::cout << "  PASS" << std::endl; }
}

}   // anonymous namespace


// ===========================================================================
// main
// ===========================================================================
int main(int argc, char* argv[])
{
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();

    test_scaled_saddle_op_mult_identity();
    test_scaled_jacobian_op_identity();
    test_minres_trajectory_identity();
    test_post_wrapper_norm_identity();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "\nAll D=I identity tests passed." << std::endl;
    }
    return 0;
}