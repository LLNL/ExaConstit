// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.G — unit test for the TRDOG diagnostic sink + SNLS-style
// two-condition convergence test on ExaTrustRegionSolver.
//
// Strategy: same 2x2 linear residual operator as the 5.11.F NR/NRLS
// tests, but driven through ExaTrustRegionSolver with a recording
// sink. We set deltaInit large enough that the full Newton step fits
// inside the trust region on iter 1, so the dogleg picks the [NR]
// branch and TRDOG converges in one accepted step.
//
// Problem: r(x) = A x - b where
//   A = [[2, 0], [0, 3]],   b = [4, 6]
// Solution: x = [2, 2].
//
// With x_0 = [0, 0]:
//   r_0      = -b = [-4, -6],            ||r_0|| = sqrt(52) ≈ 7.211
//   c        = A^{-1} r_0 = [-2, -2]
//   nr_norm  = ||-c|| = ||(2, 2)|| = sqrt(8) ≈ 2.828
//   With deltaInit = 10.0: nr_norm < delta → full NR step taken.
//   delx     = nrStep = (2, 2)
//   x_1      = x_0 + delx = [2, 2]
//   r_1      = A x_1 - b = [0, 0],       ||r_1|| = 0
//
// Expected sink calls:
//   iter=0,  norm=sqrt(52),  norm0=sqrt(52),  converged_now=false
//   iter=1,  norm=0,         norm0=sqrt(52),  converged_now=true
//
// Note: TRDOG counts iterations starting at it=1 inside the loop
// (it++ at the top), while NR/NRLS use 0-based loop indices. The
// diagnostic sink fires with iter=0 for the pre-loop initial state
// and iter=1, 2, ... for the loop iterations, consistent with the
// NR/NRLS convention used in 5.11.F.

#include "solvers/trust_region_solver.hpp"
#include "solvers/mechanics_solver.hpp"   // NewtonIterDiagnostic

#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

//------------------------------------------------------------------------------
// Test harness
//------------------------------------------------------------------------------

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

void AssertNear(double a, double b, double tol,
                const std::string& test_name,
                const std::string& detail)
{
    if (std::abs(a - b) > tol)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail
                  << "  (got " << a << ", expected " << b
                  << ", diff " << std::abs(a - b) << ", tol "
                  << tol << ")" << std::endl;
        std::exit(1);
    }
}

//------------------------------------------------------------------------------
// Mock operator: r(x) = A x - b for fixed A, b
//------------------------------------------------------------------------------
//
// GetGradient returns A as a non-owning Operator& (DenseMatrix IS-A
// Operator). TRDOG calls Mult and MultTranspose on the gradient,
// both of which DenseMatrix supports.
class LinearMockOp : public mfem::Operator
{
public:
    LinearMockOp(int n, mfem::DenseMatrix A, mfem::Vector b)
        : mfem::Operator(n), m_A(std::move(A)), m_b(std::move(b))
    {
        MFEM_VERIFY(m_A.Height() == n && m_A.Width() == n,
                    "LinearMockOp: A must be n x n");
        MFEM_VERIFY(m_b.Size() == n, "LinearMockOp: b size mismatch");
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        m_A.Mult(x, y);     // y = A * x
        y -= m_b;           // y = A x - b
    }

    mfem::Operator& GetGradient(const mfem::Vector&) const override
    {
        return const_cast<mfem::DenseMatrix&>(m_A);
    }

private:
    mfem::DenseMatrix m_A;
    mfem::Vector      m_b;
};

//------------------------------------------------------------------------------
// Mock linear solver: x = J^{-1} b via DenseMatrix::Invert
//------------------------------------------------------------------------------
class DenseInverseSolver : public mfem::Solver
{
public:
    DenseInverseSolver() : mfem::Solver() {}

    void SetOperator(const mfem::Operator& op) override
    {
        const auto* dm = dynamic_cast<const mfem::DenseMatrix*>(&op);
        MFEM_VERIFY(dm != nullptr,
                    "DenseInverseSolver::SetOperator: expected "
                    "an mfem::DenseMatrix (the Jacobian).");
        m_J     = *dm;
        m_J_inv = m_J;
        m_J_inv.Invert();
        height = m_J.Height();
        width  = m_J.Width();
    }

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override
    {
        m_J_inv.Mult(b, x);   // x = J^{-1} b
    }

private:
    mutable mfem::DenseMatrix m_J;
    mutable mfem::DenseMatrix m_J_inv;
};

//------------------------------------------------------------------------------
// Helper — build the 2x2 mock problem.
//------------------------------------------------------------------------------
struct ProblemBundle
{
    std::shared_ptr<LinearMockOp>      op;
    std::shared_ptr<DenseInverseSolver> solver;
    double                              norm0_expected;
    double                              nr_norm_expected;
};

ProblemBundle BuildProblem()
{
    mfem::DenseMatrix A(2, 2);
    A(0, 0) = 2.0; A(0, 1) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 3.0;

    mfem::Vector b(2);
    b[0] = 4.0;
    b[1] = 6.0;

    ProblemBundle p;
    p.op               = std::make_shared<LinearMockOp>(2, A, b);
    p.solver           = std::make_shared<DenseInverseSolver>();
    p.norm0_expected   = std::sqrt(4.0 * 4.0 + 6.0 * 6.0);   // sqrt(52)
    p.nr_norm_expected = std::sqrt(2.0 * 2.0 + 2.0 * 2.0);   // sqrt(8)
    return p;
}

//==============================================================================
// Test 1: TRDOG converges + sink fires with the expected pattern
//==============================================================================
void test_trdog_sink_basic()
{
    std::cout << "Test 1: ExaTrustRegionSolver sink + convergence "
                 "(full NR step path)" << std::endl;

    auto p = BuildProblem();

    ExaTrustRegionSolver trdog(MPI_COMM_WORLD);
    trdog.iterative_mode = true;
    trdog.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    trdog.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    trdog.SetRelTol(1.0e-10);
    trdog.SetAbsTol(1.0e-12);
    trdog.SetMaxIter(10);
    trdog.SetPrintLevel(-1);

    // Trust radius generous enough that the full Newton step fits
    // (nr_norm = sqrt(8) ≈ 2.83 < deltaInit = 10).
    TrDeltaControl ctrl;
    ctrl.deltaInit = 10.0;
    ctrl.deltaMax  = 1.0e3;
    trdog.SetTrustRegionControl(ctrl);

    // Recording sink.
    std::vector<NewtonIterDiagnostic> recorded;
    trdog.SetDiagnosticSink([&recorded](const NewtonIterDiagnostic& d)
    {
        recorded.push_back(d);
    });

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    trdog.Mult(dummy_b, x);

    // --- Convergence + solution ---
    AssertOrDie(trdog.GetConverged() == 1,
                "TRDOG converged flag", "expected 1");
    AssertNear(x[0], 2.0, 1.0e-10, "x[0]", "expected 2");
    AssertNear(x[1], 2.0, 1.0e-10, "x[1]", "expected 2");

    // --- Sink call count: iter 0 (initial) + iter 1 (post-step) = 2 ---
    AssertOrDie(recorded.size() == 2,
                "TRDOG sink call count",
                "expected 2 calls (iter 0 + iter 1), got "
                + std::to_string(recorded.size()));

    // --- First call (pre-loop initial state) ---
    AssertOrDie(recorded[0].iter == 0,
                "TRDOG call[0] iter", "expected 0");
    AssertNear(recorded[0].norm, p.norm0_expected, 1.0e-10,
               "TRDOG call[0] norm", "expected sqrt(52)");
    AssertNear(recorded[0].norm0, p.norm0_expected, 1.0e-10,
               "TRDOG call[0] norm0", "expected sqrt(52)");
    AssertOrDie(!recorded[0].converged_now,
                "TRDOG call[0] converged_now",
                "expected false (sqrt(52) >> tol)");

    // --- Second call (post-step, converged) ---
    AssertOrDie(recorded[1].iter == 1,
                "TRDOG call[1] iter", "expected 1");
    AssertNear(recorded[1].norm, 0.0, 1.0e-10,
               "TRDOG call[1] norm", "expected ~0");
    AssertNear(recorded[1].norm0, p.norm0_expected, 1.0e-10,
               "TRDOG call[1] norm0",
               "norm0 must stay constant — must NOT shadow with res_0");
    AssertOrDie(recorded[1].converged_now,
                "TRDOG call[1] converged_now",
                "expected true (norm <= tol)");

    // --- norm_max consistency (SNLS-style two-condition derivation) ---
    const double norm_max_expected =
        std::max(1.0e-10 * p.norm0_expected, 1.0e-12);
    AssertNear(recorded[0].norm_max, norm_max_expected, 1.0e-15,
               "TRDOG call[0] norm_max",
               "must equal max(rel_tol*norm0, abs_tol)");
    AssertNear(recorded[1].norm_max, norm_max_expected, 1.0e-15,
               "TRDOG call[1] norm_max",
               "must not change between iters");

    std::cout << "  PASS  TRDOG: 2 sink calls, full NR step taken, "
                 "converged_now false→true" << std::endl;
}

//==============================================================================
// Test 2: TRDOG with no sink installed — no-op sink, default convergence
//==============================================================================
void test_trdog_sink_unset()
{
    std::cout << "Test 2: ExaTrustRegionSolver with no sink installed"
              << std::endl;

    auto p = BuildProblem();

    ExaTrustRegionSolver trdog(MPI_COMM_WORLD);
    trdog.iterative_mode = true;
    trdog.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    trdog.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    trdog.SetRelTol(1.0e-10);
    trdog.SetAbsTol(1.0e-12);
    trdog.SetMaxIter(10);
    trdog.SetPrintLevel(-1);

    TrDeltaControl ctrl;
    ctrl.deltaInit = 10.0;
    trdog.SetTrustRegionControl(ctrl);

    // Deliberately do NOT call SetDiagnosticSink — the inherited
    // m_diagnostic_sink stays a default-constructed (empty)
    // std::function, and the null-check in Mult should skip the
    // invocation entirely.

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    trdog.Mult(dummy_b, x);

    AssertOrDie(trdog.GetConverged() == 1,
                "TRDOG no-sink converged flag", "expected 1");
    AssertNear(x[0], 2.0, 1.0e-10, "no-sink x[0]", "expected 2");
    AssertNear(x[1], 2.0, 1.0e-10, "no-sink x[1]", "expected 2");

    std::cout << "  PASS  unset sink: TRDOG converges normally"
              << std::endl;
}

//==============================================================================
// Test 3: SNLS-style two-condition convergence — abs_tol path
//==============================================================================
//
// Set rel_tol so loose that it can never fire (1.0 — any residual
// is <= initial), but rely on abs_tol to drive convergence at the
// zero-residual fixed point. The two-condition refactor must
// continue to converge on the abs_tol branch alone.
void test_trdog_abs_tol_path()
{
    std::cout << "Test 3: TRDOG converges via abs_tol branch only"
              << std::endl;

    auto p = BuildProblem();

    ExaTrustRegionSolver trdog(MPI_COMM_WORLD);
    trdog.iterative_mode = true;
    trdog.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    trdog.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));

    // rel_tol = 1.0 → rel_tol * norm0 = sqrt(52), only iter 0 itself
    // would satisfy res <= rel_tol*norm0, which is always true. To
    // make conv_rel meaningless we'd need to handle iter 0 separately
    // (it already converges trivially since res == res_initial). Set
    // rel_tol = 0.0 instead to force conv_rel to require res == 0,
    // and abs_tol = 1e-10 to fire on the post-step residual.
    trdog.SetRelTol(0.0);
    trdog.SetAbsTol(1.0e-10);
    trdog.SetMaxIter(10);
    trdog.SetPrintLevel(-1);

    TrDeltaControl ctrl;
    ctrl.deltaInit = 10.0;
    trdog.SetTrustRegionControl(ctrl);

    std::vector<NewtonIterDiagnostic> recorded;
    trdog.SetDiagnosticSink([&recorded](const NewtonIterDiagnostic& d)
    {
        recorded.push_back(d);
    });

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    trdog.Mult(dummy_b, x);

    AssertOrDie(trdog.GetConverged() == 1,
                "TRDOG abs-tol-only converged flag", "expected 1");
    AssertOrDie(recorded.back().converged_now,
                "TRDOG abs-tol-only last converged_now",
                "expected true (abs_tol branch must fire)");

    // norm_max should be abs_tol since rel_tol*norm0 = 0.
    AssertNear(recorded.back().norm_max, 1.0e-10, 1.0e-15,
               "abs-tol-only norm_max",
               "expected abs_tol (rel branch contributes 0)");

    std::cout << "  PASS  abs_tol-only convergence works" << std::endl;
}

//==============================================================================
// Test 4: SNLS-style two-condition convergence — rel_tol path
//==============================================================================
//
// Inverse of test 3: set abs_tol tiny so it can't fire on a finite
// residual, and rely on rel_tol against the initial norm. For the
// 2x2 linear problem the post-step residual is FP-zero, so both
// conditions would fire, but the test is meaningful as a
// regression check that the two-condition refactor doesn't break
// either branch.
void test_trdog_rel_tol_path()
{
    std::cout << "Test 4: TRDOG converges via rel_tol branch"
              << std::endl;

    auto p = BuildProblem();

    ExaTrustRegionSolver trdog(MPI_COMM_WORLD);
    trdog.iterative_mode = true;
    trdog.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    trdog.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    trdog.SetRelTol(1.0e-10);
    trdog.SetAbsTol(1.0e-50);   // tiny — effectively disabled
    trdog.SetMaxIter(10);
    trdog.SetPrintLevel(-1);

    TrDeltaControl ctrl;
    ctrl.deltaInit = 10.0;
    trdog.SetTrustRegionControl(ctrl);

    std::vector<NewtonIterDiagnostic> recorded;
    trdog.SetDiagnosticSink([&recorded](const NewtonIterDiagnostic& d)
    {
        recorded.push_back(d);
    });

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    trdog.Mult(dummy_b, x);

    AssertOrDie(trdog.GetConverged() == 1,
                "TRDOG rel-tol-only converged flag", "expected 1");
    AssertOrDie(recorded.back().converged_now,
                "TRDOG rel-tol-only last converged_now", "expected true");

    // norm_max = max(rel_tol*norm0, abs_tol). abs_tol is so tiny it
    // can't dominate, so norm_max ≈ rel_tol * sqrt(52).
    const double expected = 1.0e-10 * p.norm0_expected;
    AssertNear(recorded.back().norm_max, expected, 1.0e-25,
               "rel-tol-only norm_max",
               "expected rel_tol*norm0 (abs branch is negligible)");

    std::cout << "  PASS  rel_tol-only convergence works" << std::endl;
}

}   // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running TRDOG diagnostic-sink unit tests"
                  << std::endl;
        std::cout << "----------------------------------------"
                  << std::endl;
    }

    test_trdog_sink_basic();
    test_trdog_sink_unset();
    test_trdog_abs_tol_path();
    test_trdog_rel_tol_path();

    if (rank == 0)
    {
        std::cout << "----------------------------------------"
                  << std::endl;
        std::cout << "All TRDOG diagnostic-sink tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
