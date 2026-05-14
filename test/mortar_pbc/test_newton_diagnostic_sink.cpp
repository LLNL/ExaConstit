// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.F — unit test for the NewtonDiagnosticSink hook on
// ExaNewtonSolver and ExaNewtonLSSolver.
//
// Strategy: construct a tiny 2x2 linear residual operator and a
// direct dense-inverse "solver" so the Newton iteration's behavior
// is fully predictable. Wire a recording sink that captures every
// per-iter callback into a std::vector. Assert that the recorded
// callbacks match what we know the Newton loop should produce.
//
// Problem: r(x) = A x - b where
//   A = [[2, 0], [0, 3]],   b = [4, 6]
// Solution: x = [2, 2].
//
// With x_0 = [0, 0], one Newton step suffices:
//   r_0    = -b = [-4, -6],            norm_0 = sqrt(52) ≈ 7.211
//   c      = A^{-1} r_0 = [-2, -2]
//   x_1    = x_0 - c = [2, 2]
//   r_1    = A x_1 - b = [0, 0],       norm_1 = 0
//
// Expected sink calls:
//   iter=0,  norm=sqrt(52),  norm0=sqrt(52),  converged_now=false
//   iter=1,  norm=0,         norm0=sqrt(52),  converged_now=true

#include "solvers/mechanics_solver.hpp"

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
// Operator). The Newton solver feeds this into the linear-solver mock
// below via SetOperator.
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
//
// SetOperator copies the incoming DenseMatrix (the Jacobian from
// LinearMockOp::GetGradient), inverts it once, and reuses the inverse
// for subsequent Mult calls. Adequate for tiny 2x2 linear systems
// where the Jacobian is constant.
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
// Helper — build the 2x2 mock for both tests.
//------------------------------------------------------------------------------
struct ProblemBundle
{
    std::shared_ptr<LinearMockOp>      op;
    std::shared_ptr<DenseInverseSolver> solver;
    double                              norm0_expected;
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
    p.op             = std::make_shared<LinearMockOp>(2, A, b);
    p.solver         = std::make_shared<DenseInverseSolver>();
    p.norm0_expected = std::sqrt(4.0 * 4.0 + 6.0 * 6.0);   // sqrt(52)
    return p;
}

//==============================================================================
// Test 1: ExaNewtonSolver — sink fires correctly, solver converges
//==============================================================================
void test_nr_sink_basic()
{
    std::cout << "Test 1: ExaNewtonSolver sink + convergence" << std::endl;

    auto p = BuildProblem();

    ExaNewtonSolver newton(MPI_COMM_WORLD);
    newton.iterative_mode = true;
    newton.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    newton.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    newton.SetRelTol(1.0e-10);
    newton.SetAbsTol(1.0e-12);
    newton.SetMaxIter(10);
    newton.SetPrintLevel(-1);   // silent on stdout

    // Recording sink.
    std::vector<NewtonIterDiagnostic> recorded;
    newton.SetDiagnosticSink([&recorded](const NewtonIterDiagnostic& d)
    {
        recorded.push_back(d);
    });

    // Run.
    mfem::Vector x(2);
    x[0] = 0.0; x[1] = 0.0;

    mfem::Vector dummy_b;   // empty → no rhs-subtract path in Newton::Mult
    newton.Mult(dummy_b, x);

    // --- Convergence + solution ---
    AssertOrDie(newton.GetConverged() == 1,
                "NR converged flag", "expected 1");
    AssertNear(x[0], 2.0, 1.0e-10, "x[0]", "expected 2");
    AssertNear(x[1], 2.0, 1.0e-10, "x[1]", "expected 2");

    // --- Sink call count ---
    // Iter 0: prints initial residual, fails convergence, takes Newton step.
    // Iter 1: prints zero residual, passes convergence, breaks.
    // So sink fires twice.
    AssertOrDie(recorded.size() == 2,
                "NR sink call count",
                "expected 2 calls (iter 0 + iter 1), got "
                + std::to_string(recorded.size()));

    // --- First call ---
    AssertOrDie(recorded[0].iter == 0,
                "NR call[0] iter", "expected 0");
    AssertNear(recorded[0].norm, p.norm0_expected, 1.0e-10,
               "NR call[0] norm", "expected sqrt(52)");
    AssertNear(recorded[0].norm0, p.norm0_expected, 1.0e-10,
               "NR call[0] norm0", "expected sqrt(52)");
    AssertOrDie(!recorded[0].converged_now,
                "NR call[0] converged_now",
                "expected false (sqrt(52) >> tol)");

    // --- Last call ---
    AssertOrDie(recorded[1].iter == 1,
                "NR call[1] iter", "expected 1");
    AssertNear(recorded[1].norm, 0.0, 1.0e-10,
               "NR call[1] norm", "expected ~0");
    AssertNear(recorded[1].norm0, p.norm0_expected, 1.0e-10,
               "NR call[1] norm0", "expected sqrt(52) unchanged");
    AssertOrDie(recorded[1].converged_now,
                "NR call[1] converged_now",
                "expected true (norm <= norm_max)");

    // --- norm_max consistency ---
    // norm_max = max(rel_tol*norm0, abs_tol) = max(1e-10 * sqrt(52), 1e-12)
    //         ≈ 7.21e-10
    const double norm_max_expected =
        std::max(1.0e-10 * p.norm0_expected, 1.0e-12);
    AssertNear(recorded[0].norm_max, norm_max_expected, 1.0e-15,
               "NR call[0] norm_max", "must match Newton's threshold");
    AssertNear(recorded[1].norm_max, norm_max_expected, 1.0e-15,
               "NR call[1] norm_max", "should not change between iters");

    std::cout << "  PASS  NR: 2 sink calls, correct norms, converged_now "
              << "transitions false→true" << std::endl;
}

//==============================================================================
// Test 2: ExaNewtonSolver — sink unset → no calls, default behavior intact
//==============================================================================
void test_nr_sink_unset()
{
    std::cout << "Test 2: ExaNewtonSolver with no sink installed" << std::endl;

    auto p = BuildProblem();

    ExaNewtonSolver newton(MPI_COMM_WORLD);
    newton.iterative_mode = true;
    newton.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    newton.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    newton.SetRelTol(1.0e-10);
    newton.SetAbsTol(1.0e-12);
    newton.SetMaxIter(10);
    newton.SetPrintLevel(-1);
    // Note: no SetDiagnosticSink call — m_diagnostic_sink stays default
    // (no-op std::function).

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    newton.Mult(dummy_b, x);

    AssertOrDie(newton.GetConverged() == 1,
                "NR no-sink converged flag", "expected 1");
    AssertNear(x[0], 2.0, 1.0e-10, "no-sink x[0]", "expected 2");
    AssertNear(x[1], 2.0, 1.0e-10, "no-sink x[1]", "expected 2");

    std::cout << "  PASS  unset sink: solver converges normally"
              << std::endl;
}

//==============================================================================
// Test 3: ExaNewtonLSSolver — sink fires, NRLS converges on linear problem
//==============================================================================
//
// On a linear problem, the line search's three-point quadratic fit
// reduces to alpha = 1 (the full Newton step is optimal); NRLS thus
// converges in the same iteration count as NR. We verify the same
// sink pattern.
void test_nrls_sink_basic()
{
    std::cout << "Test 3: ExaNewtonLSSolver sink + convergence" << std::endl;

    auto p = BuildProblem();

    ExaNewtonLSSolver newton(MPI_COMM_WORLD);
    newton.iterative_mode = true;
    newton.SetOperator(std::static_pointer_cast<mfem::Operator>(p.op));
    newton.SetSolver(std::static_pointer_cast<mfem::Solver>(p.solver));
    newton.SetRelTol(1.0e-10);
    newton.SetAbsTol(1.0e-12);
    newton.SetMaxIter(10);
    newton.SetPrintLevel(-1);

    std::vector<NewtonIterDiagnostic> recorded;
    newton.SetDiagnosticSink([&recorded](const NewtonIterDiagnostic& d)
    {
        recorded.push_back(d);
    });

    mfem::Vector x(2); x[0] = 0.0; x[1] = 0.0;
    mfem::Vector dummy_b;
    newton.Mult(dummy_b, x);

    // --- Solver state ---
    AssertOrDie(newton.GetConverged() == 1,
                "NRLS converged flag", "expected 1");
    AssertNear(x[0], 2.0, 1.0e-9, "NRLS x[0]", "expected 2");
    AssertNear(x[1], 2.0, 1.0e-9, "NRLS x[1]", "expected 2");

    // --- Sink calls — same structure as NR ---
    AssertOrDie(recorded.size() >= 2,
                "NRLS sink call count",
                "expected at least 2 sink calls, got "
                + std::to_string(recorded.size()));

    // First call must be iter 0 at the initial norm.
    AssertOrDie(recorded[0].iter == 0,
                "NRLS call[0] iter", "expected 0");
    AssertNear(recorded[0].norm, p.norm0_expected, 1.0e-10,
               "NRLS call[0] norm", "expected sqrt(52)");
    AssertOrDie(!recorded[0].converged_now,
                "NRLS call[0] converged_now",
                "expected false at iter 0");

    // Last call must signal convergence.
    const auto& last = recorded.back();
    AssertOrDie(last.converged_now,
                "NRLS last call converged_now",
                "expected true (loop broke on convergence branch)");
    AssertOrDie(last.norm <= last.norm_max,
                "NRLS last call norm <= norm_max",
                "sink invariant violated");

    // Iter indices must be 0, 1, 2, ... contiguous.
    for (size_t i = 0; i < recorded.size(); ++i)
    {
        AssertOrDie(recorded[i].iter == static_cast<int>(i),
                    "NRLS call[" + std::to_string(i) + "] iter sequence",
                    "iter indices must be contiguous from 0");
    }

    // norm0 must be the same in every call (captured pre-loop).
    for (size_t i = 1; i < recorded.size(); ++i)
    {
        AssertNear(recorded[i].norm0, recorded[0].norm0, 1.0e-15,
                   "NRLS call[" + std::to_string(i) + "] norm0 stability",
                   "norm0 must not change after iter 0");
    }

    std::cout << "  PASS  NRLS: " << recorded.size()
              << " sink calls, converged" << std::endl;
}

}   // anonymous namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "Running Newton diagnostic-sink unit tests" << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
    }

    test_nr_sink_basic();
    test_nr_sink_unset();
    test_nrls_sink_basic();

    if (rank == 0)
    {
        std::cout << "-----------------------------------------" << std::endl;
        std::cout << "All Newton diagnostic-sink tests passed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
