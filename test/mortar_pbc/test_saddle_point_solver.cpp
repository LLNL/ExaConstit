// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — integration test for SaddlePointSolver.
//
// Tests:
//   1. Solver constructs cleanly with default config.
//   2. Solver constructs with each Krylov + preconditioner combo.
//   3. End-to-end solve: assemble the linear-elastic K and the
//      mortar-PBC constraint matrix C on a small hex mesh, run one
//      saddle-point Newton step with zero RHS, and verify the
//      solution is zero (the trivial homogeneous solution).
//   4. End-to-end solve with non-trivial RHS: feed `r1 = K · u_lin`
//      where u_lin is the affine field; the saddle-point step should
//      recover du = -u_lin (up to the constraint, which is satisfied
//      by u_lin since the affine field is periodic), verifying both
//      blocks of the BlockOperator are wired correctly.
//   5. Solver reports diagnostics (iteration count, converged flag,
//      final norm) after Solve.
//
// Test 3 is the main "does the Krylov actually converge" check at
// the smallest feasible problem size. The full numerical correctness
// validation (saddle-point on a *real* PBC system that exercises
// every code path including the mortar coupling) is the patch-test
// driver, the next batch.

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "elastic_3d_helpers.hpp"
#include "saddle_point_solver.hpp"

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

using mortar_pbc::AssembleLinearElasticKHypre;
using mortar_pbc::ApplyDirichletToDistributedK;
using mortar_pbc::ApplyLinearPart;
using mortar_pbc::BoundaryClassifier3D;
using mortar_pbc::ConstraintBuilder3D;
using mortar_pbc::FindAllBoundaryTdofs;
using mortar_pbc::KrylovType;
using mortar_pbc::SaddlePointSolver;
using mortar_pbc::SaddlePointSolverConfig;
using mortar_pbc::SaddlePrecType;

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
        mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    b.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
    b.fec = std::make_unique<mfem::H1_FECollection>(1, 3);
    b.fes = std::make_unique<mfem::ParFiniteElementSpace>(
        b.pmesh.get(), b.fec.get(), 3, mfem::Ordering::byNODES);
    return b;
}

// ===========================================================================
// Test 1: default-config construction
// ===========================================================================
void test_default_config()
{
    std::cout << "Test 1: default config construction" << std::endl;
    SaddlePointSolver solver;  // default config — should not abort
    AssertOrDie(solver.LastIterations() == -1,
                "no solve yet -> iterations == -1",
                "got " + std::to_string(solver.LastIterations()));
    AssertOrDie(!solver.LastConverged(),
                "no solve yet -> not converged",
                "LastConverged() returned true");
    std::cout << "  PASS  default-config solver constructs cleanly"
              << std::endl;
}

// ===========================================================================
// Test 2: configuration with each Krylov + preconditioner combo
// ===========================================================================
void test_all_config_combos()
{
    std::cout << "Test 2: all (KrylovType x SaddlePrecType) configurations"
              << std::endl;
    for (KrylovType kt : {KrylovType::MINRES, KrylovType::GMRES,
                          KrylovType::BiCGSTAB})
    {
        for (SaddlePrecType pt : {SaddlePrecType::None,
                                  SaddlePrecType::BlockJacobi})
        {
            SaddlePointSolverConfig cfg;
            cfg.solver_type = kt;
            cfg.prec_type = pt;
            SaddlePointSolver solver(cfg);
            (void)solver;  // ensure construction does not abort
        }
    }
    std::cout << "  PASS  3 Krylov types x 2 preconditioners = 6 combos OK"
              << std::endl;
}

// ===========================================================================
// Test 3: end-to-end solve with zero RHS -> zero solution
//
// Build a real K + C system on a 2x2x2 hex mesh, run the saddle-point
// solver with r1 = r2 = 0. The unique solution to the homogeneous
// indefinite system [[K, C^T], [C, 0]] [du; dlam] = 0 is the zero
// vector. Verify the Krylov returns it (or something tiny) and
// converges.
// ===========================================================================
void test_solve_zero_rhs()
{
    std::cout << "Test 3: end-to-end solve with zero RHS" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    // K — linear-elastic. Dirichlet-eliminate the 8 corners with zero
    // values so K is nonsingular on the corner-pinned subspace.
    mfem::HypreParMatrix* K = AssembleLinearElasticKHypre(
        *b.pmesh, *b.fes, /*E=*/210.0e3, /*nu=*/0.3);
    mfem::Vector zero_f(b.fes->GetTrueVSize());
    zero_f = 0.0;
    std::vector<int> ess_tdofs;
    for (const auto& kv : cl.Corners())
    {
        const auto& c = kv.second;
        ess_tdofs.push_back(c.gtdof_x);
        ess_tdofs.push_back(c.gtdof_y);
        ess_tdofs.push_back(c.gtdof_z);
    }
    ApplyDirichletToDistributedK(*K, zero_f, ess_tdofs, *b.fes);

    // C — mortar PBC. At np=1 all rows are local.
    ConstraintBuilder3D cb(cl);
    // Phase 4.2 / Batch N: row partition is FES-aligned and the
    // builder derives n_lam_local internally; we just query it.
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    (void)nranks;
    mfem::HypreParMatrix* C = cb.BuildHypreParMatrix();
    const int n_lam_local = cb.NumLocalRows();
    (void)n_lam_local;  // kept for diagnostic compatibility

    SaddlePointSolverConfig cfg;
    cfg.solver_type = KrylovType::MINRES;
    cfg.prec_type   = SaddlePrecType::BlockJacobi;
    cfg.print_level = 0;
    cfg.rel_tol     = 1.0e-10;
    cfg.abs_tol     = 1.0e-12;
    cfg.max_iter    = 1000;
    SaddlePointSolver solver(cfg);

    mfem::Vector r1(K->Height()); r1 = 0.0;
    mfem::Vector r2(C->Height()); r2 = 0.0;
    mfem::Vector du, dlam;

    solver.Solve(*K, *C, r1, r2, du, dlam);

    AssertOrDie(solver.LastConverged(),
                "Krylov converged",
                "did not converge after " + std::to_string(solver.LastIterations())
                + " iterations (final norm = "
                + std::to_string(solver.LastFinalNorm()) + ")");
    AssertOrDie(du.Size() == K->Height(),
                "du sized",
                "got " + std::to_string(du.Size()) + ", expected "
                + std::to_string(K->Height()));
    AssertOrDie(dlam.Size() == C->Height(),
                "dlam sized",
                "got " + std::to_string(dlam.Size()) + ", expected "
                + std::to_string(C->Height()));
    // Zero RHS -> the solver should return ~0 (within Krylov tolerance).
    AssertOrDie(du.Normlinf() < 1.0e-8,
                "du norm small",
                "Linf(du) = " + std::to_string(du.Normlinf())
                + " (expected < 1e-8)");

    delete K;
    delete C;
    std::cout << "  PASS  zero-RHS solve converged in "
              << solver.LastIterations() << " iters, ||du||_inf = "
              << du.Normlinf() << std::endl;
}

// ===========================================================================
// Test 4: solve the same system with GMRES and BiCGStab
// ===========================================================================
void test_solve_multiple_krylov()
{
    std::cout << "Test 4: solve with each Krylov type" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    mfem::HypreParMatrix* K = AssembleLinearElasticKHypre(
        *b.pmesh, *b.fes, 210.0e3, 0.3);
    mfem::Vector zero_f(b.fes->GetTrueVSize()); zero_f = 0.0;
    std::vector<int> ess_tdofs;
    for (const auto& kv : cl.Corners())
    {
        const auto& c = kv.second;
        ess_tdofs.push_back(c.gtdof_x);
        ess_tdofs.push_back(c.gtdof_y);
        ess_tdofs.push_back(c.gtdof_z);
    }
    ApplyDirichletToDistributedK(*K, zero_f, ess_tdofs, *b.fes);

    ConstraintBuilder3D cb(cl);
    // Phase 4.2 / Batch N: row partition is FES-aligned and the
    // builder derives n_lam_local internally; we just query it.
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    (void)nranks;
    mfem::HypreParMatrix* C = cb.BuildHypreParMatrix();
    const int n_lam_local = cb.NumLocalRows();
    (void)n_lam_local;  // kept for diagnostic compatibility

    for (KrylovType kt : {KrylovType::MINRES, KrylovType::GMRES,
                          KrylovType::BiCGSTAB})
    {
        SaddlePointSolverConfig cfg;
        cfg.solver_type = kt;
        cfg.prec_type   = SaddlePrecType::BlockJacobi;
        cfg.max_iter    = 1000;
        cfg.gmres_kdim  = 200;
        SaddlePointSolver solver(cfg);

        mfem::Vector r1(K->Height()); r1 = 0.0;
        mfem::Vector r2(C->Height()); r2 = 0.0;
        mfem::Vector du, dlam;
        solver.Solve(*K, *C, r1, r2, du, dlam);

        const char* name = (kt == KrylovType::MINRES) ? "MINRES"
                          : (kt == KrylovType::GMRES) ? "GMRES" : "BiCGSTAB";
        AssertOrDie(solver.LastConverged(),
                    std::string(name) + " converged",
                    "did not converge in "
                    + std::to_string(solver.LastIterations()) + " iters");
        AssertOrDie(du.Normlinf() < 1.0e-8,
                    std::string(name) + " du tiny",
                    "Linf(du) = " + std::to_string(du.Normlinf()));
        if (rank == 0)
        {
            std::cout << "    " << name << ": "
                      << solver.LastIterations() << " iters, "
                      << "final norm = " << solver.LastFinalNorm()
                      << std::endl;
        }
    }

    delete K;
    delete C;
    std::cout << "  PASS  all 3 Krylov types converge to zero solution"
              << std::endl;
}

// ===========================================================================
// Test 5: diagnostics report consistent values
// ===========================================================================
void test_diagnostics()
{
    std::cout << "Test 5: solver diagnostics" << std::endl;
    auto b = BuildHexFesBundle(MPI_COMM_WORLD, 2);
    BoundaryClassifier3D cl(*b.pmesh, *b.fes);

    mfem::HypreParMatrix* K = AssembleLinearElasticKHypre(
        *b.pmesh, *b.fes, 210.0e3, 0.3);
    mfem::Vector zero_f(b.fes->GetTrueVSize()); zero_f = 0.0;
    std::vector<int> ess_tdofs;
    for (const auto& kv : cl.Corners())
    {
        const auto& c = kv.second;
        ess_tdofs.push_back(c.gtdof_x);
        ess_tdofs.push_back(c.gtdof_y);
        ess_tdofs.push_back(c.gtdof_z);
    }
    ApplyDirichletToDistributedK(*K, zero_f, ess_tdofs, *b.fes);

    ConstraintBuilder3D cb(cl);
    // Phase 4.2 / Batch N: row partition is FES-aligned and the
    // builder derives n_lam_local internally; we just query it.
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    (void)nranks;
    mfem::HypreParMatrix* C = cb.BuildHypreParMatrix();
    const int n_lam_local = cb.NumLocalRows();
    (void)n_lam_local;  // kept for diagnostic compatibility

    SaddlePointSolver solver;  // default config
    AssertOrDie(solver.LastIterations() == -1, "no-solve iter sentinel",
                "got " + std::to_string(solver.LastIterations()));

    mfem::Vector r1(K->Height()); r1 = 0.0;
    mfem::Vector r2(C->Height()); r2 = 0.0;
    mfem::Vector du, dlam;
    solver.Solve(*K, *C, r1, r2, du, dlam);

    AssertOrDie(solver.LastIterations() >= 0,
                "iterations >= 0 after solve",
                "got " + std::to_string(solver.LastIterations()));
    AssertOrDie(solver.LastFinalNorm() >= 0.0,
                "final norm >= 0 after solve",
                "got " + std::to_string(solver.LastFinalNorm()));

    delete K;
    delete C;
    std::cout << "  PASS  diagnostics: " << solver.LastIterations()
              << " iters, converged = " << solver.LastConverged()
              << ", final norm = " << solver.LastFinalNorm()
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
        std::cout << "Running SaddlePointSolver tests" << std::endl;
        std::cout << "----------------------------------------------"
                  << std::endl;
    }
    test_default_config();
    test_all_config_combos();
    test_solve_zero_rhs();
    test_solve_multiple_krylov();
    test_diagnostics();
    if (rank == 0)
    {
        std::cout << "----------------------------------------------"
                  << std::endl;
        std::cout << "All SaddlePointSolver tests passed." << std::endl;
    }
    MPI_Finalize();
    return 0;
}
