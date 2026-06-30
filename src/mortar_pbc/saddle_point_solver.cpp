// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of SaddlePointSolver, ported from
// `mortar_pbc/saddle_point.py`. See header for design doc.

#include "saddle_point_solver.hpp"
#include "diagonal_scaler.hpp"
#include "mortar_constraint_operator.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace mortar_pbc {

//==============================================================================
// Constructor
//==============================================================================

SaddlePointSolver::SaddlePointSolver(const SaddlePointSolverConfig& cfg)
    : m_cfg(cfg)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point::ctor");
    // Defensive enum check; the enum itself has no CG, but we surface
    // an explicit error rather than silently falling through.
    switch (m_cfg.solver_type)
    {
        case KrylovType::MINRES:
        case KrylovType::GMRES:
        case KrylovType::BiCGSTAB:
            break;
        default:
            MFEM_ABORT("SaddlePointSolver: unknown KrylovType "
                       << static_cast<int>(m_cfg.solver_type));
    }
    switch (m_cfg.prec_type)
    {
        case SaddlePrecType::None:
        case SaddlePrecType::BlockJacobi:
            break;
        default:
            MFEM_ABORT("SaddlePointSolver: unknown SaddlePrecType "
                       << static_cast<int>(m_cfg.prec_type));
    }
    MFEM_VERIFY(m_cfg.rel_tol > 0.0,
                "SaddlePointSolver: rel_tol must be positive (got "
                << m_cfg.rel_tol << ")");
    MFEM_VERIFY(m_cfg.abs_tol > 0.0,
                "SaddlePointSolver: abs_tol must be positive (got "
                << m_cfg.abs_tol << ")");
    MFEM_VERIFY(m_cfg.max_iter > 0,
                "SaddlePointSolver: max_iter must be positive (got "
                << m_cfg.max_iter << ")");
}

//==============================================================================
// Solve
//==============================================================================

void SaddlePointSolver::Solve(const mfem::Operator& K,
                              const MortarConstraintOperator& C_op,
                              const mfem::Solver& K_jacobi_prec,
                              const mfem::Vector& r1,
                              const mfem::Vector& r2,
                              mfem::Vector& du,
                              mfem::Vector& dlam)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point::solve");

    const int n_v_local   = K.Height();
    const int n_lam_local = C_op.Height();

    MFEM_VERIFY(K.Width() == n_v_local,
                "SaddlePointSolver::Solve: K must be square; got ("
                << K.Height() << ", " << K.Width() << ")");
    MFEM_VERIFY(C_op.Width() == n_v_local,
                "SaddlePointSolver::Solve: C_op cols ("
                << C_op.Width() << ") must match K rows ("
                << n_v_local << ")");
    MFEM_VERIFY(K_jacobi_prec.Height() == n_v_local,
                "SaddlePointSolver::Solve: K_jacobi_prec height ("
                << K_jacobi_prec.Height() << ") must match K rows ("
                << n_v_local << ")");
    MFEM_VERIFY(K_jacobi_prec.Width() == n_v_local,
                "SaddlePointSolver::Solve: K_jacobi_prec width ("
                << K_jacobi_prec.Width() << ") must match K cols ("
                << n_v_local << ")");
    MFEM_VERIFY(r1.Size() == n_v_local,
                "SaddlePointSolver::Solve: r1 size (" << r1.Size()
                << ") must match K.Height() (" << n_v_local << ")");
    MFEM_VERIFY(r2.Size() == n_lam_local,
                "SaddlePointSolver::Solve: r2 size (" << r2.Size()
                << ") must match C_op.Height() (" << n_lam_local
                << ")");

    // Probe K_jacobi_prec for inv_diag_K. The contract is that
    // K_jacobi_prec.Mult(ones, _) returns diag(K)^{-1} elementwise.
    // See SaddlePointSolver::Solve doxygen for the list of valid
    // prec types.
    //
    // This is a local op (one elementwise Solver application). The
    // same probe runs again inside ComputeInvDiagSchur; we accept
    // the duplication to avoid a parallel-API split between
    // "Solve takes inv_diag_K Vector" and "Solve takes Solver".
    // Cost is dominated by the Allgatherv inside
    // ComputeInvDiagSchur, not the local probe.
    mfem::Vector inv_diag_K(n_v_local);
    {
        mfem::Vector ones(n_v_local);
        ones = 1.0;
        K_jacobi_prec.Mult(ones, inv_diag_K);
    }

    mfem::Vector inv_diag_S = C_op.ComputeInvDiagSchur(K_jacobi_prec);

    SolveImplInternal(
        const_cast<mfem::Operator&>(K),
        const_cast<MortarConstraintOperator&>(C_op),
        C_op.Comm(),
        inv_diag_K, inv_diag_S,
        n_v_local, n_lam_local,
        r1, r2, du, dlam);
}

//==============================================================================
// Phase 4.3 / Batch S — internal helper shared by both Solve overloads.
//
// Identical Krylov plumbing for both the HypreParMatrix path and the
// EA path. Differences land in the caller (which computes inv_diag_S
// its own way and provides the right operator references).
//
// K_op and C_op enter as mutable mfem::Operator& because mfem's
// BlockOperator::SetBlock signature takes Operator*. The caller has
// already cast away const where appropriate.
//==============================================================================
void SaddlePointSolver::SolveImplInternal(
    mfem::Operator& K_op,
    mfem::Operator& C_op,
    MPI_Comm comm,
    mfem::Vector& inv_diag_K,
    mfem::Vector& inv_diag_S,
    int n_v_local,
    int n_lam_local,
    const mfem::Vector& r1,
    const mfem::Vector& r2,
    mfem::Vector& du,
    mfem::Vector& dlam)
{
    //---- Build the block operator [[K, C^T], [C, 0]] ----
    //
    // C^T is wrapped as a TransposeOperator over C; this dispatches
    // BlockOperator's calls to C_op.MultTranspose (which both
    // HypreParMatrix and MortarConstraintOperator implement).
    mfem::Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = n_v_local;
    block_offsets[2] = n_v_local + n_lam_local;

    mfem::TransposeOperator CT_op(&C_op);

    mfem::BlockOperator block_op(block_offsets);
    block_op.SetBlock(0, 0, &K_op);
    block_op.SetBlock(0, 1, &CT_op);
    block_op.SetBlock(1, 0, &C_op);
    // (1, 1) is the zero block — not set.

    //---- Build the block-diagonal preconditioner ----
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec;
    std::unique_ptr<DiagonalScaler> jacobi_K;
    std::unique_ptr<DiagonalScaler> jacobi_S;
    if (m_cfg.prec_type == SaddlePrecType::BlockJacobi)
    {
        jacobi_K = std::make_unique<DiagonalScaler>(n_v_local,
                                                    std::move(inv_diag_K));
        jacobi_S = std::make_unique<DiagonalScaler>(n_lam_local,
                                                    std::move(inv_diag_S));

        block_prec = std::make_unique<mfem::BlockDiagonalPreconditioner>(
            block_offsets);
        block_prec->SetDiagonalBlock(0, jacobi_K.get());
        block_prec->SetDiagonalBlock(1, jacobi_S.get());
    }

    //---- Build the RHS [-r1; -r2] ----
    //
    // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean: r1 and r2 are
    // freshly-built input vectors (per-Newton-iteration); we Host-Read
    // them and Host-Write the rhs blocks via raw pointers. The block
    // views into rhs share the underlying memory with rhs itself, so
    // the writes propagate back to rhs's GetBlock as expected.
    mfem::BlockVector rhs(block_offsets);
    {
        const double* r1_d = r1.HostRead();
        const double* r2_d = r2.HostRead();
        mfem::Vector& rhs_v = rhs.GetBlock(0);
        mfem::Vector& rhs_l = rhs.GetBlock(1);
        double* rhs_v_d = rhs_v.HostWrite();
        double* rhs_l_d = rhs_l.HostWrite();
        for (int i = 0; i < n_v_local; ++i)   { rhs_v_d[i] = -r1_d[i]; }
        for (int i = 0; i < n_lam_local; ++i) { rhs_l_d[i] = -r2_d[i]; }
    }

    //---- Krylov solver ----
    std::unique_ptr<mfem::IterativeSolver> krylov;
    switch (m_cfg.solver_type)
    {
        case KrylovType::MINRES:
            krylov = std::make_unique<mfem::MINRESSolver>(comm);
            break;
        case KrylovType::GMRES:
        {
            auto* gmres = new mfem::GMRESSolver(comm);
            gmres->SetKDim(m_cfg.gmres_kdim);
            krylov.reset(gmres);
            break;
        }
        case KrylovType::BiCGSTAB:
            krylov = std::make_unique<mfem::BiCGSTABSolver>(comm);
            break;
    }
    krylov->SetRelTol(m_cfg.rel_tol);
    krylov->SetAbsTol(m_cfg.abs_tol);
    krylov->SetMaxIter(m_cfg.max_iter);
    krylov->SetPrintLevel(m_cfg.print_level);
    krylov->SetOperator(block_op);
    if (block_prec) { krylov->SetPreconditioner(*block_prec); }

    // Force the solver to ignore the input solution as initial guess
    // and start from zero. The Newton outer loop carries information
    // across iterations via u_tilde and λ; the inner linear solve is
    // for the INCREMENTAL update (du, dλ). Reusing the previous
    // step's du as initial guess is a category error.
    krylov->iterative_mode = false;

    //---- Solve ----
    mfem::BlockVector solution(block_offsets);
    solution = 0.0;  // zero initial guess
    krylov->Mult(rhs, solution);

    //---- Diagnostics ----
    m_last_iterations  = krylov->GetNumIterations();
    m_last_converged   = krylov->GetConverged();
    m_last_final_norm  = krylov->GetFinalNorm();

    //---- Extract du and dlam ----
    du.SetSize(n_v_local);
    dlam.SetSize(n_lam_local);
    {
        const mfem::Vector& sol_v = solution.GetBlock(0);
        const mfem::Vector& sol_l = solution.GetBlock(1);
        const double* sv_d = sol_v.HostRead();
        const double* sl_d = sol_l.HostRead();
        double* du_d   = du.HostWrite();
        double* dlam_d = dlam.HostWrite();
        for (int i = 0; i < n_v_local; ++i)   { du_d[i]   = sv_d[i]; }
        for (int i = 0; i < n_lam_local; ++i) { dlam_d[i] = sl_d[i]; }
    }
}

}  // namespace mortar_pbc
