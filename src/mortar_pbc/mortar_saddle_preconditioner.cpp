// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.5.B.2 / Phase D — MortarSaddlePreconditioner implementation.
//
// The generic saddle preconditioner has two setup modes:
//
//   1. Standard saddle mode:
//        - refresh the user's K-block preconditioner on K;
//        - refresh a Jacobi-style K probe on K;
//        - build the multiplier block from
//          diag(C diag(K)^-1 C^T)^-1.
//
//   2. Augmented-Lagrangian saddle mode:
//        - require a FULL-assembly Hypre K block;
//        - assemble C^T C from the active mortar rows;
//        - refresh the user's K-block preconditioner on
//          K_gamma = K + gamma C^T C;
//        - use gamma I for the multiplier block.
//
// The nonlinear residual and augmented RHS are handled outside this class; this
// file only owns the block preconditioner setup/action.

#include "mortar_saddle_preconditioner.hpp"
#include "augmented_lagrangian_saddle.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <cmath>
#include <utility>

namespace mortar_pbc {

namespace {

/**
 * @brief Compute the local trace contribution of an MFEM operator.
 *
 * @details The augmented-Lagrangian default gamma uses global traces of K and
 * \f$C^T C\f$. `AssembleDiagonal` gives a uniform interface for
 * `HypreParMatrix` and the lightweight test operators. This helper returns the
 * rank-local sum; the caller performs the MPI reduction so the communicator is
 * explicit at the gamma-selection site.
 */
double SumDiagonal(const mfem::Operator& op)
{
    mfem::Vector diag(op.Height());
    diag = 0.0;
    op.AssembleDiagonal(diag);
    return diag.Sum();
}

/**
 * @brief Choose the default augmented-Lagrangian penalty parameter.
 *
 * @details A positive user override bypasses this helper. When the configured
 * gamma is non-positive, Phase D uses
 *
 * \f[
 *   \gamma =
 *     \frac{\mathrm{tr}(K)}{\mathrm{tr}(C^T C)}
 *     \frac{n_\lambda}{n_u}.
 * \f]
 *
 * The trace ratio scales the constraint penalty to the current mechanics
 * tangent. The dimension ratio compensates for the fact that the displacement
 * and multiplier spaces generally have different sizes. If either matrix is
 * degenerate, the method falls back to gamma=1.0 and warns instead of aborting;
 * that keeps artificial tests and empty filters diagnosable while making the
 * unexpected production condition visible.
 */
double ComputeDefaultAugmentedGamma(const mfem::HypreParMatrix& K,
                                    const mfem::HypreParMatrix& CtC,
                                    HYPRE_BigInt n_lambda_global,
                                    HYPRE_BigInt n_u_global,
                                    MPI_Comm comm)
{
    double trK_local = SumDiagonal(K);
    double trCtC_local = SumDiagonal(CtC);

    double trK = 0.0;
    double trCtC = 0.0;
    MPI_Allreduce(&trK_local, &trK, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&trCtC_local, &trCtC, 1, MPI_DOUBLE, MPI_SUM, comm);

    if (trCtC <= 0.0 || n_lambda_global <= 0 || n_u_global <= 0)
    {
        MFEM_WARNING("MortarSaddlePreconditioner: default augmented "
                     "gamma could not be computed from traces "
                     "(tr(C^T C) <= 0 or empty dimensions); using gamma=1.");
        return 1.0;
    }

    return (trK / trCtC)
           * (static_cast<double>(n_lambda_global)
              / static_cast<double>(n_u_global));
}

}  // namespace

MortarSaddlePreconditioner::MortarSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> K_block_prec,
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    bool use_augmented_lagrangian,
    double gamma_override)
    : mfem::Solver(0, 0),  // size set in first SetOperator() call
      m_K_block_prec(std::move(K_block_prec)),
      m_K_jacobi_prec(std::move(K_jacobi_prec)),
      m_C_op(std::move(C_op)),
      m_block_offsets(3),
      m_use_augmented_lagrangian(use_augmented_lagrangian),
      m_gamma_override(gamma_override)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec::ctor");

    // The two K-side solvers serve different contracts. K_block_prec is the
    // user's actual upper-block preconditioner; K_jacobi_prec is a diagonal
    // probe used only by the standard Schur approximation. Keeping both
    // required, even in augmented mode, preserves constructor symmetry and
    // avoids null handling in refresh paths after option/spec changes.
    MFEM_VERIFY(m_K_block_prec,
                "MortarSaddlePreconditioner: K_block_prec must not be null");
    MFEM_VERIFY(m_K_jacobi_prec,
                "MortarSaddlePreconditioner: K_jacobi_prec must not be null");
    MFEM_VERIFY(m_C_op,
                "MortarSaddlePreconditioner: constraint operator shared_ptr "
                "must not be null");

    m_block_offsets = 0;
}

MortarSaddlePreconditioner::MortarSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> K_block_prec,
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    const MortarConstraintOperator& C_op,
    bool use_augmented_lagrangian,
    double gamma_override)
    : MortarSaddlePreconditioner(
          std::move(K_block_prec),
          std::move(K_jacobi_prec),
          std::shared_ptr<const MortarConstraintOperator>(
              &C_op, [](const MortarConstraintOperator*) {}),
          use_augmented_lagrangian,
          gamma_override)
{
}

void MortarSaddlePreconditioner::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec::set_operator");

    // ---- Step 1 — verify the operator is a saddle BlockOperator ----
    //
    // Caller is normally the inherited `mfem::IterativeSolver` path
    // inside ExaNewtonSolver::Mult, which forwards the saddle
    // Jacobian (BlockOperator) returned by
    // MortarSaddlePointSystem::GetGradient. In augmented-Lagrangian mode
    // the Krylov operator is an AugmentedLagrangianSaddleJacobian; unwrap it
    // before extracting K so this preconditioner builds K_gamma from the
    // original mechanics K block rather than adding gamma C^T C twice.
    const auto* augmented_jac =
        dynamic_cast<const AugmentedLagrangianSaddleJacobian*>(&op);
    const mfem::Operator& setup_op =
        augmented_jac ? augmented_jac->GetUnaugmentedGradient() : op;
    const auto* block_op =
        dynamic_cast<const mfem::BlockOperator*>(&setup_op);
    MFEM_VERIFY(block_op != nullptr,
                "MortarSaddlePreconditioner::SetOperator: operator is not "
                "a BlockOperator. Expected the saddle Jacobian from "
                "MortarSaddlePointSystem::GetGradient.");

    MFEM_VERIFY(block_op->NumRowBlocks() == 2 && block_op->NumColBlocks() == 2,
                "MortarSaddlePreconditioner::SetOperator: BlockOperator must "
                "be 2x2; got " << block_op->NumRowBlocks() << "x"
                << block_op->NumColBlocks());

    // ---- Step 2 — extract the K block (0,0) ----
    const mfem::Operator& K = block_op->GetBlock(0, 0);

    const int n_K   = K.Height();
    const int n_lam = m_C_op->Height();
    MFEM_VERIFY(K.Width() == n_K,
                "MortarSaddlePreconditioner: K must be square; got ("
                << K.Height() << ", " << K.Width() << ")");
    MFEM_VERIFY(m_C_op->Width() == n_K,
                "MortarSaddlePreconditioner: C_op cols (" << m_C_op->Width()
                << ") must match K rows (" << n_K << ")");

    // ---- Step 3 — refresh the K-block preconditioner ----
    //
    // Standard mode refreshes the user's preconditioner on K. Augmented
    // mode builds K_gamma = K + gamma C^T C and refreshes the same user's
    // preconditioner on the augmented operator.
    if (m_use_augmented_lagrangian)
    {
        const auto* K_hypre = dynamic_cast<const mfem::HypreParMatrix*>(&K);
        MFEM_VERIFY(K_hypre,
                    "MortarSaddlePreconditioner: augmented-Lagrangian "
                    "mode requires block (0,0) to be an mfem::HypreParMatrix "
                    "so K_gamma = K + gamma C^T C can be assembled.");

        // Build C^T C from the current active mortar rows. This is repeated
        // on every setup because active periodic specs can change the row set,
        // and because the operator's row ownership is part of the matrix
        // construction contract.
        m_CtC = m_C_op->BuildCTransposeC();
        long long n_lam_local_ll = static_cast<long long>(n_lam);
        long long n_lam_global_ll = 0;
        MPI_Allreduce(&n_lam_local_ll, &n_lam_global_ll, 1,
                      MPI_LONG_LONG_INT, MPI_SUM, m_C_op->Comm());

        m_gamma = (m_gamma_override > 0.0)
                      ? m_gamma_override
                      : ComputeDefaultAugmentedGamma(
                            *K_hypre, *m_CtC,
                            static_cast<HYPRE_BigInt>(n_lam_global_ll),
                            K_hypre->GetGlobalNumRows(), m_C_op->Comm());

        // Own K_gamma because downstream MFEM solvers keep references to the
        // operator supplied through SetOperator().
        m_K_gamma.reset(mfem::Add(1.0, *K_hypre, m_gamma, *m_CtC));
        MFEM_VERIFY(m_K_gamma,
                    "MortarSaddlePreconditioner: mfem::Add returned null "
                    "while building K_gamma");
        m_K_block_prec->SetOperator(*m_K_gamma);

        // The augmented saddle preconditioner uses gamma I in the multiplier
        // block. `DiagonalScaler` stores the diagonal action directly, so each
        // entry is gamma.
        mfem::Vector gamma_scale(n_lam);
        gamma_scale = m_gamma;
        m_S_block_prec = std::make_unique<DiagonalScaler>(
            n_lam, std::move(gamma_scale));
    }
    else
    {
        m_gamma = 0.0;
        m_CtC.reset();
        m_K_gamma.reset();

        // The user's choice (AMG, ILU, Jacobi, ...) re-runs its setup
        // against the current Newton iterate's K.
        m_K_block_prec->SetOperator(K);

        // Used only for probing diag(K)^{-1} via Mult(ones) inside
        // ComputeInvDiagSchur below.
        m_K_jacobi_prec->SetOperator(K);

        // ComputeInvDiagSchur internally:
        //   - probes K_jacobi_prec via Mult(ones) to recover diag(K)^{-1}
        //   - Allgathervs the values across ranks
        //   - walks per-pair blocks to compute
        //       inv_diag_S[i] = 1 / sum_j C_{ij}^2 * (1/diag(K))_j
        mfem::Vector inv_diag_S =
            m_C_op->ComputeInvDiagSchur(*m_K_jacobi_prec);
        MFEM_VERIFY(inv_diag_S.Size() == n_lam,
                    "MortarSaddlePreconditioner: ComputeInvDiagSchur "
                    "returned size " << inv_diag_S.Size()
                    << ", expected " << n_lam);

        m_S_block_prec = std::make_unique<DiagonalScaler>(
            n_lam, std::move(inv_diag_S));
    }

    m_block_offsets[0] = 0;
    m_block_offsets[1] = n_K;
    m_block_offsets[2] = n_K + n_lam;

    m_block_prec = std::make_unique<mfem::BlockDiagonalPreconditioner>(
        m_block_offsets);
    m_block_prec->SetDiagonalBlock(0, m_K_block_prec.get());
    m_block_prec->SetDiagonalBlock(1, m_S_block_prec.get());

    // ---- Final step — update inherited Solver size to match ----
    height = n_K + n_lam;
    width = n_K + n_lam;
}

void MortarSaddlePreconditioner::Mult(const mfem::Vector& x,
                                       mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec::mult");

    MFEM_VERIFY(m_block_prec,
                "MortarSaddlePreconditioner::Mult called before SetOperator");
    MFEM_ASSERT(x.Size() == height && y.Size() == height,
                "MortarSaddlePreconditioner::Mult: size mismatch");

    m_block_prec->Mult(x, y);
}

}  // namespace mortar_pbc
