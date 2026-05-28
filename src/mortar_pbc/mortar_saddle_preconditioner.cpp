// Phase 5.5.B.2 — MortarSaddlePreconditioner implementation.

#include "mortar_saddle_preconditioner.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <cmath>
#include <utility>

namespace mortar_pbc {

namespace {

double SumDiagonal(const mfem::Operator& op)
{
    mfem::Vector diag(op.Height());
    diag = 0.0;
    op.AssembleDiagonal(diag);
    return diag.Sum();
}

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
    // MortarSaddlePointSystem::GetGradient.
    const auto* block_op = dynamic_cast<const mfem::BlockOperator*>(&op);
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

        m_K_gamma.reset(mfem::Add(1.0, *K_hypre, m_gamma, *m_CtC));
        MFEM_VERIFY(m_K_gamma,
                    "MortarSaddlePreconditioner: mfem::Add returned null "
                    "while building K_gamma");
        m_K_block_prec->SetOperator(*m_K_gamma);

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

    // ---- Step 7 — update inherited Solver size to match ----
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
