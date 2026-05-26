// Phase 5.5.B.2 — MortarSaddlePreconditioner implementation.

#include "mortar_saddle_preconditioner.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <utility>

namespace mortar_pbc {

MortarSaddlePreconditioner::MortarSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> K_block_prec,
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    std::shared_ptr<const MortarConstraintOperator> C_op)
    : mfem::Solver(0, 0),  // size set in first SetOperator() call
      m_K_block_prec(std::move(K_block_prec)),
      m_K_jacobi_prec(std::move(K_jacobi_prec)),
      m_C_op(std::move(C_op)),
      m_block_offsets(3)
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
    const MortarConstraintOperator& C_op)
    : MortarSaddlePreconditioner(
          std::move(K_block_prec),
          std::move(K_jacobi_prec),
          std::shared_ptr<const MortarConstraintOperator>(
              &C_op, [](const MortarConstraintOperator*) {}))
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
    // The user's choice (AMG, ILU, Jacobi, ...) re-runs its setup
    // against the current Newton iterate's K. Cost is dominated by
    // this step.
    m_K_block_prec->SetOperator(K);

    // ---- Step 4 — refresh the K-Jacobi preconditioner ----
    //
    // Used only for probing diag(K)^{-1} via Mult(ones) inside
    // ComputeInvDiagSchur below. Cheap to set up since it just
    // extracts the diagonal.
    m_K_jacobi_prec->SetOperator(K);

    // ---- Step 5 — compute the Schur-complement inverse diagonal ----
    //
    // ComputeInvDiagSchur internally:
    //   - probes K_jacobi_prec via Mult(ones) to recover diag(K)^{-1}
    //   - Allgathervs the values across ranks
    //   - walks per-pair blocks to compute
    //       inv_diag_S[i] = 1 / sum_j C_{ij}^2 * (1/diag(K))_j
    mfem::Vector inv_diag_S = m_C_op->ComputeInvDiagSchur(*m_K_jacobi_prec);
    MFEM_VERIFY(inv_diag_S.Size() == n_lam,
                "MortarSaddlePreconditioner: ComputeInvDiagSchur returned "
                "size " << inv_diag_S.Size() << ", expected " << n_lam);

    // ---- Step 6 — rebuild the BlockDiagonalPreconditioner ----
    m_S_block_prec = std::make_unique<DiagonalScaler>(
        n_lam, std::move(inv_diag_S));

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
