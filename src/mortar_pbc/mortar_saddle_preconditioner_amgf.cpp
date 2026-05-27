// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// AMGF-backed block preconditioner for mortar-periodic saddle systems.

#include "mortar_saddle_preconditioner_amgf.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <utility>

namespace mortar_pbc {

MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    std::unique_ptr<mfem::HypreParMatrix> P,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override,
    int vector_dim,
    bool order_bynodes,
    int print_level)
    : mfem::Solver(0, 0),
      m_K_jacobi_prec(std::move(K_jacobi_prec)),
      m_C_op(std::move(C_op)),
      m_P(std::move(P)),
      m_subspace_solver(std::move(subspace_solver)),
      m_block_offsets(3),
      m_use_path_d(use_path_d),
      m_gamma_override(gamma_override)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::ctor");

    MFEM_VERIFY(m_K_jacobi_prec,
                "MortarSaddlePreconditionerAMGF: K_jacobi_prec must not "
                "be null");
    MFEM_VERIFY(m_C_op,
                "MortarSaddlePreconditionerAMGF: constraint operator "
                "shared_ptr must not be null");
    MFEM_VERIFY(m_P,
                "MortarSaddlePreconditionerAMGF: AMGF transfer P must not "
                "be null");
    MFEM_VERIFY(m_subspace_solver,
                "MortarSaddlePreconditionerAMGF: subspace solver must not "
                "be null");
    MFEM_VERIFY(vector_dim > 0,
                "MortarSaddlePreconditionerAMGF: vector_dim must be "
                "positive");

    m_amgf = std::make_shared<mfem::AMGFSolver>();

    // Match ExaConstit's existing full-assembly K-block AMG setup: use
    // MFEM's systems option so Hypre sees the vector-valued displacement
    // ordering, and do not retune strength/coarsening/smoother parameters
    // inside the AMGF wrapper.
    m_amgf->GetAMG().SetSystemsOptions(vector_dim, order_bynodes);
    m_amgf->GetAMG().SetPrintLevel(print_level);

    // AMGFSolver stores references to both objects, so this class owns P and
    // keeps shared ownership of the solver for the full preconditioner
    // lifetime.
    m_amgf->SetFilteredSubspaceTransferOperator(*m_P);
    m_amgf->SetFilteredSubspaceSolver(*m_subspace_solver);

    m_block_offsets = 0;
}

MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    const MortarConstraintOperator& C_op,
    std::unique_ptr<mfem::HypreParMatrix> P,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override,
    int vector_dim,
    bool order_bynodes,
    int print_level)
    : MortarSaddlePreconditionerAMGF(
          std::move(K_jacobi_prec),
          std::shared_ptr<const MortarConstraintOperator>(
              &C_op, [](const MortarConstraintOperator*) {}),
          std::move(P),
          std::move(subspace_solver),
          use_path_d,
          gamma_override,
          vector_dim,
          order_bynodes,
          print_level)
{
}

void MortarSaddlePreconditionerAMGF::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::set_operator");

    MFEM_VERIFY(!m_use_path_d,
                "MortarSaddlePreconditionerAMGF: Path D / augmented "
                "Lagrangian setup is reserved for the later Path-D partial "
                "step and is not implemented in this Path-A class-level "
                "step");
    (void)m_gamma_override;

    const auto* block_op = dynamic_cast<const mfem::BlockOperator*>(&op);
    MFEM_VERIFY(block_op != nullptr,
                "MortarSaddlePreconditionerAMGF::SetOperator: operator is "
                "not a BlockOperator. Expected the saddle Jacobian from "
                "MortarSaddlePointSystem::GetGradient.");
    MFEM_VERIFY(block_op->NumRowBlocks() == 2 && block_op->NumColBlocks() == 2,
                "MortarSaddlePreconditionerAMGF::SetOperator: "
                "BlockOperator must be 2x2; got "
                << block_op->NumRowBlocks() << "x"
                << block_op->NumColBlocks());

    m_K = dynamic_cast<const mfem::HypreParMatrix*>(
        &block_op->GetBlock(0, 0));
    MFEM_VERIFY(m_K != nullptr,
                "MortarSaddlePreconditionerAMGF::SetOperator: block (0,0) "
                "must be an mfem::HypreParMatrix because AMGF requires FULL "
                "assembly");

    const int n_K = m_K->Height();
    const int n_lam = m_C_op->Height();
    MFEM_VERIFY(m_K->Width() == n_K,
                "MortarSaddlePreconditionerAMGF: K must be square; got ("
                << m_K->Height() << ", " << m_K->Width() << ")");
    MFEM_VERIFY(m_C_op->Width() == n_K,
                "MortarSaddlePreconditionerAMGF: C_op cols ("
                << m_C_op->Width() << ") must match K rows (" << n_K
                << ")");
    MFEM_VERIFY(m_P->Height() == n_K,
                "MortarSaddlePreconditionerAMGF: AMGF transfer P local "
                "height (" << m_P->Height() << ") must match K local "
                "height (" << n_K << ")");
    MFEM_VERIFY(m_P->GetGlobalNumRows() == m_K->GetGlobalNumRows(),
                "MortarSaddlePreconditionerAMGF: AMGF transfer P global "
                "rows (" << m_P->GetGlobalNumRows()
                << ") must match K global rows ("
                << m_K->GetGlobalNumRows() << ")");

    CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::amg_setup_k");
    m_amgf->SetOperator(*m_K);
    CALI_MARK_END("mortar_pbc::saddle_prec_amgf::amg_setup_k");

    m_K_jacobi_prec->SetOperator(*m_K);

    CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");
    m_schur_diag_inv = m_C_op->ComputeInvDiagSchur(*m_K_jacobi_prec);
    CALI_MARK_END("mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");

    MFEM_VERIFY(m_schur_diag_inv.Size() == n_lam,
                "MortarSaddlePreconditionerAMGF: ComputeInvDiagSchur "
                "returned size " << m_schur_diag_inv.Size()
                << ", expected " << n_lam);

    m_S_block_prec = std::make_unique<DiagonalScaler>(
        n_lam, mfem::Vector(m_schur_diag_inv));

    m_block_offsets[0] = 0;
    m_block_offsets[1] = n_K;
    m_block_offsets[2] = n_K + n_lam;

    m_block_prec = std::make_unique<mfem::BlockDiagonalPreconditioner>(
        m_block_offsets);
    m_block_prec->SetDiagonalBlock(0, m_amgf.get());
    m_block_prec->SetDiagonalBlock(1, m_S_block_prec.get());

    height = n_K + n_lam;
    width = n_K + n_lam;
    m_gamma = 0.0;
}

void MortarSaddlePreconditionerAMGF::Mult(const mfem::Vector& x,
                                          mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::mult");

    MFEM_VERIFY(m_block_prec,
                "MortarSaddlePreconditionerAMGF::Mult called before "
                "SetOperator");
    MFEM_ASSERT(x.Size() == height && y.Size() == height,
                "MortarSaddlePreconditionerAMGF::Mult: size mismatch");

    m_block_prec->Mult(x, y);
}

}  // namespace mortar_pbc
