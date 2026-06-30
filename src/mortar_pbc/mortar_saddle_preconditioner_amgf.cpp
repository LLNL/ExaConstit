// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// AMGF-backed block preconditioner for mortar-periodic saddle systems.

#include "mortar_saddle_preconditioner_amgf.hpp"
#include "augmented_lagrangian_saddle.hpp"
#include "amgf_utils.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <iostream>
#include <utility>

#ifdef HAVE_CALIPER
#include "caliper/cali.h"
#endif

namespace mortar_pbc {

namespace {

/**
 * @brief Sum the diagonal entries of an MFEM operator on the local rank.
 *
 * @details `AssembleDiagonal` is available on both `HypreParMatrix` and the
 * small operator wrappers used in focused tests. The augmented-Lagrangian
 * default gamma heuristic only needs the global trace, so this helper computes
 * the local contribution and the caller performs the MPI reduction.
 */
double SumDiagonal(const mfem::Operator& op)
{
    mfem::Vector diag(op.Height());
    diag = 0.0;
    op.AssembleDiagonal(diag);
    return diag.Sum();
}

/**
 * @brief Compute the trace-scaled default augmented-Lagrangian gamma.
 *
 * @details A positive user override is preferred, but a non-positive gamma
 * requests the Phase D default
 *
 * \f[
 *   \gamma =
 *     \frac{\mathrm{tr}(K)}{\mathrm{tr}(C^T C)}
 *     \frac{n_\lambda}{n_u}.
 * \f]
 *
 * The trace ratio puts the penalty term on the same rough scale as the current
 * mechanics tangent, and the size ratio prevents the scalar from drifting only
 * because the displacement and multiplier spaces have different dimensions.
 * Degenerate traces fall back to 1.0 rather than aborting because an empty or
 * zero-trace artificial test matrix should not make option parsing unusable;
 * production matrices should not normally take this branch.
 */
double ComputeDefaultAugmentedGammaAMGF(const mfem::HypreParMatrix& K,
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
        MFEM_WARNING("MortarSaddlePreconditionerAMGF: default augmented "
                     "gamma could not be computed from traces "
                     "(tr(C^T C) <= 0 or empty dimensions); using gamma=1.");
        return 1.0;
    }

    return (trK / trCtC)
           * (static_cast<double>(n_lambda_global)
              / static_cast<double>(n_u_global));
}

}  // namespace

MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    std::unique_ptr<mfem::HypreParMatrix> P,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override,
    int vector_dim,
    bool order_bynodes,
    int print_level,
    mfem::HypreSolver::ErrorMode boomer_error_mode)
    : mfem::Solver(0, 0),
      m_K_jacobi_prec(std::move(K_jacobi_prec)),
      m_C_op(std::move(C_op)),
      m_P(std::move(P)),
      m_subspace_solver(std::move(subspace_solver)),
      m_print_level(print_level),
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
    m_amgf->GetAMG().SetErrorMode(boomer_error_mode);

    m_amgf->SetFilteredSubspaceSolver(*m_subspace_solver);
    if (m_P)
    {
        // AMGFSolver stores a reference to P, so this class owns P for the
        // full preconditioner lifetime in the explicit-P construction path.
        m_amgf->SetFilteredSubspaceTransferOperator(*m_P);
    }

    m_block_offsets = 0;
}

MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    std::shared_ptr<const MortarConstraintOperator> C_op,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override,
    MPI_Comm comm,
    int vector_dim,
    bool order_bynodes,
    int print_level,
    mfem::HypreSolver::ErrorMode boomer_error_mode)
    : MortarSaddlePreconditionerAMGF(
          std::move(K_jacobi_prec),
          std::move(C_op),
          std::unique_ptr<mfem::HypreParMatrix>(),
          std::move(subspace_solver),
          use_path_d,
          gamma_override,
          vector_dim,
          order_bynodes,
          print_level,
          boomer_error_mode)
{
    m_rebuild_P_from_constraint = true;
    m_comm = comm;
    MFEM_VERIFY(m_comm != MPI_COMM_NULL,
                "MortarSaddlePreconditionerAMGF: deferred-P constructor "
                "requires a valid MPI communicator");
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
    int print_level,
    mfem::HypreSolver::ErrorMode boomer_error_mode)
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
          print_level,
          boomer_error_mode)
{
}

MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    const MortarConstraintOperator& C_op,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override,
    MPI_Comm comm,
    int vector_dim,
    bool order_bynodes,
    int print_level,
    mfem::HypreSolver::ErrorMode boomer_error_mode)
    : MortarSaddlePreconditionerAMGF(
          std::move(K_jacobi_prec),
          std::shared_ptr<const MortarConstraintOperator>(
              &C_op, [](const MortarConstraintOperator*) {}),
          std::move(subspace_solver),
          use_path_d,
          gamma_override,
          comm,
          vector_dim,
          order_bynodes,
          print_level,
          boomer_error_mode)
{
}

void MortarSaddlePreconditionerAMGF::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::set_operator");

    const auto* augmented_jac =
        dynamic_cast<const AugmentedLagrangianSaddleJacobian*>(&op);
    const mfem::Operator& setup_op =
        augmented_jac ? augmented_jac->GetUnaugmentedGradient() : op;
    const auto* block_op = dynamic_cast<const mfem::BlockOperator*>(&setup_op);
    MFEM_VERIFY(block_op != nullptr,
                "MortarSaddlePreconditionerAMGF::SetOperator: operator is "
                "not a BlockOperator. Expected the saddle Jacobian from "
                "MortarSaddlePointSystem::GetGradient.");
    MFEM_VERIFY(block_op->NumRowBlocks() == 2 && block_op->NumColBlocks() == 2,
                "MortarSaddlePreconditionerAMGF::SetOperator: "
                "BlockOperator must be 2x2; got "
                << block_op->NumRowBlocks() << "x"
                << block_op->NumColBlocks());

    const auto* K_original = dynamic_cast<const mfem::HypreParMatrix*>(
        &block_op->GetBlock(0, 0));
    MFEM_VERIFY(K_original != nullptr,
                "MortarSaddlePreconditionerAMGF::SetOperator: block (0,0) "
                "must be an mfem::HypreParMatrix because AMGF requires FULL "
                "assembly");
    m_K = K_original;

    const int n_K = m_K->Height();
    const int n_lam = m_C_op->Height();
    MFEM_VERIFY(m_K->Width() == n_K,
                "MortarSaddlePreconditionerAMGF: K must be square; got ("
                << m_K->Height() << ", " << m_K->Width() << ")");
    MFEM_VERIFY(m_C_op->Width() == n_K,
                "MortarSaddlePreconditionerAMGF: C_op cols ("
                << m_C_op->Width() << ") must match K rows (" << n_K
                << ")");
    const mfem::HypreParMatrix* K_for_amgf = m_K;
    if (m_use_path_d)
    {
        m_CtC = m_C_op->BuildCTransposeC();
        long long n_lam_local_ll = static_cast<long long>(n_lam);
        long long n_lam_global_ll = 0;
        MPI_Allreduce(&n_lam_local_ll, &n_lam_global_ll, 1,
                      MPI_LONG_LONG_INT, MPI_SUM, m_C_op->Comm());

        m_gamma = (m_gamma_override > 0.0)
                      ? m_gamma_override
                      : ComputeDefaultAugmentedGammaAMGF(
                            *m_K, *m_CtC,
                            static_cast<HYPRE_BigInt>(n_lam_global_ll),
                            m_K->GetGlobalNumRows(), m_C_op->Comm());

        m_K_gamma.reset(mfem::Add(1.0, *m_K, m_gamma, *m_CtC));
        MFEM_VERIFY(m_K_gamma,
                    "MortarSaddlePreconditionerAMGF: mfem::Add returned "
                    "null while building K_gamma");
        K_for_amgf = m_K_gamma.get();
    }
    else
    {
        m_gamma = 0.0;
        m_CtC.reset();
        m_K_gamma.reset();
    }

    if (m_rebuild_P_from_constraint)
    {
        m_P.reset(exaconstit::amgf::BuildBooleanRestrictionProlongation(
            m_K->GetGlobalNumRows(),
            m_C_op->GetConstraintCoupledDofIndices(),
            m_K->GetRowStarts(),
            m_comm));
        m_amgf->SetFilteredSubspaceTransferOperator(*m_P);
    }
    MFEM_VERIFY(m_P,
                "MortarSaddlePreconditionerAMGF: AMGF transfer P must not "
                "be null");
    MFEM_VERIFY(m_P->Height() == n_K,
                "MortarSaddlePreconditionerAMGF: AMGF transfer P local "
                "height (" << m_P->Height() << ") must match K local "
                "height (" << n_K << ")");
    MFEM_VERIFY(m_P->GetGlobalNumRows() == m_K->GetGlobalNumRows(),
                "MortarSaddlePreconditionerAMGF: AMGF transfer P global "
                "rows (" << m_P->GetGlobalNumRows()
                << ") must match K global rows ("
                << m_K->GetGlobalNumRows() << ")");

    m_last_subspace_dim = m_P->GetGlobalNumCols();
    const HYPRE_BigInt n_u_global = m_P->GetGlobalNumRows();
    m_last_subspace_density =
        (n_u_global > 0)
            ? static_cast<double>(m_last_subspace_dim)
                  / static_cast<double>(n_u_global)
            : 0.0;

#ifdef HAVE_CALIPER
    cali_set_int_byname(
        "amgf.subspace_dim",
        static_cast<int>(m_last_subspace_dim));
    cali_set_double_byname(
        "amgf.subspace_density",
        m_last_subspace_density);
#endif

    if (m_print_level >= 1)
    {
        int rank = 0;
        MPI_Comm comm = m_comm;
        if (comm == MPI_COMM_NULL)
        {
            comm = m_K->GetComm();
        }
        MPI_Comm_rank(comm, &rank);
        if (rank == 0)
        {
            std::cout << "[AMGF] "
                      << (m_use_path_d
                              ? "Augmented-Lagrangian path active"
                              : "Path A active")
                      << "; |I_K|="
                      << m_last_subspace_dim
                      << " ("
                      << 100.0 * m_last_subspace_density
                      << "% of n_u=" << n_u_global
                      << "); parallel direct (SuperLU_DIST) subspace solve host-resident."
                      << std::endl;
        }
    }

    CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::amg_setup_k");
    m_amgf->SetOperator(*K_for_amgf);
    CALI_MARK_END("mortar_pbc::saddle_prec_amgf::amg_setup_k");

    if (m_use_path_d)
    {
        m_schur_diag_inv.SetSize(n_lam);
        m_schur_diag_inv = m_gamma;
        m_S_block_prec = std::make_unique<DiagonalScaler>(
            n_lam, mfem::Vector(m_schur_diag_inv));
    }
    else
    {
        m_K_jacobi_prec->SetOperator(*m_K);

        CALI_MARK_BEGIN(
            "mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");
        m_schur_diag_inv = m_C_op->ComputeInvDiagSchur(*m_K_jacobi_prec);
        CALI_MARK_END(
            "mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");

        MFEM_VERIFY(m_schur_diag_inv.Size() == n_lam,
                    "MortarSaddlePreconditionerAMGF: ComputeInvDiagSchur "
                    "returned size " << m_schur_diag_inv.Size()
                    << ", expected " << n_lam);

        m_S_block_prec = std::make_unique<DiagonalScaler>(
            n_lam, mfem::Vector(m_schur_diag_inv));
    }

    m_block_offsets[0] = 0;
    m_block_offsets[1] = n_K;
    m_block_offsets[2] = n_K + n_lam;

    m_block_prec = std::make_unique<mfem::BlockDiagonalPreconditioner>(
        m_block_offsets);
    m_block_prec->SetDiagonalBlock(0, m_amgf.get());
    m_block_prec->SetDiagonalBlock(1, m_S_block_prec.get());

    height = n_K + n_lam;
    width = n_K + n_lam;
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
