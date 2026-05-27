// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// AMGF-backed block preconditioner for mortar-periodic saddle systems.

#pragma once

#include "diagonal_scaler.hpp"
#include "mortar_constraint_operator.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

/**
 * @brief Path-A AMGF block preconditioner for the mortar saddle Jacobian.
 *
 * @details The mortar-PBC Newton Jacobian has the saddle form
 * \f[
 *   J = \begin{bmatrix} K & C^T \\ C & 0 \end{bmatrix},
 * \f]
 * where \f$K\f$ is the displacement tangent and \f$C\f$ is the mortar
 * constraint operator. This class builds the Path-A preconditioner from the
 * AMGF design document:
 * \f[
 *   M_A^{-1} =
 *   \begin{bmatrix}
 *     M_{\mathrm{AMGF}}(K, P)^{-1} & 0 \\
 *     0 &
 *     \left[\mathrm{diag}(C\,\mathrm{diag}(K)^{-1}\,C^T)\right]^{-1}
 *   \end{bmatrix}.
 * \f]
 *
 * The upper block is MFEM's `AMGFSolver`: BoomerAMG on the full displacement
 * block plus an exact filtered-subspace correction on the Boolean
 * prolongation \f$P\f$ built from the displacement true DOFs touched by active
 * mortar constraints. The filtered-subspace solve is supplied by the caller so
 * the production path can use `exaconstit::amgf::GinkgoDirectSubspaceSolver`.
 *
 * The lower block intentionally preserves ExaConstit's existing diagonal
 * Schur approximation. `K_jacobi_prec` is not the K-block preconditioner; it is
 * only the Jacobi-style probe target required by
 * `MortarConstraintOperator::ComputeInvDiagSchur` to recover
 * \f$\mathrm{diag}(K)^{-1}\f$.
 *
 * @par AMGF assumptions
 * `SetOperator()` requires the saddle operator to be an `mfem::BlockOperator`
 * with block (0,0) stored as an `mfem::HypreParMatrix`. AMGF therefore belongs
 * to the FULL-assembly CPU/OpenMP branch guarded by option validation. Partial
 * assembly, element assembly, and GPU-resident matrix-free paths are outside
 * this class' contract.
 *
 * @par Path D
 * The constructor keeps the Path-D toggle and gamma argument so the later
 * augmented-Lagrangian step can extend this class without changing the public
 * construction site. This partial step implements Path A only; requesting Path
 * D currently aborts with a clear message in `SetOperator()`.
 *
 * @par Lifetime / ownership
 * The AMGF transfer matrix \f$P\f$ is moved into this object because MFEM's
 * filtered solver keeps references to the transfer operator. The subspace
 * solver is held by shared ownership for the same reason. The constraint
 * operator can be supplied either as a shared handle or, for legacy unit tests,
 * as a non-owning reference.
 */
class MortarSaddlePreconditionerAMGF : public mfem::Solver
{
public:
    /**
     * @brief Construct an AMGF-backed saddle preconditioner.
     *
     * @param K_jacobi_prec Jacobi-style preconditioner used only to probe
     *                      `diag(K)^{-1}` for the Path-A Schur diagonal.
     * @param C_op Shared mortar constraint operator. Must outlive this object
     *             through shared ownership.
     * @param P Boolean AMGF transfer operator over coupled displacement DOFs.
     *          Moved into this object and kept alive for AMGF.
     * @param subspace_solver Solver for AMGF's filtered subspace operator,
     *                        typically `GinkgoDirectSubspaceSolver`.
     * @param use_path_d Reserved for the later augmented-Lagrangian path.
     *                   Must be false in this partial implementation.
     * @param gamma_override Reserved Path-D augmentation parameter.
     * @param vector_dim Number of displacement components passed to
     *                   BoomerAMG's systems option. ExaConstit's 3D mechanics
     *                   path uses 3.
     * @param order_bynodes Whether vector true DOFs are ordered by nodes.
     *                      Passed through to the same BoomerAMG systems setup
     *                      used by the existing K-block AMG path.
     * @param print_level BoomerAMG print level.
     * @param boomer_error_mode Hypre error handling mode for AMGF's internal
     *                          BoomerAMG. Production defaults to aborting on
     *                          Hypre setup/solve errors; focused tests may
     *                          request warning mode for artificial matrices
     *                          that trigger nonfatal Hypre warnings.
     */
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        std::unique_ptr<mfem::HypreParMatrix> P,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override = -1.0,
        int vector_dim = 3,
        bool order_bynodes = true,
        int print_level = 0,
        mfem::HypreSolver::ErrorMode boomer_error_mode =
            mfem::HypreSolver::ABORT_HYPRE_ERRORS);

    /**
     * @brief Construct with deferred AMGF transfer construction.
     *
     * @details This overload is the production SystemDriver path. The K block
     * is not available when the saddle preconditioner is constructed, so the
     * Boolean AMGF transfer matrix is rebuilt in `SetOperator()` from the
     * current K row partition and `C_op.GetConstraintCoupledDofIndices()`.
     */
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override,
        MPI_Comm comm,
        int vector_dim = 3,
        bool order_bynodes = true,
        int print_level = 0,
        mfem::HypreSolver::ErrorMode boomer_error_mode =
            mfem::HypreSolver::ABORT_HYPRE_ERRORS);

    /**
     * @brief Compatibility constructor from a non-owned constraint operator.
     */
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        std::unique_ptr<mfem::HypreParMatrix> P,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override = -1.0,
        int vector_dim = 3,
        bool order_bynodes = true,
        int print_level = 0,
        mfem::HypreSolver::ErrorMode boomer_error_mode =
            mfem::HypreSolver::ABORT_HYPRE_ERRORS);

    /**
     * @brief Compatibility deferred-P constructor from a non-owned constraint
     *        operator reference.
     */
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override,
        MPI_Comm comm,
        int vector_dim = 3,
        bool order_bynodes = true,
        int print_level = 0,
        mfem::HypreSolver::ErrorMode boomer_error_mode =
            mfem::HypreSolver::ABORT_HYPRE_ERRORS);

    ~MortarSaddlePreconditionerAMGF() override = default;

    MortarSaddlePreconditionerAMGF(
        const MortarSaddlePreconditionerAMGF&) = delete;
    MortarSaddlePreconditionerAMGF& operator=(
        const MortarSaddlePreconditionerAMGF&) = delete;

    /**
     * @brief Refresh AMGF on the K block and rebuild the Path-A Schur scaler.
     */
    void SetOperator(const mfem::Operator& op) override;

    /**
     * @brief Apply the block-diagonal AMGF / Schur preconditioner.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Current Path-D gamma value.
     *
     * @details Present for the future Path-D residual augmenter. It remains
     * zero while this class is used in Path-A mode.
     */
    const double& gamma() const { return m_gamma; }

    /**
     * @brief Path-A inverse Schur diagonal from the most recent setup.
     *
     * @details This exposes the same lower-block data passed to the internal
     * `DiagonalScaler`. It is primarily useful for diagnostics and focused
     * unit tests that need to verify the AMGF wrapper preserved the existing
     * `MortarConstraintOperator::ComputeInvDiagSchur` path without forcing a
     * full AMGF `Mult()`.
     */
    const mfem::Vector& GetPathAInverseSchurDiagonal() const
    {
        return m_schur_diag_inv;
    }

private:
    std::shared_ptr<mfem::Solver> m_K_jacobi_prec;
    std::shared_ptr<const MortarConstraintOperator> m_C_op;
    std::unique_ptr<mfem::HypreParMatrix> m_P;
    std::shared_ptr<mfem::Solver> m_subspace_solver;
    std::shared_ptr<mfem::AMGFSolver> m_amgf;
    bool m_rebuild_P_from_constraint = false;
    MPI_Comm m_comm = MPI_COMM_NULL;

    // Rebuilt on each SetOperator() call.
    mfem::Vector m_schur_diag_inv;
    std::unique_ptr<DiagonalScaler> m_S_block_prec;
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> m_block_prec;
    mfem::Array<int> m_block_offsets;
    const mfem::HypreParMatrix* m_K = nullptr;

    bool m_use_path_d = false;
    double m_gamma_override = -1.0;
    mutable double m_gamma = 0.0;
};

}  // namespace mortar_pbc
