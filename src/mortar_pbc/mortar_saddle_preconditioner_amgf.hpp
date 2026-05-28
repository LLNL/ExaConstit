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
 * @brief AMGF block preconditioner for the mortar saddle Jacobian.
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
 * @par Augmented-Lagrangian saddle method
 * When `use_path_d` is true, this class applies the same
 * augmented-Lagrangian setup used by the generic
 * `MortarSaddlePreconditioner`: it builds
 * \f$K_\gamma = K + \gamma C^T C\f$, refreshes AMGF on
 * \f$K_\gamma\f$, and replaces the diagonal Schur approximation with the
 * trivial \f$\gamma I\f$ multiplier block. This is deliberately a saddle
 * method option, not a separate AMGF preconditioner flavor. AMGF remains the
 * K-block preconditioner; the augmented formulation changes the operator and
 * the lower block seen by that preconditioner.
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
     * @param use_path_d When false, build the original Path-A AMGF/diagonal
     *                   Schur preconditioner. When true, build
     *                   \f$K_\gamma\f$ and use \f$\gamma I\f$ for the
     *                   multiplier block.
     * @param gamma_override Augmented-Lagrangian gamma. Positive values are
     *                       used directly; non-positive values request the
     *                       trace-scaled default.
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
     * @brief Refresh AMGF on the current K block and rebuild the lambda block.
     *
     * @details In standard mode, `SetOperator()` unwraps the saddle
     * `BlockOperator`, extracts the mechanics K block, refreshes AMGF on K,
     * and rebuilds the existing diagonal Schur approximation through
     * `MortarConstraintOperator::ComputeInvDiagSchur`.
     *
     * In augmented-Lagrangian mode, the method accepts either the original
     * saddle `BlockOperator` or an `AugmentedLagrangianSaddleJacobian` wrapper.
     * The wrapper is unwrapped before extracting K so the penalty term is not
     * added twice. AMGF is refreshed on `K + gamma C^T C`, while the lower
     * block is set to the `gamma I` action expected by the augmented saddle
     * preconditioner.
     */
    void SetOperator(const mfem::Operator& op) override;

    /**
     * @brief Apply the block-diagonal AMGF / Schur preconditioner.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Current augmented-Lagrangian gamma value.
     *
     * @details Zero in standard Path-A mode. In augmented-Lagrangian mode this
     * is either the positive user override or the trace-scaled default chosen
     * during the most recent `SetOperator()` call.
     */
    const double& gamma() const { return m_gamma; }

    /**
     * @brief Lambda-block diagonal data from the most recent setup.
     *
     * @details In standard mode this exposes the inverse diagonal Schur data
     * passed to the internal `DiagonalScaler`. In augmented-Lagrangian mode
     * every entry is the selected gamma, matching the \f$\gamma I\f$ lower
     * block. The accessor is primarily for diagnostics and focused unit tests
     * that need to verify setup without forcing a full AMGF `Mult()`.
     */
    const mfem::Vector& GetPathAInverseSchurDiagonal() const
    {
        return m_schur_diag_inv;
    }

    /**
     * @brief Global dimension of the latest AMGF filtered displacement subspace.
     *
     * @details This is \f$|\mathcal{I}_K|\f$: the number of unique
     * displacement true DOFs touched by active mortar constraint columns. It
     * is not the number of lambda rows, so it can exceed `C_op.Height()` when
     * one constraint row couples multiple displacement true DOFs.
     */
    HYPRE_BigInt GetLastSubspaceDimension() const
    {
        return m_last_subspace_dim;
    }

    /**
     * @brief Latest filtered-subspace density, `dim(P) / rows(P)`.
     */
    double GetLastSubspaceDensity() const
    {
        return m_last_subspace_density;
    }

private:
    std::shared_ptr<mfem::Solver> m_K_jacobi_prec;
    std::shared_ptr<const MortarConstraintOperator> m_C_op;
    std::unique_ptr<mfem::HypreParMatrix> m_P;
    std::shared_ptr<mfem::Solver> m_subspace_solver;
    std::shared_ptr<mfem::AMGFSolver> m_amgf;
    bool m_rebuild_P_from_constraint = false;
    MPI_Comm m_comm = MPI_COMM_NULL;
    int m_print_level = 0;

    // Rebuilt on each SetOperator() call.
    mfem::Vector m_schur_diag_inv;
    std::unique_ptr<DiagonalScaler> m_S_block_prec;
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> m_block_prec;
    std::unique_ptr<mfem::HypreParMatrix> m_CtC;
    std::unique_ptr<mfem::HypreParMatrix> m_K_gamma;
    mfem::Array<int> m_block_offsets;
    const mfem::HypreParMatrix* m_K = nullptr;

    bool m_use_path_d = false;
    double m_gamma_override = -1.0;
    mutable double m_gamma = 0.0;
    HYPRE_BigInt m_last_subspace_dim = 0;
    double m_last_subspace_density = 0.0;
};

}  // namespace mortar_pbc
