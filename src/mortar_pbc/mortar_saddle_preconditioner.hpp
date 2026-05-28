// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.5.B.2 / Phase D — generic mortar saddle preconditioner.
//
// This file declares the non-AMGF block preconditioner used by the mortar
// saddle Newton solve. The class is intentionally generic in the upper block:
// it consumes whatever K-block preconditioner SystemDriver has already built
// from `[Solvers.Krylov] preconditioner`, and only owns the saddle-specific
// logic needed to combine that K action with a multiplier-block approximation.
//
// Phase D extends the setup with an augmented-Lagrangian mode. That mode is a
// saddle-method choice, not a K-block preconditioner choice: this class still
// delegates the upper block to the user's K preconditioner, but refreshes that
// preconditioner on K_gamma = K + gamma C^T C and replaces the Schur diagonal
// lower block with gamma I.

#ifndef EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP
#define EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP

#include "diagonal_scaler.hpp"
#include "mortar_constraint_operator.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

/**
 * @brief Block-diagonal preconditioner for the mortar saddle-point
 *        Jacobian.
 *
 * @details Approximates the inverse of the saddle Jacobian
 * \f[
 *   J = \begin{bmatrix} K & C^T \\ C & 0 \end{bmatrix}
 * \f]
 * by a block-diagonal preconditioner
 * \f[
 *   M^{-1} = \begin{bmatrix} M_K^{-1} & 0 \\ 0 & M_S^{-1} \end{bmatrix}
 * \f]
 * where:
 *   - \f$M_K^{-1}\f$ is the user-supplied K-block preconditioner
 *     (the existing ExaConstit `J_prec` — AMG, ILU, Jacobi, etc.).
 *     Refreshed on every `SetOperator` call by forwarding the
 *     extracted K block.
 *   - \f$M_S^{-1}\f$ is a `DiagonalScaler` over the inverse Schur-
 *     complement diagonal
 *     \f$\big[\mathrm{diag}(C\,\mathrm{diag}(K)^{-1}\,C^T)\big]^{-1}\f$,
 *     computed via `MortarConstraintOperator::ComputeInvDiagSchur`.
 *
 * The reason two separate preconditioners are passed at construction
 * — rather than just one — is that:
 *   1. The K-block preconditioner can be anything (AMG, ILU, ...);
 *      MINRES requires SPD action on the (0,0) block, which any
 *      reasonable choice satisfies.
 *   2. The Schur-diagonal computation needs the actual
 *      \f$\mathrm{diag}(K)^{-1}\f$ values, not just the action of
 *      some other preconditioner. Probing those values requires a
 *      Jacobi-style preconditioner whose `Mult(ones, _)` returns
 *      \f$\mathrm{diag}(K)^{-1}\f$ directly. Forcing the K-block
 *      preconditioner to be Jacobi (so it could double as the
 *      probe target) would unnecessarily restrict the user's
 *      choice for the K block.
 *
 * Both preconditioners' `SetOperator` is called with the extracted
 * K block on every saddle `SetOperator` call, so they stay
 * consistent with the current Newton iterate.
 *
 * @par Designed-for use with MINRES
 * The block-diagonal Jacobi preconditioner is symmetric (assuming
 * symmetric K-block prec) and is the natural pair for MINRES on
 * an indefinite saddle system. Using GMRES would also work but
 * loses the short-recurrence advantage.
 *
 * @par Lifetime / ownership
 * The constructor takes shared ownership of both preconditioners
 * (`std::shared_ptr`) — the caller may continue to use them
 * elsewhere (e.g., the K-block prec may also serve as the standalone
 * `J_prec` for non-mortar branches if any) — but typically the
 * SystemDriver constructs them, hands them off, and lets the
 * preconditioner own them.
 *
 * The preconditioner stores shared access to the constraint operator.
 * In ExaConstit this shared handle comes from `MortarPbcManager`,
 * which also owns the projector-aware operator used by the saddle
 * system. A reference constructor remains available for legacy tests
 * and wraps the reference in a non-owning shared pointer.
 *
 * @par Augmented-Lagrangian mode
 * Phase D adds an opt-in augmented-Lagrangian setup path for the
 * saddle solver method. In that mode the class still wraps the user's
 * selected K-block preconditioner, but refreshes it on
 * \f$K_\gamma = K + \gamma C^T C\f$ instead of on \f$K\f$ and replaces
 * the diagonal-lumped Schur block with the trivial \f$\gamma I\f$
 * action. This keeps the algebraic method choice separate from the
 * K-block preconditioner choice: AMG, ILU, Jacobi, or AMGF-style
 * wrappers can all be tested against the augmented formulation.
 */
class MortarSaddlePreconditioner : public mfem::Solver
{
public:
    /**
     * @brief Construct from K-block + K-Jacobi preconditioners and a
     *        constraint operator.
     *
     * @param K_block_prec   Preconditioner for the (0,0) block of
     *                       the BlockDiagonal preconditioner. Any
     *                       `mfem::Solver` (AMG, ILU, Jacobi, ...).
     *                       `SetOperator(K)` will be called on every
     *                       refresh.
     * @param K_jacobi_prec  Jacobi-style preconditioner used by
     *                       `MortarConstraintOperator::ComputeInvDiagSchur`
     *                       to extract `diag(K)^{-1}` values. MUST
     *                       satisfy the contract `Mult(ones, y)` →
     *                       `y[i] = (1/diag(K))_i`. `DiagonalScaler`,
     *                       `MechOperatorJacobiSmoother` (in default
     *                       non-iterative mode), and Hypre's
     *                       `HypreDiagScale` all satisfy this.
     * @param C_op           Shared constraint operator. Must be non-null.
     *                       Kept alive by this preconditioner.
     * @param use_augmented_lagrangian
     *                       When false, use the original saddle
     *                       preconditioner. When true, build
     *                       \f$K_\gamma\f$ and use \f$\gamma I\f$ for the
     *                       lambda block.
     * @param gamma_override Augmented-Lagrangian gamma. A positive value is
     *                       used directly. A non-positive value requests the
     *                       default trace-scaled gamma
     *                       \f$\mathrm{tr}(K)/\mathrm{tr}(C^T C)\,
     *                       n_\lambda/n_u\f$.
     */
    MortarSaddlePreconditioner(
        std::shared_ptr<mfem::Solver> K_block_prec,
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        std::shared_ptr<const MortarConstraintOperator> C_op,
        bool use_augmented_lagrangian = false,
        double gamma_override = -1.0);

    /**
     * @brief Compatibility constructor from a non-owned constraint
     *        operator reference.
     *
     * @details Delegates to the shared-handle constructor through a
     * non-owning aliasing `shared_ptr`. Prefer the shared-handle
     * overload in production Phase 6 code so the preconditioner
     * participates in the manager's explicit operator ownership.
     */
    MortarSaddlePreconditioner(
        std::shared_ptr<mfem::Solver> K_block_prec,
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        bool use_augmented_lagrangian = false,
        double gamma_override = -1.0);

    ~MortarSaddlePreconditioner() override = default;

    MortarSaddlePreconditioner(const MortarSaddlePreconditioner&) = delete;
    MortarSaddlePreconditioner& operator=(
        const MortarSaddlePreconditioner&) = delete;

    /**
     * @brief Refresh both internal K-side preconditioners and rebuild
     *        the Schur-block diagonal scaler.
     *
     * @param op  Saddle Jacobian as `mfem::BlockOperator`. Caller is
     *            typically `mfem::IterativeSolver::SetPreconditioner`'s
     *            indirect path, which forwards
     *            `MortarSaddlePointSystem::GetGradient(x)` here.
     *
     * @details Steps:
     *   1. `dynamic_cast` `op` to `mfem::BlockOperator`. Aborts if
     *      `op` is not the saddle BlockOperator (mismatch is a
     *      programmer error, not a recoverable runtime condition).
     *   2. Extract `K = block_op.GetBlock(0, 0)`.
     *   3. In standard mode, forward `K` into
     *      `K_block_prec->SetOperator(K)`. In augmented-Lagrangian mode,
     *      require `K` to be a `HypreParMatrix`, build
     *      `K_gamma = K + gamma C^T C`, and forward `K_gamma` instead.
     *   4. Forward `K` into `K_jacobi_prec->SetOperator(K)` — the
     *      Jacobi probe target refreshes its `inv_diag` to match
     *      the current Newton iterate. Skipped in augmented-Lagrangian
     *      mode because the Schur block no longer uses a diagonal probe.
     *   5. Compute `inv_diag_S = C_op.ComputeInvDiagSchur(*K_jacobi_prec)`
     *      — the constraint operator probes `K_jacobi_prec` via
     *      `Mult(ones)` to extract the diagonal values, then walks
     *      its per-pair blocks to build the Schur diagonal.
     *   6. Build a fresh lambda-block `DiagonalScaler`: either the
     *      existing inverse Schur diagonal in standard mode, or the
     *      scalar \f$\gamma I\f$ action in augmented-Lagrangian mode.
     *
     * Steps 1–6 run once per Newton iteration. The cost is
     * dominated by step 3 (e.g. AMG re-setup) and is amortised
     * over the Krylov iterations that follow.
     */
    void SetOperator(const mfem::Operator& op) override;

    /**
     * @brief Apply the block-diagonal preconditioner.
     *
     * @details Delegates to the internal `BlockDiagonalPreconditioner`,
     * which applies `K_block_prec` to the upper block and the
     * Schur `DiagonalScaler` to the lower block.
     *
     * @pre `SetOperator` must have been called at least once.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Whether the latest setup used the augmented-Lagrangian mode.
     */
    bool UsesAugmentedLagrangian() const
    {
        return m_use_augmented_lagrangian;
    }

    /**
     * @brief Gamma used by the most recent augmented-Lagrangian setup.
     *
     * @details Returns zero before the first augmented setup and remains zero
     * for the standard saddle preconditioner path.
     */
    double Gamma() const { return m_gamma; }

private:
    /**
     * @brief User-selected preconditioner for the displacement block.
     *
     * @details In standard mode `SetOperator()` refreshes this solver on the
     * raw mechanics tangent K. In augmented-Lagrangian mode it is refreshed on
     * the assembled penalty-shifted matrix
     * \f$K_\gamma = K + \gamma C^T C\f$. The class never assumes a concrete
     * solver type here; it can be AMG, ILU, Jacobi, Chebyshev, or another
     * `mfem::Solver` chosen by the existing Krylov options.
     */
    std::shared_ptr<mfem::Solver> m_K_block_prec;

    /**
     * @brief Jacobi-style K probe used only by the standard Schur diagonal.
     *
     * @details `ComputeInvDiagSchur` requires a solver whose
     * `Mult(ones, y)` returns \f$\mathrm{diag}(K)^{-1}\f$ entrywise. This is
     * deliberately separate from `m_K_block_prec` so the user's upper-block
     * preconditioner is not forced to be Jacobi. The member is not used in
     * augmented-Lagrangian mode because the multiplier block is \f$\gamma I\f$.
     */
    std::shared_ptr<mfem::Solver> m_K_jacobi_prec;

    /**
     * @brief Active mortar constraint operator.
     *
     * @details Supplies C matvecs, C^T matvecs, the standard diagonal Schur
     * approximation, and the assembled \f$C^T C\f$ matrix needed by the
     * augmented-Lagrangian setup. The shared handle keeps the operator alive
     * for as long as the preconditioner may be used by MFEM's Krylov stack.
     */
    std::shared_ptr<const MortarConstraintOperator> m_C_op;

    /**
     * @brief Lambda-block diagonal solver rebuilt on every setup.
     *
     * @details Holds either the standard inverse diagonal Schur approximation
     * or the augmented \f$\gamma I\f$ action.
     */
    std::unique_ptr<DiagonalScaler> m_S_block_prec;

    /**
     * @brief Two-block preconditioner combining K and lambda actions.
     *
     * @details Recreated by `SetOperator()` after the K and lambda block
     * solvers have both been refreshed. `Mult()` delegates directly to this
     * object.
     */
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> m_block_prec;

    /**
     * @brief Assembled penalty matrix \f$C^T C\f$ for augmented mode.
     *
     * @details Null in standard mode. Rebuilt on each augmented setup because
     * active mortar rows may change across periodic-BC filter updates.
     */
    std::unique_ptr<mfem::HypreParMatrix> m_CtC;

    /**
     * @brief Assembled augmented displacement block
     *        \f$K_\gamma = K + \gamma C^T C\f$.
     *
     * @details Null in standard mode. Owned here because MFEM solvers store
     * references to the operator passed through `SetOperator()`, so the matrix
     * must outlive the K-block preconditioner setup/use interval.
     */
    std::unique_ptr<mfem::HypreParMatrix> m_K_gamma;

    /**
     * @brief Local block offsets `[0, n_u, n_u + n_lambda]`.
     */
    mfem::Array<int> m_block_offsets;

    /**
     * @brief Whether this instance uses the augmented-Lagrangian setup path.
     */
    bool m_use_augmented_lagrangian = false;

    /**
     * @brief User-provided gamma override.
     *
     * @details Positive values are used directly. Non-positive values request
     * the trace-scaled default during each augmented `SetOperator()` call.
     */
    double m_gamma_override = -1.0;

    /**
     * @brief Gamma selected during the most recent setup.
     *
     * @details Zero in standard mode. In augmented mode this is either
     * `m_gamma_override` or the computed trace-scaled default.
     */
    double m_gamma = 0.0;
};

}  // namespace mortar_pbc

#endif  // EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP
