#ifndef EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP
#define EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP

// Phase 5.5.B.2 — block-diagonal Jacobi preconditioner for the
// mortar saddle-point Jacobian. Wraps an existing K-block
// preconditioner (e.g. AMG, ILU, Jacobi — whatever the user has
// configured for J_prec) and a K-Jacobi preconditioner used to
// build the Schur-complement diagonal.

#include "diagonal_scaler.hpp"
#include "mortar_constraint_operator.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

/**
 * @brief Block-diagonal Jacobi preconditioner for the mortar
 *        saddle-point Jacobian.
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
 * The `MortarConstraintOperator&` reference must outlive this
 * preconditioner. In ExaConstit this is satisfied because the
 * constraint operator lives in the `MortarPbcManager`, which the
 * `SystemDriver` owns alongside this preconditioner.
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
     * @param C_op           Constraint operator. Reference must
     *                       outlive this preconditioner.
     */
    MortarSaddlePreconditioner(
        std::shared_ptr<mfem::Solver> K_block_prec,
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op);

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
     *   3. Forward `K` into `K_block_prec->SetOperator(K)` — the
     *      user's K-block preconditioner refreshes its internal
     *      machinery (e.g. AMG hierarchy, ILU factorisation).
     *   4. Forward `K` into `K_jacobi_prec->SetOperator(K)` — the
     *      Jacobi probe target refreshes its `inv_diag` to match
     *      the current Newton iterate.
     *   5. Compute `inv_diag_S = C_op.ComputeInvDiagSchur(*K_jacobi_prec)`
     *      — the constraint operator probes `K_jacobi_prec` via
     *      `Mult(ones)` to extract the diagonal values, then walks
     *      its per-pair blocks to build the Schur diagonal.
     *   6. Build a fresh `DiagonalScaler` on the Schur diagonal
     *      and a fresh `BlockDiagonalPreconditioner` wiring
     *      `K_block_prec` for block 0 and the Schur scaler for
     *      block 1.
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

private:
    std::shared_ptr<mfem::Solver> m_K_block_prec;
    std::shared_ptr<mfem::Solver> m_K_jacobi_prec;
    const MortarConstraintOperator& m_C_op;

    // Rebuilt on each SetOperator() call:
    std::unique_ptr<DiagonalScaler> m_S_block_prec;
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> m_block_prec;
    mfem::Array<int> m_block_offsets;
};

}  // namespace mortar_pbc

#endif  // EXACONSTIT_MORTAR_PBC_SADDLE_PRECONDITIONER_HPP
