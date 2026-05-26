// Phase 4.3 / Batch R — Saddle-point system adapter.
//
// This file declares MortarSaddlePointSystem, which composes a user-
// provided mechanical operator K (linear or nonlinear) with the EA
// constraint operator C into a single mfem::Operator presenting the
// saddle-point system
//
//     [ K(u)   C^T ] [ u      ]   [ f - r_K(u) - C^T lambda ]
//     [ C      0   ] [ lambda ] = [ -C u                    ]
//
// to higher-level MFEM machinery (BlockOperator, Newton solver,
// Krylov methods).
//
// Why this exists:
//   - In the LINEAR case (current patch tests), the user can wire
//     up an mfem::BlockOperator manually with K (HypreParMatrix*)
//     in (0,0), MortarConstraintOperator in (1,0), and
//     mfem::TransposeOperator(C_op) in (0,1). No adapter needed.
//   - In the NONLINEAR case (ExaConstit production), K's Jacobian
//     dK/du changes per Newton iteration. The user has an
//     mfem::ParNonlinearForm or similar; this adapter:
//       (a) calls user's K-residual on Mult,
//       (b) calls user's K-Jacobian on GetGradient, packaging the
//           result with C / C^T into a fresh BlockOperator that
//           lives until the next GetGradient call.
//
// The adapter does NOT own K. It owns shared access to the mortar
// constraint operator, the wrapper machinery (BlockOperator,
// TransposeOperator), and an internal copy of the user's K-residual /
// K-Jacobian function objects.
//
// API contract:
//   - Inherits mfem::Operator with Height() = Width() = u_size +
//     lambda_size.
//   - Mult(x_block, r_block) computes the saddle-point residual:
//       r_K_block = K_residual(u)  + C^T lambda
//       r_C_block = C * u  -  g_constraint_rhs
//     Note no f subtraction here — the user includes f in their
//     KResidualFn closure (allows nonzero RHS without API churn).
//     `g_constraint_rhs` is the optional non-zero constraint RHS
//     installed via SetConstraintRHS (Phase 5.0). Default = no
//     RHS installed = zero, recovering the homogeneous-constraint
//     behavior (`r_C_block = C * u`).
//   - GetGradient(x_block) returns a BlockOperator& whose blocks
//     are (K_jacobian(u), C^T_op, C_op, zero). The constraint RHS
//     does NOT enter the Jacobian (it's an additive constant on
//     the residual side).
//
// What it does NOT do:
//   - No Newton solver. The user wraps this in mfem::NewtonSolver
//     or equivalent.
//   - No preconditioner construction. The user calls
//     C_op.ComputeInvDiagSchur and K's analogous diag-K^-1 method
//     (or BuildInvDiagK if K is HypreParMatrix) externally and
//     constructs a BlockDiagonalPreconditioner outside this class.
//
#pragma once

#include "mortar_constraint_operator.hpp"
#include "mfem.hpp"

#include <functional>
#include <memory>

namespace mortar_pbc {

/**
 * @brief Saddle-point system adapter combining a user-provided
 *        mechanical operator (linear or nonlinear) with the EA
 *        constraint operator into a single `mfem::Operator`.
 *
 * @details Block layout: `[u | lambda]`. Block offsets are
 * `[0, u_size, u_size + lambda_size]`.
 *
 * Residual semantics (Mult):
 *   `r_u     = K_residual(u) + C^T * lambda`
 *   `r_lam   = C * u  -  g_constraint_rhs`
 *
 * `g_constraint_rhs` is an optional vector installed via
 * `SetConstraintRHS` (Phase 5.0). Default = no RHS installed,
 * recovering the original homogeneous-constraint behavior
 * (`r_lam = C * u`). ExaConstit's `MortarPbcManager` installs a
 * non-zero `g_constraint_rhs` once per time step to encode the
 * macroscopic deformation rate (Method D, Phase 5 plan §P5.8.4.4).
 *
 * The user's `K_residual` callback is responsible for any
 * subtraction of an external load `f`; the adapter does not
 * touch it. This matches `mfem::ParNonlinearForm::Mult` semantics
 * (which already includes the load contribution if the form has
 * been told about it).
 *
 * Jacobian semantics (GetGradient):
 *   `J = [ K_jacobian(u)   C^T ]`
 *       `[ C               0   ]`
 *
 * Returned as a `BlockOperator&` referencing internal storage
 * that lives until the next `GetGradient` call. The
 * `K_jacobian(u)` is a non-owning pointer returned by the user's
 * callback — the adapter expects it to remain valid until the
 * next `GetGradient` call as well (typical pattern: the user's
 * `mfem::ParNonlinearForm` stores its current Jacobian internally
 * and returns a pointer to it).
 *
 * @par Phase 6 ownership
 * The primary constructor stores a `std::shared_ptr` to the
 * `MortarConstraintOperator`. This matches `MortarPbcManager`, which
 * owns the projector-aware operator behind shared ownership. The
 * reference constructor remains as a compatibility path and wraps the
 * reference in a non-owning aliasing `shared_ptr`; new production code
 * should pass the shared handle.
 */
class MortarSaddlePointSystem : public mfem::Operator
{
public:
    /// Compute `r_K = K(u)` (or `K(u) - f` if f is included
    /// in the closure). Result is the local FES TDOF slice.
    using KResidualFn = std::function<void(const mfem::Vector& u,
                                            mfem::Vector& r_K)>;

    /// Return a non-owning `mfem::Operator*` for `dK/du(u)`. Pointer
    /// must remain valid until the next call. For linear K, the
    /// closure typically just returns the same `&K` every time.
    using KJacobianFn = std::function<mfem::Operator*(const mfem::Vector& u)>;

    /**
     * @brief Construct the saddle-point system.
     *
     * @param k_residual    User's K-residual callback. See
     *                      `KResidualFn` for semantics.
     * @param k_jacobian    User's K-Jacobian callback. See
     *                      `KJacobianFn` for semantics.
     * @param C_op          Shared EA constraint operator. Must be
     *                      non-null. The adapter keeps the operator
     *                      alive for as long as the saddle system
     *                      exists.
     */
    MortarSaddlePointSystem(KResidualFn k_residual,
                            KJacobianFn k_jacobian,
                            std::shared_ptr<MortarConstraintOperator> C_op);

    /**
     * @brief Compatibility constructor from a non-owned constraint
     *        operator reference.
     *
     * @details Wraps `C_op` in a non-owning shared pointer and
     * delegates to the shared-handle constructor. This preserves
     * existing tests and legacy call sites while the Phase 6 manager
     * and preconditioner paths move to explicit shared ownership.
     *
     * @param C_op  Constraint operator reference. Caller must ensure
     *              it outlives this saddle system.
     */
    MortarSaddlePointSystem(KResidualFn k_residual,
                            KJacobianFn k_jacobian,
                            const MortarConstraintOperator& C_op);

    ~MortarSaddlePointSystem() override = default;

    MortarSaddlePointSystem(const MortarSaddlePointSystem&) = delete;
    MortarSaddlePointSystem& operator=(
        const MortarSaddlePointSystem&) = delete;

    /// Block-vector layout offsets: `[0, u_size, u_size + lambda_size]`.
    const mfem::Array<int>& BlockOffsets() const { return m_block_offsets; }

    /// Number of u-block entries (= local FES TDOFs).
    int NumU() const { return m_n_u; }

    /// Number of lambda-block entries (= local constraint rows).
    int NumLambda() const { return m_n_lam; }

    /**
     * @brief Install a non-zero constraint RHS for the saddle point.
     *
     * @details Phase 5.0 extension. After this call, `Mult` returns
     *   `r_C_block = C * u - g`
     * instead of the homogeneous form. The vector `g` must have
     * size `NumLambda()`; the adapter stores a NON-OWNING POINTER
     * to it, so `g` MUST OUTLIVE any subsequent `Mult` calls (and
     * any subsequent `GetGradient` calls — though `g` does not
     * appear in the Jacobian, the lifetime contract is symmetric
     * for safety).
     *
     * Production usage (ExaConstit's `MortarPbcManager`): call
     * once per time step with a buffer member that lives on the
     * manager. The buffer is refreshed each step before the
     * Newton solve via `MortarPbcManager::UpdateConstraintRHS`.
     *
     * Calling `SetConstraintRHS` multiple times simply replaces
     * the stored pointer; the previous `g` is no longer
     * referenced.
     *
     * @param g  Constraint RHS vector. `g.Size()` must equal
     *           `NumLambda()`. Lifetime: must outlive subsequent
     *           `Mult` / `GetGradient` calls.
     */
    void SetConstraintRHS(const mfem::Vector& g);

    /**
     * @brief Remove any installed constraint RHS, returning to the
     *        homogeneous default (`r_C_block = C * u`).
     *
     * @details Phase 5.0. After this call, `HasConstraintRHS()`
     * returns `false` and `Mult` ignores any previously-installed
     * `g`. Cheap (just nulls the pointer).
     */
    void ClearConstraintRHS();

    /**
     * @brief True iff a non-null constraint RHS is currently
     *        installed via `SetConstraintRHS`.
     *
     * @details Phase 5.0. Useful for diagnostics and for the unit
     * test that verifies the default state has no RHS.
     */
    bool HasConstraintRHS() const { return m_g_rhs != nullptr; }

    /**
     * @brief Compute saddle-point residual.
     *
     * @param x_block [in]  Block vector of size `Height()`. The
     *                       u-slice is `x_block[0..NumU())`; the
     *                       lambda-slice is `x_block[NumU()..)`.
     * @param r_block [out] Saddle-point residual, same layout.
     */
    void Mult(const mfem::Vector& x_block,
              mfem::Vector& r_block) const override;

    /**
     * @brief Return saddle-point Jacobian.
     *
     * @param x_block [in]  Full block vector at which to evaluate.
     *                      **Size must equal `Width()` (= `NumU() +
     *                      NumLambda()`)**, matching `Mult`'s input
     *                      size and the `mfem::Operator` interface
     *                      convention. The adapter extracts the
     *                      u-slice (`x_block[0..NumU())`) and
     *                      forwards it to the user's `KJacobianFn`;
     *                      the lambda-slice is unused (the
     *                      saddle-point Jacobian doesn't depend on
     *                      lambda since the (1,1) block is zero).
     * @return `BlockOperator&` referencing internal storage that
     *         lives until the next `GetGradient` call. Not safe
     *         to hold across calls.
     */
    mfem::Operator& GetGradient(const mfem::Vector& x_block) const override;

    /**
     * @brief Phase 5.9.A.5 — re-read block sizes from the underlying
     *        constraint operator after its filter spec changed.
     *
     * @details `MortarSaddlePointSystem`'s `m_n_u`, `m_n_lam`,
     * `height`, `width`, and `m_block_offsets` are set at ctor time
     * from `C_op.Width()` and `C_op.Height()`. The Phase 5.9.A.3.d
     * `MortarConstraintOperator::Reset` can change `C_op.Height()`
     * at runtime (when the active periodic-BC spec switches), so
     * this method must be called once after every `Reset` to keep
     * the saddle system's sizes in sync.
     *
     * The corresponding call in `MortarPbcManager::RebuildForActiveSpec`
     * (Phase 5.9.A.4) drives this: the manager owns both the
     * constraint operator and the saddle system, so it knows when
     * a refresh is needed.
     *
     * Local — no MPI calls. Idempotent if called more than once
     * without an intervening `Reset`.
     */
    void Refresh();

private:
    KResidualFn                          m_k_residual;
    KJacobianFn                          m_k_jacobian;
    std::shared_ptr<MortarConstraintOperator> m_C_op;

    // Block layout — fixed at construction time.
    int m_n_u;
    int m_n_lam;
    mfem::Array<int> m_block_offsets;

    // Per-call Jacobian storage (mutable because GetGradient is const
    // by MFEM convention but must update internal state). The
    // BlockOperator is rebuilt on each GetGradient call to point at
    // the latest K_jacobian(u). Members are `mutable` so the const
    // accessor can refresh them.
    mutable std::unique_ptr<mfem::TransposeOperator> m_C_T_op;
    mutable std::unique_ptr<mfem::BlockOperator>     m_block_op;

    // Phase 5.0 — optional constraint RHS pointer. Non-owning;
    // the supplied vector's storage must outlive subsequent Mult
    // calls (the typical pattern is for the upstream
    // MortarPbcManager to hold a buffer member that's refreshed
    // each time step). When non-null, `Mult` subtracts (*m_g_rhs)
    // from the constraint-side residual block, giving
    //     r_C_block = C * u - (*m_g_rhs)
    // instead of the homogeneous default
    //     r_C_block = C * u.
    // Default state (no RHS installed) recovers the original
    // Phase 4.3 behavior verbatim.
    const mfem::Vector* m_g_rhs = nullptr;
};

}  // namespace mortar_pbc
