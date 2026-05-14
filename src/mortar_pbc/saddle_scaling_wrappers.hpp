// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.D — saddle scaling wrappers (Op / Solver / Prec).
// Phase 5.11.H.2 — reusable-scratch + device-aware-copy fix.
//
// Four classes implement the apply-then-call-then-unapply pattern to
// route the Newton solver and the inner saddle Krylov through the
// scaled view of the saddle system without modifying the Newton
// solver's internals:
//
//   1. ScaledSaddleOperator        wraps  mfem::Operator    (e.g.
//                                          MortarSaddlePointSystem)
//   2. ScaledJacobianOperator      wraps  mfem::Operator    (the
//                                          Jacobian/BlockOperator
//                                          returned by inner op's
//                                          GetGradient)
//   3. ScaledSaddleSolver          wraps  mfem::Solver      (e.g.
//                                          SaddlePointSolver — the
//                                          inner outer linear solver)
//   4. ScaledSaddlePreconditioner  wraps  mfem::Solver      (e.g.
//                                          MortarSaddlePreconditioner)
//
// ---------------------------------------------------------------------------
// Convention (matches Phase 5.11.C SaddleResidualScaler):
//
//     r_solver  = D^-1 r_phys     (Apply direction: phys -> solver)
//     dx_solver = D^-1 dx_phys    (Apply direction: phys -> solver)
//     r_phys    = D r_solver      (Unapply direction: solver -> phys)
//     dx_phys   = D dx_solver     (Unapply direction: solver -> phys)
//
// Where D = diag(d_u I, D_lambda), D_lambda is piecewise-constant per
// sub-block (see Phase 5.11.C).
//
// The corresponding scaled operators:
//
//     J_solver = D^-1 J D                 (NOT symmetric)
//     P_solver = D^-1 P D
//
// satisfy:
//
//     J_solver dx_solver = -r_solver   <=>   J dx_phys = -r_phys
//
// so the scaled and physical Newton steps coincide for an exact solve.
// They differ for iterative Krylov: the scaling affects convergence
// path and tolerance interpretation.
//
// ---------------------------------------------------------------------------
// Newton solver flow with the wrappers (unchanged from non-scaled flow,
// only the operators / solvers are swapped):
//
//   1. op_scaled.Mult(u_phys, r_solver)               // scaled output
//   2. norm = Norm(r_solver)                           // scaled norm
//   3. if (norm < tol) break;
//   4. solver_scaled.SetOperator(op_scaled.GetGradient(u_phys))
//                                                      // sets J_solver
//                                                      // on inner solver
//   5. r_solver.Neg();
//   6. solver_scaled.Mult(r_solver, dx_phys)           // inner iterates
//                                                      // in scaled coords,
//                                                      // wrapper unapplies
//                                                      // to dx_phys
//   7. u_phys += dx_phys
//   8. goto 1.
//
// ---------------------------------------------------------------------------
// All four wrappers expose a `Refresh` hook that the MortarPbcManager
// (Phase 5.11.E) calls after a Phase 5.9 active-spec change to update
// internal shared_ptr handles and block offsets without breaking
// any external pointers held to the wrapper itself.
//
// ---------------------------------------------------------------------------
// Phase 5.11.H.2 — reusable scratch + device-aware copy
//
// The two wrappers that need intermediate physical-coords storage
// between an Unapply/Apply call and the inner Mult call —
// `ScaledJacobianOperator` (Mult AND MultTranspose) and
// `ScaledSaddlePreconditioner` (Mult) — now hold persistent member
// scratch buffers sized at construction (and resized in Refresh if
// the active-spec change resizes the lambda block). MINRES drives
// the wrapped Jacobian's Mult hundreds of times per Newton iter and
// thousands per simulation step; allocating a fresh
// `mfem::BlockVector(m_block_offsets)` per call is pure waste, and
// the per-call allocation also leaves the scratch's MFEM memory-
// manager flag state in an "uninitialized" condition that
// interacts badly with `Vector::operator=` from a MINRES work
// vector whose flag state has been set asymmetrically by upstream
// device-aware ops (the symptom previously seen as
// `MFEM abort: No device memory controller!` at
// `MemoryManager::Copy_ -> GetDevicePtr`).
//
// The copies between scratch and caller-owned input/output buffers
// now use the canonical MFEM device-aware idiom:
//
//   const double* s = src.Read();
//   double*       d = dst.Write();
//   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { d[i] = s[i]; });
//
// where `Write()` is called on the dst's `BlockVector` view (not
// directly on the underlying storage member) so the view's flag
// state — which is what subsequent `m_scaler->Apply*/Unapply*`
// calls consult through `BlockVector::GetBlock(i).Read()` — is
// marked coherently as VALID_HOST/VALID_DEVICE matching the active
// `mfem::Device` backend. The `ScaledSaddleOperator` and
// `ScaledSaddleSolver` Mult paths are already in-place (they pass
// the caller's output buffer as the inner op's output and then run
// `m_scaler->Apply/Unapply` on a `BlockVector::Update` view) so no
// scratch is needed for them.

#pragma once

#include "saddle_residual_scaler.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc
{

//==============================================================================
// ScaledJacobianOperator
//==============================================================================

/**
 * @brief Wraps a physical Jacobian operator to present the scaled
 *        view J_solver = D^-1 J D.
 *
 * @details Typically constructed by `ScaledSaddleOperator::GetGradient`
 * and handed to the inner saddle Krylov (via its `SetOperator`). The
 * Krylov then iterates in scaled coords. The wrapper holds a
 * non-owning pointer to the inner Jacobian (whose lifetime is managed
 * by the inner operator that returned it from GetGradient).
 *
 * @par Math
 *
 *   Mult:           J_solver v = D^-1 J D v
 *                   steps:  w = D v   (Unapply input)
 *                           w' = J w  (inner.Mult)
 *                           y  = D^-1 w'  (Apply output)
 *
 *   MultTranspose:  J_solver^T v = (D^-1 J D)^T v = D J^T D^-1 v
 *                   steps:  w = D^-1 v   (Apply input)
 *                           w' = J^T w   (inner.MultTranspose)
 *                           y  = D w'    (Unapply output)
 *
 * Note the direction asymmetry: Mult unapplies-then-applies; MultTranspose
 * applies-then-unapplies. This is correct for non-symmetric D-J products.
 *
 * @par Reusable scratch (Phase 5.11.H.2)
 * The class owns a single `mfem::BlockVector` view (`m_scratch_view`)
 * over a backing `mfem::Vector` storage (`m_scratch_storage`), both
 * sized at construction and resized in `Refresh` if the active-spec
 * change resizes the lambda block. Both Mult and MultTranspose reuse
 * the same scratch for the intermediate physical-space vector
 * `w` (since Mult and MultTranspose are never called concurrently).
 * The output buffer (`Jv_solver` / `JTv_solver`) is written
 * in-place by `inner.Mult` via a stack-local `BlockVector::Update`
 * view; the final scaler call mutates that view in-place — no
 * second scratch needed, no terminal `Vector::operator=` copy.
 */
class ScaledJacobianOperator : public mfem::Operator
{
public:
    /**
     * @brief Construct from a non-owning reference to an inner
     *        Jacobian operator and a scaler.
     *
     * @param inner_jac     Reference to the physical Jacobian.
     *                      Must outlive this wrapper (typically the
     *                      caller is `ScaledSaddleOperator::GetGradient`
     *                      whose owner manages the inner Jacobian's
     *                      lifetime).
     * @param scaler        Shared ownership of the scaler. Scaler's
     *                      `Choose` is driven externally by the manager.
     * @param block_offsets Saddle block offsets [0, n_u, n_u + n_lam].
     *
     * @details At construction, allocates `m_scratch_storage` of size
     * `block_offsets.Last()` using `mfem::Device::GetMemoryType()`,
     * marks it `UseDevice(true)`, and `Update`s `m_scratch_view` over
     * it. The scratch is therefore ready for device-aware writes on
     * first call to `Mult` / `MultTranspose`.
     */
    ScaledJacobianOperator(
        mfem::Operator& inner_jac,
        std::shared_ptr<const SaddleResidualScaler> scaler,
        const mfem::Array<int>& block_offsets);

    ~ScaledJacobianOperator() override = default;

    ScaledJacobianOperator(const ScaledJacobianOperator&) = delete;
    ScaledJacobianOperator& operator=(const ScaledJacobianOperator&) = delete;

    void Mult(const mfem::Vector& v_solver,
              mfem::Vector& Jv_solver) const override;
    void MultTranspose(const mfem::Vector& v_solver,
                        mfem::Vector& JTv_solver) const override;

    /// Accessor for the wrapped physical Jacobian, used by
    /// `ScaledSaddlePreconditioner::SetOperator` to forward the
    /// physical operator into the inner prec's setup.
    mfem::Operator& GetUnscaled() const { return *m_inner_jac; }

    /// Replace the inner Jacobian pointer and update sizes. Called
    /// from `ScaledSaddleOperator::GetGradient` on each call. If
    /// `new_block_offsets.Last()` differs from the current scratch
    /// size, the scratch is resized and re-bound; otherwise the
    /// scratch is reused as-is.
    void Refresh(mfem::Operator& new_inner_jac,
                  const mfem::Array<int>& new_block_offsets);

private:
    mfem::Operator*                             m_inner_jac;
    std::shared_ptr<const SaddleResidualScaler> m_scaler;
    mfem::Array<int>                            m_block_offsets;

    // Phase 5.11.H.2 — reusable scratch for intermediate
    // physical-coords vector in Mult / MultTranspose.
    //
    // m_scratch_storage owns the bytes (sized at construction with
    // mfem::Device::GetMemoryType + UseDevice(true)). m_scratch_view
    // is a BlockVector::Update view over it; writing through the
    // view marks ITS flag state coherent for subsequent scaler
    // GetBlock(i).Read() calls. `mutable` because the public Mult /
    // MultTranspose are const but the scratch is per-instance
    // workspace, not logical state.
    mutable mfem::Vector       m_scratch_storage;
    mutable mfem::BlockVector  m_scratch_view;
};

//==============================================================================
// ScaledSaddleOperator
//==============================================================================

/**
 * @brief Wraps a saddle residual operator to scale residual output.
 *
 * @details Wraps an inner `mfem::Operator` (typically
 * `MortarSaddlePointSystem`). The wrapper:
 *
 *   - `Mult(u_phys, y)` computes `y = D^-1 (inner.Mult(u_phys))`.
 *     The Newton solver thus sees a scaled residual without itself
 *     knowing about scaling.
 *   - `GetGradient(u_phys)` returns a `ScaledJacobianOperator` that
 *     wraps the inner Jacobian to present the scaled view J_solver.
 *
 * The Newton state stays in physical coords throughout. Only the
 * residual the Newton solver sees and the Jacobian the inner Krylov
 * sees are scaled.
 *
 * @par No scratch
 * Mult is implemented in-place: `inner.Mult` writes directly into
 * the caller's `r_solver` buffer; a stack-local
 * `BlockVector::Update` view over `r_solver` then has
 * `ApplyToResidual` applied in-place. No allocated scratch needed.
 */
class ScaledSaddleOperator : public mfem::Operator
{
public:
    /**
     * @param inner_op      Shared ownership of the inner saddle operator.
     * @param scaler        Shared ownership of the scaler.
     * @param block_offsets Saddle block offsets.
     */
    ScaledSaddleOperator(
        std::shared_ptr<mfem::Operator> inner_op,
        std::shared_ptr<const SaddleResidualScaler> scaler,
        const mfem::Array<int>& block_offsets);

    ~ScaledSaddleOperator() override = default;

    ScaledSaddleOperator(const ScaledSaddleOperator&) = delete;
    ScaledSaddleOperator& operator=(const ScaledSaddleOperator&) = delete;

    /// Mult: r_solver = D^-1 (inner_op.Mult(u_phys)).
    void Mult(const mfem::Vector& u_phys,
              mfem::Vector& r_solver) const override;

    /// GetGradient: returns a `ScaledJacobianOperator` wrapping
    /// `inner_op.GetGradient(u_phys)`. The returned reference is
    /// valid until the next call to GetGradient or to Refresh.
    mfem::Operator& GetGradient(const mfem::Vector& u_phys) const override;

    /**
     * @brief Refresh the inner operator pointer and block offsets.
     *
     * @details Called by `MortarPbcManager::RebuildForActiveSpec`
     * after a Phase 5.9 spec change rebuilds the inner saddle
     * operator (and possibly resizes the lambda block). The
     * previously-returned `ScaledJacobianOperator` reference is
     * invalidated.
     */
    void Refresh(std::shared_ptr<mfem::Operator> new_inner_op,
                 const mfem::Array<int>& new_block_offsets);

    /// Accessors for testing / introspection.
    mfem::Operator&                            GetInner()   const { return *m_inner_op; }
    const SaddleResidualScaler&                GetScaler()  const { return *m_scaler;   }
    const mfem::Array<int>&                    GetOffsets() const { return m_block_offsets; }

private:
    std::shared_ptr<mfem::Operator>                 m_inner_op;
    std::shared_ptr<const SaddleResidualScaler>     m_scaler;
    mfem::Array<int>                                m_block_offsets;
    mutable std::unique_ptr<ScaledJacobianOperator> m_scaled_jac;
};

//==============================================================================
// ScaledSaddleSolver
//==============================================================================

/**
 * @brief Wraps a saddle linear solver. Output is dx_phys.
 *
 * @details The Newton solver calls `solver.Mult(r_solver_neg, dx)` to
 * solve one Newton step. Inside this wrapper:
 *
 *   1. The inner saddle solver iterates in scaled coords using the
 *      scaled Jacobian (passed through `SetOperator`).
 *   2. The wrapper unapplies (multiplies by D) the resulting
 *      `dx_solver` to produce `dx_phys` for Newton's update.
 *
 * `SetOperator` forwards the SCALED Jacobian to the inner solver —
 * the inner is set up to iterate in scaled coords. Within the inner
 * solver, the preconditioner is a `ScaledSaddlePreconditioner`
 * which unwraps the scaled Jacobian when its own `SetOperator` fires.
 *
 * @par No scratch
 * Mult is in-place: `inner.Mult` writes directly into the caller's
 * `dx_phys` buffer; a stack-local `BlockVector::Update` view over
 * `dx_phys` then has `UnapplyToIncrement` applied in-place.
 */
class ScaledSaddleSolver : public mfem::Solver
{
public:
    ScaledSaddleSolver(
        std::shared_ptr<mfem::Solver> inner_solver,
        std::shared_ptr<const SaddleResidualScaler> scaler,
        const mfem::Array<int>& block_offsets);

    ~ScaledSaddleSolver() override = default;

    ScaledSaddleSolver(const ScaledSaddleSolver&) = delete;
    ScaledSaddleSolver& operator=(const ScaledSaddleSolver&) = delete;

    /// Mult: takes b_solver (= -r_solver from Newton), returns dx_phys.
    /// Inner solver iterates in scaled coords. Wrapper unapplies on output.
    void Mult(const mfem::Vector& b_solver,
              mfem::Vector& dx_phys) const override;

    /// SetOperator forwards to inner — the operator is the SCALED Jacobian
    /// (typically a `ScaledJacobianOperator` returned by
    /// `ScaledSaddleOperator::GetGradient`).
    void SetOperator(const mfem::Operator& op) override;

    /// Refresh inner solver pointer and offsets after Phase 5.9 spec changes.
    void Refresh(std::shared_ptr<mfem::Solver> new_inner_solver,
                 const mfem::Array<int>& new_block_offsets);

    /// Accessors.
    mfem::Solver&             GetInner()   const { return *m_inner_solver; }
    const mfem::Array<int>&   GetOffsets() const { return m_block_offsets; }

private:
    std::shared_ptr<mfem::Solver>                m_inner_solver;
    std::shared_ptr<const SaddleResidualScaler>  m_scaler;
    mfem::Array<int>                             m_block_offsets;
};

//==============================================================================
// ScaledSaddlePreconditioner
//==============================================================================

/**
 * @brief Wraps a saddle preconditioner for use inside the scaled-coord
 *        Krylov.
 *
 * @details The inner saddle Krylov iterates in scaled coords with the
 * scaled Jacobian J_solver. Its preconditioner needs to act
 * consistently: P_solver^-1 r_solver = (D^-1 P D)^-1 r_solver
 *                                    = D^-1 P^-1 D r_solver.
 *
 * Mult steps:
 *   1. r_phys = D r_solver           (Unapply input, into scratch)
 *   2. z_phys = inner_prec.Mult(r_phys)   (writes into z_solver buffer
 *                                          via BlockVector::Update view)
 *   3. z_solver = D^-1 z_phys        (Apply output, in-place on view)
 *
 * `SetOperator` is called by the Krylov when the Jacobian changes
 * (typically once per Newton iter). The Krylov passes the SCALED
 * Jacobian. The wrapper unwraps it (via `ScaledJacobianOperator::GetUnscaled`)
 * to recover the physical Jacobian and forwards that to the inner
 * prec. This works because the inner prec (e.g.
 * MortarSaddlePreconditioner) is built to consume the physical
 * BlockOperator — it extracts K from block (0,0), computes the
 * Schur diagonal, etc.
 *
 * @par Reusable scratch (Phase 5.11.H.2)
 * Same pattern as `ScaledJacobianOperator`: a single
 * `mfem::BlockVector` view (`m_scratch_view`) over backing storage
 * (`m_scratch_storage`), allocated at construction, resized in
 * `Refresh` if needed. Eliminates per-call allocation across the
 * many Krylov inner iterations that fire `Mult` per Newton iter.
 */
class ScaledSaddlePreconditioner : public mfem::Solver
{
public:
    ScaledSaddlePreconditioner(
        std::shared_ptr<mfem::Solver> inner_prec,
        std::shared_ptr<const SaddleResidualScaler> scaler,
        const mfem::Array<int>& block_offsets);

    ~ScaledSaddlePreconditioner() override = default;

    ScaledSaddlePreconditioner(const ScaledSaddlePreconditioner&) = delete;
    ScaledSaddlePreconditioner& operator=(
        const ScaledSaddlePreconditioner&) = delete;

    /// Mult: z_solver = D^-1 P^-1 D r_solver.
    void Mult(const mfem::Vector& r_solver,
              mfem::Vector& z_solver) const override;

    /// SetOperator: unwraps the incoming `ScaledJacobianOperator` and
    /// forwards the physical Jacobian to inner_prec.
    void SetOperator(const mfem::Operator& op) override;

    /// Refresh inner prec pointer and offsets after Phase 5.9 spec changes.
    /// Resizes the member scratch if `new_block_offsets.Last()` differs.
    void Refresh(std::shared_ptr<mfem::Solver> new_inner_prec,
                 const mfem::Array<int>& new_block_offsets);

    /// Accessors.
    mfem::Solver&             GetInner()   const { return *m_inner_prec; }
    const mfem::Array<int>&   GetOffsets() const { return m_block_offsets; }

private:
    std::shared_ptr<mfem::Solver>                m_inner_prec;
    std::shared_ptr<const SaddleResidualScaler>  m_scaler;
    mfem::Array<int>                             m_block_offsets;

    // Phase 5.11.H.2 — reusable scratch for the intermediate
    // physical-coords input vector (post-Unapply, pre-inner-Mult).
    // See ScaledJacobianOperator's note for sizing semantics.
    mutable mfem::Vector       m_scratch_storage;
    mutable mfem::BlockVector  m_scratch_view;
};

}   // namespace mortar_pbc
