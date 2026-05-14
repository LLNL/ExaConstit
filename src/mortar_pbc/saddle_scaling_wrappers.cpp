// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.D — saddle scaling wrappers implementation.
// Phase 5.11.H.2 — reusable scratch + device-aware copy fix.
//
// See header for full design notes and math. Each wrapper's Mult /
// MultTranspose follows the apply-then-call-then-unapply pattern,
// with directions chosen per the scaling semantics:
//
//   - Operator     :  inner produces a physical residual    -> Apply (divide)
//   - JacobianOp   :  Mult unapplies-then-applies (J_solver = D^-1 J D)
//                  :  MultTranspose applies-then-unapplies (J_solver^T = D J^T D^-1)
//   - LinearSolver :  inner produces a scaled increment      -> Unapply (multiply)
//   - Preconditioner: inner consumes physical, produces physical
//                  :  Unapply input, Apply output
//
// ---------------------------------------------------------------------------
// Phase 5.11.H.2 fix details:
//
// The original 5.11.D implementation used `mfem::BlockVector w_phys`
// stack-locals constructed per call, with `static_cast<mfem::Vector&>(bv)
// = src` to copy data into them. Two problems:
//
//   (1) Per-call allocation cost. MINRES drives the wrapped Jacobian's
//       Mult hundreds of times per Newton iter, thousands per sim
//       step. Each call allocated fresh BlockVector storage of size
//       `m_block_offsets.Last()` and freed it on return.
//
//   (2) Asymmetric flag-state in `Vector::operator=`. The src vector
//       (a MINRES work vector) arrives with `VALID_DEVICE | USE_DEVICE`
//       set but `VALID_HOST` unset because upstream MINRES ops have
//       routed through device-aware Read/Write paths. The freshly-
//       constructed dst BlockVector has no valid flags set. MFEM's
//       `MemoryManager::Copy_` then sees src VALID_DEVICE without
//       VALID_HOST and tries to access src's device pointer to copy
//       device-to-host, which aborts if the linked MFEM has no
//       device backend registered (`No device memory controller!`
//       at `mem_manager.cpp:803`).
//
// Both problems are solved by the same change: keep persistent
// scratch members (sized at construction, reused per call) and
// perform the src->scratch copy via the canonical MFEM device-aware
// idiom:
//
//   const double* s = src.Read();
//   double*       d = static_cast<mfem::Vector&>(scratch_view).Write();
//   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { d[i] = s[i]; });
//
// `forall` dispatches the loop to the active mfem::Device backend
// (HIP / CUDA / host). `Read()` and `Write()` route through the dst
// view's USE_DEVICE flag (which we set at construction time to
// match `Device::GetMemoryType`) — not through src's flag state.
// The dst view's flag state is marked coherently after the Write,
// which means subsequent `m_scaler->Apply*/Unapply*` calls — which
// internally do `bv.GetBlock(i).Read()` — see VALID_HOST/VALID_DEVICE
// matching the active backend and never trigger an
// implicit cross-space sync.
//
// In addition, the output copies (`Jv_solver = y_phys` etc.) are
// eliminated entirely: we pass `inner.Mult` an output that is itself
// a `BlockVector::Update` view over the caller's output buffer, so
// the inner op writes its result directly into `Jv_solver`'s memory.
// The terminal scaler call then operates on that view in-place. One
// scratch buffer per wrapper; zero terminal copies.

#include "saddle_scaling_wrappers.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <memory>
#include <utility>

namespace mortar_pbc
{

namespace
{

//==============================================================================
// Device-aware element-wise copy: dst[i] = src[i].
//
// Replaces the `Vector::operator=` / `Memory::CopyFrom` /
// `MemoryManager::Copy_` path that was hitting "No device memory
// controller!" on the saddle-scaling code path under linked-CPU
// MFEM with `Device::UseDevice() == true` on src.
//
// Semantics:
//   - `src.Read()` returns a const pointer in src's preferred space
//     (HOST or DEVICE per Device::GetDeviceMemoryClass). On a
//     correctly-configured CPU build (Device::IsEnabled() == false),
//     this is always a host pointer.
//   - `dst.Write()` returns a writable pointer in dst's preferred
//     space and marks dst's flag state as VALID in that space
//     (clearing the other validity flag). NO sync from device to
//     host or vice versa is required because Write_ does not
//     validate — it assumes the caller is about to overwrite.
//   - `mfem::forall` dispatches the loop to the active backend.
//
// Caller responsibility:
//   - src.Size() must equal dst.Size().
//   - dst must be a writable Vector (not const).
//==============================================================================
inline void CopyVectorDeviceAware(const mfem::Vector& src,
                                   mfem::Vector& dst)
{
    MFEM_ASSERT(src.Size() == dst.Size(),
                "CopyVectorDeviceAware: size mismatch (src="
                << src.Size() << ", dst=" << dst.Size() << ")");

    const int     N = src.Size();
    const double* s = src.Read();
    double*       d = dst.Write();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { d[i] = s[i]; });
}

//==============================================================================
// Construct a BlockVector that shares storage with an existing Vector,
// laid out per the given block offsets. The returned BlockVector does
// not own memory; modifications to it modify the underlying Vector.
//
// Used by the in-place wrappers (ScaledSaddleOperator,
// ScaledSaddleSolver) to give the scaler's Apply/Unapply methods
// (which take BlockVector&) access to data passed in as Vector&.
//
// The `const_cast` overload is safe in the calling contexts: those
// callers either hold a mutable copy or have a mutable Vector
// elsewhere up the stack; the returned view's mutations do not
// propagate through the const overload back to the original src
// because we never use this overload to mutate.
//==============================================================================
mfem::BlockVector MakeBlockView(const mfem::Vector& src,
                                 const mfem::Array<int>& offsets)
{
    mfem::BlockVector v;
    v.Update(const_cast<mfem::Vector&>(src), offsets);
    return v;
}

mfem::BlockVector MakeBlockView(mfem::Vector& src,
                                 const mfem::Array<int>& offsets)
{
    mfem::BlockVector v;
    v.Update(src, offsets);
    return v;
}

//==============================================================================
// Helper: (re)size and re-Update the scratch storage + view to match
// the given block_offsets. Idempotent — if the total size is
// unchanged, only the view is re-Update'd (cheap pointer rebind);
// otherwise the storage is reallocated under the active device
// memory type and the view re-bound over it.
//==============================================================================
inline void EnsureScratchSized(mfem::Vector& storage,
                                mfem::BlockVector& view,
                                const mfem::Array<int>& offsets)
{
    const int total = offsets.Last();
    if (storage.Size() != total)
    {
        storage.SetSize(total, mfem::Device::GetMemoryType());
        storage.UseDevice(true);
    }
    // Rebind the view to (possibly-new) storage and (possibly-new) offsets.
    view.Update(storage, offsets);
}

}   // anonymous namespace

//==============================================================================
// ScaledJacobianOperator
//==============================================================================

ScaledJacobianOperator::ScaledJacobianOperator(
    mfem::Operator& inner_jac,
    std::shared_ptr<const SaddleResidualScaler> scaler,
    const mfem::Array<int>& block_offsets)
    : mfem::Operator(inner_jac.Height(), inner_jac.Width()),
      m_inner_jac(&inner_jac),
      m_scaler(std::move(scaler)),
      m_block_offsets(block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_jacobian_op::ctor");
    MFEM_VERIFY(m_scaler,
                "ScaledJacobianOperator: scaler must not be null");
    MFEM_VERIFY(m_block_offsets.Size() >= 2,
                "ScaledJacobianOperator: block_offsets must have at "
                "least one block (size >= 2)");
    MFEM_VERIFY(m_block_offsets.Last() == inner_jac.Height(),
                "ScaledJacobianOperator: block_offsets.Last() ("
                << m_block_offsets.Last() << ") must equal "
                "inner_jac.Height() (" << inner_jac.Height() << ")");

    // Phase 5.11.H.2 — allocate the reusable scratch up front.
    EnsureScratchSized(m_scratch_storage, m_scratch_view, m_block_offsets);
}

void ScaledJacobianOperator::Mult(const mfem::Vector& v_solver,
                                    mfem::Vector& Jv_solver) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_jacobian_op::mult");

    MFEM_ASSERT(v_solver.Size() == width,
                "ScaledJacobianOperator::Mult: v_solver size mismatch ("
                << v_solver.Size() << " vs " << width << ")");
    MFEM_ASSERT(Jv_solver.Size() == height,
                "ScaledJacobianOperator::Mult: Jv_solver size mismatch ("
                << Jv_solver.Size() << " vs " << height << ")");

    // J_solver v_solver = D^-1 J D v_solver
    //   stage 1: w_phys = D v_solver        (Unapply input)
    //   stage 2: y_phys = inner.Mult(w_phys) -- written directly into Jv buffer
    //   stage 3: Jv_solver = D^-1 y_phys     (Apply output, in-place)

    // Stage 1 — copy v_solver into the reusable scratch view via the
    // canonical device-aware idiom (replaces the 5.11.D
    // `static_cast<Vector&>(w_phys) = v_solver` that was routing
    // through `MemoryManager::Copy_` and hitting the missing-device-
    // controller abort). Writing through the BlockVector view
    // marks the view's flag state coherently for the subsequent
    // scaler call.
    CopyVectorDeviceAware(v_solver,
                          static_cast<mfem::Vector&>(m_scratch_view));
    m_scaler->UnapplyToIncrement(m_scratch_view);       // *= D

    // Stage 2 — inner.Mult writes directly into Jv_solver's buffer
    // via a stack-local BlockVector::Update view. No allocation,
    // no copy.
    mfem::BlockVector Jv_view = MakeBlockView(Jv_solver, m_block_offsets);
    m_inner_jac->Mult(m_scratch_view, Jv_view);

    // Stage 3 — apply scaler in-place on the output buffer (via
    // the view, which aliases Jv_solver's memory).
    m_scaler->ApplyToResidual(Jv_view);                 // /= D
}

void ScaledJacobianOperator::MultTranspose(
    const mfem::Vector& v_solver, mfem::Vector& JTv_solver) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_jacobian_op::mult_transpose");

    MFEM_ASSERT(v_solver.Size() == height,
                "ScaledJacobianOperator::MultTranspose: v_solver size mismatch ("
                << v_solver.Size() << " vs " << height << ")");
    MFEM_ASSERT(JTv_solver.Size() == width,
                "ScaledJacobianOperator::MultTranspose: JTv_solver size mismatch ("
                << JTv_solver.Size() << " vs " << width << ")");

    // J_solver^T v = (D^-1 J D)^T v = D J^T D^-1 v
    //   stage 1: w_phys = D^-1 v             (Apply input)
    //   stage 2: y_phys = inner.MultTranspose(w_phys) -- into JTv buffer
    //   stage 3: JTv_solver = D y_phys        (Unapply output, in-place)
    //
    // Note the direction asymmetry vs Mult: this branch applies
    // (divides) on input and unapplies (multiplies) on output, the
    // reverse of Mult.

    // Stage 1 — same reusable-scratch + device-aware copy pattern
    // as Mult. The scratch is reused across Mult and MultTranspose
    // calls (they never run concurrently).
    CopyVectorDeviceAware(v_solver,
                          static_cast<mfem::Vector&>(m_scratch_view));
    m_scaler->ApplyToIncrement(m_scratch_view);         // /= D

    // Stage 2 — inner.MultTranspose writes into JTv_solver via view.
    mfem::BlockVector JTv_view = MakeBlockView(JTv_solver, m_block_offsets);
    m_inner_jac->MultTranspose(m_scratch_view, JTv_view);

    // Stage 3 — unapply in-place on the output view.
    m_scaler->UnapplyToIncrement(JTv_view);             // *= D
}

void ScaledJacobianOperator::Refresh(
    mfem::Operator& new_inner_jac,
    const mfem::Array<int>& new_block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_jacobian_op::refresh");

    m_inner_jac = &new_inner_jac;
    m_block_offsets = new_block_offsets;
    height = new_inner_jac.Height();
    width = new_inner_jac.Width();

    MFEM_VERIFY(m_block_offsets.Last() == new_inner_jac.Height(),
                "ScaledJacobianOperator::Refresh: block_offsets.Last() ("
                << m_block_offsets.Last() << ") must equal "
                "new_inner_jac.Height() (" << new_inner_jac.Height() << ")");

    // Phase 5.11.H.2 — resize scratch if the lambda block changed
    // size under the new active spec; otherwise just rebind the
    // view to the new offsets (cheap pointer rebind).
    EnsureScratchSized(m_scratch_storage, m_scratch_view, m_block_offsets);
}

//==============================================================================
// ScaledSaddleOperator
//==============================================================================

ScaledSaddleOperator::ScaledSaddleOperator(
    std::shared_ptr<mfem::Operator> inner_op,
    std::shared_ptr<const SaddleResidualScaler> scaler,
    const mfem::Array<int>& block_offsets)
    : mfem::Operator(inner_op->Height(), inner_op->Width()),
      m_inner_op(std::move(inner_op)),
      m_scaler(std::move(scaler)),
      m_block_offsets(block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_op::ctor");
    MFEM_VERIFY(m_inner_op,
                "ScaledSaddleOperator: inner_op must not be null");
    MFEM_VERIFY(m_scaler,
                "ScaledSaddleOperator: scaler must not be null");
}

void ScaledSaddleOperator::Mult(const mfem::Vector& u_phys,
                                  mfem::Vector& r_solver) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_op::mult");

    MFEM_ASSERT(u_phys.Size() == width,
                "ScaledSaddleOperator::Mult: u_phys size mismatch");
    MFEM_ASSERT(r_solver.Size() == height,
                "ScaledSaddleOperator::Mult: r_solver size mismatch");

    // Inner.Mult writes directly into r_solver buffer (the inner op
    // already produces a physical residual). We then build a
    // BlockVector view over r_solver and apply the scaler in-place
    // — no scratch, no copy.
    //
    // Note: the inner.Mult call internally uses Read/Write on
    // u_phys and r_solver, so flag state on those is the inner op's
    // concern, not ours. The view we build for the scaler call
    // shares r_solver's memory, so subsequent `bv.GetBlock(i).Read()`
    // inside the scaler sees the flag state that inner.Mult's Write
    // left behind — which is coherent.
    m_inner_op->Mult(u_phys, r_solver);

    mfem::BlockVector r_view = MakeBlockView(r_solver, m_block_offsets);
    m_scaler->ApplyToResidual(r_view);                  // r_solver = D^-1 r_phys
}

mfem::Operator& ScaledSaddleOperator::GetGradient(
    const mfem::Vector& u_phys) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_op::get_gradient");

    mfem::Operator& inner_jac = m_inner_op->GetGradient(u_phys);

    if (!m_scaled_jac)
    {
        m_scaled_jac = std::make_unique<ScaledJacobianOperator>(
            inner_jac, m_scaler, m_block_offsets);
    }
    else
    {
        // Update the existing wrapper to reference the new inner
        // Jacobian and current offsets. Reusing the same object
        // keeps external references stable (e.g., the inner
        // solver may have cached the operator pointer from a
        // previous call). Refresh internally re-sizes the
        // scratch if the offsets changed.
        m_scaled_jac->Refresh(inner_jac, m_block_offsets);
    }

    return *m_scaled_jac;
}

void ScaledSaddleOperator::Refresh(
    std::shared_ptr<mfem::Operator> new_inner_op,
    const mfem::Array<int>& new_block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_op::refresh");

    MFEM_VERIFY(new_inner_op,
                "ScaledSaddleOperator::Refresh: new_inner_op must not be null");
    m_inner_op = std::move(new_inner_op);
    m_block_offsets = new_block_offsets;
    height = m_inner_op->Height();
    width = m_inner_op->Width();
    // Drop the cached scaled-Jacobian wrapper — it would otherwise
    // reference the old inner Jacobian. Next GetGradient call will
    // construct a fresh wrapper (whose own ctor sizes its scratch).
    m_scaled_jac.reset();
}

//==============================================================================
// ScaledSaddleSolver
//==============================================================================

ScaledSaddleSolver::ScaledSaddleSolver(
    std::shared_ptr<mfem::Solver> inner_solver,
    std::shared_ptr<const SaddleResidualScaler> scaler,
    const mfem::Array<int>& block_offsets)
    : mfem::Solver(inner_solver->Height(), inner_solver->Width()),
      m_inner_solver(std::move(inner_solver)),
      m_scaler(std::move(scaler)),
      m_block_offsets(block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_solver::ctor");
    MFEM_VERIFY(m_inner_solver,
                "ScaledSaddleSolver: inner_solver must not be null");
    MFEM_VERIFY(m_scaler,
                "ScaledSaddleSolver: scaler must not be null");
}

void ScaledSaddleSolver::Mult(const mfem::Vector& b_solver,
                                mfem::Vector& dx_phys) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_solver::mult");

    MFEM_ASSERT(b_solver.Size() == height,
                "ScaledSaddleSolver::Mult: b_solver size mismatch");
    MFEM_ASSERT(dx_phys.Size() == width,
                "ScaledSaddleSolver::Mult: dx_phys size mismatch");

    // Inner solver iterates J_solver dx_solver = b_solver in scaled
    // coords, returns dx_solver. We unapply (multiply by D) in
    // place to give Newton dx_phys.
    //
    // No scratch needed: inner.Mult writes directly into dx_phys's
    // memory; the BlockVector view shares that memory and the
    // unapply mutates it in-place.
    // Preserve the caller's iterative/non-iterative solve contract
    // across the wrapper boundary. Without this, the underlying
    // Krylov may reuse stale / uninitialized `dx_phys` contents as an
    // initial guess when the outer Newton solver intended a zero
    // start.
    m_inner_solver->iterative_mode = iterative_mode;
    m_inner_solver->Mult(b_solver, dx_phys);            // dx_phys buffer now
                                                         // holds dx_solver
    mfem::BlockVector dx_view = MakeBlockView(dx_phys, m_block_offsets);
    m_scaler->UnapplyToIncrement(dx_view);              // dx_phys = D dx_solver
}

void ScaledSaddleSolver::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_solver::set_operator");
    // `op` is the SCALED Jacobian (typically ScaledJacobianOperator).
    // The inner solver iterates in scaled coords and consumes the
    // scaled Jacobian directly.
    m_inner_solver->SetOperator(op);
    height = op.Height();
    width = op.Width();
}

void ScaledSaddleSolver::Refresh(
    std::shared_ptr<mfem::Solver> new_inner_solver,
    const mfem::Array<int>& new_block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_solver::refresh");
    MFEM_VERIFY(new_inner_solver,
                "ScaledSaddleSolver::Refresh: new_inner_solver must not be null");
    m_inner_solver = std::move(new_inner_solver);
    m_block_offsets = new_block_offsets;
    height = m_inner_solver->Height();
    width = m_inner_solver->Width();
}

//==============================================================================
// ScaledSaddlePreconditioner
//==============================================================================

ScaledSaddlePreconditioner::ScaledSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> inner_prec,
    std::shared_ptr<const SaddleResidualScaler> scaler,
    const mfem::Array<int>& block_offsets)
    : mfem::Solver(inner_prec->Height(), inner_prec->Width()),
      m_inner_prec(std::move(inner_prec)),
      m_scaler(std::move(scaler)),
      m_block_offsets(block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_prec::ctor");
    MFEM_VERIFY(m_inner_prec,
                "ScaledSaddlePreconditioner: inner_prec must not be null");
    MFEM_VERIFY(m_scaler,
                "ScaledSaddlePreconditioner: scaler must not be null");
    MFEM_VERIFY(m_block_offsets.Size() >= 2,
                "ScaledSaddlePreconditioner: block_offsets must have at "
                "least one block (size >= 2)");

    // Phase 5.11.H.2 — allocate the reusable scratch up front.
    EnsureScratchSized(m_scratch_storage, m_scratch_view, m_block_offsets);
}

void ScaledSaddlePreconditioner::Mult(const mfem::Vector& r_solver,
                                       mfem::Vector& z_solver) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_prec::mult");

    MFEM_ASSERT(r_solver.Size() == height,
                "ScaledSaddlePreconditioner::Mult: r_solver size mismatch ("
                << r_solver.Size() << " vs " << height << ")");
    MFEM_ASSERT(z_solver.Size() == width,
                "ScaledSaddlePreconditioner::Mult: z_solver size mismatch ("
                << z_solver.Size() << " vs " << width << ")");

    // z_solver = P_solver^-1 r_solver = D^-1 P^-1 D r_solver
    //   stage 1: r_phys = D r_solver        (Unapply input, into scratch)
    //   stage 2: z_phys = inner.Mult(r_phys) = P^-1 r_phys
    //                                          (written directly into z buffer)
    //   stage 3: z_solver = D^-1 z_phys      (Apply output, in-place)

    // Stage 1 — device-aware copy into reusable scratch, then
    // in-place unapply on the scratch view.
    CopyVectorDeviceAware(r_solver,
                          static_cast<mfem::Vector&>(m_scratch_view));
    m_scaler->UnapplyToIncrement(m_scratch_view);       // *= D

    // Stage 2 — inner prec writes directly into z_solver via view.
    mfem::BlockVector z_view = MakeBlockView(z_solver, m_block_offsets);
    m_inner_prec->Mult(m_scratch_view, z_view);

    // Stage 3 — apply scaler in-place on output view.
    m_scaler->ApplyToIncrement(z_view);                 // /= D
}

void ScaledSaddlePreconditioner::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_prec::set_operator");

    // `op` is the SCALED Jacobian. Unwrap to recover the physical
    // Jacobian and forward to inner prec. The inner prec (e.g.
    // MortarSaddlePreconditioner) needs the physical BlockOperator
    // to extract K from block (0,0), build the Schur diagonal, etc.
    const auto* scaled_jac = dynamic_cast<const ScaledJacobianOperator*>(&op);
    MFEM_VERIFY(scaled_jac != nullptr,
                "ScaledSaddlePreconditioner::SetOperator: operator is not a "
                "ScaledJacobianOperator. The Krylov inside the inner saddle "
                "solver must be configured with the scaled Jacobian returned "
                "by ScaledSaddleOperator::GetGradient.");

    m_inner_prec->SetOperator(scaled_jac->GetUnscaled());
    height = scaled_jac->Height();
    width = scaled_jac->Width();
}

void ScaledSaddlePreconditioner::Refresh(
    std::shared_ptr<mfem::Solver> new_inner_prec,
    const mfem::Array<int>& new_block_offsets)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::scaled_saddle_prec::refresh");
    MFEM_VERIFY(new_inner_prec,
                "ScaledSaddlePreconditioner::Refresh: "
                "new_inner_prec must not be null");
    m_inner_prec = std::move(new_inner_prec);
    m_block_offsets = new_block_offsets;
    height = m_inner_prec->Height();
    width = m_inner_prec->Width();

    // Phase 5.11.H.2 — resize scratch if needed.
    EnsureScratchSized(m_scratch_storage, m_scratch_view, m_block_offsets);
}

}   // namespace mortar_pbc
