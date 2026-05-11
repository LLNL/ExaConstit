// Phase 4.3 / Batch R — MortarSaddlePointSystem implementation.
//
// See mortar_saddle_point_system.hpp for design rationale.

#include "mortar_saddle_point_system.hpp"

#include "utilities/mechanics_log.hpp"
#include "mfem.hpp"

namespace mortar_pbc {

//==============================================================================
// Constructor
//==============================================================================
MortarSaddlePointSystem::MortarSaddlePointSystem(
    KResidualFn k_residual,
    KJacobianFn k_jacobian,
    const MortarConstraintOperator& C_op)
    : mfem::Operator(0, 0)
    , m_k_residual(std::move(k_residual))
    , m_k_jacobian(std::move(k_jacobian))
    , m_C_op(C_op)
    , m_n_u(C_op.Width())
    , m_n_lam(C_op.Height())
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point_system::ctor");

    // Block layout: [u | lambda].
    m_block_offsets.SetSize(3);
    m_block_offsets[0] = 0;
    m_block_offsets[1] = m_n_u;
    m_block_offsets[2] = m_n_u + m_n_lam;

    // Operator dimensions (square — same in/out block layout).
    height = m_n_u + m_n_lam;
    width  = m_n_u + m_n_lam;
}

//==============================================================================
// Refresh — Phase 5.9.A.5
//
// Re-read m_n_u, m_n_lam, m_block_offsets, height, width from the
// underlying MortarConstraintOperator. Called by
// MortarPbcManager::RebuildForActiveSpec after the operator's
// Reset (which may have changed its Height under a new filter
// spec). Local — no MPI.
//==============================================================================
void MortarSaddlePointSystem::Refresh()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point_system::refresh");

    m_n_u   = m_C_op.Width();
    m_n_lam = m_C_op.Height();

    // m_block_offsets was sized to 3 at ctor; just rewrite the entries.
    m_block_offsets[0] = 0;
    m_block_offsets[1] = m_n_u;
    m_block_offsets[2] = m_n_u + m_n_lam;

    height = m_n_u + m_n_lam;
    width  = m_n_u + m_n_lam;
}

//==============================================================================
// Mult — compute saddle-point residual.
//
// Uses block views into x_block and r_block. The TransposeOperator
// for C^T is allocated per-call (cheap — just stores a pointer).
//==============================================================================
void MortarSaddlePointSystem::Mult(const mfem::Vector& x_block,
                                   mfem::Vector& r_block) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point_system::mult");

    MFEM_VERIFY(x_block.Size() == Width(),
                "MortarSaddlePointSystem::Mult: x_block size "
                << x_block.Size() << " != Width() " << Width());
    MFEM_VERIFY(r_block.Size() == Height(),
                "MortarSaddlePointSystem::Mult: r_block size "
                << r_block.Size() << " != Height() " << Height());

    // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean block views.
    //
    // We construct sub-vectors that alias the input/output block
    // buffers without copying. The aliasing pattern requires a host
    // pointer (mfem::Vector's pointer-constructor takes a raw double*).
    // Reading and writing then go through the standard mfem::Vector
    // memory-manager interface on the SUB-VECTORS — the K-residual
    // callback calls Read/Write internally, and m_C_op's Mult /
    // MultTranspose use Read/Write themselves.
    //
    // We use ReadWrite on x_block (callbacks may both read and update
    // through views) and Write on r_block (about to be overwritten).
    // After this point the manager's host copy is the authoritative
    // one; the C-operator and K-residual will fetch device copies as
    // needed via their own Read calls.
    double* x_data = const_cast<mfem::Vector&>(x_block).HostReadWrite();
    double* r_data = r_block.HostWrite();

    mfem::Vector x_u  (x_data,           m_n_u);
    mfem::Vector x_lam(x_data + m_n_u,   m_n_lam);
    mfem::Vector r_u  (r_data,           m_n_u);
    mfem::Vector r_lam(r_data + m_n_u,   m_n_lam);

    // r_u = K_residual(u)
    m_k_residual(x_u, r_u);

    // r_u += C^T * lambda. Use a scratch buffer for the C^T product
    // to avoid in-place issues with MultTranspose's overwrite
    // semantics.
    {
        mfem::Vector ct_lam(m_n_u);
        m_C_op.MultTranspose(x_lam, ct_lam);
        r_u += ct_lam;
    }

    // r_lam = C * u  (overwrite — Mult overwrites by contract).
    m_C_op.Mult(x_u, r_lam);

    // Phase 5.0 — if a constraint RHS has been installed via
    // SetConstraintRHS, subtract it: r_lam = C * u - g.
    // Default (no RHS installed) leaves r_lam = C * u, matching
    // the original Phase 4.3 behavior.
    if (m_g_rhs != nullptr)
    {
        MFEM_ASSERT(m_g_rhs->Size() == m_n_lam,
                    "MortarSaddlePointSystem::Mult: installed "
                    "constraint RHS size " << m_g_rhs->Size()
                    << " != NumLambda() " << m_n_lam);
        r_lam.Add(-1.0, *m_g_rhs);
    }
}

//==============================================================================
// GetGradient — return saddle-point Jacobian as a BlockOperator.
//
// Rebuilds the internal BlockOperator each call to pick up a fresh
// K_jacobian(u). The lifetime of the returned reference is "until
// the next GetGradient call" — matches mfem::ParNonlinearForm
// semantics.
//==============================================================================
mfem::Operator& MortarSaddlePointSystem::GetGradient(
    const mfem::Vector& x_block) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point_system::get_gradient");

    MFEM_VERIFY(x_block.Size() == Width(),
                "MortarSaddlePointSystem::GetGradient: x_block size "
                << x_block.Size() << " != Width() " << Width());

    // Block view of u for the user's K-Jacobian closure. Use
    // HostReadWrite so the memory manager registers the access on the
    // backing buffer; the K-Jacobian callback may both read u and
    // (less commonly) write into auxiliary state through the view.
    double* x_data = const_cast<mfem::Vector&>(x_block).HostReadWrite();
    mfem::Vector x_u(x_data, m_n_u);

    // Get the user's current K-Jacobian. The pointer must remain
    // valid until the next GetGradient call (or until the user's
    // form is destroyed).
    mfem::Operator* K_jac = m_k_jacobian(x_u);
    MFEM_VERIFY(K_jac != nullptr,
                "MortarSaddlePointSystem::GetGradient: KJacobianFn "
                "returned nullptr");
    MFEM_VERIFY(K_jac->Height() == m_n_u && K_jac->Width() == m_n_u,
                "MortarSaddlePointSystem::GetGradient: K-Jacobian "
                "dimensions (" << K_jac->Height() << ", "
                << K_jac->Width() << ") do not match expected ("
                << m_n_u << ", " << m_n_u << ")");

    // Rebuild C^T wrapper and the BlockOperator. Both are cheap
    // (pointer containers); the cost is the K_jacobian callback,
    // which we can't avoid.
    m_C_T_op = std::make_unique<mfem::TransposeOperator>(&m_C_op);
    m_block_op = std::make_unique<mfem::BlockOperator>(m_block_offsets);
    m_block_op->SetBlock(0, 0, K_jac);
    m_block_op->SetBlock(0, 1, m_C_T_op.get());
    m_block_op->SetBlock(1, 0,
        const_cast<MortarConstraintOperator*>(&m_C_op));
    // (1, 1) is zero — not set.

    return *m_block_op;
}

//==============================================================================
// SetConstraintRHS / ClearConstraintRHS — Phase 5.0.
//
// Install (or clear) an optional constraint RHS `g`, modifying the
// constraint-side residual returned by Mult from r_C = C * u to
// r_C = C * u - g. Default state (no RHS installed) preserves the
// original homogeneous Phase 4.3 behavior verbatim.
//
// The pointer is non-owning. The caller (typically
// MortarPbcManager) must keep `g` alive for the lifetime of the
// install — i.e. until either the next ClearConstraintRHS call or
// the next SetConstraintRHS replacement.
//==============================================================================
void MortarSaddlePointSystem::SetConstraintRHS(const mfem::Vector& g)
{
    MFEM_VERIFY(g.Size() == m_n_lam,
                "MortarSaddlePointSystem::SetConstraintRHS: g size "
                << g.Size() << " != NumLambda() " << m_n_lam);
    m_g_rhs = &g;
}

void MortarSaddlePointSystem::ClearConstraintRHS()
{
    m_g_rhs = nullptr;
}

}  // namespace mortar_pbc
