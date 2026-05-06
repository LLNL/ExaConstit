// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of SaddlePointSolver, ported from
// `mortar_pbc/saddle_point.py`. See header for design doc.

#include "saddle_point_solver.hpp"

#include "mortar_constraint_operator.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// Diagonal-vector scaling preconditioner block
//==============================================================================
//
// Wraps an `inv_diag` vector and applies `y[i] = inv_diag[i] * x[i]`.
// Used for both the K block and the Schur block of the block-Jacobi
// preconditioner.
class DiagonalScaler : public mfem::Solver
{
public:
    DiagonalScaler(int size, mfem::Vector inv_diag)
        : mfem::Solver(size, size),
          m_inv_diag(std::move(inv_diag))
    {
        MFEM_VERIFY(m_inv_diag.Size() == size,
                    "DiagonalScaler: inv_diag size (" << m_inv_diag.Size()
                    << ") does not match operator size (" << size << ")");
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        const int n = m_inv_diag.Size();
        MFEM_ASSERT(x.Size() == n && y.Size() == n,
                    "DiagonalScaler::Mult: size mismatch");
        // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean access.
        //
        // The BlockDiagonalPreconditioner constructs sub-vector views
        // of its output `y` and passes them in. Those views are in
        // "no valid copy" memory state on first use, so the unsafe
        // GetData() call fails the DEVICE_DEBUG assertion
        //   (Empty() || (flags & VALID_HOST))
        // The typed accessors declare access intent to the memory
        // manager, which fixes this:
        //   * HostRead — declares "I will read host data; migrate
        //     from device if needed."
        //   * HostWrite — declares "I will write host data; the host
        //     copy becomes the authoritative one after this call."
        const double* xd  = x.HostRead();
        const double* idd = m_inv_diag.HostRead();
        double*       yd  = y.HostWrite();
        for (int i = 0; i < n; ++i) { yd[i] = idd[i] * xd[i]; }
    }

    /// `Solver::SetOperator` is required by the ABC; for a fixed
    /// inverse-diagonal scaler, there is nothing to update when the
    /// outer operator changes.
    void SetOperator(const mfem::Operator& /*op*/) override {}

private:
    mfem::Vector m_inv_diag;
};

//==============================================================================
// Build inv(diag(K)) for the (0, 0) Jacobi block
//==============================================================================
mfem::Vector BuildInvDiagK(const mfem::HypreParMatrix& K)
{
    const int n_local = K.Height();
    mfem::Vector diag(n_local);
    diag = 0.0;
    // Cast away const because GetDiag's signature is non-const in MFEM
    // even though the operation is logically const.
    //
    // After GetDiag, `diag` may have its VALID_HOST flag in any state
    // depending on how MFEM was built (host-only vs device build).
    // We re-declare via HostRead/HostWrite below to be DEVICE_DEBUG-safe.
    const_cast<mfem::HypreParMatrix&>(K).GetDiag(diag);

    // Invert in place; guard against zero entries (Dirichlet-eliminated
    // rows have diagonal 1 after EliminateRowsCols, so this is mostly
    // defensive — but a coefficient of 0 in some integrator setups can
    // produce true zeros).
    mfem::Vector inv_diag(n_local);
    const double tiny = 1.0e-300;
    {
        // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean access. Use raw
        // host pointers in the loop (declares intent to the memory
        // manager AND avoids per-element operator()/Memory::[] checks).
        const double* d_in  = diag.HostRead();
        double*       d_out = inv_diag.HostWrite();
        for (int i = 0; i < n_local; ++i)
        {
            const double d = d_in[i];
            d_out[i] = (std::abs(d) > tiny) ? (1.0 / d) : 0.0;
        }
    }
    return inv_diag;
}

//==============================================================================
// Build inv(diag(C * Dinv * C^T)) for the (1, 1) Schur block
//
// Method: for each local row i of C, compute
//      schur_diag[i] = sum_j C[i, j]^2 * Dinv_global[j]
//
// For this to work, every rank needs the FULL global Dinv vector
// (since C[i, :] can have non-zeros in any column). We Allgatherv the
// per-rank Dinv slices.
//
// This avoids any explicit `RAP` or `ParMult` against C, so the same
// path works whether K is HypreParMatrix or a PA Operator (the
// HypreParMatrix path is taken here only because the helper is
// instantiated on `HypreParMatrix&`).
//==============================================================================
mfem::Vector BuildInvDiagSchur(const mfem::HypreParMatrix& C,
                               const mfem::Vector& inv_diag_K_local)
{
    MPI_Comm comm = C.GetComm();
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Allgatherv the per-rank Dinv vectors into a single global array
    // ordered by rank-major. Hypre stores rows in this order for K so
    // the column ordering of C matches naturally (column partition
    // of C aligns with row partition of K).
    const int n_local = inv_diag_K_local.Size();
    std::vector<int> all_counts(nranks, 0);
    MPI_Allgather(&n_local, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

    int n_global = 0;
    std::vector<int> recv_counts(nranks);
    std::vector<int> displs(nranks);
    for (int r = 0; r < nranks; ++r)
    {
        displs[r] = n_global;
        recv_counts[r] = all_counts[r];
        n_global += all_counts[r];
    }

    std::vector<double> Dinv_global(n_global, 0.0);
    // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean: HostRead declares
    // intent before MPI consumes the host pointer.
    MPI_Allgatherv(inv_diag_K_local.HostRead(), n_local, MPI_DOUBLE,
                   Dinv_global.data(), recv_counts.data(), displs.data(),
                   MPI_DOUBLE, comm);

    // Walk C's local CSR (diag + offd parts) and compute the row-sum.
    // HypreParMatrix exposes GetDiag(SparseMatrix&) for the local-
    // column-block diagonal part and GetOffd(SparseMatrix&, int*&)
    // for the off-diagonal part with a column-map.
    mfem::SparseMatrix C_diag, C_offd;
    HYPRE_BigInt* col_map_offd = nullptr;
    const_cast<mfem::HypreParMatrix&>(C).GetDiag(C_diag);
    const_cast<mfem::HypreParMatrix&>(C).GetOffd(C_offd, col_map_offd);

    // Row offset for C's column space — global column index of the
    // first owned column on this rank. This is the row offset of K
    // (since C and K share column space = velocity-DOF space).
    // ColPart()[0] is this rank's first global column.
    HYPRE_BigInt my_col_first = C.ColPart()[0];

    const int n_lam_local = C.Height();
    mfem::Vector schur_diag(n_lam_local);
    // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean accumulation. Get a
    // host raw pointer once, zero-init through it, then accumulate
    // into the same pointer for the rest of this function.
    double* sd = schur_diag.HostWrite();
    for (int i = 0; i < n_lam_local; ++i) { sd[i] = 0.0; }

    // Diag part: column indices are LOCAL (relative to my_col_first).
    {
        const int* I = C_diag.GetI();
        const int* J = C_diag.GetJ();
        const double* A = C_diag.GetData();
        for (int i = 0; i < n_lam_local; ++i)
        {
            double s = 0.0;
            for (int k = I[i]; k < I[i + 1]; ++k)
            {
                const int j_local = J[k];
                const int j_global = static_cast<int>(my_col_first) + j_local;
                const double a = A[k];
                if (j_global >= 0 && j_global < n_global)
                {
                    s += a * a * Dinv_global[j_global];
                }
            }
            sd[i] += s;
        }
    }

    // Offd part: column indices in J are positions into col_map_offd[];
    // col_map_offd[J[k]] is the actual global column.
    if (C_offd.Width() > 0 && col_map_offd != nullptr)
    {
        const int* I = C_offd.GetI();
        const int* J = C_offd.GetJ();
        const double* A = C_offd.GetData();
        for (int i = 0; i < n_lam_local; ++i)
        {
            double s = 0.0;
            for (int k = I[i]; k < I[i + 1]; ++k)
            {
                const int j_global = static_cast<int>(col_map_offd[J[k]]);
                const double a = A[k];
                if (j_global >= 0 && j_global < n_global)
                {
                    s += a * a * Dinv_global[j_global];
                }
            }
            sd[i] += s;
        }
    }

    // Invert. Schur-diagonal entries can legitimately be zero on ranks
    // that hold no constraint rows — leave those as 0 (the multiplier-
    // block of the Krylov RHS is zero for those entries anyway).
    //
    // After the host writes above, schur_diag has VALID_HOST set; the
    // HostRead below confirms that intent and returns the same buffer.
    mfem::Vector inv_schur(n_lam_local);
    const double tiny = 1.0e-300;
    {
        const double* sd_in = schur_diag.HostRead();
        double* iv = inv_schur.HostWrite();
        for (int i = 0; i < n_lam_local; ++i)
        {
            const double d = sd_in[i];
            iv[i] = (std::abs(d) > tiny) ? (1.0 / d) : 0.0;
        }
    }
    return inv_schur;
}

}  // anonymous namespace

//==============================================================================
// Constructor
//==============================================================================

SaddlePointSolver::SaddlePointSolver(const SaddlePointSolverConfig& cfg)
    : m_cfg(cfg)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point::ctor");
    // Defensive enum check; the enum itself has no CG, but we surface
    // an explicit error rather than silently falling through.
    switch (m_cfg.solver_type)
    {
        case KrylovType::MINRES:
        case KrylovType::GMRES:
        case KrylovType::BiCGSTAB:
            break;
        default:
            MFEM_ABORT("SaddlePointSolver: unknown KrylovType "
                       << static_cast<int>(m_cfg.solver_type));
    }
    switch (m_cfg.prec_type)
    {
        case SaddlePrecType::None:
        case SaddlePrecType::BlockJacobi:
            break;
        default:
            MFEM_ABORT("SaddlePointSolver: unknown SaddlePrecType "
                       << static_cast<int>(m_cfg.prec_type));
    }
    MFEM_VERIFY(m_cfg.rel_tol > 0.0,
                "SaddlePointSolver: rel_tol must be positive (got "
                << m_cfg.rel_tol << ")");
    MFEM_VERIFY(m_cfg.abs_tol > 0.0,
                "SaddlePointSolver: abs_tol must be positive (got "
                << m_cfg.abs_tol << ")");
    MFEM_VERIFY(m_cfg.max_iter > 0,
                "SaddlePointSolver: max_iter must be positive (got "
                << m_cfg.max_iter << ")");
}

//==============================================================================
// Solve
//==============================================================================

void SaddlePointSolver::Solve(const mfem::HypreParMatrix& K,
                              const mfem::HypreParMatrix& C,
                              const mfem::Vector& r1,
                              const mfem::Vector& r2,
                              mfem::Vector& du,
                              mfem::Vector& dlam)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point::solve");

    const int n_v_local   = K.Height();
    const int n_lam_local = C.Height();

    MFEM_VERIFY(K.Width() == n_v_local,
                "SaddlePointSolver::Solve: K must be square; got ("
                << K.Height() << ", " << K.Width() << ")");
    MFEM_VERIFY(C.Width() == n_v_local,
                "SaddlePointSolver::Solve: C cols (" << C.Width()
                << ") must match K rows (" << n_v_local << ")");
    MFEM_VERIFY(r1.Size() == n_v_local,
                "SaddlePointSolver::Solve: r1 size (" << r1.Size()
                << ") must match K.Height() (" << n_v_local << ")");
    MFEM_VERIFY(r2.Size() == n_lam_local,
                "SaddlePointSolver::Solve: r2 size (" << r2.Size()
                << ") must match C.Height() (" << n_lam_local << ")");

    // Compute preconditioner pieces via the HypreParMatrix path.
    // This is the only point at which the HypreParMatrix-only entry
    // path differs from the EA entry path; everything else flows
    // through SolveImplInternal.
    mfem::Vector inv_diag_K = BuildInvDiagK(K);
    mfem::Vector inv_diag_S = BuildInvDiagSchur(C, inv_diag_K);

    // The internal helper takes K and C as mfem::Operator&. Cast away
    // const because BlockOperator::SetBlock takes Operator* (mirrors
    // the existing pattern at line 297-300 of the pre-refactor code).
    SolveImplInternal(
        const_cast<mfem::HypreParMatrix&>(K),
        const_cast<mfem::HypreParMatrix&>(C),
        K.GetComm(),
        inv_diag_K, inv_diag_S,
        n_v_local, n_lam_local,
        r1, r2, du, dlam);
}

void SaddlePointSolver::Solve(const mfem::HypreParMatrix& K,
                              const MortarConstraintOperator& C_op,
                              const mfem::Vector& r1,
                              const mfem::Vector& r2,
                              mfem::Vector& du,
                              mfem::Vector& dlam)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_point::solve_ea");

    const int n_v_local   = K.Height();
    const int n_lam_local = C_op.Height();

    MFEM_VERIFY(K.Width() == n_v_local,
                "SaddlePointSolver::Solve(EA): K must be square; got ("
                << K.Height() << ", " << K.Width() << ")");
    MFEM_VERIFY(C_op.Width() == n_v_local,
                "SaddlePointSolver::Solve(EA): C_op cols ("
                << C_op.Width() << ") must match K rows ("
                << n_v_local << ")");
    MFEM_VERIFY(r1.Size() == n_v_local,
                "SaddlePointSolver::Solve(EA): r1 size (" << r1.Size()
                << ") must match K.Height() (" << n_v_local << ")");
    MFEM_VERIFY(r2.Size() == n_lam_local,
                "SaddlePointSolver::Solve(EA): r2 size (" << r2.Size()
                << ") must match C_op.Height() (" << n_lam_local
                << ")");

    // Preconditioner pieces via the EA path. inv_diag_K is computed
    // the same way (HypreParMatrix-side); inv_diag_S uses the EA
    // operator's per-pair-block walk (Batch R) instead of a CSR walk.
    mfem::Vector inv_diag_K = BuildInvDiagK(K);
    mfem::Vector inv_diag_S = C_op.ComputeInvDiagSchur(inv_diag_K);

    SolveImplInternal(
        const_cast<mfem::HypreParMatrix&>(K),
        const_cast<MortarConstraintOperator&>(C_op),
        K.GetComm(),
        inv_diag_K, inv_diag_S,
        n_v_local, n_lam_local,
        r1, r2, du, dlam);
}

//==============================================================================
// Phase 4.3 / Batch S — internal helper shared by both Solve overloads.
//
// Identical Krylov plumbing for both the HypreParMatrix path and the
// EA path. Differences land in the caller (which computes inv_diag_S
// its own way and provides the right operator references).
//
// K_op and C_op enter as mutable mfem::Operator& because mfem's
// BlockOperator::SetBlock signature takes Operator*. The caller has
// already cast away const where appropriate.
//==============================================================================
void SaddlePointSolver::SolveImplInternal(
    mfem::Operator& K_op,
    mfem::Operator& C_op,
    MPI_Comm comm,
    mfem::Vector& inv_diag_K,
    mfem::Vector& inv_diag_S,
    int n_v_local,
    int n_lam_local,
    const mfem::Vector& r1,
    const mfem::Vector& r2,
    mfem::Vector& du,
    mfem::Vector& dlam)
{
    //---- Build the block operator [[K, C^T], [C, 0]] ----
    //
    // C^T is wrapped as a TransposeOperator over C; this dispatches
    // BlockOperator's calls to C_op.MultTranspose (which both
    // HypreParMatrix and MortarConstraintOperator implement).
    mfem::Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = n_v_local;
    block_offsets[2] = n_v_local + n_lam_local;

    mfem::TransposeOperator CT_op(&C_op);

    mfem::BlockOperator block_op(block_offsets);
    block_op.SetBlock(0, 0, &K_op);
    block_op.SetBlock(0, 1, &CT_op);
    block_op.SetBlock(1, 0, &C_op);
    // (1, 1) is the zero block — not set.

    //---- Build the block-diagonal preconditioner ----
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec;
    std::unique_ptr<DiagonalScaler> jacobi_K;
    std::unique_ptr<DiagonalScaler> jacobi_S;
    if (m_cfg.prec_type == SaddlePrecType::BlockJacobi)
    {
        jacobi_K = std::make_unique<DiagonalScaler>(n_v_local,
                                                    std::move(inv_diag_K));
        jacobi_S = std::make_unique<DiagonalScaler>(n_lam_local,
                                                    std::move(inv_diag_S));

        block_prec = std::make_unique<mfem::BlockDiagonalPreconditioner>(
            block_offsets);
        block_prec->SetDiagonalBlock(0, jacobi_K.get());
        block_prec->SetDiagonalBlock(1, jacobi_S.get());
    }

    //---- Build the RHS [-r1; -r2] ----
    //
    // Phase 4.3.B / Batch X — DEVICE_DEBUG-clean: r1 and r2 are
    // freshly-built input vectors (per-Newton-iteration); we Host-Read
    // them and Host-Write the rhs blocks via raw pointers. The block
    // views into rhs share the underlying memory with rhs itself, so
    // the writes propagate back to rhs's GetBlock as expected.
    mfem::BlockVector rhs(block_offsets);
    {
        const double* r1_d = r1.HostRead();
        const double* r2_d = r2.HostRead();
        mfem::Vector& rhs_v = rhs.GetBlock(0);
        mfem::Vector& rhs_l = rhs.GetBlock(1);
        double* rhs_v_d = rhs_v.HostWrite();
        double* rhs_l_d = rhs_l.HostWrite();
        for (int i = 0; i < n_v_local; ++i)   { rhs_v_d[i] = -r1_d[i]; }
        for (int i = 0; i < n_lam_local; ++i) { rhs_l_d[i] = -r2_d[i]; }
    }

    //---- Krylov solver ----
    std::unique_ptr<mfem::IterativeSolver> krylov;
    switch (m_cfg.solver_type)
    {
        case KrylovType::MINRES:
            krylov = std::make_unique<mfem::MINRESSolver>(comm);
            break;
        case KrylovType::GMRES:
        {
            auto* gmres = new mfem::GMRESSolver(comm);
            gmres->SetKDim(m_cfg.gmres_kdim);
            krylov.reset(gmres);
            break;
        }
        case KrylovType::BiCGSTAB:
            krylov = std::make_unique<mfem::BiCGSTABSolver>(comm);
            break;
    }
    krylov->SetRelTol(m_cfg.rel_tol);
    krylov->SetAbsTol(m_cfg.abs_tol);
    krylov->SetMaxIter(m_cfg.max_iter);
    krylov->SetPrintLevel(m_cfg.print_level);
    krylov->SetOperator(block_op);
    if (block_prec) { krylov->SetPreconditioner(*block_prec); }

    // Force the solver to ignore the input solution as initial guess
    // and start from zero. The Newton outer loop carries information
    // across iterations via u_tilde and λ; the inner linear solve is
    // for the INCREMENTAL update (du, dλ). Reusing the previous
    // step's du as initial guess is a category error.
    krylov->iterative_mode = false;

    //---- Solve ----
    mfem::BlockVector solution(block_offsets);
    solution = 0.0;  // zero initial guess
    krylov->Mult(rhs, solution);

    //---- Diagnostics ----
    m_last_iterations  = krylov->GetNumIterations();
    m_last_converged   = krylov->GetConverged();
    m_last_final_norm  = krylov->GetFinalNorm();

    //---- Extract du and dlam ----
    du.SetSize(n_v_local);
    dlam.SetSize(n_lam_local);
    {
        const mfem::Vector& sol_v = solution.GetBlock(0);
        const mfem::Vector& sol_l = solution.GetBlock(1);
        const double* sv_d = sol_v.HostRead();
        const double* sl_d = sol_l.HostRead();
        double* du_d   = du.HostWrite();
        double* dlam_d = dlam.HostWrite();
        for (int i = 0; i < n_v_local; ++i)   { du_d[i]   = sv_d[i]; }
        for (int i = 0; i < n_lam_local; ++i) { dlam_d[i] = sl_d[i]; }
    }
}

}  // namespace mortar_pbc
