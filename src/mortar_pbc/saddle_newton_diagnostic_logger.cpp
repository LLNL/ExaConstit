// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.K — implementation of `SaddleNewtonDiagnosticLogger`.
//
// See header for the file-level overview, CSV column layout, and the
// pre-/post-solve flush lifecycle.

#include "saddle_newton_diagnostic_logger.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <cmath>
#include <iomanip>
#include <utility>

namespace mortar_pbc
{

namespace {

//==============================================================================
// L2 norm of a contiguous sub-range of `v`, MPI_Allreduce'd.
//==============================================================================
double BlockL2Norm(const mfem::Vector& v, int start, int n, MPI_Comm comm)
{
    const double* d = v.HostRead();
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double x = d[start + i];
        sumsq += x * x;
    }
    double global_sumsq = 0.0;
    MPI_Allreduce(&sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global_sumsq);
}

//==============================================================================
// Per-sub-block L2 norms for the lambda half of `v`. `start` is the
// offset to the lambda block; `sb_of_row` is the scaler's
// sub-block-of-row table (size n_lam), with -1 flagging "no
// sub-block".
//==============================================================================
void SubblockNorms(const mfem::Vector& v, int start, int n_lam,
                    const mfem::Array<int>& sb_of_row, int n_sub,
                    MPI_Comm comm,
                    std::vector<double>& norms_out)
{
    std::vector<double> local_sumsq(n_sub, 0.0);
    const double* d = v.HostRead();
    const int*    sb = sb_of_row.HostRead();
    for (int i = 0; i < n_lam; ++i)
    {
        const int k = sb[i];
        if (k >= 0 && k < n_sub)
        {
            const double x = d[start + i];
            local_sumsq[k] += x * x;
        }
    }
    std::vector<double> global_sumsq(n_sub, 0.0);
    MPI_Allreduce(local_sumsq.data(), global_sumsq.data(), n_sub,
                  MPI_DOUBLE, MPI_SUM, comm);
    norms_out.resize(n_sub);
    for (int k = 0; k < n_sub; ++k)
    {
        norms_out[k] = std::sqrt(global_sumsq[k]);
    }
}

}  // anonymous namespace


//==============================================================================
// Construction / destruction
//==============================================================================

SaddleNewtonDiagnosticLogger::SaddleNewtonDiagnosticLogger(
    std::shared_ptr<const SaddleResidualScaler> scaler,
    const mfem::Array<int>& saddle_offsets,
    MPI_Comm comm,
    const std::string& filename)
    : m_scaler(std::move(scaler))
    , m_saddle_offsets(saddle_offsets)  // mfem::Array copy
    , m_comm(comm)
    , m_filename(filename)
{
    MFEM_VERIFY(m_scaler != nullptr,
                "SaddleNewtonDiagnosticLogger: scaler must not be null. "
                "On no-scaling runs, construct a scaler with "
                "IsEnabled()==false rather than passing nullptr — the "
                "logger reads partition metadata (sub-block labels + "
                "sub-block-of-row table) from it regardless of enabled "
                "state.");
    MFEM_VERIFY(m_saddle_offsets.Size() == 3,
                "SaddleNewtonDiagnosticLogger: saddle_offsets must have "
                "size 3 (got " << m_saddle_offsets.Size() << ")");

    MPI_Comm_rank(m_comm, &m_rank);
}

SaddleNewtonDiagnosticLogger::~SaddleNewtonDiagnosticLogger()
{
    if (m_pending)
    {
        // Defensive: a Newton max-iter exit can leave a buffered row
        // that never got its post-solve fill. Flush with sentinels
        // rather than silently dropping the row.
        FlushPending_();
    }
}


//==============================================================================
// Sinks
//==============================================================================

NewtonDiagnosticSink SaddleNewtonDiagnosticLogger::MakeSink()
{
    return [this](const NewtonIterDiagnostic& diag) {
        OnPreSolve_(diag);
    };
}

void SaddleNewtonDiagnosticLogger::IncrementStep()
{
    // Defensive: flush any pending row. The flush burns the old
    // m_step_index into the row before we increment.
    if (m_pending)
    {
        FlushPending_();
    }
    ++m_step_index;
}


//==============================================================================
// Sink callback bodies
//==============================================================================

void SaddleNewtonDiagnosticLogger::OnPreSolve_(
    const NewtonIterDiagnostic& diag)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_logger::pre_solve");

    MFEM_VERIFY(diag.residual != nullptr,
                "SaddleNewtonDiagnosticLogger: NewtonIterDiagnostic.residual "
                "must be non-null. Phase 5.11.J sets this; older Newton "
                "code paths that don't populate it cannot use this logger.");

    // Defensive: flush any stale pending row before overwrite.
    if (m_pending)
    {
        FlushPending_();
    }
    m_pending.reset();

    // Partition-stability check. Lock layout on first call.
    const int n_sub = m_scaler->NumSubblocks();
    if (m_n_subblocks_cached < 0)
    {
        m_n_subblocks_cached = n_sub;
        m_cached_sub_labels  = m_scaler->SubblockLabels();
    }
    else
    {
        MFEM_VERIFY(n_sub == m_n_subblocks_cached,
                    "SaddleNewtonDiagnosticLogger: scaler NumSubblocks "
                    "changed mid-run (" << m_n_subblocks_cached << " -> "
                    << n_sub << "). CSV column count is locked at first "
                    "flush; mid-run partition changes would corrupt the "
                    "layout. Restart the run for a Phase-5.9 spec change.");
    }

    PendingRow row;
    row.step           = m_step_index;
    row.iter           = diag.iter;
    row.norm           = diag.norm;
    row.norm0          = diag.norm0;
    row.norm_max       = diag.norm_max;
    row.converged_now  = diag.converged_now;
    row.scaler_enabled = m_scaler->IsEnabled();

    // Residual decomposition — un-scales internally when scaler is
    // enabled, so the per-block norms are PHYSICAL regardless of
    // wrapper state. Matches 5.11.J behavior.
    DecomposeR_(*diag.residual, row.res_K, row.res_lam, row.res_lam_sub);

    // Scaling factors.
    row.d_u = m_scaler->GetDu();
    row.d_lam_sub.resize(n_sub);
    for (int k = 0; k < n_sub; ++k)
    {
        row.d_lam_sub[k] = m_scaler->GetSubblockFactor(k);
    }

    m_pending = std::move(row);
    FlushPending_();
}


//==============================================================================
// Decomposition helpers
//==============================================================================

void SaddleNewtonDiagnosticLogger::DecomposeR_(
    const mfem::Vector& r,
    double& res_K_phys,
    double& res_lam_phys,
    std::vector<double>& res_lam_sub_phys) const
{
    const int n_u   = m_saddle_offsets[1];
    const int n_lam = m_saddle_offsets[2] - m_saddle_offsets[1];

    // Copy r and (if scaler is enabled) un-apply D to produce a
    // PHYSICAL residual. `UnapplyToIncrement` is the multiply-by-D
    // op; its name reflects its primary use (un-scaling a dx_solver
    // into dx_phys), but the math is the same for un-scaling a
    // residual: r_phys = D * r_solver. At D=I it's a no-op.
    mfem::Vector r_phys_storage(r);
    mfem::BlockVector r_phys;
    r_phys.Update(r_phys_storage, m_saddle_offsets);

    if (m_scaler->IsEnabled())
    {
        m_scaler->UnapplyToIncrement(r_phys);
    }

    res_K_phys   = BlockL2Norm(r_phys, 0,   n_u,   m_comm);
    res_lam_phys = BlockL2Norm(r_phys, n_u, n_lam, m_comm);
    SubblockNorms(r_phys, n_u, n_lam,
                   m_scaler->SubblockOfRow(),
                   m_scaler->NumSubblocks(),
                   m_comm, res_lam_sub_phys);
}

void SaddleNewtonDiagnosticLogger::EnsureFileOpen_()
{
    if (m_rank != 0)        { return; }
    if (m_file.is_open())   { return; }

    m_file.open(m_filename);
    MFEM_VERIFY(m_file.is_open(),
                "SaddleNewtonDiagnosticLogger: failed to open CSV '"
                << m_filename << "' for writing");
    // Wide precision for IEEE-double-exact diff at eps = 0.0.
    m_file << std::scientific << std::setprecision(17);
}

void SaddleNewtonDiagnosticLogger::WriteHeader_()
{
    if (m_rank != 0) { return; }

    m_file << "step,iter,norm,norm0,norm_max,converged_now,scaler_enabled,"
           << "res_K,res_lam";
    for (const auto& lbl : m_cached_sub_labels)
    {
        m_file << ",res_lam_" << lbl;
    }
    m_file << ",d_u";
    for (const auto& lbl : m_cached_sub_labels)
    {
        m_file << ",d_lam_" << lbl;
    }
    m_file << "\n";
}

void SaddleNewtonDiagnosticLogger::FlushPending_()
{
    if (!m_pending) { return; }

    if (m_rank == 0)
    {
        EnsureFileOpen_();
        if (m_n_subblocks_cached >= 0
            && m_cached_sub_labels.size() ==
                 static_cast<std::size_t>(m_n_subblocks_cached)
            && m_file.tellp() == std::streampos(0))
        {
            WriteHeader_();
        }

        const auto& row = *m_pending;
        m_file << row.step << ',' << row.iter << ','
               << row.norm << ',' << row.norm0 << ',' << row.norm_max << ','
               << (row.converged_now ? 1 : 0) << ','
               << (row.scaler_enabled ? 1 : 0) << ','
               << row.res_K << ',' << row.res_lam;
        for (double v : row.res_lam_sub) { m_file << ',' << v; }
        m_file << ',' << row.d_u;
        for (double v : row.d_lam_sub) { m_file << ',' << v; }
        m_file << '\n';
        m_file.flush();
    }

    m_pending.reset();
}

}   // namespace mortar_pbc
