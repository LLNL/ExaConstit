// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.C — SaddleResidualScaler implementation.
//
// See header for class documentation; planning doc
// `phase_5_11_saddle_residual_scaling_plan.md` §2, §4.1, §5 for the
// mathematical formulation and design rationale.

#include "saddle_residual_scaler.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace mortar_pbc
{

namespace
{

//==============================================================================
// ScaleFromNorm — Rule A (unit-balance) with floor + range-cap guards.
//
//   if r_norm < floor:  return 1.0   (identity for near-zero residual)
//   else:               return min(r_norm, range_cap)
//
// The floor guard sets d = 1.0 (not d = floor) so that residuals
// below floor pass through unchanged — dividing by floor would
// amplify them by 1/floor (~ 1e12 for the default floor), which
// would mean a "converged" block gets blown up by scaling.
//==============================================================================
double ScaleFromNorm(double r_norm, double floor, double range_cap)
{
    if (r_norm < floor)
    {
        return 1.0;
    }
    return std::min(r_norm, range_cap);
}

}   // anonymous namespace

//==============================================================================
// Constructor
//==============================================================================

SaddleResidualScaler::SaddleResidualScaler(
    const SaddleResidualScalerConfig& cfg)
    : m_cfg(cfg)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::ctor");
}

//==============================================================================
// SetPartitionDirect
//
// Copies labels and per-row IDs in; sets m_d_lambda size; resets all
// scaling factors to identity.
//==============================================================================

void SaddleResidualScaler::SetPartitionDirect(
    const std::vector<std::string>& subblock_labels,
    const mfem::Array<int>& subblock_of_row)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::set_partition_direct");

    m_subblock_labels = subblock_labels;
    m_n_subblocks = static_cast<int>(m_subblock_labels.size());

    m_subblock_of_row = subblock_of_row;
    m_d_lambda.SetSize(m_subblock_of_row.Size());

    // Phase 5.11.J — keep the per-sub-block factor parallel state
    // sized and identity-initialized alongside m_d_lambda.
    m_subblock_factor.SetSize(m_n_subblocks);
    m_subblock_factor = 1.0;

    Reset();
}

//==============================================================================
// RebuildPartition
//
// Delegates to ConstraintBuilder3D::GetRowSubblockIds + SetPartitionDirect.
//==============================================================================

void SaddleResidualScaler::RebuildPartition(
    const ConstraintBuilder3D& builder,
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::rebuild_partition");

    std::vector<std::string> labels;
    mfem::Array<int> sb_of_row;
    builder.GetRowSubblockIds(m_cfg.partition,
                              active_pair_labels, comp_mask,
                              labels, sb_of_row);
    SetPartitionDirect(labels, sb_of_row);
}

//==============================================================================
// Choose
//
// Per-step Rule A: scale each block to unit magnitude at iter 0.
//==============================================================================

void SaddleResidualScaler::Choose(
    double r_u_norm,
    const mfem::Vector& r_lambda_subblock_norms)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::choose");

    MFEM_ASSERT(r_lambda_subblock_norms.Size() == m_n_subblocks,
                "SaddleResidualScaler::Choose: r_lambda_subblock_norms "
                "size (" << r_lambda_subblock_norms.Size()
                << ") != NumSubblocks() (" << m_n_subblocks << "). "
                "Did RebuildPartition run for the current filter spec?");

    //--- u-block scalar ---
    m_d_u = ScaleFromNorm(r_u_norm, m_cfg.floor, m_cfg.range_cap);

    //--- Per-sub-block lambda scalars ---
    //
    // Build the per-sub-block array first, then broadcast to per-row
    // m_d_lambda. This factoring keeps the per_subblock = true / false
    // paths in one place (the broadcast at the end).
    mfem::Vector d_per_sb(m_n_subblocks);
    double* d_sb_data       = d_per_sb.HostWrite();
    const double* r_sb_data = r_lambda_subblock_norms.HostRead();

    if (m_cfg.per_subblock)
    {
        for (int k = 0; k < m_n_subblocks; ++k)
        {
            d_sb_data[k] = ScaleFromNorm(r_sb_data[k],
                                         m_cfg.floor, m_cfg.range_cap);
        }
    }
    else
    {
        double joint_sq = 0.0;
        for (int k = 0; k < m_n_subblocks; ++k)
        {
            joint_sq += r_sb_data[k] * r_sb_data[k];
        }
        const double joint = std::sqrt(joint_sq);
        const double d_joint = ScaleFromNorm(joint,
                                              m_cfg.floor, m_cfg.range_cap);
        for (int k = 0; k < m_n_subblocks; ++k)
        {
            d_sb_data[k] = d_joint;
        }
    }

    //--- Cache per-sub-block scalars for diagnostic logging (5.11.J) ---
    {
        double* sf = m_subblock_factor.HostWrite();
        for (int k = 0; k < m_n_subblocks; ++k)
        {
            sf[k] = d_sb_data[k];
        }
    }

    //--- Broadcast per-sub-block scalars to per-row m_d_lambda ---
    double* d_lam = m_d_lambda.HostWrite();
    const int* sb_row = m_subblock_of_row.HostRead();
    const int n = m_d_lambda.Size();
    for (int i = 0; i < n; ++i)
    {
        d_lam[i] = d_sb_data[sb_row[i]];
    }
}

//==============================================================================
// Reset
//==============================================================================

void SaddleResidualScaler::Reset()
{
    m_d_u = 1.0;
    m_subblock_factor = 1.0;
    double* d = m_d_lambda.HostWrite();
    const int n = m_d_lambda.Size();
    for (int i = 0; i < n; ++i)
    {
        d[i] = 1.0;
    }
}

//==============================================================================
// ApplyToResidual: r -> D^-1 r
//==============================================================================

void SaddleResidualScaler::ApplyToResidual(mfem::BlockVector& r) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::apply_to_residual");

    // u block: r_u[i] /= d_u
    {
        mfem::Vector& r_u = r.GetBlock(0);
        const double inv_d_u = 1.0 / m_d_u;
        double* ru = r_u.HostReadWrite();
        const int n = r_u.Size();
        for (int i = 0; i < n; ++i)
        {
            ru[i] *= inv_d_u;
        }
    }

    // lambda block: r_lam[i] /= d_lambda[i]
    {
        mfem::Vector& r_lam = r.GetBlock(1);
        MFEM_ASSERT(r_lam.Size() == m_d_lambda.Size(),
                    "ApplyToResidual: lambda block size ("
                    << r_lam.Size() << ") != m_d_lambda size ("
                    << m_d_lambda.Size() << ")");
        double* rl = r_lam.HostReadWrite();
        const double* dl = m_d_lambda.HostRead();
        const int n = r_lam.Size();
        for (int i = 0; i < n; ++i)
        {
            rl[i] /= dl[i];
        }
    }
}

//==============================================================================
// UnapplyToIncrement: dx_solver -> dx_phys = D dx_solver
//==============================================================================

void SaddleResidualScaler::UnapplyToIncrement(mfem::BlockVector& dx) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::unapply_to_increment");

    {
        mfem::Vector& dx_u = dx.GetBlock(0);
        double* du = dx_u.HostReadWrite();
        const int n = dx_u.Size();
        for (int i = 0; i < n; ++i)
        {
            du[i] *= m_d_u;
        }
    }

    {
        mfem::Vector& dx_lam = dx.GetBlock(1);
        MFEM_ASSERT(dx_lam.Size() == m_d_lambda.Size(),
                    "UnapplyToIncrement: lambda block size mismatch");
        double* dl_dx = dx_lam.HostReadWrite();
        const double* dl = m_d_lambda.HostRead();
        const int n = dx_lam.Size();
        for (int i = 0; i < n; ++i)
        {
            dl_dx[i] *= dl[i];
        }
    }
}

//==============================================================================
// ApplyToIncrement: dx_phys -> dx_solver = D^-1 dx_phys
//==============================================================================

void SaddleResidualScaler::ApplyToIncrement(mfem::BlockVector& dx) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::apply_to_increment");

    {
        mfem::Vector& dx_u = dx.GetBlock(0);
        const double inv_d_u = 1.0 / m_d_u;
        double* du = dx_u.HostReadWrite();
        const int n = dx_u.Size();
        for (int i = 0; i < n; ++i)
        {
            du[i] *= inv_d_u;
        }
    }

    {
        mfem::Vector& dx_lam = dx.GetBlock(1);
        MFEM_ASSERT(dx_lam.Size() == m_d_lambda.Size(),
                    "ApplyToIncrement: lambda block size mismatch");
        double* dl_dx = dx_lam.HostReadWrite();
        const double* dl = m_d_lambda.HostRead();
        const int n = dx_lam.Size();
        for (int i = 0; i < n; ++i)
        {
            dl_dx[i] /= dl[i];
        }
    }
}

//==============================================================================
// ScaledNorm: ||D^-1 r||_2
//==============================================================================

double SaddleResidualScaler::ScaledNorm(const mfem::BlockVector& r) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::scaled_norm");

    double sum_sq = 0.0;

    {
        const mfem::Vector& r_u = r.GetBlock(0);
        const double inv_d_u_sq = 1.0 / (m_d_u * m_d_u);
        const double* ru = r_u.HostRead();
        const int n = r_u.Size();
        for (int i = 0; i < n; ++i)
        {
            sum_sq += ru[i] * ru[i] * inv_d_u_sq;
        }
    }

    {
        const mfem::Vector& r_lam = r.GetBlock(1);
        MFEM_ASSERT(r_lam.Size() == m_d_lambda.Size(),
                    "ScaledNorm: lambda block size mismatch");
        const double* rl = r_lam.HostRead();
        const double* dl = m_d_lambda.HostRead();
        const int n = r_lam.Size();
        for (int i = 0; i < n; ++i)
        {
            const double r_scaled = rl[i] / dl[i];
            sum_sq += r_scaled * r_scaled;
        }
    }

    return std::sqrt(sum_sq);
}

//==============================================================================
// ScaledBlockNorms
//==============================================================================

void SaddleResidualScaler::ScaledBlockNorms(
    const mfem::BlockVector& r,
    double& r_u_scaled,
    mfem::Vector& r_lambda_subblock_scaled) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_residual_scaler::scaled_block_norms");

    // u block
    {
        const mfem::Vector& r_u = r.GetBlock(0);
        const double inv_d_u_sq = 1.0 / (m_d_u * m_d_u);
        const double* ru = r_u.HostRead();
        const int n = r_u.Size();
        double sum_sq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sum_sq += ru[i] * ru[i] * inv_d_u_sq;
        }
        r_u_scaled = std::sqrt(sum_sq);
    }

    // Per-sub-block lambda
    r_lambda_subblock_scaled.SetSize(m_n_subblocks);
    {
        double* out = r_lambda_subblock_scaled.HostWrite();
        for (int k = 0; k < m_n_subblocks; ++k) { out[k] = 0.0; }

        const mfem::Vector& r_lam = r.GetBlock(1);
        MFEM_ASSERT(r_lam.Size() == m_d_lambda.Size(),
                    "ScaledBlockNorms: lambda block size mismatch");
        const double* rl = r_lam.HostRead();
        const double* dl = m_d_lambda.HostRead();
        const int* sb = m_subblock_of_row.HostRead();
        const int n = r_lam.Size();
        for (int i = 0; i < n; ++i)
        {
            const double r_scaled = rl[i] / dl[i];
            out[sb[i]] += r_scaled * r_scaled;
        }
        for (int k = 0; k < m_n_subblocks; ++k)
        {
            out[k] = std::sqrt(out[k]);
        }
    }
}

//==============================================================================
// UnscaledLambdaSubblockNormsSqLocal
//
// Per-sub-block sums of squares of r_lambda. LOCAL only — caller
// must MPI_Allreduce the result across ranks.
//==============================================================================

void SaddleResidualScaler::UnscaledLambdaSubblockNormsSqLocal(
    const mfem::Vector& r_lambda,
    mfem::Vector& subblock_norms_sq) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::saddle_residual_scaler::unscaled_lambda_subblock_norms_sq_local");

    MFEM_ASSERT(r_lambda.Size() == m_subblock_of_row.Size(),
                "UnscaledLambdaSubblockNormsSqLocal: r_lambda.Size() ("
                << r_lambda.Size() << ") != m_subblock_of_row.Size() ("
                << m_subblock_of_row.Size() << ")");

    subblock_norms_sq.SetSize(m_n_subblocks);
    double* out = subblock_norms_sq.HostWrite();
    for (int k = 0; k < m_n_subblocks; ++k) { out[k] = 0.0; }

    const double* r = r_lambda.HostRead();
    const int* sb   = m_subblock_of_row.HostRead();
    const int n     = r_lambda.Size();
    for (int i = 0; i < n; ++i)
    {
        out[sb[i]] += r[i] * r[i];
    }
}

double SaddleResidualScaler::GetSubblockFactor(int b) const
{
    MFEM_ASSERT(b >= 0 && b < m_n_subblocks,
                "SaddleResidualScaler::GetSubblockFactor: index "
                << b << " out of range [0, " << m_n_subblocks << ")");
    if (m_subblock_factor.Size() == 0) { return 1.0; }
    return m_subblock_factor[b];
}

}   // namespace mortar_pbc
