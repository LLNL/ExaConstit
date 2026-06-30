// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.C — SaddleResidualScaler class.
//
// Manages the per-step symmetric block-diagonal scaling of the
// mortar-PBC saddle system. See planning doc
// `phase_5_11_saddle_residual_scaling_plan.md` §2, §4.1, §5 for the
// mathematical formulation and design rationale.
//
// At a glance:
//
//   Saddle system A = [K     C^T]
//                     [C     0 ]
//
//   Scaling matrix  D = diag(d_u * I,  D_lambda)
//
//   where D_lambda is a piecewise-constant diagonal whose value on
//   sub-block k is d_lambda^(k). Sub-blocks come from
//   ConstraintBuilder3D::GetRowSubblockIds (Phase 5.11.B) under
//   either FaceEdge or PerPair partition.
//
//   Scaled system  tilde A = D^-1 A D^-1
//   Scaled residual tilde r = D^-1 r
//   Physical increment dx_phys = D dx_solver
//
// Per-step Rule A (unit-balance) chooses scaling factors from the
// initial residual norms so that every block has scaled magnitude
// 1.0 at Newton iteration 0:
//   d_u            = ScaleFromNorm(||r_u||,            floor, range_cap)
//   d_lambda^(k)   = ScaleFromNorm(||r_lambda^(k)||,   floor, range_cap)
//
//   ScaleFromNorm(r_norm, floor, cap):
//       if r_norm < floor:  return 1.0   (floor guard — identity for
//                                         near-zero residuals)
//       else:               return min(r_norm, cap)
//
// When config.per_subblock == false, all d_lambda^(k) are set to a
// single value computed from the joint lambda block norm; this
// recovers the single-scalar-per-block formulation as a special
// case of the multi-sub-block one (no separate code path).

#pragma once

#include "constraint_builder_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <string>
#include <vector>

namespace mortar_pbc
{

/**
 * @brief Internal config for SaddleResidualScaler (Phase 5.11).
 *
 * @details The options-side `::SaddleScalingOptions` (defined in
 * `option_parser_v2.hpp`) is translated to this mortar_pbc-internal
 * config at the `MortarPbcManager` boundary (Phase 5.11.E), following
 * the same separation-of-headers pattern as `SaddlePointSolverOptions`
 * → `SaddlePointSolverConfig`. The `mortar_pbc::SubblockPartition`
 * enum is defined in `constraint_builder_3d.hpp`.
 */
struct SaddleResidualScalerConfig
{
    /// Master enable flag. When false, the manager skips routing the
    /// Newton solver through this scaler (the saddle path runs
    /// unscaled, bit-for-bit identical to pre-Phase-5.11). The scaler
    /// itself honors all method calls regardless — the early-exit
    /// happens in the calling Newton solver.
    bool enabled = false;

    /// When true, each lambda sub-block gets its own d_lambda^(k)
    /// chosen from its own residual norm. When false, all sub-blocks
    /// share a single d_lambda computed from the joint lambda norm.
    bool per_subblock = false;

    /// Partition scheme for the lambda block. See `SubblockPartition`
    /// (in `constraint_builder_3d.hpp`).
    SubblockPartition partition = SubblockPartition::FaceEdge;

    /// Floor guard. Block residual norms below this are treated as
    /// zero — the corresponding scalar is set to 1.0 (identity)
    /// rather than dividing by a tiny number.
    double floor = 1.0e-12;

    /// Range cap. Scaling factors are clipped at this high-side
    /// bound to prevent extreme values amplifying floating-point
    /// error.
    double range_cap = 1.0e12;
};

/**
 * @brief Saddle-system residual scaler (Phase 5.11).
 *
 * @details Holds the current scaling state (d_u + per-row d_lambda)
 * and provides the in-place apply/unapply operations that the
 * Newton solver and saddle operator wrappers (Phase 5.11.D) consume.
 *
 * Lifecycle:
 *
 *   1. Construct with a `SaddleResidualScalerConfig`. The scaler is
 *      in an "empty" state — partition is not yet set, d_u = 1,
 *      m_d_lambda is empty.
 *   2. Call `RebuildPartition(builder, active_pair_labels, comp_mask)`
 *      to populate the per-row partition. Sets m_d_lambda size to
 *      the local lambda row count under that filter; resets all
 *      scaling factors to 1.0 (identity).
 *   3. Each step: call `Choose(r_u_norm, r_lambda_subblock_norms)`
 *      with the initial residual norms (after MPI_Allreduce — caller
 *      responsible for the collective). Populates d_u and per-row
 *      m_d_lambda from Rule A unit-balance.
 *   4. Inside the Newton solver: call `ScaledNorm`, `ApplyToResidual`,
 *      `UnapplyToIncrement`, etc. as needed.
 *   5. On Phase 5.9 spec transitions: call `RebuildPartition` again
 *      with the new filter spec. Resets scaling factors to identity;
 *      the next step's `Choose` repopulates them.
 *
 * All operations are local — no MPI inside this class. The manager
 * is responsible for collective reductions.
 */
class SaddleResidualScaler
{
public:
    /**
     * @brief Construct with config. Partition is empty until
     *        RebuildPartition (or SetPartitionDirect) is called.
     */
    explicit SaddleResidualScaler(const SaddleResidualScalerConfig& cfg);

    /**
     * @brief Build per-row sub-block partition from a constraint
     *        builder under the given filter spec.
     *
     * @details Calls `builder.GetRowSubblockIds(m_cfg.partition,
     * active_pair_labels, comp_mask, ...)`, then populates internal
     * state (labels, per-row IDs, sized m_d_lambda). Resets d_u and
     * m_d_lambda to identity (1.0) — the next `Choose` call must
     * populate them from initial residual norms.
     *
     * Called by `MortarPbcManager` at construction and after each
     * Phase 5.9 `RebuildForActiveSpec`.
     */
    void RebuildPartition(
        const ConstraintBuilder3D& builder,
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask);

    /**
     * @brief Set the partition directly from pre-computed labels
     *        and per-row IDs.
     *
     * @details For tests (avoid building an MFEM mesh just to test
     * the math) and for the implementation of `RebuildPartition`.
     * Resets d_u and m_d_lambda to identity (1.0).
     */
    void SetPartitionDirect(
        const std::vector<std::string>& subblock_labels,
        const mfem::Array<int>& subblock_of_row);

    /**
     * @brief Pick d_u and per-row m_d_lambda from initial residual
     *        norms per Rule A (unit-balance with floor/range guards).
     *
     * @param r_u_norm                    Global ||r_u||_2 (reduced).
     * @param r_lambda_subblock_norms     Global ||r_lambda^(k)||_2
     *                                    for each sub-block (reduced).
     *                                    Size must equal `NumSubblocks()`.
     *
     * @details When `cfg.per_subblock == true`, each sub-block's
     * scalar is set independently from its own norm. When false,
     * a single joint d_lambda is computed from the L2 join of the
     * per-sub-block norms and broadcast to all rows.
     */
    void Choose(double r_u_norm,
                const mfem::Vector& r_lambda_subblock_norms);

    /**
     * @brief Reset all scaling factors to identity (d_u = 1, all
     *        m_d_lambda = 1) without changing the partition.
     */
    void Reset();

    /**
     * @brief r -> D^-1 r (in-place). r is a BlockVector with blocks
     *        (u, lambda); lambda block size must match m_d_lambda.
     */
    void ApplyToResidual(mfem::BlockVector& r) const;

    /**
     * @brief dx_solver -> dx_phys = D dx_solver (in-place). Called
     *        by `ScaledSaddlePointSolver` (Phase 5.11.D) after the
     *        inner solver returns the scaled-coordinate increment.
     */
    void UnapplyToIncrement(mfem::BlockVector& dx_solver) const;

    /**
     * @brief dx_phys -> dx_solver = D^-1 dx_phys (in-place).
     *        Inverse direction from `UnapplyToIncrement`; used by
     *        TRDOG (Phase 5.11.G) to convert a physical Newton-step
     *        direction (returned by the inner saddle solver) into
     *        scaled dogleg coordinates.
     */
    void ApplyToIncrement(mfem::BlockVector& dx_phys) const;

    /**
     * @brief Compute ||D^-1 r||_2 directly without modifying r.
     *        Used by the Newton-side convergence test.
     */
    double ScaledNorm(const mfem::BlockVector& r) const;

    /**
     * @brief Compute scaled u-block norm and per-sub-block lambda
     *        norms separately. For diagnostic logging (Phase 5.11.I).
     */
    void ScaledBlockNorms(const mfem::BlockVector& r,
                          double& r_u_scaled,
                          mfem::Vector& r_lambda_subblock_scaled) const;

    /**
     * @brief Per-sub-block sums of squares of unscaled r_lambda.
     *        LOCAL only — caller must MPI_Allreduce. Used by the
     *        manager's `ChooseScalingForStep` (Phase 5.11.E).
     */
    void UnscaledLambdaSubblockNormsSqLocal(
        const mfem::Vector& r_lambda,
        mfem::Vector& subblock_norms_sq) const;

    //--------------------------------------------------------------------------
    // Accessors
    //--------------------------------------------------------------------------

    double GetDu() const { return m_d_u; }
    const mfem::Vector& GetDLambda() const { return m_d_lambda; }
    int NumSubblocks() const { return m_n_subblocks; }
    const std::vector<std::string>& SubblockLabels() const { return m_subblock_labels; }
    const mfem::Array<int>& SubblockOfRow() const { return m_subblock_of_row; }
    /// Phase 5.11.J — current per-sub-block lambda scaling factor.
    /// One uniform value per sub-block (D_lambda is piecewise-
    /// constant per sub-block by construction). Same on every
    /// rank. Returns 1.0 (identity) for a sub-block that has not
    /// been populated by Choose yet.
    double GetSubblockFactor(int b) const;

    bool IsEnabled()    const { return m_cfg.enabled;      }
    bool PerSubblock()  const { return m_cfg.per_subblock; }
    SubblockPartition Partition() const { return m_cfg.partition; }
    double Floor()      const { return m_cfg.floor;        }
    double RangeCap()   const { return m_cfg.range_cap;    }

private:
    SaddleResidualScalerConfig m_cfg;
    double                     m_d_u = 1.0;
    mfem::Vector               m_d_lambda;        ///< size n_lambda
    mfem::Array<int>           m_subblock_of_row; ///< size n_lambda
    int                        m_n_subblocks = 0;
    std::vector<std::string>   m_subblock_labels; ///< size n_subblocks
    /// Phase 5.11.J — per-sub-block lambda scaling factor (uniform
    /// across rows in a sub-block). Size = n_subblocks. Populated
    /// in Choose; reset to 1.0 in Reset and in RebuildPartition /
    /// SetPartitionDirect. Globally identical across all MPI
    /// ranks (Choose derives factors from globally-reduced norms).
    mfem::Vector               m_subblock_factor;
};

}   // namespace mortar_pbc
