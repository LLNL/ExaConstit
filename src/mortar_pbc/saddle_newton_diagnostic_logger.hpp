// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 5.11.J — saddle Newton diagnostic logger.
//
// What 5.11.J already did
// -----------------------
// Per Newton iter the logger writes one CSV row with the residual norm,
// linear-solve summary, physical per-block / per-sub-block residual
// decomposition, and current scaling factors. The pre-solve sink is
// installed on the Newton solver via
// `newton_solver->SetDiagnosticSink(logger->MakeSink())`; the post-linear
// sink is installed via
// `newton_solver->SetLinearDiagnosticSink(logger->MakeLinearSolveSink())`.
// The host (SystemDriver) calls `IncrementStep()` once per time step to
// advance the step counter that gets stamped into each row.
//
// The destructor flushes any leftover pending row (defensive — Newton
// max-iter exit without subsequent IncrementStep would otherwise
// drop the last row).
//
// CSV columns (full, in order)
// ----------------------------
//   step                  [int]    time-step index (from IncrementStep)
//   iter                  [int]    Newton iter within step
//   norm                  [float]  ||r||_2 as Newton sees it (SCALED
//                                  when wrapper installed; PHYSICAL
//                                  otherwise)
//   norm0                 [float]  norm at iter 0 of this step
//   norm_max              [float]  Newton's convergence threshold
//   converged_now         [0|1]
//   linear_iterations     [int]    Krylov iterations for this Newton
//                                  correction, or -1 when no linear solve ran
//   linear_final_norm     [float]  Krylov final residual norm, or -1
//   linear_converged      [0|1]
//   scaler_enabled        [0|1]
//   res_K                 [float]  ||r_u||_2, PHYSICAL (un-scaled via
//                                  SaddleResidualScaler::UnapplyToIncrement
//                                  when scaler is enabled)
//   res_lam               [float]  ||r_lam||_2, PHYSICAL
//   res_lam_<label_k>     [float]  ||r_lam^(k)||_2, PHYSICAL
//   d_u                   [float]  current u-block scaling factor
//   d_lam_<label_k>       [float]  current per-sub-block lambda factor

#pragma once

#include "saddle_residual_scaler.hpp"
#include "solvers/mechanics_solver.hpp"   // diagnostic structs + sink types

#include "mfem.hpp"

#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mortar_pbc
{

/**
 * @brief Per-Newton-iter saddle-system diagnostic logger.
 *
 * @details Built once by SystemDriver during mortar setup, BEFORE
 * the Newton solver. Two sinks are exposed:
 *
 *   * `MakeSink()` — pre-solve, install on `ExaNewtonSolver` via
 *     `SetDiagnosticSink`. Buffers a row per Newton iter.
 *   * `MakeLinearSolveSink()` — post-linear-solve, install via
 *     `SetLinearDiagnosticSink`. Fills the buffered row with Krylov
 *     iteration count / final norm and flushes it.
 * Host calls `IncrementStep()` at end of each successful `Solve()`.
 *
 * @par Lifetime
 * The sink captures `this`. Logger must outlive the Newton solver.
 */
class SaddleNewtonDiagnosticLogger
{
public:
    /**
     * @brief Construct (file not yet opened).
     *
     * @param scaler          Non-null. Even on no-scaling runs the
     *                        scaler is constructed (with
     *                        `IsEnabled()==false`) to supply
     *                        partition metadata.
     * @param saddle_offsets  Size-3 `[0, n_u, n_u + n_lam]`. Stored
     *                        by value.
     * @param comm            MPI communicator for per-block norm
     *                        reductions.
     * @param filename        CSV path, default `"newton_iters.csv"`.
     */
    SaddleNewtonDiagnosticLogger(
        std::shared_ptr<const SaddleResidualScaler> scaler,
        const mfem::Array<int>& saddle_offsets,
        MPI_Comm comm,
        const std::string& filename = "newton_iters.csv");

    /// Flushes any leftover pending row.
    ~SaddleNewtonDiagnosticLogger();

    SaddleNewtonDiagnosticLogger(const SaddleNewtonDiagnosticLogger&) = delete;
    SaddleNewtonDiagnosticLogger& operator=(
        const SaddleNewtonDiagnosticLogger&) = delete;
    SaddleNewtonDiagnosticLogger(SaddleNewtonDiagnosticLogger&&) = delete;
    SaddleNewtonDiagnosticLogger& operator=(
        SaddleNewtonDiagnosticLogger&&) = delete;

    /// Pre-solve sink for `ExaNewtonSolver::SetDiagnosticSink`.
    /// Captured lambda asserts `diag.residual != nullptr`.
    NewtonDiagnosticSink MakeSink();

    /// Post-linear-solve sink for `ExaNewtonSolver::SetLinearDiagnosticSink`.
    LinearSolveDiagnosticSink MakeLinearSolveSink();

    /// Advance step counter. Call at end of each successful `Solve()`.
    /// Flushes any pending row first (defensive).
    void IncrementStep();

    int  CurrentStep() const { return m_step_index; }
    const std::string& Filename() const { return m_filename; }

private:
    struct PendingRow
    {
        int step = -1;
        int iter = -1;
        double norm = 0.0;
        double norm0 = 0.0;
        double norm_max = 0.0;
        bool   converged_now = false;
        int    linear_iterations = -1;
        double linear_final_norm = -1.0;
        bool   linear_converged = false;
        bool   scaler_enabled = false;
        double res_K = 0.0;
        double res_lam = 0.0;
        std::vector<double> res_lam_sub;
        double d_u = 1.0;
        std::vector<double> d_lam_sub;

    };

    void OnPreSolve_(const NewtonIterDiagnostic& diag);
    void OnLinearSolve_(const LinearSolveDiagnostic& diag);

    void DecomposeR_(const mfem::Vector& r,
                      double& res_K_phys,
                      double& res_lam_phys,
                      std::vector<double>& res_lam_sub_phys) const;

    void EnsureFileOpen_();
    void WriteHeader_();
    void FlushPending_();

    std::shared_ptr<const SaddleResidualScaler> m_scaler;
    mfem::Array<int>                            m_saddle_offsets;
    MPI_Comm                                    m_comm;
    int                                         m_rank = 0;
    std::string                                 m_filename;
    std::ofstream                               m_file;
    int                                         m_step_index = 0;

    int                                         m_n_subblocks_cached = -1;
    std::vector<std::string>                    m_cached_sub_labels;

    mutable std::optional<PendingRow>           m_pending;
};

}   // namespace mortar_pbc
