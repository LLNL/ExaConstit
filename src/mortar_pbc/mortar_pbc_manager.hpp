// Phase 5.3 / Phase 6 — MortarPbcManager
//
// Coordinator class that wires up the mortar-PBC machinery for use by
// SystemDriver. It owns:
//
//   - A `BoundaryClassifier3D` built on the boundary/LOR surface mesh
//     (collective on the parent ParMesh's communicator).
//   - A `SurfaceProjector` that maps boundary/LOR surface true DOFs
//     back to parent-volume true DOFs.
//   - A projector-aware `ConstraintBuilder3D` (stateless after
//     construction).
//   - A projector-aware `MortarConstraintOperator` — the EA-form C
//     operator that the saddle-point system blocks reference.
//   - A `SaddlePointSolver` — the inner Krylov for one Newton step's
//     `[K C^T; C 0] [du; dlam] = -[r1; r2]` solve.
//   - A `MortarSaddlePointSystem` — the `mfem::Operator` adapter that
//     SystemDriver hands to the Newton solver. The system holds a
//     non-owning pointer to the manager's `m_g_rhs` buffer (installed
//     in the constructor via `SetConstraintRHS`); `UpdateConstraintRHS`
//     refreshes the buffer's contents in place each time step.
//
// And it tracks:
//
//   - The macroscopic deformation gradient `F̄` and its rate `Ḟ`,
//     refreshed once per time step from the velocity-gradient BC.
//   - The accumulated Lagrange multiplier `λ` over a load history
//     (used for periodic-traction post-processing AND for the §12.1
//     Trap 3 convergence-residual contribution `F_int + C^Tλ`).
//   - Per-row reference-geometry caches for §P5.8.6.d
//     (`UpdateConstraintRHS`).
//   - The 24 corner-essential parent-volume TDOFs (8 corners × 3
//     components), pinned to remove rigid-body modes.
//
// Phasing:
//   - 5.3.A: class skeleton + constructor wiring.
//   - 5.3.B: corner essential-TDOF list construction.
//   - 5.3.C.0+1: macroscopic-F update (mesh-anchored — anchors on
//     volume-averaged F from the mesh itself to avoid forward-Euler
//     drift, per Hill-Mandel).
//   - 5.3.C.2: per-row reference factor cache + GPU-friendly
//     constraint RHS update via §P5.8.6.d.
//   - 5.3.D: fluctuation-field projection + current-configuration
//     Hill-Mandel power balance for diagnostics.
//   - 5.3.E: λ accumulation API + `C^Tλ` residual contribution.
//
// References:
//   - PHASE5_EXACONSTIT_INTEGRATION_v4.md §P5.4 (this class) and
//     §P5.8.6 (constraint-RHS formulation).
//   - MORTAR_PBC_ARCHITECTURE.md §11 (Phase 4 mortar machinery),
//     §12.1 (Trap 3 — F_int + C^Tλ convergence).
//   - Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.

#pragma once

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "mortar_constraint_operator.hpp"
#include "mortar_saddle_point_system.hpp"
#include "saddle_point_solver.hpp"
#include "saddle_residual_scaler.hpp"
#include "saddle_scaling_wrappers.hpp"
#include "surface_projector.hpp"

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Coordinator for the mortar-PBC machinery used by SystemDriver.
 *
 * @details Owns a fully-wired set of mortar PBC components and
 * exposes the high-level API SystemDriver uses to integrate
 * mortar-method PBC into the production Newton solver. After
 * construction, the manager is ready to be used as follows in a
 * time-stepping loop:
 *
 * @code
 *   // Once at SystemDriver setup:
 *   auto pbc = std::make_unique<MortarPbcManager>(
 *       sim_state, k_residual, k_jacobian);
 *
 *   // Each time step:
 *   pbc->ResetLambdaAccumulation();
 *   pbc->UpdateMacroscopicF(L_bar, dt);
 *   pbc->UpdateConstraintRHS();
 *
 *   // Each Newton iteration:
 *   nlf->Mult(velocity, residual);
 *   pbc->AddCTransposeLambdaToResidual(residual);  // F_int + C^Tλ
 *   if (||residual|| < tol) break;
 *   saddle_solve(..., dv, dλ);
 *   velocity += dv;
 *   pbc->AccumulateLambdaContribution(dλ);
 *
 *   // End of step diagnostics:
 *   auto hm = pbc->ComputeHillMandelPowerBalance(velocity, residual, L_bar);
 *   pbc->ComputeFluctuationField(velocity, L_bar, fluct_gf);
 * @endcode
 *
 * @par Lifetime
 * The manager holds a `std::shared_ptr<SimulationState>`. All access
 * to the parent mesh, primary FE space, LOR boundary surface mesh,
 * and global quadrature functions goes through the simulation state.
 *
 * @par Phase 6 LOR indexing
 * The classifier operates on `SimulationState::GetLorBoundarySubMesh`
 * and `GetLorBoundarySubMeshFes`, which are linear surface objects.
 * The mechanics solve still owns the parent-volume FE space. The
 * manager therefore constructs a `SurfaceProjector` and passes it to
 * both `ConstraintBuilder3D` and `MortarConstraintOperator`; all
 * runtime vectors, constraint columns, transposed residual
 * contributions, and corner essential TDOFs are expressed in the
 * parent-volume true-DOF numbering. At `lor_depth == 1` this is the
 * direct trace path; at larger depths it is the LOR surface path for
 * higher-order parent elements.
 *
 * @par MPI scope
 * Construction is collective on `sim_state->GetMesh()->GetComm()`.
 * Per-step methods are collective on the same communicator.
 *
 * @par GPU
 * The manager itself is host-only for configuration + small dense
 * state. The `UpdateConstraintRHS` kernel runs via `mfem::forall`
 * with `RAJA::View` for typed access; per-row caches are constructed
 * with `mfem::Device::GetMemoryType()` for GPU residency tracking.
 *
 * @par Thread safety
 * Not thread-safe. One manager per simulation, mutated only from
 * the main MPI thread.
 */
class MortarPbcManager
{
public:
    /// Closure type: compute K-residual `r_K = K(u)`.
    using KResidualFn = MortarSaddlePointSystem::KResidualFn;

    /// Closure type: return the K-Jacobian `dK/du(u)` operator.
    using KJacobianFn = MortarSaddlePointSystem::KJacobianFn;

    /**
     * @brief Diagnostic output of `ComputeHillMandelPowerBalance`.
     *
     * @details Macro side (`sigma_bar`, `d_bar`, `macro_power`,
     * `total_volume`) is always computed. Local side
     * (`integrated_internal_power`) comes from the caller-supplied
     * internal-force vector via the FE residual structure
     * `v · r_internal = ∫ σ:d dV` (σ symmetric eats antisymmetric
     * ∇v).
     *
     * The Hill-Mandel macro-homogeneity condition `⟨σ:d⟩ = σ̄:d̄`
     * equivalently means `∫σ:d dV = σ̄:d̄ · V`. `abs_residual` is the
     * absolute difference; `rel_residual` is normalized by
     * `max(|σ̄:d̄ · V|, eps)`. For a properly-enforced PBC at
     * converged equilibrium, `rel_residual` should be at machine
     * precision in the elastic limit and ~1e-8…1e-10 in nonlinear
     * crystal plasticity (Newton tolerance + integration error).
     */
    struct HillMandelDiagnostic
    {
        /// 3×3 volume-averaged Cauchy stress σ̄.
        mfem::DenseMatrix sigma_bar{3, 3};
        /// 3×3 macro rate of deformation d̄ = (L̄ + L̄^T) / 2.
        mfem::DenseMatrix d_bar{3, 3};
        /// Scalar σ̄:d̄ — macro internal-power *density*.
        double macro_power = 0.0;
        /// Total mesh volume V on the current configuration.
        double total_volume = 0.0;
        /// ∫σ:d dV computed from caller-supplied v · r_internal.
        double integrated_internal_power = 0.0;
        /// |integrated_internal_power - macro_power · V|.
        double abs_residual = 0.0;
        /// abs_residual / max(|macro_power · V|, eps).
        double rel_residual = 0.0;
    };

    /**
     * @brief Construct and wire the full mortar-PBC pipeline.
     *
     * @param sim_state    Shared simulation state. Must already be
     *                     populated with a 3D `ParMesh`, a vector
     *                     H1 FE space (vdim=3), parsed `ExaOptions`,
     *                     the LOR boundary submesh/FES accessors, and
     *                     `"kinetic_grads"` and `"cauchy_stress_end"`
     *                     global quadrature functions (both produced
     *                     by `NonlinearMechOperator` initialization).
     * @param k_residual   User's K-residual callback. See
     *                     `MortarSaddlePointSystem` for semantics.
     * @param k_jacobian   User's K-Jacobian callback. See
     *                     `MortarSaddlePointSystem` for semantics.
     *
     * @par MPI scope
     * Collective on the parent mesh's communicator.
     *
     * @par Validation
     * Aborts via `MFEM_VERIFY` if the LOR boundary submesh cannot be
     * snapped back to the parent FE space by `SurfaceProjector`, if
     * `opts.solvers.saddle_point` parses to an unknown enum value, or
     * if the rank-summed corner TDOF count from
     * `BuildCornerEssTDofs` is not exactly 24.
     */
    MortarPbcManager(std::shared_ptr<SimulationState> sim_state,
                     KResidualFn k_residual,
                     KJacobianFn k_jacobian);

    ~MortarPbcManager() = default;

    // Non-copyable / non-movable.
    MortarPbcManager(const MortarPbcManager&) = delete;
    MortarPbcManager& operator=(const MortarPbcManager&) = delete;

    //==========================================================================
    // State updates — Phase 5.3.C
    //==========================================================================

    /**
     * @brief Update the tracked macroscopic deformation gradient.
     *
     * @details Mesh-anchored Hill-Mandel formulation: anchors on
     * `F̄^{(n)}_mesh = (1/V) ∫ F dV` from the volume-averaged
     * `"kinetic_grads"` QF rather than carrying the previous step's
     * `F̄^{n}_tracked` forward. This eliminates forward-Euler drift
     * across long load histories. Then:
     *
     *     Ḟ̄^{(n+1)} = L̄ · F̄^{(n)}_mesh
     *     F̄^{(n+1)} = F̄^{(n)}_mesh + dt · Ḟ̄^{(n+1)}
     *
     * Called once per time step from SystemDriver before the Newton
     * solve. Anchoring on `F̄^{(n)}_mesh` (NOT `F̄^{(n+1)}`) when
     * computing Ḟ̄ avoids smuggling a second-order `L̄²·dt` term into
     * the rate.
     *
     * @par First step
     * If `det(F̄_mesh) < 0.5` (typically because no integrator pass
     * has touched `kinetic_grads` yet — first call before any
     * Newton solve), falls back to F̄ = I.
     *
     * @param Lbar  Velocity-gradient tensor (3×3).
     * @param dt    Time-step size.
     */
    void UpdateMacroscopicF(const mfem::DenseMatrix& Lbar, double dt);

    /**
     * @brief Refresh the constraint-RHS buffer for the current
     *        macroscopic state.
     *
     * @details Implements §P5.8.6.d: per row i,
     *
     *     g[i] = Ḟ̄_{c, k} · L_k · ℓ̂_i
     *
     * where `c = component_per_row[i]` (which row of Ḟ̄ to project),
     * `k = axis_per_row[i]` (which periodic axis the pair is on),
     * `L_k = axis_lengths[k]` (box length on axis k = ΔX_pair_k for
     * axis-aligned RVEs), and `ℓ̂_i = ell_hat_per_row[i]` (Wohlmuth
     * lumped-row factor on reference geometry).
     *
     * Implementation runs `mfem::forall` over rows with
     * `RAJA::View<const double, RAJA::Layout<2>>` for typed 3×3
     * access to Ḟ̄ — row-major default matches the
     * `kinetic_grads` flat layout.
     *
     * Called once per time step (NOT per Newton iteration); the
     * saddle-point Newton iterates against this fixed RHS until
     * convergence, per §P5.8.6 "off-equilibrium considerations."
     */
    void UpdateConstraintRHS();

    //==========================================================================
    // Diagnostics / output computation — Phase 5.3.D
    //==========================================================================

    /**
     * @brief Project the velocity fluctuation field
     *        \f$\tilde v(x) = v(x) - \bar L \cdot x\f$ onto the FES.
     *
     * @details For diagnostic / visualization. In the mortar PBC
     * formulation, the velocity decomposes additively into an affine
     * macroscopic part and a periodic fluctuation:
     *
     *     v(x) = L̄ · x + ṽ(x)
     *
     * with ṽ enforced periodic via the mortar constraint and the
     * affine part pinned via the corner Dirichlet BCs. Visualizing
     * ṽ is the most direct check that the PBC is being enforced
     * (look for periodicity, vanishing at corners).
     *
     * Implemented via `ParGridFunction::ProjectCoefficient` on a
     * `VectorCoefficient` returning `Lbar · x` at each integration
     * point, then subtracting from `velocity_tdofs`. Allocates a
     * temporary `ParGridFunction`; not a hot path.
     *
     * @param velocity_tdofs  Total velocity in TDOF space.
     * @param Lbar            Prescribed velocity gradient (3×3).
     * @param[out] fluct_gf   Fluctuation field on the manager's FES.
     *                        Sized internally by the implementation.
     */
    void ComputeFluctuationField(const mfem::Vector& velocity_tdofs,
                                 const mfem::DenseMatrix& Lbar,
                                 mfem::ParGridFunction& fluct_gf) const;

    /**
     * @brief Compute the Hill-Mandel power balance in current
     *        configuration.
     *
     * @details Computes σ̄, d̄, σ̄:d̄, V, and the volume-integrated
     * local power \f$\int σ:d \, dV\f$ from the caller-supplied
     * `internal_force_tdofs`. By the FE residual structure,
     *
     *     v · r_internal = ∫σ:∇v dV = ∫σ:d dV
     *
     * (σ symmetric eats the antisymmetric part of ∇v).
     *
     * @par Caveat — un-eliminated residual
     * `nlf->Mult(velocity)` zeros Dirichlet rows of the residual
     * (architecture-doc Trap 4). For a periodic RVE this drops the
     * boundary work term at 24 corner DOFs out of millions —
     * within diagnostic noise floor for any production-scale problem.
     *
     * If you want machine-precision Hill-Mandel, pass the
     * un-eliminated form. The recipe is in
     * `NonlinearMechOperator::GetUpdateBCsAction`
     * (`mechanics_operator.cpp`):
     *
     * @code
     *   mfem::Array<int> zero_tdofs;
     *   h_form->Setup();
     *   h_form->SetEssentialTrueDofs(zero_tdofs);
     *   h_form->Mult(velocity, r_un_eliminated);
     *   h_form->SetEssentialTrueDofs(orig_ess);
     * @endcode
     *
     * @par MPI
     * Collective on `MPI_COMM_WORLD`.
     *
     * @param velocity_tdofs        Total velocity (TDOF space).
     * @param internal_force_tdofs  `nlf->Mult(velocity)` result
     *                              (TDOF space). BC-eliminated or
     *                              not; see caveat above.
     * @param Lbar                  Prescribed velocity gradient.
     * @return Filled `HillMandelDiagnostic`.
     */
    HillMandelDiagnostic ComputeHillMandelPowerBalance(
        const mfem::Vector& velocity_tdofs,
        const mfem::Vector& internal_force_tdofs,
        const mfem::DenseMatrix& Lbar) const;

    /**
     * @brief Phase 5.7.A diagnostic — constraint consistency between
     *        the affine field L̄·x and the installed RHS g.
     *
     * @details Builds v_aff(x) = L̄·x as a FES projection (same
     * `LbarTimesXCoefficient` used by `ComputeFluctuationField`),
     * pulls it to TDOFs, applies the EA constraint operator
     * `C·v_aff`, and compares against `m_g_rhs`.
     *
     * For a consistent mortar formulation, `C·v_aff = g` to machine
     * precision (the constraint encodes the mortar projection of the
     * jump `u(+) - u(-) = L̄·L_k`, which is exactly what `g` is built
     * to enforce). Mismatches surface as one of:
     *   - `||C·v_aff - g||_inf` >> 0 and `||C·v_aff + g||_inf` small
     *     → sign error in `UpdateConstraintRHS`'s `g` formula
     *     relative to `MortarConstraintOperator`'s row convention.
     *   - both diff and sum large, but `||C·v_aff||_inf` close to
     *     `||g||_inf` → structural mismatch (wrong scaling factor,
     *     index permutation, etc.).
     *   - `||C·v_aff||_inf` >> `||g||_inf` → the affine field doesn't
     *     project to a meaningful mortar residual (rare; usually
     *     points at a builder bug).
     *
     * Translation-invariant: any rigid translation of `v_aff` adds a
     * uniform constant to all TDOFs, which `C` zeros out (its rows
     * sum to zero in each component for a matching mortar). So
     * `x_origin` is NOT needed — `L̄·x` and `L̄·(x - x_origin)` give
     * the same `C·v_aff`.
     *
     * @par MPI scope
     * Collective on the FES communicator.
     *
     * @par Cost
     * One `ParGridFunction::ProjectCoefficient` (cheap), one
     * `ParallelProject` to TDOFs, one `m_C_op.Mult`, four
     * `MPI_Allreduce` calls. Negligible compared to a Newton step.
     */
struct ConstraintConsistencyDiagnostic
    {
        double cv_norm_inf = 0.0;
        double g_norm_inf  = 0.0;
        double diff_norm_inf = 0.0;
        double sum_norm_inf = 0.0;
        // Phase 5.11.I — per-pair |Cv-g|_inf. Row r is assigned to
        // pair[k] where k is the FIRST index in {y, x, z} canonical
        // order for which |period[k]| > 0. (See
        // DiagnoseConstraintConsistency for the classification
        // logic.) Edge rows fall to their first-non-zero pair;
        // corner rows likewise. The canonical y→x→z order matches
        // 5.11.B's PER_PAIR sub-block layout and 5.11.G's TRDOG
        // diagnostic ordering.
        double diff_norm_inf_top   = 0.0;   // y-axis pair
        double diff_norm_inf_right = 0.0;   // x-axis pair
        double diff_norm_inf_back  = 0.0;   // z-axis pair

        // Phase 5.7.A extended — rank-local argmax row info.
        //
        // Reports the row at which |g| attains its max on this rank
        // plus the metadata (axis, comp, ell_hat) and the value of
        // `C·v_aff` at that SAME row. Likewise for argmax of |Cv|.
        // For np=1 these ARE the global argmax. For np>1 they are
        // per-rank — only the rank holding the global max will have
        // matching values to the corresponding `*_norm_inf` field.

        int argmax_g_row = -1;
        // Phase 5.7.A — replaces single-axis index. Full periodic
        // shift vector (Δx·L_x, Δy·L_y, Δz·L_z) at the argmax row.
        std::array<double, 3> argmax_g_period = {0.0, 0.0, 0.0};
        int argmax_g_comp = -1;
        double argmax_g_ell = 0.0;
        double argmax_g_g_val = 0.0;
        double argmax_g_cv_val = 0.0;

        int argmax_cv_row = -1;
        std::array<double, 3> argmax_cv_period = {0.0, 0.0, 0.0};
        int argmax_cv_comp = -1;
        double argmax_cv_ell = 0.0;
        double argmax_cv_g_val = 0.0;
        double argmax_cv_cv_val = 0.0;
        // Phase 5.7.A — argmax(|C·v_aff - g|) row. Localizes the
        // remaining discretization-level residual. Cv and g values
        // at this row are signed so the residual's character
        // (cancellation vs additive) is visible.
        int argmax_diff_row = -1;
        std::array<double, 3> argmax_diff_period = {0.0, 0.0, 0.0};
        int argmax_diff_comp = -1;
        double argmax_diff_ell = 0.0;
        double argmax_diff_g_val = 0.0;
        double argmax_diff_cv_val = 0.0;
        double argmax_diff_val = 0.0;   // Cv - g, signed
    };

    /**
     * @brief Compute the constraint-consistency diagnostic.
     *
     * @param Lbar  Velocity gradient L̄ (3×3). Caller supplies the
     *              same L̄ that `UpdateMacroscopicF` was called with.
     * @return Populated diagnostic.
     */
    ConstraintConsistencyDiagnostic DiagnoseConstraintConsistency(
        const mfem::DenseMatrix& Lbar) const;

    /**
     * @brief Phase 5.8 — project v_lin(x) = L̄·x onto the FES.
     *
     * @details Complementary to `ComputeFluctuationField`. Together
     * they satisfy v_total(x) = v_lin(x) + v_tilde(x) at every TDOF.
     * Reuses the `LbarTimesXCoefficient` machinery internally (same
     * coefficient used by `ComputeFluctuationField` and
     * `DiagnoseConstraintConsistency`); not a hot path.
     *
     * Useful as a reference field for visualization comparisons
     * against v_tilde, and for downstream post-processing that
     * needs the affine part isolated.
     *
     * @param Lbar           Velocity gradient (3×3). Typically
     *                       sourced from `GetLbar()` for consistency
     *                       with the most recent `UpdateMacroscopicF`
     *                       call.
     * @param[out] v_lin_gf  Grid function to populate. Sized
     *                       internally by the implementation.
     */
    void ComputeAffineVelocityField(const mfem::DenseMatrix& Lbar,
                                    mfem::ParGridFunction& v_lin_gf) const;

    /**
     * @brief Phase 5.8 — cache per-step diagnostic structs for
     *        downstream post-processing readout.
     *
     * @details Computes BOTH the `ConstraintConsistencyDiagnostic`
     * and the `HillMandelDiagnostic` from the current converged
     * state and stores them in member fields. Intended hook point:
     * `SystemDriver::Solve()` end-of-step, gated by
     * `[PostProcessing.volume_averages] periodic_validation`.
     *
     * The `PostProcessingDriver` then retrieves the cached structs
     * via `GetLastConstraintConsistencyDiagnostic()` and
     * `GetLastHillMandelDiagnostic()` for per-step text-file output.
     * Caching avoids duplicating the underlying compute work and
     * decouples the post-processor from the K-residual / Lbar
     * plumbing required by the underlying diagnostic methods.
     *
     * Uses the manager's stored `m_Lbar` (set by the most recent
     * `UpdateMacroscopicF` call).
     *
     * @par MPI
     * Collective on the FES communicator.
     *
     * @param velocity_tdofs        Total velocity (TDOF space).
     * @param internal_force_tdofs  `nlf->Mult(velocity)` result
     *                              (TDOF space). See
     *                              `ComputeHillMandelPowerBalance`
     *                              for the un-eliminated-residual
     *                              note.
     */
    void CachePerStepDiagnostics(const mfem::Vector& velocity_tdofs,
                                 const mfem::Vector& internal_force_tdofs);

    //==========================================================================
    // Lambda accumulation — Phase 5.3.E
    //==========================================================================

    /**
     * @brief Accumulate a Newton-step λ contribution into the
     *        manager's running λ buffer.
     *
     * @details `m_lambda += scale * dlam`. Called from SystemDriver
     * after each successful Newton solve to keep a running total
     * across the load history (used for periodic-traction output and
     * for the §12.1 Trap 3 convergence residual `F_int + C^Tλ`).
     *
     * @param dlam   Newton increment to the multiplier (size
     *               `NumLocalConstraints()`).
     * @param scale  Scale factor (typically 1.0; the load-step
     *               weight if Newton is sub-stepped).
     */
    void AccumulateLambdaContribution(const mfem::Vector& dlam,
                                      double scale = 1.0);

    /**
     * @brief Replace the accumulated `λ` buffer with the supplied
     *        vector.
     *
     * @details Used by SystemDriver (Phase 5.5) to write the
     * converged λ from the saddle Newton's lower block back into the
     * manager's persistent buffer, so it survives across time steps
     * as the warm-start for the next step's first Newton iteration
     * (architecture doc §12.1 Trap 3 / v4 plan §P5.14.4).
     *
     * Distinct from `AccumulateLambdaContribution` which adds an
     * incremental `δλ`. `SetAccumulatedLambda` overwrites — there's
     * no scale factor, no addition.
     *
     * @param lambda  New λ values. Size must equal
     *                `NumLocalConstraints()`.
     */
    void SetAccumulatedLambda(const mfem::Vector& lambda);
                                    
    /**
     * @brief Reset the accumulated λ buffer to zero.
     *
     * @details Typical usage: called once at the start of each
     * time step, then `AccumulateLambdaContribution` runs each
     * Newton iteration thereafter.
     */
    void ResetLambdaAccumulation();

    /**
     * @brief Add the `C^T·λ` contribution to a residual vector.
     *
     * @details At converged equilibrium of the saddle-point system,
     * `F_int = -C^T·λ` (NOT zero — that's Trap 3 of the v4
     * architecture doc). The right convergence residual is therefore
     * `F_int + C^T·λ`. This method delegates to the constraint
     * operator's `MultTranspose(m_lambda, tmp)` and adds the result
     * to `residual`.
     *
     * Allocates a single temporary `Vector(Width)` per call; not a
     * hot path but called once per Newton iteration in 5.4.
     *
     * @par MPI
     * Collective on the constraint operator's communicator.
     *
     * @param[in,out] residual  Vector to accumulate into. Size
     *                          must equal C's column count
     *                          (= FES TrueVSize).
     */
    void AddCTransposeLambdaToResidual(mfem::Vector& residual) const;

    //==========================================================================
    // Phase 5.9 — Spec-driven rebuild (Batch A.4)
    //==========================================================================

    /**
     * @brief Phase 5.9 / Batch A.4 — repopulate constraint state for
     *        a new `(essential_ids, essential_comps)` periodic-BC spec.
     *
     * @details Orchestrates the per-spec rebuild across the manager's
     * owned components:
     *
     *   1. Translate `essential_comps` (1..7 via
     *      `BCData::GetComponents` — 1=X, 2=Y, 3=Z, 4=XY, 5=XZ, 6=YZ,
     *      7=XYZ) into `std::array<bool,3> comp_mask`.
     *   2. Validate pair completeness: every face attribute in
     *      `essential_ids` must have its pair partner attribute also
     *      in the list. On failure, aborts with a message naming the
     *      missing attr + label.
     *   3. Derive canonical `active_pair_labels` (mortar-side labels)
     *      from the validated `essential_ids`.
     *   4. Call `m_C_op->Reset(active_pair_labels, comp_mask)` —
     *      rebuilds the EA constraint operator's flat-row arrays.
     *   5. Recompute `m_corner_ess_tdofs` via
     *      the projector-aware `ComputeCornerEssTDofsFromSpec` —
     *      anchor "blf" corner always pinned in all 3 components,
     *      other 7 corners pinned per `comp_mask`, and every selected
     *      boundary/LOR submesh TDOF translated to the parent FE space.
     *   6. Resize `m_lambda` and `m_g_rhs` to the new local row
     *      count `m_C_op->Height()` and zero both. (The saddle system
     *      holds a pointer to `m_g_rhs` via `SetConstraintRHS` at
     *      construction time; `SetSize` preserves the Vector's
     *      address, so the pointer remains valid.)
     *   7. Re-emit per-row reference factors
     *      (`m_period_signed_per_row`, `m_component_per_row`,
     *      `m_ell_hat_per_row`) via the filtered overload of
     *      `ConstraintBuilder3D::EmitRowFactors`.
     *
     * @par MPI scope
     * **Local — no MPI calls.** `MortarConstraintOperator::Reset`,
     * the projector-aware `ComputeCornerEssTDofsFromSpec`, and
     * `ConstraintBuilder3D::
     * EmitRowFactors` are all local on this rank. All ranks must
     * call `RebuildForActiveSpec` with identical arguments
     * (collective by convention — the same agreement requirement
     * already holds for `MortarConstraintOperator::Reset`).
     *
     * @par Rotation RBM caveat
     * Anchor pinning removes the 3 translation rigid-body modes
     * unconditionally. Rotation RBMs are NOT auto-handled. For sub-
     * XYZ specs (e.g. X-only), the user must add corner Dirichlet
     * BCs manually via the regular BC machinery if rotation modes
     * would otherwise be unconstrained for their problem.
     *
     * @param essential_ids   Boundary face attributes covered by the
     *                        periodic BC. Both halves of every pair
     *                        must be present.
     * @param essential_comps Component bitmask 1..7 per
     *                        `BCData::GetComponents`. Aborts on out-of-
     *                        range values.
     */
    void RebuildForActiveSpec(const std::vector<int>& essential_ids,
                              int essential_comps);

    /**
     * @brief Phase 5.9 / Batch A.4 — synthesize a default
     *        `(essential_ids, essential_comps)` spec covering ALL
     *        face pairs in the classifier with `comps = 7` (XYZ).
     *
     * @details Intended call site is `SystemDriver` startup when the
     * user's TOML does not contain a `[[BCs.periodic_bcs]]` block.
     * Returned spec, when passed to `RebuildForActiveSpec`, reproduces
     * the pre-5.9 fully-constrained behavior bit-for-bit.
     *
     * Both halves of every pair are emitted into `essential_ids`,
     * with deduplication (defensive — duplicates wouldn't occur for
     * a well-formed classifier but the dedup is cheap).
     *
     * @par MPI scope
     * Local — no MPI calls. The classifier's `FacePairs()` and
     * `MeshAttributeForLabel` accessors are pure lookups on
     * already-built state.
     */
    static std::pair<std::vector<int>, int> SynthesizeDefaultPbcSpec(
        const BoundaryClassifier3D& classifier);

    /**
     * @brief Phase 5.9 / Batch A.4 — current active pair labels
     *        passthrough.
     *
     * @details Equals the EA constraint operator's
     * `ActivePairLabels()` after the most recent
     * `RebuildForActiveSpec` call. Before any `RebuildForActiveSpec`
     * call, the operator's default-filter spec is in effect (all
     * mortar labels active). Exposed for diagnostic printing and
     * test introspection.
     */
    const std::vector<std::string>& GetActivePairLabels() const
    {
        return m_C_op->ActivePairLabels();
    }

    /**
     * @brief Phase 5.11.E — pick d_u and per-sub-block d_lambda from
     *        the current residual norms.
     *
     * @details Collective on the parallel-mesh communicator.
     * Computes local sums of squares for `r_phys.GetBlock(0)` (u
     * block) and per-sub-block on `r_phys.GetBlock(1)` (lambda
     * block), packs them into a single (1 + n_subblocks)-entry
     * buffer, MPI_Allreduces with `MPI_SUM`, takes sqrt to get the
     * global L2 norms, and feeds them to `m_scaler->Choose`. The
     * single Allreduce is the per-step protocol from the planning
     * doc §6.1.
     *
     * No-op when `m_scaler->IsEnabled()` is false — preserves
     * pre-5.11 bit-for-bit behavior. Otherwise, populates the
     * scaler's d_u and per-row m_d_lambda with Rule A unit-balance
     * values (floor + range-cap guarded per
     * `SaddleResidualScalerConfig`).
     *
     * Intended call site is `SystemDriver` (Phase 5.11.H), once per
     * load step after `SyncMortarPbcForStep` (which may have done a
     * filter-change `RebuildForActiveSpec` that resized the lambda
     * block) and before the Newton solver's first iteration.
     *
     * @param r_phys  Initial physical residual at the start of this
     *                load step. Block 0 = u (TDOF length); block 1 =
     *                lambda (rank-local constraint row count, must
     *                match the current `m_C_op.Height()`).
     *
     * @par MPI scope
     * Collective on `m_sim_state->GetMesh()->GetComm()`. All ranks
     * must call (the Allreduce is unconditional within the enabled
     * branch).
     */
    void ChooseScalingForStep(const mfem::BlockVector& r_phys);

    /**
     * @brief Phase 5.9 / Batch A.4 — current component mask
     *        passthrough.
     */
    const std::array<bool, 3>& GetCompMask() const
    {
        return m_C_op->CompMask();
    }

    //==========================================================================
    // Read-only accessors
    //==========================================================================

    const BoundaryClassifier3D& GetClassifier() const
    {
        return *m_classifier;
    }

    const MortarConstraintOperator& GetConstraintOperator() const
    {
        return *m_C_op;
    }

    SaddlePointSolver& GetSaddleSolver() { return m_saddle_solver; }
    const SaddlePointSolver& GetSaddleSolver() const { return m_saddle_solver; }

    std::shared_ptr<MortarSaddlePointSystem> GetSaddleSystem()
    {
        return m_saddle_system;
    }

    /**
     * @brief Phase 5.11.E — scaled view of the saddle system.
     *
     * @details The `ScaledSaddleOperator` wraps `m_saddle_system`
     * (returned by `GetSaddleSystem()`) and produces `r_solver =
     * D^-1 r_phys` from `Mult`, with `GetGradient` returning a
     * `ScaledJacobianOperator` for the inner Krylov. Always non-null;
     * when scaling is disabled it's still bit-for-bit identical to
     * the wrapped inner because identity scaling reduces all
     * Apply/Unapply operations to multiplications by 1.0 (exact in
     * IEEE-754).
     *
     * `SystemDriver` (Phase 5.11.H) chooses between this wrapper and
     * the raw `m_saddle_system` based on `GetScaler()->IsEnabled()`.
     */
    std::shared_ptr<ScaledSaddleOperator> GetScaledSaddleSystem()
    {
        return m_scaled_saddle_system;
    }

    /**
     * @brief Phase 5.11.E — scaling state for the saddle system.
     *
     * @details Always non-null. `m_scaler->IsEnabled()` indicates
     * whether the scaling path is active for this configuration;
     * when false, the scaler's d_u and d_lambda stay at 1.0
     * (identity scaling) and downstream consumers should short-
     * circuit to the unwrapped saddle operator path for bit-for-bit
     * parity with pre-5.11 behavior.
     */
    std::shared_ptr<SaddleResidualScaler>       GetScaler()       { return m_scaler; }
    std::shared_ptr<const SaddleResidualScaler> GetScaler() const { return m_scaler; }

    /**
     * @brief Phase 5.11.E — saddle-system block offsets used by the
     *        5.11.D scaling wrappers and 5.11.G TRDOG.
     *
     * @details `{0, n_u_local, n_u_local + n_lambda_local}`. Rebuilt
     * by `RebuildForActiveSpec` whenever the constraint row count
     * changes (Phase 5.9 filter spec switch).
     */
    const mfem::Array<int>& GetSaddleBlockOffsets() const {
        return m_saddle_block_offsets;
    }

    /**
     * @brief Rank-local list of corner-pinned TDOFs.
     *
     * @details Pre-5.9 (or after construction without a
     * `RebuildForActiveSpec` call): rank-summed size is 24 (8 corners
     * × 3 components — full XYZ pinning).
     *
     * Post-5.9, after `RebuildForActiveSpec(essential_ids,
     * essential_comps)`: rank-summed size depends on `essential_comps`.
     * The anchor "blf" corner contributes 3 components unconditionally;
     * the 7 other corners contribute one entry per component in the
     * derived `comp_mask`. So for `essential_comps == 7` (XYZ) → 24;
     * for `essential_comps == 1` (X-only) → 3 + 7×1 = 10; etc.
     *
     * Filled in 5.3.B via `BuildCornerEssTDofs` (default-XYZ path);
     * replaced in 5.9 via `RebuildForActiveSpec`.
     */
    const mfem::Array<int>& GetCornerEssTDofs() const
    {
        return m_corner_ess_tdofs;
    }

    /// Current macroscopic deformation gradient (3×3). Identity at
    /// construction; updated by `UpdateMacroscopicF`.
    const mfem::DenseMatrix& GetMacroscopicF() const { return m_macro_F; }

    /// Current macroscopic deformation-rate `Ḟ` (3×3). Zero at
    /// construction; updated by `UpdateMacroscopicF`.
    const mfem::DenseMatrix& GetMacroscopicFdot() const { return m_macro_Fdot; }

    /**
     * @brief Phase 5.8 — velocity gradient most recently passed to
     *        `UpdateMacroscopicF`.
     *
     * @details Zero matrix at construction. Stored so that downstream
     * callers (notably `PostProcessingDriver::PrintPeriodicValidation`)
     * can invoke the diagnostic methods without re-plumbing L̄ from
     * `BCManager`. The manager's three diagnostic methods
     * (`ComputeFluctuationField`, `ComputeHillMandelPowerBalance`,
     * `DiagnoseConstraintConsistency`) and the new
     * `ComputeAffineVelocityField` all take L̄ explicitly, so callers
     * needing consistency with the current macro state can pass
     * `GetLbar()`.
     */
    const mfem::DenseMatrix& GetLbar() const { return m_Lbar; }

    /**
     * @brief Phase 5.8 — most recently cached
     *        `ConstraintConsistencyDiagnostic`.
     *
     * @details Populated by `CachePerStepDiagnostics`.
     * Zero-initialized (cv_norm_inf = g_norm_inf = ... = 0) before
     * any call. Read by post-processing for per-step text-file
     * output.
     */
    const ConstraintConsistencyDiagnostic&
    GetLastConstraintConsistencyDiagnostic() const
    {
        return m_last_consistency_diag;
    }

    /**
     * @brief Phase 5.8 — most recently cached `HillMandelDiagnostic`.
     *
     * @details Populated by `CachePerStepDiagnostics`.
     * Zero-initialized before any call. Read by post-processing.
     */
    const HillMandelDiagnostic& GetLastHillMandelDiagnostic() const
    {
        return m_last_hill_mandel_diag;
    }

    /// Accumulated λ over the load history. Size =
    /// `NumLocalConstraints()`. Zero at construction and after
    /// `ResetLambdaAccumulation`.
    const mfem::Vector& GetAccumulatedLambda() const { return m_lambda; }

    /// Number of constraint rows owned by this rank
    /// (= `m_C_op->Height()` = `m_builder->NumLocalRows()`).
    int NumLocalConstraints() const { return m_C_op->Height(); }

    /**
     * @brief Phase 5.5.B.4 — current constraint RHS vector `g`.
     *
     * @details The saddle-point system's constraint residual is
     * `r_lam = C·u - g`; `g` is refreshed by
     * `UpdateConstraintRHS()` at each time step from the current
     * macroscopic `Ḟ̄`. The saddle system holds a non-owning
     * pointer to this buffer (installed at construction via
     * `MortarSaddlePointSystem::SetConstraintRHS`); changes to
     * `m_g_rhs` are picked up automatically by subsequent
     * `MortarSaddlePointSystem::Mult` calls.
     *
     * Used by SystemDriver's mortar `SolveInit` branch, which
     * runs a one-shot linearized saddle solve and needs to
     * compute `r2 = C·u_prev - g`.
     */
    const mfem::Vector& GetConstraintRHS() const { return m_g_rhs; }


private:
    //--------------------------------------------------------------------------
    // Private helpers
    //--------------------------------------------------------------------------

    /// Phase 5.3.B / Phase 6 — populate `m_corner_ess_tdofs` with
    /// rank-local parent-volume TDOFs for the 8 box corners (3
    /// components each, filtered to only those owned by this rank).
    /// Delegates to the projector-aware `ComputeCornerEssTDofs`
    /// overload declared below the class, then performs the global
    /// 24-entry sanity check.
    void BuildCornerEssTDofs();

    /// Phase 5.3.C.2 / Phase 6 — populate per-row caches
    /// (`period_signed_per_row`, component index, and Wohlmuth
    /// lumped-row factor) from the projector-aware builder. Called
    /// once at construction and after each active-spec rebuild.
    void BuildReferenceGeometricFactors();

    /// Phase 5.3.D — volume-averaged deformation gradient (Voigt 9
    /// row-major: `[F11, F12, F13, F21, F22, F23, F31, F32, F33]`).
    /// Wraps `ComputeVolAvgTensorFromPartial<true>` on the global
    /// `"kinetic_grads"` partial QF with `MPI_COMM_WORLD`. Used by
    /// `UpdateMacroscopicF`. Returns total mesh volume V.
    double ComputeVolumeAveragedF(mfem::Vector& F_voigt9) const;

    /// Phase 5.3.D — volume-averaged Cauchy stress (Voigt 6:
    /// `[σxx, σyy, σzz, σxy, σxz, σyz]`). Wraps
    /// `ComputeVolAvgTensorFromPartial<true>` on the global
    /// `"cauchy_stress_end"` partial QF with `MPI_COMM_WORLD`. Used
    /// by `ComputeHillMandelPowerBalance`. Returns total mesh
    /// volume V.
    double ComputeVolumeAveragedCauchyStress(mfem::Vector& sigma_voigt) const;

    //--------------------------------------------------------------------------
    // Member state
    //
    // Declaration order matters: members are initialized in declaration
    // order, not initializer-list order. The dependency chain is
    //   sim_state → classifier → builder → C_op → saddle_solver →
    //   saddle_system,
    // so they're declared in that order below.
    //--------------------------------------------------------------------------

    /// Reference to the simulation state (mesh, FES, options, QFs).
    /// Held by shared ownership.
    std::shared_ptr<SimulationState> m_sim_state;

    // Owned components (initialized in dependency order).
    //
    // Phase 6 stores these behind shared ownership because the
    // projector-aware builder and operator both need stable shared
    // handles to the LOR-boundary classifier, the surface projector,
    // and the parent FE space. Public accessors still return
    // references so downstream callers do not observe the ownership
    // change.
    std::shared_ptr<BoundaryClassifier3D>     m_classifier;
    std::shared_ptr<SurfaceProjector>         m_projector;
    std::shared_ptr<ConstraintBuilder3D>      m_builder;
    std::shared_ptr<MortarConstraintOperator> m_C_op;
    SaddlePointSolver                        m_saddle_solver;

    // Phase 5.5.B.4 — saddle system stored as shared_ptr so it can
    // be handed to ExaNewtonSolver via SetOperator(shared_ptr<Operator>).
    // The manager constructs it on the heap; SystemDriver receives a
    // copy of the shared_ptr via GetSaddleSystemShared(). Constructed
    // before m_g_rhs because m_g_rhs is the buffer the saddle system
    // points at, but we install the pointer in the ctor body so the
    // declaration order between the two is decoupled.
    std::shared_ptr<MortarSaddlePointSystem> m_saddle_system;

    // Phase 5.11.E — scaling state for the saddle system. See the
    // public accessors `GetScaler` / `GetScaledSaddleSystem` for
    // semantics. Both shared_ptrs are non-null post-ctor.
    std::shared_ptr<SaddleResidualScaler> m_scaler;
    std::shared_ptr<ScaledSaddleOperator> m_scaled_saddle_system;
    mfem::Array<int>                      m_saddle_block_offsets;


    // State buffers (Vector members initialized with explicit memory
    // type for GPU residency tracking).
    mfem::Array<int>             m_corner_ess_tdofs;
    mfem::Vector                 m_lambda;
    mfem::Vector                 m_g_rhs;

    // Macroscopic state — small dense (3×3) matrices, host-only.
    // m_macro_Fdot is copied into a Vector(9) at the top of each
    // UpdateConstraintRHS call for device-side access.
    mfem::DenseMatrix            m_macro_F;
    mfem::DenseMatrix            m_macro_Fdot;

    // Phase 5.8 — velocity gradient most recently passed to
    // UpdateMacroscopicF. Stored so post-processing can re-invoke
    // the diagnostic methods without re-plumbing Lbar through its
    // own state. Host-only 3×3 dense matrix.
    mfem::DenseMatrix            m_Lbar;

    // Phase 5.8 — cached diagnostic outputs populated by
    // CachePerStepDiagnostics (called from SystemDriver::Solve()
    // end-of-step when periodic_validation is enabled). Read by
    // PostProcessingDriver::PrintPeriodicValidation. Mutable
    // copies of the structs; default-zero-initialized.
    ConstraintConsistencyDiagnostic m_last_consistency_diag;
    HillMandelDiagnostic            m_last_hill_mandel_diag;

    // Phase 5.7.A — per-row period-signed vector replaces the prior
    // `m_axis_per_row` (single axis index) and `m_axis_lengths`
    // (3 box lengths). `period_signed_per_row` is row-major of
    // length `3 * n_rows`: for row i, components
    // `[3i, 3i+1, 3i+2]` are the physical periodic shift along
    // (x, y, z). See ConstraintBuilder3D::EmitRowFactors docstring.
    mfem::Vector                 m_period_signed_per_row;
    mfem::Array<int>             m_component_per_row;
    mfem::Vector                 m_ell_hat_per_row;
};

/**
 * @brief Compute rank-local TDOFs for the 8 box corners of a
 *        classified RVE boundary.
 *
 * @details Iterates the classifier's 8 corner records (replicated on
 * every rank); for each corner's three components (x/y/z), tests
 * whether the global TDOF is owned by this rank using
 * `classifier.GtdofOwnerRank`. Owned components are converted to
 * rank-local indices via `fes.GetMyTDofOffset()` and appended to the
 * output array.
 *
 * Exposed as a free function (rather than baked into
 * `MortarPbcManager::BuildCornerEssTDofs`) so it can be exercised
 * by `test_mortar_pbc_manager.cpp` in isolation, without the cost
 * of constructing a full `SimulationState`.
 *
 * @par Postcondition
 * Across the classifier's communicator,
 * `MPI_Allreduce(SUM, output.Size())` equals 24 (8 corners × 3
 * components). Each rank-local entry is a valid TDOF in
 * `[0, fes.GetTrueVSize())`.
 *
 * @param classifier  Fully-built `BoundaryClassifier3D`.
 * @param fes         Vector H1 FE space the classifier was built on.
 *
 * @return Rank-local list of corner essential TDOFs.
 */
mfem::Array<int> ComputeCornerEssTDofs(
    const BoundaryClassifier3D& classifier,
    const mfem::ParFiniteElementSpace& fes);

/**
 * @brief Compute parent-FES rank-local TDOFs for the 8 corners of a
 *        classifier built on a boundary/LOR submesh.
 *
 * @details Phase 6 keeps the classifier on the linear boundary/LOR
 * surface so mortar rows are built on the LOR mesh. Essential
 * boundary conditions, however, must still be applied to the parent
 * volume FE space used by the mechanics solve. This overload mirrors
 * the legacy `ComputeCornerEssTDofs(classifier, fes)` algorithm but
 * translates each classifier-side submesh global true DOF through
 * `projector.ParentGtdof()` before testing ownership in
 * `parent_fes`.
 *
 * For `lor_depth == 1` and a linear parent space, the projector is
 * the identity trace permutation and the result is bit-for-bit
 * equivalent to the legacy path. For higher-order parent spaces, the
 * returned local TDOFs are parent-volume TDOFs at the corner
 * Lagrange nodes.
 *
 * @par MPI scope
 * Local — no MPI calls. The caller may perform the same global-count
 * sanity check as the legacy path (`SUM(Size()) == 24`).
 *
 * @param classifier  Classifier built on the boundary/LOR submesh
 *                    FE space.
 * @param projector   Surface projector mapping classifier-side
 *                    submesh true DOFs to parent-FES true DOFs.
 * @param parent_fes  Parent volume FE space whose local TDOF indices
 *                    are returned.
 *
 * @return Rank-local parent-FES corner essential TDOFs.
 */
mfem::Array<int> ComputeCornerEssTDofs(
    const BoundaryClassifier3D& classifier,
    const SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes);

/**
 * @brief Phase 5.9 / Batch A.4 — compute rank-local corner-pinned
 *        TDOFs under a per-component filter, gated by which faces
 *        the corner is incident on.
 *
 * @details The anchor "blf" corner (bottom-left-front, min in all
 * three coordinates) is ALWAYS pinned in all three components,
 * removing the 3 translation rigid-body modes unconditionally.
 *
 * The 7 non-anchor corners are pinned per the **incident-face gate**
 * + `comp_mask` filter. A corner is eligible iff at least one of
 * the boundary face attributes it sits on is present in
 * `essential_ids`. For eligible corners, the c-component TDOF is
 * appended iff `comp_mask[c] == true`.
 *
 * On a standard axis-aligned 6-face RVE, the incident-face gate is
 * vacuous: every corner is on three of the six box faces, so any
 * `essential_ids` covering at least one complete axis-pair makes
 * all 8 corners eligible. (Phase 5.9.A.4's documentation has the
 * full enumeration.) The gate is implemented explicitly anyway
 * because the spec calls for it and the cost is negligible.
 *
 * For `comp_mask = {true, true, true}` and `essential_ids` covering
 * all 6 faces, the rank-summed result is 24 TDOFs, matching the
 * pre-5.9 `ComputeCornerEssTDofs` behavior. For `essential_ids =
 * {left, right}` (X-pair only) and `comp_mask = {true, false, false}`
 * (X-only): all 8 corners are incident on left or right, so the
 * rank-summed size is 3 (anchor) + 7×1 = 10.
 *
 * @par Rotation RBM caveat
 * Anchor pinning alone removes translation modes. For sub-XYZ
 * `comp_mask`, rotation modes in the filtered components may
 * remain unconstrained. Callers needing rotation pinning should add
 * additional Dirichlet BCs via the regular BC machinery.
 *
 * @par Anchor label convention
 * Uses `classifier.AnchorCornerTDofs(fes)` (Phase 5.9.A.2) to
 * obtain the anchor's 3 component TDOFs in rank-local form. The
 * anchor label is "blf" per the classifier's documentation.
 *
 * @par MPI scope
 * Local — no MPI calls. Mirrors the no-MPI scope of
 * `ComputeCornerEssTDofs`.
 *
 * @param classifier     Fully-built `BoundaryClassifier3D`.
 * @param fes            Vector H1 FE space the classifier was built
 *                       on.
 * @param essential_ids  Boundary face attributes covered by the
 *                       active periodic-BC spec. Used to determine
 *                       which non-anchor corners are eligible for
 *                       pinning (via
 *                       `classifier.CornersOnFaceAttribute`).
 * @param comp_mask      Per-spatial-component filter on eligible
 *                       corners. `comp_mask[c]` determines whether
 *                       eligible non-anchor corners contribute the
 *                       c-component TDOF.
 *
 * @return Rank-local list of corner essential TDOFs.
 */
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D& classifier,
    const mfem::ParFiniteElementSpace& fes,
    const std::vector<int>& essential_ids,
    const std::array<bool, 3>& comp_mask);

/**
 * @brief Projector-aware spec-filtered corner pinning for Phase 6.
 *
 * @details This overload is the LOR-boundary equivalent of
 * `ComputeCornerEssTDofsFromSpec(classifier, fes, essential_ids,
 * comp_mask)`. It applies the same semantic rules:
 *   - anchor corner "blf" is pinned in all three components;
 *   - non-anchor corners are gated by incident face attributes in
 *     `essential_ids`;
 *   - eligible non-anchor components are filtered by `comp_mask`.
 *
 * The difference is index space: the classifier's corner records
 * contain boundary/LOR-submesh true DOFs, while the returned list must
 * be valid for the parent volume FE space. Every selected component is
 * translated through `SurfaceProjector` before ownership and local
 * index conversion are evaluated against `parent_fes`.
 *
 * @par MPI scope
 * Local — no MPI calls. All ranks must call it with identical
 * `essential_ids` and `comp_mask`, matching the legacy filtered path.
 *
 * @param classifier     Classifier built on the boundary/LOR submesh.
 * @param projector      Submesh-to-parent true-DOF translator.
 * @param parent_fes     Parent volume FE space whose local TDOF
 *                       numbering is returned.
 * @param essential_ids  Boundary face attributes covered by the
 *                       active periodic-BC spec.
 * @param comp_mask      Per-spatial-component filter for non-anchor
 *                       corners.
 *
 * @return Rank-local parent-FES corner essential TDOFs.
 */
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D& classifier,
    const SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes,
    const std::vector<int>& essential_ids,
    const std::array<bool, 3>& comp_mask);

}  // namespace mortar_pbc
