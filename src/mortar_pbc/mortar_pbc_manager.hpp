// Phase 5.3 — MortarPbcManager
//
// Coordinator class that wires up the mortar-PBC machinery for use by
// SystemDriver. It owns:
//
//   - A `BoundaryClassifier3D` (built once at construction; collective
//     on the parent ParMesh's communicator).
//   - A `ConstraintBuilder3D` (stateless after construction).
//   - A `MortarConstraintOperator` — the EA-form C operator that the
//     saddle-point system blocks reference.
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
//   - The 24 corner-essential TDOFs (8 corners × 3 components),
//     pinned to remove rigid-body modes.
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

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <memory>

namespace mortar_pbc {

/**
 * @brief Coordinator for the Phase 5 mortar-PBC machinery.
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
 * to the parent mesh, primary FE space, and global quadrature
 * functions goes through the simulation state.
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
     *                     H1 FE space (vdim=3, order 1 in Phase 5),
     *                     parsed `ExaOptions`, and the
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
     * Aborts via `MFEM_VERIFY` if `opts.mesh.lor_depth != 1` (Phase 6
     * stub), if `opts.solvers.saddle_point` parses to an unknown
     * enum value, or if the rank-summed corner TDOF count from
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
    // Read-only accessors
    //==========================================================================

    const BoundaryClassifier3D& GetClassifier() const
    {
        return m_classifier;
    }

    const MortarConstraintOperator& GetConstraintOperator() const
    {
        return m_C_op;
    }

    SaddlePointSolver& GetSaddleSolver() { return m_saddle_solver; }
    const SaddlePointSolver& GetSaddleSolver() const { return m_saddle_solver; }

    std::shared_ptr<MortarSaddlePointSystem> GetSaddleSystem()
    {
        return m_saddle_system;
    }

    /// 24-element list of corner-pinned TDOFs (filled in 5.3.B).
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

    /// Accumulated λ over the load history. Size =
    /// `NumLocalConstraints()`. Zero at construction and after
    /// `ResetLambdaAccumulation`.
    const mfem::Vector& GetAccumulatedLambda() const { return m_lambda; }

    /// Number of constraint rows owned by this rank
    /// (= `m_C_op.Height()` = `m_builder.NumLocalRows()`).
    int NumLocalConstraints() const { return m_C_op.Height(); }

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

    /// Phase 5.3.B — populate `m_corner_ess_tdofs` with the rank-local
    /// TDOFs for the 8 box corners (3 components each, filtered to
    /// only those owned by this rank). Delegates to the free function
    /// `ComputeCornerEssTDofs` (declared below the class) plus an
    /// MPI sanity check.
    void BuildCornerEssTDofs();

    /// Phase 5.3.C.2 — populate per-row caches (axis index, component
    /// index, Wohlmuth lumped-row factor) and per-axis box lengths
    /// from the classifier's bbox. Called once at construction.
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
    BoundaryClassifier3D         m_classifier;
    ConstraintBuilder3D          m_builder;
    MortarConstraintOperator     m_C_op;
    SaddlePointSolver            m_saddle_solver;

    // Phase 5.5.B.4 — saddle system stored as shared_ptr so it can
    // be handed to ExaNewtonSolver via SetOperator(shared_ptr<Operator>).
    // The manager constructs it on the heap; SystemDriver receives a
    // copy of the shared_ptr via GetSaddleSystemShared(). Constructed
    // before m_g_rhs because m_g_rhs is the buffer the saddle system
    // points at, but we install the pointer in the ctor body so the
    // declaration order between the two is decoupled.
    std::shared_ptr<MortarSaddlePointSystem> m_saddle_system;


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

    // Phase 5.3.C.2 — reference-geometry caches for §P5.8.6.d.
    // All allocated with `mfem::Device::GetMemoryType()` so the
    // per-row kernel can run on GPU. (mfem::Array<int> doesn't have
    // `UseDevice(bool)` — only construct-time memory typing — so this
    // is the only correct pattern for the int arrays.)
    mfem::Array<int>             m_axis_per_row;
    mfem::Array<int>             m_component_per_row;
    mfem::Vector                 m_ell_hat_per_row;
    mfem::Vector                 m_axis_lengths;
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

}  // namespace mortar_pbc