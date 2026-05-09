// Phase 5.3.A — MortarPbcManager
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
//     (used for periodic-traction post-processing).
//   - The 24 corner-essential TDOFs (8 corners × 3 components),
//     pinned to remove rigid-body modes.
//
// Phasing:
//   - 5.3.A (this file): class skeleton + constructor wiring.
//     `BuildCornerEssTDofs` and `BuildReferenceGeometricFactors`
//     are declared but stubbed; the public methods that 5.3.C–E
//     will fill in MFEM_ABORT with helpful messages.
//   - 5.3.B: corner essential-TDOF list construction.
//   - 5.3.C: macroscopic-F update + constraint-RHS computation.
//     Will likely also be when the boundary `ParSubMesh` (currently
//     internal to `BoundaryClassifier3D`) gets promoted onto
//     `SimulationState` so the rest of the code can reach it from a
//     single place. Phase 6 LOR work then adds a second surface
//     mesh entry on `SimulationState` for the LOR projection.
//   - 5.3.D: fluctuation-field projection + Hill–Mandel power
//     balance for diagnostics.
//   - 5.3.E: λ accumulation API for periodic-traction outputs.
//
// References:
//   - PHASE5_EXACONSTIT_INTEGRATION_v4.md §P5.4 (this class).
//   - MORTAR_PBC_ARCHITECTURE.md §11 (Phase 4 mortar machinery).
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
 *   pbc->UpdateMacroscopicF(L_bar, dt);   // F̄ ← F̄ + L̄·F̄·dt
 *   pbc->UpdateConstraintRHS();           // refresh m_g_rhs in place
 *   newton_solver->Solve(pbc->GetSaddleSystem(), ...);
 *   pbc->AccumulateLambdaContribution(dlam, dt);
 * @endcode
 *
 * @par Lifetime
 * The manager holds a `std::shared_ptr<SimulationState>`, matching
 * the convention used elsewhere in the codebase (e.g.
 * `NonlinearMechOperator`). All access to the parent mesh and
 * primary FE space goes through the simulation state — no bare
 * references to `ParMesh` / `ParFiniteElementSpace` are stored on
 * the manager. As mortar-specific objects (e.g. the boundary
 * `ParSubMesh` in 5.3.C, the LOR variant in Phase 6) get added to
 * `SimulationState`, the manager will reach them the same way.
 *
 * @par MPI scope
 * Construction is collective on `sim_state->GetMesh()->GetComm()`
 * (delegated to `BoundaryClassifier3D`). Per-step methods are
 * collective on the same communicator.
 *
 * @par GPU
 * The manager itself is host-only (configuration + topology +
 * small dense state). The owned saddle-point solver dispatches
 * Krylov + preconditioner work via `mfem::Operator` interfaces, so
 * GPU support follows whatever K's assembly form provides
 * (HypreParMatrix path is fully supported in Phase 4.3+; PA-K is
 * Phase 6+ when `Operator::AssembleDiagonal` lands in the
 * preconditioner).
 *
 * @par Thread safety
 * Not thread-safe. Designed for one manager per simulation,
 * mutated only from the main MPI thread.
 */
class MortarPbcManager
{
public:
    /// Closure type: compute K-residual `r_K = K(u)` (or `K(u) - f` if
    /// `f` is folded into the closure). Result is the local FES TDOF
    /// slice. Forwarded directly to `MortarSaddlePointSystem`.
    using KResidualFn = MortarSaddlePointSystem::KResidualFn;

    /// Closure type: return a non-owning `mfem::Operator*` for the
    /// current K-Jacobian `dK/du(u)`. Pointer must remain valid until
    /// the next call. Forwarded directly to `MortarSaddlePointSystem`.
    using KJacobianFn = MortarSaddlePointSystem::KJacobianFn;

    /**
     * @brief Construct and wire the full mortar-PBC pipeline.
     *
     * @param sim_state    Shared simulation state. Must already be
     *                     populated with a 3D `ParMesh`, a vector
     *                     H1 FE space (vdim=3, order 1 in Phase 5),
     *                     and parsed `ExaOptions`. The manager
     *                     retains a shared-ownership reference;
     *                     reads through it on demand for every
     *                     piece of mesh / FES / configuration data
     *                     it needs. Mesh and FES accessors are
     *                     `sim_state->GetMesh()` and
     *                     `sim_state->GetMeshParFiniteElementSpace()`;
     *                     options live at `sim_state->GetOptions()`.
     * @param k_residual   User's K-residual callback. See
     *                     `MortarSaddlePointSystem` for semantics.
     * @param k_jacobian   User's K-Jacobian callback. See
     *                     `MortarSaddlePointSystem` for semantics.
     *
     * @par MPI scope
     * Collective on the parent mesh's communicator — the boundary
     * classifier does several Allgather/Allreduce/Alltoall calls
     * during construction. After return, all per-step methods are
     * also collective on the same communicator.
     *
     * @par Validation
     * Aborts via `MFEM_VERIFY` if `opts.mesh.lor_depth != 1`
     * (Phase 6 stub) or if `opts.solvers.saddle_point` parses to an
     * unknown enum value. Other validation lives in the components
     * themselves (the classifier checks dim/vdim/order).
     */
    MortarPbcManager(std::shared_ptr<SimulationState> sim_state,
                     KResidualFn k_residual,
                     KJacobianFn k_jacobian);

    ~MortarPbcManager() = default;

    // Non-copyable / non-movable: holds a non-trivial owned-component
    // graph and a shared simulation-state reference.
    MortarPbcManager(const MortarPbcManager&) = delete;
    MortarPbcManager& operator=(const MortarPbcManager&) = delete;

    //==========================================================================
    // State updates — Phase 5.3.C (stubs in 5.3.A)
    //==========================================================================

    /**
     * @brief Update the tracked macroscopic deformation gradient.
     *
     * @details Phase 5.3.C will implement this. Intended semantics:
     * given a velocity-gradient `Lbar` and time step `dt`, advance
     * `m_macro_F` by `F̄ ← F̄ + Lbar · F̄ · dt` and store
     * `m_macro_Fdot ← Lbar · F̄`. Called once per time step from
     * SystemDriver before the Newton solve.
     *
     * @param Lbar  Velocity-gradient tensor (3×3).
     * @param dt    Time-step size.
     */
    void UpdateMacroscopicF(const mfem::DenseMatrix& Lbar, double dt);

    /**
     * @brief Refresh the constraint-RHS buffer for the current
     *        macroscopic state.
     *
     * @details Phase 5.3.C will implement this. Intended semantics:
     * compute the per-row `g_k = F̄ · X_k` value the constraint
     * equation `C u = g` should equal so that `u` corresponds to
     * the prescribed macroscopic deformation, and write it into
     * the manager's `m_g_rhs` buffer. Because the saddle system was
     * given a pointer to that buffer at construction, the change
     * propagates without any further wiring.
     */
    void UpdateConstraintRHS();

    //==========================================================================
    // Diagnostics / output computation — Phase 5.3.D (stubs in 5.3.A)
    //==========================================================================

    /**
     * @brief Project the full displacement onto the fluctuation
     *        field `ũ = u − F̄·X` for visualization.
     *
     * @details Phase 5.3.D will implement this.
     *
     * @param u_tdofs  Full displacement at FES TDOFs (size
     *                 `fes.GetTrueVSize()`).
     * @param u_fluct  Output fluctuation field as a ParGridFunction
     *                 over the same FES. Sized internally by the
     *                 implementation.
     */
    void ComputeFluctuationField(const mfem::Vector& u_tdofs,
                                 mfem::ParGridFunction& u_fluct) const;

    /**
     * @brief Compute the Hill–Mandel power balance for diagnostics.
     *
     * @details Phase 5.3.D will implement this. Intended semantics:
     * compute the cell-averaged `<σ : Ḟ>` (volume integral) and
     * compare against `F̄ : <σ>` (the boundary-traction work). On
     * a converged Newton step these should agree to FP precision;
     * on a non-converged step the gap is a useful diagnostic.
     *
     * @param u_tdofs       Full displacement at FES TDOFs.
     * @param cell_power    Output: cell-averaged power.
     * @param macro_power   Output: macroscopic-state power.
     */
    void ComputeHillMandelPowerBalance(const mfem::Vector& u_tdofs,
                                       double& cell_power,
                                       double& macro_power) const;

    //==========================================================================
    // Lambda accumulation — Phase 5.3.E (stubs in 5.3.A)
    //==========================================================================

    /**
     * @brief Accumulate a Newton-step λ contribution into the
     *        manager's running λ buffer.
     *
     * @details Phase 5.3.E will implement this. Intended semantics:
     * `m_lambda += scale * dlam`. Called from SystemDriver after
     * each successful Newton solve to keep a running total of the
     * Lagrange multiplier across the load history (used downstream
     * for periodic-traction output).
     *
     * @param dlam   Newton increment to the multiplier (size
     *               `NumLocalConstraints()`).
     * @param scale  Scale factor (typically the load-step weight or
     *               1.0).
     */
    void AccumulateLambdaContribution(const mfem::Vector& dlam,
                                      double scale = 1.0);

    /**
     * @brief Reset the accumulated λ buffer to zero.
     *
     * @details Implemented in 5.3.A (trivial zero-fill); 5.3.E will
     * document the calling convention. Typical usage: called once
     * at simulation start, then `AccumulateLambdaContribution`
     * runs each Newton step thereafter.
     */
    void ResetLambdaAccumulation();

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

    /// Mutable accessor — SystemDriver wraps the Krylov solver
    /// configuration as needed. See `MortarSaddlePointSystem` for
    /// the per-Newton-iteration usage.
    SaddlePointSolver& GetSaddleSolver() { return m_saddle_solver; }
    const SaddlePointSolver& GetSaddleSolver() const { return m_saddle_solver; }

    /// Mutable accessor — the Newton solver in SystemDriver mutates
    /// the system's internal Jacobian cache via `GetGradient()`.
    MortarSaddlePointSystem& GetSaddleSystem() { return m_saddle_system; }
    const MortarSaddlePointSystem& GetSaddleSystem() const
    {
        return m_saddle_system;
    }

    /// 24-element list of corner-pinned TDOFs (filled in 5.3.B; empty
    /// in 5.3.A).
    const mfem::Array<int>& GetCornerEssTDofs() const
    {
        return m_corner_ess_tdofs;
    }

    /// Current macroscopic deformation gradient (3×3). Identity at
    /// construction time, updated by `UpdateMacroscopicF` (5.3.C).
    const mfem::DenseMatrix& GetMacroscopicF() const { return m_macro_F; }

    /// Current macroscopic deformation-rate tensor `Ḟ` (3×3).
    /// Zero at construction; updated by `UpdateMacroscopicF` (5.3.C).
    const mfem::DenseMatrix& GetMacroscopicFdot() const { return m_macro_Fdot; }

    /// Accumulated λ over the load history. Size =
    /// `NumLocalConstraints()`. Zero at construction.
    const mfem::Vector& GetAccumulatedLambda() const { return m_lambda; }

    /// Number of constraint rows owned by this rank
    /// (= `m_C_op.Height()` = `NumLocalConstraints()`).
    int NumLocalConstraints() const { return m_C_op.Height(); }

private:
    //--------------------------------------------------------------------------
    // Private helpers
    //--------------------------------------------------------------------------

    /// Phase 5.3.B — populate `m_corner_ess_tdofs` with the rank-local
    /// TDOFs for the 8 box corners (3 components each, filtered to
    /// only those owned by this rank). Stubbed in 5.3.A.
    void BuildCornerEssTDofs();

    /// Phase 5.3.C — cache reference (undeformed) coordinates of
    /// boundary nodes that participate in mortar constraints, so that
    /// `UpdateConstraintRHS` can compute `g_k = F̄ · X_k` per row
    /// without re-walking the classifier each step. Stubbed in
    /// 5.3.A — the cache layout is finalized in 5.3.C.
    void BuildReferenceGeometricFactors();

    //--------------------------------------------------------------------------
    // Member state
    //
    // Declaration order matters: members are initialized in declaration
    // order, not initializer-list order. The dependency chain is
    //   sim_state → classifier → builder → C_op → saddle_solver →
    //   saddle_system,
    // so they're declared in that order below.
    //--------------------------------------------------------------------------

    /// @brief Reference to simulation state containing mesh, fields,
    /// and configuration data. Held by shared ownership so the
    /// manager doesn't need to track parent-mesh / FES lifetimes
    /// separately. Phase 5.3.C+ will reach for additional pieces
    /// (boundary `ParSubMesh`, LOR variants in Phase 6) through this
    /// same handle once they're added to `SimulationState`.
    std::shared_ptr<SimulationState> m_sim_state;

    // Owned components (initialized in dependency order).
    BoundaryClassifier3D         m_classifier;
    ConstraintBuilder3D          m_builder;
    MortarConstraintOperator     m_C_op;
    SaddlePointSolver            m_saddle_solver;
    MortarSaddlePointSystem      m_saddle_system;

    // State buffers.
    mfem::Array<int>             m_corner_ess_tdofs;  // Phase 5.3.B fills.
    mfem::Vector                 m_lambda;            // Accumulator.
    mfem::Vector                 m_g_rhs;             // Refresh buffer.

    //==========================================================================
    // Phase 5.3.C.2 — reference-geometry caches for §P5.8.6.d.
    //
    // Built once at construction by BuildReferenceGeometricFactors;
    // consumed each time step by UpdateConstraintRHS to compute
    //
    //     g[i] = Ḟ̄[c, k] * L_k * ℓ̂_i
    //
    // where (c, k) = (component_per_row[i], axis_per_row[i]) and
    // L_k = axis_lengths[k] is the RVE box length on the k-th
    // periodic axis. All three per-row members and the axis lengths
    // have UseDevice(true) so the kernel can run on GPU; ℓ̂_i is
    // zero for degenerate rows (D_nm[k] = 0 from corner-modified
    // nodes), making g[i] = 0 there too — consistent with the
    // matching all-zero row of C.
    //==========================================================================

    /// @brief Periodic-axis index ∈ {0, 1, 2} per constraint row.
    mfem::Array<int> m_axis_per_row;
    /// @brief Spatial-component index ∈ {0, 1, 2} per constraint row.
    mfem::Array<int> m_component_per_row;
    /// @brief Wohlmuth lumped-row factor ℓ̂_i per constraint row.
    ///        Zero for degenerate (corner-modified) rows.
    mfem::Vector m_ell_hat_per_row;
    /// @brief RVE box lengths along x, y, z axes (3-vector).
    mfem::Vector m_axis_lengths;

    // Macroscopic state — small dense (3×3) matrices.
    mfem::DenseMatrix            m_macro_F;
    mfem::DenseMatrix            m_macro_Fdot;
};

/**
 * @brief Compute rank-local TDOFs for the 8 box corners of a
 *        classified RVE boundary.
 *
 * @details Iterates the classifier's 8 corner records (replicated on
 * every rank) and, for each corner's three components (x/y/z), tests
 * whether the global TDOF is owned by this rank using
 * `classifier.GtdofOwnerRank`. Owned components are converted to
 * rank-local indices via `fes.GetMyTDofOffset()` and appended to the
 * output array.
 *
 * Exposed as a free function (rather than baked into
 * `MortarPbcManager::BuildCornerEssTDofs`) so it can be exercised
 * by `test_mortar_pbc_manager.cpp` in isolation, without the cost
 * of constructing a full `SimulationState` to instantiate a
 * manager. The manager method is a thin wrapper that calls this
 * helper and adds an MPI sanity check on top.
 *
 * @par Postcondition
 * Across the classifier's communicator,
 * `MPI_Allreduce(SUM, output.Size())` equals 24 (8 corners × 3
 * components). Each rank-local entry is a valid TDOF in
 * `[0, fes.GetTrueVSize())`.
 *
 * @param classifier  Fully-built `BoundaryClassifier3D`.
 * @param fes         The vector H1 FE space the classifier was built
 *                    on. Must be the same FES used at classifier
 *                    construction (or one with an equivalent TDOF
 *                    partition).
 *
 * @return Rank-local list of corner essential TDOFs, ready to feed
 *         to MFEM's Dirichlet-elimination machinery.
 */
mfem::Array<int> ComputeCornerEssTDofs(
    const BoundaryClassifier3D& classifier,
    const mfem::ParFiniteElementSpace& fes);

}  // namespace mortar_pbc