#ifndef mechanics_system_driver_hpp
#define mechanics_system_driver_hpp

#include "fem_operators/mechanics_operator.hpp"
#include "mortar_pbc/mortar_pbc_manager.hpp"
#include "mortar_pbc/mortar_saddle_preconditioner.hpp"
#include "mortar_pbc/saddle_scaling_wrappers.hpp"
#include "mortar_pbc/saddle_newton_diagnostic_logger.hpp"
#include "models/mechanics_model.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"
#include "solvers/mechanics_solver.hpp"

#include "mfem.hpp"

#include <fstream>
#include <memory>
/**
 * @brief Primary driver class for ExaConstit's velocity-based finite element simulations.
 *
 * SystemDriver orchestrates the entire nonlinear mechanics simulation workflow for ExaConstit,
 * implementing a velocity-based, updated Lagrangian finite element framework for solid mechanics
 * problems with emphasis on crystal plasticity and micromechanics modeling. The class manages
 * the Newton-Raphson solution process, boundary condition enforcement, and model state updates
 * throughout the simulation.
 *
 * The driver integrates multiple components:
 * - Newton-Raphson nonlinear solver with optional line search
 * - Krylov iterative linear solvers (GMRES, CG, MINRES) with preconditioning
 * - Essential boundary condition management (velocity and velocity gradient BCs)
 * - Constitutive model integration and state variable updates
 * - Device-aware execution supporting CPU, OpenMP, and GPU backends
 *
 * Key simulation capabilities:
 * - Multi-material nonlinear solid mechanics with large deformations
 * - Crystal plasticity simulations with grain-specific material behavior
 * - Complex loading conditions including velocity gradient boundary conditions
 * - Automated time stepping with adaptive control
 * - Parallel MPI execution with domain decomposition
 *
 * The class follows the MFEM finite element framework conventions while extending
 * functionality for ExaConstit's specialized micromechanics applications. It serves
 * as the central coordination point between mesh management, material models,
 * solvers, and postprocessing systems.
 *
 * @ingroup ExaConstit_system
 */
class SystemDriver {
private:
    /// @brief Newton-Raphson solver instance for nonlinear equation systems
    /// Handles the main iterative solution process for F(x) = 0 using Newton's method or Newton
    /// with line search
    std::unique_ptr<ExaNewtonSolver> newton_solver;

    /// @brief Linear solver for Jacobian system solution within Newton iterations
    /// Solves the linearized system J*dx = -F at each Newton step using Krylov methods
    /// (GMRES/CG/MINRES)
    std::shared_ptr<mfem::IterativeSolver> J_solver;

    /// @brief Preconditioner for the Jacobian linear system to improve convergence
    /// Typically algebraic multigrid (BoomerAMG) or Jacobi preconditioning for efficiency
    std::shared_ptr<mfem::Solver> J_prec;

    /// @brief Material model interface for constitutive relationship evaluation
    /// Manages material property evaluation, state variable updates, and stress computation
    std::shared_ptr<ExaModel> model;

    /// @brief Nonlinear mechanics operator encapsulating the finite element discretization
    /// Provides residual evaluation, Jacobian computation, and essential DOF management for the
    /// mechanics problem
    std::shared_ptr<NonlinearMechOperator> mech_operator;

    /// @brief Number of Newton iterations performed in current solve
    int newton_iter;

    /// @brief Device execution model (CPU/OpenMP/GPU) for RAJA kernels
    RTModel class_device;

    /// @brief Flag indicating automatic time stepping is enabled
    bool auto_time = false;

    /// @brief Boundary condition attribute arrays organized by BC type
    /// Keys: "total", "ess_vel", "ess_vgrad" for combined, velocity, and velocity gradient BCs
    std::unordered_map<std::string, mfem::Array<int>> ess_bdr;

    /// @brief Scaling factors for boundary conditions per attribute and spatial component
    mfem::Array2D<double> ess_bdr_scale;

    /// @brief Component-wise BC flags indicating which spatial directions have essential BCs
    /// Keys match ess_bdr: "total", "ess_vel", "ess_vgrad"
    std::unordered_map<std::string, mfem::Array2D<bool>> ess_bdr_component;

    /// @brief Current velocity gradient tensor for uniform deformation boundary conditions
    /// Stored as flattened 3x3 or 2x2 tensor depending on spatial dimension
    mfem::Vector ess_velocity_gradient;

    /// @brief MFEM coefficient function for applying Dirichlet boundary conditions
    /// Restricted to specific boundary attributes with time-dependent scaling factors
    std::unique_ptr<mfem::VectorFunctionRestrictedCoefficient> ess_bdr_func;

    /// @brief Reference point for velocity gradient boundary condition calculations
    /// Used as origin for computing position-dependent velocity in uniform deformation
    const bool vgrad_origin_flag = false;

    /// @brief Reference point for velocity gradient boundary condition calculations
    /// Used as origin for computing position-dependent velocity in uniform deformation
    mfem::Vector vgrad_origin;

    /// @brief Flag enabling monolithic deformation mode with simplified boundary conditions
    /// Used for special loading cases with constrained degrees of freedom
    const bool mono_def_flag = false;

    /// @brief Reference to simulation state containing mesh, fields, and configuration data
    std::shared_ptr<SimulationState> m_sim_state;

    /**
     * @brief Phase 5.5 — set true when the simulation has mortar PBC
     *        enabled (periodicity + velocity-gradient BC + Phase-5
     *        prerequisites).
     *
     * @details Determined once at construction via
     * `HasVelocityGradientBC(options) && options.mesh.periodicity`,
     * then queried throughout the per-step lifecycle to gate the
     * mortar branches in `Solve()`, `SolveInit()`, `UpdateEssBdr()`,
     * and `UpdateVelocity()`. False for all non-mortar simulations
     * (i.e., the entire current production path), so the mortar
     * code paths are completely inert when not used.
     */
    bool m_mortar_enabled = false;

    /**
     * @brief Phase 5.5 — mortar PBC manager. Owns the boundary
     *        classifier, constraint builder, EA constraint operator,
     *        saddle-point system adapter, saddle-point linear solver,
     *        and the macroscopic-F state. Only constructed when
     *        `m_mortar_enabled` is true. See
     *        `mortar_pbc::MortarPbcManager`.
     */
    std::shared_ptr<mortar_pbc::MortarPbcManager> m_mortar_pbc;

    // Phase 5.5.B.4 — saddle-point preconditioner & scratch.
    //
    // Constructed only when m_mortar_enabled. SystemDriver follows
    // the existing J_prec ownership pattern: m_K_jacobi_prec is the
    // K-Jacobi preconditioner (HypreSmoother in FA mode) supplied
    // separately to MortarSaddlePreconditioner so the saddle prec
    // can probe diag(K)^{-1} for ComputeInvDiagSchur without
    // requiring the full J_prec to expose Jacobi behavior; the
    // user's chosen J_prec (AMG, ILU, L1GS, Cheby, l1Jacobi) flows
    // in as the K-block prec for the (0,0) saddle-block apply.
    //
    // Both preconditioners get SetOperator'd per Newton iteration
    // by MortarSaddlePreconditioner::SetOperator (which is itself
    // called by mfem::IterativeSolver::SetOperator propagation
    // during ExaNewtonSolver::Mult's krylov_solver call).
    std::shared_ptr<mfem::Solver> m_K_jacobi_prec;
    std::shared_ptr<mfem::Solver> m_mortar_saddle_prec;

    //==========================================================================
    // Phase 5.11.H — saddle-residual scaling wrappers.
    //
    // Always constructed when the mortar path is enabled — the
    // wrappers' Mult bodies short-circuit to pass-through when the
    // scaler is null or `IsEnabled() == false`, so they are
    // identity-transform-equivalent for production runs at no
    // measurable cost. The conditional install on `newton_solver`
    // and `J_solver` happens below in the constructor body; the
    // members live here so they outlive the Newton solve scope.
    //
    // Storage is shared_ptr for two reasons:
    //  1. The Newton solver's SetOperator / SetSolver overloads take
    //     shared_ptr (5.11.F era convention).
    //  2. The wrappers internally hold shared_ptr to their inner
    //     op / solver / prec; matching ownership at the SystemDriver
    //     layer avoids lifetime asymmetries.
    //==========================================================================
    std::shared_ptr<mortar_pbc::ScaledSaddleOperator>       m_scaled_saddle_op;
    std::shared_ptr<mortar_pbc::ScaledSaddleSolver>         m_scaled_saddle_solver;
    std::shared_ptr<mortar_pbc::ScaledSaddlePreconditioner> m_scaled_saddle_prec;

    /**
     * @brief Phase 5.9 / Batch A.5 — tracks the active periodic-BC
     *        entry installed in `m_mortar_pbc`.
     *
     * @details `m_pbc_initialized` is false until the first call to
     * `SyncMortarPbcForStep` succeeds. After that point,
     * `m_pbc_active_entry_idx` records which entry of
     * `options.boundary_conditions.periodic_bcs` is currently
     * applied, or -1 if the synthesized default (empty
     * `periodic_bcs` fallback) is in effect.
     *
     * Both members are unused (and stay at their default values)
     * for non-mortar simulations.
     */
    bool m_pbc_initialized = false;
    int  m_pbc_active_entry_idx = -1;

    // Phase 5.5.B.4 — saddle Newton scratch.
    //
    // m_x_saddle is the BlockVector the Newton iterates against:
    // [u | lambda]. The PrimalField (u-block) is packed in at the
    // start of Solve() / SolveInit() and the lambda-block is seeded
    // from the manager's accumulated lambda buffer for warm
    // starting.
    mfem::Array<int>                          m_saddle_offsets;
    std::unique_ptr<mfem::BlockVector>        m_x_saddle;

   // Phase 5.11.J — diagnostic logger replaces the Phase 5.11.I
   // raw m_newton_diag_file + manual CSV writes. The logger owns
   // its own file handle, sub-block-aware header, per-block
   // residual decomposition, and step-index counter. Constructed
   // in the SystemDriver ctor's mortar block alongside the saddle
   // scaling wrappers; destroyed alongside the SystemDriver.
    std::unique_ptr<mortar_pbc::SaddleNewtonDiagnosticLogger> m_newton_diag_logger;

public:
    /**
     * @brief Construct SystemDriver with simulation state and initialize all components.
     *
     * @param sim_state Reference to simulation state containing mesh, options, and field data
     *
     * Initializes the complete finite element system for ExaConstit simulations including
     * boundary condition management, linear and nonlinear solvers, and device configuration.
     * The constructor performs extensive setup to prepare the system for time-stepping.
     *
     * Initialization process:
     * 1. Configure device execution model from simulation options
     * 2. Initialize boundary condition arrays and component flags
     * 3. Set up velocity gradient and origin vectors for uniform deformation
     * 4. Create and configure the nonlinear mechanics operator
     * 5. Initialize linear solver (GMRES/CG/MINRES) with preconditioning
     * 6. Configure Newton solver with convergence criteria
     * 7. Handle special case initialization for monolithic deformation mode
     * 8. Set up boundary condition coefficient functions
     *
     * Boundary condition setup:
     * - Creates separate attribute arrays for total, velocity, and velocity gradient BCs
     * - Initializes component-wise flags for spatial direction control
     * - Configures scaling factors for time-dependent boundary conditions
     * - Sets up restricted coefficient functions for MFEM integration
     *
     * Linear solver configuration:
     * - Supports GMRES, Conjugate Gradient, and MINRES iterative solvers
     * - Configures algebraic multigrid (BoomerAMG) or Jacobi preconditioning
     * - Sets convergence tolerances and maximum iterations from options
     * - Enables device-aware execution for GPU acceleration when available
     *
     * Nonlinear solver setup:
     * - Creates Newton-Raphson solver with optional line search capability
     * - Configures convergence criteria (relative/absolute tolerances)
     * - Sets maximum iteration limits and printing levels
     * - Links linear solver for Jacobian system solution
     *
     * Special handling for monolithic deformation mode:
     * - Computes mesh bounding box for reference coordinate system
     * - Identifies constrained degrees of freedom based on geometric constraints
     * - Sets up essential DOF lists for simplified boundary condition enforcement
     *
     * @note Constructor performs significant computational work including mesh analysis
     * @note Memory allocation is device-aware and respects the configured execution model
     * @note All components are fully initialized and ready for time-stepping upon completion
     *
     * @throws std::runtime_error if critical initialization steps fail
     * @throws MFEM_VERIFY errors for invalid configuration combinations
     */
    SystemDriver(std::shared_ptr<SimulationState> sim_state);

    /**
     * @brief Get essential true degrees of freedom list from mechanics operator.
     *
     * @return Const reference to array of essential true DOF indices
     *
     * Retrieves the list of essential (constrained) true degrees of freedom from the
     * underlying nonlinear mechanics operator. These DOFs correspond to nodes where
     * Dirichlet boundary conditions are applied and represent constrained solution
     * components that are not solved for in the linear system.
     *
     * The essential true DOF list is used by:
     * - Linear solvers to identify constrained equations
     * - Post-processing routines for proper field reconstruction
     * - Boundary condition enforcement during assembly
     * - Solution vector manipulation and constraint application
     *
     * True DOFs represent the actual degrees of freedom in the parallel distributed
     * system after applying finite element space restrictions and MPI communication
     * patterns. This differs from local DOFs which are process-specific.
     *
     * @note Returns reference to internal data - do not modify
     * @note Valid only after mechanics operator initialization
     * @note Used primarily for linear solver configuration
     *
     * @ingroup ExaConstit_boundary_conditions
     */
    const mfem::Array<int>& GetEssTDofList();

    /**
     * @brief Execute Newton-Raphson solver for current time step.
     *
     * Performs the main nonlinear solve for the current time step using Newton-Raphson
     * iteration. This function orchestrates the complete solution process including
     * initial guess setup, Newton iteration, convergence checking, and solution update.
     *
     * Solution process:
     * 1. **Initial Guess Setup**: For automatic time stepping, uses previous solution
     * 2. **Newton Iteration**: Repeatedly solves linearized system until convergence
     * 3. **Convergence Check**: Verifies solution meets relative and absolute tolerances
     * 4. **Solution Update**: Updates velocity field and distributes to parallel processes
     * 5. **Boundary Condition Sync**: Updates time-dependent boundary conditions if converged
     *
     * Automatic time stepping mode:
     * - Uses previous time step solution as initial guess for better convergence
     * - Enables more aggressive time step sizes for improved efficiency
     * - Particularly effective for quasi-static and dynamic problems
     *
     * The function handles both standard Newton-Raphson and line search variants
     * depending on the solver type configured during initialization. Line search
     * provides better robustness for highly nonlinear problems.
     *
     * Convergence criteria:
     * - Relative tolerance: ||residual|| / ||initial_residual|| < rel_tol
     * - Absolute tolerance: ||residual|| < abs_tol
     * - Maximum iterations: Prevents infinite loops for non-convergent cases
     *
     * @note Throws MFEM_VERIFY error if Newton solver fails to converge
     * @note Updates boundary condition time if solver converges successfully
     * @note Critical function called once per time step in main simulation loop
     *
     * @throws MFEM_VERIFY if Newton solver does not converge within iteration limits
     *
     * @ingroup ExaConstit_solvers
     */
    void Solve();

    /**
     * @brief Solve Newton system for first time step with boundary condition ramp-up.
     *
     * Performs a specialized solve for the first time step that addresses convergence
     * issues with large meshes by implementing a boundary condition ramp-up strategy.
     * This corrector step ensures the solver has better initial conditions for the
     * main Newton iteration.
     *
     * **Algorithm:**
     * 1. **Setup Phase**: Create residual and increment vectors
     * 2. **BC Increment Calculation**: Compute difference between current and previous BCs
     * 3. **Linear Solve**: Use mechanics operator's UpdateBCsAction for linearized correction
     * 4. **CG Solution**: Apply conjugate gradient solver to correction equation
     * 5. **Solution Update**: Apply correction and distribute to velocity field
     *
     * **Mathematical Framework:**
     * The method solves a linearized correction equation:
     * ```
     * K_uc * (x - x_prev)_c = deltaF_u
     * ```
     * where:
     * - K_uc: Unconstrained-constrained stiffness block
     * - x, x_prev: Current and previous solution vectors
     * - deltaF_u: Force increment from boundary condition changes
     *
     * **Ramp-up Strategy:**
     * - Addresses numerical difficulties with sudden BC application
     * - Provides smooth transition from previous solution state
     * - Particularly important for large deformation problems
     * - Improves convergence rates for subsequent Newton iterations
     *
     * **Usage Context:**
     * Called automatically when BCManager detects boundary condition changes
     * between time steps, enabling complex loading scenarios:
     * - Multi-stage deformation processes
     * - Cyclic loading applications
     * - Time-dependent displacement prescriptions
     * - Load path modifications during simulation
     *
     * @note Function marked const but modifies solution through sim_state references
     * @note Uses mechanics operator's specialized BC update action
     * @note Critical for robust handling of time-dependent boundary conditions
     *
     * @ingroup ExaConstit_solvers
     */
    void SolveInit() const;

    /**
     * @brief Update material model variables after converged time step.
     *
     * Synchronizes material model state variables and simulation data structures
     * after a successful time step completion. This function ensures all internal
     * variables, history-dependent data, and kinematic measures are properly
     * updated for the next time step.
     *
     * **Update Sequence:**
     * 1. **Model Variable Update**: Calls model->UpdateModelVars() to advance material state
     * 2. **Simulation State Update**: Updates SimulationState internal data structures
     * 3. **Model Variable Setup**: Reinitializes model variables for next step
     * 4. **Deformation Gradient**: Computes current deformation gradient from solution
     *
     * **Material Model Updates:**
     * - Advances internal state variables (plastic strains, damage, etc.)
     * - Updates stress states based on converged deformation
     * - Handles crystal plasticity orientation evolution
     * - Manages history-dependent material properties
     *
     * **Kinematic Updates:**
     * - Computes deformation gradient F from current nodal positions
     * - Updates kinematic measures for large deformation analysis
     * - Maintains consistency between geometric and material nonlinearities
     *
     * **State Management:**
     * - Swaps "beginning" and "end" step quadrature function data
     * - Prepares data structures for next time increment
     * - Ensures proper history variable management
     * - Maintains simulation restart capability
     *
     * **Critical for:**
     * - Material model accuracy and stability
     * - History-dependent constitutive behavior
     * - Multi-physics coupling consistency
     * - Simulation restart and checkpointing
     *
     * @note Must be called after each converged time step
     * @note Order of operations is critical for simulation accuracy
     * @note Handles both single and multi-material regions
     *
     * @ingroup ExaConstit_material_models
     */
    void UpdateModel();

    /**
     * @brief Update essential boundary condition data for the current time step.
     *
     * Synchronizes boundary condition data with the BCManager to reflect any changes
     * in applied boundary conditions for the current simulation step. Updates the
     * essential true degrees of freedom in the mechanics operator to ensure proper
     * constraint enforcement during the solve process.
     *
     * This function is called when boundary conditions change between time steps,
     * enabling time-dependent loading scenarios such as cyclic loading, multi-stage
     * deformation, or complex loading histories typical in materials testing.
     *
     * Updates performed:
     * - Retrieves latest BC data from BCManager
     * - Updates ess_bdr, ess_bdr_scale, and ess_bdr_component arrays
     * - Refreshes essential true DOF list in mechanics operator
     * - Ensures consistency between BC data and solver constraints
     *
     * @note Called automatically when BCManager detects BC changes for current step
     * @note Only updates data if mono_def_flag is false (normal operation mode)
     */
    void UpdateEssBdr();

    /**
     * @brief Phase 5.9 / Batch A.5 — install or switch the active
     *        periodic-BC entry for the given simulation step.
     *
     * @details This method is the bridge between the user-facing
     * `[[BCs.periodic_bcs]]` TOML schema (parsed into
     * `options.boundary_conditions.periodic_bcs` +
     * `periodic_bc_entry_per_step`) and the
     * `mortar_pbc::MortarPbcManager`'s spec-driven `RebuildForActiveSpec`
     * API. The intended call sequence in the outer time-stepping
     * driver is:
     *
     * @code
     * for (int step_idx = 1; step_idx <= n_steps; ++step_idx) {
     *     BCManager::GetInstance().GetUpdateStep(step_idx);
     *     system_driver->SyncMortarPbcForStep(step_idx);   // <-- NEW
     *     system_driver->UpdateEssBdr();
     *     // ... velocity update, Solve(), update model, ...
     * }
     * @endcode
     *
     * @par State machine
     * * **Non-mortar simulation** (`m_mortar_enabled == false`):
     *   no-op.
     * * **Empty `periodic_bcs`** (default-fallback path): on the
     *   first call, synthesizes the full-PBC spec via
     *   `MortarPbcManager::SynthesizeDefaultPbcSpec` and applies it;
     *   subsequent calls are no-ops because the synthesized default
     *   is step-invariant.
     * * **Non-empty `periodic_bcs`**: looks up `step_idx` in
     *   `periodic_bc_entry_per_step`. If the lookup hits AND the
     *   target entry differs from `m_pbc_active_entry_idx`, calls
     *   `m_mortar_pbc->RebuildForActiveSpec(spec.essential_ids,
     *   spec.essential_comps)` and re-pushes the new corner subset
     *   to `mech_operator->UpdateEssTDofsCornerSubset`. If the
     *   lookup misses, the current spec is preserved (a sparse
     *   `update_steps` schedule installs entries only at transition
     *   steps — intermediate steps inherit). If the lookup misses
     *   AND the spec has never been initialized (first call with
     *   `step_idx` not in the map), aborts with a configuration
     *   error.
     *
     * @par MPI scope
     * Collective on `mech_operator`'s communicator
     * (`UpdateEssTDofsCornerSubset` may be collective);
     * `m_mortar_pbc->RebuildForActiveSpec` itself is local.
     *
     * @par Idempotence
     * If `step_idx` resolves to the same entry already active, the
     * method returns without calling either `RebuildForActiveSpec`
     * or `UpdateEssTDofsCornerSubset`. This is the common case for
     * most steps in a typical run (transitions only happen at the
     * `update_steps` boundaries).
     *
     * @param step_idx 1-based simulation step index. Same value the
     *                 outer caller passes to
     *                 `BCManager::GetInstance().GetUpdateStep`.
     */
    void SyncMortarPbcForStep(int step_idx);

    /**
     * @brief Update velocity field with current boundary condition values.
     *
     * Applies essential boundary conditions to the velocity field and updates the
     * solution vector with the prescribed velocity values. Handles both direct
     * velocity boundary conditions and velocity gradient boundary conditions that
     * require position-dependent velocity calculations.
     *
     * The function performs different operations based on boundary condition type:
     * - Direct velocity BCs: Projects boundary function onto velocity field
     * - Velocity gradient BCs: Computes position-dependent velocities from gradients
     * - Mixed BCs: Applies appropriate method for each boundary region
     *
     * For velocity gradient boundary conditions, the velocity is computed as:
     * v(x) = ∇v · (x - x_origin) where ∇v is the prescribed velocity gradient
     * and x_origin is the reference point for the deformation.
     *
     * Algorithm:
     * 1. Check for direct velocity boundary conditions
     * 2. Apply velocity boundary functions if present
     * 3. Handle velocity gradient boundary conditions
     * 4. Update solution vector with prescribed velocities
     * 5. Ensure compatibility with essential DOF constraints
     *
     * @note Called before each solve to ensure current BC values are applied
     * @note Critical for maintaining consistency between field values and constraints
     */
    void UpdateVelocity();

    /**
     * @brief Phase 5.8 — get the mortar PBC manager held by this
     *        driver, or nullptr if mortar PBC is not enabled.
     *
     * @details Returned shared_ptr is the same one held internally;
     * the manager outlives both the SystemDriver and any
     * PostProcessingDriver that consumes it as long as one
     * shared_ptr handle is kept alive.
     *
     * Used by mechanics_driver.cpp to pass the manager to the
     * PostProcessingDriver ctor, enabling fluctuation-field
     * visualization and per-step periodic validation diagnostics.
     */
    std::shared_ptr<mortar_pbc::MortarPbcManager> GetMortarPbcManager() const {
        return m_mortar_pbc;
    }

    virtual ~SystemDriver() = default;
};
#endif
