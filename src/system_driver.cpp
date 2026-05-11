
#include "system_driver.hpp"

#include "boundary_conditions/BCData.hpp"
#include "boundary_conditions/BCManager.hpp"
#include "solvers/trust_region_solver.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "ECMech_const.h"
#include "RAJA/RAJA.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <iostream>
#include <limits>

/**
 * @brief Dirichlet boundary condition function for MFEM integration
 *
 * @param attr_id Boundary attribute identifier from the mesh
 * @param y Output vector where boundary condition values will be set
 *
 * @details This function serves as the interface between MFEM's boundary condition
 * system and ExaConstit's boundary condition management. It is used as a callback
 * function during finite element assembly to apply Dirichlet boundary conditions.
 *
 * The function:
 * 1. Gets the singleton BCManager instance
 * 2. Retrieves the appropriate BCData instance for the given boundary attribute
 * 3. Applies the boundary condition values to the output vector
 *
 * This function is typically passed to MFEM's VectorFunctionRestrictedCoefficient
 * or similar boundary condition mechanisms during system setup.
 *
 * @note The attr_id corresponds to mesh boundary attributes and must match the
 * boundary IDs used during BCManager initialization.
 */
void DirBdrFunc(int attr_id, mfem::Vector& y) {
    BCManager& bcManager = BCManager::GetInstance();
    BCData& bc = bcManager.GetBCInstance(attr_id);

    bc.SetDirBCs(y);
}

namespace {

void GetTrueDofsParallel(const mfem::ParGridFunction& gf, mfem::Vector& true_dofs) {
    // used to do something like:
    // gf.GetTrueDofs(true_dofs);
    // but looks like there are issues with that on the GPUs with newer versions of MFEM
    gf.ParallelAverage(true_dofs);
}

/**
 * @brief Helper function to find mesh bounding box for velocity gradient calculations
 *
 * @tparam T Device execution policy type (CPU/GPU)
 * @param space_dim Spatial dimension of the problem (2D or 3D)
 * @param nnodes Number of nodes in the mesh
 * @param class_device Device execution policy instance
 * @param nodes Pointer to mesh node coordinates vector
 * @param origin Output vector containing min and max coordinates [min_x, min_y, min_z, max_x,
 * max_y, max_z]
 *
 * @details Calculates the minimum and maximum coordinates of the mesh nodes across all
 * spatial dimensions. This information is needed for velocity gradient boundary conditions
 * that require knowledge of the mesh extent.
 *
 * The function:
 * 1. Handles the MFEM node ordering (xxx..., yyy..., zzz... rather than xyz, xyz...)
 * 2. Uses device-compatible reduction operations for GPU execution
 * 3. Performs MPI reductions to find global min/max across all processes
 * 4. Stores results in the origin vector with min values first, then max values
 *
 * @note This is a template function to support different device execution policies.
 * The "NVCC is the bane of my existence" comment refers to CUDA compiler limitations
 * that necessitated this template approach.
 */
template <class T>
void min_max_helper(const int space_dim,
                    const size_t nnodes,
                    const T& class_device,
                    mfem::Vector* const nodes,
                    mfem::Vector& origin) {
    // Our nodes are by default saved in xxx..., yyy..., zzz... ordering rather
    // than xyz, xyz, ...
    // So, the below should get us a device reference that can be used.
    const auto X = mfem::Reshape(nodes->Read(), nnodes, space_dim);
    mfem::Vector min_origin(space_dim);
    min_origin = std::numeric_limits<double>::max();
    mfem::Vector max_origin(space_dim);
    max_origin = -std::numeric_limits<double>::max();

    min_origin.HostReadWrite();
    max_origin.HostReadWrite();
    // We need to calculate the minimum point in the mesh to get the correct velocity gradient
    // across the part.
    RAJA::RangeSegment default_range(0, static_cast<long>(nnodes));
    if (class_device == RTModel::CPU) {
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<RAJA::seq_reduce, double> seq_min(std::numeric_limits<double>::max());
            RAJA::ReduceMax<RAJA::seq_reduce, double> seq_max(-std::numeric_limits<double>::max());
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
                seq_min.min(X(i, j));
                seq_max.max(X(i, j));
            });
            min_origin(j) = seq_min.get();
            max_origin(j) = seq_max.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<RAJA::omp_reduce_ordered, double> omp_min(
                std::numeric_limits<double>::max());
            RAJA::ReduceMax<RAJA::omp_reduce_ordered, double> omp_max(
                -std::numeric_limits<double>::max());
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i) {
                omp_min.min(X(i, j));
                omp_max.max(X(i, j));
            });
            min_origin(j) = omp_min.get();
            max_origin(j) = omp_max.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<gpu_reduce, double> gpu_min(std::numeric_limits<double>::max());
            RAJA::ReduceMax<gpu_reduce, double> gpu_max(-std::numeric_limits<double>::max());
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i) {
                gpu_min.min(X(i, j));
                gpu_max.max(X(i, j));
            });
            min_origin(j) = gpu_min.get();
            max_origin(j) = gpu_max.get();
        }
    }
#endif
    MPI_Allreduce(min_origin.HostRead(),
                  origin.HostReadWrite(),
                  space_dim,
                  MPI_DOUBLE,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(max_origin.HostRead(),
                  &origin.HostReadWrite()[space_dim],
                  space_dim,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
} // End of finding max and min locations

/// @brief Check whether the user configured at least one
///        velocity-gradient BC.
///
/// Phase 5.5 — gates the mortar PBC enable. Mortar PBC requires a
/// velocity-gradient BC to be the loading mechanism (the corners
/// pinned to v = L̄·x), so absence of any vgrad BC means mortar
/// PBC is not in use even if `mesh.periodicity = true`.
///
/// Both the modern `velocity_gradient_bcs` array and the legacy
/// `essential_vel_grad` must be considered (the legacy format
/// is transformed into the modern `vgrad_bcs` vector during
/// `BoundaryOptions::validate`, so by the time SystemDriver is
/// constructed both populate the same vector).
bool HasVelocityGradientBC(const ExaOptions& opts)
{
    return !opts.boundary_conditions.vgrad_bcs.empty();
}
} // namespace

bool is_vgrad_option_flag(const std::shared_ptr<SimulationState> sim_state) {
    const auto& bo = sim_state->GetOptions().boundary_conditions;
    if (bo.vgrad_bcs.size() > 0) {
        if (bo.vgrad_bcs[0].origin) {
            return true;
        }
    }
    return false;
}

bool is_expt_mono_flag(const std::shared_ptr<SimulationState> sim_state) {
    return sim_state->GetOptions().boundary_conditions.mono_def_bcs;
}

SystemDriver::SystemDriver(std::shared_ptr<SimulationState> sim_state)
    : class_device(sim_state->GetOptions().solvers.rtmodel),
      auto_time(sim_state->GetOptions().time.time_type == TimeStepType::AUTO),
      vgrad_origin_flag(is_vgrad_option_flag(sim_state)),
      mono_def_flag(is_expt_mono_flag(sim_state)), m_sim_state(sim_state) {
    CALI_CXX_MARK_SCOPE("system_driver_init");

    const auto& options = sim_state->GetOptions();

    auto mesh = m_sim_state->GetMesh();
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    const int space_dim = mesh->SpaceDimension();
    // set the size of the essential boundary conditions attribute array
    ess_bdr["total"] = mfem::Array<int>();
    ess_bdr["total"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["total"] = 0;
    ess_bdr["ess_vel"] = mfem::Array<int>();
    ess_bdr["ess_vel"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["ess_vel"] = 0;
    ess_bdr["ess_vgrad"] = mfem::Array<int>();
    ess_bdr["ess_vgrad"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["ess_vgrad"] = 0;

    ess_bdr_component["total"] = mfem::Array2D<bool>();
    ess_bdr_component["total"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["total"] = false;
    ess_bdr_component["ess_vel"] = mfem::Array2D<bool>();
    ess_bdr_component["ess_vel"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["ess_vel"] = false;
    ess_bdr_component["ess_vgrad"] = mfem::Array2D<bool>();
    ess_bdr_component["ess_vgrad"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["ess_vgrad"] = false;

    ess_bdr_scale.SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_scale = 0.0;
    ess_velocity_gradient.SetSize(space_dim * space_dim, mfem::Device::GetMemoryType());
    ess_velocity_gradient.UseDevice(true);

    vgrad_origin.SetSize(space_dim, mfem::Device::GetMemoryType());
    vgrad_origin.UseDevice(true);
    if (vgrad_origin_flag) {
        vgrad_origin.HostReadWrite();
        vgrad_origin = 0.0;
        // already checked if this exists
        auto origin = sim_state->GetOptions().boundary_conditions.vgrad_bcs[0].origin;
        vgrad_origin(0) = (*origin)[0];
        vgrad_origin(1) = (*origin)[1];
        vgrad_origin(2) = (*origin)[2];
    }

    // Set things to the initial step
    BCManager::GetInstance().GetUpdateStep(1);
    BCManager::GetInstance().UpdateBCData(
        ess_bdr, ess_bdr_scale, ess_velocity_gradient, ess_bdr_component);
    mech_operator = std::make_shared<NonlinearMechOperator>(
        ess_bdr["total"], ess_bdr_component["total"], m_sim_state);
    model = mech_operator->GetModel();

    if (mono_def_flag) {
        const auto nodes = mesh->GetNodes();
        const int nnodes = nodes->Size() / space_dim;
        mfem::Vector origin(space_dim * 2, mfem::Device::GetMemoryType());
        origin.UseDevice(true);
        origin = 0.0;
        // Just scoping variable usage so we can reuse variables if we'd want to
        // CUDA once again is limiting us from writing normal C++
        // code so had to move to a helper function for this part...
        min_max_helper(space_dim, static_cast<size_t>(nnodes), class_device, nodes, origin);

        mfem::Array<int> ess_vdofs, ess_tdofs, ess_true_dofs;
        ess_vdofs.SetSize(fe_space->GetVSize());
        ess_vdofs = 0;
        // We need to set the ess_vdofs doing something like ess_vdofs[i] = -1;
        // However, the compiler thinks ess_vdofs is const when trying to do this in
        // the later loop, so we turn to lambda fcns to do this so the compiler picks
        // the right mfem::Array::operator[](int i) fcn.
        auto f = [&ess_vdofs](int i) {
            ess_vdofs[i] = -1;
        };
        const auto X = mfem::Reshape(nodes->HostRead(), nnodes, space_dim);
        // For this we would need to set up the true dofs at start of simulation
        // before anything actually moves
        // X's dofs would be at global min(x, z)
        // Y's dofs would be at global min(x, y, z)
        // Z's dofs would be at global min(z) | global max(z)
        RAJA::RangeSegment default_range(0, nnodes);
        RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
            const double x_diff_min = std::abs(X(i, 0) - origin(0));
            const double y_diff_min = std::abs(X(i, 1) - origin(1));
            const double z_diff_min = std::abs(X(i, 2) - origin(2));
            const double z_diff_max = std::abs(X(i, 2) - origin(5));
            if (x_diff_min < 1e-12 && z_diff_min < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 0);
                f(dof);
            }
            if (x_diff_min < 1e-12 && y_diff_min < 1e-12 && z_diff_min < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 1);
                f(dof);
            }
            if (z_diff_min < 1e-12 || z_diff_max < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 2);
                f(dof);
            }
        }); // end loop over nodes
        // Taken from mfem::FiniteElementSpace::GetEssentialTrueDofs(...)
        fe_space->Synchronize(ess_vdofs);
        fe_space->GetRestrictionMatrix()->BooleanMult(ess_vdofs, ess_tdofs);
        fe_space->MarkerToList(ess_tdofs, ess_true_dofs);
        mech_operator->UpdateEssTDofs(ess_true_dofs, mono_def_flag);
    }

    ess_bdr_func = std::make_unique<mfem::VectorFunctionRestrictedCoefficient>(
        space_dim, DirBdrFunc, ess_bdr["ess_vel"], ess_bdr_scale);

    // Partial assembly we need to use a matrix free option instead for our preconditioner
    // Everything else remains the same.
    auto& linear_solvers = options.solvers.linear_solver;
    if (options.solvers.assembly != AssemblyType::FULL) {
        J_prec = mech_operator->GetPAPreconditioner();
    } else {
        if (linear_solvers.preconditioner == PreconditionerType::AMG) {
            auto prec_amg = std::make_shared<mfem::HypreBoomerAMG>();
            const int problem_dim = m_sim_state->GetMesh()->SpaceDimension();
            const bool order_bynodes = (fe_space->GetOrdering() == mfem::Ordering::byNODES);
            // Use MFEM's supported systems-AMG configuration so Hypre sees
            // the correct vector-valued DOF ordering on newer MFEM/Hypre builds.
            prec_amg->SetSystemsOptions(problem_dim, order_bynodes);
            prec_amg->SetPrintLevel(linear_solvers.print_level);
            J_prec = prec_amg;
        } else if (linear_solvers.preconditioner == PreconditionerType::ILU) {
            auto J_hypreEuclid = std::make_shared<mfem::HypreEuclid>(fe_space->GetComm());
            J_prec = J_hypreEuclid;
        } else if (linear_solvers.preconditioner == PreconditionerType::L1GS) {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::l1GS);
            J_hypreSmoother->SetPositiveDiagonal(true);
            J_prec = J_hypreSmoother;
        } else if (linear_solvers.preconditioner == PreconditionerType::CHEBYSHEV) {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::Chebyshev);
            J_prec = J_hypreSmoother;
        } else {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
            J_hypreSmoother->SetPositiveDiagonal(true);
            J_prec = J_hypreSmoother;
        }
    }

    if (linear_solvers.solver_type == LinearSolverType::GMRES) {
        J_solver = std::make_shared<mfem::GMRESSolver>(fe_space->GetComm());
    } else if (linear_solvers.solver_type == LinearSolverType::CG) {
        J_solver = std::make_shared<mfem::CGSolver>(fe_space->GetComm());
    } else if (linear_solvers.solver_type == LinearSolverType::BICGSTAB) {
        J_solver = std::make_shared<mfem::BiCGSTABSolver>(fe_space->GetComm());
    } else {
        J_solver = std::make_shared<mfem::MINRESSolver>(fe_space->GetComm());
    }

    // The relative tolerance should be at this point or smaller
    J_solver->SetRelTol(linear_solvers.rel_tol);
    // The absolute tolerance could probably get even smaller then this
    J_solver->SetAbsTol(linear_solvers.abs_tol);
    J_solver->SetMaxIter(linear_solvers.max_iter);
    J_solver->SetPrintLevel(linear_solvers.print_level);
    J_solver->SetPreconditioner(*J_prec);

    auto nonlinear_solver = options.solvers.nonlinear_solver;
    newton_iter = nonlinear_solver.iter;
    if (nonlinear_solver.nl_solver == NonlinearSolverType::NR) {
        newton_solver = std::make_unique<ExaNewtonSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());
    }
    else if (nonlinear_solver.nl_solver == NonlinearSolverType::NRLS) {
        newton_solver = std::make_unique<ExaNewtonLSSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());
    }
    else if (nonlinear_solver.nl_solver == NonlinearSolverType::TRDOG) {
        // Build the trust-region dogleg solver and configure delta-control
        // parameters from the parsed TOML options. If the user did not supply
        // a [trust_region] sub-table, the solver's internal defaults (matching
        // SNLS's TrDeltaControl defaults) are used.
        auto tr_solver = std::make_unique<ExaTrustRegionSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());

        if (nonlinear_solver.trust_region.has_value()) {
            const auto& tr_opts = nonlinear_solver.trust_region.value();
            TrDeltaControl ctrl;
            ctrl.deltaInit         = tr_opts.delta_init;
            ctrl.deltaMin          = tr_opts.delta_min;
            ctrl.deltaMax          = tr_opts.delta_max;
            ctrl.xiLG              = tr_opts.xi_lg;
            ctrl.xiUG              = tr_opts.xi_ug;
            ctrl.xiLO              = tr_opts.xi_lo;
            ctrl.xiUO              = tr_opts.xi_uo;
            ctrl.xiIncDelta        = tr_opts.xi_inc;
            ctrl.xiDecDelta        = tr_opts.xi_dec;
            ctrl.xiForcedIncDelta  = tr_opts.xi_forced_inc;
            ctrl.rejectResIncrease = tr_opts.reject_increase;
            tr_solver->SetTrustRegionControl(ctrl);
        }

        newton_solver = std::move(tr_solver);

        // Sanity check: TRDOG requires gradient transpose support (J^T*r). For
        // PA mode, this requires the native PA transpose kernels in the
        // integrator. EA and FULL always support transpose. We warn rather than
        // hard-fail here because PA support exists once the kernels are wired.
        if (options.solvers.assembly == AssemblyType::PA) {
            mfem::out << "Note: TRDOG with PA assembly requires native PA transpose "
                      << "kernels in the gradient operator.\n";
        }
    }

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver(J_solver);
    newton_solver->SetOperator(mech_operator);
    newton_solver->SetPrintLevel(1);
    newton_solver->SetRelTol(nonlinear_solver.rel_tol);
    newton_solver->SetAbsTol(nonlinear_solver.abs_tol);
    newton_solver->SetMaxIter(nonlinear_solver.iter);

    //--------------------------------------------------------------------------
    // Phase 5.5.A — mortar PBC enable
    //
    // Detect mortar PBC, build the MortarPbcManager (which constructs
    // the boundary classifier, constraint builder, EA constraint
    // operator, saddle system adapter, and SaddlePointSolver), then
    // override the mech_operator's essential-TDOF list with the
    // 24-corner subset returned by the manager (Phase 5.4
    // UpdateEssTDofsCornerSubset).
    //
    // newton_solver / J_solver / J_prec stay wired to mech_operator
    // for the non-mortar code path (which `Solve()` will continue to
    // use when m_mortar_enabled == false). The mortar path bypasses
    // newton_solver entirely (architecture β; see Phase 5.5.A
    // insertion guide for rationale) — `Solve()` runs an explicit
    // saddle Newton loop in 5.5.B.
    //--------------------------------------------------------------------------
    {
        const bool mortar_requested =
            options.mesh.periodicity && HasVelocityGradientBC(options);

        if (mortar_requested)
        {
            CALI_CXX_MARK_SCOPE("system_driver::ctor::mortar_setup");

            MFEM_VERIFY(mech_operator != nullptr,
                        "Mortar PBC: mech_operator must be constructed "
                        "before the manager (the K closures capture it).");

            // K closures — captured by raw pointer; mech_operator
            // is held by SystemDriver as shared_ptr and outlives
            // the manager (asserted at ~MortarPbcManager via
            // §P5.14.5 — the manager doesn't outlive SystemDriver).
            auto k_residual =
                [op_ptr = mech_operator.get()](const mfem::Vector& v,
                                               mfem::Vector& r) {
                    op_ptr->Mult(v, r);
                };
            auto k_jacobian =
                [op_ptr = mech_operator.get()](const mfem::Vector& v)
                    -> mfem::Operator* {
                    return &op_ptr->GetGradient(v);
                };

            // Build the manager. Constructor is collective on the
            // mesh communicator and builds the classifier, builder,
            // C operator, saddle system, saddle solver, lambda
            // buffer, macroscopic F̄ = I, and the per-row reference
            // factor cache.
            m_mortar_pbc =
                std::make_shared<mortar_pbc::MortarPbcManager>(
                    m_sim_state, k_residual, k_jacobian);

            // m_mortar_enabled must be set before SyncMortarPbcForStep
            // because SyncMortarPbcForStep early-returns on false.
            m_mortar_enabled = true;

            // Phase 5.9 / Batch A.5 — install the initial periodic-BC
            // spec for step 1. This replaces the pre-5.9 inline call
            // to `mech_operator->UpdateEssTDofsCornerSubset(
            // m_mortar_pbc->GetCornerEssTDofs())`. The Sync method
            // handles all four cases:
            //   * empty periodic_bcs  → synthesize default full-PBC
            //     spec and install (matches pre-5.9 24-corner behavior).
            //   * periodic_bcs[0]     → install that spec.
            //   * default already installed (re-init) → no-op.
            //   * step missing from map + not initialized → abort.
            //
            // After the call, m_mortar_pbc->GetCornerEssTDofs() is
            // the spec-derived subset and mech_operator has been
            // updated accordingly.
            SyncMortarPbcForStep(1);

            // ====================================================================
            // Phase 5.5.B.4 — saddle preconditioner + saddle-system Newton wiring
            // ====================================================================
            //
            // K-Jacobi preconditioner dispatched by assembly mode,
            // following the existing J_prec pattern. Both branches
            // produce a Solver whose Mult(ones, _) returns
            // inv_diag(K), which is the contract
            // SaddlePointSolver::Solve and MortarConstraintOperator::
            // ComputeInvDiagSchur depend on.
            //
            // PA / EA: reuse the MechOperatorJacobiSmoother that
            //          mech_operator already manages. Same instance
            //          the production J_prec uses in those modes;
            //          GPU-compatible.
            //
            // FA:      HypreSmoother(type=Jacobi), default-constructed.
            //          SetOperator is called per Newton iter by
            //          MortarSaddlePreconditioner::SetOperator (and
            //          directly by SystemDriver::SolveInit's mortar
            //          branch).
            if (options.solvers.assembly != AssemblyType::FULL) {
                m_K_jacobi_prec = mech_operator->GetPAPreconditioner();
            }
            else {
                auto K_jacobi_hp = std::make_shared<mfem::HypreSmoother>();
                K_jacobi_hp->SetType(mfem::HypreSmoother::Jacobi);
                m_K_jacobi_prec = K_jacobi_hp;
            }

            // Save the user's chosen J_prec before swapping J_prec out
            // — this becomes the K-BLOCK preconditioner inside
            // MortarSaddlePreconditioner. In FA this can be AMG / ILU /
            // L1GS / Chebyshev / l1Jacobi (the user's TOML choice); in
            // PA / EA this is also MechOperatorJacobiSmoother (so
            // K_block_prec and m_K_jacobi_prec end up as the same
            // instance, harmless: SetOperator is idempotent at the
            // operator-pointer level).
            auto K_block_prec = J_prec;

            // Build the saddle preconditioner. This is the new J_prec
            // that the Krylov inside the Newton's CGSolver delegates to.
            // Its SetOperator(saddle_BlockOperator) extracts K from
            // block(0,0), refreshes K_block_prec and m_K_jacobi_prec,
            // and computes inv_diag_S via ComputeInvDiagSchur.
            m_mortar_saddle_prec =
                std::make_shared<mortar_pbc::MortarSaddlePreconditioner>(
                    K_block_prec,
                    m_K_jacobi_prec,
                    m_mortar_pbc->GetConstraintOperator());

            J_prec = m_mortar_saddle_prec;
            J_solver->SetPreconditioner(*J_prec);

            // Allocate m_x_saddle (BlockVector scratch). Block layout:
            // [u | lambda]. Sized from the mech_operator's local TDOF
            // count and the manager's local constraint count.
            const int n_K   = mech_operator->Width();
            const int n_lam = m_mortar_pbc->NumLocalConstraints();
            m_saddle_offsets.SetSize(3);
            m_saddle_offsets[0] = 0;
            m_saddle_offsets[1] = n_K;
            m_saddle_offsets[2] = n_K + n_lam;
            m_x_saddle = std::make_unique<mfem::BlockVector>(m_saddle_offsets);
            *m_x_saddle = 0.0;

            // Override the Newton solver's operator. The 5.5.A branch's
            // earlier `newton_solver->SetOperator(mech_operator)` is
            // replaced here with the saddle system, which is also an
            // mfem::Operator (post-5.5.B.1 ExaNewtonSolver accepts any
            // shared_ptr<Operator>). The Newton's Mult body now iterates
            // against [F_int(u) + C^T·lambda; C·u - g] = 0.
            newton_solver->SetOperator(m_mortar_pbc->GetSaddleSystem());
        }
    }

}

const mfem::Array<int>& SystemDriver::GetEssTDofList() {
    return mech_operator->GetEssTDofList();
}

// Solve the Newton system.
//
// Phase 5.5.B.4 — single shared body for mortar and production paths.
// The auto_time retry loop is captured in a local lambda
// (`run_with_retries`) that takes the Newton iterate by reference
// plus a `pre_attempt` callable. Production passes the PrimalField
// + a no-op pre_attempt; mortar passes m_x_saddle + a callback
// that refreshes the manager's macroscopic state and repacks
// m_x_saddle from PrimalField + accumulated lambda. Post-solve
// unpack (mortar-only) and the convergence check + ess_bdr_func
// time stamp (shared) follow.
void SystemDriver::Solve() {
    CALI_CXX_MARK_SCOPE("system_driver::solve");

    mfem::Vector zero;

    // Auto_time retry loop, shared by mortar and production paths.
    // pre_attempt() runs once before each Newton attempt (initial
    // + each retry). On retry we call SimulationState::RestartCycle
    // to roll mesh state back, then pre_attempt again so the mortar
    // path can re-anchor F̄ on the restored mesh state with the
    // new (smaller) dt.
    auto run_with_retries = [&](mfem::Vector& x_iter, auto pre_attempt) {
        if (auto_time) {
            pre_attempt();

            bool succeed_t = false;
            bool succeed   = false;
            try {
                newton_solver->Mult(zero, x_iter);
                succeed_t = newton_solver->GetConverged();
            }
            catch (const std::exception& exc) {
                MFEM_WARNING_0(exc.what());
                succeed_t = false;
            }
            catch (...) {
                MFEM_WARNING_0(
                    "An unknown exception was thrown in Krylov solver step");
                succeed_t = false;
            }
            MPI_Allreduce(&succeed_t, &succeed, 1, MPI_C_BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
            TimeStep state = m_sim_state->UpdateDeltaTime(
                newton_solver->GetNumIterations(), succeed);

            if (!succeed) {
                while (state == TimeStep::RETRIAL) {
                    MFEM_WARNING_0(
                        "Solution did not converge decreasing dt by input scale factor");
                    if (m_sim_state->GetMPIID() == 0) {
                        m_sim_state->PrintRetrialTimeStats();
                    }
                    m_sim_state->RestartCycle();
                    pre_attempt();

                    try {
                        newton_solver->Mult(zero, x_iter);
                        succeed_t = newton_solver->GetConverged();
                    }
                    catch (...) {
                        succeed_t = false;
                    }
                    MPI_Allreduce(&succeed_t, &succeed, 1, MPI_C_BOOL,
                                  MPI_LAND, MPI_COMM_WORLD);
                    state = m_sim_state->UpdateDeltaTime(
                        newton_solver->GetNumIterations(), succeed);
                }
            }
        }
        else {
            pre_attempt();
            newton_solver->Mult(zero, x_iter);
            m_sim_state->UpdateDeltaTime(
                newton_solver->GetNumIterations(), true);
        }
    };

    if (m_mortar_enabled) {
        // Mortar path. pre_attempt rebuilds L̄ from
        // ess_velocity_gradient (Vector size 9, row-major), refreshes
        // the manager's tracked F̄ + Ḟ̄ (mesh-anchored, idempotent
        // across RestartCycle), refreshes the constraint RHS buffer,
        // then packs m_x_saddle from PrimalField + accumulated lambda.
        auto pre_attempt = [&]() {
            mfem::DenseMatrix Lbar(3, 3);
            const double* L_data = ess_velocity_gradient.HostRead();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Lbar(i, j) = L_data[i * 3 + j];
                }
            }
            const double dt = m_sim_state->GetDeltaTime();
            m_mortar_pbc->UpdateMacroscopicF(Lbar, dt);
            m_mortar_pbc->UpdateConstraintRHS();
            m_x_saddle->GetBlock(0) = *m_sim_state->GetPrimalField();
            m_x_saddle->GetBlock(1) = m_mortar_pbc->GetAccumulatedLambda();
        };

        run_with_retries(*m_x_saddle, pre_attempt);

        // Unpack: copy converged u-block back to PrimalField (defensive
        // — the K-residual closure operates on a view into
        // m_x_saddle->GetBlock(0), so its UpdateEndCoords side effect
        // already syncs PrimalField; the explicit copy makes the
        // post-condition robust against future closure refactors).
        // Overwrite manager's accumulated lambda with the converged
        // multiplier.
        m_mortar_pbc->SetAccumulatedLambda(m_x_saddle->GetBlock(1));

    }
    else {
        // Production path. PrimalField is the iterate; no pre-attempt
        // setup beyond what UpdateVelocity has already done.
        run_with_retries(*m_sim_state->GetPrimalField(), [](){});
    }

    // Shared post-solve invariants. Once the system has finished
    // solving, our current coordinates configuration is based on
    // what our converged velocity field ended up being equal to.
    if (m_sim_state->GetMPIID() == 0 && newton_solver->GetConverged()) {
        ess_bdr_func->SetTime(m_sim_state->GetTime());
    }
    MFEM_VERIFY_0(newton_solver->GetConverged(),
                  "Newton Solver did not converge.");

    // Phase 5.8 — post-convergence mortar-PBC field updates and
    // diagnostic caching. Three things happen here, all gated on the
    // manager pointer being non-null (= mortar PBC enabled):
    //   1. ComputeFluctuationField:  v_tilde = v_total − L̄·x  →
    //      sim_state->GetFluctuationField()
    //   2. ComputeAffineVelocityField: v_lin = L̄·x  →
    //      sim_state->GetAffineVelocityField()
    //   3. If [PostProcessing.volume_averages] periodic_validation
    //      is true, cache the ConstraintConsistencyDiagnostic and
    //      HillMandelDiagnostic structs on the manager via
    //      CachePerStepDiagnostics. PostProcessingDriver reads
    //      these in PrintPeriodicValidation each output step.
    //
    // All three operations are cheap: ComputeFluctuationField /
    // ComputeAffineVelocityField are O(N_TDOFs) projections;
    // CachePerStepDiagnostics is one C-matvec + a couple of
    // Allreduces (DiagnoseConstraintConsistency) plus one quadrature
    // sweep over kinetic_grads + cauchy_stress_end
    // (ComputeHillMandelPowerBalance).
    if (m_mortar_pbc) {
        const mfem::DenseMatrix& Lbar = m_mortar_pbc->GetLbar();
        const mfem::Vector&      velocity = *m_sim_state->GetPrimalField();

        if (auto v_tilde_gf = m_sim_state->GetFluctuationField()) {
            m_mortar_pbc->ComputeFluctuationField(velocity, Lbar, *v_tilde_gf);
        }
        if (auto v_lin_gf = m_sim_state->GetAffineVelocityField()) {
            m_mortar_pbc->ComputeAffineVelocityField(Lbar, *v_lin_gf);
        }

        const auto& vol_opts =
            m_sim_state->GetOptions().post_processing.volume_averages;
        if (vol_opts.periodic_validation) {
            // Compute the internal-force residual at the converged
            // velocity (BC-eliminated form — Trap 4 in the
            // HillMandelDiagnostic docstring; corner DOFs out of
            // millions are diagnostic noise).
            mfem::Vector r_internal(velocity.Size(),
                                    mfem::Device::GetMemoryType());
            r_internal = 0.0;
            mech_operator->Mult(velocity, r_internal);

            m_mortar_pbc->CachePerStepDiagnostics(velocity, r_internal);
        }
    }
}

// Solve the Newton system for the 1st time step.
// It was found that for large meshes a ramp up to our desired
// applied BC might be needed.
//
// Phase 5.5.B.4 — single shared body for mortar and production
// paths. The corner-deltaF kernel, GetUpdateBCsAction call, and
// Velocity::Distribute tail are identical between paths and are
// shared. The actual linearized solve differs — production routes
// through newton_solver->CGSolver (delegates to J_solver, which
// does the K-only Krylov solve); mortar must call SaddlePointSolver
// directly because J_prec under mortar is MortarSaddlePreconditioner,
// which expects a saddle BlockOperator and would dynamic_cast-abort
// on the K-only `oper` from GetUpdateBCsAction. The two paths also
// have different sign conventions on the velocity update (production
// `X = -X + XPREV`; mortar `X = XPREV + DU`).
void SystemDriver::SolveInit() const {
    CALI_CXX_MARK_SCOPE("system_driver::solve_init");

    const auto x      = m_sim_state->GetPrimalField();
    const auto x_prev = m_sim_state->GetPrimalFieldPrev();

    // Mortar pre-step: refresh manager's macroscopic state and
    // constraint RHS so the linearized saddle solve sees the right
    // g vector.
    if (m_mortar_enabled) {
        mfem::DenseMatrix Lbar(3, 3);
        const double* L_data = ess_velocity_gradient.HostRead();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Lbar(i, j) = L_data[i * 3 + j];
            }
        }
        const double dt = m_sim_state->GetDeltaTime();
        m_mortar_pbc->UpdateMacroscopicF(Lbar, dt);
        m_mortar_pbc->UpdateConstraintRHS();
    }

    // Shared: build deltaF (corner Dirichlet contribution) and
    // the K-with-elimination operator. Phase 5.4's
    // UpdateEssTDofsCornerSubset has narrowed
    // GetEssentialTrueDofs() to the 24 corner TDOFs under mortar;
    // production keeps the full essential-TDOF set. Either way,
    // the kernel below writes deltaF only at those essential TDOFs.
    //
    // K_uc * (x - x_prev)_c = b
    mfem::Vector b(*x);      b.UseDevice(true);      b      = 0.0;
    mfem::Vector deltaF(*x); deltaF.UseDevice(true); deltaF = 0.0;
    {
        auto I        = mech_operator->GetEssentialTrueDofs().Read();
        auto size     = mech_operator->GetEssentialTrueDofs().Size();
        auto Y        = deltaF.Write();
        auto XPREV    = x_prev->Read();
        auto X_in     = x->Read();
        mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) {
            Y[I[i]] = X_in[I[i]] - XPREV[I[i]];
        });
    }
    mfem::Operator& oper =
        mech_operator->GetUpdateBCsAction(*x_prev, deltaF, b);

    // Path-specific: linearized solve + apply.
    if (m_mortar_enabled) {
        // Refresh the K-Jacobi preconditioner against this oper
        // — the saddle solver probes K_jacobi_prec for inv_diag(K)
        // internally. (In the Newton path this is done implicitly
        // by MortarSaddlePreconditioner::SetOperator.)
        m_K_jacobi_prec->SetOperator(oper);

        // r2 = C · x_prev - g. SaddlePointSolver builds RHS = -r2
        // for the bottom row, so this gives us
        //   C · du = g - C · x_prev,
        // i.e., the new state u = x_prev + du satisfies C · u = g.
        mfem::Vector r2(m_mortar_pbc->NumLocalConstraints());
        m_mortar_pbc->GetConstraintOperator().Mult(*x_prev, r2);
        r2 -= m_mortar_pbc->GetConstraintRHS();

        // Direct saddle solve. Bypasses J_prec / J_solver entirely;
        // SaddlePointSolver builds its own internal BlockOperator +
        // BlockDiagonalPreconditioner.
        mfem::Vector du, dlam;
        m_mortar_pbc->GetSaddleSolver().Solve(
            oper,
            m_mortar_pbc->GetConstraintOperator(),
            *m_K_jacobi_prec,
            b, r2, du, dlam);

        // Apply: x = x_prev + du (production sign convention is
        // flipped — see comment block below for production path).
        auto X     = x->ReadWrite();
        auto DU    = du.Read();
        auto XPREV = x_prev->Read();
        mfem::forall(x->Size(), [=] MFEM_HOST_DEVICE(int i) {
            X[i] = XPREV[i] + DU[i];
        });

        // Lambda: SolveInit is the first call of the time step;
        // the manager's accumulated lambda is the warm-start
        // baseline (zero on the very first step, the previous
        // step's converged lambda thereafter). The linearized
        // solve produced an INCREMENT dlam from that baseline,
        // so accumulate.
        m_mortar_pbc->AccumulateLambdaContribution(dlam, 1.0);
    }
    else {
        // Production path — the original pre-5.5.B.4 logic.
        x->operator=(0.0);
        // CGSolver gives us the -change in velocity, so we want to
        // add the previous velocity terms to it.
        newton_solver->CGSolver(oper, b, *x);
        auto X     = x->ReadWrite();
        auto XPREV = x_prev->Read();
        mfem::forall(x->Size(), [=] MFEM_HOST_DEVICE(int i) {
            X[i] = -X[i] + XPREV[i];
        });
    }

    // Shared tail.
    m_sim_state->GetVelocity()->Distribute(*x);
}

//==============================================================================
// SyncMortarPbcForStep — Phase 5.9 / Batch A.5
//
// Bridge between the user-facing [[BCs.periodic_bcs]] TOML schema
// and the MortarPbcManager's spec-driven RebuildForActiveSpec API.
//
// See system_driver.hpp for the state-machine narrative.
//==============================================================================
void SystemDriver::SyncMortarPbcForStep(int step_idx)
{
    CALI_CXX_MARK_SCOPE("system_driver::sync_mortar_pbc_for_step");

    if (!m_mortar_enabled)
    {
        return;
    }

    const auto& boundary_opts =
        m_sim_state->GetOptions().boundary_conditions;
    const auto& periodic_bcs       = boundary_opts.periodic_bcs;
    const auto& entry_per_step_map = boundary_opts.periodic_bc_entry_per_step;

    // -----------------------------------------------------------------
    // Branch A — empty periodic_bcs (default-fallback synthesis).
    //
    // The synthesized default is step-invariant: it covers all face
    // pairs in the classifier with essential_comps = 7 (XYZ). So
    // after the first install, every subsequent call is a no-op.
    // -----------------------------------------------------------------
    if (periodic_bcs.empty())
    {
        if (m_pbc_initialized)
        {
            return;                       // synthesized default already installed
        }

        auto synth = mortar_pbc::MortarPbcManager::SynthesizeDefaultPbcSpec(
            m_mortar_pbc->GetClassifier());
        m_mortar_pbc->RebuildForActiveSpec(synth.first, synth.second);
        mech_operator->UpdateEssTDofsCornerSubset(
            m_mortar_pbc->GetCornerEssTDofs());

        // Phase 5.9.A.5 hotfix — same as the entry-driven branch:
        // resize m_x_saddle and re-tell the Newton solver. For the
        // very-first SyncMortarPbcForStep call from the ctor this
        // is a no-op (m_x_saddle is null then).
        if (m_x_saddle)
        {
            const int n_K   = mech_operator->Width();
            const int n_lam = m_mortar_pbc->NumLocalConstraints();
            m_saddle_offsets[1] = n_K;
            m_saddle_offsets[2] = n_K + n_lam;
            m_x_saddle = std::make_unique<mfem::BlockVector>(m_saddle_offsets);
            *m_x_saddle = 0.0;
            newton_solver->SetOperator(m_mortar_pbc->GetSaddleSystem());
        }

        m_pbc_initialized = true;
        m_pbc_active_entry_idx = -1;
        return;
    }

    // -----------------------------------------------------------------
    // Branch B — non-empty periodic_bcs. Look up target entry for
    // this step in periodic_bc_entry_per_step.
    // -----------------------------------------------------------------
    int target_entry_idx = -1;
    auto it = entry_per_step_map.find(step_idx);
    if (it == entry_per_step_map.end())
    {
        // Missing transition for this step. Two cases:
        //   - Already initialized (mid-run, sparse update_steps):
        //     keep the current spec; do nothing.
        //   - Not initialized (first call, step_idx not in map):
        //     this is a configuration error — the user's
        //     update_steps schedule should contain the simulation's
        //     start step.
        if (m_pbc_initialized)
        {
            return;
        }
        MFEM_ABORT("SystemDriver::SyncMortarPbcForStep: step_idx "
                   << step_idx
                   << " has no entry in "
                      "options.boundary_conditions.periodic_bc_entry_per_step"
                   << " and no periodic-BC spec is currently installed. "
                      "The TOML's BCs.update_steps schedule should include "
                      "the simulation's start step (typically 1).");
    }
    target_entry_idx = it->second;
    MFEM_VERIFY(target_entry_idx >= 0
                && target_entry_idx < static_cast<int>(periodic_bcs.size()),
                "SystemDriver::SyncMortarPbcForStep: entry index "
                << target_entry_idx << " (for step " << step_idx
                << ") is out of range [0, " << periodic_bcs.size()
                << "). The TOML parser's periodic_bc_entry_per_step "
                "map is inconsistent with periodic_bcs.size().");

    // -----------------------------------------------------------------
    // Idempotence — skip the rebuild if we're already on this entry.
    // -----------------------------------------------------------------
    if (m_pbc_initialized && target_entry_idx == m_pbc_active_entry_idx)
    {
        return;
    }

    // -----------------------------------------------------------------
    // Apply the target spec.
    // -----------------------------------------------------------------
    const auto& spec = periodic_bcs[target_entry_idx];
    m_mortar_pbc->RebuildForActiveSpec(spec.essential_ids,
                                       spec.essential_comps);
    mech_operator->UpdateEssTDofsCornerSubset(
        m_mortar_pbc->GetCornerEssTDofs());

    // Phase 5.9.A.5 hotfix — re-size the saddle-system block vector
    // scratch to the new local row count. m_x_saddle is unset when
    // SyncMortarPbcForStep runs from the ctor before the saddle
    // prec block; in that case the existing ctor allocation site
    // (later in the same ctor) handles sizing correctly using the
    // already-updated NumLocalConstraints(). For mid-run transitions
    // (e.g. multi-entry runs switching specs at an update_step
    // boundary), m_x_saddle exists and needs reallocation.
    if (m_x_saddle)
    {
        const int n_K   = mech_operator->Width();
        const int n_lam = m_mortar_pbc->NumLocalConstraints();
        m_saddle_offsets[1] = n_K;
        m_saddle_offsets[2] = n_K + n_lam;
        m_x_saddle = std::make_unique<mfem::BlockVector>(m_saddle_offsets);
        *m_x_saddle = 0.0;

        // Re-tell the Newton solver about the saddle system. Even
        // though it's the same shared_ptr<Operator>, some Newton
        // implementations cache height/width at SetOperator time.
        // After Refresh those values changed; re-SetOperator forces
        // any such cache to refill.
        newton_solver->SetOperator(m_mortar_pbc->GetSaddleSystem());
    }

    m_pbc_initialized = true;
    m_pbc_active_entry_idx = target_entry_idx;
}

void SystemDriver::UpdateEssBdr() {
   if (!mono_def_flag) {
      BCManager::GetInstance().UpdateBCData(ess_bdr, ess_bdr_scale,
                                            ess_velocity_gradient,
                                            ess_bdr_component);

      if (m_mortar_enabled) {
         // Phase 5.5.A — corner TDOFs are step-invariant on a fixed
         // mesh, so re-asserting them is logically a no-op. Doing
         // it anyway ensures the corner subset survives in case
         // mech_operator's internal state somehow changes between
         // calls; cheap and clearer than skipping.
         mech_operator->UpdateEssTDofsCornerSubset(
            m_mortar_pbc->GetCornerEssTDofs());
      }
      else {
         mech_operator->UpdateEssTDofs(ess_bdr["total"], mono_def_flag);
      }
   }
}

// In the current form, we could honestly probably make use of velocity as our working array
void SystemDriver::UpdateVelocity() {
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    auto mesh = m_sim_state->GetMesh();
    auto velocity = m_sim_state->GetVelocity();
    auto vel_tdofs = m_sim_state->GetPrimalField();

    if (ess_bdr["ess_vel"].Sum() > 0) {
        // Now that we're doing velocity based we can just overwrite our data with the ess_bdr_func
        velocity->ProjectBdrCoefficient(*ess_bdr_func); // don't need attr list as input
                                                        // pulled off the
                                                        // VectorFunctionRestrictedCoefficient
        // populate the solution vector, v_sol, with the true dofs entries in v_cur.
        GetTrueDofsParallel(*velocity, *vel_tdofs);
    }

    if (ess_bdr["ess_vgrad"].Sum() > 0) {
        // Just scoping variable usage so we can reuse variables if we'd want to
        {
            const auto nodes = mesh->GetNodes();
            const int space_dim = mesh->SpaceDimension();
            const int nnodes = nodes->Size() / space_dim;

            // Our nodes are by default saved in xxx..., yyy..., zzz... ordering rather
            // than xyz, xyz, ...
            // So, the below should get us a device reference that can be used.
            const auto X = mfem::Reshape(nodes->Read(), nnodes, space_dim);
            const auto VGRAD = mfem::Reshape(ess_velocity_gradient.Read(), space_dim, space_dim);
            velocity->operator=(0.0);
            auto VT = mfem::Reshape(velocity->ReadWrite(), nnodes, space_dim);

            if (!vgrad_origin_flag) {
                vgrad_origin.HostReadWrite();
                // We need to calculate the minimum point in the mesh to get the correct velocity
                // gradient across the part.
                RAJA::RangeSegment default_range(0, nnodes);
                if (class_device == RTModel::CPU) {
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<RAJA::seq_reduce, double> seq_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
                            seq_min.min(X(i, j));
                        });
                        vgrad_origin(j) = seq_min.get();
                    }
                }
#if defined(RAJA_ENABLE_OPENMP)
                if (class_device == RTModel::OPENMP) {
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<RAJA::omp_reduce_ordered, double> omp_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i) {
                            omp_min.min(X(i, j));
                        });
                        vgrad_origin(j) = omp_min.get();
                    }
                }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                if (class_device == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA)
                    using gpu_reduce = RAJA::cuda_reduce;
                    using gpu_policy = RAJA::cuda_exec<1024>;
#else
                    using gpu_reduce = RAJA::hip_reduce;
                    using gpu_policy = RAJA::hip_exec<1024>;
#endif
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<gpu_reduce, double> gpu_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i) {
                            gpu_min.min(X(i, j));
                        });
                        vgrad_origin(j) = gpu_min.get();
                    }
                }
#endif
            } // End if vgrad_origin_flag
            mfem::Vector origin(space_dim, mfem::Device::GetMemoryType());
            origin.UseDevice(true);
            MPI_Allreduce(vgrad_origin.HostRead(),
                          origin.HostReadWrite(),
                          space_dim,
                          MPI_DOUBLE,
                          MPI_MIN,
                          MPI_COMM_WORLD);
            const double* dmin_x = origin.Read();
            // We've now found our minimum points so we can now go and calculate everything.
            mfem::forall(nnodes, [=] MFEM_HOST_DEVICE(int i) {
                for (int ii = 0; ii < space_dim; ii++) {
                    for (int jj = 0; jj < space_dim; jj++) {
                        // mfem::Reshape assumes Fortran memory layout
                        // which is why everything is the transpose down below...
                        VT(i, ii) += VGRAD(jj, ii) * (X(i, jj) - dmin_x[jj]);
                    }
                }
            });
        }
        {
            mfem::Vector vel_tdof_tmp(*vel_tdofs);
            vel_tdof_tmp.UseDevice(true);
            vel_tdof_tmp = 0.0;
            GetTrueDofsParallel(*velocity, vel_tdof_tmp);

            mfem::Array<int> ess_tdofs(mech_operator->GetEssentialTrueDofs());
            if (!mono_def_flag) {
                fe_space->GetEssentialTrueDofs(
                    ess_bdr["ess_vgrad"], ess_tdofs, ess_bdr_component["ess_vgrad"]);
            }
            auto I = ess_tdofs.Read();
            auto size = ess_tdofs.Size();
            auto Y = vel_tdofs->ReadWrite();
            const auto X = vel_tdof_tmp.Read();
            // vel_tdofs should already have the current solution
            mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) {
                Y[I[i]] = X[I[i]];
            });
        }
    } // end of if constant strain rate
}

void SystemDriver::UpdateModel() {
    model->UpdateModelVars();
    m_sim_state->UpdateModel();
    m_sim_state->SetupModelVariables();

    auto def_grad = m_sim_state->GetQuadratureFunction("kinetic_grads");
    mech_operator->CalculateDeformationGradient(*def_grad.get());
}