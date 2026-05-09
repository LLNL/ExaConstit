// Phase 5.3.A — MortarPbcManager implementation.
//
// The constructor wires the full mortar-PBC pipeline. The methods
// that 5.3.B–E will fill in are MFEM_ABORT'd here so that downstream
// code can be wired up against the real public API immediately while
// individual methods land incrementally.
//
// See mortar_pbc_manager.hpp for design rationale and member layout.

#include "mortar_pbc_manager.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <utility>

namespace mortar_pbc {

namespace {

//==============================================================================
// TranslateSaddleOpts — bridge between option-parser-side enums
// (SaddlePointSolverType / SaddlePointPreconditioner, defined in
// option_parser_v2.hpp) and the Phase 4.3 internal enums
// (KrylovType / SaddlePrecType, defined in saddle_point_solver.hpp).
//
// The two enum sets are deliberately separated so option_parser_v2
// can remain free of mortar_pbc dependencies. This translation
// function is the only place they meet.
//
// Aborts on unknown enum values — `ExaOptions::validate()` should
// have caught those upstream, but defensive-checking here surfaces
// any future enum additions that haven't been wired through.
//==============================================================================
SaddlePointSolverConfig TranslateSaddleOpts(const SaddlePointSolverOptions& opts)
{
    SaddlePointSolverConfig cfg;

    switch (opts.linear_solver)
    {
        case SaddlePointSolverType::MINRES:
            cfg.solver_type = KrylovType::MINRES;
            break;
        case SaddlePointSolverType::GMRES:
            cfg.solver_type = KrylovType::GMRES;
            break;
        case SaddlePointSolverType::BICGSTAB:
            cfg.solver_type = KrylovType::BiCGSTAB;
            break;
        default:
            MFEM_ABORT("MortarPbcManager: unknown SaddlePointSolverType "
                       << static_cast<int>(opts.linear_solver)
                       << ". Did ExaOptions::validate() pass?");
    }

    switch (opts.preconditioner)
    {
        case SaddlePointPreconditioner::BLOCK_JACOBI:
            cfg.prec_type = SaddlePrecType::BlockJacobi;
            break;
        case SaddlePointPreconditioner::NONE:
            cfg.prec_type = SaddlePrecType::None;
            break;
        default:
            MFEM_ABORT("MortarPbcManager: unknown SaddlePointPreconditioner "
                       << static_cast<int>(opts.preconditioner)
                       << ". Did ExaOptions::validate() pass?");
    }

    cfg.rel_tol     = opts.rel_tol;
    cfg.abs_tol     = opts.abs_tol;
    cfg.max_iter    = opts.max_iter;
    cfg.print_level = opts.print_level;
    // gmres_kdim is left at its SaddlePointSolverConfig default
    // (50). If/when ExaOptions grows a field for it, plumb it
    // through here.

    return cfg;
}

}  // anonymous namespace

//==============================================================================
// ComputeCornerEssTDofs — free function exercised by both the
// manager's BuildCornerEssTDofs (which adds an MPI sanity check on
// top) and the test_mortar_pbc_manager.cpp unit test (which avoids
// the cost of constructing a full SimulationState).
//
// Iterates the classifier's 8 corners (replicated on every rank);
// for each corner's three components (x/y/z) checks ownership via
// classifier.GtdofOwnerRank, and for owned components converts the
// global TDOF to a rank-local index using fes.GetMyTDofOffset(). The
// result is appended to the output Array<int>.
//
// Postcondition: across the classifier's communicator,
// MPI_Allreduce(SUM, output.Size()) equals 24.
//==============================================================================
mfem::Array<int> ComputeCornerEssTDofs(
    const BoundaryClassifier3D& classifier,
    const mfem::ParFiniteElementSpace& fes)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::compute_corner_ess_tdofs");

    const int my_rank = classifier.Rank();
    const HYPRE_BigInt my_offset = fes.GetMyTDofOffset();

    mfem::Array<int> out;
    out.Reserve(24);  // Upper bound: 8 corners × 3 components.

    for (const auto& kv : classifier.Corners())
    {
        const CornerInfo3D& c = kv.second;
        // After AllGather merging in the classifier, all three
        // component gtdofs should be valid (non-negative).
        MFEM_VERIFY(c.gtdof_x >= 0 && c.gtdof_y >= 0 && c.gtdof_z >= 0,
                    "ComputeCornerEssTDofs: corner '"
                        << c.label
                        << "' has invalid (negative) component gtdof");

        const std::array<int, 3> components = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};
        for (int g : components)
        {
            if (classifier.GtdofOwnerRank(g) == my_rank)
            {
                out.Append(static_cast<int>(
                    static_cast<HYPRE_BigInt>(g) - my_offset));
            }
        }
    }

    return out;
}

//==============================================================================
// Constructor
//
// All mesh / FES / configuration data is reached through the
// SimulationState; the manager itself stores no bare references to
// MFEM objects. The initializer list dereferences the shared mesh
// and FES handles (held inside SimulationState as shared_ptr) to
// satisfy the by-reference signatures of BoundaryClassifier3D and
// friends. Because m_sim_state is declared first in the header, by
// the time the classifier's initializer runs the simulation-state
// member is already valid (C++ initializes in declaration order).
//==============================================================================
MortarPbcManager::MortarPbcManager(std::shared_ptr<SimulationState> sim_state,
                                   KResidualFn k_residual,
                                   KJacobianFn k_jacobian)
    : m_sim_state(sim_state)
    // Component construction in dependency order. Each member's ctor
    // runs in declaration order (per the C++ rule), which matches the
    // dependency chain classifier → builder → C_op → saddle_solver →
    // saddle_system. SaddlePointSolver doesn't depend on the others
    // but is initialized here too for readability.
    , m_classifier(*m_sim_state->GetMesh(),
                   *m_sim_state->GetMeshParFiniteElementSpace(),
                   m_sim_state->GetOptions().mesh.snap_tol)
    , m_builder(m_classifier)
    , m_C_op(m_classifier)
    , m_saddle_solver(
          TranslateSaddleOpts(m_sim_state->GetOptions().solvers.saddle_point))
    , m_saddle_system(std::move(k_residual), std::move(k_jacobian), m_C_op)
    // State buffers — sized from the constraint operator's local row
    // count, which is set by m_C_op's constructor above.
    , m_lambda(m_C_op.Height())
    , m_g_rhs(m_C_op.Height())
    // Macroscopic state — 3×3 dense matrices, filled below.
    , m_macro_F(3, 3)
    , m_macro_Fdot(3, 3)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::ctor");

    const auto& options = m_sim_state->GetOptions();

    // Phase 5 enforces lor_depth = 1 (Phase 6 will lift this). The
    // option-parser validation already catches this when periodicity
    // is on, but we re-check here so the manager itself is robust to
    // being instantiated outside the validation path.
    MFEM_VERIFY(options.mesh.lor_depth == 1,
                "MortarPbcManager: lor_depth must be 1 in Phase 5; got "
                    << options.mesh.lor_depth
                    << ". Phase 6 will lift this restriction.");

    // Initialize macroscopic state.
    //   F̄ = I  (no deformation at simulation start)
    //   Ḟ = 0  (no deformation rate at simulation start)
    m_macro_F = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        m_macro_F(i, i) = 1.0;
    }
    m_macro_Fdot = 0.0;

    // Zero the lambda accumulator and the constraint RHS buffer.
    // Both are sized to the local lam DOF count by the initializers
    // above; we just need to zero the contents.
    m_lambda = 0.0;
    m_g_rhs  = 0.0;

    // Wire the constraint RHS buffer into the saddle system. The
    // system retains a non-owning pointer to m_g_rhs for the lifetime
    // of the manager. UpdateConstraintRHS (Phase 5.3.C) refreshes
    // the buffer's CONTENTS in place each step; the system picks up
    // the new values automatically without any further wiring.
    //
    // Installing a zero-valued g_rhs at construction time is
    // functionally identical to leaving the saddle system in its
    // homogeneous default state (r_lam = C u - 0 = C u), but it
    // simplifies the lifetime story for downstream code: the buffer
    // is always installed, never re-installed, just refreshed.
    m_saddle_system.SetConstraintRHS(m_g_rhs);

    // Build derived state. These two helpers are stubbed in 5.3.A;
    // 5.3.B fills BuildCornerEssTDofs and 5.3.C fills
    // BuildReferenceGeometricFactors. Calling them from the
    // constructor now (even as no-ops) means the public API is
    // already shaped for those batches and the call sites don't
    // need to change later.
    BuildCornerEssTDofs();
    BuildReferenceGeometricFactors();
}

//==============================================================================
// State updates — Phase 5.3.C stubs
//==============================================================================
void MortarPbcManager::UpdateMacroscopicF(const mfem::DenseMatrix& /*Lbar*/,
                                          double /*dt*/)
{
    MFEM_ABORT("MortarPbcManager::UpdateMacroscopicF: not yet implemented "
               "(Phase 5.3.C). The 5.3.A skeleton landed the class and "
               "constructor wiring; 5.3.C will fill this in.");
}

void MortarPbcManager::UpdateConstraintRHS()
{
    MFEM_ABORT("MortarPbcManager::UpdateConstraintRHS: not yet implemented "
               "(Phase 5.3.C). The 5.3.A skeleton landed the m_g_rhs "
               "buffer and wired it into the saddle system via "
               "SetConstraintRHS; 5.3.C will fill in the per-step "
               "refresh logic that uses the macroscopic F̄ and the "
               "reference geometric factors.");
}

//==============================================================================
// Diagnostics / output computation — Phase 5.3.D stubs
//==============================================================================
void MortarPbcManager::ComputeFluctuationField(
    const mfem::Vector& /*u_tdofs*/,
    mfem::ParGridFunction& /*u_fluct*/) const
{
    MFEM_ABORT("MortarPbcManager::ComputeFluctuationField: not yet "
               "implemented (Phase 5.3.D).");
}

void MortarPbcManager::ComputeHillMandelPowerBalance(
    const mfem::Vector& /*u_tdofs*/,
    double& /*cell_power*/,
    double& /*macro_power*/) const
{
    MFEM_ABORT("MortarPbcManager::ComputeHillMandelPowerBalance: not yet "
               "implemented (Phase 5.3.D).");
}

//==============================================================================
// Lambda accumulation — Phase 5.3.E stubs (ResetLambdaAccumulation
// implemented now since it's trivial)
//==============================================================================
void MortarPbcManager::AccumulateLambdaContribution(
    const mfem::Vector& /*dlam*/,
    double /*scale*/)
{
    MFEM_ABORT("MortarPbcManager::AccumulateLambdaContribution: not yet "
               "implemented (Phase 5.3.E).");
}

void MortarPbcManager::ResetLambdaAccumulation()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::reset_lambda");
    m_lambda = 0.0;
}

//==============================================================================
// Private helpers — stubs for 5.3.B and 5.3.C
//==============================================================================
void MortarPbcManager::BuildCornerEssTDofs()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::build_corner_ess_tdofs");

    // Phase 5.3.B — populate m_corner_ess_tdofs with the 8 corners'
    // (gtdof_x, gtdof_y, gtdof_z) components, filtered to only those
    // owned by this rank. The actual per-corner ownership test +
    // global→local conversion lives in ComputeCornerEssTDofs (a free
    // function in this namespace) so it can be exercised in
    // isolation by test_mortar_pbc_manager.cpp without instantiating
    // a full SimulationState.
    m_corner_ess_tdofs = ComputeCornerEssTDofs(
        m_classifier, *m_sim_state->GetMeshParFiniteElementSpace());

    // Self-check: across all ranks the corner TDOFs must total to 24
    // (8 corners × 3 components). Each rank owns a (possibly empty)
    // partition; the rank-summed count is invariant. A mismatch here
    // means the boundary classifier produced inconsistent corner
    // records across ranks, or the FES partition disagrees with the
    // classifier's GtdofOwnerRank lookup table.
    const int local_count = m_corner_ess_tdofs.Size();
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  m_classifier.Comm());
    MFEM_VERIFY(global_count == 24,
                "MortarPbcManager::BuildCornerEssTDofs: rank-summed "
                "corner TDOF count is "
                    << global_count
                    << "; expected 24 (8 corners × 3 components).");
}

void MortarPbcManager::BuildReferenceGeometricFactors()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::build_reference_geometric_factors");
    // Phase 5.3.C will fill this in. The cache holds reference
    // (undeformed) coordinates of boundary nodes that appear in
    // mortar constraint rows, so that UpdateConstraintRHS can compute
    //     g_k = F̄ · X_k
    // per row without re-walking the classifier on every step.
    //
    // Storage layout is finalized in 5.3.C — for 5.3.A this is a
    // no-op stub. The class declaration intentionally has no member
    // for the cache yet; 5.3.C will add the storage and this
    // function will populate it.
}

}  // namespace mortar_pbc