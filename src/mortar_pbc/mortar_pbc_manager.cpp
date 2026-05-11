// Phase 5.3 — MortarPbcManager implementation.
//
// See mortar_pbc_manager.hpp for design rationale and member layout.
// Cumulative across phases:
//   - 5.3.A  : constructor wiring + skeleton.
//   - 5.3.B  : ComputeCornerEssTDofs free function +
//              BuildCornerEssTDofs body.
//   - 5.3.C.0+1 : UpdateMacroscopicF mesh-anchored body. (The
//              ComputeVolumeAveragedF helper that this calls now
//              lives on the manager itself rather than on
//              SimulationState — post-processing-style calculations
//              don't belong in the state holder.)
//   - 5.3.C.2: BuildReferenceGeometricFactors + UpdateConstraintRHS
//              (RAJA::View kernel over rows).
//   - 5.3.D  : ComputeFluctuationField + ComputeHillMandelPowerBalance
//              + private ComputeVolumeAveragedCauchyStress helper.
//   - 5.3.E  : AccumulateLambdaContribution body +
//              AddCTransposeLambdaToResidual.

#include "mortar_pbc_manager.hpp"

#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// TranslateSaddleOpts — bridge between option-parser-side enums
// (SaddlePointSolverType / SaddlePointPreconditioner, defined in
// option_parser_v2.hpp) and the Phase 4.3 internal enums
// (KrylovType / SaddlePrecType, defined in saddle_point_solver.hpp).
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

    return cfg;
}

//==============================================================================
// LbarTimesXCoefficient — VectorCoefficient that returns L̄ · x at
// the integration point. Used by ComputeFluctuationField to project
// the affine velocity onto the FES.
//==============================================================================
class LbarTimesXCoefficient : public mfem::VectorCoefficient
{
public:
    explicit LbarTimesXCoefficient(const mfem::DenseMatrix& Lbar)
        : mfem::VectorCoefficient(Lbar.NumRows()), m_Lbar(Lbar)
    {
        MFEM_VERIFY(Lbar.NumRows() == Lbar.NumCols(),
                    "LbarTimesXCoefficient: Lbar must be square.");
    }

    void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
    {
        mfem::Vector x(m_Lbar.NumCols());
        T.Transform(ip, x);
        V.SetSize(m_Lbar.NumRows());
        m_Lbar.Mult(x, V);
    }

private:
    const mfem::DenseMatrix& m_Lbar;
};

}  // anonymous namespace


//==============================================================================
// ComputeCornerEssTDofs — free function exercised by both the
// manager's BuildCornerEssTDofs (which adds an MPI sanity check on
// top) and the test_mortar_pbc_manager.cpp unit test.
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
// SimulationState. The initializer list dereferences shared handles
// to satisfy the by-reference signatures of BoundaryClassifier3D
// and friends. Because m_sim_state is declared first in the header,
// by the time the classifier's initializer runs the simulation-state
// member is already valid (C++ initializes in declaration order).
//
// Vector and Array<int> members that need GPU residency tracking
// are constructed with `mfem::Device::GetMemoryType()`. mfem::Array
// has no `UseDevice(bool)` setter (only a query), so construct-time
// memory typing is the only correct pattern for the int arrays.
//==============================================================================
MortarPbcManager::MortarPbcManager(std::shared_ptr<SimulationState> sim_state,
                                   KResidualFn k_residual,
                                   KJacobianFn k_jacobian)
    : m_sim_state(sim_state)
    , m_classifier(*m_sim_state->GetMesh(),
                   *m_sim_state->GetMeshParFiniteElementSpace(),
                   m_sim_state->GetOptions().mesh.snap_tol)
    , m_builder(m_classifier)
    , m_C_op(m_classifier)
    , m_saddle_solver(
          TranslateSaddleOpts(m_sim_state->GetOptions().solvers.saddle_point))
    , m_saddle_system(std::make_shared<MortarSaddlePointSystem>(
          std::move(k_residual), std::move(k_jacobian), m_C_op))
    // State buffers — sized from the constraint operator's local
    // row count. Memory type set explicitly so device residency is
    // tracked (matters for the UpdateConstraintRHS kernel).
    , m_corner_ess_tdofs()
    , m_lambda(m_C_op.Height(), mfem::Device::GetMemoryType())
    , m_g_rhs(m_C_op.Height(), mfem::Device::GetMemoryType())
    // Macroscopic state — 3×3 dense matrices, filled below.
    , m_macro_F(3, 3)
    , m_macro_Fdot(3, 3)
    // Phase 5.7.A — per-row period-signed cache (row-major,
    // length 3 * n_rows). Sized in BuildReferenceGeometricFactors.
    , m_period_signed_per_row(0, mfem::Device::GetMemoryType())
    // Component index and ell_hat unchanged. NOTE: `m_component_per_row`
    // is `mfem::Array<int>` and constructing with
    // `Device::GetMemoryType()` does NOT translate DEVICE → HOST_64
    // the way `Vector(0, DEVICE)` does — see hotfix #1
    // (`phase_5_5_b4_hotfix_array_memtype.md`). Default-construct it.
    , m_component_per_row()
    , m_ell_hat_per_row(0, mfem::Device::GetMemoryType())
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::ctor");

    const auto& options = m_sim_state->GetOptions();

    MFEM_VERIFY(options.mesh.lor_depth == 1,
                "MortarPbcManager: lor_depth must be 1 in Phase 5; got "
                    << options.mesh.lor_depth
                    << ". Phase 6 will lift this restriction.");

    // Initialize macroscopic state.
    //   F̄ = I  (no deformation at simulation start)
    //   Ḟ = 0
    m_macro_F = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        m_macro_F(i, i) = 1.0;
    }
    m_macro_Fdot = 0.0;

    // Zero the lambda accumulator and the constraint RHS buffer.
    m_lambda = 0.0;
    m_g_rhs  = 0.0;

    // Wire the constraint RHS buffer into the saddle system.
    // UpdateConstraintRHS refreshes the buffer's CONTENTS in place
    // each step; the system picks up new values automatically.
    m_saddle_system->SetConstraintRHS(m_g_rhs);

    // Build derived state.
    BuildCornerEssTDofs();
    BuildReferenceGeometricFactors();
}

//==============================================================================
// State updates
//==============================================================================

void MortarPbcManager::UpdateMacroscopicF(const mfem::DenseMatrix& Lbar,
                                          double dt)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::update_macro_F");

    // §P5.8.6 of the v4 plan, with the mesh-anchored modification.
    // The original (P5.8.6.f) carried F̄ forward as state,
    // F̄^{n+1} = F̄^{n}_tracked + L̄·F̄^{n}_tracked·dt, which compounded
    // (a) per-step Newton residual leftover and (b) FE-time-
    // integration truncation across hundreds of load steps. The
    // corrected anchor uses the volume-averaged F from the mesh
    // itself:
    //
    //     F̄^{(n)}_mesh = (1/V) ∫ F dV
    //
    // which by Hill-Mandel is the true F̄ for a converged periodic
    // RVE — drift-free, regardless of how many steps have run.

    // Volume-averaged F as Voigt 9-vector, row-major
    // [F11, F12, F13, F21, F22, F23, F31, F32, F33].
    mfem::Vector F_voigt9(9, mfem::Device::GetMemoryType());
    const double V_unused = ComputeVolumeAveragedF(F_voigt9);
    (void)V_unused;  // Volume not needed here; we just want F̄_mesh.

    mfem::DenseMatrix F_bar_mesh(3, 3);
    {
        const double* d = F_voigt9.HostRead();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                F_bar_mesh(i, j) = d[i * 3 + j];
            }
        }
    }

    // First-step protection: if "kinetic_grads" hasn't been touched
    // by an integrator pass yet, the volume average is meaningless.
    // Detect by determinant and fall back to F̄^{(0)} = I.
    if (F_bar_mesh.Det() < 0.5)
    {
        F_bar_mesh = 0.0;
        for (int i = 0; i < 3; ++i) { F_bar_mesh(i, i) = 1.0; }
    }

    // Ḟ̄^{(n+1)} = L̄^{(n+1)} · F̄^{(n)}_mesh — the rate that goes
    // into the constraint RHS via §P5.8.6.d. Anchored on F̄^{(n)}_mesh
    // (NOT F̄^{(n+1)}) here on purpose: using F̄^{(n+1)} would smuggle
    // a second-order L̄²·dt term into Ḟ̄.
    mfem::Mult(Lbar, F_bar_mesh, m_macro_Fdot);

    // F̄^{(n+1)} = F̄^{(n)}_mesh + Ḟ̄·dt = (I + L̄·dt) · F̄^{(n)}_mesh.
    m_macro_F = m_macro_Fdot;
    m_macro_F *= dt;
    m_macro_F += F_bar_mesh;
}

void MortarPbcManager::UpdateConstraintRHS()
{
    // Phase 5.7.A — generalized §P5.8.6.d:
    //   g_i = ℓ̂_i · Σ_k Ḟ̄_{c, k} · period_signed_per_row[3i + k]
    // where
    //   c             = component_per_row[i]
    //   ℓ̂_i           = ell_hat_per_row[i]
    //   period_signed = full physical periodic shift vector for row i
    //                   (face rows: one nonzero entry; edge rows: one
    //                    or two nonzero transverse-axis entries).
    //
    // The previous formula `Ḟ̄_{c, k} · L_k · ℓ̂` used a single axis
    // index `k = axis_per_row[i]`; that worked only for faces because
    // for edges `axis_per_row` was the edge-parallel axis (not the
    // jump axis). period_signed_per_row resolves both cases uniformly.
    //
    // Per row this is now three multiply-adds rather than two
    // multiplies. Once-per-step (NOT per Newton iteration); the
    // saddle Newton iterates against this fixed RHS until convergence
    // per §P5.8.6 "off-equilibrium considerations."

    const int n_rows = m_component_per_row.Size();
    MFEM_VERIFY(m_g_rhs.Size() == n_rows,
                "MortarPbcManager::UpdateConstraintRHS: m_g_rhs size "
                << m_g_rhs.Size() << " != n_rows " << n_rows);
    MFEM_VERIFY(m_period_signed_per_row.Size() == 3 * n_rows,
                "MortarPbcManager::UpdateConstraintRHS: "
                "m_period_signed_per_row size "
                << m_period_signed_per_row.Size()
                << " != 3 * n_rows = " << 3 * n_rows);

    // Copy m_macro_Fdot (host DenseMatrix) into a device-resident
    // Vector(9), row-major. 9 doubles per step.
    mfem::Vector Fdot_vec(9, mfem::Device::GetMemoryType());
    {
        double* d = Fdot_vec.HostWrite();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                d[i * 3 + j] = m_macro_Fdot(i, j);
            }
        }
    }

    // Read-only device pointers.
    const double* Fdot_data   = Fdot_vec.Read();
    const int*    comp_data   = m_component_per_row.Read();
    const double* ell_data    = m_ell_hat_per_row.Read();
    const double* period_data = m_period_signed_per_row.Read();
    double*       g_data      = m_g_rhs.Write();

    // RAJA::View — row-major default, gives typed 2-D access inside
    // the device lambda. Fdot_view(c, k) = Fdot_data[c*3 + k]
    // = Ḟ̄_{c, k}.
    RAJA::View<const double, RAJA::Layout<2>> Fdot_view(Fdot_data, 3, 3);

    mfem::forall(n_rows, [=] MFEM_HOST_DEVICE (int i)
    {
        const int c = comp_data[i];
        // Dot product Σ_k Ḟ̄(c, k) · period_signed[3i + k]; unrolled
        // for clarity at three terms.
        const double dot = Fdot_view(c, 0) * period_data[3 * i + 0]
                         + Fdot_view(c, 1) * period_data[3 * i + 1]
                         + Fdot_view(c, 2) * period_data[3 * i + 2];
        g_data[i] = ell_data[i] * dot;
    });
}

//==============================================================================
// Diagnostics / output computation
//==============================================================================

void MortarPbcManager::ComputeFluctuationField(
    const mfem::Vector& velocity_tdofs,
    const mfem::DenseMatrix& Lbar,
    mfem::ParGridFunction& fluct_gf) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::compute_fluctuation_field");

    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    MFEM_VERIFY(velocity_tdofs.Size() == fes->GetTrueVSize(),
                "ComputeFluctuationField: velocity_tdofs size "
                << velocity_tdofs.Size() << " != fes TrueVSize "
                << fes->GetTrueVSize());

    // Project L̄·x onto the FES via VectorCoefficient.
    LbarTimesXCoefficient affine_coeff(Lbar);
    fluct_gf.SetSpace(fes.get());
    fluct_gf.ProjectCoefficient(affine_coeff);

    // Pull affine into TDOF space, subtract from velocity, push back
    // to grid-function space as the fluctuation.
    mfem::Vector affine_tdofs(fes->GetTrueVSize(),
                              mfem::Device::GetMemoryType());
    fluct_gf.ParallelProject(affine_tdofs);

    mfem::Vector tilde_v(fes->GetTrueVSize(),
                         mfem::Device::GetMemoryType());
    tilde_v = velocity_tdofs;  // deep copy
    tilde_v -= affine_tdofs;

    fluct_gf.SetFromTrueDofs(tilde_v);
}

MortarPbcManager::HillMandelDiagnostic
MortarPbcManager::ComputeHillMandelPowerBalance(
    const mfem::Vector& velocity_tdofs,
    const mfem::Vector& internal_force_tdofs,
    const mfem::DenseMatrix& Lbar) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::compute_hill_mandel");

    HillMandelDiagnostic out;

    // --- Macro side ---
    // σ̄ AND total volume in one sweep.
    mfem::Vector sigma_voigt(6, mfem::Device::GetMemoryType());
    out.total_volume = ComputeVolumeAveragedCauchyStress(sigma_voigt);

    // Voigt → 3×3.
    {
        const double* s = sigma_voigt.HostRead();
        // Voigt order: [σxx, σyy, σzz, σxy, σxz, σyz].
        out.sigma_bar(0, 0) = s[0];
        out.sigma_bar(1, 1) = s[1];
        out.sigma_bar(2, 2) = s[2];
        out.sigma_bar(0, 1) = out.sigma_bar(1, 0) = s[3];
        out.sigma_bar(0, 2) = out.sigma_bar(2, 0) = s[4];
        out.sigma_bar(1, 2) = out.sigma_bar(2, 1) = s[5];
    }

    // d̄ = (L̄ + L̄^T) / 2.
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            out.d_bar(i, j) = 0.5 * (Lbar(i, j) + Lbar(j, i));
        }
    }

    // σ̄:d̄ = sum_{i, j} σ̄_{ij} · d̄_{ij}.
    out.macro_power = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            out.macro_power += out.sigma_bar(i, j) * out.d_bar(i, j);
        }
    }

    // --- LHS: integrated local power v · r_internal ---
    // v_a · ∫B_a^Tσ dV = ∫σ:∇v dV = ∫σ:d dV (σ symmetric).
    {
        auto fes = m_sim_state->GetMeshParFiniteElementSpace();
        const double local_dot = velocity_tdofs * internal_force_tdofs;
        double global_dot = 0.0;
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM,
                      fes->GetComm());
        out.integrated_internal_power = global_dot;
    }

    // --- Residuals ---
    const double macro_integrated = out.macro_power * out.total_volume;
    out.abs_residual = std::abs(out.integrated_internal_power
                                - macro_integrated);
    const double denom = std::max(std::abs(macro_integrated), 1e-300);
    out.rel_residual = out.abs_residual / denom;

    return out;
}

//==============================================================================
// DiagnoseConstraintConsistency — Phase 5.7.A
//
// Project v_aff(x) = L̄·x onto the FES, apply C, compare against g.
// See header for what the four norms mean and how to read them.
//==============================================================================
MortarPbcManager::ConstraintConsistencyDiagnostic
MortarPbcManager::DiagnoseConstraintConsistency(
    const mfem::DenseMatrix& Lbar) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::diagnose_constraint_consistency");

    auto fes = m_sim_state->GetMeshParFiniteElementSpace();

    // 1. Build v_aff(x) = L̄·x as a ParGridFunction via the existing
    //    LbarTimesXCoefficient (defined in the anonymous namespace at
    //    the top of this file).
    LbarTimesXCoefficient affine_coeff(Lbar);
    mfem::ParGridFunction v_aff_gf(fes.get());
    v_aff_gf.ProjectCoefficient(affine_coeff);

    // 2. Pull to TDOFs.
    mfem::Vector v_aff_tdofs(fes->GetTrueVSize(),
                             mfem::Device::GetMemoryType());
    v_aff_gf.ParallelProject(v_aff_tdofs);

    // 3. Apply constraint: Cv = C * v_aff.
    mfem::Vector Cv(m_C_op.Height(), mfem::Device::GetMemoryType());
    m_C_op.Mult(v_aff_tdofs, Cv);

    // 4. diff = Cv - g, sum = Cv + g.
    mfem::Vector diff(Cv);
    diff -= m_g_rhs;
    mfem::Vector sum(Cv);
    sum += m_g_rhs;

    // 5. Local infinity norms.
    const double local_cv_inf   = Cv.Normlinf();
    const double local_g_inf    = m_g_rhs.Normlinf();
    const double local_diff_inf = diff.Normlinf();
    const double local_sum_inf  = sum.Normlinf();

    // 6. Global reductions over the FES communicator.
    ConstraintConsistencyDiagnostic out;
    MPI_Allreduce(&local_cv_inf,   &out.cv_norm_inf,   1, MPI_DOUBLE, MPI_MAX,
                  fes->GetComm());
    MPI_Allreduce(&local_g_inf,    &out.g_norm_inf,    1, MPI_DOUBLE, MPI_MAX,
                  fes->GetComm());
    MPI_Allreduce(&local_diff_inf, &out.diff_norm_inf, 1, MPI_DOUBLE, MPI_MAX,
                  fes->GetComm());
    MPI_Allreduce(&local_sum_inf,  &out.sum_norm_inf,  1, MPI_DOUBLE, MPI_MAX,
                  fes->GetComm());

// ====================================================================
    // Phase 5.7.A extended — argmax row info on this rank.
    //
    // The previous round showed all four norms equal to 0.0025,
    // indicating disjoint supports for C·v_aff vs g. Print the
    // metadata (axis, comp, ell) at each vector's argmax to pin
    // down the indexing-convention mismatch.
    // ====================================================================
    {
        // Host-side reads for the diagnostic — Cv and m_g_rhs already
        // host-resident from the operations above.
        const double* cv_data = Cv.HostRead();
        const double* g_data  = m_g_rhs.HostRead();
        const int     n_rows  = Cv.Size();
        MFEM_ASSERT(m_g_rhs.Size() == n_rows,
                      "DiagnoseConstraintConsistency: g size mismatch.");

        // Rank-local argmax of |g|.
        out.argmax_g_row = -1;
        double max_abs_g = -1.0;
        for (int i = 0; i < n_rows; ++i) {
            const double a = std::abs(g_data[i]);
            if (a > max_abs_g) {
                max_abs_g = a;
                out.argmax_g_row = i;
            }
        }
        if (out.argmax_g_row >= 0) {
            const int r = out.argmax_g_row;
            const int*    comp_h   = m_component_per_row.HostRead();
            const double* ell_h    = m_ell_hat_per_row.HostRead();
            const double* period_h = m_period_signed_per_row.HostRead();
            out.argmax_g_period[0] = period_h[3 * r + 0];
            out.argmax_g_period[1] = period_h[3 * r + 1];
            out.argmax_g_period[2] = period_h[3 * r + 2];
            out.argmax_g_comp      = comp_h[r];
            out.argmax_g_ell       = ell_h[r];
            out.argmax_g_g_val  = g_data[r];
            out.argmax_g_cv_val = cv_data[r];
        }

        // Rank-local argmax of |C·v_aff|.
        out.argmax_cv_row = -1;
        double max_abs_cv = -1.0;
        for (int i = 0; i < n_rows; ++i) {
            const double a = std::abs(cv_data[i]);
            if (a > max_abs_cv) {
                max_abs_cv = a;
                out.argmax_cv_row = i;
            }
        }
        if (out.argmax_cv_row >= 0) {
            const int r = out.argmax_cv_row;
            const int* comp_h = m_component_per_row.HostRead();
            const double* ell_h = m_ell_hat_per_row.HostRead();
            out.argmax_cv_comp   = comp_h[r];
            out.argmax_cv_ell    = ell_h[r];
            out.argmax_cv_g_val  = g_data[r];
            out.argmax_cv_cv_val = cv_data[r];
        }

        // Phase 5.7.A — argmax of |C·v_aff - g|. The `diff` vector
        // was already computed above for `||diff||_inf`; reuse it.
        out.argmax_diff_row = -1;
        double max_abs_diff = -1.0;
        const double* diff_data = diff.HostRead();
        for (int i = 0; i < n_rows; ++i)
        {
            const double a = std::abs(diff_data[i]);
            if (a > max_abs_diff)
            {
                max_abs_diff = a;
                out.argmax_diff_row = i;
            }
        }
        if (out.argmax_diff_row >= 0)
        {
            const int r = out.argmax_diff_row;
            const int* comp_h = m_component_per_row.HostRead();
            const double* ell_h = m_ell_hat_per_row.HostRead();
            const double* period_h = m_period_signed_per_row.HostRead();
            out.argmax_diff_period[0] = period_h[3 * r + 0];
            out.argmax_diff_period[1] = period_h[3 * r + 1];
            out.argmax_diff_period[2] = period_h[3 * r + 2];
            out.argmax_diff_comp   = comp_h[r];
            out.argmax_diff_ell    = ell_h[r];
            out.argmax_diff_g_val  = g_data[r];
            out.argmax_diff_cv_val = cv_data[r];
            out.argmax_diff_val    = diff_data[r];
        }
    }
    return out;
}

//==============================================================================
// Lambda accumulation
//==============================================================================

void MortarPbcManager::AccumulateLambdaContribution(
    const mfem::Vector& dlam,
    double scale)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::accumulate_lambda");
    MFEM_VERIFY(dlam.Size() == m_lambda.Size(),
                "AccumulateLambdaContribution: dlam size "
                << dlam.Size() << " != m_lambda size "
                << m_lambda.Size());
    m_lambda.Add(scale, dlam);
}

void MortarPbcManager::SetAccumulatedLambda(const mfem::Vector& lambda)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::set_lambda");
    MFEM_VERIFY(lambda.Size() == m_lambda.Size(),
                "SetAccumulatedLambda: lambda size "
                << lambda.Size() << " != m_lambda size "
                << m_lambda.Size());
    m_lambda = lambda;  // deep copy
}

void MortarPbcManager::ResetLambdaAccumulation()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::reset_lambda");
    m_lambda = 0.0;
}

void MortarPbcManager::AddCTransposeLambdaToResidual(
    mfem::Vector& residual) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::manager::add_c_transpose_lambda_to_residual");

    MFEM_VERIFY(residual.Size() == m_C_op.Width(),
                "AddCTransposeLambdaToResidual: residual size "
                << residual.Size() << " != C^T height (= C width = "
                << m_C_op.Width() << ")");

    mfem::Vector tmp(m_C_op.Width(), mfem::Device::GetMemoryType());
    tmp = 0.0;
    m_C_op.MultTranspose(m_lambda, tmp);
    residual += tmp;
}

//==============================================================================
// Private helpers
//==============================================================================

void MortarPbcManager::BuildCornerEssTDofs()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::build_corner_ess_tdofs");

    // Phase 5.3.B — populate m_corner_ess_tdofs with the 8 corners'
    // (gtdof_x, gtdof_y, gtdof_z) components, filtered to those owned
    // by this rank. Per-corner ownership test + global→local
    // conversion is in the ComputeCornerEssTDofs free function so it
    // can be exercised in isolation by test_mortar_pbc_manager.cpp.
    m_corner_ess_tdofs = ComputeCornerEssTDofs(
        m_classifier, *m_sim_state->GetMeshParFiniteElementSpace());

    // Self-check: across all ranks the corner TDOFs must total to 24.
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
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::manager::build_reference_geometric_factors");

    // Phase 5.7.A — per-row metadata now includes the full periodic
    // shift VECTOR per row (not just an axis index + global box
    // lengths). `EmitRowFactors` mirrors the row-emission pattern of
    // `EmitConstraintTriples`, so emit position k is the same row
    // index k that the constraint matrix uses. `period_signed_per_row`
    // is sized to `3 * n_local_rows` row-major; `component_per_row`
    // and `ell_hat_per_row` are sized to `n_local_rows`.
    m_builder.EmitRowFactors(m_period_signed_per_row,
                             m_component_per_row,
                             m_ell_hat_per_row);

    // The previous Cache-2 (m_axis_lengths from bbox) is gone — the
    // L_k factors are already baked into period_signed_per_row by
    // the builder (`nonmortar.plane_value - mortar.plane_value` for
    // faces; `nonmortar.coords(0, k) - mortar.coords(0, k)` for
    // edges' transverse axes). This eliminates a duplicate source of
    // truth for box lengths.

    // Sanity check: m_g_rhs (wired to the saddle system) must match
    // the local row count.
    const int n_rows = m_component_per_row.Size();
    MFEM_VERIFY(m_g_rhs.Size() == n_rows,
                "MortarPbcManager::BuildReferenceGeometricFactors: "
                "m_g_rhs size " << m_g_rhs.Size()
                << " != per-row metadata count " << n_rows
                << ". Saddle-system RHS partition disagrees with the "
                "constraint builder's NumLocalRows().");
    MFEM_VERIFY(m_period_signed_per_row.Size() == 3 * n_rows,
                "MortarPbcManager::BuildReferenceGeometricFactors: "
                "m_period_signed_per_row size "
                << m_period_signed_per_row.Size()
                << " != 3 * n_rows = " << 3 * n_rows
                << ". EmitRowFactors output is malformed.");
}

double MortarPbcManager::ComputeVolumeAveragedF(
    mfem::Vector& F_voigt9) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::compute_volume_averaged_F");

    constexpr int kSize = 9;
    if (F_voigt9.Size() != kSize)
    {
        F_voigt9.SetSize(kSize, mfem::Device::GetMemoryType());
    }
    F_voigt9 = 0.0;

    auto qf = m_sim_state->GetQuadratureFunction("kinetic_grads");
    MFEM_VERIFY(qf,
                "ComputeVolumeAveragedF: global \"kinetic_grads\" "
                "QuadratureFunction not found.");

    // The QFs in SimulationState are PartialQuadratureFunctions; the
    // global one returned by GetQuadratureFunction(name) covers the
    // whole mesh, so MPI_COMM_WORLD is the right reduction comm.
    auto& rt_model =
        const_cast<RTModel&>(m_sim_state->GetOptions().solvers.rtmodel);
    return exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        qf.get(), F_voigt9, kSize, rt_model, MPI_COMM_WORLD);
}

double MortarPbcManager::ComputeVolumeAveragedCauchyStress(
    mfem::Vector& sigma_voigt) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::manager::compute_volume_averaged_cauchy_stress");

    constexpr int kSize = 6;
    if (sigma_voigt.Size() != kSize)
    {
        sigma_voigt.SetSize(kSize, mfem::Device::GetMemoryType());
    }
    sigma_voigt = 0.0;

    auto qf = m_sim_state->GetQuadratureFunction("cauchy_stress_end");
    MFEM_VERIFY(qf,
                "ComputeVolumeAveragedCauchyStress: global "
                "\"cauchy_stress_end\" QuadratureFunction not found.");

    auto& rt_model =
        const_cast<RTModel&>(m_sim_state->GetOptions().solvers.rtmodel);
    return exaconstit::kernel::ComputeVolAvgTensorFromPartial<true>(
        qf.get(), sigma_voigt, kSize, rt_model, MPI_COMM_WORLD);
}

}  // namespace mortar_pbc