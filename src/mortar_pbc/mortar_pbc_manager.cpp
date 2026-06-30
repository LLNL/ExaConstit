// Phase 5.3 / Phase 6 — MortarPbcManager implementation.
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
//   - 6.0.G  : manager construction migrated to boundary/LOR
//              classifier + SurfaceProjector + projector-aware
//              builder/operator. Corner pinning remains applied in
//              parent-volume true-DOF numbering.

#include "mortar_pbc_manager.hpp"

#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
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
// TranslateSaddleScalingOptions — Phase 5.11.E.
//
// Bridges the option-parser-side `::SaddleScalingOptions` (nullable
// — absent if the user's TOML has no `[Solvers.SaddlePoint.Scaling]`
// table) to the mortar_pbc-internal `SaddleResidualScalerConfig`.
// Mirrors the layering of `TranslateSaddleOpts` above: the .hpp
// stays free of `option_parser_v2.hpp`; only the .cpp pulls the
// option-parser side in.
//
// When the options-side payload is `std::nullopt`, returns a
// default-constructed config (`enabled = false` etc.) so the
// downstream scaler exists but is inert — preserving pre-5.11
// behavior bit-for-bit.
//==============================================================================
SaddleResidualScalerConfig TranslateSaddleScalingOptions(
    const std::optional<SaddleScalingOptions>& opts)
{
    SaddleResidualScalerConfig cfg;

    if (!opts.has_value())
    {
        // No [Solvers.SaddlePoint.Scaling] in TOML → scaling
        // disabled, scaler is constructed but inert.
        return cfg;
    }

    cfg.enabled      = opts->enabled;
    cfg.per_subblock = opts->per_subblock;
    cfg.floor        = opts->floor;
    cfg.range_cap    = opts->range_cap;

    switch (opts->partition)
    {
        case ::SubblockPartition::FACE_EDGE:
            cfg.partition = mortar_pbc::SubblockPartition::FaceEdge;
            break;
        case ::SubblockPartition::PER_PAIR:
            cfg.partition = mortar_pbc::SubblockPartition::PerPair;
            break;
        case ::SubblockPartition::NOTYPE:
        default:
            MFEM_ABORT("MortarPbcManager: SaddleScalingOptions.partition "
                       "has invalid value " << static_cast<int>(opts->partition)
                       << ". Did ExaOptions::validate() pass?");
    }

    return cfg;
}

//==============================================================================
// Phase 5.9 / Batch A.4 — spec-interpretation helpers.
//
// Three small helpers used by RebuildForActiveSpec and the
// ComputeCornerEssTDofsFromSpec free function. Kept anonymous-ns
// local because they're TU-specific glue between the option-parser
// representation (essential_ids vector + essential_comps int) and
// the classifier/operator API (vector<string> + array<bool,3>).
//==============================================================================

/// Anchor corner label. Convention documented in
/// boundary_helpers_3d.hpp: "blf" = bottom-left-front, the corner at
/// (min_x, min_y, min_z) of the box. This corner's 3 components are
/// always pinned to remove translation rigid-body modes regardless
/// of the active spec's component mask.
constexpr const char* kAnchorCornerLabel = "blf";

/// Translate `essential_comps` (1..7 from BCData::GetComponents
/// convention) into a per-component boolean mask.
///   1 = X-only       → {T, F, F}
///   2 = Y-only       → {F, T, F}
///   3 = Z-only       → {F, F, T}
///   4 = XY           → {T, T, F}
///   5 = XZ           → {T, F, T}
///   6 = YZ           → {F, T, T}
///   7 = XYZ          → {T, T, T}
/// Aborts via MFEM_ABORT on out-of-range values.
std::array<bool, 3> CompMaskFromInt(int essential_comps)
{
    switch (essential_comps)
    {
        case 1: return {{true,  false, false}};
        case 2: return {{false, true,  false}};
        case 3: return {{false, false, true }};
        case 4: return {{true,  true,  false}};
        case 5: return {{true,  false, true }};
        case 6: return {{false, true,  true }};
        case 7: return {{true,  true,  true }};
        default:
            MFEM_ABORT("MortarPbcManager: invalid essential_comps="
                       << essential_comps
                       << "; expected 1..7 (BCData::GetComponents "
                          "convention: 1=X, 2=Y, 3=Z, 4=XY, 5=XZ, "
                          "6=YZ, 7=XYZ).");
    }
    return {{false, false, false}};  // unreachable; suppress warning
}

/// Validate pair-completeness AND derive the canonical
/// `active_pair_labels` list (mortar-side labels only).
///
/// For every attr in `essential_ids`:
///   - confirm it's a valid boundary face attribute,
///   - confirm its pair partner attribute is also in `essential_ids`.
///
/// On failure, aborts with a message naming the missing partner attr
/// and label. On success, returns a deduplicated vector of mortar-
/// side labels for the active pairs.
///
/// Walks `classifier.FacePairs()` (3 entries on a standard
/// axis-aligned RVE) to derive labels rather than iterating
/// `essential_ids` twice — fewer label↔attr round-trips.
std::vector<std::string> ValidateAndDeriveActivePairLabels(
    const BoundaryClassifier3D& classifier,
    const std::vector<int>& essential_ids)
{
    // Set for O(1) attr membership tests.
    const std::set<int> attrs_set(essential_ids.begin(),
                                  essential_ids.end());

    // First pass: validate that every attr is (a) a boundary face attr
    // and (b) has its partner present.
    for (int attr : essential_ids)
    {
        MFEM_VERIFY(classifier.IsBoundaryFaceAttribute(attr),
                    "MortarPbcManager::RebuildForActiveSpec: "
                    "essential_ids contains attribute " << attr
                    << " which is not a recognized boundary face "
                    "attribute in the classifier. Did the mesh and "
                    "TOML face attributes get out of sync?");

        const std::string label = classifier.LabelForMeshAttribute(attr);
        const std::string partner_label = classifier.PairPartnerLabel(label);
        MFEM_VERIFY(!partner_label.empty(),
                    "MortarPbcManager::RebuildForActiveSpec: face "
                    "attribute " << attr << " (label '" << label
                    << "') has no pair partner. essential_ids must "
                    "only contain attributes belonging to face pairs.");

        const int partner_attr =
            classifier.MeshAttributeForLabel(partner_label);
        MFEM_VERIFY(attrs_set.find(partner_attr) != attrs_set.end(),
                    "MortarPbcManager::RebuildForActiveSpec: periodic "
                    "BC entry references face attribute " << attr
                    << " (label '" << label
                    << "') but its required pair partner attribute "
                    << partner_attr << " (label '" << partner_label
                    << "') is missing from essential_ids. Both halves "
                    "of every pair must be listed.");
    }

    // Second pass: collect canonical mortar-side labels for active
    // pairs. A pair is active iff one half is in attrs_set; the
    // first pass guaranteed both halves are then present.
    std::set<std::string> mortar_labels_set;
    for (const auto& tup : classifier.FacePairs())
    {
        const std::string& mortar_label    = std::get<1>(tup);
        const int mortar_attr =
            classifier.MeshAttributeForLabel(mortar_label);
        if (attrs_set.find(mortar_attr) != attrs_set.end())
        {
            mortar_labels_set.insert(mortar_label);
        }
    }

    return std::vector<std::string>(mortar_labels_set.begin(),
                                    mortar_labels_set.end());
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

/// Append `submesh_gtdof` to `out` as a rank-local parent-FES TDOF
/// when this rank owns the mapped parent true DOF. The classifier
/// emits corner records in the boundary/LOR-submesh index space; the
/// mechanics Dirichlet list must be expressed in parent-volume TDOFs.
void AppendProjectedCornerComponent(
    int submesh_gtdof,
    const SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes,
    int my_rank,
    mfem::Array<int>& out)
{
    MFEM_VERIFY(submesh_gtdof >= 0,
                "AppendProjectedCornerComponent: invalid negative "
                "submesh gtdof " << submesh_gtdof);

    const int parent_gtdof = projector.ParentGtdof(submesh_gtdof);
    if (projector.ParentOwnerRank(parent_gtdof) == my_rank)
    {
        out.Append(static_cast<int>(
            static_cast<HYPRE_BigInt>(parent_gtdof)
            - parent_fes.GetMyTDofOffset()));
    }
}

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
// ComputeCornerEssTDofs — Phase 6 projector-aware overload.
//
// Same corner-walk as the legacy path, but the classifier lives on a
// boundary/LOR submesh. Each selected submesh global TDOF is projected
// back to the parent-volume FE space before ownership and local-index
// conversion. This keeps the manager's essential-BC list in the same
// TDOF space as the mechanics solve.
//==============================================================================
mfem::Array<int> ComputeCornerEssTDofs(
    const BoundaryClassifier3D& classifier,
    const SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes)
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::compute_corner_ess_tdofs_projected");

    int my_rank = -1;
    MPI_Comm_rank(parent_fes.GetComm(), &my_rank);

    mfem::Array<int> out;
    out.Reserve(24);  // Upper bound: 8 corners × 3 components.

    for (const auto& kv : classifier.Corners())
    {
        const CornerInfo3D& c = kv.second;
        MFEM_VERIFY(c.gtdof_x >= 0 && c.gtdof_y >= 0 && c.gtdof_z >= 0,
                    "ComputeCornerEssTDofs(projected): corner '"
                        << c.label
                        << "' has invalid (negative) component gtdof");

        const std::array<int, 3> components = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};
        for (int g : components)
        {
            AppendProjectedCornerComponent(
                g, projector, parent_fes, my_rank, out);
        }
    }

    return out;
}

//==============================================================================
// ComputeCornerEssTDofsFromSpec — Phase 5.9 / Batch A.4 (tightened in A.5)
//
// Spec-aware variant of ComputeCornerEssTDofs:
//   - Anchor "blf" corner: pinned in all 3 components unconditionally.
//   - 7 non-anchor corners: gated by incident-face check
//     (CornersOnFaceAttribute over essential_ids) AND filtered by
//     comp_mask.
//
// On a standard axis-aligned 6-face RVE the incident-face gate is
// vacuous (every corner is incident on three of the six box faces;
// any essential_ids covering at least one complete pair → all 8
// corners eligible). The gate is still implemented explicitly to
// match the spec docstring on PeriodicBC and to give correct
// behavior on non-RVE geometries.
//==============================================================================
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D& classifier,
    const mfem::ParFiniteElementSpace& fes,
    const std::vector<int>& essential_ids,
    const std::array<bool, 3>& comp_mask)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::compute_corner_ess_tdofs_from_spec");

    const int my_rank = classifier.Rank();
    const HYPRE_BigInt my_offset = fes.GetMyTDofOffset();

    // Step 1: anchor corner — all 3 components pinned unconditionally.
    //
    // Phase 5.9.A.2's `AnchorCornerTDofs(fes)` returns rank-local
    // TDOFs of the "blf" corner's 3 components, applying the same
    // GtdofOwnerRank / GetMyTDofOffset conversion the legacy
    // ComputeCornerEssTDofs path uses.
    mfem::Array<int> out = classifier.AnchorCornerTDofs(fes);

    // Step 2: build the set of corner labels incident on any face
    // attribute listed in essential_ids. `CornersOnFaceAttribute`
    // (Phase 5.9.A.2) returns the 4 corner labels touching the given
    // face. For a standard 6-face RVE: 4 face attrs in essential_ids
    // covers all 8 corners (incident-face gate is vacuous). A
    // single-pair entry like {left, right} also covers all 8 corners
    // because every corner is at min_x or max_x.
    std::set<std::string> incident_labels;
    for (int attr : essential_ids)
    {
        const std::vector<std::string> labels_on_face =
            classifier.CornersOnFaceAttribute(attr);
        incident_labels.insert(labels_on_face.begin(),
                               labels_on_face.end());
    }

    // Step 3: 7 non-anchor corners — pinned per the incident-face
    // gate AND per comp_mask.
    for (const auto& kv : classifier.Corners())
    {
        const CornerInfo3D& c = kv.second;
        if (c.label == kAnchorCornerLabel) { continue; }  // anchor handled

        // Incident-face gate.
        if (incident_labels.find(c.label) == incident_labels.end())
        {
            continue;
        }

        MFEM_VERIFY(c.gtdof_x >= 0 && c.gtdof_y >= 0 && c.gtdof_z >= 0,
                    "ComputeCornerEssTDofsFromSpec: corner '"
                        << c.label
                        << "' has invalid (negative) component gtdof");

        const std::array<int, 3> components = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};
        for (int comp = 0; comp < 3; ++comp)
        {
            if (!comp_mask[comp]) { continue; }
            const int g = components[comp];
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
// ComputeCornerEssTDofsFromSpec — Phase 6 projector-aware overload.
//
// Preserves the Phase 5.9 pinning semantics while changing the output
// index space from classifier/submesh TDOFs to parent-volume TDOFs:
// anchor "blf" is always fully pinned, and non-anchor corners are
// incident-face gated plus component filtered.
//==============================================================================
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D& classifier,
    const SurfaceProjector& projector,
    const mfem::ParFiniteElementSpace& parent_fes,
    const std::vector<int>& essential_ids,
    const std::array<bool, 3>& comp_mask)
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::compute_corner_ess_tdofs_from_spec_projected");

    int my_rank = -1;
    MPI_Comm_rank(parent_fes.GetComm(), &my_rank);

    mfem::Array<int> out;
    out.Reserve(24);

    // Step 1: anchor corner — all 3 components pinned
    // unconditionally. We cannot use classifier.AnchorCornerTDofs()
    // here because that helper converts in the classifier/submesh FES
    // index space; the mechanics essential list needs parent-FES
    // local true DOFs.
    const auto anchor_it = classifier.Corners().find(kAnchorCornerLabel);
    MFEM_VERIFY(anchor_it != classifier.Corners().end(),
                "ComputeCornerEssTDofsFromSpec(projected): anchor corner '"
                << kAnchorCornerLabel << "' was not found.");
    {
        const CornerInfo3D& c = anchor_it->second;
        const std::array<int, 3> components = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};
        for (int g : components)
        {
            AppendProjectedCornerComponent(
                g, projector, parent_fes, my_rank, out);
        }
    }

    // Step 2: identify non-anchor corners incident on any active face
    // attribute, matching the legacy filtered path exactly.
    std::set<std::string> incident_labels;
    for (int attr : essential_ids)
    {
        const std::vector<std::string> labels_on_face =
            classifier.CornersOnFaceAttribute(attr);
        incident_labels.insert(labels_on_face.begin(),
                               labels_on_face.end());
    }

    // Step 3: emit selected non-anchor components in parent-FES local
    // numbering.
    for (const auto& kv : classifier.Corners())
    {
        const CornerInfo3D& c = kv.second;
        if (c.label == kAnchorCornerLabel) { continue; }
        if (incident_labels.find(c.label) == incident_labels.end())
        {
            continue;
        }

        MFEM_VERIFY(c.gtdof_x >= 0 && c.gtdof_y >= 0 && c.gtdof_z >= 0,
                    "ComputeCornerEssTDofsFromSpec(projected): corner '"
                        << c.label
                        << "' has invalid (negative) component gtdof");

        const std::array<int, 3> components = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};
        for (int comp = 0; comp < 3; ++comp)
        {
            if (!comp_mask[comp]) { continue; }
            AppendProjectedCornerComponent(
                components[comp], projector, parent_fes, my_rank, out);
        }
    }

    return out;
}


//==============================================================================
// Constructor
//
// All mesh / FES / configuration data is reached through the
// SimulationState. Phase 6 cannot initialize the classifier, builder,
// operator, and saddle system directly in the initializer list because
// they share ownership of the LOR boundary surface and projector. The
// initializer list therefore constructs only dependency-free members;
// the constructor body then wires the LOR classifier, SurfaceProjector,
// projector-aware builder/operator, saddle system, and row-sized
// vectors in that order.
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
    , m_saddle_solver(
          TranslateSaddleOpts(m_sim_state->GetOptions().solvers.saddle_point))
    , m_saddle_system()
    // Phase 5.11.E — scaling state. The shared_ptrs are default-
    // constructed here (nullptr) and assigned in the body once the
    // C-op's default-filter state is fully populated; the block-
    // offsets array is sized to 3 with zeros and filled in the body
    // (the saddle system's n_u + n_lam may not be queried-ready until
    // its ctor has finished).
    , m_scaler()
    , m_scaled_saddle_system()
    , m_saddle_block_offsets(3)
    // State buffers — sized from the constraint operator's local
    // row count. Memory type set explicitly so device residency is
    // tracked (matters for the UpdateConstraintRHS kernel).
    , m_corner_ess_tdofs()
    , m_lambda(0, mfem::Device::GetMemoryType())
    , m_g_rhs(0, mfem::Device::GetMemoryType())
    // Macroscopic state — 3×3 dense matrices, filled below.
    , m_macro_F(3, 3)
    , m_macro_Fdot(3, 3)
    // Phase 5.8 — Lbar cache (refreshed by UpdateMacroscopicF).
    , m_Lbar(3, 3)
    // Phase 5.8 — cached diagnostic structs (default-constructed,
    // zero-initialized; populated by CachePerStepDiagnostics).
    , m_last_consistency_diag()
    , m_last_hill_mandel_diag()
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

    // Phase 6 — build the mortar topology on the boundary/LOR
    // surface and translate every row column/corner pin back to the
    // parent volume FE space. The three shared handles below are kept
    // alive by SimulationState and by the manager-owned components:
    //
    //   parent_fes       : mechanics unknown/residual true-vector space
    //   lor_submesh      : linear boundary/LOR surface geometry
    //   lor_submesh_fes  : classifier and multiplier-row surface space
    //
    // At lor_depth==1 this reduces to the direct boundary trace path;
    // at larger depths the LOR surface supplies the linearized mortar
    // geometry for higher-order parent elements.
    const auto parent_fes = m_sim_state->GetMeshParFiniteElementSpace();
    const auto lor_submesh = m_sim_state->GetLorBoundarySubMesh();
    const auto lor_submesh_fes = m_sim_state->GetLorBoundarySubMeshFes();

    m_classifier = std::make_shared<BoundaryClassifier3D>(
        lor_submesh,
        lor_submesh_fes,
        options.mesh.snap_tol);

    m_projector = std::make_shared<SurfaceProjector>(
        parent_fes,
        lor_submesh_fes,
        lor_submesh,
        options.mesh.snap_tol);

    m_builder = std::make_shared<ConstraintBuilder3D>(
        m_classifier, m_projector, parent_fes);
    m_C_op = std::make_shared<MortarConstraintOperator>(
        m_classifier, m_projector, parent_fes);

    m_saddle_system = std::make_shared<MortarSaddlePointSystem>(
        std::move(k_residual), std::move(k_jacobian), m_C_op);

    m_lambda.SetSize(m_C_op->Height());
    m_g_rhs.SetSize(m_C_op->Height());

    // Initialize macroscopic state.
    //   F̄ = I  (no deformation at simulation start)
    //   Ḟ = 0
    m_macro_F = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        m_macro_F(i, i) = 1.0;
    }
    m_macro_Fdot = 0.0;

    // Phase 5.8 — zero Lbar cache. Refreshed by UpdateMacroscopicF
    // at the top of each load step.
    m_Lbar = 0.0;

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

    //--------------------------------------------------------------------------
    // Phase 5.11.E — build the scaling state.
    //
    // The constraint operator is now in its default-filter state
    // (all pair labels active, all 3 comps). Build the scaler against
    // that filter so a downstream caller that uses the manager
    // BEFORE the first `SyncMortarPbcForStep`/`RebuildForActiveSpec`
    // sees a valid partition. Any subsequent `RebuildForActiveSpec`
    // call refreshes the partition + wrapper offsets to match the
    // new filter.
    //--------------------------------------------------------------------------
    {
        // Block-offsets layout: [0, n_u, n_u + n_lam].
        const int n_u   = m_C_op->Width();
        const int n_lam = m_C_op->Height();
        m_saddle_block_offsets[0] = 0;
        m_saddle_block_offsets[1] = n_u;
        m_saddle_block_offsets[2] = n_u + n_lam;

        // Scaler — translate options-side struct to mortar_pbc-internal
        // config, construct, and populate partition for the default
        // filter.
        const SaddleResidualScalerConfig scaler_cfg =
            TranslateSaddleScalingOptions(options.solvers.saddle_point.scaling);
        m_scaler = std::make_shared<SaddleResidualScaler>(scaler_cfg);
        m_scaler->RebuildPartition(*m_builder,
                                    m_C_op->ActivePairLabels(),
                                    m_C_op->CompMask());

        // ScaledSaddleOperator — wraps m_saddle_system. Always built
        // even when scaling is disabled (identity scaling is bit-for-
        // bit equivalent to the unwrapped op); SystemDriver chooses
        // which to install on the Newton solver based on
        // m_scaler->IsEnabled().
        m_scaled_saddle_system = std::make_shared<ScaledSaddleOperator>(
            std::static_pointer_cast<mfem::Operator>(m_saddle_system),
            m_scaler,
            m_saddle_block_offsets);
    }
}

//==============================================================================
// State updates
//==============================================================================

void MortarPbcManager::UpdateMacroscopicF(const mfem::DenseMatrix& Lbar,
                                          double dt)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::update_macro_F");

    // Phase 5.8 — refresh the Lbar cache so post-processing can
    // re-invoke the diagnostic methods without re-plumbing Lbar
    // through its own state. Deep-copy (mfem::DenseMatrix copy-
    // assignment resizes if needed; ours is already 3×3).
    m_Lbar = Lbar;

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
    mfem::Vector Cv(m_C_op->Height(), mfem::Device::GetMemoryType());
    m_C_op->Mult(v_aff_tdofs, Cv);

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
    // Phase 5.11.I — per-pair |Cv-g|_inf.
    //
    // Classify each row r by its period vector's first non-zero
    // component, scanned in canonical y→x→z order:
    //   period_y != 0 → top pair    (y-axis)
    //   period_x != 0 → right pair  (x-axis)
    //   period_z != 0 → back pair   (z-axis)
    // Edge rows with two non-zero components fall to whichever
    // appears first in this scan order. Corner rows likewise.
    //
    // The y→x→z order matches 5.11.B's PER_PAIR sub-block partition
    // (face_top, face_right, face_back) and 5.11.G's TRDOG
    // diagnostic column ordering, so the three numbers here line up
    // index-for-index with the saddle-system sub-block layout that
    // the scaler partitions over.
    //
    // The `diff` Vector was computed above for `||diff||_inf`; we
    // reuse its host-resident data.
    // ====================================================================
    {
        const double* diff_h   = diff.HostRead();
        const double* period_h = m_period_signed_per_row.HostRead();
        const int     n_rows   = diff.Size();

        double local_top_inf   = 0.0;
        double local_right_inf = 0.0;
        double local_back_inf  = 0.0;

        for (int i = 0; i < n_rows; ++i)
        {
            const double py = period_h[3 * i + 1];
            const double px = period_h[3 * i + 0];
            const double pz = period_h[3 * i + 2];
            const double a  = std::abs(diff_h[i]);

            // First non-zero in canonical y→x→z order wins.
            if (py != 0.0)        { if (a > local_top_inf)   local_top_inf   = a; }
            else if (px != 0.0)   { if (a > local_right_inf) local_right_inf = a; }
            else if (pz != 0.0)   { if (a > local_back_inf)  local_back_inf  = a; }
            // else: all-zero period (shouldn't happen for a valid
            // constraint row, but defend); row contributes to no pair.
        }

        MPI_Allreduce(&local_top_inf,   &out.diff_norm_inf_top,   1,
                      MPI_DOUBLE, MPI_MAX, fes->GetComm());
        MPI_Allreduce(&local_right_inf, &out.diff_norm_inf_right, 1,
                      MPI_DOUBLE, MPI_MAX, fes->GetComm());
        MPI_Allreduce(&local_back_inf,  &out.diff_norm_inf_back,  1,
                      MPI_DOUBLE, MPI_MAX, fes->GetComm());
    }

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
// ComputeAffineVelocityField — Phase 5.8
//
// Project v_lin(x) = L̄·x onto the FES. Reuses the
// LbarTimesXCoefficient defined in the anonymous namespace at the top
// of this file (same coefficient used by ComputeFluctuationField and
// DiagnoseConstraintConsistency).
//
// Together with ComputeFluctuationField, this satisfies the additive
// decomposition v_total = v_lin + v_tilde at every TDOF.
//==============================================================================
void MortarPbcManager::ComputeAffineVelocityField(
    const mfem::DenseMatrix& Lbar,
    mfem::ParGridFunction& v_lin_gf) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::compute_affine_velocity_field");

    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    LbarTimesXCoefficient affine_coeff(Lbar);
    v_lin_gf.SetSpace(fes.get());
    v_lin_gf.ProjectCoefficient(affine_coeff);
}

//==============================================================================
// CachePerStepDiagnostics — Phase 5.8
//
// Compute BOTH ConstraintConsistencyDiagnostic and
// HillMandelDiagnostic from the current converged state and cache
// them as members. Read by PostProcessingDriver::PrintPeriodicValidation
// via the GetLast*Diagnostic() accessors.
//
// Uses the manager's stored m_Lbar (set by the most recent
// UpdateMacroscopicF call).
//==============================================================================
void MortarPbcManager::CachePerStepDiagnostics(
    const mfem::Vector& velocity_tdofs,
    const mfem::Vector& internal_force_tdofs)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::cache_per_step_diagnostics");

    m_last_consistency_diag = DiagnoseConstraintConsistency(m_Lbar);
    m_last_hill_mandel_diag = ComputeHillMandelPowerBalance(
        velocity_tdofs, internal_force_tdofs, m_Lbar);
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

    MFEM_VERIFY(residual.Size() == m_C_op->Width(),
                "AddCTransposeLambdaToResidual: residual size "
                << residual.Size() << " != C^T height (= C width = "
                << m_C_op->Width() << ")");

    mfem::Vector tmp(m_C_op->Width(), mfem::Device::GetMemoryType());
    tmp = 0.0;
    m_C_op->MultTranspose(m_lambda, tmp);
    residual += tmp;
}

//==============================================================================
// RebuildForActiveSpec — Phase 5.9 / Batch A.4
//
// Repopulate constraint state for a new (essential_ids,
// essential_comps) spec. Orchestrates:
//   1. Translate essential_comps -> comp_mask.
//   2. Validate pair completeness + derive active_pair_labels.
//   3. m_C_op->Reset(active_pair_labels, comp_mask).
//   4. Recompute m_corner_ess_tdofs in parent-FES TDOF numbering.
//   5. Resize m_lambda and m_g_rhs to the new local row count.
//   6. Re-emit per-row reference factors.
//
// LOCAL — no MPI calls. All ranks must call with identical args.
//==============================================================================
void MortarPbcManager::RebuildForActiveSpec(
    const std::vector<int>& essential_ids,
    int essential_comps)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::rebuild_for_active_spec");

    // Step 1 — translate essential_comps -> per-component bool mask.
    const std::array<bool, 3> comp_mask = CompMaskFromInt(essential_comps);

    // Step 2 — validate pair completeness AND derive active mortar
    // labels. Aborts via MFEM_VERIFY on missing pair partners or
    // invalid attrs (with a message naming the missing attr + label).
    const std::vector<std::string> active_pair_labels =
        ValidateAndDeriveActivePairLabels(*m_classifier, essential_ids);

    // Step 3 — Reset the EA constraint operator under the new filter.
    // This is a local call (no MPI) that repopulates m_C_op's flat
    // per-row arrays and updates m_C_op->Height(). The construction-
    // time import/export topology is unchanged (over-imports under
    // reduced filter; see MortarConstraintOperator::Reset docs).
    m_C_op->Reset(active_pair_labels, comp_mask);

    // Phase 5.9.A.5 hotfix — refresh the saddle system's cached
    // size members so its Width()/Height() reflect the new
    // m_C_op->Height(). Without this, downstream callers that query
    // saddle_system->Width() see the stale ctor-time value while
    // m_C_op->Height() has moved.
    m_saddle_system->Refresh();

    // Step 4 — Recompute corner essential TDOFs.
    //
    // Replaces m_corner_ess_tdofs (mfem::Array<int>) via assignment —
    // the existing array's storage is freed and the new array (from
    // ComputeCornerEssTDofsFromSpec) takes its place. SystemDriver's
    // GetCornerEssTDofs() returns by const reference to the SAME
    // member, so the new contents are visible to callers without
    // re-plumbing pointers.
    //
    // Phase 5.9.A.5 — passes essential_ids so the incident-face gate
    // (CornersOnFaceAttribute) inside ComputeCornerEssTDofsFromSpec
    // can filter out corners that aren't on any listed face. Phase 6
    // additionally translates all selected classifier/submesh TDOFs
    // through m_projector so the returned local TDOFs belong to the
    // parent mechanics FE space. On an axis-aligned RVE the gate is
    // vacuous; on non-RVE geometries it matters.
    //
    // NB: SystemDriver's mech_operator->UpdateEssTDofsCornerSubset
    // needs to be re-called with the new array after this method
    // returns (handled in Phase 5.9.A.5's SystemDriver::
    // SyncMortarPbcForStep — RebuildForActiveSpec itself doesn't
    // touch mech_operator).
    m_corner_ess_tdofs = ComputeCornerEssTDofsFromSpec(
        *m_classifier,
        *m_projector,
        *m_sim_state->GetMeshParFiniteElementSpace(),
        essential_ids,
        comp_mask);

    // Step 5 — Resize state buffers to the new local row count.
    //
    // mfem::Vector::SetSize preserves the Vector object's address.
    // The saddle system holds a pointer to m_g_rhs (installed via
    // SetConstraintRHS at construction); that pointer remains valid
    // across SetSize.
    //
    // Both buffers are re-zeroed: m_lambda because the old values
    // refer to the OLD constraint system's rows and don't map onto
    // the new rows in a well-defined way; m_g_rhs because the next
    // UpdateConstraintRHS call will re-populate it from the current
    // macroscopic Ḟ̄.
    const int new_height = m_C_op->Height();
    m_lambda.SetSize(new_height);
    m_lambda = 0.0;
    m_g_rhs.SetSize(new_height);
    m_g_rhs = 0.0;

    // Step 6 — Re-emit per-row reference factors under the new
    // filter using ConstraintBuilder3D::EmitRowFactors (filtered
    // overload added in Phase 5.9.A.3). The output sizes match
    // m_C_op->Height() because both walk the same active-pair /
    // comp_mask filter.
    m_builder->EmitRowFactors(active_pair_labels, comp_mask,
                              m_period_signed_per_row,
                              m_component_per_row,
                              m_ell_hat_per_row);

    // Sanity: per-row metadata sizes must match the new height.
    MFEM_VERIFY(m_component_per_row.Size() == new_height,
                "MortarPbcManager::RebuildForActiveSpec: per-row "
                "metadata count " << m_component_per_row.Size()
                << " != m_C_op->Height() " << new_height
                << ". ConstraintBuilder3D::EmitRowFactors (filtered) "
                "disagrees with MortarConstraintOperator::Reset on "
                "the active row count.");
    MFEM_VERIFY(m_period_signed_per_row.Size() == 3 * new_height,
                "MortarPbcManager::RebuildForActiveSpec: "
                "m_period_signed_per_row size "
                << m_period_signed_per_row.Size()
                << " != 3 * new_height " << 3 * new_height
                << ". EmitRowFactors output is malformed.");
    //--------------------------------------------------------------------------
    // Phase 5.11.E — refresh scaling state for the new active spec.
    //
    // The constraint operator's filter has just changed, which may
    // have resized the lambda block. Rebuild the scaler's per-row
    // partition to match the new filter (this also resets d_u and
    // d_lambda to identity — the next `ChooseScalingForStep` call
    // will repopulate them from the post-resize residual norms).
    // Then refresh the scaled-operator wrapper's cached offsets so
    // its internal BlockVector views are sized for the new lambda
    // block count.
    //--------------------------------------------------------------------------
    m_saddle_block_offsets[1] = m_C_op->Width();   // unchanged (u block)
    m_saddle_block_offsets[2] = m_C_op->Width() + m_C_op->Height();

    m_scaler->RebuildPartition(*m_builder,
                                active_pair_labels,
                                comp_mask);

    m_scaled_saddle_system->Refresh(
        std::static_pointer_cast<mfem::Operator>(m_saddle_system),
        m_saddle_block_offsets);
}

//==============================================================================
// SynthesizeDefaultPbcSpec — Phase 5.9 / Batch A.4
//
// Static helper for SystemDriver's empty-periodic_bcs fallback path.
// Returns (essential_ids = all face attrs from classifier.FacePairs,
// essential_comps = 7 = XYZ).
//
// Local — no MPI. Pure lookup on the already-built classifier state.
//==============================================================================
std::pair<std::vector<int>, int> MortarPbcManager::SynthesizeDefaultPbcSpec(
    const BoundaryClassifier3D& classifier)
{
    std::vector<int> ids;
    ids.reserve(classifier.FacePairs().size() * 2);

    for (const auto& tup : classifier.FacePairs())
    {
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);
        ids.push_back(classifier.MeshAttributeForLabel(mortar_label));
        ids.push_back(classifier.MeshAttributeForLabel(nonmortar_label));
    }

    // Dedup defensively — duplicates wouldn't occur for a well-formed
    // classifier (mortar and nonmortar attrs are always distinct for
    // a face pair), but the dedup is cheap and protects against any
    // pathological classifier state.
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    return {ids, /*essential_comps=*/7};   // 7 = XYZ
}

//==============================================================================
// ChooseScalingForStep — Phase 5.11.E
//
// Per-step scaling-factor selection. One MPI_Allreduce of
// (1 + n_subblocks) doubles per call. Collective; all ranks must
// call.
//==============================================================================
void MortarPbcManager::ChooseScalingForStep(const mfem::BlockVector& r_phys)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::choose_scaling_for_step");

    // Disabled path — exact no-op, preserves pre-5.11 behavior.
    if (!m_scaler->IsEnabled())
    {
        return;
    }

    const int n_subblocks = m_scaler->NumSubblocks();
    MFEM_VERIFY(n_subblocks > 0,
                "MortarPbcManager::ChooseScalingForStep: scaler partition "
                "is empty — was RebuildPartition called? "
                "(Should have been done at ctor + every RebuildForActiveSpec.)");

    //--------------------------------------------------------------------------
    // Step 1 — local sums of squares.
    //
    // Layout in the packed buffer:
    //   local_sq[0]            = sum_i r_u[i]^2          (local u block)
    //   local_sq[1 + k]        = sum_{i in sb k} r_lambda[i]^2   (local)
    //
    // r_u is a TDOF vector (rank-partitioned); r_lambda is a
    // constraint-row vector (also rank-partitioned). The Allreduce
    // below sums across ranks.
    //--------------------------------------------------------------------------
    std::vector<double> local_sq(1 + n_subblocks, 0.0);

    {
        const mfem::Vector& r_u = r_phys.GetBlock(0);
        const double* d = r_u.HostRead();
        double s = 0.0;
        const int n = r_u.Size();
        for (int i = 0; i < n; ++i)
        {
            s += d[i] * d[i];
        }
        local_sq[0] = s;
    }

    {
        const mfem::Vector& r_lam = r_phys.GetBlock(1);
        mfem::Vector lam_sq_local;
        m_scaler->UnscaledLambdaSubblockNormsSqLocal(r_lam, lam_sq_local);
        MFEM_ASSERT(lam_sq_local.Size() == n_subblocks,
                    "ChooseScalingForStep: subblock sum count mismatch");
        const double* sb = lam_sq_local.HostRead();
        for (int k = 0; k < n_subblocks; ++k)
        {
            local_sq[1 + k] = sb[k];
        }
    }

    //--------------------------------------------------------------------------
    // Step 2 — single MPI_Allreduce SUM (the per-step protocol).
    //--------------------------------------------------------------------------
    std::vector<double> global_sq(1 + n_subblocks, 0.0);
    MPI_Allreduce(local_sq.data(),
                  global_sq.data(),
                  static_cast<int>(local_sq.size()),
                  MPI_DOUBLE, MPI_SUM,
                  m_sim_state->GetMesh()->GetComm());

    //--------------------------------------------------------------------------
    // Step 3 — sqrt + Choose.
    //--------------------------------------------------------------------------
    const double r_u_norm = std::sqrt(global_sq[0]);

    mfem::Vector sb_norms(n_subblocks);
    double* sbn = sb_norms.HostWrite();
    for (int k = 0; k < n_subblocks; ++k)
    {
        sbn[k] = std::sqrt(global_sq[1 + k]);
    }

    m_scaler->Choose(r_u_norm, sb_norms);
}

//==============================================================================
// Private helpers
//==============================================================================

void MortarPbcManager::BuildCornerEssTDofs()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::build_corner_ess_tdofs");

    // Phase 5.3.B / Phase 6 — populate m_corner_ess_tdofs with the 8
    // corners' (gtdof_x, gtdof_y, gtdof_z) components. The classifier
    // records live in boundary/LOR-submesh numbering; the projector-
    // aware free function translates each selected component to
    // parent-volume true-DOF numbering before filtering to this rank's
    // local partition. Keeping this conversion in a free function lets
    // test_mortar_pbc_manager.cpp validate it without constructing a
    // full SimulationState.
    m_corner_ess_tdofs = ComputeCornerEssTDofs(
        *m_classifier,
        *m_projector,
        *m_sim_state->GetMeshParFiniteElementSpace());

    // Self-check: across all ranks the corner TDOFs must total to 24.
    const int local_count = m_corner_ess_tdofs.Size();
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  m_classifier->Comm());
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
    m_builder->EmitRowFactors(m_period_signed_per_row,
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
