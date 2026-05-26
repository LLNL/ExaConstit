// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of ConstraintBuilder3D, ported from
// `mortar_pbc/constraint_builder_3d.py`. See header for design doc.
//
// Phase 5.7.A fix — EmitRowFactors now emits the full periodic shift
// VECTOR per row (period_signed) rather than a single axis index.
// Background: for edge mortars, the axis previously stored
// (`axis_per_row[i]`) was the EDGE-PARALLEL axis, but the g-formula
// in `MortarPbcManager::UpdateConstraintRHS` interpreted it as the
// JUMP axis. These are different for edges — an axis-y edge can have
// periodic shift along x and/or z, never y. The result was a g vector
// supported on the wrong constraint rows. Emitting period_signed
// directly removes the ambiguity.
//
// Phase 5.9 — Component-restricted PBC filter
// -------------------------------------------
// New overloads of `Build`, `BuildHypreParMatrix`, `NumLocalRows`,
// `NumConstraints`, and `EmitRowFactors` take a `(active_pair_labels,
// comp_mask)` filter. See the header for filter semantics. The
// parameter-less overloads forward to the filtered ones with all
// pairs active and `{true, true, true}` for `comp_mask`, exactly
// reproducing pre-5.9 behavior.
//
// Phase 6.0.F — projector-aware column translation
// ------------------------------------------------
// The builder can now be constructed with a boundary/LOR classifier
// and SurfaceProjector. The row walk is unchanged, but every emitted
// matrix column is translated to parent-volume FES true DOFs before
// insertion. This keeps the assembled HypreParMatrix path aligned with
// the projector-aware element-assembly operator.

#include "constraint_builder_3d.hpp"

#include "boundary_classifier_3d.hpp"
#include "boundary_helpers_3d.hpp"
#include "face_mortar_assembler_3d.hpp"
#include "mortar_assembler_2d.hpp"
#include "types_3d.hpp"

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// Period-vector helpers — Phase 5.7.A
//==============================================================================
// (PeriodSigned helper removed in Phase 4.2 / Batch J — was only used
// by the now-decommissioned ScatterFacePair. The classifier's
// BuildLocalPairBlocks computes its own period_signed inline from
// bbox planes.)
//
// Phase 5.7.A — period_signed reintroduced at the EmitRowFactors
// level. See `ComputeFacePeriodSigned` and `ComputeEdgePeriodSigned`
// below. The classifier still computes its own version for face
// matching in BuildLocalPairBlocks; we deliberately recompute here
// rather than threading classifier state through the LocalPairBlock
// struct, to keep the change surgical. Both compute the same value
// from the same source data (FaceInfo3D::plane_value and
// EdgeInfo3D::coords), so consistency is maintained.
//==============================================================================

int AxisStrToInt(const std::string& s)
{
    if (s == "x") { return 0; }
    if (s == "y") { return 1; }
    if (s == "z") { return 2; }
    MFEM_ABORT("ConstraintBuilder3D::AxisStrToInt: unknown axis '"
               << s << "' (expected 'x', 'y', or 'z').");
    return -1;  // unreachable
}

//==============================================================================
// ComputeFacePeriodSigned — Phase 5.7.A
//
// For a face pair (axis, mortar, nonmortar), the periodic shift
// vector is L_axis · sign · ê_axis, where the sign comes from
// (nonmortar.plane_value - mortar.plane_value). For an axis-aligned
// box RVE this is ±L_axis. Other components are zero.
//==============================================================================
std::array<double, 3> ComputeFacePeriodSigned(
    const BoundaryClassifier3D& classifier,
    const std::string& axis_str,
    const std::string& mortar_label,
    const std::string& nonmortar_label)
{
    const int axis_idx = AxisStrToInt(axis_str);
    const FaceInfo3D& mortar    = classifier.Faces().at(mortar_label);
    const FaceInfo3D& nonmortar = classifier.Faces().at(nonmortar_label);

    MFEM_VERIFY(mortar.perpendicular_axis == axis_str,
                "ComputeFacePeriodSigned: mortar face '" << mortar_label
                << "' perpendicular_axis '" << mortar.perpendicular_axis
                << "' does not match the face-pair axis '" << axis_str
                << "'. Classifier is internally inconsistent.");
    MFEM_VERIFY(nonmortar.perpendicular_axis == axis_str,
                "ComputeFacePeriodSigned: nonmortar face '" << nonmortar_label
                << "' perpendicular_axis '" << nonmortar.perpendicular_axis
                << "' does not match the face-pair axis '" << axis_str
                << "'. Classifier is internally inconsistent.");

    std::array<double, 3> ps = {0.0, 0.0, 0.0};
    ps[axis_idx] = nonmortar.plane_value - mortar.plane_value;
    return ps;
}

//==============================================================================
// ComputeEdgePeriodSigned — Phase 5.7.A
//
// For an edge pair (axis, mortar, nonmortar), the edges are parallel
// to `axis`. Their coordinates along the parametric (= edge-parallel)
// axis vary; the coordinates along the two TRANSVERSE axes are
// constant for all interior nodes of an edge. The period_signed
// vector is the difference between nonmortar and mortar transverse
// coordinates — zero along the parametric axis, possibly nonzero
// along the other two.
//
// Reads transverse coords from the FIRST interior node of each edge
// (`coords(0, k)`); any interior node would do since transverse
// coords are invariant along the edge. Asserts the edge has at least
// one interior node — should always hold post-classifier, but a bug
// upstream would manifest as a misleading silent-zero period vector
// without this assertion.
//==============================================================================
std::array<double, 3> ComputeEdgePeriodSigned(
    const BoundaryClassifier3D& classifier,
    const std::string& axis_str,
    const std::string& mortar_label,
    const std::string& nonmortar_label)
{
    const int axis_idx = AxisStrToInt(axis_str);
    const EdgeInfo3D& mortar    = classifier.Edges().at(mortar_label);
    const EdgeInfo3D& nonmortar = classifier.Edges().at(nonmortar_label);

    MFEM_VERIFY(mortar.parametric_axis == axis_str,
                "ComputeEdgePeriodSigned: mortar edge '" << mortar_label
                << "' parametric_axis '" << mortar.parametric_axis
                << "' does not match the edge-pair axis '" << axis_str
                << "'. Classifier is internally inconsistent.");
    MFEM_VERIFY(nonmortar.parametric_axis == axis_str,
                "ComputeEdgePeriodSigned: nonmortar edge '" << nonmortar_label
                << "' parametric_axis '" << nonmortar.parametric_axis
                << "' does not match the edge-pair axis '" << axis_str
                << "'. Classifier is internally inconsistent.");
    MFEM_VERIFY(mortar.coords.NumRows() > 0,
                "ComputeEdgePeriodSigned: mortar edge '" << mortar_label
                << "' has zero interior nodes; cannot read transverse "
                "coords.");
    MFEM_VERIFY(nonmortar.coords.NumRows() > 0,
                "ComputeEdgePeriodSigned: nonmortar edge '" << nonmortar_label
                << "' has zero interior nodes; cannot read transverse "
                "coords.");

    std::array<double, 3> ps = {0.0, 0.0, 0.0};
    // Transverse axes only — period along the edge-parallel axis is 0.
    for (int k = 0; k < 3; ++k)
    {
        if (k == axis_idx) { continue; }
        ps[k] = nonmortar.coords(0, k) - mortar.coords(0, k);
    }
    return ps;
}

//==============================================================================
// Phase 5.9 — filter helpers.
//==============================================================================

/// Map a face label to its perpendicular axis. Returns empty string
/// if `label` is not one of the 6 recognized face labels.
std::string LabelToAxis(const std::string& label)
{
    // Static map keeps lookup cheap and centralizes the mapping.
    static const std::map<std::string, std::string> kLabelToAxis = {
        {"left",   "x"}, {"right", "x"},
        {"bottom", "y"}, {"top",   "y"},
        {"front",  "z"}, {"back",  "z"}
    };
    auto it = kLabelToAxis.find(label);
    return (it != kLabelToAxis.end()) ? it->second : std::string();
}

/// Derive the set of active axes (subset of {"x", "y", "z"}) from a
/// list of pair labels. Labels can be mortar or nonmortar side; the
/// mapping to axis is the same. Unknown labels are silently dropped
/// (caller is responsible for upstream validation).
std::set<std::string> ActiveAxesFromPairLabels(
    const std::vector<std::string>& active_pair_labels)
{
    std::set<std::string> axes;
    for (const std::string& label : active_pair_labels)
    {
        const std::string axis = LabelToAxis(label);
        if (!axis.empty()) { axes.insert(axis); }
    }
    return axes;
}

/// Given an edge's parametric (parallel) axis, return the two
/// perpendicular axes. The edge mortar at parametric axis `a`
/// requires both perpendicular axes' face pairs to be active.
std::array<std::string, 2> EdgePerpendicularAxes(
    const std::string& edge_param_axis)
{
    if (edge_param_axis == "x") { return {"y", "z"}; }
    if (edge_param_axis == "y") { return {"x", "z"}; }
    MFEM_ASSERT(edge_param_axis == "z",
                "EdgePerpendicularAxes: unknown axis '"
                << edge_param_axis << "'");
    return {"x", "y"};
}

/// Number of active components in the mask.
int CountActiveComps(const std::array<bool, 3>& comp_mask)
{
    return (comp_mask[0] ? 1 : 0)
         + (comp_mask[1] ? 1 : 0)
         + (comp_mask[2] ? 1 : 0);
}

/// Per-component local row index within a node, given the mask.
/// Returns the position of `c` in the subsequence of true entries
/// in `comp_mask`, or -1 if `comp_mask[c]` is false.
///
/// Examples:
///   comp_mask = {true, true, true}:   c=0→0, c=1→1, c=2→2
///   comp_mask = {true, false, false}: c=0→0, c=1→-1, c=2→-1
///   comp_mask = {false, true, true}:  c=0→-1, c=1→0, c=2→1
///   comp_mask = {true, false, true}:  c=0→0, c=1→-1, c=2→1
int LocalRowOfComp(const std::array<bool, 3>& comp_mask, int c)
{
    if (!comp_mask[c]) { return -1; }
    int idx = 0;
    for (int i = 0; i < c; ++i)
    {
        if (comp_mask[i]) { ++idx; }
    }
    return idx;
}

/// Convenience: build the "all active" mortar-label list from the
/// classifier's FacePairs(). Used by the parameter-less forwarders
/// to invoke the filtered overloads with the default "all pairs"
/// argument.
std::vector<std::string> AllMortarLabels(
    const BoundaryClassifier3D& classifier)
{
    std::vector<std::string> labels;
    labels.reserve(3);
    for (const auto& tup : classifier.FacePairs())
    {
        labels.push_back(std::get<1>(tup));  // mortar label
    }
    return labels;
}

const BoundaryClassifier3D& RequireClassifier(
    const std::shared_ptr<const BoundaryClassifier3D>& classifier)
{
    MFEM_VERIFY(classifier != nullptr,
                "ConstraintBuilder3D: classifier must be non-null.");
    return *classifier;
}

}  // anonymous namespace

//==============================================================================
// Constructor
//==============================================================================

ConstraintBuilder3D::ConstraintBuilder3D(const BoundaryClassifier3D& classifier)
    : m_classifier(classifier)
    , m_parent_fes_raw(&classifier.Fes())
    , m_edge_assembler()
    , m_quad_face_assembler()
    , m_tri_face_assembler()
    , m_gtdof_lookup(classifier.GtdofXyzLookup())
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::ctor");
}

ConstraintBuilder3D::ConstraintBuilder3D(
    std::shared_ptr<const BoundaryClassifier3D> classifier,
    std::shared_ptr<const SurfaceProjector> projector,
    std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes)
    : m_classifier(RequireClassifier(classifier))
    , m_classifier_owner(std::move(classifier))
    , m_projector(std::move(projector))
    , m_parent_fes_owner(std::move(parent_fes))
    , m_parent_fes_raw(m_parent_fes_owner.get())
    , m_edge_assembler()
    , m_quad_face_assembler()
    , m_tri_face_assembler()
    , m_gtdof_lookup(m_classifier.GtdofXyzLookup())
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::projector_ctor");

    MFEM_VERIFY(m_projector != nullptr,
                "ConstraintBuilder3D: projector must be non-null.");
    MFEM_VERIFY(m_parent_fes_owner != nullptr,
                "ConstraintBuilder3D: parent FES must be non-null.");
    MFEM_VERIFY(m_projector->Width() == m_parent_fes_owner->GetTrueVSize(),
                "ConstraintBuilder3D: projector Width() does not match "
                "parent FES local true-vector size.");
    MFEM_VERIFY(m_projector->Height() == m_classifier.Fes().GetTrueVSize(),
                "ConstraintBuilder3D: projector Height() does not match "
                "classifier/submesh FES local true-vector size.");
}

int ConstraintBuilder3D::ParentGlobalTrueVSize() const
{
    return ParentFes().GlobalTrueVSize();
}

int ConstraintBuilder3D::ParentGtdofFromClassifierGtdof(
    int classifier_gtdof) const
{
    if (classifier_gtdof < 0) { return classifier_gtdof; }
    return m_projector ? m_projector->ParentGtdof(classifier_gtdof)
                       : classifier_gtdof;
}

std::array<int, 3> ConstraintBuilder3D::ParentGtdofXyzFromClassifierX(
    int classifier_g_x) const
{
    if (classifier_g_x < 0) { return {{-1, -1, -1}}; }

    const auto it = m_gtdof_lookup.find(classifier_g_x);
    MFEM_VERIFY(it != m_gtdof_lookup.end(),
                "ConstraintBuilder3D: classifier gtdof "
                << classifier_g_x << " not in gtdof_xyz_lookup.");

    std::array<int, 3> parent_xyz = {{-1, -1, -1}};
    for (int c = 0; c < kVDim; ++c)
    {
        parent_xyz[c] = ParentGtdofFromClassifierGtdof(it->second[c]);
    }
    return parent_xyz;
}

int ConstraintBuilder3D::ParentOwnerRankFromClassifierX(
    int classifier_g_x) const
{
    if (classifier_g_x < 0) { return -1; }
    const int parent_g_x = ParentGtdofFromClassifierGtdof(classifier_g_x);
    return m_projector ? m_projector->ParentOwnerRank(parent_g_x)
                       : m_classifier.GtdofOwnerRank(parent_g_x);
}

//==============================================================================
// NumConstraints — parameter-less forwarder (pre-5.9 behavior)
//==============================================================================

int ConstraintBuilder3D::NumConstraints() const
{
    return NumConstraints(AllMortarLabels(m_classifier),
                          {true, true, true});
}

//==============================================================================
// NumConstraints — Phase 5.9 filtered
//==============================================================================

int ConstraintBuilder3D::NumConstraints(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask) const
{
    const std::set<std::string> active_axes =
        ActiveAxesFromPairLabels(active_pair_labels);
    const int n_comps = CountActiveComps(comp_mask);
    if (n_comps == 0 || active_axes.empty()) { return 0; }

    int n = 0;

    // Edge pairs: each kept nonmortar edge contributes n_comps *
    // n_interior_nodes constraint rows. Gated on BOTH perpendicular
    // axes being active.
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& axis_str = std::get<0>(tup);
        const auto perps = EdgePerpendicularAxes(axis_str);
        if (active_axes.find(perps[0]) == active_axes.end()
            || active_axes.find(perps[1]) == active_axes.end())
        {
            continue;
        }
        const std::string& nonmortar_label = std::get<2>(tup);
        const EdgeInfo3D& nonmortar_edge =
            m_classifier.Edges().at(nonmortar_label);
        n += n_comps * nonmortar_edge.NumNodes();
    }

    // Face pairs: kept-nonmortar count is the size of interior_gtdofs_x.
    // Gated on the pair's axis being active.
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis_str = std::get<0>(tup);
        if (active_axes.find(axis_str) == active_axes.end())
        {
            continue;
        }
        const std::string& nonmortar_label = std::get<2>(tup);
        const FaceInfo3D& nonmortar_face =
            m_classifier.Faces().at(nonmortar_label);
        n += n_comps * nonmortar_face.interior_gtdofs_x.Size();
    }

    return n;
}

//==============================================================================
// NumLocalRows — parameter-less forwarder (pre-5.9 behavior)
//==============================================================================

int ConstraintBuilder3D::NumLocalRows() const
{
    return NumLocalRows(AllMortarLabels(m_classifier),
                        {true, true, true});
}

//==============================================================================
// NumLocalRows — Phase 5.9 filtered
//
// Phase 4.2 / Batch N — number of constraint rows owned by THIS rank
// under the FES-aligned row partition. Counts edge rows whose
// x-component nonmortar gtdof is FES-owned by this rank, plus face
// rows already routed to this rank. Under filter, the count includes
// only rows for active pairs and active components.
//==============================================================================
int ConstraintBuilder3D::NumLocalRows(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask) const
{
    // Run the emitter once and discard the buffers — it returns the
    // local row count as its return value. The emitter is the
    // authoritative source of "what rows does this rank own?", so
    // implementing this any other way risks divergence.
    //
    // Cost is O(local_rows + sum_of_local_block_nnz), which is the
    // same as one pass of BuildHypreParMatrix's emit step. For
    // typical patch tests this is microseconds; for production
    // a caller that needs the value repeatedly should cache it.
    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;
    return EmitConstraintTriples(active_pair_labels, comp_mask,
                                 rows, cols, vals);
}

//==============================================================================
// Build — parameter-less forwarder (pre-5.9 behavior)
//==============================================================================

std::unique_ptr<mfem::SparseMatrix> ConstraintBuilder3D::Build() const
{
    return Build(AllMortarLabels(m_classifier), {true, true, true});
}

//==============================================================================
// Build — Phase 5.9 filtered
//==============================================================================

std::unique_ptr<mfem::SparseMatrix> ConstraintBuilder3D::Build(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::build");

    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;

    const int n_rows = EmitConstraintTriples(active_pair_labels, comp_mask,
                                             rows, cols, vals);
    const int n_cols = ParentGlobalTrueVSize();

    // Build the SparseMatrix from COO triples. mfem::SparseMatrix
    // doesn't have a direct COO ctor, so we build it via Add() into
    // a finalize-on-Finalize() instance.
    auto C = std::make_unique<mfem::SparseMatrix>(n_rows, n_cols);
    const std::size_t n_nz = vals.size();
    for (std::size_t i = 0; i < n_nz; ++i)
    {
        C->Add(rows[i], cols[i], vals[i]);
    }
    C->Finalize();
    return C;
}

//==============================================================================
// EmitConstraintTriples — Phase 5.9 filtered shared helper
//
// Runs the edge + face scatter loop and populates the supplied COO
// buffers in this rank's local row indexing.
//
// Pre-5.9 behavior is recovered when called with all mortar labels
// active and `{true, true, true}` for comp_mask (which is what the
// parameter-less public methods do via their forwarders).
//==============================================================================

int ConstraintBuilder3D::EmitConstraintTriples(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask,
    std::vector<int>& rows,
    std::vector<int>& cols,
    std::vector<double>& vals) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::emit_triples");

    const std::set<std::string> active_axes =
        ActiveAxesFromPairLabels(active_pair_labels);

    // Reserve a generous-but-not-wasteful upper bound: each nonmortar
    // node contributes one diagonal D entry plus on the order of
    // (n_mortar_nodes_in_overlap) off-diagonal -A_m entries per
    // component. A factor of 8 per nonmortar TDOF is plenty for the
    // axis-aligned conforming case. Under filter the actual count is
    // <= this estimate (we use NumConstraints() with default filter
    // here to keep the reservation simple; it over-reserves under
    // reduced filter but never under-reserves).
    const int n_constraints_est = NumConstraints();
    rows.reserve(static_cast<std::size_t>(8) * n_constraints_est);
    cols.reserve(static_cast<std::size_t>(8) * n_constraints_est);
    vals.reserve(static_cast<std::size_t>(8) * n_constraints_est);

    int row_offset = 0;

    //--- Edge mortar blocks (up to 9 pairs) ---
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& axis_str       = std::get<0>(tup);

        // Phase 5.9 — edge-pair filter: both perpendicular axes must
        // be active for this edge group to contribute rows.
        const auto perps = EdgePerpendicularAxes(axis_str);
        if (active_axes.find(perps[0]) == active_axes.end()
            || active_axes.find(perps[1]) == active_axes.end())
        {
            continue;
        }

        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);
        const EdgeInfo3D& mortar_edge    = m_classifier.Edges().at(mortar_label);
        const EdgeInfo3D& nonmortar_edge = m_classifier.Edges().at(nonmortar_label);

        // MortarAssembler2D::AssemblePair takes (plus_edge=nonmortar,
        // minus_edge=mortar). The 2D mortar's "plus" naming aligns
        // with our nonmortar (rows-owner) per the architecture
        // glossary.
        MortarBlock2D block =
            m_edge_assembler.AssemblePair(nonmortar_edge, mortar_edge);
        row_offset = ScatterEdgeBlock(block, nonmortar_edge, mortar_edge,
                                      comp_mask,
                                      rows, cols, vals, row_offset);
    }

    //--- Face mortar blocks (up to 3 pairs) ---
    //
    // Phase 4.2 / Batch I+J: blocks are pre-matched and pre-assembled
    // by the classifier (tile-locally), then AllGather'd to every
    // rank. Read them via PairBlocks() and scatter.
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis            = std::get<0>(tup);

        // Phase 5.9 — face-pair filter: skip this axis if its pair
        // is not in the user's active set.
        if (active_axes.find(axis) == active_axes.end())
        {
            continue;
        }

        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        // Find blocks for this (axis, mortar, nonmortar). At most one
        // per geometry kind; we scatter quad first then tri to
        // preserve the row order of the legacy path.
        const BoundaryClassifier3D::LocalPairBlock* quad_block = nullptr;
        const BoundaryClassifier3D::LocalPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair != axis
                || lpb.mortar_label != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if (lpb.geometry_kind == "quad") { quad_block = &lpb; }
            else if (lpb.geometry_kind == "tri") { tri_block = &lpb; }
        }

        if (quad_block != nullptr)
        {
            row_offset = ScatterFaceBlock(quad_block->block, comp_mask,
                                          rows, cols, vals, row_offset);
        }
        if (tri_block != nullptr)
        {
            row_offset = ScatterFaceBlock(tri_block->block, comp_mask,
                                          rows, cols, vals, row_offset);
        }
    }

    return row_offset;
}

//==============================================================================
// EmitRowFactors — parameter-less forwarder (pre-5.9 behavior)
//==============================================================================

void ConstraintBuilder3D::EmitRowFactors(
    mfem::Vector& period_signed_per_row,
    mfem::Array<int>& component_index,
    mfem::Vector& ell_hat) const
{
    EmitRowFactors(AllMortarLabels(m_classifier), {true, true, true},
                   period_signed_per_row, component_index, ell_hat);
}

//==============================================================================
// EmitRowFactors — Phase 5.9 filtered
//
// Per-row reference-geometry metadata. Mirrors the row-enumeration
// pattern of EmitConstraintTriples exactly so that emit position k
// corresponds to constraint row k. Edges go through the row-owner
// filter (FES ownership of the x-component nonmortar gtdof); face
// pair blocks are pre-routed by the classifier so they require no
// per-row filter.
//
// Phase 5.7.A — emits `period_signed_per_row` (Vector of length
// 3 * n_local_rows, row-major), `component_index`, and `ell_hat`.
// See header for the downstream g formula in
// `MortarPbcManager::UpdateConstraintRHS`.
//
// Phase 5.9 — same iteration as the unfiltered version, but gated on
// `active_pair_labels` and `comp_mask`. Only emitted rows are pushed
// to the output buffers; row count matches `EmitConstraintTriples`
// under the same filter.
//==============================================================================
void ConstraintBuilder3D::EmitRowFactors(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask,
    mfem::Vector& period_signed_per_row,
    mfem::Array<int>& component_index,
    mfem::Vector& ell_hat) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::emit_row_factors");

    const std::set<std::string> active_axes =
        ActiveAxesFromPairLabels(active_pair_labels);

    // Build into std::vector first (cheap, growable); copy out at the
    // end to mfem::Vector / mfem::Array. The upper-bound row count is
    // NumConstraints(); local count is at most that.
    const int n_constraints_est = NumConstraints();
    std::vector<double> period_buf;   // 3 doubles per row, row-major
    std::vector<int>    comp_buf;
    std::vector<double> ell_buf;
    period_buf.reserve(static_cast<std::size_t>(3 * n_constraints_est));
    comp_buf.reserve(static_cast<std::size_t>(n_constraints_est));
    ell_buf.reserve(static_cast<std::size_t>(n_constraints_est));

    const int my_rank = m_classifier.Rank();

    //--- Edge mortar blocks ---
    //
    // We re-run the edge assembler here. The cost is up to 9 small
    // dense assemblies per call — negligible at construction time, and
    // matching EmitConstraintTriples' pattern keeps the row order
    // identical. (Future refactor: cache the assembled blocks once
    // and reuse across both methods. Not required here.)
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& axis_str        = std::get<0>(tup);

        // Phase 5.9 — edge-pair filter.
        const auto perps = EdgePerpendicularAxes(axis_str);
        if (active_axes.find(perps[0]) == active_axes.end()
            || active_axes.find(perps[1]) == active_axes.end())
        {
            continue;
        }

        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        // Phase 5.7.A — compute the period_signed VECTOR for this
        // edge pair. For an edge parallel to axis_str, the parallel-
        // axis component is always 0; the two transverse-axis
        // components encode the (Δa · L_a, Δb · L_b) shift between
        // mortar and nonmortar edge positions.
        const std::array<double, 3> period_signed =
            ComputeEdgePeriodSigned(m_classifier, axis_str,
                                    mortar_label, nonmortar_label);

        const EdgeInfo3D& mortar_edge    = m_classifier.Edges().at(mortar_label);
        const EdgeInfo3D& nonmortar_edge = m_classifier.Edges().at(nonmortar_label);

        MortarBlock2D block =
            m_edge_assembler.AssemblePair(nonmortar_edge, mortar_edge);

        const int n_n = nonmortar_edge.NumNodes();
        for (int k = 0; k < n_n; ++k)
        {
            // Row-owner filter — same as ScatterEdgeBlock.
            const int g_n_x = nonmortar_edge.gtdofs_x[k];
            const int owner = (g_n_x >= 0)
                              ? ParentOwnerRankFromClassifierX(g_n_x) : -1;
            if (owner != my_rank) { continue; }

            const double D_kk = block.D_nm(k);
            // Phase 5.9 — emit one entry per ACTIVE component.
            for (int c = 0; c < kVDim; ++c)
            {
                if (!comp_mask[c]) { continue; }
                period_buf.push_back(period_signed[0]);
                period_buf.push_back(period_signed[1]);
                period_buf.push_back(period_signed[2]);
                comp_buf.push_back(c);
                ell_buf.push_back(D_kk);
            }
        }
    }

    //--- Face mortar blocks (pre-routed by the classifier) ---
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis_str        = std::get<0>(tup);

        // Phase 5.9 — face-pair filter.
        if (active_axes.find(axis_str) == active_axes.end())
        {
            continue;
        }

        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        // Phase 5.7.A — for a face pair, period_signed is L_axis ·
        // sign · ê_axis. One nonzero component (the face normal axis).
        const std::array<double, 3> period_signed =
            ComputeFacePeriodSigned(m_classifier, axis_str,
                                    mortar_label, nonmortar_label);

        // Find quad and tri blocks for this pair. Same lookup
        // pattern EmitConstraintTriples uses.
        const FaceMortarPairBlock* quad_block = nullptr;
        const FaceMortarPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair       != axis_str
                || lpb.mortar_label    != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if      (lpb.geometry_kind == "quad") { quad_block = &lpb.block; }
            else if (lpb.geometry_kind == "tri")  { tri_block  = &lpb.block; }
        }

        auto emit_face_block = [&](const FaceMortarPairBlock& block)
        {
            const int n_n = block.NumNonmortarKept();
            for (int k = 0; k < n_n; ++k)
            {
                const double D_kk = block.D(k);
                // Phase 5.9 — emit one entry per ACTIVE component.
                for (int c = 0; c < kVDim; ++c)
                {
                    if (!comp_mask[c]) { continue; }
                    period_buf.push_back(period_signed[0]);
                    period_buf.push_back(period_signed[1]);
                    period_buf.push_back(period_signed[2]);
                    comp_buf.push_back(c);
                    ell_buf.push_back(D_kk);
                }
            }
        };

        if (quad_block != nullptr) { emit_face_block(*quad_block); }
        if (tri_block  != nullptr) { emit_face_block(*tri_block);  }
    }

    // Copy out to mfem::Vector / mfem::Array outputs.
    //
    // HostWrite()-based population, matching the ecmech idiom (see
    // Hotfix #2 — phase_5_5_b4_hotfix_2_emit_row_factors.md). The
    // caller in MortarPbcManager constructs these with
    // Device::GetMemoryType(); SetSize() on the Vector members sets
    // both VALID_HOST and VALID_DEVICE flags, so the indexed-write
    // assertion in mem_manager.hpp fires without an explicit
    // HostWrite() to clear VALID_DEVICE.
    const int n_local = static_cast<int>(comp_buf.size());
    period_signed_per_row.SetSize(3 * n_local);
    component_index.SetSize(n_local);
    ell_hat.SetSize(n_local);
    double* period_data = period_signed_per_row.HostWrite();
    int*    comp_data   = component_index.HostWrite();
    double* ell_data    = ell_hat.HostWrite();
    for (int i = 0; i < n_local; ++i)
    {
        period_data[3*i + 0] = period_buf[3*i + 0];
        period_data[3*i + 1] = period_buf[3*i + 1];
        period_data[3*i + 2] = period_buf[3*i + 2];
        comp_data[i] = comp_buf[i];
        ell_data[i]  = ell_buf[i];
    }
}

//==============================================================================
// GetRowSubblockIds — parameter-less forwarder (defaults: all pairs / all comps)
//==============================================================================

void ConstraintBuilder3D::GetRowSubblockIds(
    SubblockPartition partition,
    std::vector<std::string>& subblock_labels,
    mfem::Array<int>& subblock_of_row) const
{
    GetRowSubblockIds(partition,
                      AllMortarLabels(m_classifier),
                      {true, true, true},
                      subblock_labels,
                      subblock_of_row);
}

//==============================================================================
// GetRowSubblockIds — Phase 5.11
//
// Walks the constraint-row index space in EmitConstraintTriples'
// order and emits per-row sub-block IDs. Pair-iteration filters and
// per-component row strides match EmitConstraintTriples /
// EmitRowFactors exactly, so `subblock_of_row[i]` aligns with row `i`
// of the constraint matrix produced by `Build(active_pair_labels,
// comp_mask)`.
//
// The walk:
//   1. Edge pairs (m_classifier.EdgePairs() order), filtered on both
//      perpendicular axes ∈ active_axes. Per kept (active + owned)
//      nonmortar node: emit n_comps_a sub-block IDs.
//   2. Face pairs (m_classifier.FacePairs() order), filtered on axis
//      ∈ active_axes. For each, find quad and tri blocks (quad first,
//      then tri, matching ScatterFaceBlock's emission order). Per
//      kept nonmortar node: emit n_comps_a sub-block IDs.
//
// For FaceEdge: all edge rows → ID 0, all face rows → ID 1; labels
// always {"edge", "face"} regardless of filter (empty sub-blocks OK
// — see header note on diagnostic-column stability).
//
// For PerPair: each active pair → its own sequential ID in walk
// order; labels include only active pairs.
//==============================================================================

void ConstraintBuilder3D::GetRowSubblockIds(
    SubblockPartition partition,
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask,
    std::vector<std::string>& subblock_labels,
    mfem::Array<int>& subblock_of_row) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::get_row_subblock_ids");

    const std::set<std::string> active_axes =
        ActiveAxesFromPairLabels(active_pair_labels);
    const int n_comps_a = CountActiveComps(comp_mask);
    const int my_rank   = m_classifier.Rank();

    // Pre-size the output. NumLocalRows under the same filter is the
    // authoritative count; we'll MFEM_VERIFY against this at the end
    // to catch any walk-order divergence with EmitConstraintTriples.
    const int n_local = NumLocalRows(active_pair_labels, comp_mask);
    subblock_of_row.SetSize(n_local);

    //--------------------------------------------------------------------------
    // Build subblock_labels.
    //--------------------------------------------------------------------------
    subblock_labels.clear();
    if (partition == SubblockPartition::FaceEdge)
    {
        // Two labels — edge first to match walk order, then face.
        // Always emit BOTH even if one is empty under the filter,
        // for diagnostic-column stability across Phase 5.9 spec
        // transitions.
        subblock_labels.push_back("edge");
        subblock_labels.push_back("face");
    }
    else
    {
        // PerPair: one label per ACTIVE pair, in walk order. Edges
        // first (m_classifier.EdgePairs()), then faces
        // (m_classifier.FacePairs()).
        for (const auto& tup : m_classifier.EdgePairs())
        {
            const std::string& axis_str = std::get<0>(tup);
            const auto perps = EdgePerpendicularAxes(axis_str);
            if (active_axes.find(perps[0]) == active_axes.end()
                || active_axes.find(perps[1]) == active_axes.end())
            {
                continue;
            }
            const std::string& nm_label = std::get<2>(tup);
            subblock_labels.push_back("edge_" + nm_label);
        }
        for (const auto& tup : m_classifier.FacePairs())
        {
            const std::string& axis_str = std::get<0>(tup);
            if (active_axes.find(axis_str) == active_axes.end())
            {
                continue;
            }
            const std::string& mortar_label = std::get<1>(tup);
            subblock_labels.push_back("face_" + mortar_label);
        }
    }

    // Empty-row early exit (the walk below is a no-op anyway, but this
    // saves an unnecessary classifier traversal on degenerate filter
    // configurations).
    if (n_local == 0)
    {
        return;
    }

    //--------------------------------------------------------------------------
    // Walk rows in EmitConstraintTriples order, assigning sub-block IDs.
    //--------------------------------------------------------------------------
    int row_idx = 0;
    int per_pair_sb_next = 0;   // running ID for PerPair partition

    //--- Edge mortar blocks ---
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& axis_str = std::get<0>(tup);

        const auto perps = EdgePerpendicularAxes(axis_str);
        if (active_axes.find(perps[0]) == active_axes.end()
            || active_axes.find(perps[1]) == active_axes.end())
        {
            continue;
        }

        const std::string& nm_label    = std::get<2>(tup);
        const EdgeInfo3D& nonmortar_edge =
            m_classifier.Edges().at(nm_label);

        // Sub-block ID for this edge pair.
        const int sb_id = (partition == SubblockPartition::FaceEdge)
                          ? 0
                          : per_pair_sb_next++;

        const int n_nm = nonmortar_edge.NumNodes();
        for (int k = 0; k < n_nm; ++k)
        {
            // Row-owner filter on the x-component nonmortar gtdof.
            // Off-rank: skip entirely (no row_idx advance), matching
            // ScatterEdgeBlock's behavior.
            const int g_n_x = nonmortar_edge.gtdofs_x[k];
            const int owner = (g_n_x >= 0)
                              ? ParentOwnerRankFromClassifierX(g_n_x) : -1;
            if (owner != my_rank) { continue; }

            // Owned: emit n_comps_a IDs (one per active component).
            // D_kk == 0 vs nonzero doesn't matter for ROW emission —
            // both branches advance row_offset by n_comps_a in
            // ScatterEdgeBlock; we match that.
            for (int c = 0; c < n_comps_a; ++c)
            {
                subblock_of_row[row_idx++] = sb_id;
            }
        }
    }

    //--- Face mortar blocks ---
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis_str = std::get<0>(tup);
        if (active_axes.find(axis_str) == active_axes.end())
        {
            continue;
        }

        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const int sb_id = (partition == SubblockPartition::FaceEdge)
                          ? 1
                          : per_pair_sb_next++;

        // Find quad and tri blocks for this pair; emit in quad-then-
        // tri order to match EmitConstraintTriples' ScatterFaceBlock
        // calls.
        const FaceMortarPairBlock* quad_block = nullptr;
        const FaceMortarPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair       != axis_str
                || lpb.mortar_label    != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if      (lpb.geometry_kind == "quad") { quad_block = &lpb.block; }
            else if (lpb.geometry_kind == "tri")  { tri_block  = &lpb.block; }
        }

        auto emit_for_face_block = [&](const FaceMortarPairBlock& blk)
        {
            const int n_nm = blk.NumNonmortarKept();
            for (int k = 0; k < n_nm; ++k)
            {
                // Face blocks are pre-routed to row owners by the
                // classifier — no off-rank skip needed here, matching
                // ScatterFaceBlock.
                for (int c = 0; c < n_comps_a; ++c)
                {
                    subblock_of_row[row_idx++] = sb_id;
                }
            }
        };

        if (quad_block != nullptr) { emit_for_face_block(*quad_block); }
        if (tri_block  != nullptr) { emit_for_face_block(*tri_block);  }
    }

    MFEM_VERIFY(row_idx == n_local,
                "ConstraintBuilder3D::GetRowSubblockIds: emitted row "
                "count (" << row_idx << ") does not match NumLocalRows "
                "(" << n_local << "). Walk-order divergence from "
                "EmitConstraintTriples / EmitRowFactors.");
}

//==============================================================================
// BuildHypreParMatrix — parameter-less forwarder (pre-5.9 behavior)
//==============================================================================

mfem::HypreParMatrix* ConstraintBuilder3D::BuildHypreParMatrix() const
{
    return BuildHypreParMatrix(AllMortarLabels(m_classifier),
                               {true, true, true});
}

//==============================================================================
// BuildHypreParMatrix — Phase 5.9 filtered, distributed form
//==============================================================================

mfem::HypreParMatrix* ConstraintBuilder3D::BuildHypreParMatrix(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>& comp_mask) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::build_hypre");

    // Phase 4.2 / Batch N: row partition is FES-aligned. Each rank's
    // n_lam_local is determined by the data — the count of rows
    // EmitConstraintTriples emits on this rank, which (post-Batch-N)
    // equals the sum of:
    //   - edge mortar rows with x-component nonmortar gtdof owned
    //     by this rank in FES, and
    //   - face mortar rows present in m_classifier.PairBlocks()
    //     (already pre-routed by RoutePairBlocksToRowOwners).
    //
    // The caller no longer chooses n_lam_local; that info is exposed
    // separately via NumLocalRows() if needed downstream.
    //
    // Phase 5.9 — under filter, n_lam_local reflects only the active
    // rows (active pair labels × active components).

    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;
    const int n_lam_local   = EmitConstraintTriples(
        active_pair_labels, comp_mask, rows, cols, vals);
    const int n_global_cols = ParentGlobalTrueVSize();

    MPI_Comm comm = m_classifier.Comm();
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Gather per-rank row counts to build the row partition.
    std::vector<int> all_n_lam(nranks, 0);
    MPI_Allgather(&n_lam_local, 1, MPI_INT,
                  all_n_lam.data(), 1, MPI_INT, comm);

    // Sum to get global row count.
    int n_global_rows = 0;
    for (int r = 0; r < nranks; ++r) { n_global_rows += all_n_lam[r]; }

    // Hypre row_starts: 2 entries (begin, end) on this rank.
    std::vector<HYPRE_BigInt> row_starts(2);
    HYPRE_BigInt acc = 0;
    for (int r = 0; r < rank; ++r) { acc += all_n_lam[r]; }
    row_starts[0] = acc;
    row_starts[1] = acc + n_lam_local;

    // Column partition: MUST match the FES's true-DOF partition
    // (§P4.8.9). For C·u to be valid as a parallel matvec where u
    // lives in the FES TDOF space (the layout K's rows use), C's
    // columns must be partitioned IDENTICALLY to K's rows — i.e.,
    // according to the FES's TDOF offsets, which come from METIS
    // partitioning of the mesh and are NOT a uniform chunk split.
    HYPRE_BigInt* fes_tdof_offsets = ParentFes().GetTrueDofOffsets();
    std::vector<HYPRE_BigInt> col_starts(2);
    col_starts[0] = fes_tdof_offsets[0];
    col_starts[1] = fes_tdof_offsets[1];

    // Sanity-check: this rank's local FES TDOF count must equal
    // (col_starts[1] - col_starts[0]).
    {
        const int n_loc_fes = ParentFes().GetTrueVSize();
        const int n_loc_col = static_cast<int>(col_starts[1] - col_starts[0]);
        MFEM_VERIFY(n_loc_fes == n_loc_col,
                    "ConstraintBuilder3D::BuildHypreParMatrix: FES local "
                    "TDOF count (" << n_loc_fes << ") does not match the "
                    "partition span derived from GetTrueDofOffsets ("
                    << n_loc_col << "). FES partition state inconsistent.");
    }

    // Phase 4.2 / Batch N: triples are already in this rank's local
    // row indexing (EmitConstraintTriples emits only this rank's rows
    // and uses 0-based local row indices via row_offset). No filter
    // step needed; just build the local SparseMatrix directly.
    mfem::SparseMatrix local_block(n_lam_local, n_global_cols);
    const std::size_t n_triples = vals.size();
    for (std::size_t k = 0; k < n_triples; ++k)
    {
        local_block.Add(rows[k], cols[k], vals[k]);
    }
    local_block.Finalize();

    // Construct the HypreParMatrix using the same 9-arg ctor as
    // before (comm, global_rows, global_cols, row_starts, col_starts,
    // CSR I/J/data taken from the local SparseMatrix).
    auto* H = new mfem::HypreParMatrix(
        comm,
        static_cast<HYPRE_BigInt>(n_lam_local),
        static_cast<HYPRE_BigInt>(n_global_rows),
        static_cast<HYPRE_BigInt>(n_global_cols),
        const_cast<int*>(local_block.GetI()),
        const_cast<int*>(local_block.GetJ()),
        const_cast<double*>(local_block.GetData()),
        row_starts.data(),
        col_starts.data());

    // The HypreParMatrix copies the data on construction; local_block
    // can be discarded as it goes out of scope. Caller owns H.
    return H;
}

//==============================================================================
// ScatterEdgeBlock — Phase 5.9 filtered
//
// Append rows for one (block, nonmortar, mortar) triplet, respecting
// the component mask.
//
// Row layout per nonmortar node:
//   - Off-rank skip (owner != my_rank): no rows emitted, row_offset
//     unchanged.
//   - Owned node, D_kk == 0: row_offset advances by
//     CountActiveComps(comp_mask) to preserve the per-node stride.
//   - Owned node, D_kk != 0: emit diagonal D entries and off-diagonal
//     -A_m entries for each active component, then advance row_offset
//     by CountActiveComps(comp_mask).
//==============================================================================

int ConstraintBuilder3D::ScatterEdgeBlock(
    const MortarBlock2D& block,
    const EdgeInfo3D& nonmortar_edge,
    const EdgeInfo3D& mortar_edge,
    const std::array<bool, 3>& comp_mask,
    std::vector<int>& rows,
    std::vector<int>& cols,
    std::vector<double>& vals,
    int row_offset) const
{
    const int n_nonmortar = nonmortar_edge.NumNodes();
    const int n_mortar    = mortar_edge.NumNodes();

    MFEM_VERIFY(block.D_nm.Size() == n_nonmortar,
                "ConstraintBuilder3D: edge block D_nm size ("
                << block.D_nm.Size() << ") does not match nonmortar "
                "edge node count (" << n_nonmortar << ")");
    MFEM_VERIFY(block.A_m.NumRows() == n_nonmortar
                && block.A_m.NumCols() == n_mortar,
                "ConstraintBuilder3D: edge block A_m shape ("
                << block.A_m.NumRows() << ", " << block.A_m.NumCols()
                << ") does not match (n_nonmortar, n_mortar) = ("
                << n_nonmortar << ", " << n_mortar << ")");

    // Phase 4.2 / Batch N — filter rows by FES ownership of the
    // x-component nonmortar gtdof. Edge mortars are produced
    // redundantly on every rank (cheap 9 small-dense assemblies),
    // and the row-owner filter makes each rank emit only the rows
    // it owns under the FES TDOF partition.
    //
    // Convention: a constraint row's "owner" is the rank that owns
    // the corresponding nonmortar node's x-component gtdof. This
    // matches RoutePairBlocksToRowOwners (which routes by x gtdof)
    // and ensures all three component rows for a node land on the
    // same rank.
    //
    // At np=1 the filter is trivial (every gtdof is owned by rank 0);
    // the row layout matches Batches K/L exactly.
    const int my_rank   = m_classifier.Rank();
    const int n_comps_a = CountActiveComps(comp_mask);

    for (int k = 0; k < n_nonmortar; ++k)
    {
        const double D_kk = block.D_nm(k);
        const std::array<int, 3> nonmortar_g_xyz = {
            nonmortar_edge.gtdofs_x[k],
            nonmortar_edge.gtdofs_y[k],
            nonmortar_edge.gtdofs_z[k],
        };

        // Row-owner test on the x gtdof. Skip the row entirely if
        // owned by another rank — do NOT increment row_offset, since
        // row_offset counts rows this rank emits (used as the local
        // row index in BuildHypreParMatrix's local_block).
        const int owner =
            (nonmortar_g_xyz[0] >= 0)
            ? ParentOwnerRankFromClassifierX(nonmortar_g_xyz[0])
            : -1;
        if (owner != my_rank) { continue; }

        if (D_kk == 0.0)
        {
            // Degenerate row (could happen if a nonmortar node is
            // entirely covered by a corner-modified element). Skip
            // entry emission but still consume the per-node row
            // indices to keep the layout deterministic. Under filter
            // we advance by n_comps_a (was kVDim pre-5.9).
            row_offset += n_comps_a;
            continue;
        }

        // Diagonal D entry per active spatial component.
        for (int c = 0; c < kVDim; ++c)
        {
            const int local_row = LocalRowOfComp(comp_mask, c);
            if (local_row < 0) { continue; }  // component filtered out
            const int gd = ParentGtdofFromClassifierGtdof(nonmortar_g_xyz[c]);
            if (gd < 0) { continue; }
            rows.push_back(row_offset + local_row);
            cols.push_back(gd);
            vals.push_back(D_kk);
        }

        // Off-diagonal -A_m entries over mortar interior nodes.
        for (int l = 0; l < n_mortar; ++l)
        {
            const double A_kl = block.A_m(k, l);
            if (A_kl == 0.0) { continue; }
            const std::array<int, 3> mortar_g_xyz = {
                mortar_edge.gtdofs_x[l],
                mortar_edge.gtdofs_y[l],
                mortar_edge.gtdofs_z[l],
            };
            for (int c = 0; c < kVDim; ++c)
            {
                const int local_row = LocalRowOfComp(comp_mask, c);
                if (local_row < 0) { continue; }  // component filtered out
                const int gd = ParentGtdofFromClassifierGtdof(mortar_g_xyz[c]);
                if (gd < 0) { continue; }
                rows.push_back(row_offset + local_row);
                cols.push_back(gd);
                vals.push_back(-A_kl);
            }
        }

        row_offset += n_comps_a;
    }

    return row_offset;
}

//==============================================================================
// ScatterFaceBlock — Phase 5.9 filtered
//
// Same per-component row gating as ScatterEdgeBlock; differs in that
// the off-rank filter is not applied here (face pair blocks are
// pre-routed to row owners by the classifier in
// RoutePairBlocksToRowOwners, so every block on this rank IS owned
// by this rank).
//==============================================================================

int ConstraintBuilder3D::ScatterFaceBlock(
    const FaceMortarPairBlock& block,
    const std::array<bool, 3>& comp_mask,
    std::vector<int>& rows,
    std::vector<int>& cols,
    std::vector<double>& vals,
    int row_offset) const
{
    const int n_nonmortar_kept = block.NumNonmortarKept();
    const int n_mortar_kept    = block.NumMortarKept();

    MFEM_VERIFY(block.D.Size() == n_nonmortar_kept,
                "ConstraintBuilder3D: face block D size ("
                << block.D.Size() << ") does not match "
                "n_nonmortar_kept (" << n_nonmortar_kept << ")");
    MFEM_VERIFY(block.A_m.NumRows() == n_nonmortar_kept
                && block.A_m.NumCols() == n_mortar_kept,
                "ConstraintBuilder3D: face block A_m shape ("
                << block.A_m.NumRows() << ", " << block.A_m.NumCols()
                << ") does not match (kept_nonmortar, kept_mortar) = ("
                << n_nonmortar_kept << ", " << n_mortar_kept << ")");

    // Phase 4.2 / Batch L: A_m is now sparse (mfem::SparseMatrix).
    // Walk it via its CSR arrays rather than `(k, l)` indexing —
    // the per-element `operator()` does a binary search per call,
    // which would be O(nnz_per_row * n_mortar_kept) total. The CSR
    // walk is O(nnz) total.
    const int* A_I    = block.A_m.GetI();
    const int* A_J    = block.A_m.GetJ();
    const double* A_V = block.A_m.GetData();

    const int n_comps_a = CountActiveComps(comp_mask);

    for (int k = 0; k < n_nonmortar_kept; ++k)
    {
        const double D_kk = block.D(k);
        const int nonmortar_gx = block.nonmortar_gtdofs[k];

        const std::array<int, 3> nonmortar_g_xyz =
            ParentGtdofXyzFromClassifierX(nonmortar_gx);

        if (D_kk == 0.0)
        {
            row_offset += n_comps_a;
            continue;
        }

        // Diagonal D entries — active components only.
        for (int c = 0; c < kVDim; ++c)
        {
            const int local_row = LocalRowOfComp(comp_mask, c);
            if (local_row < 0) { continue; }  // component filtered out
            const int gd = nonmortar_g_xyz[c];
            if (gd < 0) { continue; }
            rows.push_back(row_offset + local_row);
            cols.push_back(gd);
            vals.push_back(D_kk);
        }

        // Off-diagonal -A_m entries — CSR row walk, active components only.
        for (int idx = A_I[k]; idx < A_I[k + 1]; ++idx)
        {
            const int l = A_J[idx];
            const double A_kl = A_V[idx];
            if (A_kl == 0.0) { continue; }
            const int mortar_gx = block.mortar_gtdofs[l];
            const std::array<int, 3> mortar_g_xyz =
                ParentGtdofXyzFromClassifierX(mortar_gx);
            for (int c = 0; c < kVDim; ++c)
            {
                const int local_row = LocalRowOfComp(comp_mask, c);
                if (local_row < 0) { continue; }  // component filtered out
                const int gd = mortar_g_xyz[c];
                if (gd < 0) { continue; }
                rows.push_back(row_offset + local_row);
                cols.push_back(gd);
                vals.push_back(-A_kl);
            }
        }

        row_offset += n_comps_a;
    }

    return row_offset;
}

}  // namespace mortar_pbc
