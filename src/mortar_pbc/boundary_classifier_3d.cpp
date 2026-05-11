// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of BoundaryClassifier3D, ported from
// `mortar_pbc/boundary_3d.py`. See header for design doc.

#include "boundary_classifier_3d.hpp"

#include "boundary_helpers_3d.hpp"
#include "face_mortar_assembler_3d.hpp"
#include "types_3d.hpp"

#ifdef MORTAR_PBC_HAS_AXOM
// Phase 4.4 / Batch 4.4-E — clipped-path fallback for non-conforming
// face mortar pairs. Headers only included when Axom is available; the
// dispatch in BuildLocalPairBlocks below conditionally uses them.
#include "face_mortar_match_3d.hpp"
#include "face_mortar_assembler_clipped_3d.hpp"
#endif

#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mortar_pbc {

//==============================================================================
// Internal record types (implementation detail; not exposed in the header).
//==============================================================================

/// One unique boundary vertex, post Allgatherv-merge.
///
/// The `parent_attrs` set has cardinality 1, 2, or 3:
///   - 1 -> face-interior vertex (no shared box edge or corner)
///   - 2 -> box-edge vertex (sits on two faces' shared edge)
///   - 3 -> box-corner vertex (sits on three faces' shared corner)
///
/// `synth_id` is a stable index into m_vertex_records, assigned during
/// the gather/merge step and used as a synthetic global vertex
/// identifier downstream (the actual ParMesh vertex index is rank-
/// local and meaningless globally).
struct BoundaryClassifier3D::VertexRecord
{
    int synth_id = -1;
    std::array<double, 3> coord = {0.0, 0.0, 0.0};
    std::array<int, 3> gtdof_xyz = {-1, -1, -1};
    // Sorted, deduplicated attribute list. Size 1, 2, or 3.
    std::vector<int> parent_attrs;
};

// Note: the FaceElementRecord struct has been removed in Phase 4.2 /
// Batch J. Face elements no longer flow through the global AllGather
// (they travel via TileShuffleFaceElements on the boundary subcomm
// instead). The per-pair mortar blocks are produced tile-locally by
// BuildLocalPairBlocks; the constraint builder consumes them via
// PairBlocks(). Face-element diagnostics that were once read from
// m_face_element_records are now read from m_tile_shuffled_face_elements
// (per-rank tile slice; full set at np=1).

namespace {

//==============================================================================
// Snap-coord helpers
//==============================================================================
//
// Cross-rank vertex identity uses snapped physical coordinates as the
// global key. Each (x, y, z) is snapped to integer multiples of the
// classifier's `tol`; vertices snapping to the same triple are
// "the same" vertex regardless of rank-local ParMesh indices.
//
// Architecture: §11.7.1 (cross-rank keying).

inline std::array<long long, 3> SnapKey(double x, double y, double z, double snap_unit)
{
    auto rnd = [snap_unit](double v) -> long long
    {
        return static_cast<long long>(std::llround(v / snap_unit));
    };
    return {rnd(x), rnd(y), rnd(z)};
}

inline int AxisIdx(const std::string& axis)
{
    if (axis == "x") { return 0; }
    if (axis == "y") { return 1; }
    if (axis == "z") { return 2; }
    MFEM_ABORT("AxisIdx: unknown axis '" << axis << "'");
    return -1;
}

}  // anonymous namespace

//==============================================================================
// Constructor — orchestrates the Python __init__ flow
//==============================================================================

BoundaryClassifier3D::BoundaryClassifier3D(mfem::ParMesh& pmesh,
                                           mfem::ParFiniteElementSpace& fes,
                                           double tol_rel,
                                           double pair_match_tol_rel)
    : m_pmesh(pmesh)
    , m_fes(fes)
    , m_comm(pmesh.GetComm())
    , m_tol_rel(tol_rel)
    , m_pair_match_tol_rel(pair_match_tol_rel)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::ctor");

    MFEM_VERIFY(m_pmesh.Dimension() == 3,
                "BoundaryClassifier3D: requires a 3D mesh (got dim "
                << m_pmesh.Dimension() << ")");
    MFEM_VERIFY(m_fes.GetVDim() == 3,
                "BoundaryClassifier3D: expected vector FE space with vdim=3, "
                "got vdim=" << m_fes.GetVDim());
    MFEM_VERIFY(m_fes.GetOrder(0) == 1,
                "BoundaryClassifier3D: order-1 H1 only (Phase 4 scope); got "
                "order " << m_fes.GetOrder(0));

    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_nranks);

    // Boundary subcomm (Phase 4.2 §P4.4.0): split off the ranks that
    // actually own boundary elements on the parent ParMesh. This is
    // a WORLD-collective `MPI_Comm_split`; interior ranks pass color =
    // MPI_UNDEFINED and receive `MPI_COMM_NULL`. Boundary ranks pass
    // color = 0 and join the new comm.
    //
    // The Phase 4.1 internals (face-element AllGatherv) still run on
    // `m_comm` for now; Phase 4.2's tile-partitioned shuffle (Batch H)
    // will move them to `m_boundary_comm`. This batch (G) is purely
    // additive — it creates the subcomm so subsequent batches can use
    // it.
    {
        const bool has_boundary = (m_pmesh.GetNBE() > 0);
        const int color = has_boundary ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(m_comm, color, m_rank, &m_boundary_comm);
        if (m_boundary_comm != MPI_COMM_NULL)
        {
            MPI_Comm_rank(m_boundary_comm, &m_bdy_rank);
            MPI_Comm_size(m_boundary_comm, &m_n_bdy_ranks);
        }
    }

    // Cache global TDOF count once — every rank knows its own value
    // without a fresh collective at access time.
    m_n_global_tdofs = m_fes.GlobalTrueVSize();

    // Phase 4.2 / Batch N — Allgather every rank's FES TDOF starting
    // offset so we can answer GtdofOwnerRank() locally via binary
    // search. Layout: m_fes_tdof_offsets_all[r] = first global TDOF
    // owned by rank r; m_fes_tdof_offsets_all[m_nranks] = total
    // (sentinel). FES.GetTrueDofOffsets() returns a 2-element local
    // [start, end) array; we Allgather the start values and append
    // the global total as a sentinel.
    //
    // CRITICAL: use HYPRE_MPI_BIG_INT (defined by HYPRE) as the MPI
    // datatype, NOT a hardcoded MPI_LONG_LONG. HYPRE_BigInt resolves
    // to either `int` or `long long` depending on the HYPRE build's
    // --enable-bigint flag. Hardcoding the wrong width corrupts the
    // Allgather: the send buffer is `sizeof(HYPRE_BigInt)` bytes per
    // element but MPI reads/writes `sizeof(MPI_LONG_LONG) == 8` bytes.
    // Most production HYPRE builds (including ExaConstit's) keep the
    // default `int` width, so this would manifest as a corrupted
    // monotone-check failure with garbage values like "108 -> 0".
    {
        const HYPRE_BigInt my_start =
            m_fes.GetTrueDofOffsets()[0];
        m_fes_tdof_offsets_all.assign(
            static_cast<std::size_t>(m_nranks + 1), 0);
        MPI_Allgather(&my_start, 1, HYPRE_MPI_BIG_INT,
                      m_fes_tdof_offsets_all.data(), 1,
                      HYPRE_MPI_BIG_INT, m_comm);
        m_fes_tdof_offsets_all[m_nranks] =
            static_cast<HYPRE_BigInt>(m_n_global_tdofs);
        // Sanity: offsets must be monotonically non-decreasing.
        for (int r = 1; r <= m_nranks; ++r)
        {
            MFEM_VERIFY(
                m_fes_tdof_offsets_all[r] >= m_fes_tdof_offsets_all[r - 1],
                "BoundaryClassifier3D: Allgather'd FES TDOF offsets are "
                "not monotone at rank " << r << " ("
                << m_fes_tdof_offsets_all[r - 1] << " -> "
                << m_fes_tdof_offsets_all[r] << "). FES partition is "
                "inconsistent across ranks.");
        }
    }

    // Step 1: bbox + tolerance (collective)
    ComputeBbox();
    {
        const double dx = m_bbox_max[0] - m_bbox_min[0];
        const double dy = m_bbox_max[1] - m_bbox_min[1];
        const double dz = m_bbox_max[2] - m_bbox_min[2];
        const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
        m_tol = m_tol_rel * diag;
        MFEM_VERIFY(m_tol > 0.0,
                    "BoundaryClassifier3D: bbox diagonal evaluated to "
                    << diag << "; cannot proceed.");
    }

    // Step 1b: discover MFEM's attribute -> face-label mapping (collective).
    DiscoverFaceLabelByAttr();
    for (const auto& kv : m_face_label_by_attr)
    {
        m_face_attr_by_label[kv.second] = kv.first;
    }

    // Step 2: build the boundary ParSubMesh (collective).
    BuildBoundarySubmesh();

    // Step 2b (Phase 4.2 / Batch H): build the deterministic tile
    // partition. Only on boundary ranks — interior ranks have no
    // boundary work to do and don't need it. The TilePartition3D is
    // pure arithmetic (no MPI), but every boundary rank constructs an
    // identical instance so OwnerRank() lookups agree across the
    // subcomm.
    if (IsBoundaryRank())
    {
        m_tile_partition.reset(new TilePartition3D(
            m_bbox_min, m_bbox_max, m_n_bdy_ranks));
    }

    // Step 3: gather per-rank boundary records, AllGather, dedup. (collective)
    GatherBoundaryRecords();

    // Step 3b (Phase 4.2 / Batch H): tile-shuffle local face elements
    // on the boundary subcomm in parallel with the AllGather path.
    // Both data streams coexist for now; downstream consumers
    // (BuildFaces, ConstraintBuilder) still read the AllGather'd
    // catalogue. Batch I will switch them to the tile-shuffled path
    // and decommission the global AllGather.
    if (IsBoundaryRank())
    {
        TileShuffleFaceElements();
    }

    // Step 4: classify vertices into corners / edges / faces (local).
    BuildCorners();
    BuildEdges();
    BuildFaces();

    // Step 5 (Phase 4.2 / Batch I): assemble per-pair mortar blocks
    // tile-locally, then AllGatherv them across WORLD so every rank
    // (boundary or interior) has the full set. The constraint
    // builder (refactored in this same batch) consumes these blocks
    // instead of running its own matching against the AllGather'd
    // face element list.
    //
    // Note ordering: GatherBoundaryRecords (step 3) must run before
    // BuildLocalPairBlocks because the latter needs vertex gtdofs
    // (via m_snap_key_to_record_idx → m_vertex_records).
    //
    // The AllGather happens on m_comm (WORLD) — see
    // GatherPairBlocksAcrossBoundary docstring. Interior ranks
    // contribute zero blocks but must participate in the collective
    // to receive the complete set.
    if (IsBoundaryRank())
    {
        BuildLocalPairBlocks();
    }
    RoutePairBlocksToRowOwners();
}

// Out-of-line destructor: VertexRecord is forward-declared in the
// header but defined in this .cpp. Defaulting the destructor here
// ensures the std::vector<VertexRecord> member destructs with the
// complete type in scope.
//
// Also responsible for freeing `m_boundary_comm` if non-null.
BoundaryClassifier3D::~BoundaryClassifier3D()
{
    if (m_boundary_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&m_boundary_comm);
    }
}

//==============================================================================
// Step 1 — bbox via Allreduce
//==============================================================================

void BoundaryClassifier3D::ComputeBbox()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::compute_bbox");

    double local_min[3] = {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity()};
    double local_max[3] = {-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};

    const int nv = m_pmesh.GetNV();
    for (int v = 0; v < nv; ++v)
    {
        const double* xyz = m_pmesh.GetVertex(v);
        for (int d = 0; d < 3; ++d)
        {
            local_min[d] = std::min(local_min[d], xyz[d]);
            local_max[d] = std::max(local_max[d], xyz[d]);
        }
    }

    double global_min[3];
    double global_max[3];
    MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, m_comm);
    MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, m_comm);

    for (int d = 0; d < 3; ++d)
    {
        m_bbox_min[d] = global_min[d];
        m_bbox_max[d] = global_max[d];
    }
}

//==============================================================================
// Step 1b — runtime discovery of MFEM's attribute-to-label mapping
//
// For each boundary attribute 1..n_attrs, find one parent boundary
// element with that attribute, read its vertex coords, determine
// which axis is invariant (zero spread) and at which extreme
// (matching bbox_min vs bbox_max), then look up the canonical label
// via AxisExtremeToLabel().
//
// Discovery is collective-free locally (every rank scans its own
// boundary elements); we use Allgather to build a consistent global
// view since not every rank owns elements with every attribute. This
// lets us also catch the "two ranks discover different labels for the
// same attribute" failure mode.
//==============================================================================

void BoundaryClassifier3D::DiscoverFaceLabelByAttr()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::discover_face_labels");

    MFEM_VERIFY(m_pmesh.bdr_attributes.Size() > 0,
                "BoundaryClassifier3D: parent ParMesh has no boundary "
                "attributes. The mesh must have boundary elements with "
                "attributes 1..6 covering all 6 RVE faces.");
    const int n_attrs = m_pmesh.bdr_attributes.Max();

    // Per-rank findings: attr -> (axis_idx, is_min) packed into one int per
    // attr. Encoding: 0..2 = axis index for "min" extreme; 3..5 = axis
    // index + 3 for "max" extreme; -1 = not found on this rank.
    //
    // Allgather a fixed-size array per rank: indices 1..n_attrs (we
    // skip slot 0 to keep attribute numbering 1-based).
    std::vector<int> local_findings(n_attrs + 1, -1);

    const int nbe = m_pmesh.GetNBE();
    for (int be = 0; be < nbe; ++be)
    {
        const int attr = m_pmesh.GetBdrAttribute(be);
        MFEM_VERIFY(attr >= 1 && attr <= n_attrs,
                    "BoundaryClassifier3D: bdr element " << be
                    << " has attribute " << attr
                    << " outside the declared range 1.." << n_attrs);
        if (local_findings[attr] >= 0) { continue; }  // already found

        mfem::Array<int> verts;
        m_pmesh.GetBdrElementVertices(be, verts);
        const int nv = verts.Size();
        MFEM_VERIFY(nv == 3 || nv == 4,
                    "BoundaryClassifier3D: bdr element " << be
                    << " has " << nv << " vertices (expected 3 or 4)");

        // Compute per-axis min/max over this element's vertices.
        double v_min[3] = { std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
        double v_max[3] = {-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
        double v_sum[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < nv; ++k)
        {
            const double* xyz = m_pmesh.GetVertex(verts[k]);
            for (int d = 0; d < 3; ++d)
            {
                v_min[d] = std::min(v_min[d], xyz[d]);
                v_max[d] = std::max(v_max[d], xyz[d]);
                v_sum[d] += xyz[d];
            }
        }
        const double v_mean[3] = {v_sum[0] / nv, v_sum[1] / nv, v_sum[2] / nv};
        const double spread[3] = {v_max[0] - v_min[0],
                                  v_max[1] - v_min[1],
                                  v_max[2] - v_min[2]};

        // Invariant axis: the one with smallest spread.
        int invariant_axis = 0;
        if (spread[1] < spread[invariant_axis]) { invariant_axis = 1; }
        if (spread[2] < spread[invariant_axis]) { invariant_axis = 2; }

        // Sanity: invariant-axis spread must be within tolerance.
        MFEM_VERIFY(spread[invariant_axis] <= m_tol,
                    "BoundaryClassifier3D: bdr attr " << attr
                    << " is not axis-aligned. Invariant-axis ("
                    << "xyz"[invariant_axis] << ") spread = "
                    << spread[invariant_axis] << ", tol = " << m_tol
                    << ". Phase 4 supports axis-aligned RVE boundaries only.");

        // Determine extreme by comparing invariant-axis mean to bbox.
        const double inv_val = v_mean[invariant_axis];
        const double d_min = std::abs(inv_val - m_bbox_min[invariant_axis]);
        const double d_max = std::abs(inv_val - m_bbox_max[invariant_axis]);
        const bool is_min = (d_min < d_max);
        // Encoding: 0..2 = (axis, min); 3..5 = (axis, max).
        local_findings[attr] = invariant_axis + (is_min ? 0 : 3);
    }

    // Allgather across ranks; consistency-check every (attr -> finding).
    std::vector<int> all_findings(static_cast<std::size_t>(n_attrs + 1)
                                  * static_cast<std::size_t>(m_nranks), -1);
    MPI_Allgather(local_findings.data(), n_attrs + 1, MPI_INT,
                  all_findings.data(),  n_attrs + 1, MPI_INT, m_comm);

    std::vector<int> merged(n_attrs + 1, -1);
    for (int r = 0; r < m_nranks; ++r)
    {
        for (int attr = 1; attr <= n_attrs; ++attr)
        {
            const int f = all_findings[r * (n_attrs + 1) + attr];
            if (f < 0) { continue; }
            if (merged[attr] >= 0)
            {
                MFEM_VERIFY(merged[attr] == f,
                            "BoundaryClassifier3D: inconsistent face-label "
                            "discovery for attr " << attr << ": encoding "
                            << merged[attr] << " vs " << f
                            << " on different ranks.");
            }
            else
            {
                merged[attr] = f;
            }
        }
    }

    // Map findings to canonical labels.
    std::set<std::string> seen_labels;
    for (int attr = 1; attr <= n_attrs; ++attr)
    {
        const int f = merged[attr];
        MFEM_VERIFY(f >= 0,
                    "BoundaryClassifier3D: no rank found a boundary element "
                    "with attribute " << attr
                    << ". The mesh must have at least one boundary element "
                    "per attribute 1.." << n_attrs);
        const int axis = f % 3;
        const bool is_min = (f / 3 == 0);
        const std::string ax_name(1, "xyz"[axis]);
        const std::string extreme = is_min ? "min" : "max";
        const std::string label = AxisExtremeToLabel(ax_name, extreme);
        MFEM_VERIFY(seen_labels.find(label) == seen_labels.end(),
                    "BoundaryClassifier3D: two attributes map to the same "
                    "label '" << label << "'. Discovery inconsistent.");
        seen_labels.insert(label);
        m_face_label_by_attr[attr] = label;
    }
}

//==============================================================================
// Step 2 — boundary ParSubMesh
//==============================================================================

void BoundaryClassifier3D::BuildBoundarySubmesh()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::build_submesh");

    const int n_attrs = m_pmesh.bdr_attributes.Max();
    // ParSubMesh::CreateFromBoundary expects an Array<int> whose
    // CONTENTS are the actual attribute values, NOT a boolean mask.
    // (Robert's macOS pyMFEM debugging note from the Python
    // prototype: a [1,1,1,1,1,1] mask was misinterpreted as "select
    // attribute 1, six times" and returned only the bottom face.)
    mfem::Array<int> bdr_attrs(n_attrs);
    for (int a = 0; a < n_attrs; ++a) { bdr_attrs[a] = a + 1; }

    m_bdr_submesh.reset(new mfem::ParSubMesh(
        mfem::ParSubMesh::CreateFromBoundary(m_pmesh, bdr_attrs)));
}

//==============================================================================
// Step 3 — gather per-rank boundary records, AllGather, dedup
//
// Why snap-coord keying, not parent_vertex_id keying
// ---------------------------------------------------
// ParMesh's vertex indices are RANK-LOCAL: vertex 27 on rank 0 is
// unrelated to vertex 27 on rank 1. AllGather'ing records keyed by
// parent_vertex_id therefore collides across ranks and produces
// nonsense merges. We snap physical coordinates to a tolerance grid
// (`round(x / tol)`) and use the snapped tuple as the global key.
//
// Per-rank pack layout (fixed-width, fits cleanly in MPI_Allgatherv):
//
//   Vertex int pack:  10 int64s per vertex =
//       [snap_kx, snap_ky, snap_kz,
//        gtdof_x, gtdof_y, gtdof_z,
//        attr1, attr2, attr3, _pad]
//     attr2/attr3 = -1 if unused (vertex on fewer than 2/3 faces).
//   Vertex double pack: 3 doubles per vertex = [x, y, z]
//
//   Face element packs are split by geometry into separate streams
//   for fixed-width handling:
//     Quad int pack:    13 int64s per quad =
//         [parent_attr,
//          snap_kx_v0, snap_ky_v0, snap_kz_v0,  ... (4 verts × 3 keys)]
//     Quad double pack: 12 doubles per quad (4 × 3 coords)
//     Tri int pack:     10 int64s per tri  (1 + 3 × 3)
//     Tri double pack:   9 doubles per tri  (3 × 3)
//
// All four streams go through MPI_Allgatherv; merging happens locally.
//==============================================================================

namespace {

// Vertex int-pack stride (per-vertex layout in GatherBoundaryRecords).
// Phase 4.2 / Batch J: the kQPack* / kTPack* face-element packs are gone;
// face elements are no longer AllGather'd globally — they reach their
// destination via the per-rank tile-shuffle (see TileShuffleFaceElements).
constexpr int kVPackInts    = 10;
constexpr int kVPackDoubles = 3;

}  // anonymous namespace

void BoundaryClassifier3D::GatherBoundaryRecords()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::gather_records");

    mfem::ParSubMesh& sub = *m_bdr_submesh;
    const mfem::Array<int>& parent_vmap = sub.GetParentVertexIDMap();
    const mfem::Array<int>& parent_emap = sub.GetParentElementIDMap();

    // ---------- Local vertex pass ----------
    //
    // Build a snap_key -> {coord, attr_set, gtdof_xyz} map locally by
    // walking the boundary submesh elements (each element's vertices
    // tally their parent-attr set and TDOF triple). We re-key from
    // snap_key to a flat int-pack at the end. No face-element data
    // is accumulated here — Phase 4.2 / Batch J: face elements
    // travel via TileShuffleFaceElements on the boundary subcomm,
    // not via this AllGather.
    struct LocalVertexData
    {
        std::array<double, 3> coord = {0.0, 0.0, 0.0};
        std::set<int> attrs;
        std::array<int, 3> gtdofs = {-1, -1, -1};
    };
    std::map<std::array<long long, 3>, LocalVertexData> local_verts;

    const int n_sub_elems = sub.GetNE();
    for (int se = 0; se < n_sub_elems; ++se)
    {
        const int parent_be = parent_emap[se];
        const int parent_attr = m_pmesh.GetBdrAttribute(parent_be);

        mfem::Array<int> sub_verts;
        sub.GetElementVertices(se, sub_verts);
        const int n_verts = sub_verts.Size();
        MFEM_VERIFY(n_verts == 3 || n_verts == 4,
                    "BoundaryClassifier3D: face element with " << n_verts
                    << " vertices (expected 3 or 4)");

        for (int k = 0; k < n_verts; ++k)
        {
            const int parent_v = parent_vmap[sub_verts[k]];
            const double* xyz = m_pmesh.GetVertex(parent_v);
            const auto key = SnapKey(xyz[0], xyz[1], xyz[2], m_tol);

            // Tally vertex.
            auto it = local_verts.find(key);
            if (it == local_verts.end())
            {
                LocalVertexData lvd;
                for (int d = 0; d < 3; ++d) { lvd.coord[d] = xyz[d]; }
                lvd.attrs.insert(parent_attr);

                // Look up TDOFs via the parent FES.
                mfem::Array<int> scalar_ldofs;
                m_fes.GetVertexDofs(parent_v, scalar_ldofs);
                if (scalar_ldofs.Size() > 0)
                {
                    const int s_ldof = scalar_ldofs[0];
                    for (int c = 0; c < 3; ++c)
                    {
                        const int comp_ldof = m_fes.DofToVDof(s_ldof, c);
                        if (comp_ldof >= 0)
                        {
                            const int g = m_fes.GetGlobalTDofNumber(comp_ldof);
                            if (g >= 0) { lvd.gtdofs[c] = g; }
                        }
                    }
                }
                local_verts[key] = lvd;
            }
            else
            {
                it->second.attrs.insert(parent_attr);
            }
        }
    }

    // ---------- Pack local arrays for Allgatherv ----------
    //
    // Vertex pack: kVPackInts ints + kVPackDoubles doubles per vertex.
    // We need separate int / double Allgatherv calls because MPI
    // doesn't have a native heterogeneous gather.
    const int n_local_verts = static_cast<int>(local_verts.size());
    std::vector<long long> v_int_pack(n_local_verts * kVPackInts);
    std::vector<double>    v_dbl_pack(n_local_verts * kVPackDoubles);
    {
        int idx = 0;
        for (const auto& kv : local_verts)
        {
            const auto& key = kv.first;
            const auto& lvd = kv.second;
            long long* slot = v_int_pack.data() + idx * kVPackInts;
            slot[0] = key[0];
            slot[1] = key[1];
            slot[2] = key[2];
            slot[3] = lvd.gtdofs[0];
            slot[4] = lvd.gtdofs[1];
            slot[5] = lvd.gtdofs[2];
            // Up to 3 attrs, padded with -1.
            int a_idx = 0;
            for (int a : lvd.attrs)
            {
                if (a_idx >= 3) { break; }
                slot[6 + a_idx++] = a;
            }
            for (; a_idx < 3; ++a_idx) { slot[6 + a_idx] = -1; }
            slot[9] = 0;  // _pad
            v_dbl_pack[idx * 3 + 0] = lvd.coord[0];
            v_dbl_pack[idx * 3 + 1] = lvd.coord[1];
            v_dbl_pack[idx * 3 + 2] = lvd.coord[2];
            ++idx;
        }
    }

    // Face-element packs are gone — see Phase 4.2 / Batch J. Tile-shuffle
    // (TileShuffleFaceElements) handles face-element distribution
    // separately, on m_boundary_comm. The vertex pack continues
    // through the existing AllGatherv path below.

    // ---------- Allgatherv vertex pack ----------
    //
    // For each pack: gather counts (Allgather), build displacements
    // and recv-counts (in element units, then in MPI scalar units),
    // resize global buffer, Allgatherv.
    auto gather_long = [&](const std::vector<long long>& local,
                           int stride_per_elem,
                           std::vector<long long>& global) -> int /* total elems */
    {
        const int n_local_elems = static_cast<int>(local.size()) / stride_per_elem;
        std::vector<int> all_counts(m_nranks, 0);
        MPI_Allgather(&n_local_elems, 1, MPI_INT,
                      all_counts.data(), 1, MPI_INT, m_comm);
        int total_elems = 0;
        std::vector<int> recv_counts(m_nranks);
        std::vector<int> displs(m_nranks);
        for (int r = 0; r < m_nranks; ++r)
        {
            displs[r] = total_elems * stride_per_elem;
            recv_counts[r] = all_counts[r] * stride_per_elem;
            total_elems += all_counts[r];
        }
        global.assign(static_cast<std::size_t>(total_elems) * stride_per_elem, 0);
        MPI_Allgatherv(local.data(), n_local_elems * stride_per_elem,
                       MPI_LONG_LONG,
                       global.data(), recv_counts.data(), displs.data(),
                       MPI_LONG_LONG, m_comm);
        return total_elems;
    };
    auto gather_double = [&](const std::vector<double>& local,
                             int stride_per_elem,
                             std::vector<double>& global) -> int
    {
        const int n_local_elems = static_cast<int>(local.size()) / stride_per_elem;
        std::vector<int> all_counts(m_nranks, 0);
        MPI_Allgather(&n_local_elems, 1, MPI_INT,
                      all_counts.data(), 1, MPI_INT, m_comm);
        int total_elems = 0;
        std::vector<int> recv_counts(m_nranks);
        std::vector<int> displs(m_nranks);
        for (int r = 0; r < m_nranks; ++r)
        {
            displs[r] = total_elems * stride_per_elem;
            recv_counts[r] = all_counts[r] * stride_per_elem;
            total_elems += all_counts[r];
        }
        global.assign(static_cast<std::size_t>(total_elems) * stride_per_elem, 0.0);
        MPI_Allgatherv(local.data(), n_local_elems * stride_per_elem, MPI_DOUBLE,
                       global.data(), recv_counts.data(), displs.data(),
                       MPI_DOUBLE, m_comm);
        return total_elems;
    };

    std::vector<long long> v_int_global;
    std::vector<double>    v_dbl_global;
    const int n_v_global = gather_long(v_int_pack, kVPackInts, v_int_global);
    (void)gather_double(v_dbl_pack, kVPackDoubles, v_dbl_global);

    // ---------- Merge vertex records by snap key ----------
    std::map<std::array<long long, 3>, VertexRecord> merged;
    for (int i = 0; i < n_v_global; ++i)
    {
        const long long* islot = v_int_global.data() + i * kVPackInts;
        const double*    dslot = v_dbl_global.data() + i * kVPackDoubles;
        std::array<long long, 3> key = {islot[0], islot[1], islot[2]};

        auto it = merged.find(key);
        if (it == merged.end())
        {
            VertexRecord rec;
            for (int d = 0; d < 3; ++d) { rec.coord[d] = dslot[d]; }
            for (int c = 0; c < 3; ++c)
            {
                rec.gtdof_xyz[c] = static_cast<int>(islot[3 + c]);
            }
            for (int a_idx = 0; a_idx < 3; ++a_idx)
            {
                const long long a = islot[6 + a_idx];
                if (a > 0) { rec.parent_attrs.push_back(static_cast<int>(a)); }
            }
            std::sort(rec.parent_attrs.begin(), rec.parent_attrs.end());
            rec.parent_attrs.erase(
                std::unique(rec.parent_attrs.begin(), rec.parent_attrs.end()),
                rec.parent_attrs.end());
            merged[key] = std::move(rec);
        }
        else
        {
            VertexRecord& rec = it->second;
            // Merge attrs (union of sets).
            for (int a_idx = 0; a_idx < 3; ++a_idx)
            {
                const long long a = islot[6 + a_idx];
                if (a > 0
                    && std::find(rec.parent_attrs.begin(),
                                 rec.parent_attrs.end(),
                                 static_cast<int>(a))
                       == rec.parent_attrs.end())
                {
                    rec.parent_attrs.push_back(static_cast<int>(a));
                }
            }
            std::sort(rec.parent_attrs.begin(), rec.parent_attrs.end());
            // Merge per-component gtdofs (take first positive).
            for (int c = 0; c < 3; ++c)
            {
                if (rec.gtdof_xyz[c] < 0 && islot[3 + c] >= 0)
                {
                    rec.gtdof_xyz[c] = static_cast<int>(islot[3 + c]);
                }
            }
        }
    }

    // Validate that every merged vertex has all 3 gtdofs.
    int n_bad = 0;
    for (auto& kv : merged)
    {
        if (kv.second.gtdof_xyz[0] < 0
            || kv.second.gtdof_xyz[1] < 0
            || kv.second.gtdof_xyz[2] < 0)
        {
            ++n_bad;
        }
    }
    MFEM_VERIFY(n_bad == 0,
                "BoundaryClassifier3D: " << n_bad << " boundary vertex(es) "
                "did not get a TDOF for at least one component across all "
                "ranks. Total merged: " << merged.size());

    // ---------- Convert merged map to indexed vector ----------
    m_vertex_records.clear();
    m_vertex_records.reserve(merged.size());
    m_snap_key_to_record_idx.clear();
    int next_id = 0;
    for (auto& kv : merged)
    {
        VertexRecord& rec = kv.second;
        rec.synth_id = next_id;
        m_snap_key_to_record_idx[kv.first] = next_id;
        m_vertex_records.push_back(std::move(rec));
        ++next_id;
    }

    // Phase 4.2 / Batch J — face-element AllGather is gone. Face
    // elements travel via TileShuffleFaceElements on the boundary
    // subcomm; per-pair mortar blocks are produced tile-locally by
    // BuildLocalPairBlocks and AllGather'd as blocks (smaller than
    // raw elements) by GatherPairBlocksAcrossBoundary. The
    // build_dedup_key + face_seen + process_face_pack scaffolding
    // that lived here previously has been removed.
}

//==============================================================================
// Step 4a — corners (8 total, |attr_set| == 3)
//==============================================================================

void BoundaryClassifier3D::BuildCorners()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::build_corners");

    std::vector<const VertexRecord*> corner_records;
    for (const VertexRecord& r : m_vertex_records)
    {
        if (r.parent_attrs.size() == 3) { corner_records.push_back(&r); }
    }
    MFEM_VERIFY(corner_records.size() == 8,
                "BoundaryClassifier3D: expected 8 corner vertices "
                "(|attr_set| == 3), found " << corner_records.size()
                << ". Mesh may not be a topologically axis-aligned box. "
                "Total boundary vertices gathered: " << m_vertex_records.size());

    const double xmin = m_bbox_min[0], xmax = m_bbox_max[0];
    const double ymin = m_bbox_min[1], ymax = m_bbox_max[1];
    const double zmin = m_bbox_min[2], zmax = m_bbox_max[2];

    // Label convention per CornerInfo3D: "blf" = bottom-left-front, etc.
    //   first letter:  b = bottom(y_min) / t = top(y_max)
    //   second letter: l = left(x_min)   / r = right(x_max)
    //   third letter:  f = front(z_min)  / b = back(z_max)
    struct Target { const char* label; std::array<double, 3> coord; };
    std::array<Target, 8> targets = {{
        {"blf", {xmin, ymin, zmin}},
        {"brf", {xmax, ymin, zmin}},
        {"blb", {xmin, ymin, zmax}},
        {"brb", {xmax, ymin, zmax}},
        {"tlf", {xmin, ymax, zmin}},
        {"trf", {xmax, ymax, zmin}},
        {"tlb", {xmin, ymax, zmax}},
        {"trb", {xmax, ymax, zmax}},
    }};
    for (const Target& t : targets)
    {
        const VertexRecord* best = nullptr;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (const VertexRecord* r : corner_records)
        {
            const double dx = r->coord[0] - t.coord[0];
            const double dy = r->coord[1] - t.coord[1];
            const double dz = r->coord[2] - t.coord[2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) { best_d2 = d2; best = r; }
        }
        MFEM_VERIFY(best != nullptr && std::sqrt(best_d2) <= m_tol,
                    "BoundaryClassifier3D: no corner record within tol="
                    << m_tol << " of target ('" << t.label << "', "
                    << t.coord[0] << ", " << t.coord[1] << ", " << t.coord[2]
                    << "). Best distance was " << std::sqrt(best_d2));

        CornerInfo3D ci;
        ci.label = t.label;
        ci.coord = best->coord;
        ci.gtdof_x = best->gtdof_xyz[0];
        ci.gtdof_y = best->gtdof_xyz[1];
        ci.gtdof_z = best->gtdof_xyz[2];
        m_corners[ci.label] = std::move(ci);
    }
}

//==============================================================================
// Step 4b — edges (12 total, |attr_set| == 2)
//==============================================================================

void BoundaryClassifier3D::BuildEdges()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::build_edges");

    // Group |attr_set| == 2 vertices by their (sorted) attr pair.
    std::map<std::pair<int, int>, std::vector<const VertexRecord*>> edge_groups;
    for (const VertexRecord& r : m_vertex_records)
    {
        if (r.parent_attrs.size() != 2) { continue; }
        std::pair<int, int> key{r.parent_attrs[0], r.parent_attrs[1]};
        edge_groups[key].push_back(&r);
    }
    MFEM_VERIFY(edge_groups.size() == 12,
                "BoundaryClassifier3D: expected 12 distinct (attr1, attr2) "
                "pairs for box edges, found " << edge_groups.size());

    const auto& mortar_set = MortarLabels();

    for (auto& kv : edge_groups)
    {
        const std::pair<int, int>& attr_pair = kv.first;
        std::vector<const VertexRecord*>& recs = kv.second;

        // Determine parametric axis: the variance-based answer for
        // multi-vertex edges, attr-based for the degenerate
        // single-vertex case.
        std::string param_axis;
        if (recs.size() >= 2)
        {
            double mins[3] = { std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
            double maxs[3] = {-std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()};
            for (const VertexRecord* r : recs)
            {
                for (int d = 0; d < 3; ++d)
                {
                    mins[d] = std::min(mins[d], r->coord[d]);
                    maxs[d] = std::max(maxs[d], r->coord[d]);
                }
            }
            int best_d = 0;
            double best_spread = maxs[0] - mins[0];
            for (int d = 1; d < 3; ++d)
            {
                const double s = maxs[d] - mins[d];
                if (s > best_spread) { best_spread = s; best_d = d; }
            }
            param_axis = std::string(1, "xyz"[best_d]);
        }
        else
        {
            // Single-vertex edge: derive from face attrs.
            param_axis = ParamAxisFromAttrs(attr_pair, m_face_label_by_attr);
        }

        const std::string label = EdgeLabel(param_axis, attr_pair,
                                            m_face_label_by_attr);
        const int axis_idx = AxisIdx(param_axis);

        // Sort interior records along the parametric axis.
        std::sort(recs.begin(), recs.end(),
                  [axis_idx](const VertexRecord* a, const VertexRecord* b)
                  { return a->coord[axis_idx] < b->coord[axis_idx]; });

        const int n_interior = static_cast<int>(recs.size());
        EdgeInfo3D edge;
        edge.label = label;
        edge.parametric_axis = param_axis;
        edge.edge_min = m_bbox_min[axis_idx];
        edge.edge_max = m_bbox_max[axis_idx];
        edge.coords.SetSize(n_interior, 3);
        edge.gtdofs_x.SetSize(n_interior);
        edge.gtdofs_y.SetSize(n_interior);
        edge.gtdofs_z.SetSize(n_interior);
        for (int k = 0; k < n_interior; ++k)
        {
            edge.coords(k, 0) = recs[k]->coord[0];
            edge.coords(k, 1) = recs[k]->coord[1];
            edge.coords(k, 2) = recs[k]->coord[2];
            edge.gtdofs_x[k]  = recs[k]->gtdof_xyz[0];
            edge.gtdofs_y[k]  = recs[k]->gtdof_xyz[1];
            edge.gtdofs_z[k]  = recs[k]->gtdof_xyz[2];
        }

        // Connectivity: [(-1, 0), (0, 1), ..., (n-1, -2)].
        edge.elements.reserve(n_interior + 1);
        edge.elements.emplace_back(kEdgeNodeLeftCornerSentinel, 0);
        for (int k = 0; k < n_interior - 1; ++k)
        {
            edge.elements.emplace_back(k, k + 1);
        }
        edge.elements.emplace_back(n_interior - 1, kEdgeNodeRightCornerSentinel);

        // Determine corner labels at endpoints.
        const std::string& f1_name = m_face_label_by_attr.at(attr_pair.first);
        const std::string& f2_name = m_face_label_by_attr.at(attr_pair.second);
        auto face_value = [this](const std::string& face_name)
            -> std::pair<std::string, double>
        {
            const auto& fa = FaceAxes(face_name);
            const std::string& perp = fa.first;
            const int ax = AxisIdx(perp);
            const bool high =
                (face_name == "top" || face_name == "right" || face_name == "back");
            return {perp, high ? m_bbox_max[ax] : m_bbox_min[ax]};
        };
        const auto fv1 = face_value(f1_name);
        const auto fv2 = face_value(f2_name);
        const int ax_idx_p1 = AxisIdx(fv1.first);
        const int ax_idx_p2 = AxisIdx(fv2.first);

        std::array<double, 3> tgt_min = {0, 0, 0};
        std::array<double, 3> tgt_max = {0, 0, 0};
        tgt_min[axis_idx]   = edge.edge_min;
        tgt_max[axis_idx]   = edge.edge_max;
        tgt_min[ax_idx_p1]  = fv1.second;
        tgt_max[ax_idx_p1]  = fv1.second;
        tgt_min[ax_idx_p2]  = fv2.second;
        tgt_max[ax_idx_p2]  = fv2.second;

        auto find_corner = [this](const std::array<double, 3>& tgt) -> std::string
        {
            for (const auto& cv : m_corners)
            {
                const auto& c = cv.second;
                if (std::abs(c.coord[0] - tgt[0]) < m_tol
                    && std::abs(c.coord[1] - tgt[1]) < m_tol
                    && std::abs(c.coord[2] - tgt[2]) < m_tol)
                {
                    return cv.first;
                }
            }
            MFEM_ABORT("BoundaryClassifier3D: no corner found at target ("
                       << tgt[0] << ", " << tgt[1] << ", " << tgt[2] << ")");
            return {};
        };
        edge.corner_min_label = find_corner(tgt_min);
        edge.corner_max_label = find_corner(tgt_max);

        // Mortar/nonmortar: edge is mortar iff BOTH adjacent faces are
        // nonmortars (the "low-low corner" edge along its parametric axis).
        const bool both_nonmortar =
            (mortar_set.find(f1_name) == mortar_set.end()) &&
            (mortar_set.find(f2_name) == mortar_set.end());
        edge.is_mortar = both_nonmortar;

        m_edges[label] = std::move(edge);
    }
}

//==============================================================================
// Step 4c — faces (6 total) and per-face element lists
//==============================================================================

void BoundaryClassifier3D::BuildFaces()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::build_faces");

    // Phase 4.2 / Batch J — `face.interior_gtdofs_x/y/z` is now
    // computed from `m_vertex_records` directly (vertices with
    // `parent_attrs.size() == 1` are face-interior on the unique
    // face named by their single parent_attr), without needing the
    // AllGather'd per-face element list. The face.quad_elements /
    // face.tri_elements vectors are a per-rank diagnostic populated
    // from `m_tile_shuffled_face_elements`; at np=1 this is the
    // global set, at np>1 it is the per-rank tile slice.
    // Downstream consumers (ConstraintBuilder3D) read PairBlocks()
    // instead.

    // Build a primary-gtdof -> sentinel-class map.
    std::map<int, int> sentinel_class;
    for (const VertexRecord& r : m_vertex_records)
    {
        if (r.parent_attrs.size() == 3)
        {
            sentinel_class[r.gtdof_xyz[0]] = kGtdofCornerSentinel;
        }
        else if (r.parent_attrs.size() == 2)
        {
            sentinel_class[r.gtdof_xyz[0]] = kGtdofEdgeSentinel;
        }
    }

    const auto& mortar_set = MortarLabels();

    // Step 1 — face metadata (label, is_mortar, axes, plane_value,
    // bounding_edge_labels). Cheap; no element data needed.
    for (const auto& attr_label : m_face_label_by_attr)
    {
        const int attr = attr_label.first;
        const std::string& face_label = attr_label.second;
        const auto fa = FaceAxes(face_label);
        const std::string& perp_axis = fa.first;
        const auto& param_axes = fa.second;
        const int perp_idx = AxisIdx(perp_axis);
        const bool high_side =
            (face_label == "top" || face_label == "right" || face_label == "back");
        const double plane_value = high_side ? m_bbox_max[perp_idx]
                                             : m_bbox_min[perp_idx];

        FaceInfo3D face;
        face.label = face_label;
        face.is_mortar = (mortar_set.find(face_label) != mortar_set.end());
        face.perpendicular_axis = perp_axis;
        face.plane_value = plane_value;
        face.parametric_axes = param_axes;
        face.bounding_edge_labels =
            FaceBoundingEdgeLabels(attr, m_face_label_by_attr);
        m_faces[face_label] = std::move(face);
    }

    // Step 2 — populate interior_gtdofs_x/y/z from vertex_records.
    // A vertex with parent_attrs.size() == 1 is in the interior of
    // exactly one face (corners have 3 attrs, edges have 2). Use a
    // per-face std::set to dedup defensively, then unload to mfem::Array.
    std::map<std::string, std::set<int>> interior_x_per_face;
    std::map<std::string, std::set<int>> interior_y_per_face;
    std::map<std::string, std::set<int>> interior_z_per_face;
    for (const VertexRecord& vr : m_vertex_records)
    {
        if (vr.parent_attrs.size() != 1) { continue; }
        const int face_attr = vr.parent_attrs[0];
        auto it = m_face_label_by_attr.find(face_attr);
        MFEM_VERIFY(it != m_face_label_by_attr.end(),
                    "BuildFaces: vertex parent_attr=" << face_attr
                    << " has no face label");
        const std::string& face_label = it->second;
        interior_x_per_face[face_label].insert(vr.gtdof_xyz[0]);
        interior_y_per_face[face_label].insert(vr.gtdof_xyz[1]);
        interior_z_per_face[face_label].insert(vr.gtdof_xyz[2]);
    }
    for (auto& kv : m_faces)
    {
        const std::string& label = kv.first;
        FaceInfo3D& face = kv.second;
        const auto& sx = interior_x_per_face[label];
        const auto& sy = interior_y_per_face[label];
        const auto& sz = interior_z_per_face[label];
        face.interior_gtdofs_x.SetSize(static_cast<int>(sx.size()));
        face.interior_gtdofs_y.SetSize(static_cast<int>(sy.size()));
        face.interior_gtdofs_z.SetSize(static_cast<int>(sz.size()));
        int k = 0; for (int g : sx) { face.interior_gtdofs_x[k++] = g; }
        k = 0;     for (int g : sy) { face.interior_gtdofs_y[k++] = g; }
        k = 0;     for (int g : sz) { face.interior_gtdofs_z[k++] = g; }
    }

    // Step 3 — diagnostic-only: populate face.quad_elements /
    // face.tri_elements from m_tile_shuffled_face_elements (per-rank
    // slice, deduped by (parent_attr, sorted snap_keys)). At np=1 this
    // is the global set; at np>1 it is partial. Constraint builder
    // doesn't use these — they exist for unit-test introspection
    // (test_sentinel_rewriting, test_faces_count_and_mortar_flags) and
    // for any debugging / visualization that wants per-element data.
    {
        std::set<std::vector<long long>> seen;
        auto build_dedup_key = [](int attr,
            const std::vector<std::array<long long, 3>>& sk)
            -> std::vector<long long>
        {
            std::vector<std::array<long long, 3>> sorted = sk;
            std::sort(sorted.begin(), sorted.end());
            std::vector<long long> key;
            key.reserve(1 + 3 * sorted.size());
            key.push_back(attr);
            for (const auto& k : sorted)
            {
                key.push_back(k[0]); key.push_back(k[1]); key.push_back(k[2]);
            }
            return key;
        };

        // Group shuffled elements by parent_attr (face), deduped.
        std::map<int, std::vector<const ShuffledFaceElement*>> per_attr;
        for (const auto& sfe : m_tile_shuffled_face_elements)
        {
            std::vector<long long> dk = build_dedup_key(sfe.parent_attr,
                                                        sfe.snap_keys);
            if (!seen.insert(std::move(dk)).second) { continue; }
            per_attr[sfe.parent_attr].push_back(&sfe);
        }

        // Convert per-face shuffled elements to QuadFaceElement /
        // TriFaceElement, splitting by geometry. Reuse the existing
        // ConvertShuffledToQuads / ConvertShuffledToTris helpers.
        for (const auto& kv : per_attr)
        {
            const int attr = kv.first;
            auto label_it = m_face_label_by_attr.find(attr);
            if (label_it == m_face_label_by_attr.end()) { continue; }
            const std::string& face_label = label_it->second;
            FaceInfo3D& face = m_faces[face_label];

            std::vector<const ShuffledFaceElement*> quad_p;
            std::vector<const ShuffledFaceElement*> tri_p;
            for (const ShuffledFaceElement* sfe : kv.second)
            {
                if (sfe->geometry_kind == "quad") { quad_p.push_back(sfe); }
                else                              { tri_p.push_back(sfe); }
            }
            if (!quad_p.empty())
            {
                auto qe = ConvertShuffledToQuads(quad_p, face_label,
                                                 sentinel_class);
                face.n_quad_elements = static_cast<int>(qe.size());
                face.quad_elements = std::move(qe);
            }
            if (!tri_p.empty())
            {
                auto te = ConvertShuffledToTris(tri_p, face_label,
                                                sentinel_class);
                face.n_tri_elements = static_cast<int>(te.size());
                face.tri_elements = std::move(te);
            }
        }
    }
}

//==============================================================================
// Public helpers used by the constraint builder
//==============================================================================

std::map<int, std::array<int, 3>> BoundaryClassifier3D::GtdofXyzLookup() const
{
    std::map<int, std::array<int, 3>> out;
    for (const VertexRecord& r : m_vertex_records)
    {
        const int gx = r.gtdof_xyz[0];
        if (gx >= 0)
        {
            out[gx] = {gx, r.gtdof_xyz[1], r.gtdof_xyz[2]};
        }
    }
    return out;
}

std::vector<std::tuple<std::string, std::string, std::string>>
BoundaryClassifier3D::EdgePairs() const
{
    std::map<std::string, std::string> mortar_by_axis;
    std::map<std::string, std::vector<std::string>> nonmortars_by_axis;
    nonmortars_by_axis["x"]; nonmortars_by_axis["y"]; nonmortars_by_axis["z"];

    for (const auto& kv : m_edges)
    {
        const std::string& label = kv.first;
        const EdgeInfo3D& e = kv.second;
        if (e.is_mortar)
        {
            MFEM_VERIFY(mortar_by_axis.find(e.parametric_axis) ==
                            mortar_by_axis.end(),
                        "BoundaryClassifier3D: multiple mortar edges along "
                        "axis '" << e.parametric_axis << "'");
            mortar_by_axis[e.parametric_axis] = label;
        }
        else
        {
            nonmortars_by_axis[e.parametric_axis].push_back(label);
        }
    }

    std::vector<std::tuple<std::string, std::string, std::string>> out;
    out.reserve(9);
    for (const std::string& axis : {std::string("x"), std::string("y"),
                                    std::string("z")})
    {
        auto m_it = mortar_by_axis.find(axis);
        MFEM_VERIFY(m_it != mortar_by_axis.end(),
                    "BoundaryClassifier3D: no mortar edge along axis '"
                    << axis << "'");
        std::vector<std::string>& nm = nonmortars_by_axis.at(axis);
        MFEM_VERIFY(nm.size() == 3,
                    "BoundaryClassifier3D: axis '" << axis << "': expected "
                    "3 nonmortar edges, found " << nm.size());
        std::sort(nm.begin(), nm.end());
        for (const std::string& nm_label : nm)
        {
            out.emplace_back(axis, m_it->second, nm_label);
        }
    }
    return out;
}

std::vector<std::tuple<std::string, std::string, std::string>>
BoundaryClassifier3D::FacePairs() const
{
    std::vector<std::tuple<std::string, std::string, std::string>> out;
    out.reserve(3);
    for (const auto& mp : mortar_pbc::FacePairs())
    {
        const std::string& mortar = mp.first;
        const std::string& nonmortar = mp.second;
        const auto fa = FaceAxes(mortar);
        out.emplace_back(fa.first, mortar, nonmortar);
    }
    return out;
}

//==============================================================================
// Phase 5.9 — face-attribute / corner-pinning topology accessors
//
// Used by MortarPbcManager (Phase 5.9.A.4) to:
//   - Resolve PeriodicBC::essential_ids → corner-vertex set
//     (CornersOnFaceAttribute).
//   - Validate pair completeness across user-specified attrs
//     (ArePaired, PairPartnerLabel, LabelForMeshAttribute,
//      MeshAttributeForLabel, IsBoundaryFaceAttribute).
//   - Identify the unconditional anchor TDOFs (AnchorCornerTDofs).
//
// All six are local (no MPI collectives) and read-only — replicated
// state guarantees same answer on every rank.
//==============================================================================

std::vector<std::string> BoundaryClassifier3D::CornersOnFaceAttribute(
    int face_attr) const
{
    // Reverse-lookup attr → face label. Returns empty if attr isn't a
    // known boundary face attribute on this classifier.
    auto attr_it = m_face_label_by_attr.find(face_attr);
    if (attr_it == m_face_label_by_attr.end()) {
        return {};
    }
    const std::string& face_label = attr_it->second;

    // Map face label → (position in corner label, expected letter).
    // Corner labels are 3 letters: positions 0/1/2 encode the
    // y / x / z axis halves respectively. See CornerInfo3D's docstring
    // in types_3d.hpp for the convention.
    int pos = -1;
    char letter = ' ';
    if      (face_label == "bottom") { pos = 0; letter = 'b'; }
    else if (face_label == "top"   ) { pos = 0; letter = 't'; }
    else if (face_label == "left"  ) { pos = 1; letter = 'l'; }
    else if (face_label == "right" ) { pos = 1; letter = 'r'; }
    else if (face_label == "front" ) { pos = 2; letter = 'f'; }
    else if (face_label == "back"  ) { pos = 2; letter = 'b'; }
    else {
        // Label is in the attr↔label map but isn't one of the 6
        // recognized face labels. Shouldn't happen post-construction
        // (classifier enforces the 6-face contract) but defend
        // anyway.
        return {};
    }

    std::vector<std::string> result;
    result.reserve(4);  // each face has exactly 4 corners
    for (const auto& kv : m_corners) {
        const std::string& corner_label = kv.first;
        if (corner_label.size() >= 3 && corner_label[pos] == letter) {
            result.push_back(corner_label);
        }
    }
    return result;
}

std::string BoundaryClassifier3D::PairPartnerLabel(
    const std::string& label) const
{
    // Fixed cuboid pair topology — same on every classifier.
    // `std::map` over `std::unordered_map` because the table is tiny
    // (6 entries) and `<map>` is already included for
    // `m_face_label_by_attr`.
    static const std::map<std::string, std::string> partners = {
        {"bottom", "top"  }, {"top",   "bottom"},
        {"left",   "right"}, {"right", "left"  },
        {"front",  "back" }, {"back",  "front" }
    };
    auto it = partners.find(label);
    return (it != partners.end()) ? it->second : std::string();
}

bool BoundaryClassifier3D::ArePaired(int attr_a, int attr_b) const
{
    const std::string label_a = LabelForMeshAttribute(attr_a);
    if (label_a.empty()) { return false; }
    const std::string partner = PairPartnerLabel(label_a);
    if (partner.empty()) { return false; }
    return MeshAttributeForLabel(partner) == attr_b;
}

int BoundaryClassifier3D::MeshAttributeForLabel(
    const std::string& label) const
{
    // Linear scan; m_face_label_by_attr has at most 6 entries.
    for (const auto& kv : m_face_label_by_attr) {
        if (kv.second == label) {
            return kv.first;
        }
    }
    return -1;
}

std::string BoundaryClassifier3D::LabelForMeshAttribute(int attr) const
{
    auto it = m_face_label_by_attr.find(attr);
    return (it != m_face_label_by_attr.end()) ? it->second : std::string();
}

bool BoundaryClassifier3D::IsBoundaryFaceAttribute(int attr) const
{
    return m_face_label_by_attr.find(attr) != m_face_label_by_attr.end();
}

mfem::Array<int> BoundaryClassifier3D::AnchorCornerTDofs(
    const mfem::ParFiniteElementSpace& fes) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::boundary_classifier::anchor_corner_tdofs");

    // The "blf" corner is the (bbox_min[0], bbox_min[1], bbox_min[2])
    // vertex by classifier convention (see BuildCorners in this file).
    // Construction guarantees the 8 corners are populated; if "blf"
    // is somehow missing, return empty rather than abort — caller's
    // coverage check will catch it via the global-count = 3 invariant.
    auto it = m_corners.find("blf");
    if (it == m_corners.end()) {
        return mfem::Array<int>();
    }
    const CornerInfo3D& anchor = it->second;

    const int my_rank = Rank();
    const HYPRE_BigInt my_offset = fes.GetMyTDofOffset();

    mfem::Array<int> result;
    result.Reserve(3);

    const std::array<int, 3> gtdofs = anchor.GTDofs();
    for (int comp = 0; comp < 3; ++comp) {
        const int gtdof = gtdofs[comp];
        if (gtdof < 0) { continue; }  // unowned-on-this-rank sentinel

        // Ownership test via classifier's binary search over the
        // Allgather'd TDOF offsets (Phase 4.2 / Batch N).
        if (GtdofOwnerRank(gtdof) == my_rank) {
            const int local = gtdof - static_cast<int>(my_offset);
            result.Append(local);
        }
    }

    return result;
}

std::string BoundaryClassifier3D::Summary() const
{
    std::ostringstream oss;
    oss << "BoundaryClassifier3D summary:\n";
    oss << "  bbox: ["
        << m_bbox_min[0] << ", " << m_bbox_min[1] << ", " << m_bbox_min[2]
        << "] -> ["
        << m_bbox_max[0] << ", " << m_bbox_max[1] << ", " << m_bbox_max[2]
        << "]\n";
    oss << "  tol:  " << m_tol << "\n";
    oss << "  attribute -> face label:\n";
    for (const auto& kv : m_face_label_by_attr)
    {
        oss << "    attr " << kv.first << " -> " << kv.second << "\n";
    }
    oss << "  corners (8): ";
    for (const auto& kv : m_corners) { oss << kv.first << " "; }
    oss << "\n";
    oss << "  edges (" << m_edges.size() << "):";
    int n_mortar_edges = 0;
    for (const auto& kv : m_edges)
    {
        if (kv.second.is_mortar) { ++n_mortar_edges; }
    }
    oss << " " << n_mortar_edges << " mortar + "
        << (m_edges.size() - n_mortar_edges) << " nonmortar\n";
    oss << "  faces (" << m_faces.size() << "):";
    for (const auto& kv : m_faces)
    {
        oss << " " << kv.first
            << "(" << kv.second.NumElements() << " elems"
            << (kv.second.is_mortar ? ", M" : ", N") << ")";
    }
    oss << "\n";
    return oss.str();
}


//==============================================================================
// Phase 4.2 / Batch H — TileShuffleFaceElements
//
// Pack each rank's local boundary face elements per destination tile,
// AllToAllv on m_boundary_comm, unpack into m_tile_shuffled_face_elements.
//
// Pack format (per element, fixed-width — fits cleanly in MPI_Alltoallv):
//
//   ints (per elem, kSPackInts longs):
//     [ 0]  parent_attr
//     [ 1]  n_verts (3 for tri, 4 for quad)
//     [ 2.. 4]  snap_key[0]
//     [ 5.. 7]  snap_key[1]
//     [ 8..10]  snap_key[2]
//     [11..13]  snap_key[3]   (zero-filled for tri elements)
//
//   doubles (per elem, kSPackDoubles doubles):
//     [ 0.. 2]  coords[0]
//     [ 3.. 5]  coords[1]
//     [ 6.. 8]  coords[2]
//     [ 9..11]  coords[3]     (zero-filled for tri elements)
//
// Two parallel streams: one long, one double, each their own
// MPI_Alltoallv on m_boundary_comm. Required to keep MPI types clean
// (MPI does not support heterogeneous Alltoall).
//
// Routing decision (per local element):
//   1. Look up face_label from m_face_label_by_attr[parent_attr].
//   2. Look up (perp_axis, {param_a, param_b}) from FaceAxes(face_label).
//      The axis_pair is the perpendicular axis (e.g. face "front" has
//      perp = "z" → tile-route on the (x, y) parametric plane = the
//      tile partition's "z" axis-pair).
//   3. Compute parametric centroid (average of vertex coords).
//   4. Use m_tile_partition->OwnerRank(axis_pair, centroid) to get the
//      destination boundary-comm rank.
//==============================================================================

namespace {

constexpr int kSPackInts    = 14;  // see pack layout above
constexpr int kSPackDoubles = 12;

}  // anonymous namespace

void BoundaryClassifier3D::TileShuffleFaceElements()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::tile_shuffle");

    MFEM_VERIFY(IsBoundaryRank(),
                "TileShuffleFaceElements: must only be called on boundary "
                "ranks. The caller is responsible for guarding with "
                "IsBoundaryRank().");
    MFEM_VERIFY(m_tile_partition != nullptr,
                "TileShuffleFaceElements: m_tile_partition is null on a "
                "boundary rank — did the constructor build it?");

    mfem::ParSubMesh& sub = *m_bdr_submesh;
    const mfem::Array<int>& parent_vmap = sub.GetParentVertexIDMap();
    const mfem::Array<int>& parent_emap = sub.GetParentElementIDMap();
    const int n_sub_elems = sub.GetNE();

    //------------------------------------------------------------------
    // Pass 1 — for each local face element, determine destination rank
    //          and build the per-destination element list.
    //------------------------------------------------------------------
    // send_buckets[dest_bdy_rank] = vector of element indices.
    std::vector<std::vector<int>> send_buckets(m_n_bdy_ranks);
    // Per-element cached metadata to avoid recomputing during the pack.
    struct LocalElem
    {
        int parent_attr = 0;
        int n_verts = 0;
        std::array<std::array<long long, 3>, 4> snap_keys = {};
        std::array<std::array<double, 3>, 4>    coords    = {};
    };
    std::vector<LocalElem> local_elems(n_sub_elems);

    for (int se = 0; se < n_sub_elems; ++se)
    {
        const int parent_be = parent_emap[se];
        const int parent_attr = m_pmesh.GetBdrAttribute(parent_be);

        mfem::Array<int> sub_verts;
        sub.GetElementVertices(se, sub_verts);
        const int n_verts = sub_verts.Size();
        MFEM_VERIFY(n_verts == 3 || n_verts == 4,
                    "TileShuffleFaceElements: face element with " << n_verts
                    << " vertices (expected 3 or 4)");

        LocalElem& le = local_elems[se];
        le.parent_attr = parent_attr;
        le.n_verts = n_verts;

        double centroid[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < n_verts; ++k)
        {
            const int parent_v = parent_vmap[sub_verts[k]];
            const double* xyz = m_pmesh.GetVertex(parent_v);
            for (int d = 0; d < 3; ++d)
            {
                le.coords[k][d] = xyz[d];
                centroid[d] += xyz[d];
            }
            le.snap_keys[k] = SnapKey(xyz[0], xyz[1], xyz[2], m_tol);
        }
        for (int d = 0; d < 3; ++d)
        {
            centroid[d] /= static_cast<double>(n_verts);
        }

        // Determine the axis-pair for this face element. The face's
        // PERPENDICULAR axis IS the axis-pair name in TilePartition3D's
        // convention (axis-pair "z" tiles the (x, y) plane, i.e. the
        // perpendicular axis is z).
        auto attr_it = m_face_label_by_attr.find(parent_attr);
        MFEM_VERIFY(attr_it != m_face_label_by_attr.end(),
                    "TileShuffleFaceElements: parent attribute "
                    << parent_attr << " has no face label in "
                    "m_face_label_by_attr.");
        const std::string& face_label = attr_it->second;
        const auto fa = FaceAxes(face_label);
        const std::string& axis_pair = fa.first;

        const std::array<double, 3> centroid_arr = {
            centroid[0], centroid[1], centroid[2]};
        const int dest_bdy_rank = m_tile_partition->OwnerRank(
            axis_pair, centroid_arr);
        MFEM_VERIFY(dest_bdy_rank >= 0 && dest_bdy_rank < m_n_bdy_ranks,
                    "TileShuffleFaceElements: OwnerRank returned "
                    << dest_bdy_rank << " out of range [0, "
                    << m_n_bdy_ranks << ")");
        send_buckets[dest_bdy_rank].push_back(se);
    }

    //------------------------------------------------------------------
    // Pass 2 — pack send buffers in dest-rank order.
    //------------------------------------------------------------------
    std::vector<int> send_counts(m_n_bdy_ranks, 0);
    for (int r = 0; r < m_n_bdy_ranks; ++r)
    {
        send_counts[r] = static_cast<int>(send_buckets[r].size());
    }
    std::vector<int> send_displs(m_n_bdy_ranks, 0);
    int total_send_elems = 0;
    for (int r = 0; r < m_n_bdy_ranks; ++r)
    {
        send_displs[r] = total_send_elems;
        total_send_elems += send_counts[r];
    }

    std::vector<long long> send_int_pack(
        static_cast<std::size_t>(total_send_elems) * kSPackInts);
    std::vector<double>    send_dbl_pack(
        static_cast<std::size_t>(total_send_elems) * kSPackDoubles);

    {
        int write_idx = 0;
        for (int r = 0; r < m_n_bdy_ranks; ++r)
        {
            for (int se : send_buckets[r])
            {
                const LocalElem& le = local_elems[se];
                long long* islot = send_int_pack.data()
                                 + write_idx * kSPackInts;
                double*    dslot = send_dbl_pack.data()
                                 + write_idx * kSPackDoubles;
                islot[0] = le.parent_attr;
                islot[1] = le.n_verts;
                for (int k = 0; k < 4; ++k)
                {
                    if (k < le.n_verts)
                    {
                        islot[2 + k * 3 + 0] = le.snap_keys[k][0];
                        islot[2 + k * 3 + 1] = le.snap_keys[k][1];
                        islot[2 + k * 3 + 2] = le.snap_keys[k][2];
                        dslot[k * 3 + 0]     = le.coords[k][0];
                        dslot[k * 3 + 1]     = le.coords[k][1];
                        dslot[k * 3 + 2]     = le.coords[k][2];
                    }
                    else
                    {
                        // Padding for tri (k=3 unused).
                        islot[2 + k * 3 + 0] = 0;
                        islot[2 + k * 3 + 1] = 0;
                        islot[2 + k * 3 + 2] = 0;
                        dslot[k * 3 + 0]     = 0.0;
                        dslot[k * 3 + 1]     = 0.0;
                        dslot[k * 3 + 2]     = 0.0;
                    }
                }
                ++write_idx;
            }
        }
    }

    //------------------------------------------------------------------
    // Exchange counts (Alltoall of 1 int per rank).
    //------------------------------------------------------------------
    std::vector<int> recv_counts(m_n_bdy_ranks, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 m_boundary_comm);

    int total_recv_elems = 0;
    std::vector<int> recv_displs(m_n_bdy_ranks, 0);
    for (int r = 0; r < m_n_bdy_ranks; ++r)
    {
        recv_displs[r] = total_recv_elems;
        total_recv_elems += recv_counts[r];
    }

    //------------------------------------------------------------------
    // Alltoallv the packed buffers (int stream + double stream).
    //
    // Counts and displacements must be expressed in MPI scalar units,
    // not element units, for MPI_Alltoallv. So multiply each by the
    // pack stride.
    //------------------------------------------------------------------
    std::vector<int> send_int_counts(m_n_bdy_ranks);
    std::vector<int> send_int_displs(m_n_bdy_ranks);
    std::vector<int> recv_int_counts(m_n_bdy_ranks);
    std::vector<int> recv_int_displs(m_n_bdy_ranks);
    std::vector<int> send_dbl_counts(m_n_bdy_ranks);
    std::vector<int> send_dbl_displs(m_n_bdy_ranks);
    std::vector<int> recv_dbl_counts(m_n_bdy_ranks);
    std::vector<int> recv_dbl_displs(m_n_bdy_ranks);
    for (int r = 0; r < m_n_bdy_ranks; ++r)
    {
        send_int_counts[r] = send_counts[r] * kSPackInts;
        send_int_displs[r] = send_displs[r] * kSPackInts;
        recv_int_counts[r] = recv_counts[r] * kSPackInts;
        recv_int_displs[r] = recv_displs[r] * kSPackInts;
        send_dbl_counts[r] = send_counts[r] * kSPackDoubles;
        send_dbl_displs[r] = send_displs[r] * kSPackDoubles;
        recv_dbl_counts[r] = recv_counts[r] * kSPackDoubles;
        recv_dbl_displs[r] = recv_displs[r] * kSPackDoubles;
    }

    std::vector<long long> recv_int_pack(
        static_cast<std::size_t>(total_recv_elems) * kSPackInts);
    std::vector<double>    recv_dbl_pack(
        static_cast<std::size_t>(total_recv_elems) * kSPackDoubles);

    MPI_Alltoallv(send_int_pack.data(), send_int_counts.data(),
                  send_int_displs.data(), MPI_LONG_LONG,
                  recv_int_pack.data(), recv_int_counts.data(),
                  recv_int_displs.data(), MPI_LONG_LONG,
                  m_boundary_comm);
    MPI_Alltoallv(send_dbl_pack.data(), send_dbl_counts.data(),
                  send_dbl_displs.data(), MPI_DOUBLE,
                  recv_dbl_pack.data(), recv_dbl_counts.data(),
                  recv_dbl_displs.data(), MPI_DOUBLE,
                  m_boundary_comm);

    //------------------------------------------------------------------
    // Unpack into m_tile_shuffled_face_elements.
    //
    // For each received element, decode its axis_pair and (tile_i,
    // tile_j) using the same OwnerRank inversion that the sender used.
    //------------------------------------------------------------------
    m_tile_shuffled_face_elements.clear();
    m_tile_shuffled_face_elements.reserve(total_recv_elems);

    int read_idx = 0;
    for (int src = 0; src < m_n_bdy_ranks; ++src)
    {
        for (int e = 0; e < recv_counts[src]; ++e)
        {
            const long long* islot = recv_int_pack.data()
                                   + read_idx * kSPackInts;
            const double*    dslot = recv_dbl_pack.data()
                                   + read_idx * kSPackDoubles;
            ShuffledFaceElement sfe;
            sfe.parent_attr = static_cast<int>(islot[0]);
            const int n_v = static_cast<int>(islot[1]);
            MFEM_VERIFY(n_v == 3 || n_v == 4,
                        "TileShuffleFaceElements: unpack got n_verts="
                        << n_v << " (expected 3 or 4)");
            sfe.geometry_kind = (n_v == 4) ? "quad" : "tri";
            sfe.snap_keys.resize(n_v);
            sfe.coords.SetSize(n_v, 3);
            double centroid[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < n_v; ++k)
            {
                sfe.snap_keys[k] = {islot[2 + k * 3 + 0],
                                    islot[2 + k * 3 + 1],
                                    islot[2 + k * 3 + 2]};
                for (int d = 0; d < 3; ++d)
                {
                    sfe.coords(k, d) = dslot[k * 3 + d];
                    centroid[d] += dslot[k * 3 + d];
                }
            }
            for (int d = 0; d < 3; ++d)
            {
                centroid[d] /= static_cast<double>(n_v);
            }

            // Decode axis_pair from parent_attr.
            auto attr_it = m_face_label_by_attr.find(sfe.parent_attr);
            MFEM_VERIFY(attr_it != m_face_label_by_attr.end(),
                        "TileShuffleFaceElements unpack: parent attr "
                        << sfe.parent_attr << " has no face label");
            const std::string& face_label = attr_it->second;
            sfe.axis_pair = FaceAxes(face_label).first;

            // Decode (tile_i, tile_j) using OwnerRankFast on this
            // rank's grid for the matching axis. The owner is by
            // construction this rank, so we can recover (i, j) by
            // inverting the rank → tile mapping.
            const AxisTileGrid& grid = m_tile_partition->Grid(sfe.axis_pair);
            const int local_rank_in_axis = m_bdy_rank - grid.axis_rank_start;
            // Defensive sanity check: the element we received MUST be
            // from a rank whose tile we own. If this ever fires, the
            // sender computed a different OwnerRank than we do — a
            // determinism failure that cannot happen by design but
            // would be catastrophic if it did.
            MFEM_VERIFY(local_rank_in_axis >= 0
                        && local_rank_in_axis < grid.n_axis_ranks,
                        "TileShuffleFaceElements unpack: received an "
                        "element on the '" << sfe.axis_pair
                        << "' axis but this rank (m_bdy_rank="
                        << m_bdy_rank << ") does not own any tile on "
                        "that axis. Likely sender/receiver disagree on "
                        "the partition.");
            sfe.tile_i = local_rank_in_axis % grid.n_tx;
            sfe.tile_j = local_rank_in_axis / grid.n_tx;

            sfe.source_bdy_rank = src;
            m_tile_shuffled_face_elements.push_back(std::move(sfe));
            ++read_idx;
        }
    }
}

//==============================================================================
// Phase 4.2 / Batch I — ConvertShuffledToQuads
//
// Convert a list of ShuffledFaceElement* (already filtered to one
// face_label and one geometry_kind == "quad") into QuadFaceElement
// objects with CCW reordering and sentinel-rewritten gtdofs.
//
// Performs the same per-element work that the legacy BuildFaces did
// when it walked the AllGather'd face-element records — CCW reorder
// against the face label, then sentinel rewriting on primary gtdofs
// using the precomputed sentinel-class map. Inputs come from
// ShuffledFaceElement (snap_keys + coords) instead of any global
// element list (the global list no longer exists post-Batch J).
//
// `sentinel_class` is a precomputed gtdof → sentinel-class map
// (kGtdofCornerSentinel for corner gtdofs, kGtdofEdgeSentinel for
// edge gtdofs); the caller builds it once per call to
// BuildLocalPairBlocks for efficiency.
//==============================================================================
std::vector<QuadFaceElement>
BoundaryClassifier3D::ConvertShuffledToQuads(
    const std::vector<const ShuffledFaceElement*>& shuffled,
    const std::string& face_label,
    const std::map<int, int>& sentinel_class) const
{
    std::vector<QuadFaceElement> out;
    out.reserve(shuffled.size());

    const auto fa = FaceAxes(face_label);
    const std::string& perp_axis = fa.first;
    const auto& param_axes = fa.second;

    for (const ShuffledFaceElement* sfe : shuffled)
    {
        MFEM_ASSERT(sfe->geometry_kind == "quad",
                    "ConvertShuffledToQuads: non-quad element");
        const int n_v = static_cast<int>(sfe->snap_keys.size());
        MFEM_ASSERT(n_v == 4, "ConvertShuffledToQuads: snap_keys.size() != 4");

        // CCW-reorder a copy of coords + ids together. We need a
        // per-vertex "id" index for the reorder; use the snap-key
        // lookup to get vertex_record_idx.
        mfem::DenseMatrix coords = sfe->coords;  // copy
        std::vector<int> ids(n_v);
        for (int k = 0; k < n_v; ++k)
        {
            auto it = m_snap_key_to_record_idx.find(sfe->snap_keys[k]);
            MFEM_VERIFY(it != m_snap_key_to_record_idx.end(),
                        "ConvertShuffledToQuads: snap key ("
                        << sfe->snap_keys[k][0] << ", "
                        << sfe->snap_keys[k][1] << ", "
                        << sfe->snap_keys[k][2] << ") not in vertex catalogue. "
                        "Tile-shuffled element does not match a known "
                        "boundary vertex; classifier state inconsistent.");
            ids[k] = it->second;
        }
        ReorderFaceVerticesCcw(coords, ids, face_label);

        // Sentinel rewriting on primary gtdofs.
        std::array<int, 4> sentinel_gtdofs;
        for (int k = 0; k < 4; ++k)
        {
            const VertexRecord& vr = m_vertex_records[ids[k]];
            const int primary = vr.gtdof_xyz[0];
            auto it = sentinel_class.find(primary);
            sentinel_gtdofs[k] = (it != sentinel_class.end())
                ? it->second
                : primary;
        }

        QuadFaceElement qe;
        qe.coords = coords;
        qe.gtdofs = sentinel_gtdofs;
        qe.parametric_axes = param_axes;
        qe.perpendicular_axis = perp_axis;
        qe.boundary_tag = ClassifyQuadBoundaryTag(qe.gtdofs);
        out.push_back(std::move(qe));
    }
    return out;
}

//==============================================================================
// Phase 4.2 / Batch I — ConvertShuffledToTris (mirror of quad version)
//==============================================================================
std::vector<TriFaceElement>
BoundaryClassifier3D::ConvertShuffledToTris(
    const std::vector<const ShuffledFaceElement*>& shuffled,
    const std::string& face_label,
    const std::map<int, int>& sentinel_class) const
{
    std::vector<TriFaceElement> out;
    out.reserve(shuffled.size());

    const auto fa = FaceAxes(face_label);
    const std::string& perp_axis = fa.first;
    const auto& param_axes = fa.second;

    for (const ShuffledFaceElement* sfe : shuffled)
    {
        MFEM_ASSERT(sfe->geometry_kind == "tri",
                    "ConvertShuffledToTris: non-tri element");
        const int n_v = static_cast<int>(sfe->snap_keys.size());
        MFEM_ASSERT(n_v == 3, "ConvertShuffledToTris: snap_keys.size() != 3");

        mfem::DenseMatrix coords = sfe->coords;
        std::vector<int> ids(n_v);
        for (int k = 0; k < n_v; ++k)
        {
            auto it = m_snap_key_to_record_idx.find(sfe->snap_keys[k]);
            MFEM_VERIFY(it != m_snap_key_to_record_idx.end(),
                        "ConvertShuffledToTris: snap key not in vertex "
                        "catalogue.");
            ids[k] = it->second;
        }
        ReorderFaceVerticesCcw(coords, ids, face_label);

        std::array<int, 3> sentinel_gtdofs;
        for (int k = 0; k < 3; ++k)
        {
            const VertexRecord& vr = m_vertex_records[ids[k]];
            const int primary = vr.gtdof_xyz[0];
            auto it = sentinel_class.find(primary);
            sentinel_gtdofs[k] = (it != sentinel_class.end())
                ? it->second
                : primary;
        }

        TriFaceElement te;
        te.coords = coords;
        te.gtdofs = sentinel_gtdofs;
        te.parametric_axes = param_axes;
        te.perpendicular_axis = perp_axis;
        te.boundary_tag = ClassifyTriBoundaryTag(te.gtdofs);
        out.push_back(std::move(te));
    }
    return out;
}

//==============================================================================
// Phase 4.2 / Batch I — BuildLocalPairBlocks
//
// Walk m_tile_shuffled_face_elements; bucket by (axis_pair,
// face_label, geometry_kind); dedup within each bucket by
// (parent_attr, sorted snap_keys); convert to QuadFaceElement /
// TriFaceElement; run MatchConformingFacePairs +
// AssemblePairConforming per (axis_pair, geom) sub-pair; store the
// resulting blocks in m_local_pair_blocks.
//==============================================================================

//==============================================================================
// GtdofOwnerRank — Phase 4.2 / Batch N — binary search on the
// Allgather'd FES TDOF offsets to find the owning rank.
//==============================================================================
int BoundaryClassifier3D::GtdofOwnerRank(int gtdof) const
{
    MFEM_ASSERT(gtdof >= 0 && gtdof < m_n_global_tdofs,
                "GtdofOwnerRank: gtdof " << gtdof << " out of range "
                "[0, " << m_n_global_tdofs << ")");
    MFEM_ASSERT(static_cast<int>(m_fes_tdof_offsets_all.size())
                == m_nranks + 1,
                "GtdofOwnerRank: m_fes_tdof_offsets_all not initialized");

    // Standard upper_bound trick: find first index i such that
    // offsets[i] > gtdof, then owner = i - 1. (Range is monotone non-
    // decreasing; an equal-offset case occurs only for ranks owning
    // zero TDOFs, which shouldn't happen for FES partitions but the
    // upper_bound handles it correctly by returning the rank just
    // before any zero-width run.)
    auto it = std::upper_bound(m_fes_tdof_offsets_all.begin(),
                                       m_fes_tdof_offsets_all.end(),
                                       static_cast<HYPRE_BigInt>(gtdof));
    const int owner = static_cast<int>(
        (it - m_fes_tdof_offsets_all.begin()) - 1);
    MFEM_ASSERT(owner >= 0 && owner < m_nranks,
                "GtdofOwnerRank: computed owner " << owner
                << " out of range for gtdof " << gtdof);
    return owner;
}

void BoundaryClassifier3D::BuildLocalPairBlocks()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::build_local_pair_blocks");
    m_local_pair_blocks.clear();

    if (m_tile_shuffled_face_elements.empty()) { return; }

    // Build the sentinel-class map (corner = 3 attrs, edge = 2 attrs).
    // Mirrors the BuildFaces logic.
    std::map<int, int> sentinel_class;
    for (const VertexRecord& r : m_vertex_records)
    {
        if (r.parent_attrs.size() == 3)
        {
            sentinel_class[r.gtdof_xyz[0]] = kGtdofCornerSentinel;
        }
        else if (r.parent_attrs.size() == 2)
        {
            sentinel_class[r.gtdof_xyz[0]] = kGtdofEdgeSentinel;
        }
    }

    // Stateless assemblers — same as the constraint builder uses.
    QuadFaceMortarAssembler quad_assembler;
    TriFaceMortarAssembler  tri_assembler;

    const auto& mortar_set = MortarLabels();

    // Iterate the 3 face pairs (one per axis-pair).
    // FacePairs() returns (axis, mortar_label, nonmortar_label) tuples.
    for (const auto& tup : FacePairs())
    {
        const std::string& axis = std::get<0>(tup);
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const int mortar_attr    = m_face_attr_by_label.at(mortar_label);
        const int nonmortar_attr = m_face_attr_by_label.at(nonmortar_label);

        // Filter + dedup shuffled elements for this axis-pair.
        // Dedup by (parent_attr, sorted snap_keys) — mirrors the
        // existing AllGather'd dedup. Ranks may have received the
        // same element multiple times if it sat on a partition
        // boundary on the sender side.
        std::set<std::vector<long long>> seen;
        std::vector<const ShuffledFaceElement*> mortar_quads_p;
        std::vector<const ShuffledFaceElement*> mortar_tris_p;
        std::vector<const ShuffledFaceElement*> nonmortar_quads_p;
        std::vector<const ShuffledFaceElement*> nonmortar_tris_p;

        auto build_dedup_key = [](int attr,
            const std::vector<std::array<long long, 3>>& sk)
            -> std::vector<long long>
        {
            std::vector<std::array<long long, 3>> sorted = sk;
            std::sort(sorted.begin(), sorted.end());
            std::vector<long long> key;
            key.reserve(1 + 3 * sorted.size());
            key.push_back(attr);
            for (const auto& k : sorted)
            {
                key.push_back(k[0]); key.push_back(k[1]); key.push_back(k[2]);
            }
            return key;
        };

        for (const auto& sfe : m_tile_shuffled_face_elements)
        {
            if (sfe.axis_pair != axis) { continue; }

            const bool is_mortar    = (sfe.parent_attr == mortar_attr);
            const bool is_nonmortar = (sfe.parent_attr == nonmortar_attr);
            if (!is_mortar && !is_nonmortar)
            {
                // This face element belongs to a different axis-pair
                // OR a different parent_attr (shouldn't happen on
                // axis-aligned RVEs, but tolerated).
                continue;
            }

            std::vector<long long> dk = build_dedup_key(sfe.parent_attr,
                                                       sfe.snap_keys);
            if (!seen.insert(std::move(dk)).second) { continue; }

            if (is_mortar)
            {
                if (sfe.geometry_kind == "quad")
                {
                    mortar_quads_p.push_back(&sfe);
                }
                else
                {
                    mortar_tris_p.push_back(&sfe);
                }
            }
            else
            {
                if (sfe.geometry_kind == "quad")
                {
                    nonmortar_quads_p.push_back(&sfe);
                }
                else
                {
                    nonmortar_tris_p.push_back(&sfe);
                }
            }
        }
        // Defensive: confirm mortar_set assignment matches face label.
        MFEM_ASSERT(mortar_set.find(mortar_label) != mortar_set.end(),
                    "BuildLocalPairBlocks: mortar_label '" << mortar_label
                    << "' not in MortarLabels() set");
        MFEM_ASSERT(mortar_set.find(nonmortar_label) == mortar_set.end(),
                    "BuildLocalPairBlocks: nonmortar_label '"
                    << nonmortar_label << "' is in MortarLabels() set");

        // plane_values for periodicity.
        const auto fa_nonmortar = FaceAxes(nonmortar_label);
        const int perp_idx = AxisIdx(fa_nonmortar.first);
        const bool nm_high =
            (nonmortar_label == "top" || nonmortar_label == "right"
             || nonmortar_label == "back");
        const bool m_high =
            (mortar_label == "top" || mortar_label == "right"
             || mortar_label == "back");
        const double plane_nm = nm_high ? m_bbox_max[perp_idx]
                                        : m_bbox_min[perp_idx];
        const double plane_m  = m_high  ? m_bbox_max[perp_idx]
                                        : m_bbox_min[perp_idx];
        const double period_signed = plane_m - plane_nm;

        // Match + assemble quad sub-pair if both sides have quads.
        if (!nonmortar_quads_p.empty() && !mortar_quads_p.empty())
        {
            std::vector<QuadFaceElement> nm_q = ConvertShuffledToQuads(
                nonmortar_quads_p, nonmortar_label, sentinel_class);
            std::vector<QuadFaceElement> m_q  = ConvertShuffledToQuads(
                mortar_quads_p, mortar_label, sentinel_class);

            // Phase 4.4 / Batch 4.4-E — try the conforming path first;
            // on non-1:1 match (zero-candidate or many-candidate
            // nonmortar element), fall back to the clipped path. The
            // try-style API returns std::nullopt when the meshes are
            // non-matching.
            //
            // Match tolerance comes from the classifier's
            // m_pair_match_tol_rel member (Phase 4.2 / Batch K).
            // Default 1e-9, configurable via the ctor.
            auto matches_opt = TryMatchConformingFacePairs(
                nm_q, m_q, axis, period_signed, m_pair_match_tol_rel);

            FaceMortarPairBlock blk;
            if (matches_opt.has_value())
            {
                // Conforming fast path.
                blk = quad_assembler.AssemblePairConforming(
                    nm_q, m_q, *matches_opt, nonmortar_label, mortar_label);
            }
            else
            {
#ifdef MORTAR_PBC_HAS_AXOM
                // Non-conforming fallback (Axom-gated).
                auto cands    = MatchClippedQuadFacePairs(nm_q, m_q, axis);
                auto sub_tris = ClipQuadFacePairs(nm_q, m_q, cands, axis);
                blk = AssembleQuadFacePairClipped(
                    nm_q, m_q, sub_tris, axis, nonmortar_label, mortar_label);
#else
                MFEM_ABORT("BuildLocalPairBlocks (quad): non-conforming "
                           "face pair detected on axis '" << axis
                           << "' but ExaConstit was built with ENABLE_AXOM=OFF. "
                           "Rebuild with ENABLE_AXOM=ON to enable clipped-path "
                           "support for non-matching meshes.");
#endif
            }

            LocalPairBlock lpb;
            lpb.axis_pair       = axis;
            lpb.mortar_label    = mortar_label;
            lpb.nonmortar_label = nonmortar_label;
            lpb.geometry_kind   = "quad";
            lpb.block           = std::move(blk);
            m_local_pair_blocks.push_back(std::move(lpb));
        }

        // Match + assemble tri sub-pair if both sides have tris.
        if (!nonmortar_tris_p.empty() && !mortar_tris_p.empty())
        {
            std::vector<TriFaceElement> nm_t = ConvertShuffledToTris(
                nonmortar_tris_p, nonmortar_label, sentinel_class);
            std::vector<TriFaceElement> m_t  = ConvertShuffledToTris(
                mortar_tris_p, mortar_label, sentinel_class);

            // Phase 4.4 / Batch 4.4-E — same try-style dispatch as
            // the quad path above.
            auto matches_opt = TryMatchConformingFacePairs(
                nm_t, m_t, axis, period_signed, m_pair_match_tol_rel);

            FaceMortarPairBlock blk;
            if (matches_opt.has_value())
            {
                blk = tri_assembler.AssemblePairConforming(
                    nm_t, m_t, *matches_opt, nonmortar_label, mortar_label);
            }
            else
            {
#ifdef MORTAR_PBC_HAS_AXOM
                auto cands    = MatchClippedTriFacePairs(nm_t, m_t, axis);
                auto sub_tris = ClipTriFacePairs(nm_t, m_t, cands, axis);
                blk = AssembleTriFacePairClipped(
                    nm_t, m_t, sub_tris, axis, nonmortar_label, mortar_label);
#else
                MFEM_ABORT("BuildLocalPairBlocks (tri): non-conforming "
                           "face pair detected on axis '" << axis
                           << "' but ExaConstit was built with ENABLE_AXOM=OFF. "
                           "Rebuild with ENABLE_AXOM=ON to enable clipped-path "
                           "support for non-matching meshes.");
#endif
            }

            LocalPairBlock lpb;
            lpb.axis_pair       = axis;
            lpb.mortar_label    = mortar_label;
            lpb.nonmortar_label = nonmortar_label;
            lpb.geometry_kind   = "tri";
            lpb.block           = std::move(blk);
            m_local_pair_blocks.push_back(std::move(lpb));
        }
    }
}

//==============================================================================
// Phase 4.2 / Batch N — RoutePairBlocksToRowOwners
//
// Replaces Batch I/K's GatherPairBlocksAcrossBoundary. Each boundary
// rank, for each local pair block, partitions its nonmortar rows by
// FES owner rank, packs one block-fragment per destination, and
// MPI_Alltoallv-routes them on m_comm. Each receiving rank ends up
// with only the fragments whose nonmortar gtdofs it owns in FES.
//
// Pack format
// -----------
// Same per-block layout as Batch L (nine-int header + payload),
// reused unchanged for fragments. A fragment is just a smaller
// per-block record whose nonmortar_gtdofs is a subset and whose
// A_m has the corresponding row slice. The full mortar_gtdofs and
// the unmodified A_m column structure are kept (rows are routed,
// columns are not).
//
// Per-block ints (variable length):
//   [0]   geom_kind          (0 = quad, 1 = tri)
//   [1]   axis_pair_idx      (0 = x, 1 = y, 2 = z)
//   [2,3] mortar_label       16 chars zero-padded, cast as 2 longs
//   [4,5] nonmortar_label    16 chars zero-padded, cast as 2 longs
//   [6]   n_n                (number of nonmortar gtdofs / rows in
//                             THIS fragment, possibly < producer's
//                             original block n_n)
//   [7]   n_m                (number of mortar gtdofs / cols)
//   [8]   nnz                (number of A_m nonzeros in fragment)
//   [9 .. 9 + n_n)                                 nonmortar_gtdofs
//   [9 + n_n .. 9 + n_n + n_m)                     mortar_gtdofs
//   [9 + n_n + n_m .. 9 + n_n + n_m + (n_n + 1))   A_m CSR I array
//   [9 + n_n + n_m + n_n + 1 .. ... + nnz)         A_m CSR J array
// Header is 9 longs; payload is (2*n_n + n_m + 1 + nnz) longs.
//
// Per-block doubles (variable length):
//   [0 .. nnz)         A_m CSR data values
//   [nnz .. nnz+n_n)   D
// Total = nnz + n_n doubles.
//
// Phase 4.2 / Batch N changes from Batch L's gather:
//   - Pack format identical (fragments use the same header).
//   - Communicator: m_comm (was m_boundary_comm + Bcast). Required
//     because nonmortar gtdofs may be FES-owned by interior ranks.
//   - Collective: MPI_Alltoallv (was MPI_Allgatherv + MPI_Bcast).
//     Each rank sends n_destinations × variable-size streams; each
//     rank receives 0 or more fragments per source.
//   - Per-rank receive volume: O(global_blocks / n_bdy_ranks) under
//     a uniform partition of nonmortar gtdofs, vs Batch L's
//     O(global_blocks). On a 100³ RVE at np=10⁶ this is the
//     dominant memory win for Phase 4.2.
//
// Multiple source ranks may route fragments for the same
// (axis_pair, mortar_label, nonmortar_label, geom) bucket to the
// same destination. The merge step at the end uses gtdof-keyed
// accumulation (§P4.8.10) to handle shared DOFs across fragments.
//==============================================================================
namespace {

constexpr int kBlockHeaderInts = 9;

// Pack a 16-byte zero-padded char array into 2 long longs.
// Returns std::pair<long long, long long>.
std::pair<long long, long long> PackLabel16(const std::string& label)
{
    char buf[16];
    std::memset(buf, 0, sizeof(buf));
    const std::size_t n = std::min<std::size_t>(label.size(), 16);
    std::memcpy(buf, label.data(), n);
    long long a, b;
    std::memcpy(&a, buf, 8);
    std::memcpy(&b, buf + 8, 8);
    return {a, b};
}

// Inverse: 2 longs → 16-byte zero-padded char array → std::string.
std::string UnpackLabel16(long long a, long long b)
{
    char buf[16];
    std::memcpy(buf, &a, 8);
    std::memcpy(buf + 8, &b, 8);
    // Find first NUL.
    int len = 0;
    while (len < 16 && buf[len] != '\0') { ++len; }
    return std::string(buf, len);
}

int AxisPairIdx(const std::string& s)
{
    if (s == "x") { return 0; }
    if (s == "y") { return 1; }
    if (s == "z") { return 2; }
    MFEM_ABORT("AxisPairIdx: unknown axis_pair '" << s << "'");
    return -1;
}
const char* AxisPairName(int idx)
{
    switch (idx) { case 0: return "x"; case 1: return "y"; case 2: return "z"; }
    MFEM_ABORT("AxisPairName: invalid idx " << idx);
    return nullptr;
}

}  // anonymous namespace

void BoundaryClassifier3D::RoutePairBlocksToRowOwners()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::boundary_classifier::route_pair_blocks");
    m_gathered_pair_blocks.clear();

    // Phase 4.2 / Batch N implementation. Each boundary rank, for
    // each m_local_pair_blocks entry, partitions the entry's
    // nonmortar rows by FES owner rank (via GtdofOwnerRank), then
    // packs one fragment per (destination rank) pair using the same
    // per-block format as Batch L. After all fragments are packed,
    // MPI_Alltoallv on m_comm exchanges them. Receivers unpack,
    // bucket by (axis, mortar, nonmortar, geom), and merge fragments
    // sharing a bucket via gtdof-keyed accumulation.
    //
    // Communicator: m_comm (WORLD). Required because nonmortar
    // gtdofs may be FES-owned by interior ranks (METIS partitioning
    // does NOT guarantee co-location of FES TDOFs and boundary-
    // element-owning ranks).
    //
    // The merge logic at the bottom is identical to Batch L's
    // (gtdof-keyed accumulation per §P4.8.10); only the input source
    // (Alltoallv result) differs.

    //------------------------------------------------------------------
    // Stage 1 — fragment each local block by destination rank.
    //
    // For each local block, we walk its nonmortar_gtdofs[] once,
    // grouping rows by GtdofOwnerRank. Then we slice the A_m CSR by
    // the row groups and produce one DestinationFragment per
    // (rank, original block) where the rank actually receives at
    // least one row.
    //------------------------------------------------------------------
    struct DestinationFragment
    {
        int dest_rank = -1;
        // Header info — shared across all fragments derived from one
        // original m_local_pair_blocks entry.
        std::string axis_pair;
        std::string mortar_label;
        std::string nonmortar_label;
        std::string geometry_kind;
        // Subset content.
        std::vector<int>    frag_nonmortar_gtdofs;
        std::vector<double> frag_D;
        // Source-block-row indices that ended up in this fragment
        // (used to slice A_m's CSR rows).
        std::vector<int>    src_row_indices;
        // Pointer back to source A_m (CSR walk during pack).
        const FaceMortarPairBlock* src_block = nullptr;
    };

    std::vector<DestinationFragment> all_fragments;
    all_fragments.reserve(m_local_pair_blocks.size() * 2);

    for (const auto& lpb : m_local_pair_blocks)
    {
        const int n_n = lpb.block.NumNonmortarKept();
        if (n_n == 0) { continue; }

        // Group source rows by destination rank.
        std::map<int, std::vector<int>> rows_by_dest;
        for (int i = 0; i < n_n; ++i)
        {
            const int g = lpb.block.nonmortar_gtdofs[i];
            const int dest = GtdofOwnerRank(g);
            rows_by_dest[dest].push_back(i);
        }

        for (auto& kv : rows_by_dest)
        {
            DestinationFragment frag;
            frag.dest_rank       = kv.first;
            frag.axis_pair       = lpb.axis_pair;
            frag.mortar_label    = lpb.mortar_label;
            frag.nonmortar_label = lpb.nonmortar_label;
            frag.geometry_kind   = lpb.geometry_kind;
            frag.src_block       = &lpb.block;
            frag.src_row_indices = std::move(kv.second);

            const int frag_n_n = static_cast<int>(frag.src_row_indices.size());
            frag.frag_nonmortar_gtdofs.resize(frag_n_n);
            frag.frag_D.resize(frag_n_n);
            for (int k = 0; k < frag_n_n; ++k)
            {
                const int i_src = frag.src_row_indices[k];
                frag.frag_nonmortar_gtdofs[k] =
                    lpb.block.nonmortar_gtdofs[i_src];
                frag.frag_D[k] = lpb.block.D(i_src);
            }
            all_fragments.push_back(std::move(frag));
        }
    }

    //------------------------------------------------------------------
    // Stage 2 — count and pack per-destination streams.
    //
    // Per destination, we concatenate all fragments destined for it
    // into a single int-stream + double-stream. The Alltoallv counts
    // are these per-destination byte/element totals.
    //------------------------------------------------------------------
    std::vector<int> send_counts_int(m_nranks, 0);
    std::vector<int> send_counts_dbl(m_nranks, 0);
    std::vector<int> send_n_frags(m_nranks, 0);

    for (const auto& frag : all_fragments)
    {
        const int n_n_f = static_cast<int>(frag.frag_nonmortar_gtdofs.size());
        const int n_m   = frag.src_block->NumMortarKept();

        // Count nnz in the row-sliced CSR by walking source CSR rows
        // selected by src_row_indices.
        int nnz_f = 0;
        const int* src_I = frag.src_block->A_m.GetI();
        for (int k = 0; k < n_n_f; ++k)
        {
            const int i_src = frag.src_row_indices[k];
            nnz_f += src_I[i_src + 1] - src_I[i_src];
        }

        // Per-fragment ints: header + nm_gtdofs + m_gtdofs + I + J.
        const int frag_ints = kBlockHeaderInts + n_n_f + n_m
                               + (n_n_f + 1) + nnz_f;
        // Per-fragment doubles: A_m data (nnz_f) + D (n_n_f).
        const int frag_dbls = nnz_f + n_n_f;

        send_counts_int[frag.dest_rank] += frag_ints;
        send_counts_dbl[frag.dest_rank] += frag_dbls;
        send_n_frags[frag.dest_rank]    += 1;
    }

    // Compute send displs.
    std::vector<int> send_displs_int(m_nranks, 0);
    std::vector<int> send_displs_dbl(m_nranks, 0);
    int total_send_int = 0;
    int total_send_dbl = 0;
    for (int r = 0; r < m_nranks; ++r)
    {
        send_displs_int[r] = total_send_int;
        send_displs_dbl[r] = total_send_dbl;
        total_send_int += send_counts_int[r];
        total_send_dbl += send_counts_dbl[r];
    }

    std::vector<long long> send_int_pack(total_send_int);
    std::vector<double>    send_dbl_pack(total_send_dbl);

    // Per-destination cursors.
    std::vector<int> int_cursor = send_displs_int;
    std::vector<int> dbl_cursor = send_displs_dbl;

    // Walk fragments again and emit into per-destination slots.
    for (const auto& frag : all_fragments)
    {
        const int n_n_f = static_cast<int>(frag.frag_nonmortar_gtdofs.size());
        const int n_m   = frag.src_block->NumMortarKept();

        const int* src_I    = frag.src_block->A_m.GetI();
        const int* src_J    = frag.src_block->A_m.GetJ();
        const double* src_V = frag.src_block->A_m.GetData();

        // First pass: build the fragment-local CSR I row-pointers,
        // and accumulate nnz_f.
        std::vector<int> frag_I(n_n_f + 1, 0);
        for (int k = 0; k < n_n_f; ++k)
        {
            const int i_src = frag.src_row_indices[k];
            frag_I[k + 1] = frag_I[k]
                + (src_I[i_src + 1] - src_I[i_src]);
        }
        const int nnz_f = frag_I[n_n_f];

        const int dest = frag.dest_rank;
        int& iw = int_cursor[dest];
        int& dw = dbl_cursor[dest];

        // Header (9 longs).
        const auto m_lbl = PackLabel16(frag.mortar_label);
        const auto n_lbl = PackLabel16(frag.nonmortar_label);
        send_int_pack[iw + 0] = (frag.geometry_kind == "quad") ? 0 : 1;
        send_int_pack[iw + 1] = AxisPairIdx(frag.axis_pair);
        send_int_pack[iw + 2] = m_lbl.first;
        send_int_pack[iw + 3] = m_lbl.second;
        send_int_pack[iw + 4] = n_lbl.first;
        send_int_pack[iw + 5] = n_lbl.second;
        send_int_pack[iw + 6] = n_n_f;
        send_int_pack[iw + 7] = n_m;
        send_int_pack[iw + 8] = nnz_f;

        // nonmortar_gtdofs.
        for (int k = 0; k < n_n_f; ++k)
        {
            send_int_pack[iw + kBlockHeaderInts + k] =
                frag.frag_nonmortar_gtdofs[k];
        }
        // mortar_gtdofs (full set, unmodified).
        for (int j = 0; j < n_m; ++j)
        {
            send_int_pack[iw + kBlockHeaderInts + n_n_f + j] =
                frag.src_block->mortar_gtdofs[j];
        }
        // CSR I.
        for (int k = 0; k < n_n_f + 1; ++k)
        {
            send_int_pack[iw + kBlockHeaderInts + n_n_f + n_m + k] =
                frag_I[k];
        }
        // CSR J — walk source rows in src_row_indices order.
        int j_out = 0;
        for (int k = 0; k < n_n_f; ++k)
        {
            const int i_src = frag.src_row_indices[k];
            for (int idx = src_I[i_src]; idx < src_I[i_src + 1]; ++idx)
            {
                send_int_pack[iw + kBlockHeaderInts + n_n_f + n_m
                              + (n_n_f + 1) + j_out] = src_J[idx];
                ++j_out;
            }
        }

        iw += kBlockHeaderInts + n_n_f + n_m + (n_n_f + 1) + nnz_f;

        // Doubles: A_m data (in same order as J), then D.
        int v_out = 0;
        for (int k = 0; k < n_n_f; ++k)
        {
            const int i_src = frag.src_row_indices[k];
            for (int idx = src_I[i_src]; idx < src_I[i_src + 1]; ++idx)
            {
                send_dbl_pack[dw + v_out] = src_V[idx];
                ++v_out;
            }
        }
        dw += nnz_f;
        for (int k = 0; k < n_n_f; ++k)
        {
            send_dbl_pack[dw + k] = frag.frag_D[k];
        }
        dw += n_n_f;
    }

    // Verify cursors landed exactly at the next destination's start.
    for (int r = 0; r < m_nranks; ++r)
    {
        const int expected_int_end = send_displs_int[r] + send_counts_int[r];
        const int expected_dbl_end = send_displs_dbl[r] + send_counts_dbl[r];
        MFEM_ASSERT(int_cursor[r] == expected_int_end,
                    "RoutePairBlocksToRowOwners: int pack cursor mismatch "
                    "for dest " << r << " (expected "
                    << expected_int_end << ", got " << int_cursor[r] << ")");
        MFEM_ASSERT(dbl_cursor[r] == expected_dbl_end,
                    "RoutePairBlocksToRowOwners: dbl pack cursor mismatch "
                    "for dest " << r);
    }

    //------------------------------------------------------------------
    // Stage 3 — exchange counts (per-rank Alltoall) so receivers
    // know how big to size their recv buffers.
    //------------------------------------------------------------------
    std::vector<int> recv_counts_int(m_nranks, 0);
    std::vector<int> recv_counts_dbl(m_nranks, 0);
    MPI_Alltoall(send_counts_int.data(), 1, MPI_INT,
                 recv_counts_int.data(), 1, MPI_INT, m_comm);
    MPI_Alltoall(send_counts_dbl.data(), 1, MPI_INT,
                 recv_counts_dbl.data(), 1, MPI_INT, m_comm);

    std::vector<int> recv_displs_int(m_nranks, 0);
    std::vector<int> recv_displs_dbl(m_nranks, 0);
    int total_recv_int = 0, total_recv_dbl = 0;
    for (int r = 0; r < m_nranks; ++r)
    {
        recv_displs_int[r] = total_recv_int;
        recv_displs_dbl[r] = total_recv_dbl;
        total_recv_int += recv_counts_int[r];
        total_recv_dbl += recv_counts_dbl[r];
    }

    std::vector<long long> recv_int_pack(total_recv_int);
    std::vector<double>    recv_dbl_pack(total_recv_dbl);

    //------------------------------------------------------------------
    // Stage 4 — exchange the actual streams via Alltoallv on m_comm.
    //------------------------------------------------------------------
    MPI_Alltoallv(send_int_pack.data(), send_counts_int.data(),
                  send_displs_int.data(), MPI_LONG_LONG,
                  recv_int_pack.data(), recv_counts_int.data(),
                  recv_displs_int.data(), MPI_LONG_LONG,
                  m_comm);
    MPI_Alltoallv(send_dbl_pack.data(), send_counts_dbl.data(),
                  send_displs_dbl.data(), MPI_DOUBLE,
                  recv_dbl_pack.data(), recv_counts_dbl.data(),
                  recv_displs_dbl.data(), MPI_DOUBLE,
                  m_comm);

    //------------------------------------------------------------------
    // Stage 5 — unpack received fragments into per-bucket lists.
    //
    // Bucket key: (axis_pair_name, mortar_label, nonmortar_label,
    // geom_kind). Multiple fragments may share a bucket if multiple
    // source ranks contributed rows for the same (axis, mortar,
    // nonmortar, geom). Each unpacked fragment becomes a
    // FaceMortarPairBlock with build-mode A_m → Finalize(), then the
    // bucket's fragments are merged via the gtdof-keyed accumulator.
    //------------------------------------------------------------------
    using BucketKey = std::tuple<std::string, std::string,
                                  std::string, std::string>;
    std::map<BucketKey, std::vector<FaceMortarPairBlock>> per_bucket;

    long long ip = 0, dp = 0;
    while (ip < static_cast<long long>(total_recv_int))
    {
        const long long* hdr = recv_int_pack.data() + ip;
        const int geom_kind     = static_cast<int>(hdr[0]);
        const int axis_idx      = static_cast<int>(hdr[1]);
        const std::string m_lbl = UnpackLabel16(hdr[2], hdr[3]);
        const std::string n_lbl = UnpackLabel16(hdr[4], hdr[5]);
        const int n_n = static_cast<int>(hdr[6]);
        const int n_m = static_cast<int>(hdr[7]);
        const int nnz = static_cast<int>(hdr[8]);

        FaceMortarPairBlock blk;
        blk.nonmortar_face_name = n_lbl;
        blk.mortar_face_name    = m_lbl;
        blk.nonmortar_gtdofs.SetSize(n_n);
        blk.mortar_gtdofs.SetSize(n_m);
        blk.D.SetSize(n_n);
        blk.A_m = mfem::SparseMatrix(n_n, n_m);

        for (int i = 0; i < n_n; ++i)
        {
            blk.nonmortar_gtdofs[i] = static_cast<int>(
                recv_int_pack[ip + kBlockHeaderInts + i]);
        }
        for (int j = 0; j < n_m; ++j)
        {
            blk.mortar_gtdofs[j] = static_cast<int>(
                recv_int_pack[ip + kBlockHeaderInts + n_n + j]);
        }

        // Reconstruct A_m via Add() walking the packed CSR.
        const long long* A_I_pack = recv_int_pack.data()
            + ip + kBlockHeaderInts + n_n + n_m;
        const long long* A_J_pack = A_I_pack + (n_n + 1);
        for (int i = 0; i < n_n; ++i)
        {
            const long long row_start = A_I_pack[i];
            const long long row_end   = A_I_pack[i + 1];
            for (long long idx = row_start; idx < row_end; ++idx)
            {
                const int j = static_cast<int>(A_J_pack[idx]);
                const double v = recv_dbl_pack[dp + idx];
                blk.A_m.Add(i, j, v);
            }
        }
        blk.A_m.Finalize();

        for (int i = 0; i < n_n; ++i)
        {
            blk.D(i) = recv_dbl_pack[dp + nnz + i];
        }

        const std::string geom = (geom_kind == 0) ? "quad" : "tri";
        per_bucket[BucketKey(AxisPairName(axis_idx), m_lbl, n_lbl, geom)]
            .push_back(std::move(blk));

        ip += kBlockHeaderInts + n_n + n_m + (n_n + 1) + nnz;
        dp += nnz + n_n;
    }
    MFEM_ASSERT(ip == static_cast<long long>(total_recv_int),
                "RoutePairBlocksToRowOwners: int unpack cursor "
                << ip << " != total_recv_int " << total_recv_int);
    MFEM_ASSERT(dp == static_cast<long long>(total_recv_dbl),
                "RoutePairBlocksToRowOwners: dbl unpack cursor "
                << dp << " != total_recv_dbl " << total_recv_dbl);

    //------------------------------------------------------------------
    // Stage 6 — merge fragments within each bucket via gtdof-keyed
    // accumulation (§P4.8.10). This handles shared nonmortar DOFs at
    // tile boundaries — different source ranks may both have
    // contributed rows for the same nonmortar gtdof in the same
    // bucket, and their A_m / D entries must SUM, not concatenate.
    //
    // The lambda is identical to Batch L's MergeBlocks. The semantic
    // change in Batch N is upstream (which fragments arrive here),
    // not in the merge itself.
    //------------------------------------------------------------------
    auto MergeBlocks = [](const std::vector<FaceMortarPairBlock>& parts)
        -> FaceMortarPairBlock
    {
        if (parts.size() == 1) { return parts[0]; }
        FaceMortarPairBlock out;
        out.nonmortar_face_name = parts[0].nonmortar_face_name;
        out.mortar_face_name    = parts[0].mortar_face_name;

        std::map<int, int> nm_gtdof_to_row;
        std::map<int, int> m_gtdof_to_col;
        for (const auto& p : parts)
        {
            for (int i = 0; i < p.NumNonmortarKept(); ++i)
            {
                const int g = p.nonmortar_gtdofs[i];
                if (nm_gtdof_to_row.find(g) == nm_gtdof_to_row.end())
                {
                    const int next = static_cast<int>(nm_gtdof_to_row.size());
                    nm_gtdof_to_row[g] = next;
                }
            }
            for (int j = 0; j < p.NumMortarKept(); ++j)
            {
                const int g = p.mortar_gtdofs[j];
                if (m_gtdof_to_col.find(g) == m_gtdof_to_col.end())
                {
                    const int next = static_cast<int>(m_gtdof_to_col.size());
                    m_gtdof_to_col[g] = next;
                }
            }
        }
        const int merged_n_n = static_cast<int>(nm_gtdof_to_row.size());
        const int merged_n_m = static_cast<int>(m_gtdof_to_col.size());

        out.nonmortar_gtdofs.SetSize(merged_n_n);
        out.mortar_gtdofs.SetSize(merged_n_m);
        for (const auto& kv : nm_gtdof_to_row)
        {
            out.nonmortar_gtdofs[kv.second] = kv.first;
        }
        for (const auto& kv : m_gtdof_to_col)
        {
            out.mortar_gtdofs[kv.second] = kv.first;
        }

        out.D.SetSize(merged_n_n);
        out.D = 0.0;
        out.A_m = mfem::SparseMatrix(merged_n_n, merged_n_m);

        for (const auto& p : parts)
        {
            const int pn = p.NumNonmortarKept();
            const int pm = p.NumMortarKept();

            std::vector<int> row_map(pn);
            for (int i = 0; i < pn; ++i)
            {
                row_map[i] = nm_gtdof_to_row.at(p.nonmortar_gtdofs[i]);
            }
            std::vector<int> col_map(pm);
            for (int j = 0; j < pm; ++j)
            {
                col_map[j] = m_gtdof_to_col.at(p.mortar_gtdofs[j]);
            }

            for (int i = 0; i < pn; ++i)
            {
                out.D(row_map[i]) += p.D(i);
            }
            const int* p_I    = p.A_m.GetI();
            const int* p_J    = p.A_m.GetJ();
            const double* p_V = p.A_m.GetData();
            for (int i = 0; i < pn; ++i)
            {
                const int mr = row_map[i];
                for (int idx = p_I[i]; idx < p_I[i + 1]; ++idx)
                {
                    const int j = p_J[idx];
                    out.A_m.Add(mr, col_map[j], p_V[idx]);
                }
            }
        }
        out.A_m.Finalize();
        return out;
    };

    for (auto& kv : per_bucket)
    {
        const auto& key = kv.first;
        LocalPairBlock lpb;
        lpb.axis_pair       = std::get<0>(key);
        lpb.mortar_label    = std::get<1>(key);
        lpb.nonmortar_label = std::get<2>(key);
        lpb.geometry_kind   = std::get<3>(key);
        lpb.block = MergeBlocks(kv.second);
        m_gathered_pair_blocks.push_back(std::move(lpb));
    }
}

}  // namespace mortar_pbc
