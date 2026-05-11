// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/boundary_3d.py`'s
// BoundaryClassifier3D class. Pure helpers (boundary-tag dispatch,
// edge-label composition, CCW reordering) live in
// boundary_helpers_3d.{hpp,cpp}; this header carries the
// MFEM-aware, MPI-collective class itself.
//
// What it does
// ------------
// Given a 3D ParMesh + 3D vector ParFiniteElementSpace (vdim=3, P1),
// construct at __init__ time:
//   * 8  CornerInfo3D records (one per box vertex)
//   * 12 EdgeInfo3D   records (4 edges per axis × 3 axes)
//   * 6  FaceInfo3D   records (one per box face) with face-element
//                     lists already populated as QuadFaceElement /
//                     TriFaceElement objects with sentinel-tagged
//                     gtdofs and Wohlmuth boundary tags.
//
// All 3 catalogues are fully replicated: every rank holds the same
// classification — same data on rank 0 and rank N-1 — so downstream
// constraint assembly is rank-symmetric (architecture §10.4).
//
// Constructor cost: one ParSubMesh build + several Allgatherv calls
// + bounded local work. Done once at init time; not on the hot path.
//
// References
// ----------
//   * MORTAR_PBC_ARCHITECTURE.md §11.7 (cross-rank keying via snap-coord)
//   * MORTAR_PBC_ARCHITECTURE.md §10.4 (collective rank-symmetry rule)

#pragma once

#include "tile_partition_3d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Classify the boundary of a 3D ParMesh into corners / edges /
 *        faces, with sentinel-tagged face elements ready for the
 *        face-mortar assemblers.
 *
 * @details Constructs the classification at construction time. After
 * construction the per-component catalogues are accessible via
 * Corners(), Edges(), Faces(); each is a std::map keyed by label
 * string. Labels follow the conventions in boundary_helpers_3d.hpp:
 * 8 corner labels ("blf", "brf", ..., "trb"); 12 edge labels of form
 * "{axis}-{face1}-{face2}"; 6 face labels ("bottom", "top", "front",
 * "back", "left", "right").
 *
 * Construction is **collective on the parent mesh's MPI communicator**.
 * After construction, all read accessors are local and rank-symmetric.
 *
 * @par Lifetime
 * The classifier holds **non-owning references** to `pmesh` and `fes`.
 * Caller must ensure both outlive the classifier.
 *
 * @par GPU
 * The classifier itself is host-only (it operates on parent-mesh
 * topology, attribute lists, and TDOF maps — no field data).
 * Downstream constraint assembly may be GPU-parallel; the
 * classification step is not on any inner loop.
 *
 * @par Mesh requirements (Phase 4 scope)
 *   - 3D mesh (Dimension() == 3)
 *   - Vector H1 FE space with vdim == 3
 *   - Order 1 (linear) for Phase 4 — higher order is Phase 6+ via LOR
 *   - Axis-aligned box-shaped RVE (boundary attributes 1..6 each
 *     correspond to one axis-extreme face of the bounding box).
 *     Mesh attributes need NOT follow any particular ordering — the
 *     classifier discovers attr -> face-label mapping at runtime by
 *     inspecting actual boundary-element coordinates (architecture
 *     §11.7.2).
 *
 * Failures (non-3D mesh, wrong vdim, wrong order, non-axis-aligned
 * boundary, missing or extra corners/edges/faces) abort via
 * MFEM_VERIFY / MFEM_ABORT with a diagnostic message.
 *
 * @see CornerInfo3D, EdgeInfo3D, FaceInfo3D in types_3d.hpp.
 */
class BoundaryClassifier3D
{
public:
    /**
     * @brief Construct and run the full classification (collective).
     *
     * @param pmesh    The 3D parent ParMesh.
     * @param fes      Vector H1, vdim=3, order 1, defined on `pmesh`.
     * @param tol_rel  Relative tolerance for coordinate comparisons.
     *                 Default 1e-9. Absolute tolerance is
     *                 `tol_rel * |bbox_diagonal|`.
     *
     * MPI scope: **collective on `pmesh.GetComm()`** —
     *   - 1 Allreduce (bbox)
     *   - 1 Allgather  (per-rank face-attr findings)
     *   - 1 Allgatherv (per-rank vertex pack — Phase 4.2 / Batch J:
     *     the per-rank face-element pack was removed; face elements
     *     travel via tile-shuffle on `m_boundary_comm` instead)
     *   - 2 Alltoall + 2 Alltoallv on `m_boundary_comm` (tile shuffle)
     *   - 3 Allgather + 2 Allgatherv on `m_boundary_comm`
     *     (per-pair mortar block pack, produced tile-locally)
     *   - 1 Allreduce + 3 Bcast on `m_comm` (fanout of the gathered
     *     blocks to interior ranks for the fair-split row partition)
     *
     * @param pair_match_tol_rel Relative tolerance for face-pair
     *                           centroid matching during
     *                           BuildLocalPairBlocks. Default 1e-9.
     *                           Phase 4.2 / Batch K: matching now
     *                           lives in the classifier (was in the
     *                           constraint builder), so the tolerance
     *                           is configured here.
     */
    BoundaryClassifier3D(mfem::ParMesh& pmesh,
                         mfem::ParFiniteElementSpace& fes,
                         double tol_rel = 1e-9,
                         double pair_match_tol_rel = 1e-9);

    /// Destructor — defined out-of-line in the .cpp where the internal
    /// VertexRecord type is complete (the std::vector<...> member's
    /// destructor instantiation needs it).
    ~BoundaryClassifier3D();

    // Non-copyable / non-movable. The classifier holds references and
    // catalogues that don't survive a default copy meaningfully; it's
    // built once and read.
    BoundaryClassifier3D(const BoundaryClassifier3D&) = delete;
    BoundaryClassifier3D& operator=(const BoundaryClassifier3D&) = delete;

    //==========================================================================
    // Read-only accessors
    //==========================================================================

    /// 8 box-corner records, keyed by 3-letter label ("blf" / "brf" / ...).
    const std::map<std::string, CornerInfo3D>& Corners() const { return m_corners; }
    /// 12 box-edge records, keyed by "{axis}-{face1}-{face2}" label.
    const std::map<std::string, EdgeInfo3D>& Edges() const { return m_edges; }
    /// 6 box-face records, keyed by face label.
    const std::map<std::string, FaceInfo3D>& Faces() const { return m_faces; }

    /// Bounding-box minimum corner (after Allreduce-MIN over all ranks).
    const std::array<double, 3>& BboxMin() const { return m_bbox_min; }
    /// Bounding-box maximum corner (after Allreduce-MAX over all ranks).
    const std::array<double, 3>& BboxMax() const { return m_bbox_max; }
    /// Absolute tolerance: `tol_rel * |bbox_diagonal|`.
    double Tol() const { return m_tol; }

    /// MPI communicator used by this classifier (== parent ParMesh's comm).
    MPI_Comm Comm() const { return m_comm; }

    /// Phase 4.2 / Batch N — this rank's index in `m_comm`.
    int Rank() const { return m_rank; }

    /// Total number of ranks in `m_comm`.
    int NRanks() const { return m_nranks; }

    /// Boundary-only subcommunicator (Phase 4.2 §P4.4.0).
    ///
    /// Returns `MPI_COMM_NULL` on interior ranks. Callers that
    /// invoke collectives on this comm MUST guard with
    /// `IsBoundaryRank()` first — collective calls on a null comm
    /// from an interior rank are undefined behaviour.
    MPI_Comm BoundaryComm() const { return m_boundary_comm; }

    /// True if this rank has at least one boundary element on the
    /// parent ParMesh and therefore participates in `m_boundary_comm`.
    bool IsBoundaryRank() const { return m_boundary_comm != MPI_COMM_NULL; }

    /// This rank's index in the boundary subcomm; -1 on interior ranks.
    int BdyRank() const { return m_bdy_rank; }

    /// Size of the boundary subcomm; -1 on interior ranks (call
    /// `IsBoundaryRank()` first).
    int NBdyRanks() const { return m_n_bdy_ranks; }

    /// The parallel FE space this classifier was built against.
    /// Used by ConstraintBuilder3D::BuildHypreParMatrix to align the
    /// constraint matrix's column partition with the FES's true-DOF
    /// partition (which is determined by METIS, NOT by uniform chunk
    /// splitting).
    mfem::ParFiniteElementSpace& Fes() const { return m_fes; }

    /// Total number of global true-DOFs in the parent FES.
    /// Used by ConstraintBuilder3D to size the global C matrix.
    int NGlobalTdofs() const { return m_n_global_tdofs; }

    /**
     * @brief Phase 4.2 / Batch N — return the rank in `m_comm` that
     *        owns a given gtdof under the FES's true-DOF partition.
     *
     * @details Used by Batch N's row-owner routing: a constraint row
     * derived from nonmortar gtdof `g` is owned by the rank that owns
     * `g` in FES, so that C's row partition aligns with K's column
     * partition (and therefore the saddle-point block matrix's blocks
     * are partition-consistent).
     *
     * Implemented as a binary search on the cached
     * `m_fes_tdof_offsets_all` vector (size `m_nranks + 1`,
     * Allgather'd at construction time).
     *
     * @param gtdof Global true-DOF index. Must be in
     *              `[0, NGlobalTdofs())`.
     * @return The owning rank, in `[0, m_nranks)`.
     */
    int GtdofOwnerRank(int gtdof) const;

    /// Runtime-discovered mapping from MFEM boundary attribute to
    /// canonical face label. Exposed for the constraint builder to walk
    /// face attributes in deterministic order.
    const std::map<int, std::string>& FaceLabelByAttr() const
    {
        return m_face_label_by_attr;
    }

    //==========================================================================
    // Helpers used by the constraint builder
    //==========================================================================

    /**
     * @brief Build a lookup `gtdof_x -> (gtdof_x, gtdof_y, gtdof_z)`.
     *
     * @details ConstraintBuilder3D uses this to expand the
     * primary-component gtdofs stored in
     * `FaceMortarPairBlock::nonmortar_gtdofs` / `mortar_gtdofs` into
     * per-component gtdofs for vdim=3 constraint rows.
     *
     * @return A fresh map on each call (cheap; ~100 entries on a
     *         4×4×4 RVE).
     */
    std::map<int, std::array<int, 3>> GtdofXyzLookup() const;

    /**
     * @brief The 9 mortar-nonmortar edge pairs.
     *
     * @return Vector of `(axis, mortar_label, nonmortar_label)` tuples.
     *         3 axes × 3 nonmortar edges per axis = 9 pairs.
     *
     * @details For each parametric axis (x, y, z), there is 1 mortar
     * edge (the one with both adjacent faces being nonmortars) and 3
     * nonmortar edges. This pairs the mortar against each nonmortar
     * individually.
     */
    std::vector<std::tuple<std::string, std::string, std::string>>
    EdgePairs() const;

    /**
     * @brief The 3 mortar-nonmortar face pairs.
     *
     * @return Vector of `(axis, mortar_label, nonmortar_label)` tuples
     *         in canonical order: y-pair (top/bottom), x-pair
     *         (right/left), z-pair (back/front).
     */
    std::vector<std::tuple<std::string, std::string, std::string>>
    FacePairs() const;

    /**
     * @brief Phase 5.9 — corner labels lying on the given mesh face
     *        attribute.
     *
     * @param face_attr  Mesh face attribute (1-based, matching MFEM
     *                   convention and `velocity_gradient_bcs.essential_ids`).
     * @return Vector of 3-letter corner labels (e.g., `{"blf",
     *         "brf", "blb", "brb"}` for the bottom face). Empty if
     *         `face_attr` is not a known boundary attribute on
     *         this classifier.
     *
     * @details Resolved by label matching: each corner label encodes
     * its membership in the 6 box faces via positional letters
     * (pos 0: 'b'/'t' for bottom/top; pos 1: 'l'/'r' for left/right;
     * pos 2: 'f'/'b' for front/back). The face attribute is first
     * mapped to its label via `LabelForMeshAttribute`; then the
     * corners are filtered by the corresponding positional letter.
     *
     * For a topologically axis-aligned box (the classifier's
     * precondition), each face attribute returns exactly 4 corners.
     * Replicated state — same answer on every rank.
     */
    std::vector<std::string> CornersOnFaceAttribute(int face_attr) const;

    /**
     * @brief Phase 5.9 — label of the periodic pair partner.
     *
     * @param label  One of the 6 face labels (`"bottom"`, `"top"`,
     *               `"left"`, `"right"`, `"front"`, `"back"`).
     * @return The label of the opposite face in the same pair
     *         (`"bottom"`↔`"top"`, `"left"`↔`"right"`,
     *         `"front"`↔`"back"`). Empty string if `label` is not
     *         one of the 6 recognized face labels.
     *
     * @details The mapping is fixed by the cuboid topology and
     * doesn't depend on classifier state — but exposed as a method
     * (not a free function) for consistency with the rest of the
     * label-handling API.
     */
    std::string PairPartnerLabel(const std::string& label) const;
    
    /**
     * @brief Phase 5.9 — test whether two mesh attributes are
     *        periodic pair partners.
     *
     * @param attr_a  First mesh face attribute.
     * @param attr_b  Second mesh face attribute.
     * @return true iff `attr_a` and `attr_b` are on opposite sides
     *         of the same spatial axis (e.g., the left and right
     *         face attributes for the x-axis pair).
     *
     * @details Convenience composition:
     * `MeshAttributeForLabel(PairPartnerLabel(LabelForMeshAttribute(a)))
     *  == b`. Returns false (rather than asserting) if either attr is
     * unknown to the classifier.
     */
    bool ArePaired(int attr_a, int attr_b) const;

    /**
     * @brief Phase 5.9 — reverse lookup: face label → mesh attribute.
     *
     * @param label  One of the 6 face labels. (Corner labels and
     *               edge labels return -1.)
     * @return Mesh face attribute number (1-based) for that label,
     *         or -1 if the label is not in the classifier's
     *         attr↔label table.
     *
     * @details Linear scan over the (at most 6) entries of
     * `m_face_label_by_attr`. The inverse map isn't stored
     * explicitly because the table is tiny and constructed once.
     */
    int MeshAttributeForLabel(const std::string& label) const;

    /**
     * @brief Phase 5.9 — forward lookup: mesh attribute → face label.
     *
     * @param attr  Mesh face attribute (1-based).
     * @return Face label string (`"bottom"`, `"top"`, etc.), or
     *         empty string if the attribute is not a known boundary
     *         face attribute.
     *
     * @details Public accessor over the private
     * `m_face_label_by_attr` map. Empty-string return (rather than
     * abort) lets callers detect and report the missing-attribute
     * case with their own context-appropriate error message — used
     * by Phase A.4's pair-completeness validator.
     */
    std::string LabelForMeshAttribute(int attr) const;

    /**
     * @brief Phase 5.9 — test whether an integer is a known
     *        boundary face attribute on this classifier.
     *
     * @param attr  Mesh attribute number (1-based).
     * @return true iff `attr` appears as a key in the classifier's
     *         attr↔label map (i.e., it identifies one of the 6 box
     *         faces this classifier was constructed against).
     *
     * @details Cheap presence check; equivalent to
     * `!LabelForMeshAttribute(attr).empty()` but with a slightly
     * clearer call site.
     */
    bool IsBoundaryFaceAttribute(int attr) const;

    /**
     * @brief Phase 5.9 — rank-local TDOFs of the (min, min, min)
     *        anchor corner in all 3 components.
     *
     * @param fes  Vector H1 ParFiniteElementSpace this classifier
     *             was constructed against (or one with matching
     *             ownership partition).
     * @return Up to 3 rank-local TDOF indices, one per spatial
     *         component, for the components owned by this rank.
     *         Empty on ranks that don't own the anchor corner.
     *
     * @details The "blf" corner — `(bbox_min[0], bbox_min[1],
     * bbox_min[2])` — is by classifier convention the kinematic
     * anchor point for mortar PBC. Pinning all 3 components at this
     * corner unconditionally removes the 3 translation rigid-body
     * modes regardless of what the user specified for the broader
     * corner-pinning set in `[[BCs.periodic_bcs]]`.
     *
     * Ownership is tested via the existing `GtdofOwnerRank` binary
     * search; rank-local TDOFs are computed by subtracting
     * `fes.GetMyTDofOffset()` from the global TDOFs.
     *
     * @par MPI scope
     * Local. The cumulative anchor TDOF count across all ranks is
     * exactly 3 (one per component, owned by exactly one rank each).
     */
    mfem::Array<int> AnchorCornerTDofs(
        const mfem::ParFiniteElementSpace& fes) const;

    /**
     * @brief Human-readable diagnostic summary. Suitable for rank-0
     *        printing.
     */
    std::string Summary() const;

    //==========================================================================
    // Phase 4.2 — tile-shuffled face elements
    //==========================================================================

    /**
     * @brief One face element after the Phase 4.2 tile-shuffle.
     *
     * @details The classifier tile-shuffles each rank's local boundary
     * face elements on `m_boundary_comm` so each tile-owning rank
     * receives exactly the elements whose parametric centroid falls
     * into its tile. After the shuffle, this rank holds a
     * `std::vector<ShuffledFaceElement>` listing only the elements
     * routed to it.
     *
     * Mortar/nonmortar partners route identically (same parametric
     * centroid modulo period), so per-pair matching becomes
     * tile-local with no further communication.
     *
     * Phase 4.2 / Batch H exposes this as a read-only diagnostic
     * (validated via `test_boundary_classifier_3d`); Batch I will
     * wire it into the constraint builder's per-pair matching.
     */
    struct ShuffledFaceElement
    {
        /// Original boundary attribute on the parent ParMesh.
        int parent_attr = 0;
        /// "quad" or "tri" — geometry of the face element.
        std::string geometry_kind;
        /// 3 (tri) or 4 (quad) snap-keys identifying the face vertices.
        /// Cross-rank-stable identity per §11.7 of the architecture doc.
        std::vector<std::array<long long, 3>> snap_keys;
        /// (n × 3) physical coordinates of the face vertices.
        mfem::DenseMatrix coords;
        /// Axis-pair this face belongs to ("x", "y", or "z").
        /// Derived from the face's perpendicular axis via FaceAxes().
        std::string axis_pair;
        /// Tile (i, j) in the axis-pair's grid that this element
        /// landed in. Always equal to
        /// `m_tile_partition.OwnerRank(axis_pair, centroid)`'s decoded
        /// `(tile_i, tile_j)` on the receiving rank.
        int tile_i = -1;
        int tile_j = -1;
        /// Source rank (in `m_boundary_comm`) — for debugging only.
        int source_bdy_rank = -1;
    };

    /**
     * @brief Read-only access to this rank's tile-shuffled face elements.
     *
     * @return Empty if this rank is interior (`!IsBoundaryRank()`),
     *         otherwise the elements whose centroids fall into a
     *         tile owned by this rank in `m_boundary_comm`.
     *
     * @details The shuffle was performed once during construction
     * (Phase 4.2 §P4.4.4 step 5); this is a free read accessor.
     */
    const std::vector<ShuffledFaceElement>& TileShuffledFaceElements() const
    {
        return m_tile_shuffled_face_elements;
    }

    /**
     * @brief Read-only access to the deterministic tile partition.
     *
     * @return Reference to the per-rank `TilePartition3D` instance.
     *         Only valid on boundary ranks; aborting on interior ranks
     *         is a contract violation.
     */
    const TilePartition3D& TilePartition() const
    {
        MFEM_VERIFY(m_tile_partition != nullptr,
                    "BoundaryClassifier3D::TilePartition: this rank is "
                    "interior (no TilePartition3D was constructed). "
                    "Guard with IsBoundaryRank() first.");
        return *m_tile_partition;
    }

    //==========================================================================
    // Phase 4.2 / Batch I — pre-matched per-pair mortar blocks
    //==========================================================================

    /**
     * @brief One pre-matched face-mortar block, keyed by the
     *        face-pair and geometry it came from.
     *
     * @details Phase 4.1 had `ConstraintBuilder3D::ScatterFacePair`
     * call `MatchConformingFacePairs` + `AssemblePairConforming`
     * directly against `face.quad_elements` / `face.tri_elements`
     * (which were globally complete after AllGatherv). Phase 4.2
     * moves that work into the classifier so it runs *tile-locally*
     * on the receiver of the tile-shuffle. The classifier then
     * AllGatherv's the resulting blocks across `m_boundary_comm`
     * so every boundary rank holds the full set; the constraint
     * builder reads them via `PairBlocks()` and scatters them.
     *
     * The block AllGather is strictly smaller than the face-element
     * AllGatherv it replaces because (a) only matched (mortar,
     * nonmortar) pairs produce blocks (interior face elements alone
     * don't), and (b) the dense matrices store match products
     * (`A_m`) and lumped diagonals (`D`), not raw vertex coords.
     *
     * @par Phase 4.2.B follow-up
     * The block AllGather still has O(total_blocks) per-rank memory.
     * The asymptotic scaling fix (AllToAllv-to-row-owner + nonmortar-
     * DOF-aligned row partition) is Batch J. This batch lifts the
     * matching out of the constraint builder and removes the
     * face-element AllGatherv; the block AllGather is the
     * next-bottleneck.
     */
    struct LocalPairBlock
    {
        /// Axis-pair this block belongs to ("x", "y", or "z").
        std::string axis_pair;
        /// Mortar face label ("top", "right", "back").
        std::string mortar_label;
        /// Nonmortar face label ("bottom", "left", "front").
        std::string nonmortar_label;
        /// "quad" or "tri" — the geometry of the face elements
        /// that produced this block.
        std::string geometry_kind;
        /// The assembled pair block (`A_m`, `D`, gtdof arrays).
        FaceMortarPairBlock block;
    };

    /**
     * @brief Read-only access to the gathered face-mortar pair blocks.
     *
     * @return Empty if this rank is interior; otherwise the full set
     *         of (axis_pair, mortar_label, nonmortar_label, geom)
     *         blocks contributed across all boundary ranks.
     *
     * @details Each (axis_pair, mortar, nonmortar, geometry) tuple
     * maps to **at most one** block in this list. A 4×4×4 hex RVE
     * yields 3 entries (one per axis-pair, all `geometry_kind=="quad"`);
     * a tet RVE yields 3 entries with `"tri"`; a mixed mesh yields up
     * to 6 entries.
     */
    const std::vector<LocalPairBlock>& PairBlocks() const
    {
        return m_gathered_pair_blocks;
    }

private:
    //==========================================================================
    // Construction-time helpers (all collective unless noted otherwise)
    //==========================================================================

    /// Compute global RVE bounding box via Allreduce. [collective]
    void ComputeBbox();

    /// Discover attr -> face-label by inspecting boundary-element
    /// coords. Locally per-rank; merged via Allgather. [collective]
    void DiscoverFaceLabelByAttr();

    /// Build a single ParSubMesh covering the full boundary. [collective]
    void BuildBoundarySubmesh();

    /// Walk submesh elements (purely as a vertex-discovery pass),
    /// gather per-rank vertex records, Allgatherv across `m_comm`,
    /// dedup by snap-coord key. Phase 4.2 / Batch J: face-element
    /// records are NOT gathered here anymore — they travel via
    /// `TileShuffleFaceElements` on `m_boundary_comm`. The vertex
    /// catalogue is still globally replicated (corner / edge
    /// classification needs it). [collective]
    void GatherBoundaryRecords();

    /// Identify the 8 corner vertices and build CornerInfo3D records. [local]
    void BuildCorners();

    /// Identify the 12 box edges and build EdgeInfo3D records. [local]
    void BuildEdges();

    /// Build 6 FaceInfo3D records with sentinel-tagged face-element
    /// lists. [local]
    void BuildFaces();

    /// Phase 4.2 / Batch H — perform the tile-partitioned face-element
    /// shuffle on `m_boundary_comm`. Pack local face elements per
    /// destination tile (using `m_tile_partition`), AllToAllv on
    /// `m_boundary_comm`, and store the received per-rank tile-local
    /// elements in `m_tile_shuffled_face_elements`.
    ///
    /// Runs in parallel with the existing `GatherBoundaryRecords`
    /// for now; downstream consumers (BuildFaces / ConstraintBuilder)
    /// still read the AllGather'd records. Switching to the
    /// tile-shuffled path is Batch I.
    ///
    /// MPI scope: collective on `m_boundary_comm`. No-op on interior
    /// ranks. [collective on bdry comm]
    void TileShuffleFaceElements();

    /// Phase 4.2 / Batch I — assemble the per-pair mortar blocks
    /// tile-locally from `m_tile_shuffled_face_elements`. Output goes
    /// into `m_local_pair_blocks` (this rank's contribution).
    ///
    /// Algorithm: walk `m_tile_shuffled_face_elements`; bucket by
    /// (axis_pair, mortar/nonmortar, geometry_kind, tile_idx);
    /// for each (axis, geom) bucket on each tile owned by this rank,
    /// run `MatchConformingFacePairs` + `AssemblePairConforming` on
    /// the tile-local mortar / nonmortar element vectors; store the
    /// resulting `FaceMortarPairBlock` (with geometry_kind metadata).
    ///
    /// Concatenation across the rank's tiles within a single
    /// (axis, mortar, nonmortar, geom) bucket: each tile contributes
    /// its own block; the per-tile blocks share the same
    /// (mortar, nonmortar) labels and geometry. They get concatenated
    /// into a single `LocalPairBlock` per bucket — `D` gets stacked,
    /// `A_m` gets row-stacked, and the gtdof arrays append.
    ///
    /// MPI scope: local (no collectives). [local on bdry rank]
    void BuildLocalPairBlocks();

    /// Phase 4.2 / Batch N — route per-pair blocks to the rank that
    /// owns each row's nonmortar gtdof under the FES TDOF partition.
    ///
    /// @details This replaces Batch I/K's
    /// `GatherPairBlocksAcrossBoundary` (which AllGather'd every
    /// block to every boundary rank, then Bcast'd to interior ranks).
    /// The new flow:
    ///   1. Each boundary rank, for each local pair block, groups its
    ///      nonmortar rows by FES owner rank. Each group becomes a
    ///      "block fragment" — same header info (axis_pair, geom,
    ///      labels) and full mortar_gtdofs, but only the subset of
    ///      nonmortar rows / D entries / A_m rows for one destination.
    ///   2. Per-destination fragment streams are packed and exchanged
    ///      via MPI_Alltoallv on `m_comm` (must be `m_comm`, not
    ///      `m_boundary_comm`, because nonmortar gtdofs may be FES-
    ///      owned by interior ranks).
    ///   3. Receiving ranks unpack fragments and merge same-bucket
    ///      contributions via gtdof-keyed accumulation (preserving
    ///      §P4.8.10's correctness for shared DOFs).
    ///
    /// After this runs, every rank's `m_gathered_pair_blocks`
    /// contains only the block (fragments) whose nonmortar rows fall
    /// within this rank's FES TDOF range. The replicated-on-every-
    /// rank storage of Batches I/K is gone — per-rank memory is now
    /// O(boundary_blocks / n_bdy_ranks).
    ///
    /// MPI scope: collective on `m_comm`.
    ///                  [collective on world]
    void RoutePairBlocksToRowOwners();

    /// Helper for `BuildLocalPairBlocks`: take a list of shuffled
    /// face elements (already filtered to one face_label / one
    /// geometry kind) and convert each into a fully-formed
    /// QuadFaceElement (CCW-reordered, sentinel-rewritten gtdofs).
    /// Looks up vertex gtdofs via `m_snap_key_to_record_idx` +
    /// `m_vertex_records`.
    std::vector<QuadFaceElement> ConvertShuffledToQuads(
        const std::vector<const ShuffledFaceElement*>& shuffled,
        const std::string& face_label,
        const std::map<int, int>& sentinel_class) const;

    /// Sibling of ConvertShuffledToQuads for tri elements.
    std::vector<TriFaceElement> ConvertShuffledToTris(
        const std::vector<const ShuffledFaceElement*>& shuffled,
        const std::string& face_label,
        const std::map<int, int>& sentinel_class) const;

    //==========================================================================
    // Member state — all in m_-prefixed snake_case per ExaConstit
    // developer's guide, *Name Formatting*.
    //==========================================================================

    // Non-owning references to caller-supplied mesh + FE space.
    mfem::ParMesh& m_pmesh;
    mfem::ParFiniteElementSpace& m_fes;

    MPI_Comm m_comm;
    int m_rank = -1;
    int m_nranks = -1;

    // Boundary subcommunicator (Phase 4.2 §P4.4.0 / §P4.4.4).
    //
    // Ranks with at least one boundary element on the parent ParMesh
    // join `m_boundary_comm`; others get `MPI_COMM_NULL`. The rank ID
    // and size relative to this subcomm are cached as
    // `m_bdy_rank` / `m_n_bdy_ranks` (both -1 for interior ranks).
    //
    // Phase 4.1 internals still use `m_comm` (WORLD) for all
    // collectives. Phase 4.2 introduces the subcomm here so it's
    // available for the tile-partitioned AllToAllv path. **Interior
    // ranks must never participate in collectives on `m_boundary_comm`**
    // — they hold `MPI_COMM_NULL` and any such call would be UB.
    MPI_Comm m_boundary_comm = MPI_COMM_NULL;
    int m_bdy_rank = -1;
    int m_n_bdy_ranks = -1;

    // Geometry
    std::array<double, 3> m_bbox_min;
    std::array<double, 3> m_bbox_max;
    double m_tol = 0.0;
    double m_tol_rel = 1e-9;
    double m_pair_match_tol_rel = 1e-9;

    // Runtime-discovered attribute mapping.
    std::map<int, std::string> m_face_label_by_attr;
    std::map<std::string, int> m_face_attr_by_label;

    // Boundary submesh (owning unique_ptr — ParSubMesh is heavy).
    std::unique_ptr<mfem::ParSubMesh> m_bdr_submesh;

    // Internal (gathered, replicated) record buffers — implementation-
    // detail forward declarations live in the .cpp file.
    //
    // Phase 4.2 / Batch J — `FaceElementRecord` and
    // `m_face_element_records` were removed. Face elements no longer
    // flow through the global AllGather; they travel via
    // TileShuffleFaceElements (boundary subcomm) and per-pair
    // mortar blocks via GatherPairBlocksAcrossBoundary.
    struct VertexRecord;
    std::vector<VertexRecord> m_vertex_records;

    // Snap-key (cross-rank vertex identity) -> index into
    // m_vertex_records. Built during gather, used in BuildFaces to
    // resolve face-element vertex identities.
    std::map<std::array<long long, 3>, int> m_snap_key_to_record_idx;

    // Output catalogues.
    std::map<std::string, CornerInfo3D> m_corners;
    std::map<std::string, EdgeInfo3D>   m_edges;
    std::map<std::string, FaceInfo3D>   m_faces;

    // Phase 4.2 / Batch H — tile partition (Strategy B per §P4.4.4).
    // Built once on boundary ranks during construction; null on
    // interior ranks. unique_ptr because TilePartition3D doesn't have
    // a default ctor (it requires bbox + n_bdy_ranks).
    std::unique_ptr<TilePartition3D> m_tile_partition;

    // Phase 4.2 / Batch H — this rank's tile-shuffled face elements.
    // After TileShuffleFaceElements() runs, holds exactly the
    // elements whose parametric centroid falls into a tile owned by
    // this rank in m_boundary_comm. Empty on interior ranks.
    std::vector<ShuffledFaceElement> m_tile_shuffled_face_elements;

    // Phase 4.2 / Batch I — per-pair mortar blocks assembled on this
    // rank from its tile-local face elements. Empty on interior ranks.
    std::vector<LocalPairBlock> m_local_pair_blocks;

    // Phase 4.2 / Batch N — per-pair block fragments routed TO this
    // rank by `RoutePairBlocksToRowOwners()`. After routing, every
    // entry's nonmortar_gtdofs belong to this rank's FES TDOF range.
    // Multiple source ranks may have routed fragments for the same
    // (axis, mortar, nonmortar, geom) bucket; their contributions are
    // merged via gtdof-keyed accumulation during the routing step
    // (preserving §P4.8.10 for shared DOFs). On the producer side,
    // a single `m_local_pair_blocks` entry may be split into up to
    // `m_nranks` fragments (one per destination); each fragment ships
    // only the subset of nonmortar rows it carries.
    //
    // Phase 4.2 / Batches I/K: this used to be the FULLY-replicated
    // (every rank holds every block) gathered set — that's gone.
    std::vector<LocalPairBlock> m_gathered_pair_blocks;

    // Phase 4.2 / Batch N — FES TDOF partition offsets for every
    // rank in `m_comm`. Layout: m_fes_tdof_offsets_all[r] is the
    // first global TDOF owned by rank r, with a sentinel
    // m_fes_tdof_offsets_all[m_nranks] == NGlobalTdofs(). Built at
    // ctor time via Allgather of FES.GetTrueDofOffsets()[0]. Used
    // by GtdofOwnerRank() to dispatch routing destinations.
    std::vector<HYPRE_BigInt> m_fes_tdof_offsets_all;

    // Total global TDOFs. Cached at construction time.
    int m_n_global_tdofs = 0;
};

}  // namespace mortar_pbc
