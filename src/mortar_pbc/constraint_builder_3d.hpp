// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — port of Python `mortar_pbc/constraint_builder_3d.py`.
//
// What this layer does
// --------------------
// `ConstraintBuilder3D` consumes a `BoundaryClassifier3D` (Phase
// 4.1.A Batch B) and the three element-type-specific assemblers
// (Batches A & B from Phase 3) and produces the global mortar-
// periodic constraint matrix `C`.
//
// `C` has shape `(n_constraint_rows, n_global_tdofs)` and encodes:
//
//      C[(k, c), :] · u  =  D[k] u_nonmortar_c[k]
//                         - Σ_l A_m[k, l] u_mortar_c[l]
//                        =  0   (nonmortar/mortar coupling, per spatial
//                                component c ∈ {x, y, z})
//
// This is the orchestration layer that ties together:
//   * The 3D edge mortar (9 pairs: 3 axes × 3 nonmortar edges each
//     paired against 1 mortar edge per axis) — uses
//     `MortarAssembler2D::AssemblePair` with the axis-generic dispatch
//     on `EdgeInfo3D`.
//   * The 3D face mortar (3 pairs: 1 per axis) — uses
//     `QuadFaceMortarAssembler` and `TriFaceMortarAssembler`. Mixed
//     hex+tet faces dispatch by element type and accumulate row-stacked.
//
// Stacking these into one global `C` lets the saddle-point solve
// (next batch in this phase) pick up the 3D periodicity without any
// further structural change.
//
// Design notes
// ------------
//   * **Replicated CSR.** Per the architecture's Phase 4 Round-1 plan
//     ("AllGather"), the classifier's per-face / per-edge records are
//     already replicated on every rank. The constraint builder
//     therefore builds the same global `C` on every rank — no further
//     collectives at constraint-assembly time.
//
//   * **HypreParMatrix conversion is separate.** The replicated
//     `mfem::SparseMatrix` is the natural intermediate form. The
//     `BuildHypreParMatrix` method takes the replicated CSR and
//     produces a distributed `HypreParMatrix` with empty rows on
//     interior ranks — using an `MPI_Allgather` of the per-rank LM
//     row count to compute the row partition. This is the input to
//     the saddle-point solver.
//
//   * **vdim=3 expansion is explicit.** Edge and face mortar blocks
//     index by *scalar* gtdofs (one per node). Each scalar constraint
//     expands to 3 vector constraints by replicating the row across
//     the (x, y, z) gtdofs of the same node, looked up via the
//     classifier's `GtdofXyzLookup()`.
//
//   * **Sentinel handling is upstream.** The classifier already
//     stripped corner/edge sentinels from face-element gtdofs; the
//     face assembler returns `FaceMortarPairBlock` with sentinel
//     rows/cols ALREADY DROPPED. Edge records hold only edge-interior
//     nodes by construction. So this builder treats every gtdof as a
//     real, positive global TDOF index.
//
// Phase 5.9 — Component-restricted PBC filter
// -------------------------------------------
// Filtered overloads of `Build`, `BuildHypreParMatrix`, `NumLocalRows`,
// `NumConstraints`, and `EmitRowFactors` accept a `(active_pair_labels,
// comp_mask)` pair that gates which constraint rows are emitted.
//
//   * `active_pair_labels` — list of MORTAR-SIDE face labels (per the
//     classifier's convention: `"top"`, `"right"`, `"back"`). A face
//     pair is "active" iff its mortar label appears here. The
//     corresponding "active axes" are derived internally:
//
//         "left"/"right"   → "x"
//         "bottom"/"top"   → "y"
//         "front"/"back"   → "z"
//
//     (The function accepts any of the 6 labels for convenience; the
//     caller may pass the mortar side or the nonmortar side and the
//     result is the same set of active axes.) See
//     `ActiveAxesFromPairLabels` in the cpp for the mapping.
//
//   * `comp_mask` — 3-bool array gating per-component row emission.
//     For each kept nonmortar node, only rows for components `c`
//     with `comp_mask[c] == true` are emitted; the row count per
//     node is `count(comp_mask)` instead of `kVDim`.
//
// Active-pair rules:
//   - Face mortars (`m_classifier.FacePairs()`): a pair is emitted
//     iff its axis (`std::get<0>(tup)`) ∈ active_axes.
//   - Edge mortars (`m_classifier.EdgePairs()`): a group is emitted
//     iff BOTH of its perpendicular axes ∈ active_axes. An x-axis
//     edge mortar (edges parallel to x) requires `"y"` AND `"z"`
//     active; analogously for y and z. This is the conservative
//     choice — when both perpendicular axes are active the edges
//     work as before, and when either is dropped the edges are too
//     (avoiding over-constraint of edge nodes whose face-pair
//     correspondences are inconsistent with the user's reduced PBC
//     specification).
//
// The parameter-less overloads (`Build()`, etc.) forward to the
// filtered overloads with all face pairs active and `{true, true,
// true}` for `comp_mask`, exactly reproducing pre-5.9 behavior.
//
// References
// ----------
//   * MORTAR_PBC_ARCHITECTURE.md §11.8 (this layer).
//   * MORTAR_PBC_ARCHITECTURE.md §11.5 (3D edge mortar).
//   * MORTAR_PBC_ARCHITECTURE.md §11.6 (face-mortar geometric matching).
//
// Phase 6.0.F — projector-aware HypreParMatrix path
// -------------------------------------------------
// A second constructor accepts a classifier built on the boundary/LOR
// submesh plus a SurfaceProjector. Matrix rows still come from the
// classifier, but every emitted column is translated to the parent
// volume FES. This keeps the assembled C path aligned with the
// projector-aware MortarConstraintOperator.

#pragma once

#include "boundary_classifier_3d.hpp"
#include "face_mortar_assembler_3d.hpp"
#include "mortar_assembler_2d.hpp"
#include "surface_projector.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Lambda block sub-block partition scheme (Phase 5.11).
 *
 * @details Used by `ConstraintBuilder3D::GetRowSubblockIds` to
 * partition the constraint-row index space into sub-blocks for
 * per-sub-block residual scaling. The mortar_pbc-side enum is
 * deliberately kept distinct from the options-side
 * `::SubblockPartition` so mortar_pbc headers don't pull in
 * `option_parser_v2.hpp` (same pattern as `KrylovType` vs
 * `SaddlePointSolverType`). Translation happens at the
 * `MortarPbcManager` boundary.
 *
 * Partition schemes:
 *   - `FaceEdge` (default): 2 sub-blocks. Sub-block 0 contains all
 *     rows from active edge mortar groups; sub-block 1 contains
 *     all rows from active face mortar pairs. Coarsest physically
 *     meaningful partition; always exposes 2 labels regardless of
 *     filter state (empty sub-blocks possible).
 *   - `PerPair`: one sub-block per ACTIVE mortar pair, in walk order
 *     (edges from `m_classifier.EdgePairs()` first, then faces from
 *     `m_classifier.FacePairs()`). Label count varies with the
 *     Phase 5.9 filter spec; full-XYZ unfiltered yields 9 + 3 = 12
 *     sub-blocks; X-only filter yields 1 (the x-face pair, all
 *     edges dropped).
 */
enum class SubblockPartition
{
    FaceEdge, /**< 2 sub-blocks: edges (0), faces (1). */
    PerPair   /**< One per active edge pair + one per active face pair. */
};

/**
 * @brief Assemble the global mortar-periodic constraint matrix `C`.
 *
 * @details After construction, call `Build()` to produce a replicated
 * `mfem::SparseMatrix` of shape `(n_constraints, n_global_tdofs)`.
 * Optionally call `BuildHypreParMatrix()` to convert to a distributed
 * `HypreParMatrix` for use with the saddle-point solver.
 *
 * The class is **stateless after construction** — no caches between
 * `Build()` calls. Calling `Build()` twice produces equivalent
 * matrices (the constraint matrix only depends on the classifier's
 * already-fixed catalogue).
 *
 * Phase 5.9 — filtered overloads `Build(active_pair_labels, comp_mask)`
 * etc. emit a subset of rows according to the filter, supporting
 * component-restricted PBC (e.g., periodicity in X only for monotonic
 * X-direction loading with stress-free Y/Z).
 *
 * Phase 6 — when constructed with a `SurfaceProjector`, the classifier
 * is interpreted as a boundary/LOR-submesh classifier and matrix
 * columns are parent-volume FES true DOFs. Runtime users therefore see
 * the same column space as `MortarConstraintOperator::Width()`.
 *
 * @par Lifetime
 * Legacy construction holds a non-owning reference to the classifier.
 * Projector construction stores shared ownership of the classifier,
 * projector, and parent FES supplied by the caller.
 *
 * @par MPI scope
 * `Build()` is **local** (no collectives) — every rank builds the
 * same global matrix. `BuildHypreParMatrix()` is **collective** on
 * the classifier's communicator (one `MPI_Allgather` of int row
 * counts).
 */
class ConstraintBuilder3D
{
public:
    /// Vector dimension; locked at 3 for 3D vector elasticity.
    static constexpr int kVDim = 3;

    /**
     * @brief Construct the builder around a fully-classified boundary.
     *
     * @param classifier  Output of `BoundaryClassifier3D`, required.
     *
     * Phase 4.2 / Batch K: the previous `pair_match_tol_rel`
     * parameter was removed. Face-pair matching now happens inside
     * the classifier (`BuildLocalPairBlocks`) rather than in this
     * builder, so the matching tolerance is configured on the
     * classifier itself (its 4th constructor argument). The builder
     * just consumes the pre-matched pair blocks.
     */
    explicit ConstraintBuilder3D(const BoundaryClassifier3D& classifier);

    /**
     * @brief Construct the builder for a boundary/LOR classifier with
     *        parent-FES column translation.
     *
     * @param classifier  Fully-built classifier whose FES is the
     *                    boundary/LOR submesh FES.
     * @param projector   Surface projector translating classifier-side
     *                    submesh true DOFs to parent-volume true DOFs.
     * @param parent_fes  Parent volume FES defining the matrix column
     *                    space and Hypre column partition.
     *
     * @details The builder remains otherwise stateless. `Build()` and
     * `BuildHypreParMatrix()` emit the same rows as the classifier
     * supplies, but every column index is translated through
     * `projector` before insertion. `EmitRowFactors()` is unchanged
     * because it emits geometry-only row metadata, not TDOF columns.
     *
     * @par MPI scope
     * Constructor is local after `classifier` and `projector` have been
     * constructed. Later `BuildHypreParMatrix()` calls remain
     * collective on `classifier->Comm()`.
     */
    ConstraintBuilder3D(
        std::shared_ptr<const BoundaryClassifier3D> classifier,
        std::shared_ptr<const SurfaceProjector> projector,
        std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes);

    // Non-copyable / non-movable: holds a reference and a small set of
    // assemblers.
    ConstraintBuilder3D(const ConstraintBuilder3D&) = delete;
    ConstraintBuilder3D& operator=(const ConstraintBuilder3D&) = delete;

    //==========================================================================
    // Parameter-less (unfiltered) public API — preserves pre-5.9 behavior.
    //==========================================================================

    /**
     * @brief Build the replicated global constraint matrix.
     *
     * @return A `unique_ptr<mfem::SparseMatrix>` of shape
     *         `(NumConstraints(), ParentGlobalTrueVSize())`. Entries
     *         are: diagonal `D[k]` per kept nonmortar row, off-diagonal
     *         `-A_m[k, l]` per (kept nonmortar, kept mortar) pair, all
     *         vdim-replicated per spatial component.
     *
     * @par MPI scope
     * Local — no collective communication. Every rank builds the same
     * matrix.
     *
     * @par Layout
     * Row order: edge constraints first (9 pairs in the order
     * `BoundaryClassifier3D::EdgePairs()` returns), face constraints
     * second (3 pairs in `FacePairs()` order). Within each pair, rows
     * are vdim-replicated per kept nonmortar node.
     *
     * Equivalent to `Build(all_mortar_labels, {true, true, true})`.
     */
    std::unique_ptr<mfem::SparseMatrix> Build() const;

    /**
     * @brief Build a distributed `HypreParMatrix` form of `C`.
     *
     * @details Phase 4.2 / Batch N: the row partition is now derived
     * from the data — each rank owns the constraint rows whose
     * x-component nonmortar gtdof is FES-owned by this rank. The
     * caller no longer specifies `n_lam_local`. Use `NumLocalRows()`
     * if you need the value (e.g. to size a Lagrange-multiplier
     * vector).
     *
     * @return A heap-allocated `HypreParMatrix*`. Caller owns and must
     *         `delete` it.
     *
     * @par MPI scope
     * Collective on `classifier.Comm()`. One `MPI_Allgather` (int).
     *
     * Equivalent to `BuildHypreParMatrix(all_mortar_labels,
     * {true, true, true})`.
     */
    mfem::HypreParMatrix* BuildHypreParMatrix() const;

    /**
     * @brief Phase 4.2 / Batch N — number of constraint rows owned
     *        by this rank under the FES-aligned row partition.
     *
     * @details Computed by running `EmitConstraintTriples` once and
     * counting the emitted rows.
     *
     * Useful for sizing the Lagrange-multiplier `Vector` (the dual
     * variable in the saddle-point system has one entry per local
     * constraint row).
     *
     * Equivalent to `NumLocalRows(all_mortar_labels, {true, true,
     * true})`.
     */
    int NumLocalRows() const;

    /**
     * @brief Number of constraint rows the build will emit.
     *
     * @details Sum over edge pairs of `kVDim × n_interior_nonmortar_nodes`,
     * plus sum over face pairs of `kVDim × n_kept_nonmortar_face_dofs`
     * (using the classifier's pre-computed `interior_gtdofs_x` size).
     *
     * Equivalent to `NumConstraints(all_mortar_labels, {true, true,
     * true})`.
     */
    int NumConstraints() const;

    /**
     * @brief Per-row reference-geometry metadata used by
     *        `MortarPbcManager::UpdateConstraintRHS` to build the
     *        constraint RHS `g`.
     *
     * @param[out] period_signed_per_row  Vector of length
     *                                    `3 * n_local_rows` in
     *                                    row-major layout. For each
     *                                    constraint row i,
     *                                    `period_signed_per_row[3i..3i+3)`
     *                                    is the physical periodic
     *                                    shift vector
     *                                    `(Δ_x·L_x, Δ_y·L_y, Δ_z·L_z)`
     *                                    that the row enforces. For
     *                                    face rows exactly one
     *                                    component is nonzero (the
     *                                    face normal axis); for edge
     *                                    rows the parallel-axis
     *                                    component is zero and the
     *                                    two transverse components
     *                                    can each be nonzero.
     * @param[out] component_index         Per-row spatial component
     *                                    constrained: 0=x, 1=y, 2=z.
     * @param[out] ell_hat                 Per-row Wohlmuth-lumped
     *                                    diagonal weight `D_kk`.
     *
     * @details Phase 5.7.A — previously emitted a single integer
     * axis index per row (`axis_index`). That was correct only for
     * face rows; for edge rows the axis index encoded the
     * edge-parallel axis, which is NOT the periodic jump direction.
     * The `period_signed_per_row` output replaces it and works for
     * both face and edge rows. The downstream g formula in
     * `MortarPbcManager::UpdateConstraintRHS` is now
     *   `g[i] = ell_hat[i] * Σ_k Ḟ̄(c, k) · period_signed_per_row[3i + k]`.
     *
     * Mirrors the row-enumeration pattern of `EmitConstraintTriples`
     * so that emit position k corresponds to constraint matrix row k.
     *
     * Equivalent to `EmitRowFactors(all_mortar_labels, {true, true,
     * true}, ...)`.
     */
    void EmitRowFactors(mfem::Vector& period_signed_per_row,
                        mfem::Array<int>& component_index,
                        mfem::Vector& ell_hat) const;

    //==========================================================================
    // Phase 5.9 — filtered public API
    //==========================================================================

    /**
     * @brief Phase 5.9 — build the replicated `C` with a face-pair
     *        and component filter.
     *
     * @param active_pair_labels  Mortar-side face labels of the pairs
     *                            to include. Any of the 6 face labels
     *                            (`"left"`, `"right"`, `"bottom"`,
     *                            `"top"`, `"front"`, `"back"`) is
     *                            accepted; the function derives the
     *                            set of active axes from these.
     * @param comp_mask           3-bool mask gating per-component
     *                            row emission. `comp_mask[c] == false`
     *                            skips row `c` at every kept nonmortar
     *                            node.
     *
     * @details Face-pair filter: a face pair is emitted iff its axis
     * is in the set of active axes. Edge-mortar filter: an edge group
     * is emitted iff BOTH of its perpendicular axes are active. The
     * comp-mask is applied per-row inside the scatter helpers.
     *
     * The row count is
     *   `count(comp_mask) × (Σ over active edges of n_interior_nodes
     *                       + Σ over active face pairs of n_kept_nm_dofs)`.
     */
    std::unique_ptr<mfem::SparseMatrix> Build(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask) const;

    /// Phase 5.9 — distributed-form `BuildHypreParMatrix` with filter.
    /// See `Build(active_pair_labels, comp_mask)` for filter semantics.
    mfem::HypreParMatrix* BuildHypreParMatrix(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask) const;

    /// Phase 5.9 — local row count under filter. Re-runs the emitter
    /// with the filter and discards buffers; cost is O(local_rows).
    int NumLocalRows(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask) const;

    /// Phase 5.9 — global row count under filter, computed without
    /// running the emitter (cheap, just walks classifier topology).
    int NumConstraints(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask) const;

    /// Phase 5.9 — row-factor emission under filter.
    /// `period_signed_per_row` is still 3 doubles per row in row-
    /// major layout; under filter the row count is reduced and the
    /// per-row content is preserved (same period_signed,
    /// component_index, ell_hat as the unfiltered emission for the
    /// rows that ARE emitted).
    void EmitRowFactors(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask,
        mfem::Vector& period_signed_per_row,
        mfem::Array<int>& component_index,
        mfem::Vector& ell_hat) const;

    //==========================================================================
    // Phase 5.11 — sub-block partition accessor
    //==========================================================================

    /**
     * @brief Phase 5.11 — partition the local lambda row index space
     *        into sub-blocks per the given scheme.
     *
     * @param[in]  partition           Partition scheme — `FaceEdge` (2
     *                                 sub-blocks) or `PerPair` (one
     *                                 per active pair).
     * @param[in]  active_pair_labels  Mortar-side face labels of active
     *                                 pairs (same convention as
     *                                 `Build`/`NumLocalRows`/etc.).
     * @param[in]  comp_mask           3-bool spatial-component mask.
     * @param[out] subblock_labels     Human-readable labels, one per
     *                                 sub-block. Used as column-name
     *                                 stems in `periodic_consistency`
     *                                 output.
     *                                 - `FaceEdge`: always 2 entries
     *                                   `{"edge", "face"}` regardless
     *                                   of filter state.
     *                                 - `PerPair`: one entry per active
     *                                   pair in walk order. Edge
     *                                   labels are `"edge_<nm_label>"`;
     *                                   face labels are
     *                                   `"face_<mortar_label>"`.
     * @param[out] subblock_of_row     Per-row sub-block ID (in
     *                                 `[0, n_subblocks)`). Sized to
     *                                 `NumLocalRows(active_pair_labels,
     *                                 comp_mask)`. Row order matches
     *                                 `EmitConstraintTriples` /
     *                                 `EmitRowFactors` exactly.
     *
     * @details Walks the constraint-row index space in the same order
     * as the emitter:
     *   1. Edge mortar blocks in `m_classifier.EdgePairs()` order,
     *      gated on BOTH perpendicular axes ∈ active_axes. Per kept
     *      (active + row-owned) nonmortar node, emit
     *      `CountActiveComps(comp_mask)` sub-block IDs.
     *   2. Face mortar blocks in `m_classifier.FacePairs()` order,
     *      gated on the pair's axis ∈ active_axes. Within each pair,
     *      quad block first then tri block (matching the emitter's
     *      ScatterFaceBlock order). Per kept nonmortar node, emit
     *      `CountActiveComps(comp_mask)` sub-block IDs.
     *
     * The row-owner filter (edge side) and the pre-routed face-pair
     * convention (face side) match the emitter's behavior exactly,
     * so `subblock_of_row[i]` corresponds to row `i` in the
     * `Build(active_pair_labels, comp_mask)` output. The sub-block
     * ID for a given row depends only on which pair the row came
     * from — all per-component rows from the same nonmortar node
     * share the same sub-block ID.
     *
     * For `FaceEdge` partition: `subblock_labels` is always
     * `{"edge", "face"}` (size 2) even if one or both sub-blocks
     * have no rows under the current filter. This keeps the
     * downstream `periodic_consistency` column set stable across
     * Phase 5.9 spec transitions.
     *
     * For `PerPair` partition: `subblock_labels` contains one entry
     * per ACTIVE pair only. The label count varies under filter; the
     * downstream post-processor must handle column-set changes
     * across spec transitions (see Phase 5.11 plan §10.8).
     */
    void GetRowSubblockIds(
        SubblockPartition partition,
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask,
        std::vector<std::string>& subblock_labels,
        mfem::Array<int>& subblock_of_row) const;

    /**
     * @brief Phase 5.11 — parameter-less forwarder for
     *        `GetRowSubblockIds`. Equivalent to calling with all
     *        mortar labels active and `{true, true, true}` for
     *        `comp_mask` (matches the pre-5.9 default behavior of
     *        the other accessors).
     */
    void GetRowSubblockIds(
        SubblockPartition partition,
        std::vector<std::string>& subblock_labels,
        mfem::Array<int>& subblock_of_row) const;

private:
    /**
     * @brief Append rows for one edge mortar block to the COO buffers.
     *
     * @details `nonmortar_edge.gtdofs_*` index into the per-component
     * arrays directly; the vdim expansion is just the per-c loop.
     *
     * Phase 5.9 — `comp_mask` filters which spatial-component rows
     * are emitted. The `row_offset` advances by `count(comp_mask)`
     * per kept nonmortar node (not by `kVDim`), and the per-component
     * row within a node is determined by the position of `c` in the
     * subsequence of true entries in `comp_mask`. The off-rank skip
     * (row owner ≠ my_rank) and the degenerate D_kk == 0 branch both
     * compose with the filter: they consume `count(comp_mask)` rows
     * worth of `row_offset` (or none, for off-rank skip).
     *
     * @return The new (post-append) row offset.
     */
    int ScatterEdgeBlock(const MortarBlock2D& block,
                         const EdgeInfo3D& nonmortar_edge,
                         const EdgeInfo3D& mortar_edge,
                         const std::array<bool, 3>& comp_mask,
                         std::vector<int>& rows,
                         std::vector<int>& cols,
                         std::vector<double>& vals,
                         int row_offset) const;

    // Note: `ScatterFacePair` was removed in Phase 4.2 / Batch J.
    // The face-pair matching + assembly that used to live here is now
    // performed tile-locally inside `BoundaryClassifier3D::BuildLocalPairBlocks`,
    // and the constraint builder's `Build()` consumes the pre-assembled
    // blocks via `m_classifier.PairBlocks()` and dispatches them
    // through `ScatterFaceBlock` directly.

    /**
     * @brief Append rows for one (already-sentinel-stripped) face mortar
     *        block to the COO buffers.
     *
     * @details `block.nonmortar_gtdofs[k]` is the primary-component (x)
     * gtdof of nonmortar node `k`; the per-component triple is looked
     * up via `m_gtdof_lookup`.
     *
     * Phase 5.9 — `comp_mask` filters which spatial-component rows
     * are emitted; same semantics as in `ScatterEdgeBlock`.
     *
     * @return The new (post-append) row offset.
     */
    int ScatterFaceBlock(const FaceMortarPairBlock& block,
                         const std::array<bool, 3>& comp_mask,
                         std::vector<int>& rows,
                         std::vector<int>& cols,
                         std::vector<double>& vals,
                         int row_offset) const;

    /**
     * @brief Phase 4.2 / Batch M — internal helper that runs the
     *        edge + face scatter loop into the supplied COO buffers,
     *        and returns the total number of constraint rows.
     *
     * @details Both `Build()` (full replicated matrix) and
     * `BuildHypreParMatrix()` (per-rank local slice) call this helper
     * to do the actual row emission.
     *
     * Phase 5.9 — accepts the `(active_pair_labels, comp_mask)`
     * filter. Face-pair iteration is gated on whether the pair's
     * axis ∈ active_axes; edge-pair iteration is gated on whether
     * BOTH perpendicular axes ∈ active_axes; the comp-mask is
     * threaded into the scatter helpers.
     *
     * @return Total number of constraint rows emitted.
     */
    int EmitConstraintTriples(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>& comp_mask,
        std::vector<int>& rows,
        std::vector<int>& cols,
        std::vector<double>& vals) const;

    /// Parent FES whose global true DOFs define matrix columns.
    const mfem::ParFiniteElementSpace& ParentFes() const
    {
        return *m_parent_fes_raw;
    }

    /// Number of parent-FES global true DOFs used as C's column count.
    int ParentGlobalTrueVSize() const;

    /// Translate classifier-side true DOF to parent-FES true DOF.
    /// Negative sentinels are preserved.
    int ParentGtdofFromClassifierGtdof(int classifier_gtdof) const;

    /// Return parent-FES component true DOFs corresponding to a
    /// classifier-side x-component true DOF key.
    std::array<int, 3> ParentGtdofXyzFromClassifierX(
        int classifier_g_x) const;

    /// Return the owner rank of a classifier-side x-component true DOF
    /// after parent-FES translation.
    int ParentOwnerRankFromClassifierX(int classifier_g_x) const;

    //==========================================================================
    // Member state
    //==========================================================================

    const BoundaryClassifier3D& m_classifier;

    // Phase 6 ownership hooks. Legacy construction leaves these empty
    // and uses classifier.Fes() as the parent FES. Projector
    // construction fills them so the builder can translate submesh-FES
    // columns into the parent-volume FES column space.
    std::shared_ptr<const BoundaryClassifier3D> m_classifier_owner;
    std::shared_ptr<const SurfaceProjector> m_projector;
    std::shared_ptr<const mfem::ParFiniteElementSpace> m_parent_fes_owner;
    const mfem::ParFiniteElementSpace* m_parent_fes_raw = nullptr;

    // Stateless assemblers — cheap to default-construct, kept as
    // members so the builder owns its own working set.
    //
    // Phase 4.2 / Batch I+J: these assemblers no longer run any
    // `AssemblePairConforming` here in production builds (the
    // classifier does that tile-locally and AllGather's the resulting
    // blocks). They are kept on the off-chance that a future debug
    // path needs to re-run an assembler against a single block.
    MortarAssembler2D       m_edge_assembler;
    QuadFaceMortarAssembler m_quad_face_assembler;
    TriFaceMortarAssembler  m_tri_face_assembler;

    // Cached classifier-side gtdof lookup: primary x-component gtdof
    // -> (gx, gy, gz). In projector mode these are submesh-FES gtdofs
    // and must be translated before they are emitted as matrix columns.
    std::map<int, std::array<int, 3>> m_gtdof_lookup;
};

}  // namespace mortar_pbc
