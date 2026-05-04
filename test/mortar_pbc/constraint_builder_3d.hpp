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
// References
// ----------
//   * MORTAR_PBC_ARCHITECTURE.md §11.8 (this layer).
//   * MORTAR_PBC_ARCHITECTURE.md §11.5 (3D edge mortar).
//   * MORTAR_PBC_ARCHITECTURE.md §11.6 (face-mortar geometric matching).

#pragma once

#include "boundary_classifier_3d.hpp"
#include "face_mortar_assembler_3d.hpp"
#include "mortar_assembler_2d.hpp"
#include "types_3d.hpp"

#include "mfem.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mortar_pbc {

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
 * @par Lifetime
 * The builder holds a non-owning reference to the classifier. The
 * caller must ensure the classifier outlives the builder.
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

    // Non-copyable / non-movable: holds a reference and a small set of
    // assemblers.
    ConstraintBuilder3D(const ConstraintBuilder3D&) = delete;
    ConstraintBuilder3D& operator=(const ConstraintBuilder3D&) = delete;

    /**
     * @brief Build the replicated global constraint matrix.
     *
     * @return A `unique_ptr<mfem::SparseMatrix>` of shape
     *         `(NumConstraints(), classifier.NGlobalTdofs())`. Entries
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
     * Internally:
     *   1. Calls `EmitConstraintTriples` which (after Batch N) emits
     *      only this rank's rows.
     *   2. `MPI_Allgather`s the per-rank row count to compute Hypre
     *      row_starts.
     *   3. Constructs a local-sized `SparseMatrix` and wraps it in
     *      a `HypreParMatrix` using the FES TDOF column partition
     *      (§P4.8.9 — must match K's column partition for valid
     *      C·u parallel matvec).
     *
     * @return A heap-allocated `HypreParMatrix*`. Caller owns and must
     *         `delete` it.
     *
     * @par MPI scope
     * Collective on `classifier.Comm()`. One `MPI_Allgather` (int).
     */
    mfem::HypreParMatrix* BuildHypreParMatrix() const;

    /**
     * @brief Phase 4.2 / Batch N — number of constraint rows owned
     *        by this rank under the FES-aligned row partition.
     *
     * @details Computed by running `EmitConstraintTriples` once and
     * counting the emitted rows. Cached on first call; subsequent
     * calls are O(1).
     *
     * Useful for sizing the Lagrange-multiplier `Vector` (the dual
     * variable in the saddle-point system has one entry per local
     * constraint row).
     */
    int NumLocalRows() const;

    /**
     * @brief Number of constraint rows the build will emit.
     *
     * @details Sum over edge pairs of `kVDim × n_interior_nonmortar_nodes`,
     * plus sum over face pairs of `kVDim × n_kept_nonmortar_face_dofs`
     * (using the classifier's pre-computed `interior_gtdofs_x` size).
     */
    int NumConstraints() const;

private:
    /**
     * @brief Append rows for one edge mortar block to the COO buffers.
     *
     * @details `nonmortar_edge.gtdofs_*` index into the per-component
     * arrays directly; the vdim expansion is just the per-c loop.
     *
     * @return The new (post-append) row offset.
     */
    int ScatterEdgeBlock(const MortarBlock2D& block,
                         const EdgeInfo3D& nonmortar_edge,
                         const EdgeInfo3D& mortar_edge,
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
     * @return The new (post-append) row offset.
     */
    int ScatterFaceBlock(const FaceMortarPairBlock& block,
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
     * to do the actual row emission. `Build()` constructs a
     * `SparseMatrix` from all triples; `BuildHypreParMatrix()`
     * filters by this rank's row range and constructs only the local
     * slice. Sharing the helper guarantees both paths produce
     * mathematically identical row content (modulo floating-point
     * order in `SparseMatrix::Finalize`).
     *
     * @param[out] rows COO row indices (0-indexed in global row space).
     * @param[out] cols COO column indices (0-indexed in global TDOF
     *                  space; matches FES TDOF numbering).
     * @param[out] vals COO values.
     * @return Total number of constraint rows emitted.
     */
    int EmitConstraintTriples(std::vector<int>& rows,
                              std::vector<int>& cols,
                              std::vector<double>& vals) const;

    //==========================================================================
    // Member state
    //==========================================================================

    const BoundaryClassifier3D& m_classifier;
    // Phase 4.2 / Batch K: m_pair_match_tol_rel was removed from this
    // class. Matching happens inside the classifier now; the
    // tolerance is configured on the classifier's constructor.

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

    // Cached gtdof lookup: primary x-component gtdof -> (gx, gy, gz).
    std::map<int, std::array<int, 3>> m_gtdof_lookup;
};

}  // namespace mortar_pbc
