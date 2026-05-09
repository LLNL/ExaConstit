// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Phase 4.1.A — implementation of ConstraintBuilder3D, ported from
// `mortar_pbc/constraint_builder_3d.py`. See header for design doc.

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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mortar_pbc {

namespace {

//==============================================================================
// Period-vector helper
//==============================================================================
// (PeriodSigned helper removed in Phase 4.2 / Batch J — was only used
// by the now-decommissioned ScatterFacePair. The classifier's
// BuildLocalPairBlocks computes its own period_signed inline from
// bbox planes.)
//==============================================================================

}  // anonymous namespace

//==============================================================================
// Constructor
//==============================================================================

ConstraintBuilder3D::ConstraintBuilder3D(const BoundaryClassifier3D& classifier)
    : m_classifier(classifier)
    , m_edge_assembler()
    , m_quad_face_assembler()
    , m_tri_face_assembler()
    , m_gtdof_lookup(classifier.GtdofXyzLookup())
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::ctor");
}

//==============================================================================
// NumConstraints — pre-compute the row count without running assembly
//==============================================================================

int ConstraintBuilder3D::NumConstraints() const
{
    int n = 0;

    // Edge pairs: each kept nonmortar edge contributes vdim *
    // n_interior_nodes constraint rows. EdgeInfo3D::n_nodes is the
    // size of any of the per-component gtdof arrays (they all match;
    // see types_3d.hpp).
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& nonmortar_label = std::get<2>(tup);
        const EdgeInfo3D& nonmortar_edge =
            m_classifier.Edges().at(nonmortar_label);
        n += kVDim * nonmortar_edge.NumNodes();
    }

    // Face pairs: kept-nonmortar count is the size of interior_gtdofs_x
    // (face interior dofs, with corner/edge sentinels already excluded
    // by the classifier).
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& nonmortar_label = std::get<2>(tup);
        const FaceInfo3D& nonmortar_face =
            m_classifier.Faces().at(nonmortar_label);
        n += kVDim * nonmortar_face.interior_gtdofs_x.Size();
    }

    return n;
}

//==============================================================================
// NumLocalRows — Phase 4.2 / Batch N — number of constraint rows
// owned by THIS rank under the FES-aligned row partition. Counts
// edge rows whose x-component nonmortar gtdof is FES-owned by this
// rank, plus face rows already routed to this rank.
//==============================================================================
int ConstraintBuilder3D::NumLocalRows() const
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
    return EmitConstraintTriples(rows, cols, vals);
}

//==============================================================================
// Build — produce the replicated CSR matrix
//==============================================================================

std::unique_ptr<mfem::SparseMatrix> ConstraintBuilder3D::Build() const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::build");

    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;

    const int n_rows = EmitConstraintTriples(rows, cols, vals);
    const int n_cols = m_classifier.NGlobalTdofs();

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
// EmitConstraintTriples — shared helper between Build() and
// BuildHypreParMatrix(). Runs the edge + face scatter loop and
// populates the supplied COO buffers in global-row indexing.
//==============================================================================

int ConstraintBuilder3D::EmitConstraintTriples(
    std::vector<int>& rows,
    std::vector<int>& cols,
    std::vector<double>& vals) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::emit_triples");

    // Reserve a generous-but-not-wasteful upper bound: each nonmortar
    // node contributes one diagonal D entry plus on the order of
    // (n_mortar_nodes_in_overlap) off-diagonal -A_m entries per
    // component. A factor of 8 per nonmortar TDOF is plenty for the
    // axis-aligned conforming case.
    const int n_constraints_est = NumConstraints();
    rows.reserve(static_cast<std::size_t>(8) * n_constraints_est);
    cols.reserve(static_cast<std::size_t>(8) * n_constraints_est);
    vals.reserve(static_cast<std::size_t>(8) * n_constraints_est);

    int row_offset = 0;

    //--- Edge mortar blocks (9 pairs) ---
    for (const auto& tup : m_classifier.EdgePairs())
    {
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
                                      rows, cols, vals, row_offset);
    }

    //--- Face mortar blocks (3 pairs) ---
    //
    // Phase 4.2 / Batch I+J: blocks are pre-matched and pre-assembled
    // by the classifier (tile-locally), then AllGather'd to every
    // rank. Read them via PairBlocks() and scatter.
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis            = std::get<0>(tup);
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
            row_offset = ScatterFaceBlock(quad_block->block, rows, cols, vals,
                                          row_offset);
        }
        if (tri_block != nullptr)
        {
            row_offset = ScatterFaceBlock(tri_block->block, rows, cols, vals,
                                          row_offset);
        }
    }

    return row_offset;
}

//==============================================================================
// AxisStrToInt — local helper. EdgePairs / FacePairs return axis as a
// single-character string; collapse to {0, 1, 2}.
//==============================================================================
namespace {
int AxisStrToInt(const std::string& s)
{
    if (s == "x") { return 0; }
    if (s == "y") { return 1; }
    if (s == "z") { return 2; }
    MFEM_ABORT("ConstraintBuilder3D::AxisStrToInt: unknown axis '"
               << s << "' (expected 'x', 'y', or 'z').");
    return -1;  // unreachable
}
}  // anonymous namespace

//==============================================================================
// EmitRowFactors — per-row reference-geometry metadata. Mirrors the
// row-enumeration pattern of EmitConstraintTriples exactly so that
// emit position k corresponds to constraint row k. Edges go through
// the row-owner filter (FES ownership of the x-component nonmortar
// gtdof); face pair blocks are pre-routed by the classifier so they
// require no per-row filter.
//==============================================================================
void ConstraintBuilder3D::EmitRowFactors(
    mfem::Array<int>& axis_index,
    mfem::Array<int>& component_index,
    mfem::Vector& ell_hat) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::constraint_builder::emit_row_factors");

    // Build into std::vector first (cheap, growable); copy out at the
    // end to mfem::Array / mfem::Vector. The upper-bound row count
    // is NumConstraints(); local count is at most that.
    const int n_constraints_est = NumConstraints();
    std::vector<int>    axis_buf;
    std::vector<int>    comp_buf;
    std::vector<double> ell_buf;
    axis_buf.reserve(static_cast<std::size_t>(n_constraints_est));
    comp_buf.reserve(static_cast<std::size_t>(n_constraints_est));
    ell_buf.reserve(static_cast<std::size_t>(n_constraints_est));

    const int my_rank = m_classifier.Rank();

    //--- Edge mortar blocks ---
    //
    // We re-run the edge assembler here. The cost is 9 small dense
    // assemblies per call — negligible at construction time, and
    // matching EmitConstraintTriples' pattern keeps the row order
    // identical. (Future refactor: cache the assembled blocks once
    // and reuse across both methods. Not required here.)
    for (const auto& tup : m_classifier.EdgePairs())
    {
        const std::string& axis_str       = std::get<0>(tup);
        const std::string& mortar_label   = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const int axis_idx = AxisStrToInt(axis_str);
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
                              ? m_classifier.GtdofOwnerRank(g_n_x) : -1;
            if (owner != my_rank) { continue; }

            const double D_kk = block.D_nm(k);
            for (int c = 0; c < kVDim; ++c)
            {
                axis_buf.push_back(axis_idx);
                comp_buf.push_back(c);
                ell_buf.push_back(D_kk);
            }
        }
    }

    //--- Face mortar blocks (pre-routed by the classifier) ---
    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis_str       = std::get<0>(tup);
        const std::string& mortar_label   = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const int axis_idx = AxisStrToInt(axis_str);

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
                for (int c = 0; c < kVDim; ++c)
                {
                    axis_buf.push_back(axis_idx);
                    comp_buf.push_back(c);
                    ell_buf.push_back(D_kk);
                }
            }
        };

        if (quad_block != nullptr) { emit_face_block(*quad_block); }
        if (tri_block  != nullptr) { emit_face_block(*tri_block);  }
    }

    // Copy out to mfem::Array<int> / mfem::Vector outputs.
    const int n_local = static_cast<int>(axis_buf.size());
    axis_index.SetSize(n_local);
    component_index.SetSize(n_local);
    ell_hat.SetSize(n_local);
    for (int i = 0; i < n_local; ++i)
    {
        axis_index[i]      = axis_buf[i];
        component_index[i] = comp_buf[i];
        ell_hat[i]         = ell_buf[i];
    }
}

//==============================================================================
// BuildHypreParMatrix — distributed form, row-partitioned via Allgather
//==============================================================================

mfem::HypreParMatrix* ConstraintBuilder3D::BuildHypreParMatrix() const
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

    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> vals;
    const int n_lam_local   = EmitConstraintTriples(rows, cols, vals);
    const int n_global_cols = m_classifier.NGlobalTdofs();

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
    HYPRE_BigInt* fes_tdof_offsets = m_classifier.Fes().GetTrueDofOffsets();
    std::vector<HYPRE_BigInt> col_starts(2);
    col_starts[0] = fes_tdof_offsets[0];
    col_starts[1] = fes_tdof_offsets[1];

    // Sanity-check: this rank's local FES TDOF count must equal
    // (col_starts[1] - col_starts[0]).
    {
        const int n_loc_fes = m_classifier.Fes().GetTrueVSize();
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
// ScatterEdgeBlock — append rows for one (block, nonmortar, mortar) triplet
//==============================================================================

int ConstraintBuilder3D::ScatterEdgeBlock(
    const MortarBlock2D& block,
    const EdgeInfo3D& nonmortar_edge,
    const EdgeInfo3D& mortar_edge,
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
    const int my_rank = m_classifier.Rank();

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
            ? m_classifier.GtdofOwnerRank(nonmortar_g_xyz[0])
            : -1;
        if (owner != my_rank) { continue; }

        if (D_kk == 0.0)
        {
            // Degenerate row (could happen if a nonmortar node is
            // entirely covered by a corner-modified element). Skip,
            // but still consume the kVDim row indices to keep the
            // vdim-aligned layout deterministic.
            row_offset += kVDim;
            continue;
        }

        // Diagonal D entry per spatial component.
        for (int c = 0; c < kVDim; ++c)
        {
            const int gd = nonmortar_g_xyz[c];
            if (gd < 0) { continue; }
            rows.push_back(row_offset + c);
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
                const int gd = mortar_g_xyz[c];
                if (gd < 0) { continue; }
                rows.push_back(row_offset + c);
                cols.push_back(gd);
                vals.push_back(-A_kl);
            }
        }

        row_offset += kVDim;
    }

    return row_offset;
}

//==============================================================================
// ScatterFaceBlock — append rows for one face mortar block
//==============================================================================

int ConstraintBuilder3D::ScatterFaceBlock(
    const FaceMortarPairBlock& block,
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

    for (int k = 0; k < n_nonmortar_kept; ++k)
    {
        const double D_kk = block.D(k);
        const int nonmortar_gx = block.nonmortar_gtdofs[k];

        auto it = m_gtdof_lookup.find(nonmortar_gx);
        MFEM_VERIFY(it != m_gtdof_lookup.end(),
                    "ConstraintBuilder3D: nonmortar gtdof "
                    << nonmortar_gx << " (face block) has no entry in "
                    "classifier's gtdof_xyz_lookup. The face assembler "
                    "emitted a nonmortar gtdof not seen by the boundary "
                    "classifier.");
        const std::array<int, 3>& nonmortar_g_xyz = it->second;

        if (D_kk == 0.0)
        {
            row_offset += kVDim;
            continue;
        }

        // Diagonal D entries.
        for (int c = 0; c < kVDim; ++c)
        {
            const int gd = nonmortar_g_xyz[c];
            if (gd < 0) { continue; }
            rows.push_back(row_offset + c);
            cols.push_back(gd);
            vals.push_back(D_kk);
        }

        // Off-diagonal -A_m entries — CSR row walk.
        for (int idx = A_I[k]; idx < A_I[k + 1]; ++idx)
        {
            const int l = A_J[idx];
            const double A_kl = A_V[idx];
            if (A_kl == 0.0) { continue; }
            const int mortar_gx = block.mortar_gtdofs[l];
            auto it2 = m_gtdof_lookup.find(mortar_gx);
            MFEM_VERIFY(it2 != m_gtdof_lookup.end(),
                        "ConstraintBuilder3D: mortar gtdof " << mortar_gx
                        << " has no entry in classifier's "
                        "gtdof_xyz_lookup.");
            const std::array<int, 3>& mortar_g_xyz = it2->second;
            for (int c = 0; c < kVDim; ++c)
            {
                const int gd = mortar_g_xyz[c];
                if (gd < 0) { continue; }
                rows.push_back(row_offset + c);
                cols.push_back(gd);
                vals.push_back(-A_kl);
            }
        }

        row_offset += kVDim;
    }

    return row_offset;
}

}  // namespace mortar_pbc
