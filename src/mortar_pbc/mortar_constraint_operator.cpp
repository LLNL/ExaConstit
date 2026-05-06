// Phase 4.3 / Batch O — MortarConstraintOperator skeleton.
//
// The constructor builds the off-rank import / export topology;
// Mult and MultTranspose are stubbed for Batch P to implement. The
// stubs MFEM_ABORT with a clear message so callers wiring the type
// in early get an immediate, traceable failure rather than silent
// zero-output.
//
// See mortar_constraint_operator.hpp for design rationale.

#include "mortar_constraint_operator.hpp"

#include "mortar_assembler_2d.hpp"
#include "utilities/mechanics_log.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace mortar_pbc {

//==============================================================================
// Constructor — builds local edge-mortar blocks + import/export topology.
//
// Phase 4.3 / Batch O scaffolds these; Batch P fleshes them out and
// adds testing. The current implementation:
//   1. Assembles 9 edge-mortar blocks locally (cheap; matches
//      ConstraintBuilder3D::EmitConstraintTriples's per-rank
//      redundant assembly).
//   2. Caches the gtdof_xyz_lookup from the classifier.
//   3. Computes the off-rank gtdof set: all mortar gtdofs across
//      this rank's pair blocks (face mortars from PairBlocks() +
//      edge mortars whose row-owner is this rank) that are NOT
//      FES-owned locally.
//   4. Builds the Alltoallv import topology (counts, displs, slot
//      maps).
//   5. Builds the export topology by inverting the import topology
//      via Alltoall on counts.
//==============================================================================
MortarConstraintOperator::MortarConstraintOperator(
    const BoundaryClassifier3D& classifier)
    : mfem::Operator(/* height */ 0, /* width */ 0)
    , m_classifier(classifier)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::mortar_constraint_operator::ctor");

    m_gtdof_lookup = classifier.GtdofXyzLookup();

    // -----------------------------------------------------------------
    // Step 1 — assemble local edge-mortar blocks. We need the same 9
    // blocks ConstraintBuilder3D produces in EmitConstraintTriples.
    // Reusing MortarAssembler2D directly (it's stateless and cheap to
    // default-construct).
    // -----------------------------------------------------------------
    MortarAssembler2D edge_assembler;
    m_local_edge_pairs.reserve(classifier.EdgePairs().size());
    for (const auto& tup : classifier.EdgePairs())
    {
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);
        const EdgeInfo3D& mortar_edge =
            classifier.Edges().at(mortar_label);
        const EdgeInfo3D& nonmortar_edge =
            classifier.Edges().at(nonmortar_label);

        LocalEdgePair lep;
        lep.block = edge_assembler.AssemblePair(nonmortar_edge, mortar_edge);
        lep.nonmortar_edge = nonmortar_edge;
        lep.mortar_edge    = mortar_edge;
        m_local_edge_pairs.push_back(std::move(lep));
    }

    // -----------------------------------------------------------------
    // Step 2 — compute Operator height/width.
    //
    // Width  = this rank's local FES TDOF count (matches the column
    //          partition of HypreParMatrix path).
    // Height = number of constraint rows owned by this rank under
    //          the FES-aligned partition. Uses a temporary
    //          ConstraintBuilder3D to delegate to
    //          NumLocalRows() — keeps the row-counting logic in one
    //          place.
    // -----------------------------------------------------------------
    {
        ConstraintBuilder3D temp_builder(classifier);
        const int n_lam_local = temp_builder.NumLocalRows();
        const int n_loc_fes   = classifier.Fes().GetTrueVSize();
        // Operator base class doesn't expose protected setters in
        // older MFEM; use the (h, w) ctor pattern via a placement
        // assignment. Cleanest portable form:
        height = n_lam_local;
        width  = n_loc_fes;
    }

    // -----------------------------------------------------------------
    // Step 3 — build the off-rank import / export topology.
    //
    // The "import" side: this rank needs `x[g_m]` for every mortar
    // gtdof `g_m` referenced by ANY block on this rank that is NOT
    // FES-owned locally. The set is enumerated, sorted by owner rank,
    // and Alltoallv recv counts/displs are precomputed. The mortar
    // gtdofs in face blocks are x-component only (per Batch L
    // convention); we route by x-gtdof and assume y/z components are
    // co-located (matches Batch N's row-owner convention — y/z FES
    // ownership SHOULD match x in MFEM's standard byNODES vector
    // ordering).
    //
    // The "export" side (mirror of import, used by MultTranspose):
    // every other rank tells us "I need these LOCAL gtdofs from you"
    // via an Alltoall on counts followed by an Alltoallv on the
    // gtdof-index lists. We store those as `m_export_local_gtdofs`
    // in destination-rank-sorted order matching the export send
    // counts/displs.
    // -----------------------------------------------------------------
    MPI_Comm comm = classifier.Comm();
    const int my_rank = classifier.Rank();
    const int n_ranks = classifier.NRanks();

    // FES TDOF range owned by this rank.
    const HYPRE_BigInt my_first_tdof =
        classifier.Fes().GetTrueDofOffsets()[0];
    const HYPRE_BigInt my_end_tdof =
        classifier.Fes().GetTrueDofOffsets()[1];

    // ----------- collect off-rank mortar gtdofs (x-component) -----------
    //
    // Walk every block and every mortar column; check FES ownership;
    // collect off-rank gtdofs in a set (dedup automatic).
    std::set<int> off_rank_gtdofs_set;

    auto consider_mortar_gtdof = [&](int g_x)
    {
        // g_x is the x-component gtdof of the mortar node.
        if (g_x < 0) { return; }
        if (g_x >= static_cast<int>(my_first_tdof)
            && g_x < static_cast<int>(my_end_tdof))
        {
            return;  // FES-owned locally; no exchange needed
        }
        off_rank_gtdofs_set.insert(g_x);
    };

    // Face mortar blocks (already row-routed to this rank in Batch N).
    for (const auto& lpb : classifier.PairBlocks())
    {
        const int n_m = lpb.block.NumMortarKept();
        for (int j = 0; j < n_m; ++j)
        {
            consider_mortar_gtdof(lpb.block.mortar_gtdofs[j]);
        }
    }

    // Edge mortar blocks (assembled redundantly per rank — only
    // consider the ones where this rank owns the row).
    for (const auto& lep : m_local_edge_pairs)
    {
        const int n_n = lep.nonmortar_edge.NumNodes();
        const int n_m = lep.mortar_edge.NumNodes();
        // Filter: only need mortar values for rows we own (those whose
        // x-component nonmortar gtdof is FES-owned locally).
        bool any_row_owned = false;
        for (int k = 0; k < n_n; ++k)
        {
            const int g_n_x = lep.nonmortar_edge.gtdofs_x[k];
            if (g_n_x < 0) { continue; }
            if (g_n_x >= static_cast<int>(my_first_tdof)
                && g_n_x < static_cast<int>(my_end_tdof))
            {
                any_row_owned = true;
                break;
            }
        }
        if (!any_row_owned) { continue; }
        // For each owned row, its mortar columns might be off-rank.
        for (int l = 0; l < n_m; ++l)
        {
            consider_mortar_gtdof(lep.mortar_edge.gtdofs_x[l]);
        }
    }

    // ----------- partition by FES owner; build import topology -----------
    //
    // Sort the off-rank set by owner rank, store the resulting
    // sequence in m_import_off_rank_gtdofs. Build per-source-rank
    // recv counts and a (gtdof -> slot) lookup.
    {
        // Bucket gtdofs by owner.
        std::vector<std::vector<int>> by_owner(n_ranks);
        for (int g : off_rank_gtdofs_set)
        {
            const int owner = classifier.GtdofOwnerRank(g);
            MFEM_ASSERT(owner != my_rank,
                        "MortarConstraintOperator: off-rank gtdof "
                        << g << " has GtdofOwnerRank == my_rank "
                        << my_rank << " — set classification bug");
            by_owner[owner].push_back(g);
        }

        m_import_off_rank_gtdofs.clear();
        m_import_recv_counts.assign(n_ranks, 0);
        m_import_recv_displs.assign(n_ranks, 0);
        int cumulative = 0;
        for (int r = 0; r < n_ranks; ++r)
        {
            // Stable order for reproducibility.
            std::sort(by_owner[r].begin(), by_owner[r].end());
            m_import_recv_displs[r] = cumulative;
            m_import_recv_counts[r] = static_cast<int>(by_owner[r].size());
            for (int g : by_owner[r])
            {
                const int slot = static_cast<int>(
                    m_import_off_rank_gtdofs.size());
                m_import_off_rank_gtdofs.push_back(g);
                m_import_gtdof_to_slot[g] = slot;
            }
            cumulative += m_import_recv_counts[r];
        }
    }

    // ----------- mirror to export topology via Alltoall + Alltoallv -----
    //
    // (a) Alltoall the per-source recv counts so each rank learns
    //     how many of ITS gtdofs each peer wants.
    // (b) Alltoallv the gtdof index lists themselves (each rank sends
    //     m_import_off_rank_gtdofs sliced by m_import_recv_displs to
    //     each owner; each owner receives the gtdofs it must export).
    // (c) Store results in m_export_local_gtdofs (destination-rank-
    //     sorted order matching m_import_send_counts/displs).
    {
        m_import_send_counts.assign(n_ranks, 0);
        MPI_Alltoall(m_import_recv_counts.data(), 1, MPI_INT,
                     m_import_send_counts.data(), 1, MPI_INT,
                     comm);

        m_import_send_displs.assign(n_ranks, 0);
        int total_send = 0;
        for (int r = 0; r < n_ranks; ++r)
        {
            m_import_send_displs[r] = total_send;
            total_send += m_import_send_counts[r];
        }

        m_export_local_gtdofs.assign(total_send, 0);

        // Send our import requests; receive the requests destined for us.
        // Note: from THIS rank's perspective, m_import_off_rank_gtdofs
        // is the SEND buffer for the gtdof exchange (we're telling
        // each owner "send me these"), and m_export_local_gtdofs is
        // what we RECEIVE (other ranks telling us "send these to me").
        MPI_Alltoallv(m_import_off_rank_gtdofs.data(),
                      m_import_recv_counts.data(),
                      m_import_recv_displs.data(),
                      MPI_INT,
                      m_export_local_gtdofs.data(),
                      m_import_send_counts.data(),
                      m_import_send_displs.data(),
                      MPI_INT,
                      comm);

        // Sanity: every received gtdof should be FES-owned locally.
        for (int g : m_export_local_gtdofs)
        {
            MFEM_VERIFY(g >= static_cast<int>(my_first_tdof)
                        && g < static_cast<int>(my_end_tdof),
                        "MortarConstraintOperator: peer rank requested "
                        "gtdof " << g << " from this rank, but it is "
                        "outside this rank's FES TDOF range ["
                        << my_first_tdof << ", " << my_end_tdof << "). "
                        "Topology mismatch — likely a GtdofOwnerRank "
                        "inconsistency.");
        }
    }

    // Phase 4.3.B / Batch X — pre-flatten per-pair-block data into
    // GPU-friendly arrays. After this call the matvec hot path is a
    // single mfem::forall over m_n_active_rows, with no std::map or
    // std::vector lookups in the kernel.
    BuildFlatRowArrays();
}

//==============================================================================
// BuildFlatRowArrays — Phase 4.3.B / Batch X
//
// Walks the SAME iteration order as Mult / MultTranspose (edges first
// with row-owner filter, then face mortars in FacePairs() order with
// quad-then-tri). Populates m_row_D, m_row_g_n_local, m_row_csr_off,
// m_csr_A, m_csr_g_m_local, m_csr_g_m_recv. After this point the
// per-pair lookup machinery (m_local_edge_pairs, classifier.PairBlocks(),
// m_gtdof_lookup, m_import_gtdof_to_slot) is unused at matvec time —
// it's all baked into the flat arrays.
//
// Encoding contract (must be respected by the kernel):
//   * Sentinel rows (D_kk == 0): emit a row entry with D = 0, an
//     empty CSR slice (csr_off[i+1] == csr_off[i]), and -1 for all
//     g_n_local components. This preserves row-count alignment with
//     the lambda vector layout.
//   * Sentinel components on a non-sentinel row: g_n_local[c] = -1
//     for that component; the kernel writes 0 into y for that
//     component (matching the existing CPU code which simply skips
//     the component, leaving y[ro+c] at its initialized 0.0).
//   * Mortar component encoding (m_csr_g_m_local / m_csr_g_m_recv):
//     - both -1: sentinel; kernel skips.
//     - g_m_local[c] >= 0, g_m_recv[c] == -1: local FES TDOF.
//     - g_m_local[c] == -1, g_m_recv[c] >= 0: imported off-rank.
//==============================================================================
void MortarConstraintOperator::BuildFlatRowArrays()
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::mortar_constraint_operator::build_flat_row_arrays");

    const int my_rank = m_classifier.Rank();
    const HYPRE_BigInt my_first_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[0];
    const HYPRE_BigInt my_end_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[1];

    // ------------------------------------------------------------------
    // Pass 1 — count active rows and total CSR entries.
    //
    // We need the totals to size the flat arrays before populating.
    // The walk must be identical to pass 2 (and to Mult / MultTranspose)
    // so that sizes match.
    // ------------------------------------------------------------------
    int n_active = 0;
    int n_csr    = 0;

    // Edge pairs: row-owner filter; if D_kk == 0, row is still emitted
    // (counts towards n_active) with empty CSR slice. The CSR slice
    // counts ALL non-zero A_kl entries; A_m for edges is dense, so
    // n_m entries per row before pruning. We prune zeros at population
    // time (the sentinel-skip logic mirrors the existing Mult body).
    for (const auto& lep : m_local_edge_pairs)
    {
        const int n_n = lep.nonmortar_edge.NumNodes();
        const int n_m = lep.mortar_edge.NumNodes();
        for (int k = 0; k < n_n; ++k)
        {
            const int g_n_x = lep.nonmortar_edge.gtdofs_x[k];
            const int owner = (g_n_x >= 0)
                              ? m_classifier.GtdofOwnerRank(g_n_x) : -1;
            if (owner != my_rank) { continue; }
            ++n_active;
            const double D_kk = lep.block.D_nm(k);
            if (D_kk == 0.0) { continue; }
            // count non-zero A_kl entries
            for (int l = 0; l < n_m; ++l)
            {
                if (lep.block.A_m(k, l) != 0.0) { ++n_csr; }
            }
        }
    }

    // Face pairs (FacePairs() order, quad-then-tri).
    auto count_face_block = [&](const FaceMortarPairBlock& block)
    {
        const int n_n = block.NumNonmortarKept();
        const int* A_I    = block.A_m.GetI();
        const double* A_V = block.A_m.GetData();
        for (int k = 0; k < n_n; ++k)
        {
            ++n_active;
            if (block.D(k) == 0.0) { continue; }
            for (int idx = A_I[k]; idx < A_I[k + 1]; ++idx)
            {
                if (A_V[idx] != 0.0) { ++n_csr; }
            }
        }
    };

    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis            = std::get<0>(tup);
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const FaceMortarPairBlock* quad_block = nullptr;
        const FaceMortarPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair != axis
                || lpb.mortar_label != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if (lpb.geometry_kind == "quad") { quad_block = &lpb.block; }
            else if (lpb.geometry_kind == "tri") { tri_block = &lpb.block; }
        }

        if (quad_block != nullptr) { count_face_block(*quad_block); }
        if (tri_block  != nullptr) { count_face_block(*tri_block);  }
    }

    m_n_active_rows = n_active;

    // ------------------------------------------------------------------
    // Pass 2 — allocate and populate.
    // ------------------------------------------------------------------
    m_row_lambda_off.SetSize(n_active);
    m_row_D.SetSize(n_active);
    m_row_g_n_local.SetSize(n_active * kVDim);
    m_row_csr_off.SetSize(n_active + 1);
    m_csr_A.SetSize(n_csr);
    m_csr_g_m_local.SetSize(n_csr * kVDim);
    m_csr_g_m_recv.SetSize(n_csr * kVDim);

    // Init host-side via raw GetData; this is setup time, not a hot
    // path, so just write through host pointers and let the memory
    // manager's first Read on device migrate as needed.
    for (int i = 0; i < n_active; ++i)              { m_row_lambda_off[i] = i * kVDim; }
    for (int i = 0; i < n_active; ++i)              { m_row_D[i] = 0.0; }
    for (int i = 0; i < n_active * kVDim; ++i)      { m_row_g_n_local[i] = -1; }
    for (int i = 0; i <= n_active; ++i)             { m_row_csr_off[i] = 0; }
    for (int i = 0; i < n_csr; ++i)                 { m_csr_A[i] = 0.0; }
    for (int i = 0; i < n_csr * kVDim; ++i)         { m_csr_g_m_local[i] = -1; }
    for (int i = 0; i < n_csr * kVDim; ++i)         { m_csr_g_m_recv[i]  = -1; }

    // Helper — encode one mortar component lookup into the two
    // tagged-index arrays. Returns silently on sentinel.
    auto encode_mortar = [&](int g_m_x, int component, int csr_entry)
    {
        const auto it = m_gtdof_lookup.find(g_m_x);
        MFEM_VERIFY(it != m_gtdof_lookup.end(),
                    "BuildFlatRowArrays: mortar gtdof " << g_m_x
                    << " not in m_gtdof_lookup");
        const int gd = it->second[component];
        if (gd < 0)
        {
            // sentinel — both arrays already -1; nothing to do
            return;
        }
        const int slot_idx = csr_entry * kVDim + component;
        if (gd >= static_cast<int>(my_first_tdof)
            && gd <  static_cast<int>(my_end_tdof))
        {
            m_csr_g_m_local[slot_idx] = gd - static_cast<int>(my_first_tdof);
        }
        else
        {
            const auto slot_it = m_import_gtdof_to_slot.find(g_m_x);
            MFEM_VERIFY(slot_it != m_import_gtdof_to_slot.end(),
                        "BuildFlatRowArrays: off-rank mortar gtdof "
                        << g_m_x
                        << " missing from import topology");
            m_csr_g_m_recv[slot_idx] = slot_it->second * kVDim + component;
        }
    };

    int row_i = 0;
    int csr_i = 0;

    // Edge pairs.
    for (const auto& lep : m_local_edge_pairs)
    {
        const int n_n = lep.nonmortar_edge.NumNodes();
        const int n_m = lep.mortar_edge.NumNodes();

        for (int k = 0; k < n_n; ++k)
        {
            const int g_n_x = lep.nonmortar_edge.gtdofs_x[k];
            const int owner = (g_n_x >= 0)
                              ? m_classifier.GtdofOwnerRank(g_n_x) : -1;
            if (owner != my_rank) { continue; }

            const double D_kk = lep.block.D_nm(k);
            m_row_D[row_i] = D_kk;
            m_row_csr_off[row_i] = csr_i;

            // Per-component nonmortar local index (always FES-local
            // for owned rows under Batch N; or -1 sentinel).
            int g_n_xyz[kVDim];
            g_n_xyz[0] = lep.nonmortar_edge.gtdofs_x[k];
            g_n_xyz[1] = lep.nonmortar_edge.gtdofs_y[k];
            g_n_xyz[2] = lep.nonmortar_edge.gtdofs_z[k];
            for (int c = 0; c < kVDim; ++c)
            {
                const int gd = g_n_xyz[c];
                if (gd < 0) { continue; }   // leave -1
                MFEM_ASSERT(gd >= static_cast<int>(my_first_tdof)
                            && gd <  static_cast<int>(my_end_tdof),
                            "BuildFlatRowArrays: edge nonmortar gtdof "
                            << gd << " not FES-local despite row-owner "
                            "filter");
                m_row_g_n_local[row_i * kVDim + c]
                    = gd - static_cast<int>(my_first_tdof);
            }

            if (D_kk != 0.0)
            {
                // CSR entries (one per non-zero A_kl in this dense row).
                for (int l = 0; l < n_m; ++l)
                {
                    const double A_kl = lep.block.A_m(k, l);
                    if (A_kl == 0.0) { continue; }
                    m_csr_A[csr_i] = A_kl;
                    const int g_m_x = lep.mortar_edge.gtdofs_x[l];
                    // Per-component encoding. The edge struct exposes
                    // per-component gtdofs directly; we re-route through
                    // m_gtdof_lookup via the x-component key, which gives
                    // the same answer (the lookup was built from the
                    // edge / face metadata in the first place).
                    for (int c = 0; c < kVDim; ++c)
                    {
                        encode_mortar(g_m_x, c, csr_i);
                    }
                    ++csr_i;
                }
            }
            ++row_i;
        }
    }

    // Face pairs (FacePairs order, quad-then-tri).
    auto populate_face_block = [&](const FaceMortarPairBlock& block)
    {
        const int n_n = block.NumNonmortarKept();
        const int* A_I    = block.A_m.GetI();
        const int* A_J    = block.A_m.GetJ();
        const double* A_V = block.A_m.GetData();

        for (int k = 0; k < n_n; ++k)
        {
            const double D_kk = block.D(k);
            const int g_n_x = block.nonmortar_gtdofs[k];

            const auto it = m_gtdof_lookup.find(g_n_x);
            MFEM_VERIFY(it != m_gtdof_lookup.end(),
                        "BuildFlatRowArrays: face nonmortar gtdof "
                        << g_n_x << " not in m_gtdof_lookup");
            const std::array<int, 3>& g_n_xyz = it->second;

            m_row_D[row_i] = D_kk;
            m_row_csr_off[row_i] = csr_i;

            for (int c = 0; c < kVDim; ++c)
            {
                const int gd = g_n_xyz[c];
                if (gd < 0) { continue; }
                MFEM_ASSERT(gd >= static_cast<int>(my_first_tdof)
                            && gd <  static_cast<int>(my_end_tdof),
                            "BuildFlatRowArrays: face nonmortar gtdof "
                            "component " << gd
                            << " not FES-local despite Batch N routing");
                m_row_g_n_local[row_i * kVDim + c]
                    = gd - static_cast<int>(my_first_tdof);
            }

            if (D_kk != 0.0)
            {
                for (int idx = A_I[k]; idx < A_I[k + 1]; ++idx)
                {
                    const int l = A_J[idx];
                    const double A_kl = A_V[idx];
                    if (A_kl == 0.0) { continue; }
                    m_csr_A[csr_i] = A_kl;
                    const int g_m_x = block.mortar_gtdofs[l];
                    for (int c = 0; c < kVDim; ++c)
                    {
                        encode_mortar(g_m_x, c, csr_i);
                    }
                    ++csr_i;
                }
            }
            ++row_i;
        }
    };

    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis            = std::get<0>(tup);
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const FaceMortarPairBlock* quad_block = nullptr;
        const FaceMortarPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair != axis
                || lpb.mortar_label != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if (lpb.geometry_kind == "quad") { quad_block = &lpb.block; }
            else if (lpb.geometry_kind == "tri") { tri_block = &lpb.block; }
        }

        if (quad_block != nullptr) { populate_face_block(*quad_block); }
        if (tri_block  != nullptr) { populate_face_block(*tri_block);  }
    }

    // Final sentinel of the prefix-sum.
    m_row_csr_off[n_active] = csr_i;

    MFEM_ASSERT(row_i == n_active,
                "BuildFlatRowArrays: row count mismatch ("
                << row_i << " vs " << n_active << ")");
    MFEM_ASSERT(csr_i == n_csr,
                "BuildFlatRowArrays: CSR count mismatch ("
                << csr_i << " vs " << n_csr << ")");
}

//==============================================================================
// Mult — y = C * x
//
// Step 1 — import off-rank mortar u-values via Alltoallv.
// Step 2 — zero y.
// Step 3 — walk face mortar blocks; per-pair scatter into local row range.
// Step 4 — walk edge mortar blocks; per-pair scatter (with row-owner filter).
//
// The row ordering matches ConstraintBuilder3D::EmitConstraintTriples:
// edge mortars first (in EdgePairs() order), then face mortars (in
// FacePairs() order). Same iteration order as the HypreParMatrix path
// emits triples — and since at np=1 the routing is a self-loop, the
// HypreParMatrix path's row layout matches this one bit-for-bit.
//
// Wait — note the order: EmitConstraintTriples does edges THEN faces.
// We mirror that exactly (edges first, faces second). Otherwise the
// row layout would differ from BuildHypreParMatrix's and the A/B
// validation in Batch Q would diverge.
//==============================================================================
void MortarConstraintOperator::Mult(const mfem::Vector& x,
                                    mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::mortar_constraint_operator::mult");

    MFEM_VERIFY(x.Size() == Width(),
                "MortarConstraintOperator::Mult: input size "
                << x.Size() << " != Width() " << Width());
    MFEM_VERIFY(y.Size() == Height(),
                "MortarConstraintOperator::Mult: output size "
                << y.Size() << " != Height() " << Height());

    MPI_Comm comm = m_classifier.Comm();
    const int n_ranks = m_classifier.NRanks();
    const HYPRE_BigInt my_first_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[0];
    const HYPRE_BigInt my_end_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[1];

    // -----------------------------------------------------------------
    // Step 1 (HOST) — pack send buffer of off-rank u-values.
    //
    // MPI is host-only in standard implementations, so the send buffer
    // is constructed on the host. We use HostRead on x to get a stable
    // host pointer (the memory manager will migrate from device if
    // needed, and DEVICE_DEBUG will validate the access pattern).
    //
    // Layout: AOS, three doubles per slot (x, y, z components for one
    // mortar gtdof). One MPI_Alltoallv carries the whole exchange.
    // -----------------------------------------------------------------
    const int n_export = static_cast<int>(m_export_local_gtdofs.size());
    const int n_import = static_cast<int>(m_import_off_rank_gtdofs.size());

    std::vector<double> send_buf(static_cast<std::size_t>(n_export) * kVDim);
    // The recv buffer is an mfem::Vector so it can flow into the
    // device-side kernel via Read(). MPI fills it on the host; the
    // memory manager will migrate it to the device on first Read.
    mfem::Vector recv_buf(n_import * kVDim);
    {
        const double* x_host = x.HostRead();
        double* recv_host = recv_buf.HostWrite();  // mark as host-written
                                                   // (we will fill via MPI)
        (void)recv_host;

        for (int s = 0; s < n_export; ++s)
        {
            const int g_x = m_export_local_gtdofs[s];
            const auto it = m_gtdof_lookup.find(g_x);
            MFEM_VERIFY(it != m_gtdof_lookup.end(),
                        "MortarConstraintOperator::Mult: requested gtdof "
                        << g_x << " has no entry in gtdof_xyz_lookup");
            const std::array<int, 3>& g_xyz = it->second;
            for (int c = 0; c < kVDim; ++c)
            {
                const int gd = g_xyz[c];
                if (gd < 0)
                {
                    send_buf[s * kVDim + c] = 0.0;
                    continue;
                }
                MFEM_ASSERT(gd >= static_cast<int>(my_first_tdof)
                            && gd < static_cast<int>(my_end_tdof),
                            "MortarConstraintOperator::Mult: peer requested "
                            "gtdof component " << gd << " not in this "
                            "rank's FES TDOF range");
                const int local_idx = gd - static_cast<int>(my_first_tdof);
                send_buf[s * kVDim + c] = x_host[local_idx];
            }
        }
    }

    // Compute Alltoallv counts/displs in element units of (vdim doubles).
    std::vector<int> send_counts_dbl(n_ranks);
    std::vector<int> send_displs_dbl(n_ranks);
    std::vector<int> recv_counts_dbl(n_ranks);
    std::vector<int> recv_displs_dbl(n_ranks);
    for (int r = 0; r < n_ranks; ++r)
    {
        send_counts_dbl[r] = m_import_send_counts[r] * kVDim;
        send_displs_dbl[r] = m_import_send_displs[r] * kVDim;
        recv_counts_dbl[r] = m_import_recv_counts[r] * kVDim;
        recv_displs_dbl[r] = m_import_recv_displs[r] * kVDim;
    }

    // MPI_Alltoallv operates on host pointers. Get a host-write
    // pointer to recv_buf so the memory manager registers the
    // imminent host write (DEVICE_DEBUG will validate this).
    MPI_Alltoallv(send_buf.data(), send_counts_dbl.data(),
                  send_displs_dbl.data(), MPI_DOUBLE,
                  recv_buf.HostWrite(), recv_counts_dbl.data(),
                  recv_displs_dbl.data(), MPI_DOUBLE,
                  comm);

    // -----------------------------------------------------------------
    // Step 2 (DEVICE) — zero y, then mfem::forall over m_n_active_rows.
    //
    // Each thread handles one row, computing its kVDim outputs:
    //
    //   for c in 0..kVDim:
    //     g_n = m_row_g_n_local[i*kVDim + c];
    //     if (g_n < 0) continue;            // sentinel
    //     y_c = D_kk * x[g_n];
    //     for csr_entry in [csr_off[i], csr_off[i+1]):
    //       g_m_local = m_csr_g_m_local[csr_entry*kVDim + c];
    //       g_m_recv  = m_csr_g_m_recv [csr_entry*kVDim + c];
    //       if (g_m_local >= 0)      u_m = x[g_m_local];
    //       else if (g_m_recv >= 0)  u_m = recv_buf[g_m_recv];
    //       else                     continue;       // both -1: sentinel
    //       y_c -= A[csr_entry] * u_m;
    //     y[lambda_off + c] = y_c;
    //
    // Reads: x (FES-local), recv_buf (off-rank import), all of the
    //   m_row_* / m_csr_* flat arrays.
    // Writes: y (lambda-local).
    // -----------------------------------------------------------------
    y = 0.0;  // mfem::Vector::operator=(double) is device-aware

    if (m_n_active_rows == 0) { return; }   // nothing to do

    const double* d_x        = x.Read();
    const double* d_recv     = recv_buf.Read();
    const double* d_row_D    = m_row_D.Read();
    const int*    d_g_n_loc  = m_row_g_n_local.Read();
    const int*    d_csr_off  = m_row_csr_off.Read();
    const int*    d_lam_off  = m_row_lambda_off.Read();
    const double* d_csr_A    = m_csr_A.Read();
    const int*    d_g_m_loc  = m_csr_g_m_local.Read();
    const int*    d_g_m_recv = m_csr_g_m_recv.Read();
    double*       d_y        = y.Write();

    // Capture kVDim by value for the kernel — it's a constexpr int but
    // some toolchains warn on capturing static constexpr in lambdas.
    const int vdim = kVDim;

    mfem::forall(m_n_active_rows, [=] MFEM_HOST_DEVICE (int i)
    {
        const double D_kk = d_row_D[i];
        const int    csr_a = d_csr_off[i];
        const int    csr_b = d_csr_off[i + 1];
        const int    lam_off = d_lam_off[i];

        for (int c = 0; c < vdim; ++c)
        {
            const int gn_loc = d_g_n_loc[i * vdim + c];
            if (gn_loc < 0)            // sentinel: skip; y already zero
            {
                continue;
            }
            double y_c = D_kk * d_x[gn_loc];
            for (int e = csr_a; e < csr_b; ++e)
            {
                const int gm_loc  = d_g_m_loc [e * vdim + c];
                const int gm_recv = d_g_m_recv[e * vdim + c];
                double u_m;
                if (gm_loc >= 0)        { u_m = d_x[gm_loc]; }
                else if (gm_recv >= 0)  { u_m = d_recv[gm_recv]; }
                else                    { continue; }   // sentinel
                y_c -= d_csr_A[e] * u_m;
            }
            d_y[lam_off + c] = y_c;
        }
    });
}

//==============================================================================
// MultTranspose — y = C^T * x
//
// Reverse of Mult: x is the lambda-side vector (local row range),
// y is the FES TDOF residual contribution (local FES TDOF range
// for THIS rank's contributions; off-rank contributions are staged
// in an export buffer and Alltoallv'd to the owners, who element-
// wise ADD them into their local y).
//
// Step 1 — zero y AND the export staging buffer.
// Step 2 — walk edge mortars (with row-owner filter), face mortars;
//          per-pair scatter writing to local y or to export staging.
// Step 3 — Alltoallv export staging back to owners; receivers ADD
//          received values into their local y.
//
// The staging buffer is sized to mirror the IMPORT recv buffer
// (n_import * vdim doubles) and uses the same per-rank counts /
// displs in reverse — i.e., the buffer for rank r's import slots
// becomes this rank's export-to-rank-r staging area.
//==============================================================================
void MortarConstraintOperator::MultTranspose(const mfem::Vector& x,
                                             mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::mortar_constraint_operator::mult_transpose");

    MFEM_VERIFY(x.Size() == Height(),
                "MortarConstraintOperator::MultTranspose: input size "
                << x.Size() << " != Height() " << Height());
    MFEM_VERIFY(y.Size() == Width(),
                "MortarConstraintOperator::MultTranspose: output size "
                << y.Size() << " != Width() " << Width());

    MPI_Comm comm = m_classifier.Comm();
    const int n_ranks = m_classifier.NRanks();
    const HYPRE_BigInt my_first_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[0];
    const HYPRE_BigInt my_end_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[1];

    // -----------------------------------------------------------------
    // Phase 4.3.B / Batch X — first-pass GPU port note.
    //
    // The forward Mult is parallelizable as a single mfem::forall over
    // m_n_active_rows because each row's OUTPUT y entry is unique
    // (no row-row collisions). MultTranspose is NOT directly
    // parallelizable the same way: multiple rows can scatter into the
    // same y entry (a mortar gtdof FES-local on this rank can be
    // referenced from many pair blocks), and the off-rank export
    // staging is also a many-to-one accumulation.
    //
    // For "first pass" GPU readiness we keep MultTranspose as a single
    // sequential walk over the flat arrays on the host. The flat
    // arrays themselves are mfem::Vector / mfem::Array<int>, so they
    // remain DEVICE_DEBUG-clean — we just don't yet use mfem::forall
    // here. A follow-up batch can convert to atomic-add scatter on
    // device once the rest of the GPU stack is validated.
    // -----------------------------------------------------------------
    const int n_import = static_cast<int>(m_import_off_rank_gtdofs.size());
    const int n_export = static_cast<int>(m_export_local_gtdofs.size());

    // Zero y. On real builds this happens through the memory manager
    // — if y was last touched on device, this clears device memory.
    y = 0.0;

    // Host-side staging buffer for off-rank contributions. AOS
    // (slot, component). Filled by the host walk below; sent via
    // MPI_Alltoallv.
    std::vector<double> export_stage(
        static_cast<std::size_t>(n_import) * kVDim, 0.0);

    // -----------------------------------------------------------------
    // Host walk over the flat arrays. Reads x (lambda-side), writes
    // y (FES-local) and export_stage (off-rank staging).
    //
    // The flat arrays already encode every (row, csr_entry, c) tuple
    // we need to scatter to. Sentinels are -1 in m_csr_g_m_local /
    // m_csr_g_m_recv and skipped just like Mult does.
    // -----------------------------------------------------------------
    if (m_n_active_rows > 0)
    {
        const double* h_x        = x.HostRead();
        const double* h_row_D    = m_row_D.HostRead();
        const int*    h_g_n_loc  = m_row_g_n_local.HostRead();
        const int*    h_csr_off  = m_row_csr_off.HostRead();
        const int*    h_lam_off  = m_row_lambda_off.HostRead();
        const double* h_csr_A    = m_csr_A.HostRead();
        const int*    h_g_m_loc  = m_csr_g_m_local.HostRead();
        const int*    h_g_m_recv = m_csr_g_m_recv.HostRead();
        double*       h_y        = y.HostReadWrite();   // we += into y

        const int vdim = kVDim;

        for (int i = 0; i < m_n_active_rows; ++i)
        {
            const double D_kk    = h_row_D[i];
            const int    csr_a   = h_csr_off[i];
            const int    csr_b   = h_csr_off[i + 1];
            const int    lam_off = h_lam_off[i];

            for (int c = 0; c < vdim; ++c)
            {
                const int gn_loc = h_g_n_loc[i * vdim + c];
                if (gn_loc < 0) { continue; }   // sentinel
                const double xi = h_x[lam_off + c];

                // Diagonal contribution: y[gn_loc] += D_kk * xi.
                // Always FES-local under Batch N's row-owner invariant.
                h_y[gn_loc] += D_kk * xi;

                // Off-diagonal -A_kl * xi contributions over csr.
                for (int e = csr_a; e < csr_b; ++e)
                {
                    const double A_kl = h_csr_A[e];
                    const int gm_loc  = h_g_m_loc [e * vdim + c];
                    const int gm_recv = h_g_m_recv[e * vdim + c];
                    const double v = -A_kl * xi;
                    if (gm_loc >= 0)
                    {
                        h_y[gm_loc] += v;
                    }
                    else if (gm_recv >= 0)
                    {
                        // Off-rank: gm_recv is already (slot * vdim + c),
                        // so it indexes directly into export_stage.
                        export_stage[gm_recv] += v;
                    }
                    // else: sentinel — drop.
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // MPI_Alltoallv — return off-rank contributions to their owners.
    //
    // The IMPORT topology shipped each off-rank gtdof FROM its owner
    // TO us. The EXPORT topology is the mirror: ship contributions
    // FROM us TO the owner. Counts/displs swap roles correspondingly.
    // -----------------------------------------------------------------
    std::vector<double> recv_export(
        static_cast<std::size_t>(n_export) * kVDim, 0.0);

    std::vector<int> send_counts_dbl(n_ranks);
    std::vector<int> send_displs_dbl(n_ranks);
    std::vector<int> recv_counts_dbl(n_ranks);
    std::vector<int> recv_displs_dbl(n_ranks);
    for (int r = 0; r < n_ranks; ++r)
    {
        // Reverse direction: what we IMPORTED in Mult is what we EXPORT
        // here, and vice versa.
        send_counts_dbl[r] = m_import_recv_counts[r] * kVDim;
        send_displs_dbl[r] = m_import_recv_displs[r] * kVDim;
        recv_counts_dbl[r] = m_import_send_counts[r] * kVDim;
        recv_displs_dbl[r] = m_import_send_displs[r] * kVDim;
    }

    MPI_Alltoallv(export_stage.data(), send_counts_dbl.data(),
                  send_displs_dbl.data(), MPI_DOUBLE,
                  recv_export.data(), recv_counts_dbl.data(),
                  recv_displs_dbl.data(), MPI_DOUBLE,
                  comm);

    // -----------------------------------------------------------------
    // Add received off-rank contributions into our local y.
    //
    // For each export slot s (= peer-requested gtdof we own), the
    // received doubles are the contribution PEERS computed for OUR
    // local gtdof m_export_local_gtdofs[s], component c. Look up the
    // actual local component gtdof via gtdof_xyz_lookup and add into y.
    // -----------------------------------------------------------------
    if (n_export > 0)
    {
        double* h_y = y.HostReadWrite();
        for (int s = 0; s < n_export; ++s)
        {
            const int g_x = m_export_local_gtdofs[s];
            const auto it = m_gtdof_lookup.find(g_x);
            MFEM_VERIFY(it != m_gtdof_lookup.end(),
                        "MultTranspose: peer-requested gtdof " << g_x
                        << " not in gtdof_xyz_lookup");
            const std::array<int, 3>& g_xyz = it->second;
            for (int c = 0; c < kVDim; ++c)
            {
                const int gd = g_xyz[c];
                if (gd < 0) { continue; }  // sentinel — peer sent 0
                MFEM_ASSERT(gd >= static_cast<int>(my_first_tdof)
                            && gd < static_cast<int>(my_end_tdof),
                            "MultTranspose: peer-requested gtdof component "
                            "not in our FES TDOF range");
                h_y[gd - static_cast<int>(my_first_tdof)]
                    += recv_export[s * kVDim + c];
            }
        }
    }
}

//==============================================================================
// ComputeInvDiagSchur — Phase 4.3 / Batch R
//
// Computes diag(C * diag(K)^{-1} * C^T) directly from the per-pair
// blocks, matching the formula used in saddle_point_solver.cpp's
// BuildInvDiagSchur(HypreParMatrix C, ...).
//
// Per-pair-block contribution to row (block, k, c):
//   S = D[k]^2 * inv_diag_K[g_n_c]
//       + sum_l (A_{kl}^2 * inv_diag_K[g_m_c])
//
// where g_n_c, g_m_c are the c-component global TDOFs of the
// nonmortar and mortar nodes. The mortar TDOFs may be off-rank, so
// we Allgatherv the full inv_diag_K array once at the start —
// matching how the existing HypreParMatrix-path BuildInvDiagSchur
// gathers inv_diag_K, since the size is small (Width() per rank,
// summing to NGlobalTdofs() globally).
//==============================================================================
mfem::Vector MortarConstraintOperator::ComputeInvDiagSchur(
    const mfem::Vector& inv_diag_K_local) const
{
    CALI_CXX_MARK_SCOPE(
        "mortar_pbc::mortar_constraint_operator::compute_inv_diag_schur");

    MFEM_VERIFY(inv_diag_K_local.Size() == Width(),
                "ComputeInvDiagSchur: inv_diag_K_local size "
                << inv_diag_K_local.Size() << " != Width() " << Width());

    // ------------------------------------------------------------------
    // Phase 4.3.B / Batch X — host-only by design.
    //
    // ComputeInvDiagSchur runs ONCE per Newton step (called by
    // SaddlePointSolver during preconditioner setup, before the
    // Krylov iterations begin). It is not in the matvec hot path.
    //
    // Two reasons to keep it host-only for now:
    //   1. The MPI_Allgatherv of inv_diag_K is host-only anyway.
    //   2. The body uses std::map (m_gtdof_lookup) which is not
    //      GPU-friendly. Refactoring this into flat arrays is
    //      possible but provides little benefit since the cost is
    //      amortised across thousands of Krylov iterations.
    //
    // We use HostRead / HostReadWrite on input and output Vectors
    // so the memory manager validates the access pattern under
    // DEVICE_DEBUG.
    // ------------------------------------------------------------------

    MPI_Comm comm = m_classifier.Comm();
    const int my_rank = m_classifier.Rank();
    const int n_ranks = m_classifier.NRanks();
    const HYPRE_BigInt my_first_tdof =
        m_classifier.Fes().GetTrueDofOffsets()[0];

    // -----------------------------------------------------------------
    // Step 1 — Allgatherv inv_diag_K_local into a global array.
    // The mortar gtdofs in our pair blocks may belong to any rank,
    // so we need a global lookup. Mirrors the existing pattern in
    // saddle_point_solver.cpp::BuildInvDiagSchur.
    // -----------------------------------------------------------------
    const int n_local = inv_diag_K_local.Size();
    std::vector<int> all_counts(n_ranks, 0);
    MPI_Allgather(&n_local, 1, MPI_INT, all_counts.data(), 1,
                  MPI_INT, comm);

    int n_global = 0;
    std::vector<int> recv_counts(n_ranks);
    std::vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r)
    {
        displs[r] = n_global;
        recv_counts[r] = all_counts[r];
        n_global += all_counts[r];
    }

    std::vector<double> Dinv_global(static_cast<std::size_t>(n_global), 0.0);
    // Read inv_diag_K_local from host (will migrate from device if
    // dirty there). MPI consumes the host pointer.
    MPI_Allgatherv(inv_diag_K_local.HostRead(), n_local, MPI_DOUBLE,
                   Dinv_global.data(), recv_counts.data(),
                   displs.data(), MPI_DOUBLE, comm);

    // -----------------------------------------------------------------
    // Step 2 — walk per-pair blocks and accumulate S_i for each
    // local constraint row. Same FacePairs() iteration order as
    // Mult / MultTranspose so row indices align with Height().
    // -----------------------------------------------------------------
    mfem::Vector schur_diag(Height());
    // Mark the entire vector as host-written for the upcoming
    // accumulation, AND keep a raw host pointer in scope to use for
    // all subsequent writes. Going through operator()/[] for every
    // index is more fragile under DEVICE_DEBUG (each access re-checks
    // the memory manager state) and slower than a single raw pointer.
    double* sd_data = schur_diag.HostWrite();
    for (int i = 0; i < schur_diag.Size(); ++i) { sd_data[i] = 0.0; }

    int row_offset = 0;

    // ----- edge mortar contributions (with row-owner filter) -----
    for (const auto& lep : m_local_edge_pairs)
    {
        const int n_n = lep.nonmortar_edge.NumNodes();
        const int n_m = lep.mortar_edge.NumNodes();

        for (int k = 0; k < n_n; ++k)
        {
            const int g_n_x = lep.nonmortar_edge.gtdofs_x[k];
            const int owner =
                (g_n_x >= 0)
                ? m_classifier.GtdofOwnerRank(g_n_x)
                : -1;
            if (owner != my_rank) { continue; }

            const double D_kk = lep.block.D_nm(k);
            if (D_kk == 0.0)
            {
                row_offset += kVDim;
                continue;
            }

            for (int c = 0; c < kVDim; ++c)
            {
                int g_n_c;
                if (c == 0) { g_n_c = lep.nonmortar_edge.gtdofs_x[k]; }
                else if (c == 1) { g_n_c = lep.nonmortar_edge.gtdofs_y[k]; }
                else              { g_n_c = lep.nonmortar_edge.gtdofs_z[k]; }
                if (g_n_c < 0) { continue; }

                // Diagonal term: D[k]^2 * (K^-1)_{g_n_c}.
                double s = D_kk * D_kk * Dinv_global[g_n_c];

                // Off-diagonal terms: sum_l A_kl^2 * (K^-1)_{g_m_c}.
                for (int l = 0; l < n_m; ++l)
                {
                    const double A_kl = lep.block.A_m(k, l);
                    if (A_kl == 0.0) { continue; }
                    int g_m_c;
                    if (c == 0) { g_m_c = lep.mortar_edge.gtdofs_x[l]; }
                    else if (c == 1) { g_m_c = lep.mortar_edge.gtdofs_y[l]; }
                    else              { g_m_c = lep.mortar_edge.gtdofs_z[l]; }
                    if (g_m_c < 0) { continue; }
                    s += A_kl * A_kl * Dinv_global[g_m_c];
                }

                sd_data[row_offset + c] = s;
            }
            row_offset += kVDim;
        }
    }

    // ----- face mortar contributions (in FacePairs() order) -----
    auto accumulate_face_block = [&](const FaceMortarPairBlock& block,
                                     int& ro)
    {
        const int n_n = block.NumNonmortarKept();
        const int* A_I    = block.A_m.GetI();
        const int* A_J    = block.A_m.GetJ();
        const double* A_V = block.A_m.GetData();

        for (int k = 0; k < n_n; ++k)
        {
            const double D_kk = block.D(k);
            const int g_n_x = block.nonmortar_gtdofs[k];
            const auto it = m_gtdof_lookup.find(g_n_x);
            MFEM_VERIFY(it != m_gtdof_lookup.end(),
                        "ComputeInvDiagSchur: face nonmortar gtdof "
                        << g_n_x << " not in gtdof_xyz_lookup");
            const std::array<int, 3>& g_n_xyz = it->second;

            if (D_kk == 0.0)
            {
                ro += kVDim;
                continue;
            }

            for (int c = 0; c < kVDim; ++c)
            {
                const int g_n_c = g_n_xyz[c];
                if (g_n_c < 0) { continue; }

                double s = D_kk * D_kk * Dinv_global[g_n_c];

                for (int idx = A_I[k]; idx < A_I[k + 1]; ++idx)
                {
                    const int l = A_J[idx];
                    const double A_kl = A_V[idx];
                    if (A_kl == 0.0) { continue; }
                    const int g_m_x = block.mortar_gtdofs[l];
                    const auto it_m = m_gtdof_lookup.find(g_m_x);
                    MFEM_VERIFY(it_m != m_gtdof_lookup.end(),
                                "ComputeInvDiagSchur: face mortar gtdof "
                                << g_m_x << " not in gtdof_xyz_lookup");
                    const int g_m_c = it_m->second[c];
                    if (g_m_c < 0) { continue; }
                    s += A_kl * A_kl * Dinv_global[g_m_c];
                }

                sd_data[ro + c] = s;
            }
            ro += kVDim;
        }
    };

    for (const auto& tup : m_classifier.FacePairs())
    {
        const std::string& axis            = std::get<0>(tup);
        const std::string& mortar_label    = std::get<1>(tup);
        const std::string& nonmortar_label = std::get<2>(tup);

        const FaceMortarPairBlock* quad_block = nullptr;
        const FaceMortarPairBlock* tri_block  = nullptr;
        for (const auto& lpb : m_classifier.PairBlocks())
        {
            if (lpb.axis_pair != axis
                || lpb.mortar_label != mortar_label
                || lpb.nonmortar_label != nonmortar_label) { continue; }
            if (lpb.geometry_kind == "quad") { quad_block = &lpb.block; }
            else if (lpb.geometry_kind == "tri") { tri_block = &lpb.block; }
        }
        if (quad_block != nullptr) { accumulate_face_block(*quad_block,
                                                            row_offset); }
        if (tri_block  != nullptr) { accumulate_face_block(*tri_block,
                                                            row_offset); }
    }

    MFEM_ASSERT(row_offset == Height(),
                "ComputeInvDiagSchur: emitted " << row_offset
                << " rows but Height() = " << Height());

    // -----------------------------------------------------------------
    // Step 3 — invert (matching BuildInvDiagSchur's tiny-tolerance
    // convention; entries with magnitude < 1e-300 stay at zero, which
    // is correct because the corresponding block-Jacobi action is a
    // no-op on those rows).
    //
    // Suppress unused-variable warning for my_first_tdof — it's
    // unused here because Dinv_global is indexed by GLOBAL TDOF, not
    // local. We keep the binding in case future maintainers add a
    // local-only optimization that needs it.
    // -----------------------------------------------------------------
    (void)my_first_tdof;

    mfem::Vector inv_schur(Height());
    constexpr double kTiny = 1.0e-300;
    {
        // sd_data is the host-resident schur_diag we wrote into above.
        // inv_schur is fresh; declare the host write before the loop.
        double* iv_data = inv_schur.HostWrite();
        for (int i = 0; i < Height(); ++i)
        {
            const double d = sd_data[i];
            iv_data[i] = (std::abs(d) > kTiny) ? (1.0 / d) : 0.0;
        }
    }
    return inv_schur;
}

}  // namespace mortar_pbc
