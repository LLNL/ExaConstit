// Phase 4.3 / Batch O — Element-assembly constraint operator skeleton.
//
// This file declares MortarConstraintOperator, the element-assembly (EA)
// counterpart to the HypreParMatrix path in ConstraintBuilder3D::
// BuildHypreParMatrix(). The EA path keeps per-pair local D and A_m
// blocks and applies them matrix-free in Mult / MultTranspose, instead
// of assembling a global sparse C and using HypreParMatrix's matvec.
//
// Why both paths exist:
//   - HypreParMatrix path: needed for setup-style validation
//     (Build() returns a CSR for offline inspection / row-wise checks),
//     and for prototype runs where Hypre's matvec is the simpler
//     thing.
//   - EA path: needed for production. The HypreParMatrix path requires
//     Hypre's vector-type matvec to be GPU-correct (still a known
//     issue across Hypre versions for vector-DOF problems), and it
//     forces global sparsity-pattern management. The EA path matches
//     the matrix-free style ExaConstit already uses for K and slots
//     into mfem::forall over pairs naturally.
//
// API contract:
//   - Inherits mfem::Operator. Mult and MultTranspose follow MFEM's
//     standard semantics (overwrite y on the way out — no
//     accumulation).
//   - Works inside an mfem::BlockOperator alongside K (the saddle-
//     point solver wires it as `BlockOperator(0,1) = &mortar_op` and
//     uses mfem::TransposeOperator(&mortar_op) for the (1,0) block).
//   - Works inside an mfem::BlockNonlinearForm Jacobian path. Since
//     C is linear in u, the Jacobian-of-the-residual returned via
//     GetGradient(x) is the operator itself, independent of x. A
//     thin BlockNonlinearFormIntegrator-style adapter (Phase 4.3 /
//     Batch R) wraps this.
//
// What is NOT in scope here:
//   - Non-conforming face mortars. The Python prototype's Phase 3.5
//     (Sutherland-Hodgman polygon clipping) was never implemented;
//     the C++ port mirrors that. Non-conforming faces are deferred
//     to a future phase. 2D edge mortars ARE non-conforming-capable
//     (interval overlap) on both sides — we picked that up because
//     the Python 2D code had it from the start.
//   - GPU port. Phase 4.3.A is CPU only. Phase 4.3.B (Batch X+1)
//     ports Mult / MultTranspose to mfem::forall.
//
// Phase 4.3 batch sequence:
//   - Batch O (this batch): design + skeleton + doc.
//   - Batch P: Mult / MultTranspose CPU implementation.
//   - Batch Q: A/B validation harness (HypreParMatrix vs EA matvec
//     equivalence to FP precision; EA-path patch test).
//   - Batch R: BlockNonlinearForm adapter.
//   - Batch S: --constraint-storage=ea CLI flag and CMake option.
//
#pragma once

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "types_3d.hpp"
#include "utilities/mechanics_log.hpp"
#include "mfem.hpp"

#include <map>
#include <memory>
#include <vector>

namespace mortar_pbc {

/**
 * @brief Element-assembly constraint operator — applies C and C^T
 *        matrix-free using per-pair local D and A_m blocks.
 *
 * @details
 * `MortarConstraintOperator` inherits `mfem::Operator` and provides
 * `Mult(u, lambda) = C u` and `MultTranspose(lambda, u_residual) =
 * C^T lambda`. It consumes the same per-pair block infrastructure
 * built up through Phase 4.2 (boundary classifier's
 * `PairBlocks()` + `EdgePairs()`), so no new mortar-mathematics
 * code is required — only a new way of applying the same blocks.
 *
 * @par Vector layout
 * - Domain (`Width()`): the FES TDOF vector `u`. Each rank holds
 *   the local TDOFs in `[FES.GetTrueDofOffsets()[0], ...)`. Mortar
 *   gtdofs needed by this rank's pair blocks may be on other ranks
 *   and must be gathered each `Mult` (off-rank import). Built once
 *   at construction time.
 * - Range (`Height()`): the constraint multiplier vector `lambda`,
 *   partitioned per rank in the same FES-aligned scheme as
 *   `BuildHypreParMatrix` (Batch N). `Height()` equals
 *   `ConstraintBuilder3D::NumLocalRows()`.
 *
 * @par Per-pair scatter pattern
 * For each face-mortar block on this rank, with `n_n` local
 * nonmortar rows and `n_m` mortar columns:
 * - `Mult` reads `u_x[g]`, `u_y[g]`, `u_z[g]` for every nonmortar
 *   gtdof `g` (this rank's local TDOF; cheap) and every mortar
 *   gtdof `g'` (potentially off-rank; needs the import buffer).
 * - For each spatial component `c` (x, y, z): writes
 *   `lambda[r+c] += D[k] * u_c[g_n[k]] - sum_l A_m[k,l] u_c[g_m[l]]`.
 * - `MultTranspose` reverses: each lambda entry's contribution
 *   adds to `u_residual[g]` for the corresponding nonmortar /
 *   mortar gtdof. Writes to off-rank `u_residual` entries are
 *   handled via an export buffer (computed at construction).
 *
 * @par Edge-mortar handling
 * Edge mortars are produced redundantly on every rank in
 * `ConstraintBuilder3D::EmitConstraintTriples` (post-Batch-N).
 * The EA path mirrors this: each rank holds its own copy of the 9
 * `MortarBlock2D` blocks (assembled locally at construction time)
 * and applies them with the same row-owner filter
 * (`GtdofOwnerRank(nonmortar_g_xyz[0]) == this rank`).
 *
 * @par Off-rank vector import / export
 * At construction time, the operator computes:
 * - `m_off_rank_mortar_gtdofs`: unique mortar gtdofs (across all
 *   pair blocks on this rank) that are NOT FES-owned by this rank.
 * - `m_off_rank_owner`: per-entry, the FES owner rank.
 * The per-`Mult` exchange uses `MPI_Alltoallv` to gather these
 * values from owner ranks — collective on `m_classifier.Comm()`,
 * but with volume bounded by the rank's portion of the periodic
 * boundary surface (a small fraction of `Width()`). For
 * `MultTranspose`, the same pattern reversed scatters local
 * contributions to off-rank `u_residual` entries.
 *
 * @par Why an MPI_Alltoallv per matvec is acceptable
 * Krylov methods do O(iters) matvecs. Each Alltoallv has volume
 * O(boundary_surface_per_rank / 3), payload size = (boundary
 * vertices touched by this rank's mortar gtdofs) * (vdim doubles).
 * For a 100^3 RVE on 10^6 ranks with ~6% boundary, this is on the
 * order of 100 doubles per matvec per rank. Negligible vs the
 * Krylov work K * u (which dominates). The HypreParMatrix path's
 * matvec also does an off-rank exchange under the hood (Hypre's
 * column-comm pattern); we are not trading off latency, only
 * implementation control.
 *
 * @par GPU portability
 * Phase 4.3.A (CPU): the inner loop over pair blocks runs on host.
 * Phase 4.3.B will port to `mfem::forall` over a flattened pair
 * array. The block-fragment data structure is already CSR-friendly
 * (post-Batch-L `A_m` is `mfem::SparseMatrix`), which makes the
 * forall port mechanical. Off-rank import / export buffers are
 * staged through host memory in Phase 4.3.A; Phase 4.3.B uses
 * pinned buffers + GPU-direct where supported.
 *
 * @par Lifetime
 * The operator holds a `const BoundaryClassifier3D&` reference and
 * does not own it. The classifier must outlive the operator.
 *
 * @see ConstraintBuilder3D::BuildHypreParMatrix — the dual
 *      HypreParMatrix path.
 * @see MortarFaceMortarPairBlock — the per-pair block storage.
 */
class MortarConstraintOperator : public mfem::Operator
{
public:
    /**
     * @brief Construct from a fully-built classifier.
     *
     * @param classifier  The classifier whose `PairBlocks()` and
     *                    `EdgePairs()` provide the per-pair block
     *                    data. Must be fully built (post-
     *                    `RoutePairBlocksToRowOwners`).
     *
     * @par MPI scope
     * Collective on `classifier.Comm()`. Performs:
     *   - 1 `MPI_Alltoall` (off-rank gtdof set sizes)
     *   - 2 `MPI_Alltoallv` (off-rank gtdof index exchange,
     *     building the import/export tables)
     *
     * Construction is intentionally heavyweight; per-`Mult` cost is
     * just one Alltoallv and one local pair-loop.
     */
    explicit MortarConstraintOperator(const BoundaryClassifier3D& classifier);

    ~MortarConstraintOperator() override = default;

    // No copy / move — holds an internal MPI exchange topology that
    // would be cheap to rebuild but expensive to maintain in a
    // valid state under copying.
    MortarConstraintOperator(const MortarConstraintOperator&) = delete;
    MortarConstraintOperator& operator=(const MortarConstraintOperator&) = delete;

    /**
     * @brief Apply C: y = C * x.
     *
     * @param x [in]  FES TDOF vector (this rank's local slice; size
     *                must equal `Width()`).
     * @param y [out] Constraint multiplier vector (this rank's local
     *                slice; size must equal `Height()`). Overwritten,
     *                not accumulated.
     *
     * @par Algorithm (Phase 4.3 / Batch P will implement)
     * @code
     * 1. Import off-rank mortar u-values via Alltoallv.
     * 2. Zero y.
     * 3. For each edge-mortar block whose nonmortar gtdofs are
     *    FES-owned locally:
     *      For each component c in {x, y, z}:
     *        For each nonmortar row k:
     *          y[row_off + c] += D[k] * u_c[g_n[k]]
     *          For each mortar col l:
     *            y[row_off + c] -= A_m(k, l) * u_c[g_m[l]]
     *        row_off += vdim
     * 4. For each face-mortar block in PairBlocks() (already
     *    pre-routed to this rank in Batch N):
     *      Same per-component loop, walking A_m via CSR.
     * @endcode
     *
     * @par MPI scope
     * Collective on `classifier.Comm()`. One Alltoallv (off-rank
     * mortar u-value import).
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Apply C^T: y = C^T * x.
     *
     * @param x [in]  Constraint multiplier vector (this rank's local
     *                slice; size must equal `Height()`).
     * @param y [out] FES TDOF residual vector (this rank's local
     *                slice; size must equal `Width()`). Overwritten,
     *                not accumulated.
     *
     * @par Algorithm (Phase 4.3 / Batch P will implement)
     * @code
     * 1. Zero y AND the off-rank export staging buffer.
     * 2. For each edge-mortar block (with row-owner filter):
     *      For each component c, for each row k, for each col l:
     *        y[g_n[k] for c] += D[k] * x[row_off + c]
     *        y[g_m[l] for c] -= A_m(k, l) * x[row_off + c]
     *           ^-- if g_m[l] is off-rank, write to export[c, off_rank_slot]
     * 3. For each face-mortar block (CSR walk + same logic).
     * 4. Export off-rank contributions via Alltoallv (reverse of
     *    Mult's import); each owner rank ADDS the received entries
     *    into its local y.
     * @endcode
     *
     * @par MPI scope
     * Collective on `classifier.Comm()`. One Alltoallv (off-rank
     * residual export, with element-wise ADD on receive).
     */
    void MultTranspose(const mfem::Vector& x,
                       mfem::Vector& y) const override;

    /**
     * @brief Number of constraint rows owned by this rank.
     *
     * Equal to `Height()`, exposed under a more descriptive name
     * for callers who want to size the multiplier vector.
     */
    int NumLocalRows() const { return Height(); }

    /**
     * @brief Phase 4.3 / Batch R — compute the diagonal of the
     *        Schur-complement preconditioner approximation
     *        \f$\mathrm{diag}(C\,\mathrm{diag}(K)^{-1}\,C^T)\f$,
     *        and return its element-wise reciprocal (the
     *        inverse-Schur diagonal used by block-Jacobi
     *        preconditioning).
     *
     * @details This mirrors `saddle_point_solver.cpp`'s
     * `BuildInvDiagSchur(HypreParMatrix C, ...)` but works directly
     * on the EA per-pair blocks — no global CSR is required, so
     * the EA path can be preconditioned without first building a
     * `HypreParMatrix` form of C.
     *
     * The Schur diagonal entry for constraint row `i` is
     * \f[
     *   S_i = \sum_j C_{ij}^2 \, (K^{-1})_{jj}
     * \f]
     * which decomposes per-pair-block as
     * \f[
     *   S_{(\text{block},k,c)} =
     *     D_k^2 \, (K^{-1})_{g_n^c}
     *     + \sum_l A_{kl}^2 \, (K^{-1})_{g_m^c}
     * \f]
     * where \f$g_n^c\f$ and \f$g_m^c\f$ are the global TDOFs of
     * the nonmortar and mortar nodes' c-components. The mortar
     * `\f$g_m^c\f$` may be off-rank; we Allgatherv the full
     * `inv_diag_K` array once at the start so the lookup is local.
     *
     * @param inv_diag_K_local The local slice of \f$\mathrm{diag}(K)^{-1}\f$
     *                         on this rank (size `Width()`).
     * @return Vector of size `Height()` containing the inverse
     *         Schur-complement diagonal: `inv_schur[i] = 1 / S_i`,
     *         with zero replacing any entry where `|S_i| < 1e-300`
     *         (matching the HypreParMatrix-path convention).
     *
     * @par MPI scope
     * Collective on `m_classifier.Comm()`. One `MPI_Allgather` (int
     * counts) + one `MPI_Allgatherv` (`inv_diag_K` doubles).
     */
    mfem::Vector ComputeInvDiagSchur(
        const mfem::Vector& inv_diag_K_local) const;

    /// Spatial vector dimension. Public so test/diagnostic code can
    /// share it. The mortar machinery is hardcoded to kVDim=3 (3D);
    /// generalising to other vdims would require revisiting the
    /// per-pair scatter contracts.
    static constexpr int kVDim = 3;

    /// Sentinel returned by the flat-array `m_csr_g_m[]` table when
    /// a mortar component is absent (Dirichlet-stripped). The matvec
    /// kernel checks for this and skips the contribution.
    static constexpr int kSentinelIdx = -2147483647;  // INT_MIN+1

private:
    const BoundaryClassifier3D& m_classifier;

    // Edge-mortar blocks for this rank. Assembled at construction
    // (cheap — 9 small dense pairs). Held WITH their (nonmortar,
    // mortar) edge metadata so we can do the row-owner filter.
    struct LocalEdgePair
    {
        MortarBlock2D block;
        EdgeInfo3D    nonmortar_edge;
        EdgeInfo3D    mortar_edge;
    };
    std::vector<LocalEdgePair> m_local_edge_pairs;

    // Cached gtdof_xyz lookup (matches ConstraintBuilder3D's).
    std::map<int, std::array<int, 3>> m_gtdof_lookup;

    // ---- Off-rank import / export topology ----
    //
    // m_import_off_rank_gtdofs:  for each unique mortar gtdof not
    //   FES-owned locally, the global index. Size = total off-rank
    //   gtdofs needed.
    // m_import_local_slot:       for each off-rank gtdof, the slot
    //   in the import buffer. Used during pair-block scatter to
    //   look up u-values.
    // m_import_recv_counts /
    // m_import_recv_displs:      Alltoallv parameters for the
    //   import (per-source-rank counts/displs).
    // m_export_send_counts /
    // m_export_send_displs:      Alltoallv parameters for the
    //   transpose export. Mirror of the import side: what this rank
    //   produces locally for off-rank u_residual destinations.
    //
    // Computed at construction. Re-used on every Mult / MultTranspose.
    std::vector<int> m_import_off_rank_gtdofs;
    std::map<int, int> m_import_gtdof_to_slot;
    std::vector<int> m_import_recv_counts;
    std::vector<int> m_import_recv_displs;
    std::vector<int> m_import_send_counts;
    std::vector<int> m_import_send_displs;
    // Per-source-rank list of which LOCAL gtdofs to send out (the
    // "mirror image" of m_import_off_rank_gtdofs from each owner's
    // perspective). Built via the inverse of the import topology.
    std::vector<int> m_export_local_gtdofs;

    // ---- Phase 4.3.B / Batch X — flat per-row arrays for GPU matvec --
    //
    // The CPU implementation walks per-pair blocks via std::map and
    // raw CSR pointers. That is not GPU-portable. The flat-array
    // form, built once at construction time, mirrors what the matvec
    // hot path needs:
    //
    // m_n_active_rows:       count of constraint rows this rank owns
    //                        (excludes edge rows the row-owner filter
    //                        skips). Equal to Height() / kVDim.
    //
    // m_row_lambda_off[i]:   first lambda index this row writes
    //                        (= i * kVDim, but stored to be explicit
    //                        for readers).
    //
    // m_row_D[i]:            D_kk value for row i. Pre-baked diagonal
    //                        coefficient; same for all kVDim
    //                        components of the row.
    //
    // m_row_g_n_local[i*3+c]: index into the local FES TDOF vector
    //                        (= x slice on this rank) for the
    //                        c-component of row i's nonmortar node.
    //                        -1 means sentinel (Dirichlet-stripped
    //                        component); kernel skips such entries.
    //                        By Batch N's invariant the nonmortar
    //                        component is ALWAYS FES-local for owned
    //                        rows, so this never encodes an off-rank
    //                        index — only "local" or "sentinel".
    //
    // m_row_csr_off[i]:      prefix-sum start index into m_csr_A /
    //                        m_csr_g_m_local / m_csr_g_m_recv for
    //                        row i's off-diagonal contributions.
    //                        m_row_csr_off[N] is the total CSR entry
    //                        count.
    //
    // m_csr_A[k]:            A_kl value for CSR entry k.
    //
    // m_csr_g_m_local[k*3+c]: local FES TDOF index for the mortar
    //                        component c of CSR entry k, or -1 if
    //                        this component is off-rank (look in
    //                        m_csr_g_m_recv) or sentinel-stripped
    //                        (in which case m_csr_g_m_recv is also
    //                        -1, signalling "skip").
    //
    // m_csr_g_m_recv[k*3+c]: recv-buffer slot index (already
    //                        multiplied by kVDim and offset by c, so
    //                        ready to use as recv_buf[idx]). -1 if
    //                        the component is local or sentinel.
    //
    // Kernel decision tree (per (k, c)):
    //     li = m_csr_g_m_local[k*3+c];
    //     ri = m_csr_g_m_recv [k*3+c];
    //     if (li < 0 && ri < 0)     skip;             // sentinel
    //     else if (li >= 0)         u_m = x[li];      // local
    //     else                      u_m = recv_buf[ri];   // off-rank
    //
    // All these are mfem::Vector / mfem::Array<int> so the memory
    // manager owns them and Read/Write annotations work.
    int m_n_active_rows = 0;
    mfem::Array<int> m_row_lambda_off;
    mfem::Vector     m_row_D;
    mfem::Array<int> m_row_g_n_local;     // size = m_n_active_rows * kVDim
    mfem::Array<int> m_row_csr_off;       // size = m_n_active_rows + 1
    mfem::Vector     m_csr_A;             // size = total CSR entries
    mfem::Array<int> m_csr_g_m_local;     // size = total CSR entries * kVDim
    mfem::Array<int> m_csr_g_m_recv;      // size = total CSR entries * kVDim

    // Helper called once at construction to populate all of the
    // m_row_* and m_csr_* flat arrays from the per-pair-block data
    // (m_local_edge_pairs + classifier.PairBlocks()). Consolidates
    // what was the per-pair-block walk in Mult / MultTranspose's
    // host-side code into a one-shot setup pass, leaving the matvec
    // free to run as a single mfem::forall over m_n_active_rows.
    void BuildFlatRowArrays();
};

}  // namespace mortar_pbc
