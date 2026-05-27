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
// Phase 5.9 / Batch A.3.d — Component-restricted PBC filter
// ----------------------------------------------------------
// The operator now carries a runtime-mutable filter spec
// `(m_active_pair_labels, m_comp_mask)` that gates which constraint
// rows are emitted (matching `ConstraintBuilder3D::Build(labels,
// mask)`). The defaults at construction time are "all pairs active,
// all components active" — exactly reproducing pre-5.9 behavior.
//
// `Reset(active_pair_labels, comp_mask)` repopulates the flat
// per-row arrays under a new filter spec, updating `Height()` to
// match. It is **local — no MPI calls** — and must be called with
// the same arguments on every rank (collective by convention, like
// `MPI_Allreduce` parameters). The import/export topology built at
// construction time is unchanged by `Reset`; under a reduced filter
// it over-imports off-rank mortar gtdofs (correct, just wasteful),
// which is acceptable because the import volume is already a small
// fraction of the matvec cost.
//
// Phase 6.0.E — projector-aware construction
// ----------------------------------------------------------
// The operator can now be constructed from a classifier that lives on
// the boundary/LOR submesh plus a SurfaceProjector that translates
// classifier-side submesh true DOFs to parent-volume true DOFs. The
// legacy constructor remains available and is treated as an identity
// projection path until the manager and builder are fully migrated.
//
#pragma once

#include "boundary_classifier_3d.hpp"
#include "constraint_builder_3d.hpp"
#include "surface_projector.hpp"
#include "types_3d.hpp"
#include "utilities/mechanics_log.hpp"
#include "mfem.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
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
 * - Domain (`Width()`): the parent-FES TDOF vector `u`. In legacy
 *   construction the parent FES is the classifier FES. In Phase 6
 *   projector construction the classifier FES is the boundary/LOR
 *   submesh FES, while the parent FES is the volume FES supplied to
 *   the constructor. The flat arrays always store parent-FES local
 *   indices before the first matvec.
 * - Range (`Height()`): the constraint multiplier vector `lambda`,
 *   partitioned per rank in the same FES-aligned scheme as
 *   `BuildHypreParMatrix` (Batch N). `Height()` equals
 *   `ConstraintBuilder3D::NumLocalRows(active_pair_labels,
 *   comp_mask)` under the operator's current filter spec — for the
 *   default "all pairs, all comps" spec this matches the pre-5.9
 *   `NumLocalRows()` value exactly.
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
 * @par Phase 5.9 filter
 * `Reset(active_pair_labels, comp_mask)` rebuilds the per-row flat
 * arrays under a new filter spec. The filter rules match
 * `ConstraintBuilder3D`: a face pair contributes iff its axis is in
 * the active set (derived from labels by the
 * `left/right -> x`, `bottom/top -> y`, `front/back -> z` mapping);
 * an edge mortar group contributes iff BOTH of its perpendicular
 * axes are active. Within active pairs, `comp_mask` filters
 * per-component rows.
 *
 * @par Projector-mediated indexing
 * The classifier may emit either parent-FES true DOFs (legacy path) or
 * submesh-FES true DOFs (Phase 6 path). Setup code routes every
 * classifier-side true DOF through `SurfaceProjector` when one is
 * present. Runtime `Mult` and `MultTranspose` never call the projector;
 * they only read parent-FES local indices and parent-FES off-rank
 * import slots from the flat arrays.
 *
 * @par Higher-order / LOR contract
 * In the Phase 6 Day-1 path the parent mechanics space is a P2
 * tetrahedral H1 space and the classifier space is a P1 boundary
 * submesh obtained by one uniform refinement of the extracted boundary
 * mesh (`lor_depth = 2`). The refined boundary vertices coincide with
 * the parent P2 boundary Lagrange nodes, so projector construction is a
 * true DOF permutation rather than interpolation. For an affine parent
 * field `u(x) = L x`, `Mult(u)` must equal the reference geometric RHS
 * assembled from `ConstraintBuilder3D::EmitRowFactors`:
 *
 * @code
 * g_i = ell_hat_i * sum_k L(component_i, k)
 *                     * period_signed_per_row(3*i + k)
 * @endcode
 *
 * This relation is the smoke-testable sign and indexing invariant for
 * higher-order mortar PBC. If it fails, either classifier row
 * enumeration, surface projection, or the signed-period convention is
 * inconsistent.
 *
 * @par Lifetime
 * Legacy construction holds a `const BoundaryClassifier3D&` reference
 * and does not own it. Projector construction stores shared ownership
 * of the classifier, projector, and parent FES supplied by the caller.
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
     *
     * @par Phase 5.9 default filter
     * The filter spec is initialized to "all face pairs active, all
     * components active" — equivalent to pre-5.9 behavior. Use
     * `Reset(active_pair_labels, comp_mask)` to change this without
     * destroying and rebuilding the operator (which would re-run
     * the construction-time MPI collectives).
     */
    explicit MortarConstraintOperator(const BoundaryClassifier3D& classifier);

    /**
     * @brief Construct from a submesh-side classifier and parent-FES
     *        projector.
     *
     * @param classifier  Fully-built classifier whose FES is the
     *                    boundary/LOR submesh FES.
     * @param projector   Surface projector translating classifier-side
     *                    submesh true DOFs to parent-volume true DOFs.
     * @param parent_fes  Parent volume FES that defines `Width()`, the
     *                    input vector to `Mult`, and the output vector
     *                    of `MultTranspose`.
     *
     * @details This is the Phase 6 constructor. All pair-block metadata
     * is still read from `classifier`, but every true-DOF reference is
     * translated through `projector` while building import/export
     * topology and flat row arrays. The matvec kernels remain unchanged
     * at runtime because those arrays store parent-FES local indices.
     * For `lor_depth = 1` the projector is a boundary-trace
     * permutation from the unrefined surface space to the linear parent
     * space; for `lor_depth = 2` it maps the uniformly refined P1
     * boundary vertices to coincident parent P2 boundary nodes.
     *
     * @par MPI scope
     * Collective on `classifier->Comm()`, matching the legacy
     * constructor. `projector` must have been constructed on the same
     * communicator and against `parent_fes`.
     */
    MortarConstraintOperator(
        std::shared_ptr<const BoundaryClassifier3D> classifier,
        std::shared_ptr<const SurfaceProjector> projector,
        std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes);

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
     * @par Phase 5.9 filter
     * The kernel applies `m_comp_mask` at the per-component loop
     * (skipping filtered components) and uses `m_local_c[c]` as the
     * row-local offset into the lambda vector. Filtered edge / face
     * pairs are already absent from the flat arrays (handled in
     * `BuildFlatRowArrays`).
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
     * @par Phase 5.9 filter
     * Same component-filter mechanism as `Mult` — the host walk
     * reads `x[lam_off + m_local_c[c]]` and skips filtered components.
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
     * @details Phase 5.5 — argument relaxed from a raw
     * `mfem::Vector& inv_diag_K_local` to `const mfem::Solver&
     * K_jacobi_prec` so the function works with any preconditioner
     * that mathematically implements diagonal scaling, without
     * needing the caller to extract its inverse-diagonal values
     * first.
     *
     * The implementation probes `K_jacobi_prec` by applying it to
     * a vector of ones:
     *
     *   y = K_jacobi_prec.Mult(ones)
     *
     * For any solver whose action is `y[i] = inv_diag(K)[i] * x[i]`
     * (the documented contract for this argument — Jacobi /
     * diagonal scaling), `Mult(ones, _)` returns `inv_diag(K)`
     * directly. The remainder of the algorithm (Allgatherv +
     * per-pair-block walk) is unchanged from the previous
     * Vector-based API.
     *
     * Solvers satisfying the contract:
     *   - `mortar_pbc::DiagonalScaler` (always)
     *   - `mfem::OperatorJacobiSmoother` (when iterative_mode == false)
     *   - ExaConstit's `MechOperatorJacobiSmoother` (when
     *     iterative_mode == false)
     *   - Hypre's `HypreDiagScale` (always)
     *
     * Solvers NOT satisfying the contract (do NOT pass these):
     *   - AMG, ILU, GMG, Gauss-Seidel, Chebyshev, ... — these
     *     implement non-diagonal actions; the probe would return
     *     non-diagonal values and the resulting inv_diag_S would be
     *     wrong (silently — there is no runtime check against this).
     *
     * The contract is documented rather than runtime-enforced
     * because the set of valid Jacobi-style solvers is open-ended
     * and a runtime check would require either a marker base class
     * or a Vector-of-ones probe + sparsity check, neither of which
     * is justified given the small set of call sites and the
     * unambiguous responsibility (caller picks the right prec).
     *
     * Phase 5.9 — the per-pair-block walk uses the same filter as
     * `BuildFlatRowArrays` so the Schur diagonal aligns with the
     * filtered `Height()`. Filtered pairs are skipped at the outer
     * iteration; filtered components are skipped at the inner
     * per-c loop; `row_offset` strides by `m_n_comps_active`.
     *
     * @param K_jacobi_prec  Preconditioner whose `Mult(ones, _)`
     *                       action returns `diag(K)^{-1}`. Sized so
     *                       that `K_jacobi_prec.Height() == Width()`.
     * @return Vector of size `Height()` containing the inverse
     *         Schur-complement diagonal: `inv_schur[i] = 1 / S_i`,
     *         with zero replacing any entry where `|S_i| < 1e-300`
     *         (matching the HypreParMatrix-path convention).
     *
     * @par MPI scope
     * Collective on `m_classifier.Comm()`. One `MPI_Allgather`
     * (int counts) + one `MPI_Allgatherv` (`inv_diag_K` doubles)
     * — same as before. The added `Mult(ones)` probe is local
     * (no extra collectives).
     */
    mfem::Vector ComputeInvDiagSchur(
        const mfem::Solver& K_jacobi_prec) const;

    /**
     * @brief Phase 5.9 / Batch A.3.d — repopulate flat-row arrays
     *        under a new `(active_pair_labels, comp_mask)` filter
     *        spec.
     *
     * @param active_pair_labels  Mortar-side face labels of pairs to
     *                            include. Same convention as
     *                            `ConstraintBuilder3D::Build(labels,
     *                            mask)`. May be passed as either
     *                            mortar or nonmortar side; the
     *                            label→axis mapping is the same
     *                            either way.
     * @param comp_mask           Per-spatial-component gate. Rows for
     *                            components `c` with
     *                            `comp_mask[c] == false` are skipped.
     *
     * @details
     * Resets the operator's per-row flat arrays (`m_row_D`,
     * `m_row_g_n_local`, `m_row_csr_off`, `m_csr_A`,
     * `m_csr_g_m_local`, `m_csr_g_m_recv`, `m_row_lambda_off`,
     * `m_n_active_rows`) and updates `Height()` to match. The
     * import/export topology is **not** rebuilt — it was sized at
     * construction time for the "all pairs, all comps" spec, and
     * under any reduced filter it correctly over-imports off-rank
     * mortar gtdofs (some imported values are simply never read).
     *
     * @par Pair-completeness validation
     * `Reset` itself does NOT validate that `active_pair_labels`
     * contains both halves of every pair (the classifier's
     * `ArePaired` check). That validation is the responsibility of
     * the calling layer (`MortarPbcManager::RebuildForActiveSpec`
     * in Phase 5.9.A.4) where the user-facing TOML spec is
     * interpreted and friendly error messages can be issued.
     *
     * @par MPI scope
     * **Local — no MPI calls.** All ranks must call `Reset` with
     * identical arguments (collective by convention), because the
     * import/export topology is symmetric and any inconsistency
     * between ranks' filter specs would cause a per-`Mult` matvec
     * to write into the wrong lambda slots on one side. The
     * topology itself is unchanged, so all-ranks exchange the same
     * data they did before; only the kernel's per-component skip
     * pattern differs across ranks if the filter args do.
     */
    void Reset(const std::vector<std::string>& active_pair_labels,
               const std::array<bool, 3>& comp_mask);

    /**
     * @brief Phase 5.9 / Batch A.3.d — current active pair labels.
     */
    const std::vector<std::string>& ActivePairLabels() const
    {
        return m_active_pair_labels;
    }

    /**
     * @brief Phase 5.9 / Batch A.3.d — current component mask.
     */
    const std::array<bool, 3>& CompMask() const { return m_comp_mask; }

    /**
     * @brief MPI communicator for this operator.
     *
     * @details Equal to `classifier.Comm()`. Exposed so callers
     * (e.g. `SaddlePointSolver`) can drive collectives on the same
     * communicator as the underlying constraint topology without
     * having to also accept a comm argument.
     */
    MPI_Comm Comm() const { return m_classifier.Comm(); }

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

    // Phase 6 ownership hooks. The legacy constructor leaves these
    // empty and relies on caller-owned objects. The projector-aware
    // constructor fills them so the operator can share lifetime with
    // MortarPbcManager and the setup infrastructure.
    std::shared_ptr<const BoundaryClassifier3D> m_classifier_owner;
    std::shared_ptr<const SurfaceProjector> m_projector;
    std::shared_ptr<const mfem::ParFiniteElementSpace> m_parent_fes_owner;
    const mfem::ParFiniteElementSpace* m_parent_fes_raw = nullptr;

    // Edge-mortar blocks for this rank. Assembled at construction
    // (cheap — 9 small dense pairs). Held WITH their (nonmortar,
    // mortar) edge metadata so we can do the row-owner filter.
    //
    // Phase 5.9 / Batch A.3.d — these are NOT filtered at
    // construction; all 9 edge pairs are always assembled here.
    // BuildFlatRowArrays applies the current filter spec
    // (m_active_pair_labels) when walking these pairs to populate
    // the flat arrays.
    struct LocalEdgePair
    {
        MortarBlock2D block;
        EdgeInfo3D    nonmortar_edge;
        EdgeInfo3D    mortar_edge;
    };
    std::vector<LocalEdgePair> m_local_edge_pairs;

    // Cached classifier-side gtdof_xyz lookup (matches
    // ConstraintBuilder3D's). In legacy construction these are already
    // parent-FES gtdofs. In projector construction these are submesh-FES
    // gtdofs and must be translated before indexing the runtime parent
    // vector.
    std::map<int, std::array<int, 3>> m_gtdof_lookup;

    // Cached parent-side component lookup keyed by parent x-component
    // true DOF. This intentionally stores MFEM's actual global true
    // DOF numbers instead of deriving y/z by arithmetic. With
    // Ordering::byNODES, sibling components are node-associated and
    // co-owned, but their global true DOF numbers are not the
    // component-block layout `x + scalar_true_size` used by byVDIM.
    // The MPI import/export path stores one slot per node keyed by
    // the x-component; this map supplies the real component gtdofs
    // for packing and transpose accumulation.
    std::map<int, std::array<int, 3>> m_parent_gtdof_lookup;

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
    //
    // Phase 5.9 / Batch A.3.d — this topology is NOT rebuilt by
    // Reset. Under reduced filter the topology over-imports (the
    // import buffer holds values for some off-rank gtdofs that are
    // never read by the filtered kernel), which is correct but
    // wasteful. The waste is bounded by the original topology size
    // and is negligible for typical filter specs (X-only PBC drops
    // ~2/3 of rows but only ~0% of imports since the import set
    // counts UNIQUE scalar gtdofs, and each scalar gtdof contributes
    // to all three component rows regardless of filter).
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

    // ---- Phase 5.9 — current filter spec ----
    //
    // m_active_pair_labels:   list of MORTAR-SIDE face labels of
    //                         active pairs. Defaults at construction
    //                         to all mortar labels from
    //                         classifier.FacePairs() ("top", "right",
    //                         "back" on a standard axis-aligned box).
    //                         Reset() replaces this.
    //
    // m_comp_mask:            per-component gate. Defaults to
    //                         {true, true, true}. Reset() replaces.
    //
    // m_n_comps_active:       count of true entries in m_comp_mask.
    //                         Equal to 3 for default. Used as the
    //                         per-row stride in m_row_lambda_off and
    //                         as the lambda-side row count multiplier
    //                         (Height() = m_n_active_rows * m_n_comps_active).
    //
    // m_local_c[c]:           position of c in the subsequence of
    //                         true entries in m_comp_mask, or -1 if
    //                         m_comp_mask[c] is false. The matvec
    //                         kernel captures these as 3 ints and
    //                         uses them to (a) skip filtered
    //                         components and (b) compute the
    //                         row-local lambda offset for active
    //                         components.
    std::vector<std::string> m_active_pair_labels;
    std::array<bool, 3> m_comp_mask = {{true, true, true}};
    int m_n_comps_active = kVDim;
    int m_local_c[3] = {0, 1, 2};

    // ---- Phase 4.3.B / Batch X — flat per-row arrays for GPU matvec --
    //
    // The CPU implementation walks per-pair blocks via std::map and
    // raw CSR pointers. That is not GPU-portable. The flat-array
    // form, built once at construction time (and re-built by Reset
    // under a new filter spec), mirrors what the matvec hot path
    // needs:
    //
    // m_n_active_rows:       count of constraint NODES this rank
    //                        owns and that pass the active-pair
    //                        filter. Each node contributes
    //                        m_n_comps_active rows to the lambda
    //                        vector, so Height() == m_n_active_rows
    //                        * m_n_comps_active.
    //
    // m_row_lambda_off[i]:   first lambda index this row writes
    //                        (= i * m_n_comps_active). Stored
    //                        explicitly to allow trivial change of
    //                        stride under filter without re-deriving.
    //
    // m_row_D[i]:            D_kk value for row i. Pre-baked diagonal
    //                        coefficient; same for all m_n_comps_active
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
    //                        Note this array remains size n_active*kVDim
    //                        regardless of comp_mask — the kernel
    //                        uses m_local_c[c] to decide which
    //                        components to read.
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
    //     lc = m_local_c[c];
    //     if (lc < 0) skip;                  // filtered (Phase 5.9)
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

    /**
     * @brief Shared implementation for both constructors.
     *
     * @details Builds edge blocks, filter defaults, import/export
     * topology, and flat row arrays after constructor-specific lifetime
     * and parent-FES members have been initialized.
     */
    void Initialize();

    /// Parent volume FES defining the operator runtime vector space.
    const mfem::ParFiniteElementSpace& ParentFes() const
    {
        return *m_parent_fes_raw;
    }

    /// Translate a classifier-side global true DOF to a parent-FES
    /// global true DOF. Negative sentinels are preserved.
    int ParentGtdofFromClassifierGtdof(int classifier_gtdof) const;

    /// Return parent-FES component true DOFs corresponding to a
    /// classifier-side x-component true DOF key.
    std::array<int, 3> ParentGtdofXyzFromClassifierX(
        int classifier_g_x) const;

    /// Return parent-FES component true DOFs for a parent x-component
    /// true DOF. Uses the cached MFEM/projector map; does not assume
    /// a global true-DOF arithmetic layout.
    std::array<int, 3> ParentGtdofXyzFromParentX(int parent_g_x) const;

    /// Return the owner rank of a classifier-side x-component true DOF
    /// after translation to the parent FES.
    int ParentOwnerRankFromClassifierX(int classifier_g_x) const;

    // Helper called at construction (and by Reset under Phase 5.9)
    // to populate all of the m_row_* and m_csr_* flat arrays from
    // the per-pair-block data (m_local_edge_pairs +
    // classifier.PairBlocks()), respecting the current filter
    // (m_active_pair_labels, m_comp_mask). Consolidates what was the
    // per-pair-block walk in Mult / MultTranspose's host-side code
    // into a one-shot setup pass, leaving the matvec free to run as
    // a single mfem::forall over m_n_active_rows.
    void BuildFlatRowArrays();
};

}  // namespace mortar_pbc
