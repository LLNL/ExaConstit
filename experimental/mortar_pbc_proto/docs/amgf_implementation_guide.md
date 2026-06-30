# Step-by-Step Implementation Guide: AMGF for ExaConstit Mortar PBC

A practical, action-oriented companion to `amgf_for_mortar_pbc_v2.md`. This guide distills §7 of that document into concrete code-modification tasks, ordered chronologically, with file paths, validation gates, and decision points.

The Phase numbering here matches §9 of the analysis document: **Phase 1 = Path A**, **Phase 2 = Path D**, **Phase 3 = Production benchmarks**, **Phase 4 = Path B** (conditional), **Phase 5 = Alternative AMG** (conditional). This guide covers Phases 1 and 2 in detail and notes the trigger conditions for Phases 4 and 5.

---

## 0. Context: the saddle/PBC structure is the bottleneck

Your bulk BoomerAMG on the K block is already well-tuned (HMIS coarsening, strength threshold 0.5, 4-level hierarchy with complexity 1.24, ~80× residual reduction in 6 iterations on representative non-PBC problems). The bulk AMG is doing its job. The 10k+ iteration stagnation under mortar PBC is **structural** — BoomerAMG, however well-configured, sees only matrix entries and has no mechanism to detect the constraint-induced near-null modes that emerge once the mortar coupling is wired in. That is precisely the failure mode AMGF was designed to fix.

For higher-order ($Q_2$) elements the situation is worse than at $Q_1$ for three compounding reasons:

| Element | Nodes per face | Nodes per edge | Typical $|\mathcal{I}_K| / n_u$ |
|---|---|---|---|
| $Q_1$ hex | 4 | 2 | 3–5% |
| $Q_2$ hex | 9 | 3 | 10–15% |
| $Q_3$ hex | 16 | 4 | 15–25% |

1. **The constraint matrix $C$ has more rows and denser support.** Each pair of opposite $Q_2$ periodicity faces produces ~9 multipliers per surface element vs. ~4 for $Q_1$, and the Wohlmuth dual basis with corner modifications interacts with edge and face DOFs in addition to corner DOFs.
2. **The Schur block conditioning gets correspondingly worse**, so the existing diagonal-lumped Schur preconditioner $\text{diag}(\hat{S})^{-1}$ becomes a much worse approximation than at $Q_1$.
3. **The constraint-induced near-null subspace is larger** because the boundary-adjacent DOF count $|\mathcal{I}_K|$ scales up. That subspace is still small relative to $n_u$ (typically 10–15% for $Q_2$, vs. 3–5% for $Q_1$) but the *absolute* count and the *conditioning* on that subspace are both worse.

Given (1) and (2), Path A alone — which improves only the K block — is structurally likely to be a partial fix at $Q_2$. The Schur block lumping remains broken and that's the dominant bottleneck. **Phase 2 (Path D) is the actual answer**; Phase 1 (Path A) is still worth implementing first because it puts the AMGF infrastructure in place and Path D is a small increment on top of it, but plan the calendar with Phase 2 baked in.

---

## 1. Baseline measurement (before any code changes)

You already know AMG-on-K is fine and the PBC saddle is broken; the "diagnostic" step is therefore not to *verify* the diagnosis but to *capture baseline numbers* you'll compare against when each phase comes online.

Pick a representative $Q_2$ PBC problem that reliably reproduces the slow convergence (a single-grain or small polycrystal cube is fine; you do not need the full production-scale RVE for this) and record, with the current code path:

- Linear-solver iteration count to reach $\|r\| \leq 10^{-5}$ (or whatever it actually reaches before the iteration cap).
- Wall-clock time for one full Newton step.
- Per-iteration Krylov cost (one V-cycle + one Schur-block apply); use Caliper or hand timing.
- Newton outer-iteration count over the load history.
- Throughout the Newton iteration, print $\|r_K\|$ and $\|r_\lambda\|$ at each step. Note the ratio. (For $Q_2$ PBC, I'd guess $\|r_K\|/\|r_\lambda\| > 10^3$ throughout — confirming the residual imbalance §6 of the analysis doc discusses.)

These numbers serve two purposes: they're the validation gate for Phase 1 (Path A should reduce the iteration count), and they reveal whether the residual imbalance is bad enough that Path D's automatic rebalancing is a major win on top of the conditioning improvement (it almost certainly is for $Q_2$ PBC, but having the numbers makes the case quantitative).

Save these baseline numbers somewhere durable. You'll reference them five separate times during the implementation.

---

## 1.5. Pre-implementation verification (do these BEFORE any coding)

Three quick checks that prevent wasted work. Each takes minutes; total time under one hour.

### Step 0 — Verify the actual MFEM 4.9 `AMGFSolver` API

**Why this step exists:** the analysis doc's first revision paraphrased the AMGF API from workshop slides; the names did not match the actual MFEM 4.9 source on first inspection. The verified API below is from the MFEM 4.9 Doxygen reference (`docs.mfem.org/html/classmfem_1_1AMGFSolver.html`) and from `linalg/filteredsolver.hpp` on the MFEM master branch. Confirm against your local MFEM checkout (`rcarson3/mfem` branch `exaconstit-latest`, or wherever your ExaConstit build pulls MFEM from) before relying on these names.

**The verified API** (sufficient to drive the AMGF preconditioner):

```cpp
namespace mfem {

class FilteredSolver : public Solver {
public:
    FilteredSolver();
    void SetFilteredSubspaceTransferOperator(const Operator& P);  // base
    void SetFilteredSubspaceSolver(Solver& S);
    void Mult(const Vector& x, Vector& y) const override;
};

class AMGFSolver : public FilteredSolver {
public:
    AMGFSolver();                            // default ctor; constructs internal HypreBoomerAMG
    void SetOperator(const Operator& A) override;
    void SetSolver(Solver& B) override;
    void SetFilteredSubspaceTransferOperator(const HypreParMatrix& Pop); // HypreParMatrix-typed override
    HypreBoomerAMG& GetAMG();
    const HypreBoomerAMG& GetAMG() const;
};

}  // namespace mfem
```

**Notes for a coding agent or human:**
- The ctor takes **no `MPI_Comm`** — internal AMG handles MPI.
- Access the AMG via **`GetAMG()`**, not `.AMG()`.
- The transfer operator is set via the **`HypreParMatrix`-typed** override on `AMGFSolver` (the generic base-class version exists but the typed version is the one to use).
- Configure AMG settings **after** construction via `GetAMG().Set...()` calls.
- The subspace solver `S` must be `mfem::Solver`-derived. Adapter for Ginkgo: see Step 1.1.
- Header is `mfem/linalg/filteredsolver.hpp`, included transitively via `mfem.hpp` or `mfem/linalg/linalg.hpp`.

**Validation gate:** a one-file test program that constructs an `AMGFSolver`, calls `GetAMG().SetSystemsOptions(3)`, sets a dummy P and dummy subspace solver, and links cleanly. Five minutes; mostly verifies the build picked up MFEM 4.9.

---

### Step 0.5 — Verify FULL assembly is required and error out cleanly otherwise

**The architectural constraint (§7.4 of the analysis doc):**
- AMGF requires a fully assembled `HypreParMatrix` for both BoomerAMG and the Ginkgo boundary submatrix factorization.
- ExaConstit currently prefers Element Assembly (EA) on GPU because EA is faster than PA and gives access to per-element block values.
- The existing `src/options/option_solvers.cpp::validate()` enforces three rules: (a) GPU + FULL is forbidden, (b) GPU forces JACOBI prec, (c) EA/PA forces JACOBI prec.

These rules contradict what AMGF needs. The new AMGF option **cannot silently fall through them** — that would silently downgrade to Jacobi and the user would never see AMGF. Instead, the validation must add a new clause:

```cpp
// In src/options/option_solvers.cpp::SolverOptions::validate(),
// AFTER the existing GPU + FULL check and BEFORE the GPU/EA/PA prec-correction:

if (linear_solver.preconditioner == PreconditionerType::AMGF ||
    linear_solver.preconditioner == PreconditionerType::AMGF_AUG_LAGRANGIAN)
{
    if (assembly != AssemblyType::FULL) {
        WARNING_0_OPT(
            "Error: AMGF preconditioner requires FULL assembly. "
            "Element Assembly (EA) is preferred on GPU for performance but "
            "the AMGF preconditioner branch requires a fully assembled "
            "HypreParMatrix for BoomerAMG and the Ginkgo subspace solver. "
            "Either set `assembly = \"FULL\"` and switch rtmodel to "
            "CPU or OPENMP, or disable AMGF.");
        return false;
    }
    if (rtmodel == RTModel::GPU) {
        WARNING_0_OPT(
            "Error: AMGF requires FULL assembly, which is not supported on "
            "GPU runtime. Switch rtmodel to CPU or OPENMP.");
        return false;
    }
}
```

Add the corresponding `PreconditionerType::AMGF` and `PreconditionerType::AMGF_AUG_LAGRANGIAN` enum values, the TOML keyword parsing, and the documentation in the preconditioner help text. Do not auto-correct silently; the user must explicitly choose between (a) FULL assembly + CPU/OpenMP + AMGF, or (b) EA/PA + GPU + Jacobi (the existing path, unchanged).

**Future Option 2** (hybrid EA-for-residual + FA-for-preconditioner on GPU) is documented in §10.3 of the analysis doc as future work. It requires MFEM machinery that doesn't currently exist (`HypreParMatrix` assembly from `ParNonlinearForm::GetGradient` element matrices when the form is configured with EA). Skip for the initial implementation.

**Validation gate:** with `assembly = "EA"` and `preconditioner = "AMGF"` in the TOML, the run aborts at validation time with the message above. With `assembly = "FULL"` and `rtmodel = "CPU"`, the run proceeds.

---

### Step 0.7 — Run the backward-compatibility regression test before any code changes

Run the existing ExaConstit test suite end-to-end **before** any of the changes below. Record the pass/fail status of each test as a baseline. The new AMGF code paths are gated behind the `preconditioner = "AMGF"` flag and should preserve existing behavior bit-for-bit when the flag is off, but every code change carries regression risk. Capturing the pre-change state makes regressions trivial to spot.

After each Phase 1 step completes, re-run the same suite with `preconditioner` unset (defaulting to JACOBI). All tests should continue to pass.

---

## 2. Phase 1 — Path A: Step-by-Step Implementation

Goal: wrap the existing BoomerAMG K-block preconditioner with `mfem::AMGFSolver`, using a subspace basis matrix $P$ that restricts to boundary-adjacent displacement DOFs. Estimated effort: 4–6 working days.

### Step 1.1 — Build-system: enable Ginkgo in MFEM

**Files:** `CMakeLists.txt` (top-level), MFEM build configuration, Ginkgo as an external dependency.

**Action:**
- Rebuild MFEM with `MFEM_USE_GINKGO=YES`. MFEM's Ginkgo wrapper lives in `mfem/linalg/ginkgo.hpp` and provides `mfem::Ginkgo::GinkgoExecutor` plus wrapper classes for Ginkgo's solvers and preconditioners.
- Ginkgo itself should be built with at minimum `GINKGO_BUILD_OMP=ON`. For the initial AMGF implementation (Step 0.5: FA required → CPU/OpenMP rtmodel), the **OMP executor is the one actually used**; `GINKGO_BUILD_CUDA=ON` / `GINKGO_BUILD_HIP=ON` are needed only for the future hybrid mode (§7.4 Option 2 of the analysis doc) and can be omitted from the first build.
- Ginkgo's sparse direct factorizations live in `gko::experimental::factorization` — specifically `Cholesky` (for the symmetric K case) and `Lu` (for non-associated-flow non-symmetric K). Both have OMP executors. Verify the build by running the contact miniapp or any Ginkgo example shipped with MFEM.

**Why Ginkgo instead of MUMPS:**
- BSD-3 license; no CeCILL (MUMPS) or commercial MKL (CPardiso) complications.
- Already in MFEM as a first-class wrapper, with the executor model letting you pick CUDA / HIP / OneAPI / OpenMP at runtime — important for the future hybrid mode even if the initial implementation uses OMP only.
- Active development with consistent performance work; well-supported on the ECP HPC stack ExaConstit targets.
- GPU-resident factorization is available when the future hybrid mode (§7.4 Option 2) becomes possible; choosing Ginkgo now avoids a re-port later.

**Build-system caveat:** Ginkgo and MFEM both have CMake-flavor dependencies; the integration is well-supported but the first build on a new HPC environment can take a few hours to chase down (CUDA arch flags, OpenMP linker quirks, etc.). Start it in parallel with the other steps so it isn't on the critical path.

**Validation gate:** A trivial test program that creates a `gko::experimental::factorization::Cholesky` factorization of a small SPD matrix on an OMP executor and solves a system completes successfully. The MFEM example `ex1p` or `ex9p` runs with a Ginkgo-based preconditioner unchanged in behavior.

**Note on the subspace solver interface.** AMGFSolver expects an `mfem::Solver*` for the subspace solve. MFEM's Ginkgo wrapper has high-level support for iterative solvers and preconditioners but the *direct* factorization usage may require a thin adapter class — perhaps 50 lines — that holds a `gko::experimental::factorization::Cholesky` (or `Lu`) and exposes `mfem::Solver::SetOperator` and `Mult`. The skeleton:

```cpp
namespace mortar_pbc {

class GinkgoDirectSubspaceSolver : public mfem::Solver {
public:
    GinkgoDirectSubspaceSolver(std::shared_ptr<gko::Executor> exec, bool symmetric);

    void SetOperator(const mfem::Operator& op) override {
        // 1. Cast op to mfem::HypreParMatrix (AMGF subspace operator P^T A P
        //    is always a HypreParMatrix; verify via dynamic_cast).
        // 2. Convert the HypreParMatrix's local diagonal block to
        //    gko::matrix::Csr<double, int> on the chosen executor.
        // 3. Build factorization: Cholesky if symmetric_, else Lu.
        //    Cache the factorization for reuse across Krylov iterations.
    }

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
        // Apply forward/back-substitution via the factored Ginkgo operator.
        // The input/output are mfem::Vector; mirror to/from
        // gko::matrix::Dense<double> on the executor.
    }

private:
    std::shared_ptr<gko::Executor> exec_;
    bool symmetric_;
    std::shared_ptr<gko::LinOp> factored_op_;  // The Cholesky or Lu object
};

}  // namespace mortar_pbc
```

This adapter lives in `src/mortar_pbc/ginkgo_direct_subspace_solver.hpp/cpp`, is instantiated once in `system_driver.cpp` (Step 1.6), and is passed to `AMGFSolver::SetFilteredSubspaceSolver`. Cost to write: half a day, mostly format conversion boilerplate.

**Executor selection helper.** The TOML option `amgf_subspace_executor` (from Step 1.5) drives the executor choice:

```cpp
std::shared_ptr<gko::Executor> MakeGinkgoExecutor(const std::string& spec) {
    if (spec == "omp" || spec == "auto") {
        return gko::OmpExecutor::create();
    }
    if (spec == "cuda") {
        // For future hybrid mode (§7.4 Option 2); not used in initial impl.
        return gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    }
    if (spec == "hip") {
        return gko::HipExecutor::create(0, gko::OmpExecutor::create());
    }
    MFEM_ABORT("Unknown Ginkgo executor: " << spec);
    return nullptr;
}
```

---

### Step 1.2 — Expose boundary-DOF index set from `MortarConstraintOperator`

**File:** `src/mortar_pbc/mortar_constraint_operator.hpp` and `.cpp`.

**The existing API** (verified in the current source, namespace `mortar_pbc`):
- `void Mult(const mfem::Vector& u, mfem::Vector& lambda) const override` — applies $C$.
- `void MultTranspose(const mfem::Vector& lambda, mfem::Vector& u_residual) const override` — applies $C^T$; host-side walk plus `MPI_Alltoallv` back-scatter (not yet GPU-parallelized).
- `mfem::Vector ComputeInvDiagSchur(const mfem::Solver& K_jacobi_prec) const` — returns $\text{diag}(C \cdot \text{diag}(K)^{-1} \cdot C^T)^{-1}$ via `WeightedRowSqSum` internally (no explicit $C C^T$ product).
- `void Reset(const std::vector<int>& active_pair_labels, const std::array<bool,3>& comp_mask)` — Phase 5.9 sub-XYZ rebuild.
- `int Height() const` — local lambda dimension.
- `int Width() const` — local FES TrueVSize (the u dimension).

**What we are adding:** a new public accessor for the index set $\mathcal{I}_K$ — displacement DOFs that appear as nonzero columns in $C$. This is the set used to build the AMGF subspace prolongation $P$ (see §3.3 of the analysis doc).

```cpp
class MortarConstraintOperator {
public:
    // ... existing methods (unchanged) ...

    /**
     * @brief Returns the global TDOF indices that participate in any
     *        active mortar periodicity constraint.
     *
     * @details This is the index set I_K used to build the AMGF subspace
     * prolongation P. See §3.3 and §7.2 of the AMGF analysis document.
     *
     * The returned vector is sorted ascending, holds globally unique
     * `HYPRE_BigInt` indices (global numbering, not per-rank local), and
     * respects the current Phase 5.9 filter spec — only DOFs touched by
     * active pair-labels and active components contribute.
     *
     * @par Cost
     * O(nnz(C)) on first call; cached thereafter. The cache is invalidated
     * by `Reset` (because the filter spec changes which rows are active).
     *
     * @par MPI
     * Local — no collective. Each rank returns its own local view of the
     * global index set, gathered from its owned constraint rows plus its
     * already-imported off-rank mortar gtdofs.
     */
    const std::vector<HYPRE_BigInt>& GetConstraintCoupledDofIndices() const;

private:
    mutable std::vector<HYPRE_BigInt> m_constraint_coupled_dofs;
    mutable bool m_constraint_coupled_dofs_built = false;
};
```

The implementation iterates the operator's per-pair block structure (already walked by `Mult` and `MultTranspose`) and collects column indices encountered as nonzeros. For the dual-basis mortar with Wohlmuth corner modifications, this naturally includes both nonmortar-side and mortar-side DOFs across the active pair labels.

**Sub-XYZ filter handling:** when `Reset(active_pair_labels, comp_mask)` is called (Phase 5.9), invalidate the cache by setting `m_constraint_coupled_dofs_built = false` so the next `GetConstraintCoupledDofIndices()` call rebuilds against the new filter spec.

**Header includes:** the .hpp file already includes `"mfem.hpp"` and `<vector>`; no additional includes needed.

**Caliper:** wrap the first-call build with `CALI_CXX_MARK_SCOPE("mortar_pbc::mortar_constraint_operator::get_constraint_coupled_dofs")`. The cached return path is too cheap to bother instrumenting.

**Doxygen:** include the §-reference to the AMGF document, the cost characterization, the local-vs-global semantics, and the Reset interaction.

**Validation gate (unit test):** in `test/mortar_pbc/` following the existing `test_mortar_saddle_preconditioner.cpp` pattern (custom `AssertOrDie`-style framework, gtest-free):
- Build a 2×2×2 hex mesh with $Q_2$ elements and full XYZ periodicity.
- Confirm `GetConstraintCoupledDofIndices().size()` matches a hand-enumerated count of boundary nodes × 3 components.
- Call `Reset(...)` with one axis dropped; confirm the returned size shrinks to match the new active-face count.
- Confirm sorted-ascending and uniqueness.

---

### Step 1.3 — Add the `BuildBooleanRestrictionProlongation` utility

**File:** new file `src/mortar_pbc/amgf_utils.hpp` and `amgf_utils.cpp`.

**Action:** Implement a free function that builds a parallel `HypreParMatrix` $P \in \mathbb{R}^{n \times n_c}$ with one Boolean nonzero per column, given a global index set $\mathcal{I}$:

```cpp
namespace exaconstit::amgf {

/**
 * @brief Build a Boolean prolongation matrix P with one nonzero per column.
 *
 * @details P has shape (n_global_rows, n_global_cols) where
 * n_global_cols = idx_global.size(). P[idx_global[j], j] = 1.0 for
 * j = 0, ..., n_global_cols - 1.
 *
 * P's row partitioning matches K's row partitioning (passed as row_starts)
 * so that P^T K P assembles correctly via HypreParMatrix::Mult.
 * Columns are partitioned compactly: each MPI rank owns a contiguous
 * range of columns, with the partition derived from how many entries of
 * idx_global fall in each rank's row range.
 *
 * See §3.3 and §7.2 of the AMGF analysis document for the mathematical
 * role of P in the AMGF preconditioner.
 *
 * @par Empty-partition edge case
 * Some MPI ranks may hold no boundary DOFs (entirely interior
 * subdomains). For those ranks the local row count of P is the same as
 * for K, but the local column count is zero. The HypreParMatrix
 * constructor handles this correctly; no special-casing is needed beyond
 * making sure the column-partition computation handles n_local_cols = 0
 * on the affected ranks. Verify on a deliberately ill-balanced run (one
 * rank owning a corner subdomain with no boundary faces).
 *
 * @param n_global_rows  Total displacement DOF count n_u.
 * @param idx_global     Sorted unique global indices in [0, n_global_rows),
 *                       from MortarConstraintOperator::GetConstraintCoupledDofIndices().
 * @param k_row_starts   The row partition of the K matrix
 *                       (HypreParMatrix::RowPart() / GetRowStarts()).
 * @param comm           MPI communicator.
 *
 * @return Heap-allocated HypreParMatrix*; caller takes ownership
 *         (wrap in unique_ptr).
 */
mfem::HypreParMatrix* BuildBooleanRestrictionProlongation(
    HYPRE_BigInt n_global_rows,
    const std::vector<HYPRE_BigInt>& idx_global,
    const HYPRE_BigInt* k_row_starts,
    MPI_Comm comm);

}  // namespace exaconstit::amgf
```

**Header includes** (in the .hpp file): `#include "mfem.hpp"`, `#include <vector>`, `#include <mpi.h>` (if not already pulled in).

**Implementation outline:**
1. Determine the local row range of K on this rank from `k_row_starts`.
2. Walk `idx_global` to identify which entries are *locally owned* (fall in the local row range).
3. Compute global column partitioning: an `MPI_Allgather` of local nonzero counts gives the per-rank column block sizes; cumulative sum gives the column row-starts.
4. Build a `mfem::SparseMatrix` for the local diagonal block (square in the local-rows × local-cols sense) and an off-diagonal block (local-rows × off-rank-cols, empty in our Boolean case since each column has exactly one row).
5. Hand to the `HypreParMatrix` four-argument constructor (`MPI_Comm, global_num_rows, global_num_cols, row_starts, col_starts, diag, offd, cmap`).

Allocate the underlying CSR on the host. P is touched once per Newton step (in `SetOperator`), so host residency does not hurt; the BoomerAMG and Ginkgo solvers will pull P to their preferred memory if needed.

**Caliper:** `CALI_CXX_MARK_SCOPE("exaconstit::amgf::build_boolean_restriction_prolongation")`.

**Validation gate (unit test):** in `test/amgf/`:
- Construct a 10-DOF problem with $\mathcal{I} = \{0, 3, 7\}$, build P by hand and via the utility, compare.
- Build a tiny $A$ (a 10×10 SPD `HypreParMatrix`), form $P^T A P$ via `mfem::RAP` or `HypreParMatrix::Mult`, compare against the expected 3×3 principal submatrix.
- Run with 1, 2, and 4 MPI ranks; confirm consistent behavior including the case where one rank ends up with zero local columns of P.

---

### Step 1.4 — New preconditioner class `MortarSaddlePreconditionerAMGF`

**File:** `src/mortar_pbc/mortar_saddle_preconditioner_amgf.hpp` and `.cpp` (new file; do not modify the existing `mortar_pbc::MortarSaddlePreconditioner` so the old code path remains available as fallback).

**Existing class as a reference** (verified): `mortar_pbc::MortarSaddlePreconditioner` (in `src/mortar_pbc/mortar_saddle_preconditioner.hpp`) takes the ctor

```cpp
MortarSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> K_block_prec,    // the K-block preconditioner (any Solver)
    std::shared_ptr<mfem::Solver> K_jacobi_prec,   // Jacobi-style probe for diag(K)^{-1}
    const MortarConstraintOperator& C_op);
```

`SetOperator` extracts $K$ from the saddle `BlockOperator`, calls `K_block_prec->SetOperator(K)` and `K_jacobi_prec->SetOperator(K)`, then builds the inverse-Schur-diagonal via `C_op.ComputeInvDiagSchur(*K_jacobi_prec)`. The result is wrapped in a `DiagonalScaler` and the two blocks are wired into an `mfem::BlockDiagonalPreconditioner`.

**The new class.** AMGF *is* the K-block preconditioner (it wraps the AMG), so there is no separate `K_block_prec` argument; the class owns the `mfem::AMGFSolver` internally. The `K_jacobi_prec` is still needed for Path A's Schur-diagonal probe (the existing `ComputeInvDiagSchur` contract requires a Jacobi-style probe target). Path D bypasses both the probe and the Schur scaling.

```cpp
namespace mortar_pbc {

class MortarSaddlePreconditionerAMGF : public mfem::Solver
{
public:
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        std::unique_ptr<mfem::HypreParMatrix> P,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override = -1.0);

    void SetOperator(const mfem::Operator& op) override;
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Phase 5.11.G: needed by SaddlePathDAugmenter to read the current gamma.
    const double& gamma() const { return gamma_; }

private:
    // Shared infrastructure
    std::shared_ptr<mfem::Solver> K_jacobi_prec_;
    const MortarConstraintOperator& C_op_;
    std::unique_ptr<mfem::HypreParMatrix> P_;
    std::shared_ptr<mfem::Solver> subspace_solver_;
    std::shared_ptr<mfem::AMGFSolver> amgf_;
    mfem::HypreParMatrix* K_ = nullptr;       // not owned; from BlockOperator

    // Path A state
    mfem::Vector schur_diag_inv_;

    // Path D state
    bool use_path_d_;
    double gamma_override_;
    mutable double gamma_ = 0.0;
    std::unique_ptr<mfem::HypreParMatrix> K_gamma_;

    // Dimensions
    HYPRE_BigInt n_u_global_ = 0;
    int n_lambda_local_ = 0;
};

}  // namespace mortar_pbc
```

**Constructor body** (in `.cpp`):

```cpp
MortarSaddlePreconditionerAMGF::MortarSaddlePreconditionerAMGF(
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    const MortarConstraintOperator& C_op,
    std::unique_ptr<mfem::HypreParMatrix> P,
    std::shared_ptr<mfem::Solver> subspace_solver,
    bool use_path_d,
    double gamma_override)
    : mfem::Solver(0, 0),
      K_jacobi_prec_(std::move(K_jacobi_prec)),
      C_op_(C_op),
      P_(std::move(P)),
      subspace_solver_(std::move(subspace_solver)),
      use_path_d_(use_path_d),
      gamma_override_(gamma_override)
{
    MFEM_VERIFY(K_jacobi_prec_, "AMGF: K_jacobi_prec must be non-null");
    MFEM_VERIFY(P_,             "AMGF: P must be non-null");
    MFEM_VERIFY(subspace_solver_, "AMGF: subspace_solver must be non-null");

    amgf_ = std::make_shared<mfem::AMGFSolver>();

    // Copy ExaConstit's verified BoomerAMG configuration verbatim.
    // Do NOT retune here; the whole point of Path A is to preserve the
    // proven bulk-AMG configuration while wrapping it.
    auto& amg = amgf_->GetAMG();
    amg.SetSystemsOptions(3);     // 3-component displacement field
    amg.SetRelaxType(8);          // l1-symmetric Gauss-Seidel
    amg.SetStrongThreshold(0.5);  // ExaConstit production value
    amg.SetPrintLevel(0);

    // Wire P and the subspace solver into the AMGF. These are not refreshed
    // per Newton step; only the operator A changes (in SetOperator below).
    amgf_->SetFilteredSubspaceTransferOperator(*P_);
    amgf_->SetFilteredSubspaceSolver(*subspace_solver_);

    n_u_global_     = P_->GetGlobalNumRows();
    n_lambda_local_ = C_op_.Height();

    // Sanity check: C^T C must not be entirely zero (catches the
    // pathological case from §10.1 of the analysis doc).
    if (use_path_d_) {
        // Cheap check: the C_op cache will build C^T C on first call;
        // we can defer to SetOperator. Just verify the constraint op
        // has nonzero Height.
        MFEM_VERIFY(n_lambda_local_ > 0 || mpi_size_check(),
                    "AMGF Path D: zero local lambda count on all ranks");
    }
}
```

**Key implementation points** (specific to ExaConstit conventions):
- Use `std::shared_ptr` for the AMG and Ginkgo subspace solver, never raw pointers (matches the `oper_mech` / `prec_mech` shared-pointer discipline ExaConstit uses everywhere for lifetime safety).
- Copy the bulk-AMG configuration from the existing `MortarSaddlePreconditioner` construction site in `src/system_driver.cpp` (where `mfem::HypreBoomerAMG` is currently constructed). Use whatever ExaConstit actually deploys there; do not let the implementation step become an opportunity to retune AMG.
- `SetOperator` extracts the $(1,1)$ block from `mfem::BlockOperator` and routes it through `AMGFSolver::SetOperator` (Path A) or wraps it in $K_\gamma$ first (Path D — see Step 2.4).
- `Mult` applies the block-diagonal preconditioner: AMGF on the displacement block, diagonal-lumped Schur (Path A) or $\gamma I$ (Path D) on the multiplier block.
- Match the existing `prec_mech->SetOperator(J)` + `prec_mech->Mult(r, c)` pattern that ExaConstit's outer solver (`ExaNewtonSolver`, `ExaNewtonLSSolver`, `ExaTrustRegionSolver`) expects, so this is a drop-in replacement.

**The `oper_mech` / `prec_mech` convention:** ExaConstit's solver code holds the gradient operator as `oper_mech` (an `std::shared_ptr<mfem::Operator>`, owned at `SystemDriver` level) and the preconditioner as `prec_mech` (`std::shared_ptr<mfem::Solver>`). Linear solves go `prec_mech->SetOperator(J)` followed by Krylov iteration with `prec_mech` set on the iterative solver via `SetPreconditioner`. Never use raw pointers here — lifetime safety in the trust-region solver depends on the shared_ptr discipline.

**Doxygen:** full header documenting the mathematical role (Path A from §2.2 of the analysis doc), the convergence guarantee (§3.10), the ctor parameter requirements, and the Path D toggle.

**Caliper:** `CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::set_operator")` and `...::mult` (the latter runs once per Krylov iteration so will reveal the per-iter cost of the subspace solve).

**Validation gate:** with the new class compiled and linked, replace the existing `mortar_pbc::MortarSaddlePreconditioner` in a small single-grain test problem (toggling via the new TOML option from Step 1.5) and verify (a) the linear solver still converges to the same tolerance, (b) the residual reduction is monotonic, (c) per-iteration cost is reasonable (not orders of magnitude slower than baseline — though iteration counts should be substantially lower).

---

### Step 1.5 — Plumb new options through the TOML parser

**File:** `src/options/option_solvers.hpp/cpp` (where `SolverOptions`, `LinearSolverOptions`, and the `PreconditionerType` enum live) plus the TOML parsing site.

**Action:** Add two new values to `PreconditionerType`:

```cpp
enum class PreconditionerType {
    NOTYPE = 0,
    JACOBI,
    AMG,
    ILU,
    L1GS,
    CHEBYSHEV,
    AMGF,                  // Path A: AMGF on K
    AMGF_AUG_LAGRANGIAN    // Path D: AMGF on K_gamma + augmented RHS
};
```

Wire up the corresponding TOML string parsing (`"amgf"` → `AMGF`, `"amgf_aug_lagrangian"` → `AMGF_AUG_LAGRANGIAN`) wherever the existing `"amg"` / `"ilu"` / etc. cases are handled.

Also add a configuration struct field for $\gamma$ (Path D's augmentation parameter) and the Ginkgo subspace executor selection:

```cpp
struct LinearSolverOptions {
    // ... existing fields ...

    /// Augmentation parameter gamma for Path D (AMGF_AUG_LAGRANGIAN).
    /// If <= 0, use the natural scaling gamma = tr(K) / tr(C^T C) * (n_lambda / n_u).
    /// See §5.10 of the AMGF analysis document for sensitivity analysis.
    double amgf_gamma = -1.0;

    /// Ginkgo executor for the AMGF subspace solver.
    /// Values: "cuda", "hip", "omp" (host), "auto" (matches rtmodel when possible).
    /// Note: AMGF currently requires FULL assembly which forbids GPU rtmodel
    /// (per the option validation), so in practice this resolves to OpenMP
    /// for the initial implementation. Reserved for the future hybrid mode.
    std::string amgf_subspace_executor = "omp";
};
```

Default values are conservative (AMGF off) so existing runs continue to use the JACOBI / AMG / ILU / L1GS / CHEBYSHEV paths unchanged.

The Step 0.5 validation rule already guards against `AMGF + EA/PA` and `AMGF + GPU`. Make sure that rule fires *before* the existing GPU-forces-JACOBI auto-correct, so the user gets the explicit AMGF error message rather than a silent downgrade to JACOBI.

**Validation gate:** running with `preconditioner = "amgf"` in the TOML constructs the new preconditioner; running without it preserves the existing behavior bit-for-bit. The Step 0.7 regression suite still passes.

---

### Step 1.6 — Wire it up in `system_driver.cpp`

**File:** `src/system_driver.cpp`.

**Action:** In the preconditioner-construction block (right after the existing `MortarSaddlePreconditioner` is currently instantiated for the mortar PBC path), add a branch on the new `PreconditionerType::AMGF` / `AMGF_AUG_LAGRANGIAN` cases. Existing `JACOBI`/`AMG`/`ILU`/`L1GS`/`CHEBYSHEV` paths are unchanged.

```cpp
// (Inside the existing if-block that constructs the saddle preconditioner
// when mortar PBC is active.)
std::shared_ptr<mfem::Solver> saddle_prec;

const auto& linear_solvers = options.solvers.linear_solver;
const bool path_d_active =
    (linear_solvers.preconditioner == PreconditionerType::AMGF_AUG_LAGRANGIAN);
const bool path_a_active =
    (linear_solvers.preconditioner == PreconditionerType::AMGF) || path_d_active;

if (path_a_active) {
    // Build P. The MortarConstraintOperator already owns the index set;
    // we just need K's row-partition info, which we get from J_prec or
    // wherever the assembled K-as-HypreParMatrix lives at this point in
    // the driver. The K matrix is constructed earlier in this function
    // when assembly == FULL (the option validation guarantees that here,
    // per Step 0.5).
    const auto& boundary_idx = mortar_op->GetConstraintCoupledDofIndices();
    auto P = std::unique_ptr<mfem::HypreParMatrix>(
        exaconstit::amgf::BuildBooleanRestrictionProlongation(
            K_global_size,            // HYPRE_BigInt
            boundary_idx,
            K_matrix->GetRowStarts(), // HypreParMatrix::GetRowStarts
            MPI_COMM_WORLD));

    // Build the Ginkgo direct subspace solver (Cholesky if K is symmetric,
    // Lu otherwise — see Step 1.1).
    auto gko_exec = MakeGinkgoExecutor(linear_solvers.amgf_subspace_executor);
    auto subspace_solver = std::make_shared<GinkgoDirectSubspaceSolver>(
        gko_exec, /*symmetric=*/!options.use_non_associated_flow);

    // The K_jacobi_prec probe is still needed for Path A's Schur diagonal.
    // It's the same Jacobi-style preconditioner used in the existing
    // MortarSaddlePreconditioner path — typically an mfem::HypreDiagScale
    // or our own DiagonalScaler. Build it just as the existing code does.
    auto K_jacobi_prec = std::make_shared<mfem::HypreDiagScale>();

    saddle_prec = std::make_shared<mortar_pbc::MortarSaddlePreconditionerAMGF>(
        K_jacobi_prec, *mortar_op, std::move(P), subspace_solver,
        /*use_path_d=*/path_d_active, linear_solvers.amgf_gamma);
} else {
    // Existing code path: plain block-diagonal Schur preconditioner.
    // Verified ctor: (K_block_prec, K_jacobi_prec, C_op).
    saddle_prec = std::make_shared<mortar_pbc::MortarSaddlePreconditioner>(
        K_block_prec,    // existing AMG/ILU/Jacobi/etc. choice from J_prec
        K_jacobi_prec,   // existing Jacobi-style probe target
        *mortar_op);
}

prec_mech = saddle_prec;

// Phase 5.11.G: Path D additionally hooks a SaddlePathDAugmenter scaler
// into the trust-region solver. See Step 2.6.
if (path_d_active) {
    auto& amgf_prec = dynamic_cast<mortar_pbc::MortarSaddlePreconditionerAMGF&>(
        *saddle_prec);
    auto path_d_scaler = std::make_shared<mortar_pbc::SaddlePathDAugmenter>(
        *mortar_op, amgf_prec.gamma());
    if (auto* tr_solver = dynamic_cast<ExaTrustRegionSolver*>(newton_solver.get())) {
        tr_solver->SetScaler(path_d_scaler, saddle_block_offsets);
    } else {
        // ExaNewtonSolver / ExaNewtonLSSolver don't yet support the scaler
        // hook. Either error out at option-validation time, or extend those
        // solvers with an equivalent slot. For initial implementation we
        // require TRDOG.
        MFEM_ABORT("AMGF_AUG_LAGRANGIAN currently requires nl_solver = TRDOG");
    }
}
```

The rest of the Newton/Krylov flow is unchanged — `prec_mech->SetOperator(J)` and `prec_mech->Mult(r, c)` work identically on either implementation.

**Header includes** (in `system_driver.cpp`, near the top):

```cpp
#include "mortar_pbc/mortar_saddle_preconditioner.hpp"          // existing
#include "mortar_pbc/mortar_saddle_preconditioner_amgf.hpp"     // new
#include "mortar_pbc/amgf_utils.hpp"                            // new
#include "mortar_pbc/ginkgo_direct_subspace_solver.hpp"         // new
#include "mortar_pbc/saddle_path_d_augmenter.hpp"               // new (Step 2.6)
```

**Validation gate:** smoke test with `preconditioner = "jacobi"` and `preconditioner = "amgf"` on a single test problem; verify identical final solution (to within Newton tolerance), just different iteration counts. Backward-compat regression suite from Step 0.7 still passes.

---

### Step 1.6.5 — CMake updates for the new files

**File:** `CMakeLists.txt` (top-level) and any subdirectory `CMakeLists.txt` files that enumerate source files for the ExaConstit library target.

**Action:** Add the new files to the ExaConstit library's source list:

```cmake
# In the source-file list for the main ExaConstit library target (look for
# where mortar_pbc/mortar_constraint_operator.cpp is already listed):
set(EXACONSTIT_SOURCES
    # ... existing entries ...
    src/mortar_pbc/amgf_utils.cpp
    src/mortar_pbc/ginkgo_direct_subspace_solver.cpp
    src/mortar_pbc/mortar_saddle_preconditioner_amgf.cpp
    src/mortar_pbc/saddle_path_d_augmenter.cpp
)
```

Plus the Ginkgo find_package + linking:

```cmake
# Ginkgo discovery. Mirrors the existing find_package patterns for MFEM,
# Caliper, etc. ExaConstit's Ginkgo requirement is transitive via MFEM
# (MFEM_USE_GINKGO=YES), so this find_package is mostly for the header
# paths used by the GinkgoDirectSubspaceSolver adapter.
find_package(Ginkgo 1.6 REQUIRED)
target_link_libraries(exaconstit PRIVATE Ginkgo::ginkgo)
```

Add new test executables under `test/CMakeLists.txt`:

```cmake
add_executable(test_amgf_utils test/amgf/test_amgf_utils.cpp)
target_link_libraries(test_amgf_utils PRIVATE exaconstit)
add_test(NAME test_amgf_utils COMMAND test_amgf_utils)

add_executable(test_mortar_saddle_preconditioner_amgf
    test/mortar_pbc/test_mortar_saddle_preconditioner_amgf.cpp)
target_link_libraries(test_mortar_saddle_preconditioner_amgf PRIVATE exaconstit)
add_test(NAME test_mortar_saddle_preconditioner_amgf
    COMMAND test_mortar_saddle_preconditioner_amgf)
```

**Test framework notes:** ExaConstit's existing tests (e.g. `test/mortar_pbc/test_mortar_saddle_preconditioner.cpp`) use a custom `AssertOrDie`-style assertion macro (no GTest/Catch2 dependency). New tests should follow that pattern — define the test as a `main()` returning 0 on success, use `AssertOrDie(condition, "test name", "details")` for checks, and structure as a sequence of `test_*` free functions called from `main`. The CMake `add_test` then captures pass/fail via the exit code. See the existing file for the full pattern.

**Version requirements** (call out at the top of any new file's header comment):
- MFEM ≥ 4.9 (December 11, 2025 release; `AMGFSolver` added in this version).
- Ginkgo ≥ 1.6 (has `gko::experimental::factorization::Cholesky` and `Lu`).
- Hypre ≥ 2.31 (matches MFEM 4.9's GPU-aware MPI auto-config; you likely have this).
- Caliper ≥ 2.10 (the version ExaConstit currently builds against; new attributes added per Step 1.7 are compatible).

**Validation gate:** `cmake --build` succeeds; `ctest -R amgf` runs the new tests and they all pass.

---

### Step 1.7 — Caliper instrumentation and logging

**Files:** the new preconditioner class, the new utility.

**Action:** Beyond the `CALI_CXX_MARK_SCOPE` calls already specified per step, add the following Caliper attributes via the C-API helpers (`cali_set_int_byname`, `cali_set_double_byname`). Header: `#include "caliper/cali.h"` (already pulled in by ExaConstit's `utilities/mechanics_log.hpp`).

Inside `MortarSaddlePreconditionerAMGF::SetOperator` (Path A branch):

```cpp
cali_set_int_byname("amgf.subspace_dim", static_cast<int>(P_->GetGlobalNumCols()));
cali_set_double_byname("amgf.subspace_density",
                       double(P_->GetGlobalNumCols()) /
                       double(P_->GetGlobalNumRows()));
```

Inside `system_driver.cpp` after each linear solve (so the iteration count is queryable per Newton step):

```cpp
cali_set_int_byname("amgf.krylov_iters_last", J_solver->GetNumIterations());
```

These attributes make the Phase 1 validation runs trivial to compare across problems: a `runtime-report` Caliper config will surface the attributes alongside the region times.

**Logging:** when `options.solvers.linear_solver.print_level >= 1`, log a one-line summary at the start of each Newton step:

```
[AMGF] Path A active; |I_K|=523891 (5.2% of n_u=10.0M); Ginkgo Cholesky factor: 247 MB on host.
```

(The factor lives on host in the initial implementation per Step 0.5 / §7.4 of the analysis doc.)

---

### Step 1.8 — Validation gate for Phase 1 complete

Before declaring Phase 1 done and proceeding to Phase 2 / Phase 3 / etc., verify all of:

1. **Correctness:** the final converged $(u, \lambda)$ from a Path A run matches the baseline run to within Newton tolerance on a representative problem. (Path A only changes the preconditioner, not the system being solved — solutions must be identical.)
2. **Iteration count reduction:** Path A should reduce the linear-solver iteration count below the baseline, but for a $Q_2$ PBC problem where the Schur block is the dominant bottleneck, *don't expect dramatic reduction from Path A alone*. A 2–5× improvement is realistic; if Path A drops you from 10k to ~2k iterations you're seeing exactly the partial fix the structural analysis predicts (constraint-induced near-null modes in K block handled, Schur block lumping still bad). **This is the signal that Phase 2 is mandatory, not optional.**
3. **Per-iteration cost:** Path A adds at most 2× the per-iteration cost of baseline. (One extra BoomerAMG V-cycle + one Ginkgo forward/back substitution per iteration. The Cholesky factorization is amortized across iterations within a Newton step.)
4. **Infrastructure correctness:** the new `MortarSaddlePreconditionerAMGF` class wires correctly into `system_driver.cpp`, the `oper_mech`/`prec_mech` flow is preserved, Caliper instrumentation reports sensible numbers, and the Ginkgo-enabled build is reproducible. This is the more important Phase 1 outcome than the iteration-count win.
5. **Newton outer convergence:** the Newton iteration count is unaffected by the new preconditioner (or, ideally, decreases because tighter linear solves are now feasible). Path A should not destabilize the outer Newton flow.
6. **Backward compatibility:** running with `preconditioner = "jacobi"` (or `"amg"` etc.) still works exactly as before. The full ExaConstit test suite, run with each existing preconditioner option, still passes.

If 1, 3, 4, 5, 6 are satisfied and 2 shows the expected partial improvement, **Phase 1 is complete** and you should proceed to Phase 2 immediately. The Phase-1-as-permanent-solution outcome (where Path A alone makes the problem tractable) is unlikely for your $Q_2$ PBC case; treat Phase 1 as infrastructure-laying.

#### Diagnostic procedure when Path A does not reduce iteration count

If the iteration-count check (#2) shows *no* improvement — i.e. Path A's count matches the baseline — work the following decision tree before assuming the implementation is broken:

1. **Confirm AMGF is actually being applied.** Run with `CALI_CONFIG=runtime-report` and verify the `mortar_pbc::saddle_prec_amgf::set_operator` and `...::mult` regions are appearing in the Caliper output. If they aren't, the `system_driver.cpp` wiring from Step 1.6 didn't take the new branch. Inspect: is `preconditioner = "amgf"` parsed correctly? Is the `path_a_active` flag true at the construction site?
2. **Confirm the subspace solve is reasonable.** Log $n_c = $ `P_->GetGlobalNumCols()`. For a $Q_2$ PBC problem at $n_u \sim 10^6$ this should be roughly $10^5 - 1.5 \times 10^5$. If it's wildly off (e.g. zero, or comparable to $n_u$), `GetConstraintCoupledDofIndices` is wrong — return there and re-verify the index set.
3. **Confirm the AMG configuration carried through.** Print `amgf_->GetAMG().GetMaxLevels()`, strength threshold, etc., and compare against the existing `J_prec` AMG that the production code uses. If AMGF's AMG defaults overrode your `SetSystemsOptions(3)` / `SetRelaxType(8)` / `SetStrongThreshold(0.5)` calls, the configuration was applied at the wrong time (try moving the configuration into `SetOperator` instead of the constructor).
4. **Inspect the residual breakdown.** For each Krylov iteration, log $\|r_K\|$ and $\|r_\lambda\|$ separately. If $\|r_K\|$ converges but $\|r_\lambda\|$ stagnates, the Schur block (not the K block) is the bottleneck — *this is the expected outcome for $Q_2$ PBC* and confirms Phase 2 is mandatory rather than indicating a Phase 1 bug.
5. **Verify the boundary submatrix is reasonable.** Print $\|P^T K P\|_F$ vs $\|K\|_F$. The ratio should be on the order of $|\mathcal{I}_K|/n_u$. If the ratio is unreasonably small or zero, $P$ was built against a wrong index set (most likely the wrong global numbering — check that `idx_global` from `GetConstraintCoupledDofIndices` is in the same global numbering as $K$'s row partition).

The most common "Path A doesn't help" cause is *not* a bug; it's the residual-imbalance outcome from point 4, which is a feature: it's exactly the diagnostic that proves Phase 2's augmented Lagrangian is the right next step. Don't spend more than a day on diagnostic loops if the residual breakdown shows the Schur-block stagnation.

---

## 3. Phase 2 — Path D: Augmented Lagrangian Saddle Method + Optional AMGF on $K_\gamma$

**Goal:** Add the augmented-Lagrangian formulation as a saddle solver method:
`[Solvers.SaddlePoint] method = "AUGMENTED_LAGRANGIAN"`. The selected K-block
preconditioner is then applied to $K_\gamma = K + \gamma C^T C$. With
`[Solvers.Krylov] preconditioner = "AMGF"`, AMGF filters $K_\gamma$; with
`preconditioner = "AMG"`, this tests the augmented-Lagrangian method by itself.
This separation is intentional so the augmented method can be evaluated
independently of AMGF. The old `AMGF_AUG_LAGRANGIAN` preconditioner spelling is
compatibility only, not the primary Phase 2 switch.

This is the structural fix: it addresses *both* the K-block conditioning *and*
the catastrophically-bad Schur-block lumping *and* the
$\|r_K\| \gg \|r_\lambda\|$ residual imbalance, simultaneously. For your $Q_2$
PBC case where Phase 1's diagonal Schur lumping is the dominant bottleneck, this
is where the iteration count actually drops to the 30–80 range. Estimated
effort: 8–12 working days, building on the Phase 1 scaffolding.

The implementation has more moving parts than Phase 1, but each individual change is small. The order below is deliberate: each step is independently testable.

### Step 2.1 — Decide the class structure: toggle, subclass, or composition

**File:** `src/mortar_pbc/mortar_saddle_preconditioner_amgf.hpp/cpp`.

**The design question:** Path A and Path D share roughly 80% of their code (the AMGF wrapping on the (1,1) block is identical; only the operator being wrapped and the (2,2) block preconditioner differ). Three reasonable structures:

- **(a) Single class with a toggle.** `MortarSaddlePreconditionerAMGF` has a `bool use_path_d_` flag set at construction. `SetOperator` and `Mult` branch on the flag. Simplest. Downside: the class accumulates both paths' state even when only one is active.
- **(b) Two sibling classes.** `MortarSaddlePreconditionerAMGF_PathA` and `MortarSaddlePreconditionerAMGF_PathD`, both `mfem::Solver`-derived. Cleaner separation, slight code duplication.
- **(c) Base + derived.** Base class has the shared AMGF-on-(1,1) machinery; derived classes implement the Path-A-specific or Path-D-specific (2,2) block. Most "elegant" but adds polymorphism overhead and harder to read.

**Recommendation:** Option (a) for the initial implementation, but drive the
toggle from `SaddlePointSolverOptions::method`, not from
`PreconditionerType::AMGF_AUG_LAGRANGIAN`. `PreconditionerType::AMGF` should
mean only "use AMGF for the K block"; the saddle method decides whether that
block is `K` or `K_gamma`. Refactor to (b) only if the class grows unwieldy.

**The class declaration** (matches Step 1.4 — repeated here for the Phase 2 reader, but it's the same definition):

```cpp
namespace mortar_pbc {

class MortarSaddlePreconditionerAMGF : public mfem::Solver
{
public:
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        std::unique_ptr<mfem::HypreParMatrix> P,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override = -1.0);

    void SetOperator(const mfem::Operator& op) override;
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Required by SaddlePathDAugmenter (Phase 5.11.G): the augmenter
    /// reads the current gamma_ at each ApplyToResidual call.
    const double& gamma() const { return gamma_; }

private:
    // Shared infrastructure
    std::shared_ptr<mfem::Solver> K_jacobi_prec_;
    const MortarConstraintOperator& C_op_;
    std::unique_ptr<mfem::HypreParMatrix> P_;
    std::shared_ptr<mfem::Solver> subspace_solver_;
    std::shared_ptr<mfem::AMGFSolver> amgf_;  // owns its internal HypreBoomerAMG
    mfem::HypreParMatrix* K_ = nullptr;       // not owned; from BlockOperator

    // Path A state (Schur-diagonal lumping; ComputeInvDiagSchur returns a Vector)
    mfem::Vector schur_diag_inv_;

    // Path D state
    bool use_path_d_;
    double gamma_override_;
    mutable double gamma_ = 0.0;
    std::unique_ptr<mfem::HypreParMatrix> K_gamma_;

    // Dimensions (set in ctor from P->GetGlobalNumRows() and C_op_.Height())
    HYPRE_BigInt n_u_global_ = 0;
    int n_lambda_local_ = 0;
};

}  // namespace mortar_pbc
```

**Note** on the `bulk_amg_` member that earlier drafts had as a separate field: it isn't needed. `mfem::AMGFSolver` owns its `HypreBoomerAMG` internally and exposes it via `GetAMG()`; the AMG is configured in the ctor (`amgf_->GetAMG().SetSystemsOptions(3)`, etc.) and the AMG hierarchy is rebuilt automatically inside `amgf_->SetOperator(...)`. Don't accidentally create a second BoomerAMG instance — the duplication wastes memory and can cause subtle hierarchy-setup ordering bugs.

**Validation gate:** Phase 1's tests continue to pass with `use_path_d_ = false`. Code paths are clearly separated by the flag and visible in the diff.

---

### Step 2.2 — $C^T C$ assembly (cached, rebuilt only when mesh changes)

**File:** `src/mortar_pbc/mortar_constraint_operator.hpp/cpp`.

**Action:** Add a method that returns $C^T C \in \mathbb{R}^{n_u \times n_u}$ as a parallel sparse matrix. The implementation is one `HypreParMatrix::Mult` call (or `RAP` if you want to be explicit), but the *caching* is what matters: $C^T C$ depends only on the mesh and the periodicity-face DOF lists, neither of which changes during a nonlinear solve, so it can be built once at constraint-operator setup and reused for every Newton step. Phase 5.9's `Reset` (sub-XYZ filter change) invalidates the cache.

```cpp
class MortarConstraintOperator {
public:
    /**
     * @brief Returns C^T C as a parallel sparse matrix.
     *
     * @details Computed lazily on first call and cached for the lifetime of
     * the object (or until `Reset` invalidates it via filter-spec change).
     *
     * Used by Path D of the AMGF preconditioner to build
     * K_gamma = K + gamma * C^T C. See §5 of the AMGF analysis document
     * for the mathematical role.
     *
     * @par Cost (one-time)
     * O(nnz(C)^2 / n_lambda) for the sparse-matrix-matrix multiplication.
     * For typical ExaConstit mortar PBC, this is roughly
     * O(n_lambda * boundary_stencil_size^2), so cheap relative to K assembly.
     *
     * @par Sparsity
     * Nonzeros concentrated on boundary-adjacent rows/columns (DOFs in
     * I_K, obtainable via GetConstraintCoupledDofIndices). Interior DOFs
     * not coupled by any constraint contribute exactly zero rows.
     *
     * @par Validation contract
     * The returned matrix is guaranteed to have non-trivial Frobenius
     * norm (the build asserts this internally before returning the cache).
     * If C is rank-deficient — e.g. via the Wohlmuth-modified corner case
     * described in §10.1 of the analysis doc — this method still returns
     * a valid C^T C, but downstream code must handle the rank deficiency
     * (the augmentation gamma * C^T C contributes nothing in its kernel).
     */
    const mfem::HypreParMatrix& GetCTransposeC() const;

private:
    mutable std::unique_ptr<mfem::HypreParMatrix> m_CT_C_cached;
    mutable bool m_CT_C_built = false;
};
```

**Implementation outline** (in the .cpp):

1. On first call: build a `HypreParMatrix C_pm` from the existing per-pair block representation (the existing `BuildHypreParMatrix` path in the `MortarConstraintOperator`/`ConstraintBuilder3D`).
2. Compute `C^T C = mfem::ParMult(C_pm.Transpose(), C_pm)` (or equivalent).
3. Validate `C^T C->FNorm() > 0` (or use the cheaper local-diag Frobenius check via `AssembleDiagonal`). If zero, `MFEM_ABORT` with the §10.1 message — this is an upstream bug, not a recoverable condition. The constraint builder should never emit an all-zero $C$.
4. Cache and return reference.

**Sub-XYZ filter handling:** `Reset(active_pair_labels, comp_mask)` invalidates by setting `m_CT_C_built = false`. The next call rebuilds against the new active set.

**Memory budget check:** $C^T C$ has nonzeros only on rows/columns indexed by $\mathcal{I}_K$. For $Q_2$ at $n_u = 10^7$ and $|\mathcal{I}_K| \approx 1.5 \times 10^6$ with a per-row stencil of ~50 nonzeros, $C^T C$ is roughly $7.5 \times 10^7$ nonzeros total — about 1.2 GB across all ranks at 16 bytes per nonzero (int + double). Significant memory but fits comfortably on HPC partitions.

**Validation gate (unit test):**
- On a 5-node 1D test problem (where $C$ can be written down by hand), verify $C^T C \cdot v = C^T (C \cdot v)$ for random $v$.
- Verify the sparsity pattern matches expectation: zeros on interior rows.
- Re-call after `Reset(...)` with a different filter spec and verify the new $C^T C$ matches the new filter (smaller support if a sub-axis spec drops some pairs).

**Caliper:** `CALI_CXX_MARK_SCOPE("mortar_pbc::mortar_constraint_operator::get_c_transpose_c")` — the first call (build) and subsequent calls (cache hit) both go through this scope so you'll see both in the Caliper output.

---

### Step 2.3 — $\gamma$ parameter selection

**File:** `MortarSaddlePreconditionerAMGF::SetOperator` (compute the default) and `option_parser_v2.hpp` (allow override).

**The natural-scaling formula:**

$$\gamma_{\text{default}} = \frac{\text{tr}(K)}{\text{tr}(C^T C)} \cdot \frac{n_\lambda}{n_u}$$

The motivation (§5.10 of the analysis doc): this scaling makes the Frobenius norms of $K$ and $\gamma C^T C$ comparable when restricted to the boundary subspace, putting $\gamma$ in the "productive range" where the augmentation is strong enough to dominate the constraint Schur but not so strong that $K_\gamma$ becomes ill-conditioned in the bulk.

**Implementation:**

```cpp
double ComputeDefaultGamma(const mfem::HypreParMatrix& K,
                           const mfem::HypreParMatrix& CTC,
                           HYPRE_BigInt n_lambda_global,
                           HYPRE_BigInt n_u_global,
                           MPI_Comm comm)
{
    // Use AssembleDiagonal (the canonical mfem::Operator method) so this
    // code works uniformly across HypreParMatrix, PA, EA, and FA forms.
    // For HypreParMatrix specifically, AssembleDiagonal fills the local-
    // diagonal-block diagonal entries (which is what tr() needs — see the
    // existing block_jacobi preconditioner in the prototype's saddle_point.py
    // for the same idiom).
    mfem::Vector diag_K(K.Height());
    diag_K = 0.0;
    K.AssembleDiagonal(diag_K);
    const double trK_local = diag_K.Sum();
    double trK = 0.0;
    MPI_Allreduce(&trK_local, &trK, 1, MPI_DOUBLE, MPI_SUM, comm);

    mfem::Vector diag_CTC(CTC.Height());
    diag_CTC = 0.0;
    CTC.AssembleDiagonal(diag_CTC);
    const double trCTC_local = diag_CTC.Sum();
    double trCTC = 0.0;
    MPI_Allreduce(&trCTC_local, &trCTC, 1, MPI_DOUBLE, MPI_SUM, comm);

    if (trCTC <= 0.0) {
        // §10.1 of the analysis doc: degenerate case (C entirely zero,
        // or C has rank zero after dropping corner rows). Falling back
        // to gamma=1 keeps the run going but the upstream constraint
        // builder almost certainly has a bug if this fires.
        MFEM_WARNING("ComputeDefaultGamma: tr(C^T C) = 0 -- check that the "
                     "MortarConstraintOperator has non-trivial active rows. "
                     "Falling back to gamma=1.");
        return 1.0;
    }

    return (trK / trCTC) *
           (static_cast<double>(n_lambda_global) /
            static_cast<double>(n_u_global));
}
```

**Why `AssembleDiagonal` and not `GetDiag`:** the existing prototype code (`experimental/mortar_pbc_proto/mortar_pbc/saddle_point.py`) documents this idiom explicitly: `AssembleDiagonal` is the canonical `mfem::Operator` method that works on PA, EA, FA, and HypreParMatrix forms uniformly, while `GetDiag(Vector&)` only works on `HypreParMatrix`. Even though the AMGF path requires FULL assembly per Step 0.5, using `AssembleDiagonal` keeps the code robust under future hybrid-assembly modes (§10.3 of the analysis doc).

**Practical $\gamma$ tuning sequence:**

1. **First run:** compute $\gamma_{\text{default}}$ on a small problem, log it, run. Measure iteration count.
2. **Sensitivity sweep:** $\gamma \in \{0.1, 0.3, 1, 3, 10\} \cdot \gamma_{\text{default}}$. The iteration count should be roughly flat in the middle of this range. If it isn't, the formula needs adjustment for your specific problem class.
3. **Production:** set $\gamma$ in the option parser to the value that gave best iteration count on the small problem. The analysis (§5.10) says this scales fairly well with problem size, but verify on the next-size-up problem.

**Logging:** at the start of each Newton step, log the $\gamma$ value used. If $\gamma$ varies (because it's recomputed each Newton step from the current $K$), that's informative diagnostic content.

---

### Step 2.4 — $K_\gamma$ assembly (once per Newton step)

**File:** `MortarSaddlePreconditionerAMGF::SetOperator`.

**Action:** When Path D is active, assemble $K_\gamma = K + \gamma C^T C$ inside `SetOperator`, which is called once per Newton step when $K$ is rebuilt:

```cpp
void MortarSaddlePreconditionerAMGF::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::set_operator");

    // Extract K from the saddle BlockOperator.
    // The dynamic_cast must succeed: Step 0.5's option-validation guarantees
    // FULL assembly when AMGF is selected, so the (0,0) block of the saddle
    // BlockOperator is an assembled HypreParMatrix. A failed cast here means
    // a Step-0.5 bypass somewhere upstream.
    const auto& block_op = dynamic_cast<const mfem::BlockOperator&>(op);
    K_ = dynamic_cast<mfem::HypreParMatrix*>(&block_op.GetBlock(0, 0));
    MFEM_VERIFY(K_, "BlockOperator (0,0) must be a HypreParMatrix for AMGF. "
                    "This indicates a Step-0.5 option-validation bypass: "
                    "AMGF requires FULL assembly.");

    // Refresh K_jacobi_prec on the current K, used for Path A's Schur-diagonal
    // probe via ComputeInvDiagSchur. This matches the pattern in the existing
    // mortar_pbc::MortarSaddlePreconditioner::SetOperator.
    K_jacobi_prec_->SetOperator(*K_);

    if (use_path_d_) {
        // Compute gamma (cheap; tr() calls are O(n) plus one MPI_Allreduce).
        const mfem::HypreParMatrix& CTC = C_op_.GetCTransposeC();
        gamma_ = (gamma_override_ > 0.0)
                     ? gamma_override_
                     : ComputeDefaultGamma(*K_, CTC, n_lambda_global_,
                                           n_u_global_, MPI_COMM_WORLD);

        // Assemble K_gamma = 1.0 * K + gamma * C^T C.
        // mfem::Add returns a heap-allocated HypreParMatrix; reset()
        // disposes the previous K_gamma_ and stores the new one cleanly.
        K_gamma_.reset(mfem::Add(1.0, *K_, gamma_, CTC));

        // AMGF operates on K_gamma now (rebuilds the AMG hierarchy).
        amgf_->SetOperator(*K_gamma_);
    } else {
        // Path A: AMGF operates on K directly. The Schur-diagonal lumping
        // is rebuilt via the existing C_op API.
        amgf_->SetOperator(*K_);
        // ComputeInvDiagSchur RETURNS a Vector (verified signature; not an
        // out-parameter — the prototype's pattern was different).
        schur_diag_inv_ = C_op_.ComputeInvDiagSchur(*K_jacobi_prec_);
    }

    // Update inherited Solver dimensions to match the BlockOperator.
    height = K_->Height() + n_lambda_local_;
    width = height;
}
```

**Subtleties:**

1. **`mfem::Add` ownership:** the returned pointer is heap-allocated; wrap in `unique_ptr` to avoid leaks. Replace it cleanly on each `SetOperator` call (the `reset` does both deletion of the prior and storage of the new).

2. **AMG hierarchy rebuild cost:** BoomerAMG's hierarchy is rebuilt on every `SetOperator` (this is true for both Path A and Path D — it's not a Path D-specific cost). Profile it. If it's a substantial fraction of Newton-step time, Hypre exposes some hierarchy-reuse options (`HYPRE_BoomerAMGSetKeepTranspose`, plus careful setup-management) but they require sparsity-pattern invariance across calls. For typical crystal-plasticity Newton iterations the sparsity is invariant, so reuse is feasible — but it's an optimization to investigate only if the AMG setup proves to dominate. The default behavior (full rebuild per Newton step) is correct.

3. **The AMG configuration applies to $K_\gamma$, not $K$.** Your verified HMIS / strength threshold 0.5 / `RelaxType=8` was tuned for $K$. The augmented operator $K_\gamma$ has the same sparsity structure as $K$ plus rank-$n_\lambda$ fill-in on the boundary. *Empirically* the same configuration works for $K_\gamma$ — the §5.7 analysis shows $\kappa(K_\gamma) \approx \kappa(K)$ for reasonable $\gamma$ — but verify with a Caliper-instrumented run that the per-V-cycle iteration count on $K_\gamma$ alone is similar to that on $K$ alone.

4. **Memory: storing both $K$ and $K_\gamma$.** Both are needed during a Newton step: $K$ for residual computation in the next outer iteration; $K_\gamma$ for the preconditioner. Memory roughly doubles for the displacement-block operators. With $C^T C$ also cached, total is roughly $3 \times $ baseline. For $n_u = 10^7$ at $Q_2$, this is on the order of ten GB across the parallel partition — fine on HPC.

5. **FULL assembly is required here.** $K$ must be a `HypreParMatrix`, not a PA/EA operator. Step 0.5's validation rule enforces this at option-parse time; the `dynamic_cast` in `SetOperator` is a defensive runtime check for the same constraint. If `dynamic_cast` returns `nullptr` here, something has slipped past Step 0.5 — abort with a clear message, do not silently fall through.

**Validation gate:** with `use_path_d_=true` and `gamma_override_=0.0`, the preconditioner reduces exactly to Path A behavior. Use this as a regression test for the Path D wiring.

---

### Step 2.5 — Trivial $\gamma I$ Schur preconditioner

**File:** `MortarSaddlePreconditionerAMGF::Mult`.

**Action:** Replace the existing Path A Schur scaling with a scalar multiply by $\gamma$ when Path D is active. **Use `BlockVector::GetBlock(...)` to access the displacement and multiplier blocks** — do not slice the flat underlying memory via `GetData() + offset`, because the saddle BlockVector blocks may not be contiguous in memory (especially with MFEM's device-memory tracking) and the resulting `mfem::Vector(double*, int)` view would fight the memory manager.

```cpp
void MortarSaddlePreconditionerAMGF::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::mult");

    // x and y are saddle BlockVectors with layout (u, lambda).
    // The caller (MINRESSolver / CGSolver / GMRESSolver) always passes the
    // saddle BlockVector here; the static_cast is safe and matches the
    // pattern in mortar_pbc::MortarSaddlePreconditioner::Mult.
    const auto& xb = static_cast<const mfem::BlockVector&>(x);
    auto&       yb = static_cast<mfem::BlockVector&>(y);

    const mfem::Vector& x_u      = xb.GetBlock(0);
    const mfem::Vector& x_lambda = xb.GetBlock(1);
    mfem::Vector&       y_u      = yb.GetBlock(0);
    mfem::Vector&       y_lambda = yb.GetBlock(1);

    // (1,1) block: AMGF (Path A: on K; Path D: on K_gamma)
    amgf_->Mult(x_u, y_u);

    // (2,2) block: Path A uses the existing diagonal Schur lumping;
    // Path D uses the trivial scalar multiplication by gamma.
    if (use_path_d_) {
        // y_lambda = gamma * x_lambda. DEVICE_DEBUG-clean via typed
        // accessors, matching the pattern in DiagonalScaler::Mult.
        const int n_lam = x_lambda.Size();
        MFEM_ASSERT(y_lambda.Size() == n_lam, "Path D Mult: size mismatch");
        const double  g  = gamma_;
        const double* xd = x_lambda.HostRead();
        double*       yd = y_lambda.HostWrite();
        for (int i = 0; i < n_lam; ++i) { yd[i] = g * xd[i]; }
    } else {
        // Path A: same diag(C * diag(K)^{-1} * C^T)^{-1} scaling that
        // the existing MortarSaddlePreconditioner uses (matches the
        // DiagonalScaler::Mult pattern verbatim).
        const int n_lam = x_lambda.Size();
        MFEM_ASSERT(schur_diag_inv_.Size() == n_lam,
                    "Path A Mult: schur_diag_inv_ size mismatch");
        const double* xd  = x_lambda.HostRead();
        const double* idd = schur_diag_inv_.HostRead();
        double*       yd  = y_lambda.HostWrite();
        for (int i = 0; i < n_lam; ++i) { yd[i] = idd[i] * xd[i]; }
    }
}
```

**Headers:** `#include "mfem.hpp"`, `#include "caliper/cali.h"` if Caliper is enabled. The `BlockVector` class is in `mfem/linalg/blockvector.hpp` (already pulled in via `mfem.hpp`).

The $\gamma I$ preconditioner is *exact* in the limit $\gamma \to \infty$ and *uniformly good* (condition number bounded) for moderate $\gamma$ — see §5.8 of the analysis doc. The result is that the entire (2,2) block apply collapses from a vector-of-divisions (Path A) to a single scaling (Path D).

**Validation gate:** on a tiny test problem, verify that with $\gamma \to \infty$ the Path D solution converges to the same $(u, \lambda)$ as Path A (both should converge to the same Newton step at the linear solve level).

---

### Step 2.6 — Right-hand side augmentation via `SaddleResidualScaler`

**File:** new `src/mortar_pbc/saddle_path_d_augmenter.hpp/cpp`, plus minor wiring in `src/system_driver.cpp`.

**Critical correctness point:** This is the single change most likely to be done incorrectly. Read this section twice before implementing.

**What changes.** The augmented saddle system has a modified displacement-block residual:

$$r_K^{\text{aug}} = r_K + \gamma \, C^T r_\lambda$$

The multiplier-block residual $r_\lambda$ is unchanged. The modification is applied *only* when forming the right-hand side of the linear solve, *not* in Newton residual evaluation, *not* in convergence checks, *not* in trust-region merit functions.

**Where to put it: use the existing `SaddleResidualScaler` slot, not `system_driver.cpp`.** ExaConstit already has a Phase 5.11 architectural pattern for coordinate-converting saddle residuals around the trust-region Newton solve and dogleg output — `mortar_pbc::SaddleResidualScaler`, wired into `ExaTrustRegionSolver::SetScaler(scaler, block_offsets)`. That class exists precisely so the RHS transformations live in one place instead of being scattered across `system_driver.cpp`. Path D's augmentation is exactly this kind of transformation, so the implementation follows the same pattern.

The trust-region linear solve (per the `ExaTrustRegionSolver::Mult` algorithm in `src/solvers/trust_region_solver.cpp`) runs:
1. Compute Newton residual `r = oper_mech->Mult(x) - b` (unaugmented; unchanged).
2. Compute Jacobian `J = oper_mech->GetGradient(x)` (the saddle BlockOperator; unchanged).
3. Compute steepest descent `grad = J^T * r` (used for Cauchy step length; **unchanged** — operates on the unaugmented residual).
4. Compute `Jg_2 = ||J * grad||^2` (used for the Cauchy step length; **unchanged**).
5. Solve `J * c = r` via the Krylov solver `prec_mech` → `nrStep = -c`. **This is where the RHS augmentation is applied.**
6. Build the dogleg step from `nrStep` and `grad`; apply trial step; evaluate residual; rho update.

The Cauchy direction (step 3) is computed from the *unaugmented* residual `r`. The Cauchy step length (step 4) uses the unaugmented `J`. The Newton step is the only place that sees the augmented form, and we are free to internally rewrite it however we like *provided the returned `c` is the solution to the unaugmented system* — which §5.4 of the analysis doc proves it is.

**Implementation:**

```cpp
// In src/mortar_pbc/saddle_path_d_augmenter.hpp:

#pragma once
#include "mortar_constraint_operator.hpp"
#include "mfem.hpp"
#include <memory>

namespace mortar_pbc {

/**
 * @brief Phase 5.11.G saddle residual scaler that applies the Path D
 *        augmented-Lagrangian RHS modification.
 *
 * Implements the rule (used ONLY at the Newton-solve RHS, not in
 * residual evaluation or merit-function computation):
 *
 *     r_K_augmented := r_K + gamma * C^T * r_lambda
 *     r_lambda      := r_lambda  (unchanged)
 *
 * This is the §5.4 theorem of the AMGF analysis: the augmented
 * saddle system has the same (du, dlambda) solution as the
 * unaugmented one, but with the displacement-block RHS rewritten
 * as above.
 *
 * The class implements the SaddleResidualScaler interface
 * (`ApplyToResidual`, `UnapplyToIncrement` — names match the
 * existing Phase 5.11 family), so it slots into
 * `ExaTrustRegionSolver::SetScaler(scaler, block_offsets)`.
 *
 * @note `UnapplyToIncrement` is the identity here: the augmented
 * linear solve already returns the unaugmented (du, dlambda), so no
 * back-conversion is needed.
 *
 * @note Re-reads gamma from the preconditioner on each call so the
 * scaler stays in sync with the current Newton step's gamma value
 * (gamma is recomputed each Newton step from tr(K) / tr(C^T C)).
 */
class SaddlePathDAugmenter : public SaddleResidualScaler {
public:
    SaddlePathDAugmenter(const MortarConstraintOperator& C_op,
                         const double& gamma_ref);

    void ApplyToResidual(mfem::Vector& r) const override;
    void UnapplyToIncrement(mfem::Vector& delx) const override { /* identity */ }

private:
    const MortarConstraintOperator& m_C_op;
    const double& m_gamma;       // reference to the preconditioner's gamma_
    const mfem::Array<int>& m_block_offsets;
};

}  // namespace mortar_pbc
```

```cpp
// In src/mortar_pbc/saddle_path_d_augmenter.cpp:

void SaddlePathDAugmenter::ApplyToResidual(mfem::Vector& r) const
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::path_d_augmenter::apply_to_residual");

    // r is a saddle BlockVector with layout (u, lambda).
    auto& rb = static_cast<mfem::BlockVector&>(r);
    mfem::Vector& r_K      = rb.GetBlock(0);
    mfem::Vector& r_lambda = rb.GetBlock(1);

    // tmp = C^T r_lambda. (C^T is the existing MultTranspose; see
    // mortar_constraint_operator.cpp for the host-side walk + Alltoallv
    // back-scatter.)
    mfem::Vector tmp(m_C_op.Width());
    m_C_op.MultTranspose(r_lambda, tmp);

    // r_K += gamma * tmp.  r_lambda is unchanged.
    r_K.Add(m_gamma, tmp);
}
```

**Wiring it in (one change in `system_driver.cpp`, right after the preconditioner is constructed):**

The wiring snippet shown in Step 1.6 already includes the scaler hook (it's the `if (path_d_active) { ... }` block right after the `saddle_prec` is built). For clarity, the body of that block is:

```cpp
if (path_d_active) {  // same flag derived from PreconditionerType in Step 1.6
    auto& amgf_prec = dynamic_cast<mortar_pbc::MortarSaddlePreconditionerAMGF&>(
        *saddle_prec);
    auto path_d_scaler = std::make_shared<mortar_pbc::SaddlePathDAugmenter>(
        *mortar_op, amgf_prec.gamma());

    // ExaTrustRegionSolver's SetScaler hooks the scaler into Mult so it
    // is applied around the Newton-solve and the dogleg-output (see the
    // existing Phase 5.11.G code in src/solvers/trust_region_solver.cpp).
    if (auto* tr_solver = dynamic_cast<ExaTrustRegionSolver*>(newton_solver.get())) {
        tr_solver->SetScaler(path_d_scaler, saddle_block_offsets);
    } else {
        // ExaNewtonSolver / ExaNewtonLSSolver don't yet have an equivalent
        // scaler slot. Either extend them, or require TRDOG when Path D is
        // selected. For initial implementation we require TRDOG.
        MFEM_ABORT("AMGF_AUG_LAGRANGIAN currently requires nl_solver = TRDOG");
    }
}
```

**Why this is the right place — and what it preserves:**
- The Cauchy direction `grad = J^T r` (step 3 above) uses the unaugmented `r`. Trust-region geometry is preserved.
- The Cauchy step length `Jg_2 = ||J * grad||^2` (step 4) uses the unaugmented `J`. Step-length scaling is preserved.
- The Newton-step residual is the only quantity that the scaler touches, and only inside the `prec_mech->Mult(r_aug, c)` call. The returned `c = -nrStep` is the exact solution to the unaugmented saddle system (§5.4 theorem).
- `UnapplyToIncrement` is the identity, so no back-conversion is needed after the dogleg.

**What does NOT change:**
- The Newton residual `r = oper_mech->Mult(x) - b` uses the unaugmented saddle BlockOperator. Newton convergence checks are on the unaugmented residual.
- The trust-region merit function is computed from the unaugmented residual.
- The solution `(u, lambda)` is interpreted as the solution to the unaugmented problem. Path D is *only* a preconditioning trick.

**Validation gate (the most important one in this entire guide):** on a tiny problem with known solution, run Path A and Path D to full Newton + Krylov convergence and verify the converged $(u, \lambda)$ match componentwise to within Newton tolerance. If they don't, the scaler is wrong and *do not proceed* until it's fixed.

**Validation gate 2:** on a problem that exercises the trust-region machinery (one with at least one rejected step before acceptance), verify the sequence of trust-region radii and the sequence of accepted/rejected steps are *identical* between Path A and Path D. If they differ, the scaler is touching something it shouldn't (most likely you've applied it outside the Newton-solve step, e.g. inside the residual evaluation).

---

### Step 2.7 — $\gamma$ sensitivity study (validation, not a code change)

**Action:** Run the same test problem with $\gamma \in \{0.01, 0.1, 1, 10, 100\} \cdot \gamma_{\text{default}}$ — five log-spaced values across four decades. Record:

- Linear-solver iteration count at each $\gamma$.
- Newton iteration count at each $\gamma$.
- Converged $(u, \lambda)$ — verify all five give the same solution (they should; the theorem says so).

**Expected pattern:**

| $\gamma$ relative to default | Expected iteration count behavior |
|---|---|
| $0.01 \times$ | High iter count — augmentation too weak, behaves like unaugmented |
| $0.1 \times$ | Moderate — getting closer to productive range |
| $1 \times$ (default) | Low — should be near the minimum |
| $10 \times$ | Low (similar to default, possibly slightly higher) |
| $100 \times$ | Rising — $K_\gamma$ becoming dominated by $\gamma C^T C$, AMG quality degrades |

A flat plateau across $\{0.1, 1, 10\} \cdot \gamma_{\text{default}}$ is the success signal: the natural scaling is in the productive range. If the plateau is narrow or absent, the formula needs adjustment for your problem class — usually a different exponent on $n_\lambda / n_u$, or a switch from trace ratio to Frobenius norm ratio. See §5.10 for alternatives.

**Time cost:** 5 runs × small problem ≈ 30 minutes. Do this once at the start of Phase 2 validation, not every time you tweak the code.

---

### Step 2.8 — Trust-region dogleg interaction verification

**File:** No code change typically needed; this is a verification step that the Step 2.6 scaler is correctly wired.

The Step 2.6 design routes the Path D RHS augmentation through `SaddleResidualScaler` → `ExaTrustRegionSolver::SetScaler`. The scaler is applied inside `ExaTrustRegionSolver::Mult` around the Newton-solve only — the Cauchy direction and Cauchy step-length use the unaugmented residual and unaugmented Jacobian respectively. So Path A and Path D trust-region trajectories should be *identical* (same sequence of trust-region radii, same accept/reject pattern).

**Verification protocol:**
1. Pick a test problem that exercises the trust-region machinery — one with at least one rejected step before final acceptance. The polycrystal-with-localized-plasticity benchmarks typically do this.
2. Run with `preconditioner = "AMGF"` (Path A) and record:
   - Sequence of trust-region radii $\delta$ across outer iterations.
   - Sequence of accepted/rejected steps.
   - Sequence of $\rho$ values.
   - Final converged $(u, \lambda)$.
3. Run with `preconditioner = "AMGF_AUG_LAGRANGIAN"` (Path D) on the same input. Compare:
   - All four sequences must be identical (modulo bit-level Krylov noise within tolerance).

**If they differ:** the most likely culprits, in order of probability:
1. The scaler is being applied somewhere it shouldn't be (e.g. inside residual evaluation, or before the Cauchy direction computation). Check that `SetScaler` is called *after* `SetSolver` and *before* the first `Mult`.
2. The scaler holds a stale reference to gamma. Check that `gamma_` is updated in `SetOperator` *before* the next Newton solve.
3. The scaler's `UnapplyToIncrement` is not the identity (the existing 5.11.G scalers do back-conversion; the Path D augmenter must override `UnapplyToIncrement` to be a no-op).

---

### Step 2.9 — Outer Powell-Hestenes multiplier update (optional, defer)

**Action:** Don't implement this yet.

For the linearized single-Newton-step problem, the augmented linear solve already returns the exact $\lambda$ — no outer multiplier update is needed (§5.9 of the analysis doc).

The outer Powell-Hestenes update becomes relevant only if you later choose to solve the inner linear system with a loose tolerance to save Krylov iterations. That's a Phase 3+ optimization, not a Phase 2 implementation step. Mention it in code comments so the next developer knows the option exists, but don't build it.

---

### Step 2.10 — Caliper instrumentation specific to Path D

**Action:** Add Caliper instrumentation at three levels: region scopes (already established with `CALI_CXX_MARK_SCOPE` throughout), per-Newton-step scalar attributes, and per-iteration counters.

Caliper exposes a few API surfaces; ExaConstit's existing code already uses `CALI_CXX_MARK_SCOPE` for region timing. For the scalar attributes below, use the C-API helpers (`cali_set_double_byname`, `cali_set_int_byname`) — they are the simplest way to surface a scalar value attached to the current region. Header: `#include "caliper/cali.h"` (already pulled in by ExaConstit's `utilities/mechanics_log.hpp`).

**Per-Newton-step scalar attributes** (set inside `MortarSaddlePreconditionerAMGF::SetOperator`):

```cpp
void MortarSaddlePreconditionerAMGF::SetOperator(const mfem::Operator& op)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::saddle_prec_amgf::set_operator");

    // ... K extraction, K_jacobi_prec refresh, etc. (existing body) ...

    if (use_path_d_) {
        CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::ctc_lookup");
        const mfem::HypreParMatrix& CTC = C_op_.GetCTransposeC();
        CALI_MARK_END("mortar_pbc::saddle_prec_amgf::ctc_lookup");

        gamma_ = (gamma_override_ > 0.0)
                     ? gamma_override_
                     : ComputeDefaultGamma(*K_, CTC, n_lambda_global_,
                                           n_u_global_, MPI_COMM_WORLD);

        // Surface gamma as a queryable attribute on this Newton step.
        cali_set_double_byname("amgf.path_d.gamma_value", gamma_);

        CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::k_gamma_assembly");
        K_gamma_.reset(mfem::Add(1.0, *K_, gamma_, CTC));
        CALI_MARK_END("mortar_pbc::saddle_prec_amgf::k_gamma_assembly");

        CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::amg_setup_k_gamma");
        amgf_->SetOperator(*K_gamma_);
        CALI_MARK_END("mortar_pbc::saddle_prec_amgf::amg_setup_k_gamma");
    } else {
        CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::amg_setup_k");
        amgf_->SetOperator(*K_);
        CALI_MARK_END("mortar_pbc::saddle_prec_amgf::amg_setup_k");

        CALI_MARK_BEGIN("mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");
        schur_diag_inv_ = C_op_.ComputeInvDiagSchur(*K_jacobi_prec_);
        CALI_MARK_END("mortar_pbc::saddle_prec_amgf::compute_inv_diag_schur");
    }

    cali_set_int_byname("amgf.subspace_dim",
                        static_cast<int>(P_->GetGlobalNumCols()));

    height = K_->Height() + n_lambda_local_;
    width = height;
}
```

**Per-iteration counters** — the Krylov iteration count is captured by the outer iterative solver's standard reporting. To compare Path A vs Path D explicitly, log it at the end of each linear solve via the iterative solver's `GetNumIterations()`. This is typically done in `system_driver.cpp` after the call to `newton_solver->Mult(...)`; one extra `cali_set_int_byname("amgf.krylov_iters_last", J_solver->GetNumIterations())` per Newton step is enough.

**Attribute summary** for post-processing:

- `amgf.path_d.gamma_value` — gamma at each Newton step (double).
- `amgf.subspace_dim` — global column count of P (int).
- `amgf.krylov_iters_last` — Krylov iterations of the last linear solve (int).
- Region scopes: `mortar_pbc::saddle_prec_amgf::set_operator`, `...::mult`, `...::k_gamma_assembly`, `...::amg_setup_k_gamma`, `...::amg_setup_k`, `...::ctc_lookup`, `...::compute_inv_diag_schur`, plus the new `mortar_pbc::path_d_augmenter::apply_to_residual` from Step 2.6.

Run with `CALI_CONFIG=runtime-report,profile.mpi` or your existing Caliper config to surface the per-region wall-clock breakdown. The output will reveal whether AMG setup on $K_\gamma$ is comparable to or much more expensive than AMG setup on $K$ alone — the §5.7 analysis predicts comparable cost; if the actual numbers diverge significantly, the strength threshold or coarsening may need re-examination for $K_\gamma$.

---

### Step 2.11 — Validation gate for Phase 2 complete

Before declaring Phase 2 done and moving to Phase 3, verify all of:

1. **Correctness (most important):** Path D's converged $(u, \lambda)$ matches Path A's (and the baseline's) converged $(u, \lambda)$ on a representative problem, componentwise to within Newton tolerance. This is the §5.4 theorem expressed as a regression test. If this fails, Step 2.6 is the most likely culprit.

2. **Iteration count target reached:** Path D should reduce the linear-solver iteration count from your baseline 10k+ to the 30–100 range on a $Q_2$ PBC test problem. This is the actual win.

3. **$\gamma$-robustness:** the sensitivity sweep from Step 2.7 shows a flat plateau of at least one decade in $\gamma$ around the default value. If the plateau is absent or very narrow, the formula needs adjustment but the implementation is correct.

4. **Residual rebalancing visible:** $\|r_K^{\text{aug}}\|$ and $\|r_\lambda\|$ are now within 1–2 orders of magnitude of each other throughout the Newton iteration, in contrast to the 3+ orders of magnitude observed in the baseline run. This is the §6 prediction made manifest.

5. **Wall-clock improvement:** total Newton-step wall-clock for Path D is substantially better than baseline. Per-iteration cost goes up modestly (denser $K_\gamma$, larger AMG setup), but the dramatic iteration count drop dominates. If Path D is somehow slower than baseline, the AMG setup on $K_\gamma$ is the most likely cause — profile it.

6. **Newton outer convergence unchanged:** Path D should not increase the number of Newton outer iterations. If it does, the trust-region interaction needs investigation (Step 2.8 territory).

7. **Trust-region machinery works correctly:** the dogleg solver accepts/rejects steps identically to Path A on a problem that exercises the trust region.

If all seven are satisfied, **Phase 2 is complete and you have a production-quality saddle preconditioner.** Phase 3 (production benchmarks across the full ExaConstit envelope) follows; expect mostly cleanup, scaling tests, and edge-case validation. Phase 4 (Path B) and Phase 5 (alternative AMG) are unlikely to be needed if Phase 2 numbers come in as expected.

---

## 4. Higher-order ($Q_2$/$Q_3$) specific notes

Practical considerations for your $Q_2$ case beyond what's already in the steps above:

1. **$|\mathcal{I}_K|$ scales differently from $Q_1$.** For $Q_2$ hex elements on a 6-face periodicity cube, expect $|\mathcal{I}_K| \approx 0.10\, n_u$ to $0.15\, n_u$ depending on mesh aspect ratio. With Ginkgo's Cholesky factorization on host (per the Step 0.5 / §7.4 FA-on-host constraint of the initial implementation), the factor for the boundary submatrix at $n_u = 10^7$ with $|\mathcal{I}_K| = 1.5 \times 10^6$ is roughly $(1.5 \times 10^6)^{1.5} \approx 1.8 \times 10^9$ nonzeros total — about 25 GB at 16 bytes per nonzero, distributed across host memory of all participating ranks. Comfortable on typical HPC nodes (100s of GB host RAM per node). For $Q_3$ at the same $n_u$, plan for closer to 100 GB across the node — verify the factor fits before launching.

2. **Wohlmuth corner modifications interact non-trivially with $Q_2$.** For $Q_2$ faces, the partition-of-unity corner modifications also need to be considered at the *edge* DOFs of the periodicity faces (not just the geometric corners of the cube). Make sure `GetConstraintCoupledDofIndices()` correctly includes the edge DOFs — verify with a hand-counted small test problem before scaling up. The verification: build a tiny $Q_2$ PBC problem (say, 2×2×2 mesh) and confirm the count of returned indices matches what you'd expect by hand-enumerating the boundary nodes.

3. **The Path D augmentation cost is $Q_2$-friendly.** $C^T C$ has the same sparsity structure regardless of element order — its sparsity is governed by the geometry of the periodicity coupling, not the polynomial order. So $C^T C$ assembly (Step 2.2) doesn't get more expensive at higher order. What gets more expensive is the cost of one BoomerAMG V-cycle on $K_\gamma$ — but that's exactly what you're already paying for the V-cycle on $K$, so the relative cost is roughly unchanged. The dominant new cost is the Ginkgo Cholesky factorization, which scales as $(n_c)^{1.5}$ for the typical surface-stencil sparsity.

4. **Host-side execution is acceptable for the initial implementation.** Because Step 0.5 forces CPU/OpenMP runtime when AMGF is enabled, the AMGF machinery is host-resident — both BoomerAMG and Ginkgo run on host. No host-device transfer overhead per Krylov iteration since there is no device. The slowdown from giving up EA-on-GPU is more than offset by the iteration count dropping from 10k+ to 30–80. Future work (§7.4 Option 2 of the analysis doc) enables the hybrid EA-for-residual + FA-for-preconditioner mode for full GPU residency; choosing Ginkgo now (rather than MUMPS or CPardiso) keeps that future path open without a re-port.

5. **Expected per-Newton-step Krylov count after Path D.** For your well-tuned bulk AMG ($\beta \approx 5\text{--}10$, consistent with the BoomerAMG output you showed) and a properly-implemented Path D, the Petrides bound $\kappa(M^{-1}A) \leq 2(\beta + 3)$ gives $\kappa \leq 26$. CG/MINRES then converges to $\epsilon = 10^{-8}$ in roughly $\sqrt{26} \log(10^8) \approx 38$ iterations. Realistically, expect Phase 2 to bring you from 10k+ iterations to **30–80 iterations per Newton step**, with the variation depending on polycrystal heterogeneity and how aggressively you've tuned $\gamma$. That's the order-of-magnitude target to aim for and the gate that signals you're done.

6. **Non-associated flow at $Q_2$.** When $K$ is non-symmetric (from non-associated plastic flow), the boundary submatrix $P^T K_\gamma P$ is also non-symmetric. Switch the Ginkgo subspace solver from `Cholesky` to `Lu` — the wrapper class adapter from Step 1.1 handles this via a constructor flag. The outer Krylov method also switches from PCG to GMRES (your `LinearSolverType` enum already supports this). The Petrides theorem bound is no longer rigorous for non-symmetric $A$, but empirically the AMGF preconditioner remains effective for the mild non-symmetry typical of non-associated plastic flow.

---

## 5. Estimated timeline

Assuming a single developer (you) working on this:

| Step | Duration | Cumulative |
|---|---|---|
| Step 0 + 0.5 + 0.7: pre-impl verification (MFEM API, FA gating, baseline regression) | 0.5–1 day | 0.5–1 day |
| Baseline measurement (§1) | 0.5 day | 1–1.5 days |
| Step 1.1: build Ginkgo into MFEM + adapter class | 1–2 days | 2–3.5 days |
| Steps 1.2–1.6 (Path A core implementation, including CMake additions at 1.6.5) | 3.5 days | 5.5–7 days |
| Steps 1.7–1.8 (Caliper, Path A validation) | 1 day | 6.5–8 days |
| Steps 2.1–2.5 (Path D core implementation) | 4 days | 10.5–12 days |
| Step 2.6 (RHS augmentation via SaddleResidualScaler — critical correctness step) | 1.5 days | 12–13.5 days |
| Step 2.7 ($\gamma$ sensitivity sweep) | 0.5 day | 12.5–14 days |
| Step 2.8 (trust-region interaction verification) | 0.5 day | 13–14.5 days |
| Step 2.10–2.11 (Caliper, full validation) | 1 day | 14–15.5 days |
| Phase 3 (production benchmarks, scaling, edge cases) | 5–10 days | 19–25.5 days |

So roughly **4–5 working weeks** for Phases 1+2+3 if everything goes smoothly. Add 50% for unexpected issues with Ginkgo integration, parallel matrix assembly subtleties, MFEM API quirks, or the FA-vs-EA architectural wrinkle from Step 0.5 — say **6–8 working weeks** as a realistic estimate.

Phase 4 (Path B) and Phase 5 (alternative AMG) are unlikely to be triggered if Phase 2 numbers come in as expected (30–80 iterations per Newton step). If triggered, each would add another 2–4 weeks.

The two most likely sources of unexpected delay are: (a) Ginkgo–MFEM build integration on first compile (CUDA / HIP / RAJA flag juggling); (b) Step 2.6 — if the `SaddleResidualScaler` wiring isn't exactly right, the symptoms are subtle (Path D solution drifts from Path A on hard problems) and the debugging is non-trivial.

---

## 6. What this guide does NOT cover

A few aspects of the original document deliberately not turned into action items here:

1. **Path B implementation** — Phase 4 — is conditional and the §3.6 $K^{-1}$-surrogate decision (option ii vs iii) is a design choice that should be made fresh when/if you actually need Path B, based on what you learn from Phases 1–3.
2. **Phase 5 alternatives** (PCBDDC, MueLu, GenEO) involve external dependencies and a substantially different code path; treat them as a separate project if ever needed.
3. **The sub-XYZ periodicity case** (already supported in the ExaConstit 5.9 development cycle via `MortarConstraintOperator::Reset(active_pair_labels, comp_mask)` and `MortarPbcManager::RebuildForActiveSpec`) requires verifying that the new `GetConstraintCoupledDofIndices()` and `GetCTransposeC()` caches correctly invalidate on `Reset` — a unit-test item rather than a separate implementation step.
4. **Outer Powell-Hestenes optimization** (Step 2.9) is a future-work item, not a Phase 2 task. It only becomes relevant if the inner Krylov tolerance is intentionally loosened.
5. **The hybrid EA-for-residual + FA-for-preconditioner mode** (§7.4 Option 2 of the analysis doc) is documented as future work. The initial implementation forces CPU/OpenMP runtime when AMGF is selected, via Step 0.5's validation.

---

## 7. Quick recap: the critical path

If you want to compress everything above into one paragraph:

> Verify the MFEM 4.9 `AMGFSolver` API in your local checkout. Add the `PreconditionerType::AMGF` and `AMGF_AUG_LAGRANGIAN` enum values and the validation rule that aborts cleanly when AMGF is requested without FULL assembly. Build Ginkgo into MFEM and write the `GinkgoDirectSubspaceSolver` adapter. Add `GetConstraintCoupledDofIndices()` and `GetCTransposeC()` to `MortarConstraintOperator`. Add the `BuildBooleanRestrictionProlongation` utility. Build `MortarSaddlePreconditionerAMGF` wrapping `mfem::AMGFSolver`, with a `use_path_d_` toggle. Wire it into `system_driver.cpp` and add the new files to CMake. With the toggle off (Path A), validate the AMGF infrastructure works correctly — solution unchanged, partial iteration-count drop confirms structure. Then flip the toggle on (Path D): assemble $K_\gamma = K + \gamma C^T C$ each Newton step, route AMGF to $K_\gamma$, replace the Schur block with $\gamma I$, and apply the RHS augmentation $r_K + \gamma C^T r_\lambda$ via a `SaddlePathDAugmenter` (a `SaddleResidualScaler` subclass wired into `ExaTrustRegionSolver::SetScaler`). Validate that converged $(u, \lambda)$ are identical to Path A's *and* that the trust-region trajectory is identical. Sensitivity-sweep $\gamma$. Production benchmark. Iteration count goes from your current 10k+ to 30–80. Done.

The rest is detail, but Step 2.6 (the RHS augmentation wired through `SaddleResidualScaler`, *not* via direct edits in `system_driver.cpp`) is the detail that, if gotten wrong, silently produces wrong answers. Double-check it.
