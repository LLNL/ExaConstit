# Mortar PBC Prototype: Status & Forward Plan

> **For the comprehensive theory + practice + 3D-extension document, see
> `docs/MORTAR_PBC_ARCHITECTURE.md`.** That is the all-guiding reference; this
> file is the shorter pre-Phase-3 status snapshot.

This document is the chat-restart summary for the mortar non-conforming
periodic-BC prototype.  It captures (1) what's done and verified,
(2) the architectural decisions locked in along the way, (3) traps
encountered (so we don't re-encounter them), and (4) the forward
plan with open design questions.

Last updated: end of Phase 2 (heterogeneous + checkerboard), 2D PASS on
np = 1, 2, 4, 8 in both layouts.

---

## Goal

Mortar-method non-conforming periodic boundary conditions for an RVE
solid mechanics problem.  Built first as a pyMFEM prototype, then ported
to MFEM C++ for integration into ExaConstit (LLNL crystal-plasticity
code, MFEM/RAJA, updated-Lagrangian, partial-assembly GPU).

Reference paper: Lopes, Ferreira, Andrade Pires (2021), CMAME 384,
113930.  Copy at `/mnt/user-data/uploads/1-s2_0-S004578252100267X-main.pdf`
in the original conversation environment.

---

## Status: what's done

### Phase 1: distributed Krylov saddle-point on linear elasticity

**1A: unpreconditioned distributed Krylov.**  GMRES + BlockOperator
formulation.  C represented as a Python Operator wrapping a scipy CSR;
the operator's `Mult`/`MultTranspose` do an Allgatherv of the input,
multiply by the (replicated) global CSR, and slice this rank's output.
K is consumed strictly via its operator interface — never gathered to
root, never converted to scipy CSR for the actual solve.

**1B: block-Jacobi preconditioner.**  Two diagonal blocks:
- `(0,0)` = `diag(K)^{-1}`, extracted via `Operator.AssembleDiagonal`
  (works uniformly on PA, EA, FA, HypreParMatrix forms).
- `(1,1)` = `diag(C diag(K)^{-1} C^T)^{-1}`, computed without ever
  forming the explicit C C^T product.  The C operator exposes a
  method `WeightedRowSqSum(weights, out)` that computes
  `out[i] = sum_j C[i,j]^2 * weights[j]` for owned rows; this is a
  collective (Allgatherv) call, parallel-safe.  The element-wise-squared
  C is cached at construction.

Wrapped as Python `_DiagonalScaler` operators (`y[i] = inv_diag[i]*x[i]`)
and assembled via `mfem.BlockDiagonalPreconditioner`.  Iteration counts
drop ~5x on the patch test.  Verified PASS at machine precision
(`||du||_inf ~ 5e-15`) on np = 1, 2, 4, 8.

### Phase 2: Newton on neo-Hookean

**2.1 (homogeneous neo-Hookean).**  Switched from BilinearForm K to
ParNonlinearForm.  Newton outer loop wrapping the saddle-point solver
as the linear inner step.  Verified Newton converges in 1 iteration on
the homogeneous patch (the linear deformation IS the exact solution and
the constraint reactions absorb all the imbalance — `u_tilde = 0` at
convergence).  PASS np = 1–8.

**2.2 (heterogeneous strip-split, 5× contrast).**  Vertical strip:
elements with `centroid_x < L/2` get attribute 1 (matrix, E = 70e3);
others get attribute 2 (stiff, E = 350e3).  `PWConstCoefficient(mu_vec)`
and `PWConstCoefficient(K_vec)` indexed by attribute, fed into
`NeoHookeanModel(mu_coef, K_coef)`.  Quadratic Newton convergence
observed:

```
iter 0:  1.07e+06
iter 1:  4.39e+05
iter 2:  7.03e+04
iter 3:  5.73e+03
iter 4:  3.75e+01
iter 5:  1.71e-03   (relative: 1.61e-09 — converged)
```

`||u_tilde||_inf = 8.04e-02` (non-trivial — the soft strip takes most
of the deformation).  PASS np = 1–8.

**2.4 (checkerboard, 5× contrast).**  Same machinery, four-quadrant
diagonal-pair layout.  Both periodic directions cross material
discontinuities; two intersecting internal interfaces.  Closest 2D
analogue to the 3D RVE case.  Driver: `examples/patch_test_2d_checkerboard.py`.

(Step 2.3, "100× contrast stress test," skipped for now — the design
is solid enough that a contrast-bumping test isn't required before
moving to 3D.  Easy to revisit if needed.)

---

## Architectural decisions (locked)

These are deliberate calls made during Phase 1/2; revisiting them needs
explicit justification, not casual drift.

1. **UT (uniform traction) deferred but not blocked.**  ConstraintAssembler
   ABC + `stack_constraints` helper exists.  Mortar PBC is the first
   instantiation; UT can plug in later as another `ConstraintAssembler`
   subclass.

2. **K-block consumed as `mfem::Operator` only.**  Never `tocsr()`,
   never RAP, never gathered for the actual solve.  This is the
   GPU-portability requirement: PA-K must work without ever materializing
   a CSR.  Block-Jacobi prec uses only `AssembleDiagonal`.

3. **Krylov runtime-selectable.**  MINRES (default for symmetric K),
   GMRES (non-symmetric K), BiCGStab.  CG explicitly rejected (saddle-point
   system is indefinite; CG diverges).

4. **`SaddlePointSolver` is a mirror of `mfem::SchurConstrainedSolver`
   but with operator-only K.**  Current MFEM `constraints.hpp`
   implementations (`SchurConstrainedHypreSolver`, `EliminationCGSolver`,
   `PenaltyConstrainedSolver`) all require an assembled HypreParMatrix
   K and use HypreBoomerAMG.  Not GPU-friendly for PA-K.  Our class
   inherits the same external API (matches the ABC) but takes K as a
   plain `Operator` and uses block-Jacobi prec.  This is a candidate
   upstream contribution to MFEM: a fourth `ConstrainedSolver` variant
   for matrix-free K.

5. **Solve-step API uses pre-assembled Newton residuals.**  After a
   sign-bug class encountered around the C^T λ contribution to the top
   RHS, refactored to take `(r1_local, r2_local)` directly — the caller
   assembles the FULL Newton residuals (including the `+ C^T λ_k`
   contribution).  Solver simply negates them.  Eliminates sign-error
   class entirely.

6. **`SetIterativeMode(False)` on the inner Krylov solver.**  Newton's
   outer loop warm-starts at the OUTER level via `u_tilde` and `λ` —
   those carry information across iterations correctly because they're
   the actual unknowns.  The inner linear solve is for the INCREMENTAL
   update `(du, dλ)`; the previous step's `du` has no relevance to the
   current step's, so inner warm-starting is a category error.  Especially
   important for CG (Lanczos breakdowns); also defensively correct for
   GMRES.

7. **Tribol deferred until working version exists.**  We're not relying
   on Tribol's mortar implementation; we built our own to learn the
   mortar machinery + own the integration into ExaConstit's PA path.

8. **SciPy direct solver quarantined to verification path only.**  Lives
   in `mortar_pbc/_verify_solver.py`.  Not exported from package.  Used
   only as cross-check for the Krylov path.  Production solve always
   goes through `SaddlePointSolver`.

9. **Newton convergence: relative force-balance + absolute constraint
   + stagnation detection.**  Three criteria:
   - `||F_int + C^T λ||_2 < max(rtol * r0, atol)` (relative, with
     absolute floor; `r0` = iter-0 residual norm).
   - `||C u_tilde||_2 < atol_constraint` (absolute, constraint residual
     is dimensionless).
   - `||du||_2 < du_floor` (stagnation: linear solver can't improve
     further; declare converged).

10. **C++ build exposes all three MFEM ConstrainedSolver classes for
    optional cross-check** (Schur/Elim/Penalty) — confirmed available
    in pyMFEM build.

---

## Critical lessons (the trap list)

These came up the hard way.  Worth keeping forefront.

1. **Every collective must run on every rank.**  No rank-0-only or
   `n_lam_local > 0` guards around `C_op.Mult`, `CT_op.Mult`,
   `WeightedRowSqSum`, `comm.allreduce`, `nlf.Mult`, `nlf.GetGradient`,
   `BoundaryClassifier2D` construction, etc.  Local guards only wrap
   purely local computation (sentinel checks, negation loops over a
   per-rank slice).

2. **`BoundaryClassifier2D` collective construction must precede any
   rank-0-only prints** to avoid asymmetric collective entry causing
   deadlocks.

3. **Element-wise `vec[i] = float(...)` writes are robust against
   pyMFEM `GetDataArray` view-vs-copy ambiguity.**  On some pyMFEM builds
   `GetDataArray()` returns a view; on others it's a copy.  Element-wise
   assignment via `__setitem__` always works correctly.

4. **`nlf.GetGradient` returns `mfem::Operator&` (base class).**  The
   dynamic type is normally `HypreParMatrix`, but pyMFEM exposes only
   the base.  For verification gather paths, attempt `mfem.Opr2HypreParMat`
   downcast if exposed; else duck-type-check `hasattr(op, "MergeDiagAndOffd")`;
   else gracefully skip the SciPy-direct verify path.  Newton convergence
   itself doesn't depend on this.

5. **`ParNonlinearForm` handles essential DOFs internally.**  Once
   `nlf.SetEssentialTrueDofs(ess_tdof_list)` is called:
   - `nlf.Mult(x, residual)` returns residual with essential DOFs
     already zeroed.
   - `nlf.GetGradient(x)` returns tangent with essential rows/cols
     already eliminated.
   Calling our own `apply_dirichlet_to_distributed_K` on the result
   would corrupt K (double-elimination).  Only the LINEAR-elastic
   driver (`patch_test_2d.py`) uses the manual path; the nonlinear
   drivers MUST NOT.

6. **The Newton residual MUST include the `C^T λ_k` contribution.**
   `||F_int||_2` alone stagnates at the natural force scale of the
   problem (~2.7e5 for our case, same as iter 0) regardless of how
   converged the actual equilibrium is.  The quantity that goes to
   zero at equilibrium is `||F_int + C^T λ||_2`.  Iter 0 has λ=0 so
   the term is zero; iter 1+ must add `C^T λ_k` before the convergence
   check AND pass the augmented residual to `solve_step`.

7. **Verification gather block must mirror the in-loop residual
   construction.**  After Newton converges, the post-loop verify path
   recomputes `nlf.Mult(x, final_residual)` (giving F_int alone) and
   gathers it.  Without re-adding `C^T λ`, the gathered residual is
   the natural-scale F_int (~1e5) rather than the converged residual
   (~1e-9 relative).  Easy bug to miss because Newton trace looked
   right; only the verification panel showed the wrong number.

8. **Absolute Newton tolerance ignores problem scale.**  For Lamé
   modulus O(1e4) and natural force O(1e5), an `atol = 1e-10` is
   physically meaningless — orders of magnitude below floating-point
   noise floor at this problem scale.  Use relative drop from `r0`
   with absolute floor as safety net for trivially-tiny problems.

9. **Krylov stagnation when the linear solve has nothing to do.**
   When Newton has already converged on a previous iteration but the
   outer loop hasn't recognized it yet, the next Krylov call sees a
   tiny RHS, exits with 0 iterations, returns du=0.  Without
   stagnation detection in the Newton outer loop, this loops to
   max_iter pretending Newton failed.  Always include `||du|| < floor`
   as a convergence path.

10. **Pointer/lifetime conventions in pyMFEM.**  `BlockDiagonalPreconditioner`
    does NOT own its diagonal blocks.  Python GC will collect them
    mid-Krylov-solve unless explicit references are kept alive in
    a list outside the function scope.  `SaddlePointSolver._build_block_jacobi_prec`
    returns a `keepalive` list specifically for this; the caller stashes
    it on `self._last_prec_refs`.

---

## Warm-start commentary (for future multi-load-step driver)

ExaConstit handles BC changes between time steps via `SystemDriver::SolveInit`
(`src/system_driver.cpp:441-478`).  The motivation, captured in
ExaConstit issue #8 (github.com/llnl/ExaConstit/issues/8):

The constrained DOFs (the essential boundary) are NOT being warm-started
in any approximate sense — they're set EXACTLY to their prescribed
values for step `n+1`.  The issue is the **unconstrained DOFs**: at the
start of step `n+1`, their previous-step values `v_u^n` are no longer
in equilibrium with the new boundary values `v_c^{n+1}`, and starting
Newton from `(v_u^n, v_c^{n+1})` injects a large artificial residual at
the first Newton iterate.  For severe BC changes, this can put Newton's
first iterate into a bad region (e.g. `J < 0` for hyperelastic).

The SolveInit projection works as follows:

```
Step 1 (warm-start projection, before Newton):
  1a. K_n  := tangent stiffness from previous converged state.
  1b. ΔR_u := -K_{uc} (v_c^{n+1} - v_c^{n})
              The change in residual at unconstrained DOFs caused by the
              change in CONSTRAINED-DOF values from step n to n+1.
              K_{uc} is the sub-matrix coupling unconstrained rows to
              constrained columns.
  1c. Solve  K_n Δv^{n+1} = -(R^n + ΔR_u)   for Δv.
              R^n is the previous step's residual (zero at converged
              state; non-zero if step n didn't fully converge —
              captured here).
  1d. Initial guess for Newton: v^{n+1}_initial = v^n + Δv^{n+1}.
              The unconstrained DOFs now have a sensible starting value
              that reflects the BC change linearly through the
              previous-step tangent.

Step 2 (Newton solve, as normal):
  2a. Apply v_c^{n+1} EXACTLY to the constrained DOFs.
  2b. Run Newton from v^{n+1}_initial.
```

ExaConstit's primal field is **velocity**, and the prescribed velocity
gradient changes every load step — so without SolveInit, every step
starts Newton from a state that's non-equilibrium at the unconstrained
DOFs because the constrained values just jumped.

**For our PBC mortar formulation:** the unknown is `u_tilde` (the
periodic fluctuation), and `u_tilde`'s essential BCs are the corner
Dirichlets fixed at zero — these don't change between load steps.
What changes is `u_lin = (F_macro - I) Y`, added to `u_tilde` to form
the total state.  The SolveInit equivalent for our setup would be:

```
Δu_lin       := u_lin^{n+1} - u_lin^{n}
ΔR_unconstr  := -K_{uc} Δu_lin       (NOT -K_{uc}(v_c^{n+1} - v_c^{n});
                                       our "constrained values" of u_tilde
                                       are zero at corners and don't change.
                                       But the LINEAR PART u_lin DOES change,
                                       and that's the analogue here.)
Solve  K Δu_tilde = -(R^n + ΔR_unconstr)
u_tilde^{n+1}_initial = u_tilde^n + Δu_tilde
```

So we DO need a SolveInit equivalent for multi-load-step F_macro
ramping — it's just expressed in terms of `u_lin` change rather than
constrained-DOF value change.  This wasn't relevant in single-step
testing (Phases 1–2) because we only had one load step: cold-start
`u_tilde = 0` and let Newton converge.  For Phase 6+ multi-step
loading, this projection becomes mandatory.

**Where this becomes additionally relevant beyond F_macro ramping:**
- Velocity-based primal formulation (rate-dependent crystal plasticity)
  follows ExaConstit's setup directly — `v_c` is the prescribed
  velocity at each step and SolveInit applies as written.
- Prescribed displacements on boundaries beyond the corner Dirichlets
  (e.g. displacement-controlled loading on an entire edge) — same
  thing, with `u_c^{n+1} - u_c^n` driving the projection.

Both are post-port concerns.  Recommendation: when we get to Phase 6
multi-step driver, port ExaConstit's SolveInit pattern (it's a single
linear solve, cheap), generalized to also handle the `Δu_lin` case.

---

## Code layout

```
mortar_pbc_proto/
├── mortar_pbc/                 # the package
│   ├── __init__.py             # exports public API
│   ├── types_2d.py             # EdgeNodes2D, CornerInfo dataclasses
│   ├── boundary_2d.py          # BoundaryClassifier2D (with DofToVDof fix)
│   ├── mortar_2d.py            # N_line2, M_line2_dual, MortarBlock2D,
│   │                              MortarAssembler2D
│   ├── constraint_builder.py   # ConstraintBuilder2D — scipy CSR build
│   ├── constraint_assembler.py # ABC + MortarPbcConstraintAssembler +
│   │                              stack_constraints helper
│   ├── saddle_point.py         # SaddlePointSolver (Krylov + block-Jacobi
│   │                              prec); make_constraint_operators
│   │                              factory; _DiagonalScaler helper
│   └── _verify_solver.py       # SciPyDirectSolver (quarantined)
├── examples/
│   ├── patch_test_2d.py                  # Phase 1B regression baseline
│   │                                       (linear elastic, single solve)
│   ├── patch_test_2d_heterogeneous.py    # Step 2.2: strip-split, 5x
│   └── patch_test_2d_checkerboard.py     # Step 2.4: 4-quadrant, 5x
└── tests/
    └── test_mortar_2d_unit.py            # 5 unit tests:
                                              dual basis bi-orthogonality,
                                              partition of unity,
                                              conforming pair lumping,
                                              non-conforming linear-field
                                              reproduction,
                                              ConstraintAssembler ABC +
                                              stack_constraints
```

---

## Forward plan

### Phase 3: 3D mortar (next major work)

**Wirebasket structure.**  3D RVE has:
- 8 corners — must be Dirichlet-pinned (3 components each → 24 TDOFs).
- 12 edge wirebaskets — periodic in their direction; 4 wirebaskets per
  spatial direction, each pairing 4 edges.
- 6 face pairs — periodic; 3 pairs (one per spatial direction).

Each face pair has the same kind of mortar coupling we built for 2D
edges, but on 2D surface integrals over face geometry.  Each edge
wirebasket couples 4 line edges (not 2), and the corner constraint
involves 8 corners, not 4.

**Polygon clipping for 2D segmentation pieces.**  When the non-mortar
face's elements aren't aligned with the mortar face's, each pair of
overlapping element faces must be intersected to form a polygon, then
quadrature is built on this polygon.  Robust polygon clipping in 3D is
non-trivial; Sutherland-Hodgman or similar.

**Triangular vs quadrilateral non-mortar elements.**  For our
extruded-quad-on-quad ExaConstit meshes, both faces are quads.  But
we should design for general — the Lopes paper covers triangular
non-mortar elements too (Appendix C).

**Dual basis modifications.**  Lopes Eq. C.1 gives the line-2 (1D)
dual basis.  For 3D faces, we need the 2D analogue — Wohlmuth's
biorthogonal basis on quad and triangle reference elements.  The
corner+edge wirebasket modifications (Wohlmuth) are subtle: dual
basis functions near corners need correction terms to maintain
biorthogonality across the geometric singularities.

**Open Phase 3 design questions:**

1. **Constraint storage layout.**  In 2D, C is replicated on every
   rank (28x162, only 92 nnz; cheap).  In 3D with O(10K) face pairs and
   O(100) wirebasket constraints per direction, replicated C is no
   longer free.  Options:
   (a) Distribute C — owned-row partitioning matching face-element
       distribution.  Mult/MultTranspose become more complex.
   (b) Replicate per constraint group (faces, edges, corners
       separately), block-diagonalized.
   (c) Stay replicated and just accept the memory cost (probably
       fine through 100K elements).
   
   Recommend starting with (c) and migrating to (a) only if memory
   becomes a real bottleneck.

2. **Reference vs spatial configuration for mortar integration.**  In
   updated Lagrangian, the reference mesh and spatial mesh differ.
   Mortar integrals can be evaluated on either.  Lopes uses reference
   (the formulation is reference-Lagrangian).  ExaConstit is updated
   Lagrangian — at each load step, reference resets.  This matches the
   reference-mortar convention naturally; just rebuild C at each load
   step's reset.

3. **Dual basis integration order.**  The Wohlmuth-modified dual basis
   has discontinuities along corner/edge boundaries.  Quadrature must
   be subdivided at these discontinuities.  Tricky; need to think
   through the subdivision logic before coding.

### Phase 4: MPI for 3D

Same template as 2D — operators wrap distributed CSRs; collective
correctness baked into every Mult.  Bigger Allgatherv volumes; might
push us into "distributed C" sooner than just memory-driven.

### Phase 5: C++ port to ExaConstit

**Class design.**  `MortarPbcSchurSolver` (or similar) inherits from
`mfem::ConstrainedSolver`, mirroring the existing
`SchurConstrainedHypreSolver` API but with operator-only K and
block-Jacobi prec.  The ConstraintAssembler ABC pattern carries over
to C++ as a virtual interface; mortar-PBC is one implementation,
UT will be another, and Tribol-based contact would be a third.

**Possible upstream MFEM contribution.**  MFEM's existing
`mfem::ConstrainedSolver` family doesn't have a matrix-free / PA-friendly
variant.  Our `MortarPbcSchurSolver` IS that variant.  After ExaConstit
integration is solid, propose upstream as a new ConstrainedSolver
subclass.  Reference: `mfem/linalg/constraints.hpp` for the existing
ABC and three implementations.

**Hooks to existing ExaConstit infrastructure:**
- `SystemDriver::SolveInit` — warm-start path; needs extension to handle
  PBC if/when we add prescribed displacements beyond corner Dirichlets.
- `BCManager` — currently handles essential BCs by attribute; PBC is
  a different beast (constraint-based, not essential-BC-based).  May
  need a new manager class or a generalized `ConstraintManager`.
- `mech_operator` — the ParNonlinearForm equivalent.  Wires into our
  saddle-point solver as the K-operator source.

**What's NOT going to MFEM upstream.**  The mortar assembly itself
(`MortarAssembler2D` and friends).  That's domain-specific to our PBC
setup; lives in ExaConstit.  Upstream contribution is the
`ConstrainedSolver` subclass only.

### Phase 6+: extensions (post-port)

- **Multi-load-step driver** with proper warm-start handling.
- **Velocity-based primal formulation** (rate-dependent constitutive
  models need this; SolveInit-style projection at each step).
- **Tribol integration** as a third `ConstraintAssembler` for contact
  problems.
- **Uniform traction (UT) BCs** as a second `ConstraintAssembler` —
  the ABC was designed with UT in mind from the start.

---

## Open questions before resuming

1. **Should we run the 100× contrast stress test before moving to 3D?**
   (Step 2.3, deferred.)  Cheap to do; would add confidence that
   Newton + block-Jacobi prec hold up under aggressive contrast.

2. **Phase 3 Q1: distributed vs replicated C in 3D?**  Recommendation
   above is "start replicated, migrate if needed."  Confirm before
   starting.

3. **Phase 3 Q2: which 3D mesh source?**  pyMFEM has `MakeCartesian3D`
   for the prototype.  For meaningful non-conforming tests, we need
   meshes whose face pairs really don't match — need to either build
   them by hand or extend `build_nonconforming_square` to a
   `build_nonconforming_cube` analog.

4. **Polygon clipping library or hand-roll?**  Sutherland-Hodgman is
   simple enough to hand-roll for convex-on-convex (which is our case
   for quad-on-quad face pairs).  shapely has it but is a heavy
   dependency.  Recommend hand-rolling.

---

## Run reference (validated as of last session)

All on np = 1, 2, 4, 8 — PASS in every case.

```
python examples/patch_test_2d.py                    # Phase 1B regression
python examples/patch_test_2d_heterogeneous.py      # Step 2.2 strip-split
python examples/patch_test_2d_checkerboard.py       # Step 2.4 checkerboard

python tests/test_mortar_2d_unit.py                 # 5 unit tests
```

---

## Environment

- pyMFEM commit 7e99b925, MFEM 4.9, conda-forge openmpi
- Python 3.9, conda env `mortar-pbc`
- macOS, `MACOSX_DEPLOYMENT_TARGET=11.0`
- Build: `pip install ./ -C"with-parallel=Yes" --verbose` (from PyMFEM
  source)

pyMFEM exposed (verified in use):
- `PyOperatorBase`, `BlockOperator`, `BlockDiagonalPreconditioner`
- `MINRESSolver`, `GMRESSolver`, `BiCGSTABSolver` (no CG — see note)
- `ParNonlinearForm`, `HyperelasticNLFIntegrator`,
  `NeoHookeanModel(mu_coef, K_coef)`
- `SchurConstrainedHypreSolver`, `EliminationCGSolver`,
  `PenaltyConstrainedSolver` (all three available; not currently used
  except as design reference)
- `ToScipyCSR`, `ToHypreParCSR`, `Opr2HypreParMat` (the last is the
  Operator → HypreParMatrix downcast helper)
- `PWConstCoefficient(mfem.Vector)` for per-attribute material
- `intArray`, `Array` various utility types

---

End of project status.  When resuming, start by re-reading this file
and verifying the runs above still pass.  Pick from "Open questions"
or proceed directly to Phase 3 planning.
