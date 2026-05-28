# Phase 5 (v7) — ExaConstit Integration: Mortar PBC into Production Solver

> **This document is v7 and supersedes v6 in its entirety.** v7 adds the
> Phase 5.11 implementation batch (saddle-system residual scaling) that
> shipped after v6 was written, plus two new hazard entries in §P5.14
> for traps surfaced during that work. The substantive additions:
>
> 1. **New §P5.19 content section** documenting the saddle residual
>    scaling stack end-to-end: the asymmetric block-diagonal scaling
>    formulation, Rule-A unit-balance selection with floor/cap guards,
>    the `FaceEdge` / `PerPair` sub-block partition choice, the three
>    operator/solver/preconditioner wrappers, NR/NRLS/TRDOG integration,
>    the per-Newton-iter diagnostic logger, the `InspectingIterativeSolver`
>    post-solve telemetry hook, and the two-run-diff workflow for
>    wrapper-transparency validation.
> 2. **Phase 5.11 entry added to §P5.13 phasing** with sub-batches A
>    through K covering the actual delivery sequence (intermediate
>    hot-fix bundles and superseded dead-end identity tests are not
>    surfaced in the phasing — only the conceptual progression is).
> 3. **Two new §P5.14 hazards** (§P5.14.15 and §P5.14.16):
>    `iterative_mode` flag forwarding in `mfem::Solver` wrappers (a
>    classic non-determinism trap), and `SetSolver` overload selection
>    when wiring shared_ptr vs reference (caused the inspector wrap to
>    silently bypass the scaled-solver chain during 5.11.K integration).
>
> No content from v6 was removed. All §P5.1–§P5.18 sections, the
> §P5.13 Phase 5.0–5.10 phasing, and the §P5.14.1–§P5.14.14 hazard
> entries survive unchanged.
>
> ---
>
> **Previous version notes (v6 inherits v5 preamble; v5 inherits v4):**
>
> **This document is v6 and supersedes v5 in its entirety.** v6 inlines
> the §P5.18 component-restricted-PBC content section, the new Phase 5.9
> phasing batch, and the four new §P5.14 hazard entries that were
> originally delivered as a separate addendum after Phase 5.9
> (component-restricted PBC) shipped and passed validation. **The
> meaningful renumbering**: the v5 doc reserved "Phase 5.9" for the
> performance / GPU / documentation push that closes the Phase 5
> programme of work; in v6 the component-restricted PBC work (which
> was the actual next implementation batch and is what the source-code
> comments labeled "Phase 5.9 / Batch A.X" during development) takes
> the **Phase 5.9** slot, and the perf/GPU/docs push is renumbered to
> **Phase 5.10**. Code comments and planning doc now agree.
>
> **What v6 adds:**
>
> 1. **New §P5.18 content section** documenting the spec-driven
>    constraint-filter machinery end-to-end: the `PeriodicBC` TOML
>    interface, the two-axis (pair + component) filter, the EA operator
>    row layout under filter, the manager-level rebuild orchestration,
>    spec-aware corner essential TDOFs, the SystemDriver per-step hook,
>    the stale-cache refresh cascade (with the X-only sizing example
>    that caught it), TOML examples, test coverage, backward-compat
>    invariants, and future-work pointers.
> 2. **§P5.13 phasing reshuffled**: Phase 5.9 is now "Component-
>    restricted PBC (spec-driven constraint filter)" with nine
>    sub-batches A through I. The original Phase 5.9 ("Performance
>    benchmarks + GPU validation + documentation") becomes Phase 5.10
>    with the same four sub-batches A through D.
> 3. **Four new §P5.14 hazards** (§P5.14.11 through §P5.14.14):
>    stale size caches across `MortarSaddlePointSystem` / `m_x_saddle`,
>    the SystemDriver mortar ctor ordering invariant, rotation RBMs
>    under sub-XYZ specs, and the collective-by-convention scope of
>    `Reset`.
> 4. **§P5.6.1 lifecycle code block updated** to inject the
>    `SyncMortarPbcForStep(ti)` call inside the `GetUpdateStep`
>    transition block, with a §P5.18.6 cross-reference.
>
> No content from v5 was removed. The §P5.8 derivation, the
> classifier-as-cache argument, all existing §P5.4–§P5.7 manager and
> SystemDriver descriptions, and the Phase 5.0–5.8 phasing all
> survive unchanged.
>
> ---
>
> **Previous version notes (v5 inherits v4 preamble; v4 supersedes v3):**
>
> **v4 applies the 13 edits identified during the
> code-review pass against the actual Phase 4 classifier internals
> (`PHASE5_v3_code_review.md`). v3's conceptual core — the §P5.8 UL
> derivation, the build-once-on-reference prescription, the
> classifier-as-cache argument — survived review intact. v4 fixes
> implementation details and resolves one design ambiguity v3 had left
> open. Changes from v3:
>
> **Substantive (v4)**
>
> 1. **Method D explicit** (§P5.8.4 + new §P5.8.4.4). v3 implicitly
>    assumed Method-D semantics (iterate on total $v$, constraint
>    $Cv = g$ with non-zero $g$) but didn't say so against the alternative
>    (Method C — fluctuation primal, homogeneous constraint). v4 picks
>    Method D explicitly, justifies the choice, and documents Method C as
>    the fallback if the chosen path runs into trouble.
> 2. **Phase 5.0 batch added** (§P5.13). New first batch:
>    `MortarSaddlePointSystem` constraint-RHS extension. Phase 4.3
>    modification (~10 LOC) — adds `SetConstraintRHS(const Vector& g)` /
>    `ClearConstraintRHS()` methods, modifies `Mult` to subtract `g` from
>    the constraint-side residual when set. Default behavior unchanged so
>    existing Phase 4.3 tests don't regress. Prerequisite to Phase 5.3
>    (manager construction).
>
> **Interface alignment (v4)**
>
> 3. **K closures passed to manager constructor** (§P5.4.1, §P5.5.3).
>    v3's `SetKResidual` / `SetKJacobian` setters don't exist on
>    `MortarSaddlePointSystem`; the actual API takes them at construction.
>    `MortarPbcManager`'s constructor now takes the closures and forwards
>    to the saddle system.
> 4. **`SetConstraintRHS` instead of `GetConstraintRHS`** (§P5.4.1, §P5.4.4).
>    The saddle system doesn't expose its RHS for the manager to mutate;
>    after the Phase 5.0 extension, the manager calls `SetConstraintRHS(g)`.
> 5. **Saddle-point enums use Phase 4.3 `KrylovType` / `SaddlePrecType`
>    directly** (§P5.3.4) with a translation note explaining the option-parser
>    boundary.
> 6. **`UpdateMacroscopicF` takes `const mfem::Vector&`, not `DenseMatrix`**
>    (§P5.4.1, §P5.4.4). Matches BCManager's flat 9-vector convention.
> 7. **`m_saddle_solver` renamed to `m_saddle_solver_for_init`** (§P5.4.1,
>    §P5.4.3). Distinguishes the direct linear solver (used in `SolveInit`)
>    from the Newton-loop saddle system.
> 8. **`m_C_op` "refreshed per step" comment removed** (§P5.4.1). It's
>    built once.
>
> **Implementation notes (v4)**
>
> 9. **`ProjectVelocityGradientToCornerTDofs` factoring note added**
>    (§P5.5.4) — describes the small refactor needed to reuse existing
>    `essential_vel_grad` projection logic on a TDOF subset.
> 10. **`ComputeMacroscopicP` LOC estimate added** (§P5.10.3) — Lopes
>     eq. 37 implementation is ~30 LOC of new code in `MortarPbcManager`.
>
> **New traps (v4)**
>
> 11. **§P5.14.8** — saddle-point residual must use the set constraint
>     RHS, not the homogeneous default. Failure mode: forgetting to
>     call `SetConstraintRHS(g)` after `UpdateConstraintRHS()` makes
>     Newton converge to the wrong fluctuation.
> 12. **§P5.14.9** — refactoring the corner-pin projection may regress
>     non-mortar tests if `UpdateVelocity` is touched too aggressively.
>
> **Open questions deferred to implementation** (from code-review §RR.5)
>
> - Does `SaddlePointSolver` have a generic `Operator&` overload for K?
>   (Phase 5.6 SolveInit needs it; verify when implementing.)
> - Hill-Mandel diagnostic implementation feasibility from $\lambda$ +
>   classifier data. (Phase 5.8 work.)
> - Restart format support for $\bar F^{(n)}$. (Phase 5.7 / Phase 5.9.)
> - Whether `essential_vel_grad` projection can be factored without
>   touching `UpdateVelocity` significantly. (Phase 5.5 work; trap §P5.14.9
>   covers the regression risk.)
>
> **Diff vs v2** (kept here for context across the v2 → v3 → v4 chain):
> v2's per-step `RebuildConstraintBlocks` machinery is gone; the constraint
> matrix is built once and never rebuilt. v3's §P5.8 rewrite established
> this; v4 inherits it without change. The Hill-Mandel diagnostic added in
> v3 stays. Phasing went from 9 batches in v2 → 9 in v3 → 10 in v4 (Phase
> 5.0 added).
>
> **Companion documents**: `MORTAR_PBC_ARCHITECTURE.md`,
> `PHASE4_CPP_PORT_PLAN.md`, `PHASE6_HIGHER_ORDER_LOR.md`.
> **Cross-references**: §X.Y (architecture doc), §P4.X.Y (Phase 4 plan),
> §P5.X.Y (this doc), §P6.X.Y (Phase 6).
>
> **Loading this document into a fresh conversation**: this v3 is sufficient
> context to resume the integration from any phase boundary. Pair with
> `MORTAR_PBC_ARCHITECTURE.md` and `PHASE4_CPP_PORT_PLAN.md` for the
> upstream context.

---

## §P5.1 Goals and non-goals

### Goals

1. **Promote `test/mortar_pbc/` to `src/mortar_pbc/`.** After Phase 5 the
   user enables mortar PBC by (a) setting `mesh.periodicity = true` in the
   TOML and (b) supplying a `velocity_gradient` BC; everything else routes
   through existing ExaConstit pathways.
2. **Velocity-primal in updated Lagrangian.** ExaConstit's primal is $v$
   on the current configuration $\Omega^{(n)}$. The mortar PBC constraint
   $C\,v = g^{(n)}$ has $C$ built once on the reference (undeformed) face
   geometry and an RHS $g^{(n)}$ updated each step from current $\bar L^{(n)}$
   and tracked $\bar F^{(n)}$. See §P5.8 for the rigorous derivation.
3. **Compatible with NR / NRLS / TRDOG; PA / EA / FULL; CPU / GPU; linear
   elastic, ExaCMech, UMAT; multi-region heterogeneous meshes.**
4. **Zero regression when mortar is disabled.** The default code path is
   bit-identical to pre-Phase-5 ExaConstit.
5. **Reuse `MortarSaddlePointSystem`** (Phase 4.3 Batch R) so the existing
   `ExaNewtonSolver` family receives a saddle-point Operator and a
   `BlockOperator` Jacobian without per-solver modification.
6. **Reuse `BCManager` and `essential_vel_grad`** for the corner pin; the
   only change is restricting the projection from "all TDOFs on tagged
   faces" to "8 corner TDOFs from those faces."

### Non-goals (deferred)

- $p \ge 2$ primal: see Phase 6 (LOR machinery).
- Semi-periodic (mixed periodic + Dirichlet boundaries): API designed to
  accept it cleanly; not Day 1.
- UT BCs as a second `ConstraintAssembler`: long-term roadmap.
- FE² coupling: downstream user.
- Non-axis-aligned RVEs (curvilinear / hexagonal). See §P5.8.9 Gotcha 8 for
  the standard fallback if this becomes a need (consistent biorthogonalization
  on reference geometry — Phase 5+ infrastructure).

---

## §P5.2 Architectural overview

Five points of contact with existing code, no new top-level table:

```
┌────────────────────────────────────────────────────────────────────┐
│ TOML — distributed across existing tables                          │
│   [Mesh].periodicity, [Mesh].snap_tol, [Mesh].lor_depth (Phase 6)  │
│   [BCs] essential_vel_grad — already exists                        │
│   [Solvers.SaddlePoint] — new sibling of [Solvers.Krylov]          │
└────────────────────────────────────┬───────────────────────────────┘
                                     │
              ┌──────────────────────┴────────────────────┐
              ▼                                            ▼
   ┌─────────────────────┐               ┌──────────────────────────────┐
   │ BCManager (existing)│               │ SystemDriver (modified)      │
   │   single source for │               │   m_mortar_pbc                │
   │   essential_vel_grad│               │   detects mortar via          │
   │   data              │               │     mesh.periodicity          │
   └──────────┬──────────┘               │   per-step F̄ update           │
              │                          │   essential-DOF projection    │
              │  (vgrad, comp, ids)      └──────────────────┬───────────┘
              ▼                                             │
   ┌──────────────────────┐                                 ▼
   │ MortarPbcManager     │◀───────────────┐  ┌──────────────────────────────┐
   │ (NEW — owns          │                │  │ ExaNewton(LS|TR)Solver        │
   │   classifier,        │                │  │   ALMOST UNCHANGED.           │
   │   builder,           │                │  │   Receives a different        │
   │   constraint op,     │                │  │   Operator (the wrapping      │
   │   MortarSaddlePoint- │                │  │   MortarSaddlePointSystem);   │
   │   System adapter,    │                │  │   GetGradient returns a       │
   │   F̄^(n) state)       │                │  │   BlockOperator. Krylov on    │
   │                      │                │  │   the inner block uses the    │
   │   Provides:          │                │  │   user-chosen MINRES/GMRES/   │
   │   - corner ess TDOFs │                │  │   BiCGStab from               │
   │   - F̄^(n) update     │                │  │   [Solvers.SaddlePoint]       │
   │   - λ warm-start     │                │  └───────────────────────────────┘
   │   - C built once,    │                │
   │     never rebuilt    │                │
   └──────────────────────┘                │
                                           │
                            ┌──────────────┘
                            │  saddle-point Operator
                            │  (constructed once)
                            ▼
```

The new artifact is `MortarPbcManager`. Everything else is wiring. The
manager's lifecycle is simple:
- **Constructed once** at `SystemDriver` initialization (after
  `mech_operator`).
- **Per step**: `UpdateMacroscopicF()` advances $\bar F^{(n)}$ via
  $\bar F^{(n+1)} = \bar F^{(n)} + \bar L^{(n+1)} \bar F^{(n)} \Delta t$;
  the saddle-point RHS is recomputed.
- **Per Newton iteration**: nothing — the matrix $C$ is constant; only the
  bulk equilibrium and the $\lambda$ update happen.

---

## §P5.3 Configuration: distributed across existing TOML tables

No new `[BCs.MortarPBC]` table. Five additions, each in the table where the
parameter conceptually belongs.

### §P5.3.1 `[Mesh]` additions

```toml
[Mesh]
filename = "polycrystal.mesh"
order = 1

# Existing flag — promoted to "enable periodic BCs of any flavour".
# When true and a velocity_gradient BC is present, mortar PBC is on.
periodicity = true

# NEW: snap-coordinate tolerance for face/edge/corner identification
# in BoundaryClassifier3D. Defaults safely; users typically don't touch.
snap_tol = 1.0e-10

# NEW (Phase 6 stub): LOR refinement depth on the periodic-face
# ParSubMesh. depth = 1 means no LOR (direct mortar). depth > 1 enables
# higher-order primal via LOR projection. Phase 5 enforces depth = 1;
# Phase 6 lifts the restriction.
lor_depth = 1
```

### §P5.3.2 `[BCs]` — no schema change

The user supplies the macroscopic deformation rate via the existing
`essential_vel_grad` BC. Same TOML schema as the standard non-mortar path:

```toml
[BCs]
expt_mono_def_flag = false   # existing; unrelated

[[BCs.essential_vel_grad]]
essential_ids   = [1, 2, 3, 4, 5, 6]   # all 6 face attributes
essential_comps = [-1, -1, -1, -1, -1, -1]  # xyz on each
velocity_gradient = [
  [0.001,  0.0, 0.0],
  [0.0,   -0.0005, 0.0],
  [0.0,    0.0,   -0.0005]
]
origin = [0.0, 0.0, 0.0]
```

When mortar PBC is enabled, `MortarPbcManager` consumes the
`essential_vel_grad` data the same way `BCManager` does for non-mortar
case — but applies it only to the corner TDOF subset. See §P5.5.

For semi-periodic (future): the user lists only the *periodic* face
attributes here, with a separate `essential_velocity` block for the
conventionally-Dirichlet faces. The mortar machinery operates on the
`essential_vel_grad`-tagged set; the standard Dirichlet machinery operates
on the rest. Composes naturally because the TDOF sets are disjoint.

### §P5.3.3 `[Solvers.SaddlePoint]` — new sibling under [Solvers]

```toml
[Solvers]
assembly = "ea"

[Solvers.Krylov]            # existing — used for K linearization
linear_solver = "GMRES"
rel_tol = 1.0e-12
abs_tol = 1.0e-30

[Solvers.SaddlePoint]       # NEW — used for the saddle-point Krylov
linear_solver = "MINRES"    # MINRES (default, K symmetric) | GMRES | BICGSTAB
rel_tol = 1.0e-10
abs_tol = 1.0e-12
max_iter = 500
preconditioner = "block_jacobi"   # block_jacobi (default) | none
print_level = 0
```

When mortar PBC is disabled, this block is parsed but ignored. Validation
rejects CG (the saddle-point system is indefinite).

### §P5.3.4 Options struct additions

In `option_parser_v2.hpp`:

```cpp
struct MeshOptions {
    // ... existing fields ...
    bool periodicity = false;       // existing
    double snap_tol = 1.0e-10;      // NEW
    int lor_depth = 1;              // NEW (Phase 6 stub; enforce == 1 in Phase 5)
};

// NEW — option-parser-side enums for the [Solvers.SaddlePoint] table.
// These are TRANSLATED at the construction boundary of MortarPbcManager
// into the Phase 4.3 internal types `mortar_pbc::KrylovType` and
// `mortar_pbc::SaddlePrecType` (defined in saddle_point_solver.hpp).
// Translation lives in `MortarPbcManager` so that option_parser_v2
// doesn't need to pull in mortar-pbc headers.
enum class SaddlePointSolverType { MINRES, GMRES, BICGSTAB, NOTYPE };
enum class SaddlePointPreconditioner { BLOCK_JACOBI, NONE, NOTYPE };

struct SaddlePointSolverOptions {
    SaddlePointSolverType linear_solver = SaddlePointSolverType::MINRES;
    double rel_tol = 1.0e-10;
    double abs_tol = 1.0e-12;
    int max_iter = 500;
    SaddlePointPreconditioner preconditioner = SaddlePointPreconditioner::BLOCK_JACOBI;
    int print_level = 0;

    bool validate() const;
    static SaddlePointSolverOptions from_toml(const toml::value& toml_input);
};

struct SolverOptions {
    // ... existing fields ...
    SaddlePointSolverOptions saddle_point;   // NEW
};
```

The translation step inside `MortarPbcManager`:

```cpp
// In mortar_pbc_manager.cpp:
mortar_pbc::SaddlePointSolverConfig
TranslateSaddleOpts(const SaddlePointSolverOptions& opts) {
    mortar_pbc::SaddlePointSolverConfig c;
    switch (opts.linear_solver) {
        case SaddlePointSolverType::MINRES:   c.solver_type = mortar_pbc::KrylovType::MINRES;   break;
        case SaddlePointSolverType::GMRES:    c.solver_type = mortar_pbc::KrylovType::GMRES;    break;
        case SaddlePointSolverType::BICGSTAB: c.solver_type = mortar_pbc::KrylovType::BiCGStab; break;
        default: MFEM_ABORT("invalid SaddlePointSolverType");
    }
    switch (opts.preconditioner) {
        case SaddlePointPreconditioner::BLOCK_JACOBI: c.prec_type = mortar_pbc::SaddlePrecType::BlockJacobi; break;
        case SaddlePointPreconditioner::NONE:         c.prec_type = mortar_pbc::SaddlePrecType::None;        break;
        default: MFEM_ABORT("invalid SaddlePointPreconditioner");
    }
    c.rel_tol = opts.rel_tol;
    c.abs_tol = opts.abs_tol;
    c.max_iter = opts.max_iter;
    c.print_level = opts.print_level;
    return c;
}
```

This way the user-facing TOML strings (`"MINRES"`, `"GMRES"`, etc.) drive
option-parser enums, and option-parser enums translate cleanly to the
Phase 4.3 internal types at the manager boundary. Renaming Phase 4.3's
`KrylovType` / `SaddlePrecType` is explicitly *not* in scope for Phase 5.

`MortarPbcOptions` from earlier drafts does **not** exist. The configuration
is distributed; `MortarPbcManager` reads what it needs from each existing
struct.

---

## §P5.4 The `MortarPbcManager` class

Lives in `src/mortar_pbc/mortar_pbc_manager.{hpp,cpp}`. Owns the classifier,
builder, constraint operator, `MortarSaddlePointSystem` adapter, and the
$\bar F^{(n)}$ macroscopic state.

### §P5.4.1 What the manager owns

```cpp
namespace exaconstit::mortar_pbc {

class MortarPbcManager {
public:
    /// Function-object types matching the Phase 4.3
    /// `MortarSaddlePointSystem` API.
    using KResidualFn = mortar_pbc::MortarSaddlePointSystem::KResidualFn;
    using KJacobianFn = mortar_pbc::MortarSaddlePointSystem::KJacobianFn;

    /// Construct: build classifier, builder, EA constraint op, the
    /// SaddlePointSolver-for-init, the MortarSaddlePointSystem adapter
    /// (using the supplied K closures), the λ warm-start vector, and
    /// initialize macroscopic F̄ to identity.
    /// Collective on pmesh.GetComm().
    ///
    /// The K residual / K Jacobian closures must be supplied at
    /// construction time because `MortarSaddlePointSystem` (Phase 4.3 /
    /// Batch R) takes them as constructor arguments — there are no
    /// setters on the saddle system. SystemDriver constructs both
    /// closures from `mech_operator` and passes them in here.
    MortarPbcManager(mfem::ParMesh& pmesh,
                     mfem::ParFiniteElementSpace& fes,
                     const ExaOptions& opts,
                     KResidualFn k_residual,
                     KJacobianFn k_jacobian);

    // ----- Per-step macroscopic F̄ update -----

    /// Advance F̄ from F̄^(n) to F̄^(n+1) using the current macroscopic L̄
    /// and the time step Δt. First-order update consistent with ExaConstit's
    /// existing time stepping:
    ///     F̄^(n+1) = F̄^(n) + L̄^(n+1) F̄^(n) Δt
    /// Called once per time step before the Newton solve. See §P5.8.6
    /// for the derivation. The vgrad input is the same flat 9-vector
    /// (row-major 3×3) that BCManager's `UpdateBCData` populates;
    /// the manager reshapes it internally to a 3×3 DenseMatrix.
    void UpdateMacroscopicF(const mfem::Vector& vgrad_9vec, double dt);

    /// Read access to the current F̄^(n) for diagnostics and the
    /// constraint RHS computation.
    const mfem::DenseMatrix& GetMacroscopicF() const;

    /// Compute the velocity gradient time derivative Ḟ̄^(n) = L̄^(n) F̄^(n)
    /// for the constraint RHS. Cached after each UpdateMacroscopicF.
    const mfem::DenseMatrix& GetMacroscopicFdot() const;

    // ----- Corner-Dirichlet TDOFs -----

    /// The 24 essential TDOFs (8 RVE corners × 3 components). When
    /// mortar PBC is enabled, these REPLACE the full-face TDOF list
    /// derived from BCManager's essential_vel_grad attributes —
    /// that's the entire BC override. See §P5.5.
    const mfem::Array<int>& GetCornerEssTDofs() const;

    // ----- The saddle-point Operator -----

    /// Returns the MortarSaddlePointSystem adapter that wraps the user's
    /// K residual/Jacobian closures with the constraint operator. This is
    /// what gets handed to ExaNewtonSolver. (Phase 4.3 / Batch R machinery
    /// extended in Phase 5.0 with constraint-RHS support.)
    MortarSaddlePointSystem& GetSaddleSystem();

    /// The direct-linear-saddle-point solver used by SystemDriver's
    /// SolveInit (one-shot linear correction at BC ramp boundaries).
    /// NOT used inside the Newton loop — that goes through the
    /// MortarSaddlePointSystem adapter above.
    SaddlePointSolver& GetSaddleSolverForInit();

    // ----- Constraint RHS update -----

    /// Recompute the saddle-point RHS using the current F̄^(n) and L̄^(n),
    /// and call `m_saddle_system->SetConstraintRHS(g)` to install it.
    /// Called once per time step after UpdateMacroscopicF, before the
    /// Newton solve. The RHS depends on macroscopic state but not on the
    /// current iterate, so it is fixed for the duration of the Newton
    /// iteration. See §P5.8.6.
    void UpdateConstraintRHS();

    // ----- λ accumulation -----

    /// The Lagrange multiplier vector. Persists across Newton iterations
    /// and time steps for warm-starting.
    mfem::Vector& GetLambda();
    void ResetLambda();
    int NumLocalLmRows() const;

    // ----- Diagnostics -----

    /// Compute the rank-local C·v for printing the constraint residual
    /// after a Newton solve.
    void ComputeConstraintResidual(const mfem::Vector& v,
                                   mfem::Vector& Cv) const;

    /// Compute v_tilde = v - v_lin on the current mesh, for visualization.
    /// v_lin is computed from the corner pin formula in spatial form
    /// (§P5.8.7). Cached reference corner coords are NOT used here —
    /// v_lin is the spatial-form macroscopic field on the current mesh.
    void ComputeFluctuationField(const mfem::Vector& v_tdofs,
                                 mfem::ParGridFunction& v_tilde_gf) const;

    /// Hill-Mandel power balance diagnostic. Returns
    ///     |P̄ : Ḟ̄ - (1/|Ω₀|) ∫_Ω₀ P : Ḟ dΩ|
    /// which should be zero to FP precision at any converged step
    /// (Hill 1963; Mandel 1971). See §P5.8.11. The macroscopic P̄ is
    /// recovered from the assembled λ via Lopes et al. eq. 37 — see
    /// §P5.10.3 for the implementation note.
    double ComputeHillMandelPowerBalance() const;

private:
    mfem::ParMesh& m_pmesh;
    mfem::ParFiniteElementSpace& m_fes;
    const ExaOptions& m_opts;

    // Built once at construction; never modified after.
    std::unique_ptr<BoundaryClassifier3D> m_classifier;
    std::unique_ptr<ConstraintBuilder3D> m_builder;
    std::unique_ptr<MortarConstraintOperator> m_C_op;
    std::unique_ptr<SaddlePointSolver> m_saddle_solver_for_init;
    std::unique_ptr<MortarSaddlePointSystem> m_saddle_system;

    mfem::Array<int> m_corner_ess_tdofs;
    mfem::Vector m_lambda;

    // Per-step macroscopic state.
    mfem::DenseMatrix m_macro_F;     // F̄^(n) — initialized to I, advanced per step
    mfem::DenseMatrix m_macro_Fdot;  // Ḟ̄^(n) = L̄^(n) F̄^(n) — cached for RHS

    // Buffer for the per-step constraint RHS g^(n) handed to
    // m_saddle_system->SetConstraintRHS(g). Persists across the Newton
    // iteration; refreshed in UpdateConstraintRHS each step.
    mfem::Vector m_g_rhs;

    // Cached at construction; used in UpdateConstraintRHS (§P5.8.6.d).
    // Per LM row: ∫_{Γ₀} M_i dΓ₀ × (X⁺ - X⁻)_pair_axis(i)
    mfem::DenseMatrix m_rhs_geometric_factors;
};

}  // namespace
```

### §P5.4.2 The `MortarSaddlePointSystem` integration

`MortarSaddlePointSystem` (Phase 4.3 Batch R) is an `mfem::Operator` that
composes user-supplied K residual + Jacobian closures with the EA constraint
operator into a unified saddle-point Operator:

- `Mult(x_block, y_block)` returns the saddle-point residual:
  $y = (\,F_\text{int}(v) + C^T \lambda - 0,\; C v - g^{(n)}\,)^T$.
- `GetGradient(x_block)` returns the `BlockOperator` representing
  $\begin{bmatrix} K & C^T \\ C & 0 \end{bmatrix}$.

The manager constructs this once at startup with the user's
`NonlinearMechOperator` driving the K closures. The constraint matrix $C$
in the adapter never changes; only the RHS $g^{(n)}$ in the residual
computation updates per step (controlled by `UpdateConstraintRHS`).

This is what makes §P5.7 simple: existing solvers see a regular
`mfem::Operator`; they don't need to know about saddle-point structure.

### §P5.4.3 Construction sequence

```cpp
MortarPbcManager::MortarPbcManager(mfem::ParMesh& pmesh,
                                   mfem::ParFiniteElementSpace& fes,
                                   const ExaOptions& opts,
                                   KResidualFn k_residual,
                                   KJacobianFn k_jacobian)
    : m_pmesh(pmesh), m_fes(fes), m_opts(opts),
      m_macro_F(3), m_macro_Fdot(3)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::construct");

    // 1. Classifier on the parent ParMesh — built once on the initial
    //    (undeformed) mesh. Phase 4.2 distributed pair matching runs here.
    //    The classifier caches reference face geometry in
    //    QuadFaceElement::coords / TriFaceElement::coords arrays
    //    transiently during BuildLocalPairBlocks; after Initialize()
    //    returns, only the assembled D / A_m blocks remain. The
    //    classifier is therefore decoupled from pmesh's subsequent
    //    motion (§P5.8.5).
    m_classifier = std::make_unique<BoundaryClassifier3D>(
        pmesh, fes, opts.mesh.snap_tol);
    m_classifier->Initialize();

    // 2. Builder — also built once; topology only.
    m_builder = std::make_unique<ConstraintBuilder3D>(*m_classifier);

    // 3. EA constraint op on the reference (initial) mesh. Built once;
    //    NEVER rebuilt. See §P5.8.5 for why this is correct under UL.
    m_C_op = std::make_unique<MortarConstraintOperator>(*m_classifier);

    // 4. Direct-linear saddle-point solver for SolveInit (one-shot
    //    linear corrections at BC ramp boundaries). Configured from
    //    [Solvers.SaddlePoint] options via the option-parser-side
    //    enum translation (see §P5.3.4).
    m_saddle_solver_for_init = std::make_unique<SaddlePointSolver>(
        TranslateSaddleOpts(opts.solvers.saddle_point));

    // 5. Adapter: composes the K closures with the constraint op into
    //    a single mfem::Operator presenting the saddle-point system to
    //    the Newton solver. The K closures must be supplied at
    //    construction time (Phase 4.3 / Batch R API; no setters
    //    available). With Phase 5.0's constraint-RHS extension, the
    //    adapter also accepts a non-zero constraint RHS via
    //    `SetConstraintRHS`; we install that per step in
    //    UpdateConstraintRHS().
    m_saddle_system = std::make_unique<MortarSaddlePointSystem>(
        std::move(k_residual), std::move(k_jacobian), *m_C_op);

    // 6. Corner TDOFs from the classifier's CornerInfo3D records.
    BuildCornerEssTDofs();

    // 7. λ warm-start.
    m_lambda.SetSize(m_builder->NumLocalRows());
    m_lambda.UseDevice(true);
    m_lambda = 0.0;

    // 8. Constraint-RHS buffer.
    m_g_rhs.SetSize(m_builder->NumLocalRows());
    m_g_rhs.UseDevice(true);
    m_g_rhs = 0.0;

    // 9. Macroscopic F̄ initialized to identity (no prior deformation).
    m_macro_F = 0.0;
    m_macro_F(0, 0) = m_macro_F(1, 1) = m_macro_F(2, 2) = 1.0;
    m_macro_Fdot = 0.0;

    // 10. Cache RHS geometric factors (§P5.8.6.d). Computed once on the
    //     reference; reused for every step's UpdateConstraintRHS.
    BuildReferenceGeometricFactors();
}
```

The `BuildReferenceGeometricFactors()` step computes, per LM row, the
quantities $\hat \ell^{\hat E_0}_i \cdot (X^+ - X^-)_{\text{pair}(i)}$ from
(P5.8.6.d). These are constants over the simulation lifetime.

### §P5.4.4 Per-step state update

```cpp
void MortarPbcManager::UpdateMacroscopicF(const mfem::Vector& vgrad_9vec,
                                          double dt)
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::update_macro_F");

    // BCManager populates vgrad_9vec as a flat row-major 3×3 (the same
    // convention used by the existing essential_vel_grad code path).
    // Reshape it locally into a 3×3 DenseMatrix.
    mfem::DenseMatrix L_macro(3);
    {
        const auto VG = vgrad_9vec.HostRead();
        auto LM = L_macro.HostWrite();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                LM[i * 3 + j] = VG[i * 3 + j];
            }
        }
    }

    // F̄^(n+1) = F̄^(n) + L̄^(n+1) F̄^(n) Δt
    mfem::DenseMatrix delta_F(3);
    mfem::Mult(L_macro, m_macro_F, delta_F);
    delta_F *= dt;
    m_macro_F += delta_F;

    // Ḟ̄^(n+1) = L̄^(n+1) F̄^(n+1) (post-update value used in RHS)
    mfem::Mult(L_macro, m_macro_F, m_macro_Fdot);
}

void MortarPbcManager::UpdateConstraintRHS()
{
    CALI_CXX_MARK_SCOPE("mortar_pbc::manager::update_rhs");

    // Per LM row i: g_i = Ḟ̄^(n) · m_rhs_geometric_factors[i,:]
    // where m_rhs_geometric_factors[i,:] is the cached
    // ∫M_i dΓ₀ · (X⁺-X⁻)_pair(i) vector.
    auto FACTORS = m_rhs_geometric_factors.HostRead();
    auto FDOT    = m_macro_Fdot.HostRead();
    auto G       = m_g_rhs.HostWrite();
    const int nrows = m_builder->NumLocalRows();

    for (int i = 0; i < nrows; ++i) {
        double gi = 0.0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                // Component-aware unrolling: the LM row's component is
                // determined by which displacement component the
                // homologous-pair offset acts in.
                gi += FDOT[3*k + l] * FACTORS[i*3 + l];
            }
        }
        G[i] = gi;
    }

    // Install the new RHS into the saddle-point adapter. The adapter
    // (Phase 4.3 + Phase 5.0 extension) will subtract this from r_C
    // in its Mult, so the Newton converges when C v - g = 0 (the
    // Method-D form, see §P5.8.4.4).
    m_saddle_system->SetConstraintRHS(m_g_rhs);
}
```

### §P5.4.5 Lifetime and ownership

- `MortarPbcManager` is held by `SystemDriver` as a `std::unique_ptr`.
- It receives the parent `ParMesh` and `ParFiniteElementSpace` by reference;
  lifetime is managed by `SimulationState` upstream.
- All mortar machinery (classifier, builder, $C$ op, saddle solver / system)
  is owned by the manager.
- $\lambda$ persists across Newton iterations (warm-start within step) and
  across time steps (warm-start at next step's first Newton iteration).
- $\bar F$ persists across time steps as macroscopic state and must be
  saved / restored on checkpoint / restart (§P5.14.1).

---

## §P5.5 Corner-Dirichlet via the existing BCManager / `mono_def_flag` pattern

`BCManager` continues to own all BC data. Mortar PBC introduces a single
override: when projecting `essential_vel_grad` BCs to the mech operator's
essential-TDOF list, use the *corner-only subset* of TDOFs from those
attributes, not the full-face set.

This is structurally identical to how `mono_def_flag` already operates: the
flag redirects the BC application path through a different branch in
`SystemDriver::UpdateEssBdr` and `NonlinearMechOperator::UpdateEssTDofs`.
Mortar PBC adds a parallel flag `m_mortar_enabled` in `SystemDriver` that
does the same kind of selective override.

### §P5.5.1 The override hook

`NonlinearMechOperator` gains one new method:

```cpp
class NonlinearMechOperator : public mfem::NonlinearForm {
public:
    // ... existing ...

    /// Phase 5 — replace the mech operator's essential TDOF list with
    /// the supplied set. Mirrors UpdateEssTDofs's mono_def_flag branch
    /// but takes a TDOF list directly instead of an attribute mask.
    /// Used by mortar PBC to inject the 8-corner-only TDOF set.
    void UpdateEssTDofsCornerSubset(const mfem::Array<int>& corner_tdofs);
};
```

Internally this calls `ParNonlinearForm::SetEssentialTrueDofs` with the
corner list. From the operator's perspective, nothing else changes — `Mult`
and `GetGradient` zero-eliminate the corner rows just as they would for any
Dirichlet TDOF.

### §P5.5.2 BC value computation: the corner pin in spatial form

The corner pin in UL spatial form is

$$
v_\text{corner}^{(n)} = \bar L^{(n)} \cdot (x_\text{corner}^{(n)} - x_0)
$$

where $x_\text{corner}^{(n)}$ is the *current* corner coordinate and $x_0$ is
the user-supplied origin. ExaConstit's existing `essential_vel_grad`
machinery computes exactly this (verified in `system_driver.cpp`:
`mesh->GetNodes()` returns current spatial coords). The mortar override
just restricts the projection to corner TDOFs; the formula is unchanged.

**Origin assumption**: this works correctly for $X_0 = (0,0,0)$ (the
standard convention for axis-aligned RVEs). For $X_0 \ne 0$, see
§P5.8.9 Gotcha 3 — the recommended workflow is to pre-translate the mesh
so the loading origin is at $(0,0,0)$.

### §P5.5.3 SystemDriver wiring

```cpp
// In SystemDriver constructor, AFTER mech_operator is constructed
// (the K closures need to capture it):
if (options.mesh.periodicity && HasVelocityGradientBC(options)) {
    m_mortar_enabled = true;

    // Build the K closures up front. They capture mech_operator as
    // a raw pointer; the manager outlives mech_operator only if the
    // SystemDriver guarantees it (asserted at manager destruction).
    auto k_residual =
        [op_ptr = mech_operator.get()](const mfem::Vector& v,
                                       mfem::Vector& r) {
            op_ptr->Mult(v, r);
        };
    auto k_jacobian =
        [op_ptr = mech_operator.get()](const mfem::Vector& v)
            -> mfem::Operator* {
            // mech_operator->GetGradient returns by reference; convert
            // to the pointer the saddle-system's KJacobianFn expects.
            return &op_ptr->GetGradient(v);
        };

    // Construct the manager with the closures up front. The manager's
    // ctor builds the MortarSaddlePointSystem internally (Phase 4.3
    // Batch R adapter takes K closures at construction; there are
    // no setters).
    m_mortar_pbc = std::make_unique<MortarPbcManager>(
        *m_sim_state->GetMesh(),
        *m_sim_state->GetMeshParFiniteElementSpace(),
        options,
        std::move(k_residual),
        std::move(k_jacobian));

    // Override essential TDOFs to the corner subset.
    mech_operator->UpdateEssTDofsCornerSubset(
        m_mortar_pbc->GetCornerEssTDofs());

    // Hand the saddle-point Operator to the Newton solver in place
    // of mech_operator.
    newton_solver->SetOperator(m_mortar_pbc->GetSaddleSystem());
} else {
    // Standard non-mortar path — unchanged.
    newton_solver->SetOperator(mech_operator);
}
```

### §P5.5.4 `UpdateEssBdr` and `UpdateVelocity`

`SystemDriver::UpdateEssBdr`'s existing logic queries `BCManager` for the
essential_vel_grad TDOF mask and updates `mech_operator`. Under mortar PBC
we keep the corner-only TDOF override:

```cpp
void SystemDriver::UpdateEssBdr() {
    if (!mono_def_flag) {
        BCManager::GetInstance().UpdateBCData(
            ess_bdr, ess_bdr_scale, ess_velocity_gradient, ess_bdr_component);
        if (m_mortar_enabled) {
            // Corner TDOFs are step-invariant; re-asserting them is
            // a no-op but cheap and clearer than skipping the call.
            mech_operator->UpdateEssTDofsCornerSubset(
                m_mortar_pbc->GetCornerEssTDofs());
        } else {
            mech_operator->UpdateEssTDofs(ess_bdr["total"], mono_def_flag);
        }
    }
}
```

`UpdateVelocity` similarly grows one branch — when mortar enabled, project
the velocity-gradient onto the corner TDOFs only, using the *current* mesh
nodes through the existing essential_vel_grad spatial-form code path.

**Refactoring note**: the body of the existing `UpdateVelocity`'s
essential_vel_grad branch (in `system_driver.cpp`) does
$v(x) = \nabla v \cdot (x - x_0)$ across all TDOFs of the tagged
attributes. To reuse this on a TDOF subset, factor the per-node loop
into a small free function:

```cpp
void ProjectVelocityGradientToCornerTDofs(
    const mfem::ParGridFunction& mesh_nodes,
    const mfem::Vector& vgrad_9vec,
    const mfem::Vector& origin,
    const mfem::Array<int>& corner_tdofs,
    mfem::Vector& v_tdofs);
```

The function reads `mesh->GetNodes()` (current spatial coords) and
applies $v = \nabla v \cdot (x - x_0)$ at each TDOF in `corner_tdofs`,
writing into `v_tdofs` at those positions and leaving other entries
unchanged. The non-mortar `UpdateVelocity` path can be refactored to
call this same helper with `corner_tdofs = ess_tdofs` (all
essential_vel_grad TDOFs), making the mortar branch a one-line
substitution. **Trap §P5.14.9 covers the regression risk** if this
refactor is done sloppily.

---

## §P5.6 SystemDriver `Solve()`, `SolveInit()`, and the per-step lifecycle

### §P5.6.1 The new step lifecycle

The main time-stepping loop in `mechanics_driver.cpp` already has the right
shape:

```cpp
while (!sim_state->IsFinished()) {
    if (BCManager::GetInstance().GetUpdateStep(ti)) {
        oper.SyncMortarPbcForStep(ti);   // §P5.18.6 — install or
                                         // switch active periodic-BC
                                         // entry for this step. No-op
                                         // for non-mortar runs and for
                                         // steps that don't transition.
        oper.UpdateEssBdr();
        oper.UpdateVelocity();
        oper.SolveInit();
    }
    if (m_mortar_enabled) {
        // Advance F̄^(n) using the current macroscopic L̄^(n+1) and Δt.
        // No constraint matrix rebuild — see §P5.8 for the derivation.
        // ess_velocity_gradient is the SystemDriver member that BCManager
        // has just populated for this step (in UpdateEssBdr above).
        m_mortar_pbc->UpdateMacroscopicF(ess_velocity_gradient, GetCurrentDt());
        m_mortar_pbc->UpdateConstraintRHS();
    }
    oper.UpdateModel();
    oper.Solve();
    sim_state->FinishCycle();
    // ... output, advance time ...
    ti++;
}
```

The mortar-specific addition is two cheap calls: `UpdateMacroscopicF` (one
3x3 matrix multiply) and `UpdateConstraintRHS` (one matvec on a small dense
table). Both run before `UpdateModel` and `Solve`. **No constraint matrix
rebuild, no per-Newton-iteration refresh.** The matrix $C$ inside
`m_C_op` is what was constructed at simulation start and stays.

### §P5.6.2 `Solve()` — minimal change

Because the `MortarSaddlePointSystem` adapter is what's been handed to
`newton_solver` (§P5.5.3), the Newton body itself doesn't need a new code
path:

```cpp
void SystemDriver::Solve() {
    auto x = m_sim_state->GetPrimalField();
    if (auto_time) { /* ... existing initial-guess setup ... */ }

    mfem::Vector zero;
    newton_solver->Mult(zero, *x);   // operates on the saddle-point
                                     // BlockVector when m_mortar_enabled
    // ... existing convergence assertion ...
}
```

The "BlockVector when mortar enabled" detail is handled by the adapter:
`MortarSaddlePointSystem::Mult` knows to pack/unpack the $\lambda$ block
internally, so the outer `Mult` signature is still `(input, output)` on
`mfem::Vector`. The existing `ExaNewtonSolver::Mult` body — convergence
check, residual norm, line search — operates on the saddle-point operator
transparently.

The convergence criterion correction (the residual must include the
$C^T \lambda$ term and the constraint residual, see architecture doc §12
Trap 3) is internal to `MortarSaddlePointSystem::Mult`: it returns the
combined residual that the Newton solver sees, so existing rel/abs
tolerances apply naturally.

### §P5.6.3 `SolveInit()` — corrector with constraint awareness

The `SolveInit` corrector solves
$K_\text{eliminated} \cdot \Delta v_\text{free} = -K \cdot \Delta v_\text{ess}$
for the BC ramp-up. With mortar PBC the corrector becomes a saddle-point
system:

$$
\begin{bmatrix} K_\text{eliminated} & C^T \\ C & 0 \end{bmatrix}
\begin{bmatrix} \Delta v \\ \Delta \lambda \end{bmatrix} =
\begin{bmatrix} -K \cdot \Delta v_\text{ess} \\ -(C v_\text{prev} - g^{(n)}) \end{bmatrix}
$$

The right-hand side $-(Cv_\text{prev} - g^{(n)})$ is typically tiny if the
previous step converged (constraint was satisfied) and the new step's $g$
is close to the prev's (small loading change); it captures the correction
needed for any accumulated drift plus the loading delta. The left block is
the same $C$ used in the Newton, with $K$ from
`mech_operator->GetUpdateBCsAction(...)`.

Implementation: the existing `SolveInit` body is augmented with a mortar
branch that constructs the saddle-point RHS and calls
`m_mortar_pbc->GetSaddleSolver().Solve(K_uc, *m_C_op, b, r2, du, dlam)`
directly (bypassing the Newton wrapper since this is a single linear solve).

---

## §P5.7 Newton-solver integration via `MortarSaddlePointSystem`

`MortarSaddlePointSystem` does the heavy lifting; the existing
`ExaNewtonSolver` family stays nearly unchanged.

### §P5.7.1 What stays the same in `ExaNewtonSolver` family

- Class structure (NR, NRLS, TRDOG).
- `Mult(b, x)` signature.
- The convergence loop.
- The line-search logic (NRLS).
- The trust-region logic (TRDOG).

### §P5.7.2 What changes

Nothing structural. The solver is handed a different `mfem::Operator`:
when mortar PBC is on, the operator is `MortarSaddlePointSystem` and
`GetGradient` returns a `BlockOperator`. The adapter's internals take care
of the saddle-point Krylov on the inner solve.

The only solver-side adjustment is that the inner-Krylov preconditioner —
what would normally be the user's choice from `[Solvers.Krylov]` — for the
mortar case is bypassed in favour of the `[Solvers.SaddlePoint]`
block-Jacobi preconditioner that the adapter constructs from
`MortarConstraintOperator::ComputeInvDiagSchur`. This is automatic: the
adapter's `GetGradient` returns a `BlockOperator` plus an associated
`BlockDiagonalPreconditioner`; the existing `ExaNewtonSolver::Mult` calls
`prec->SetOperator(grad)` and then `prec->Mult(r, c)`, which routes through
MFEM's saddle-point Krylov infrastructure.

### §P5.7.3 NRLS: line search merit function

The NRLS line search picks $\varepsilon$ minimizing $\|F\|$. With the
saddle-point Operator, $F$ is the combined residual; the adapter's `Mult`
returns the right thing automatically. NRLS line search correctness flows
from the adapter; no NRLS code modification needed.

### §P5.7.4 TRDOG: the K^T concern

Trust region uses $J$ and $J^T$ on the block Jacobian. The block transpose is

$$
\mathcal{J}^T = \begin{bmatrix} K^T & C^T \\ C & 0 \end{bmatrix}
$$

(remember: the off-diagonal $C^T$ appears in both $\mathcal{J}$ and
$\mathcal{J}^T$ — they don't swap when you take the block transpose). When
$K$ is symmetric (linear elastic, hyperelastic, most ExaCMech tangents
under standard formulations), $\mathcal{J}^T = \mathcal{J}$ and TRDOG works
without modification.

Your recent BBar PA `MultTranspose` work covers the case of $K^T$ being
explicitly needed; when wired into `mech_operator->GetGradient(x)` returning
a `MultTranspose`-capable operator, the `MortarSaddlePointSystem`'s block
Jacobian also exposes `MultTranspose` correctly (composes the per-block
transposes through the standard `BlockOperator::MultTranspose` path).

---

## §P5.8 The mortar PBC constraint under Updated Lagrangian — theoretical justification, mathematical derivation, and failure modes

This section is the mathematical and conceptual core of v3. It justifies
why the constraint matrix $C$ can be built once on reference geometry and
reused throughout the simulation, despite ExaConstit's bulk being UL with
velocity primal — a combination that does not appear to have been published
elsewhere.

### §P5.8.1 The continuous PBC constraint in UL spatial form

#### Kinematic statement

For a representative volume element (RVE) on the cube $\Omega_0 = [0, L]^3$
in the *reference* (initial, undeformed) configuration, periodicity is the
statement that the displacement fluctuation $\tilde u$ is identical on
homologous points across opposing faces:

$$
\tilde u(X^+) = \tilde u(X^-)
\qquad \forall\, (X^+, X^-) \text{ homologous on } \partial \Omega_0
$$

where the total displacement is split as $u(X) = (\bar F - I)\, X + \tilde u(X)$
into a macroscopic affine part and a periodic fluctuation, and homologous
pairs $(X^+, X^-)$ are reference-coordinate pairs related by lattice
translation: $X^+ = X^- + L\,\hat e_k$ for some axis $\hat e_k$.

This is the standard formulation found in Hill (1963), Mandel (1971), and
all subsequent computational homogenization literature. In displacement
form, the homologous-pair constraint is

$$
u(X^+) - u(X^-) = (\bar F - I)\,(X^+ - X^-).
\tag{P5.8.1.a}
$$

#### Conversion to velocity-primal UL form

ExaConstit solves for the spatial velocity field $v(x, t) = \dot x$ on the
*current* (deformed) configuration $\Omega^{(n)}$. The mesh nodes advance
per step via $x^{(n+1)} = x^{(n)} + v^{(n+1)}\,\Delta t$, and the constitutive
law sees the relative deformation gradient $F_{\text{rel}}^{(n)} = \partial x^{(n+1)}/\partial x^{(n)}$.

Differentiating (P5.8.1.a) in time:

$$
\dot u(X^+) - \dot u(X^-) = \dot{\bar F}\,(X^+ - X^-).
\tag{P5.8.1.b}
$$

Since $\dot u$ at a material point equals the spatial velocity at the
current position of that point, this is

$$
v(x^+(t)) - v(x^-(t)) = \dot{\bar F}\,(X^+ - X^-).
\tag{P5.8.1.c}
$$

To convert to fully spatial form, observe that for a homologous pair the
*current* coordinate offset is

$$
x^+ - x^- = X^+ + u(X^+) - X^- - u(X^-)
            = (X^+ - X^-) + \bigl[u(X^+) - u(X^-)\bigr]
            = (X^+ - X^-) + (\bar F - I)\,(X^+ - X^-)
            = \bar F\,(X^+ - X^-),
\tag{P5.8.1.d}
$$

where the second-to-last step uses (P5.8.1.a) at converged equilibrium.
Substituting:

$$
\dot{\bar F}\,(X^+ - X^-) = \dot{\bar F}\, \bar F^{-1}\,(x^+ - x^-) = \bar L\,(x^+ - x^-),
$$

since the macroscopic spatial velocity gradient is defined as
$\bar L = \dot{\bar F}\,\bar F^{-1}$. So (P5.8.1.c) becomes

$$
\boxed{\;v(x^+) - v(x^-) = \bar L\,(x^+ - x^-)\;}
\tag{P5.8.1.e}
$$

This is the **velocity-primal spatial-form periodic constraint** that
ExaConstit must enforce. Both $v(x^\pm)$ and $x^\pm$ are evaluated at the
*current* configuration $\Omega^{(n)}$; $\bar L^{(n)}$ is the macroscopic
spatial velocity gradient at step $n$ (a user-supplied loading rate).

#### Remarks

(a) The equivalence (P5.8.1.c) ↔ (P5.8.1.e) is exact for the converged
periodic field, regardless of how large the macroscopic deformation has
become. The constraint can be expressed purely in current-config quantities
($v$, $x$, $\bar L$) without explicit reference to $\bar F$. This is the
form ExaConstit naturally works with — and crucially, the form the existing
`essential_vel_grad` BC machinery already uses for non-mortar loading.

(b) Equation (P5.8.1.d) is *not* generally true off-equilibrium (during a
non-converged Newton iteration, the fluctuation field $\tilde u$ is not yet
periodic), so the relation $x^+ - x^- = \bar F(X^+ - X^-)$ is approximate
mid-iteration. The right-hand side of the discrete constraint will use
*current* $x^\pm$ as read from the moving mesh; the constraint itself drives
the field toward (P5.8.1.e) at convergence. We discuss this iteration-vs-converged
distinction further in §P5.8.6.

### §P5.8.2 The mortar weak form and its integration domain

#### Discrete weak form

Introduce a dual-basis Lagrange multiplier $\lambda_h \in M_h(\partial \Omega^-)$
on the nonmortar surface (Wohlmuth, *SIAM J. Numer. Anal.* 38, 2000;
Wohlmuth, *Discretization Techniques and Iterative Solvers* book, 2001). The
Lagrange multiplier $\lambda$ has the physical interpretation of the
surface traction needed to enforce periodicity (Lopes et al. 2021, eq. 37,
recovers this for the closely related uniform-traction case). The mortar
weak form of (P5.8.1.e) is

$$
\int_\Gamma M_i(\xi) \cdot \bigl[\,v_h^+(\xi) - v_h^-(\xi) - \bar L\,(x_h^+(\xi) - x_h^-(\xi))\,\bigr]\, d\Gamma = 0,
\quad \forall i,
\tag{P5.8.2.a}
$$

where $M_i$ is a Wohlmuth dual-basis function and $\Gamma$ is the integration
domain. The *choice of $\Gamma$* is the central operational question.

Two natural options:

- **Choice A (reference)**: $\Gamma = \Gamma_0$, the reference (undeformed)
  nonmortar surface. The integrand uses reference parametrization; the area
  measure $d\Gamma_0$ is constant per element on an axis-aligned RVE.

- **Choice B (current)**: $\Gamma = \Gamma^{(n)}$, the deformed nonmortar
  surface at step $n$. The integrand uses current parametrization; the area
  measure $d\Gamma^{(n)}$ varies per quadrature point under non-trivial
  deformation.

Both are mathematically valid weak forms of the same continuous constraint.
Both give convergent discrete approximations as the mesh refines. They
differ in the *discrete approximation properties* — most importantly, in
whether the Wohlmuth dual basis preserves its biorthogonality.

#### Conversion of integration measures

Under deformation, the relationship between $d\Gamma_0$ and $d\Gamma^{(n)}$
is given by Nanson's formula:

$$
d\Gamma^{(n)} = J^{(n)}\,\bigl|F^{-T,(n)} \hat n_0\bigr|\, d\Gamma_0
\equiv \theta^{(n)}(\xi)\, d\Gamma_0,
\tag{P5.8.2.b}
$$

where $J^{(n)} = \det F^{(n)}$ is the bulk Jacobian and $\hat n_0$ is the
reference outward normal. The factor $\theta^{(n)}(\xi)$ varies pointwise on
the parent face element under non-trivial deformation. This is the source
of the biorthogonality issue under Choice B (next subsection).

#### Discretization

Using $v_h^\pm = \sum_j v_j^\pm N_j(\xi)$ and the dual-basis property, the
weak form (P5.8.2.a) discretizes to

$$
D_{ii} v_i^- - \sum_j A^m_{ij} v_j^+ = g_i^{(n)},
\tag{P5.8.2.c}
$$

with

$$
D_{ii} = \int_\Gamma M_i \, N_i \, d\Gamma,
\qquad
A^m_{ij} = \int_\Gamma M_i \, (N_j \circ \Pi)\, d\Gamma,
\qquad
g_i^{(n)} = \bar L^{(n)} \cdot \int_\Gamma M_i\,(x^+ - x^-)\, d\Gamma.
\tag{P5.8.2.d}
$$

Here $\Pi$ is the homologous-pair projection. Stack over all nonmortar DOFs
to get the system $C v = g^{(n)}$ where $C = [-D \;|\; A^m]$ acts on the
stacked nonmortar / mortar TDOF vector.

The matrices $D$ and $A^m$ depend on $\Gamma$ — i.e., on Choice A vs B. The
right-hand side $g^{(n)}$ depends on both $\Gamma$ and the current
$\bar L^{(n)}$ / $x^\pm$.

### §P5.8.3 Wohlmuth biorthogonality on reference faces — why this gives clean discrete behavior

#### The biorthogonality property

Wohlmuth (2000) constructs the dual basis $M_i$ on a reference parent
element $\hat E$ via a coefficient transformation

$$
M_i(\xi) = \sum_j A^{\hat E}_{ij}\, N_j(\xi),
\qquad
A^{\hat E} = \bigl(\hat M^{\hat E}\bigr)^{-1}\, \mathrm{diag}\bigl(\hat \ell^{\hat E}\bigr),
\tag{P5.8.3.a}
$$

with $\hat M^{\hat E}_{ij} = \int_{\hat E} N_i N_j \, d\hat\xi$ the parent
mass matrix and $\hat \ell^{\hat E}_j = \int_{\hat E} N_j \, d\hat\xi$ the
lumped row-sums. By construction (Wohlmuth 2000, Lemma 2.1):

$$
\int_{\hat E} M_i(\xi)\, N_j(\xi)\, d\hat\xi = \delta_{ij}\, \hat \ell^{\hat E}_j.
\tag{P5.8.3.b}
$$

This is the **biorthogonality property**: in the reference parametric
domain, the dual basis is orthogonal to the standard basis up to the lumped
row-sums. This makes $D$ diagonal (cheap to invert) — the operational
advantage of dual mortar over Lagrange-multiplier mortar.

#### What happens under deformation

When we evaluate $D$ via Choice A (reference integration), we compute

$$
D^A_{ii} = \int_{E_0} M_i(\xi)\, N_i(\xi)\, |J^{E_0}_\text{face}|\, d\hat\xi
        = |J^{E_0}_\text{face}| \int_{\hat E} M_i\, N_i\, d\hat\xi
        = |J^{E_0}_\text{face}|\, \hat \ell^{\hat E}_i,
\tag{P5.8.3.c}
$$

where $|J^{E_0}_\text{face}|$ is the *constant* parent-to-physical Jacobian
of the reference axis-aligned face. The integral factors cleanly because
$|J|$ is constant; biorthogonality is preserved at the assembled level.
**$D$ is exactly diagonal.**

Under Choice B (current integration), we compute

$$
D^B_{ii} = \int_{E^{(n)}} M_i\, N_i\, |J^{E,(n)}_\text{face}(\xi)|\, d\hat\xi.
\tag{P5.8.3.d}
$$

The factor $|J^{E,(n)}_\text{face}(\xi)|$ varies *pointwise* on the parent
under non-trivial deformation (per Nanson's formula, P5.8.2.b — the factor
$\theta^{(n)}(\xi)$ is non-constant when the deformation gradient has any
spatial variation across the face). The integral does not factor; the
parent-element coefficients $A^{\hat E}_{ij}$ from (P5.8.3.a) no longer give
biorthogonality at the assembled level. **$D$ becomes full (no longer
diagonal), and the dual mortar's algorithmic advantage is lost.**

#### Restoring biorthogonality: consistent biorthogonalization

Popp, Gee, & Wall (*IJNME* 79, 2009) and Popp, Wohlmuth, Gee, & Wall
(*SIAM J. Sci. Comput.* 34, 2012) developed **consistent biorthogonalization**
to restore diagonality of $D$ under Choice B: reconstruct $A^{\hat E}_{ij}$
*per element per iteration* using the deformed-element mass matrix
$\hat M^{E,(n)}_{ij} = \int_{\hat E} N_i N_j\, |J^{(n)}|\, d\hat\xi$. This
makes the dual basis itself deformation-dependent; correctness then
requires the dual basis to be linearized w.r.t. nodal positions in the
consistent Newton tangent (Popp et al. 2010, *IJNME* 84:543).

This is what mortar contact codes do (Tribol uses *standard* — non-dual —
multipliers and avoids the issue; BACI/4C does consistent biorthogonalization;
see Wohlmuth & Popp's overview chapters in *Mortar Methods for Single- and
Multi-Field Applications*, Springer 2014). It works but is substantially
more expensive than the reference-integration approach — every Newton
iteration rebuilds $A^{\hat E}$ for every face element, and the linearization
adds tangent terms. **For RVE-PBC where this expense is unnecessary, Choice
A is strictly better.**

#### The RVE-PBC literature consensus

Reis & Pires (*CMAME* 274, 2014, eq. 14–17) and Lopes, Ferreira, & Pires
(*CMAME* 384, 2021, §3) build $D$ and $A^m$ on the reference RVE faces, in
total Lagrangian frameworks with displacement primal. In both cases the
matrices are constructed once at preprocessing and reused throughout the
simulation. This is the standard practice in computational homogenization
mortar PBC, justified by the constant-Jacobian property of axis-aligned RVE
reference faces and the resulting biorthogonality preservation.

### §P5.8.4 The constraint matrix is configuration-blind and primal-blind

This is the key conceptual point that justifies applying the literature's
reference-integration technique to ExaConstit's UL bulk.

#### Configuration-independence of $C_{ij}$

The matrix entries (P5.8.2.d) are *pure geometric integrals* on whichever
$\Gamma$ is chosen. They do not contain $u$, $v$, or any field variable.
Once $\Gamma$ is fixed, the matrix entries are deterministic functions of
the chosen face geometry alone:

$$
C^A_{ij} = \int_{\Gamma_0} M_i\, N_j\, d\Gamma_0
\quad\text{(pure geometric integral on reference faces)}.
\tag{P5.8.4.a}
$$

Construct (P5.8.4.a) once at simulation start, store the resulting sparse
matrix, and the *values* $C^A_{ij}$ are valid for the rest of the simulation.
The mesh advancing under UL does not change these values because the
integration domain $\Gamma_0$ is reference geometry, not pmesh's current
state.

#### Primal-independence of $C_{ij}$

The matrix $C^A$ is a linear operator on a TDOF vector. The TDOF vector can
hold *any* field defined on the same FES — displacement $u$, velocity $v$,
displacement increment $\Delta u$, or anything else. The operator's action

$$
[C^A w]_i = \sum_j C^A_{ij}\, w_j
\tag{P5.8.4.b}
$$

is a pure linear-algebra operation; it doesn't know what $w$ represents
physically. So the *same* matrix $C^A$ that encodes the displacement-form
constraint $C^A u = g^{TL}(\bar F, X)$ in a TL framework also encodes the
velocity-form constraint $C^A v = g^{UL}(\bar L, x)$ in a UL framework
**provided we update the right-hand side appropriately**.

The discrete velocity constraint becomes

$$
C^A\, v = g^{(n)}
\qquad\text{with}\qquad
g_i^{(n)} = \bar L^{(n)} \cdot \int_{\Gamma_0} M_i\,(x^{+,(n)} - x^{-,(n)})\, d\Gamma_0,
\tag{P5.8.4.c}
$$

which uses the same matrix $C^A$ but a UL-aware RHS computed from current
$x^\pm$ (read from current pmesh).

#### Why this works across formulation boundaries

The kinematic relationship $v^+ - v^- = \bar L(x^+ - x^-)$ is a statement
about *fields and their spatial coordinates*, not about which integration
domain the discrete weak form uses. Picking $\Gamma_0$ for the discrete
weak form gives one consistent discrete approximation; picking $\Gamma^{(n)}$
gives a different but equally valid one. Both converge to the same
continuous constraint as $h \to 0$ (standard mortar convergence theory —
Wohlmuth 2000, Theorem 4.1; Bernardi, Maday, & Patera 1994, *IMA J.* 14).

So the right thing for ExaConstit is to:

1. Build $C^A$ once on $\Gamma_0$ (reference, undeformed faces). Get
   biorthogonality automatically.
2. Apply $C^A$ to velocity TDOFs — this is the UL-primal velocity constraint.
3. Update the RHS $g^{(n)}$ each step using current $x^\pm$ and current
   $\bar L^{(n)}$.
4. The mesh advancing under UL doesn't touch $C^A$ — the matrix is
   configuration-blind.

This is operationally the same as what Reis & Pires / Lopes do, but with
the TDOFs interpreted as velocity rather than displacement and the RHS
computed in spatial form rather than material form. The **technique is
established; the application to UL bulk is novel** (see §P5.8.8).

#### §P5.8.4.4 Method D explicit: iterate on total $v$, with non-zero $g$

The continuous constraint (P5.8.1.e) discretizes (§P5.8.4.c) to
$C^A v = g^{(n)}$ with $g^{(n)} \ne 0$ in general. There are two
operationally distinct ways to *enforce* this discrete constraint
inside a Newton solver:

- **Method C — fluctuation primal, homogeneous constraint.** Define
  $v_\text{lin}(x) = \bar L^{(n)} \cdot (x - x_0)$ as a separate field
  computed each step; let the iterate be the fluctuation
  $\tilde v = v_\text{total} - v_\text{lin}$; the constraint becomes
  homogeneous: $C^A \tilde v = 0$. The corner pin sets
  $\tilde v_\text{corner} = 0$. This matches the existing
  `MortarSaddlePointSystem::Mult` API as written (which hardcodes
  $r_C = C \cdot u$ — homogeneous). Bulk equilibrium computes
  $F_\text{int}(v_\text{lin} + \tilde v)$ inside the K residual closure.

- **Method D — total primal, non-zero RHS.** The iterate is the total
  velocity $v$; the constraint is $C^A v = g^{(n)}$ with $g^{(n)}$
  computed from $\dot{\bar F}^{(n)}$ as in (P5.8.6.d). The corner pin
  sets $v_\text{corner} = \bar L^{(n)} \cdot x^{(n)}_\text{corner}$.
  Bulk equilibrium computes $F_\text{int}(v)$ directly — no addition
  step needed inside the K closure.

**ExaConstit Phase 5 picks Method D.** The reasons:

1. **Primal-field convention compatibility.** ExaConstit's
   `m_sim_state->GetPrimalField()` is the total velocity vector; all
   downstream plumbing (`UpdateEndCoords`, the constitutive law,
   `PostProcessingDriver`) expects total $v$. Method C would require
   maintaining $\tilde v$ as the iterate while still synthesizing total
   $v$ everywhere downstream — significant plumbing churn.

2. **Matches the patch test pattern.** The Phase 4 patch test driver
   uses Method D (total primal, $u_\text{init} = u_\text{lin}$,
   constraint RHS computed via $-(C u_\text{init} - g)$). Production
   stays consistent with the validated linear-case path.

3. **Localized API extension.** Method D needs one small Phase 4.3
   modification: a `SetConstraintRHS(const Vector&)` method on
   `MortarSaddlePointSystem` to install $g^{(n)}$ each step. Default
   behavior with no $g$ set remains the homogeneous form, so existing
   Phase 4.3 tests don't regress. This is the **Phase 5.0 batch**
   in the phasing (§P5.13).

4. **No K-closure-side complication.** The K closures in Method D are
   simply `mech_operator->Mult(v, r)` and `mech_operator->GetGradient(v)`
   — direct delegation. Method C's K closures would need to add
   $v_\text{lin}$ to the iterate before delegating, which couples the
   K closure to per-step macroscopic state and forces explicit
   $v_\text{lin}$ tracking.

The trade-off Method D accepts is the Phase 4.3 modification (Phase 5.0
batch) and the macroscopic $\bar F^{(n)}$ tracking in `MortarPbcManager`
(needed to compute $g$ from $\dot{\bar F}^{(n)} = \bar L^{(n)} \bar F^{(n)}$).
Both are small and contained.

**Method C as fallback.** If Method D runs into trouble during
implementation (e.g., the saddle-point Krylov struggles with the
non-zero RHS for some material configuration, or the $\bar F$ tracking
has a subtle drift the diagnostic doesn't catch), Method C is the
documented fallback. It requires an internal pivot — turning the
mech-operator's K closure into a wrapper that adds $v_\text{lin}$ —
but doesn't touch the saddle-point system itself. The validation tests
in §P5.8.11 should detect Method-D-specific failures early enough to
make the pivot before deep production integration.

### §P5.8.5 Operational realization: the classifier as reference-face cache

#### The cache mechanism

`BoundaryClassifier3D::Initialize()` walks `pmesh.GetVertex(...)` for each
boundary face element at simulation start and stores the corner coordinates
in `QuadFaceElement::coords` / `TriFaceElement::coords` arrays. These arrays
are owned by the classifier and are never updated thereafter.

`face_mortar_assembler_3d.cpp`'s `NonmortarJacobian`,
`NonmortarJacobianAxisAligned`, and the per-quadrature integration loops in
`AssemblePairConforming` and `AssembleQuadFacePairClipped` read from those
cached arrays — never from `pmesh`. So once the classifier is initialized,
the constraint matrix is built against a frozen-in-time copy of the initial
face geometry.

This is the operational realization of "build $C^A$ on $\Gamma_0$" without
maintaining a separate reference mesh: the classifier is itself the
reference-face cache. The cost is minimal (one `Vector` of corner coords
per boundary face element; ~$O(n^2)$ memory for an $n^3$ RVE).

#### What is not cached

Three things are *not* cached and must be computed per step:

- **The current $\bar L^{(n)}$** — pulled from `BCManager::GetVelocityGradient`
  on the time-dependent series.
- **The current $\bar F^{(n)}$** — tracked as `MortarPbcManager::m_macro_F`
  state, advanced per step via (P5.8.6.f).
- **The current macroscopic loading rate $\dot{\bar F}^{(n)} = \bar L^{(n)} \bar F^{(n)}$** —
  computed once per step from the above two, used in the constraint RHS.

#### What stays constant

- Sparsity pattern of $C^A$ (pair list, gtdof maps, Wohlmuth boundary tags).
- Numerical values of $C^A$ (the integrals (P5.8.4.a) on reference geometry).
- The per-element parent-coefficient matrices $A^{\hat E}_{ij}$.
- The non-conforming clipping topology (Phase 4.4 BVH and clip vertices)
  if non-conforming faces are used — these too are reference-mesh quantities
  cached at classifier init.
- The cached `m_rhs_geometric_factors` table in `MortarPbcManager`
  (§P5.4.4) — per LM row, $\hat \ell^{\hat E_0}_i \cdot (X^+ - X^-)_{\text{pair}(i)}$
  precomputed once.

#### Opposite-normal face pairs and the `mortar_node_perm`

A geometric fact worth stating explicitly here, because implementers
routinely get this wrong: **the conforming face-pair node permutation
$\texttt{mortar\_node\_perm}$ is never the identity on opposite-normal
periodic faces of an axis-aligned cube.** Face elements are conventionally
ordered CCW from outside the volume; for $+x$ and $-x$ faces "outside"
points in opposite directions, so the in-plane $(y, z)$ CCW order is
reversed between the two faces. `NodePermByCoordMatch` correctly reports
the involution $(0, 3, 2, 1)$ for matched quad-quad pairs (or $(0, 2, 1)$
for tri-tri pairs).

The conforming assembler's inner loop must account for this by mapping the
nm reference Gauss point into the *mortar's own* reference frame via the
perm-defined affine map (which `MortarRefFromPermutation` does correctly)
and then evaluating the standard reference basis there. **No further
reordering of the resulting shape values is needed or correct.** Applying
a second permutation on the shape values is a subtle bug class that is
invisible at $n = 2$ resolution (because every face element is then
corner-tagged with $M \equiv 1$ on its support, which makes the shape
permutation algebraically irrelevant) but breaks the Wohlmuth row-sum
identity at $n \geq 3$ with a precise $C v / g = 1.5$ signature at
corner-adjacent face-interior rows. See §P5.14.10 for the full failure
mode, mathematical reproduction, and detection diagnostic.

### §P5.8.6 The right-hand side update under UL

#### Spatial form RHS

From (P5.8.4.c):

$$
g_i^{(n)} = \bar L^{(n)} \cdot \int_{\Gamma_0} M_i(\xi)\,(x^{+,(n)}(\xi) - x^{-,(n)}(\xi))\, d\Gamma_0.
\tag{P5.8.6.a}
$$

This is a "weighted current-spatial-offset" integral. We can simplify using
the reference homologous-pair structure of the RVE. For axis-aligned RVEs,
the reference offset is constant per face pair: $(X^+ - X^-)_k = L\,\hat e_k$
on the $k$-th periodic axis. Under macroscopic deformation, by (P5.8.1.d) at
converged equilibrium and on the corner-pinned faces:

$$
x^{+,(n)} - x^{-,(n)} = \bar F^{(n)}\,(X^+ - X^-) + (\tilde u^+ - \tilde u^-).
\tag{P5.8.6.b}
$$

The fluctuation difference $\tilde u^+ - \tilde u^-$ vanishes at converged
equilibrium (that's exactly the constraint we're enforcing), so

$$
x^{+,(n)} - x^{-,(n)} \to \bar F^{(n)}\,(X^+ - X^-) = L\,\bar F^{(n)}\hat e_k
\quad\text{(at equilibrium)}.
\tag{P5.8.6.c}
$$

This is *constant per face pair* in the converged state, independent of
$\xi$. The integral in (P5.8.6.a) factors:

$$
g_i^{(n)} = \bar L^{(n)}\, \bar F^{(n)}\, (X^+ - X^-)_{\text{pair}(i)}\, \int_{\Gamma_0} M_i\, d\Gamma_0
        = \dot{\bar F}^{(n)}\, (X^+ - X^-)_{\text{pair}(i)}\, \hat \ell^{\hat E_0}_i,
\tag{P5.8.6.d}
$$

since $\bar L \bar F = \dot{\bar F}$ and $\int_{\Gamma_0} M_i\, d\Gamma_0
= \hat \ell^{\hat E_0}_i$ by Wohlmuth's lumped-row property on reference
geometry.

So in spatial form *at converged equilibrium*, the RHS reduces to a product
of (constant reference quantities) × ($\dot{\bar F}^{(n)}$) — the current
macroscopic deformation rate. We don't need to evaluate quadrature-point-level
current spatial offsets at all in the converged state.

#### Off-equilibrium considerations

Mid-Newton-iteration, the fluctuation difference $\tilde u^+ - \tilde u^-$
is *not* zero — that's what the Newton iteration is correcting. A naive
implementation of (P5.8.6.a) using current quadrature-point spatial offsets
would feed the *unconverged* fluctuation back into the constraint RHS,
causing the Newton to chase a moving target. The cleaner choice is to use
the converged-form RHS (P5.8.6.d), which depends only on $\dot{\bar F}^{(n)}$
and reference geometry, independent of the current $\tilde u$.

This is a subtle but important point: **the constraint RHS should be
expressed in terms of the macroscopic loading rate, not the current boundary
spatial offsets**. Operationally this means tracking $\bar F^{(n)}$ as
state and computing $\dot{\bar F}^{(n)} = \bar L^{(n)} \bar F^{(n)}$ for
the RHS update.

#### Practical $\bar F^{(n)}$ tracking

ExaConstit's existing infrastructure does not currently track macroscopic
$\bar F$ as state. We add a small piece of state in `MortarPbcManager`:
a $3 \times 3$ `mfem::DenseMatrix m_macro_F` updated each step via

$$
\bar F^{(n+1)} = \exp\bigl(\bar L^{(n+1)}\,\Delta t\bigr)\, \bar F^{(n)}
\quad\text{(matrix exponential)}
\tag{P5.8.6.e}
$$

or first-order via

$$
\bar F^{(n+1)} = \bar F^{(n)} + \bar L^{(n+1)}\,\bar F^{(n)}\,\Delta t.
\tag{P5.8.6.f}
$$

For consistency with ExaConstit's existing first-order time integration,
(P5.8.6.f) is the natural choice. The extra cost is one matrix-matrix
product per step — negligible.

The RHS (P5.8.6.d) is then computed once per step (not per Newton iteration)
using $\bar F^{(n)}$, and the saddle-point Newton iterates against this
fixed RHS until convergence. Fluctuations correct themselves; the
macroscopic loading is locked in.

#### Equivalence to the standard `essential_vel_grad` corner pin

For a corner DOF, where the fluctuation is *defined* to be zero (the corner
is pinned by Dirichlet, not by mortar), $x^{(n)}_\text{corner} = \bar F^{(n)} X_\text{corner}$
holds *exactly*, not just at equilibrium. The corner pin then is:

$$
v_\text{corner}^{(n)} = \dot{\bar F}^{(n)}\, X_\text{corner}
                    = \bar L^{(n)} \bar F^{(n)}\, X_\text{corner}
                    = \bar L^{(n)}\, x_\text{corner}^{(n)},
\tag{P5.8.6.g}
$$

where the last equality uses $x_\text{corner}^{(n)} = \bar F^{(n)} X_\text{corner}$
(pinned, no fluctuation). This is exactly what ExaConstit's existing
`essential_vel_grad` BC machinery computes via $v(x) = \bar L \cdot (x^{(n)} - x_0)$
(with $x_0 = 0$ for axis-aligned RVEs with the standard origin). **No
formula change is needed for the corner pin in mortar PBC; the existing
`essential_vel_grad` projection, restricted to the corner TDOF subset,
gives the right answer.**

### §P5.8.7 Why the corner pin alone is not sufficient — the need for the mortar constraint

A common question: if the corner pin enforces $v_\text{corner} = \bar L\, x_\text{corner}$
at 8 vertex DOFs, why isn't that enough to enforce periodicity?

**Because the corner pin enforces only the macroscopic affine velocity at 8
points. The macroscopic field at any other point on the face is determined
by the macroscopic loading, but the *fluctuation* part of the velocity
field is not controlled.** The mortar constraint enforces that the
fluctuation part is periodic — i.e., that homologous interior face DOFs
have *equal* fluctuation values across opposing faces.

Together:
- The corner pin (8 corners × 3 components = 24 strong Dirichlet TDOFs)
  fixes the macroscopic affine velocity exactly at the 8 RVE corners.
- The mortar constraint (one LM row per interior face / edge nonmortar TDOF
  after Wohlmuth corner / edge sentinel removal) enforces fluctuation
  periodicity on all other boundary TDOFs.

This split — strong corner pin + weak mortar fluctuation periodicity — is
what gives the formulation its variational consistency and what makes the
homogenized stress recoverable from the assembled Lagrange multipliers
(Lopes et al. 2021, eq. 37).

### §P5.8.8 What is novel and what is established

#### Established techniques being reused

1. **Reference-integrated mortar PBC** with Wohlmuth dual basis —
   established by Reis & Pires 2014, Lopes et al. 2021 in TL
   displacement-primal frameworks for computational homogenization. The
   technique itself is mature; the biorthogonality preservation argument is
   rigorous and well-cited.
2. **Wohlmuth dual basis with corner / edge modifications** — established by
   Wohlmuth 2000–2007 (collected in *Discretization Techniques and Iterative
   Solvers Based on Domain Decomposition*, Springer 2001). The wirebasket
   hierarchy used by Phase 4 is a direct application of these modifications.
3. **Updated Lagrangian crystal plasticity** — established for ExaConstit's
   bulk equilibrium via standard finite-element formulations (Bonet & Wood,
   *Nonlinear Continuum Mechanics for FEM*, CUP 2008; Belytschko, Liu, &
   Moran, *Nonlinear FEM*, Wiley 2014). The mech_operator + UpdateEndCoords
   pattern is conventional.
4. **`MortarSaddlePointSystem` adapter** for embedding the saddle-point
   structure into a regular `mfem::Operator` interface — Phase 4.3 / Batch
   R, conceptually based on standard saddle-point Krylov techniques (Benzi,
   Golub, & Liesen, *Acta Numer.* 14, 2005).

#### What is novel — to my knowledge

The novelty is in the *combination*: applying the reference-integration
mortar PBC technique to a UL bulk crystal plasticity solver, and doing so
with velocity primal in the mortar weak form. Specifically:

(N1) **The matrix-construction-is-primal-blind argument** in §P5.8.4 is, I
believe, the right way to justify the application but does not appear
explicitly in either the RVE-PBC literature (which is TL) or the contact
mortar literature (which is UL but for a different problem with sliding
interfaces). The argument hinges on the observation that the matrix
*values* are pure geometric integrals — the same numbers in both
formulations — so the literature's reference-integration technique transfers
to UL with no algebraic modification.

(N2) **The classifier-as-reference-face-cache realization** in §P5.8.5 is
specific to the ExaConstit / MFEM architecture. By keeping the classifier
authoritative for face geometry and decoupled from the moving pmesh, we get
the operational equivalent of "integrate on reference" without maintaining
a full reference mesh.

(N3) **The macroscopic $\bar F$ tracking via (P5.8.6.f)** is needed for the
constraint RHS in spatial form. ExaConstit's existing infrastructure
doesn't track this — it's a new piece of state. The convention that
$\bar F$ is updated per step using the current $\bar L^{(n+1)}$ via a
first-order matrix product is the same convention used in TL homogenization
codes (e.g., the macro-scale time integration in Lopes et al.) but applied
inside a UL solver loop.

To my knowledge based on basic literature searches, no paper combines (i)
mortar PBC, (ii) UL bulk formulation with velocity primal, and (iii)
finite-strain crystal plasticity. Most computational homogenization papers
using mortar PBC (Reis & Pires; Lopes; Schneider & Andra et al.) work in TL
displacement-primal frameworks; UL crystal plasticity papers (e.g. Nervi &
Idiart 2015, *IJP* 65; Roters et al. *DAMASK*) typically use Lagrange
multipliers or master-slave constraints, not mortar.

#### Why this should work despite being novel

The reasoning chain is:

(R1) The **continuous constraint** (P5.8.1.e) is well-defined on the current
configuration regardless of bulk formulation choice.

(R2) **The mortar weak form** (P5.8.2.a) admits two integration domain
choices, both valid; each gives a convergent discrete approximation
(Wohlmuth 2000, Theorem 4.1).

(R3) Choice A (reference integration) preserves Wohlmuth biorthogonality
exactly (P5.8.3.c) on axis-aligned RVE faces. This is the same property
exploited by all RVE-PBC mortar literature.

(R4) The **matrix entries** under Choice A are configuration-blind
geometric integrals; the matrix is built once and reused for the simulation
lifetime regardless of how the pmesh advances under UL.

(R5) The **operator action** $C v$ on velocity TDOFs is a pure linear
operation; the matrix doesn't know whether it's being applied to $u$ or $v$.
So the RVE-PBC literature's matrix construction transfers verbatim to a
velocity-primal UL setting.

(R6) The **right-hand side** updates per step using current $\bar L^{(n)}$
and tracked $\bar F^{(n)}$, giving the discrete velocity-form constraint
$C v = g^{(n)}$ (P5.8.4.c).

(R7) The **convergence properties** of the discrete approximation are
inherited from the standard mortar method (Wohlmuth 2000; Bernardi-Maday-Patera
1994).

The argument is reasonably tight. The places where it could fail are
documented in §P5.8.9.

### §P5.8.9 Gotchas — where this picture may break down

The following are the documented or anticipated failure modes. Each is
flagged with **severity** and a **diagnostic** (how the user / developer
would notice the failure).

#### Gotcha 1 (severe): Severe boundary distortion violates homologous-pair geometric meaning

**Failure mode**: When some part of the RVE boundary undergoes severe local
distortion — e.g., a face element collapses near zero area, a face flips
over, or strong macro-shear bands cause faces to rotate by > 90° — the
physical meaning of "homologous-point pair" can become ambiguous. The
reference-coordinate-defined pairing $(X^+, X^-)$ is well-defined
mathematically but may no longer correspond to physically meaningful
periodicity.

**Diagnostic**: Volume-averaged $\langle F \rangle$ drifts substantially
from the prescribed $\bar F^{(n)}$; the $\tilde v$ visualization shows
non-trivial (non-periodic) fluctuation that doesn't go to zero at converged
homogeneous steps; Newton convergence rate degrades (typically linear or
worse). The Hill-Mandel diagnostic (§P5.10.3) flags drift > 1e-10 relative.

**Mitigation in Phase 5**: this regime is outside the scope of mortar PBC
based on either configuration (current-integration mortar would suffer from
the same physical breakdown). Document that ExaConstit's mortar PBC is
intended for "macro-scale-uniform" deformation regimes where the RVE
remains a meaningful representative volume; for shear-band-forming or
strain-localizing problems, the user should consider domain-decomposition
methods or adaptive remeshing instead.

#### Gotcha 2 (significant): The constraint RHS expression assumes $\bar F^{(n)}$ tracking is consistent with the bulk integration

**Failure mode**: $\bar F^{(n)}$ is updated externally using the user's
prescribed $\bar L^{(n)}$. If the user's prescribed $\bar L^{(n)}$ does not
faithfully describe the macroscopic loading the bulk equilibrium "wants"
(e.g., because of large boundary fluctuation that the corner pin cannot
suppress, or because of nonlinearity in the macroscopic response that the
user did not account for in time-stepping $\bar L$), then the RHS computed
from $\bar F^{(n)}$ is inconsistent with the actual macroscopic deformation
in the bulk.

**Diagnostic**: $\langle F \rangle$ from the bulk solution does not match
the externally-tracked $\bar F^{(n)}$ at converged steps. Hill-Mandel power
balance non-zero.

**Mitigation in Phase 5**: register $\langle F \rangle$ computation and the
Hill-Mandel power balance as per-step outputs. Cross-check against
$\bar F^{(n)}$. If the user's $\bar L$ time-history gives a $\bar F^{(n)}$
inconsistent with $\langle F \rangle$, the right action is *on the user
side*: they need to refine their loading description. This is not an
implementation bug.

#### Gotcha 3 (moderate): The corner pin assumes $X_0 = (0,0,0)$ for axis-aligned RVEs with origin-anchored loading

**Failure mode**: ExaConstit's `essential_vel_grad` allows a user-supplied
`vgrad_origin` $x_0$. For mortar PBC consistency, $x_0$ should be the
*reference* origin, and the corner pin should use $\bar L \cdot (x^{(n)} - x_0^{(n)})$
where $x_0^{(n)} = \bar F^{(n)} X_0$ tracks the deforming origin. With the
existing `essential_vel_grad` machinery, $x_0$ is treated as fixed in space.
For $X_0 = 0$, $x_0^{(n)} = 0$ trivially and no issue arises. For
$X_0 \ne 0$, the existing machinery gives wrong corner pin values.

**Diagnostic**: Volume-averaged $\langle F \rangle$ has a non-zero offset
from the prescribed $\bar F^{(n)}$ proportional to $X_0$.

**Mitigation in Phase 5**: document that mortar PBC assumes $X_0 = 0$ at
construction. If the user's RVE has a non-zero reference origin, recommend
they translate the mesh to put the origin at $(0,0,0)$ before running. This
is the standard convention in computational homogenization.

**Future fix (Phase 5+ or Phase 6)**: extend `essential_vel_grad` to
optionally accept a "reference origin" interpretation that uses
$x_0^{(n)} = \bar F^{(n)} X_0$. Small change; deferred until a real user
needs it.

#### Gotcha 4 (moderate): Wohlmuth biorthogonality at $p \ge 2$ breaks even on the reference

**Failure mode**: At $p \ge 2$, the reference parent-element mass matrix
$\hat M^{\hat E}_{ij}$ has off-diagonal entries that don't go away with
constant $|J|$. The closed-form coefficients $A^{\hat E}_{ij}$ no longer
give exact biorthogonality on assembly; they give an *approximate*
biorthogonality with $O(h)$ consistency error in $H^1$ rates.

**Diagnostic**: Patch test passes at engineering tolerance but $H^1$
convergence rate falls below the optimal $h^p$.

**Mitigation in Phase 5**: $p = 1$ only. Phase 6's LOR machinery resolves
this — refining the periodic-face submesh by $p-1$ levels and running the
mortar pipeline at LOR Q1/P1 restores the constant-$|J|$ property at the
LOR refinement level.

#### Gotcha 5 (low): Time integrator mismatch under multi-stage schemes

**Failure mode**: ExaConstit currently uses first-order time stepping,
which is consistent with the (P5.8.6.f) update for $\bar F^{(n)}$. If a
higher-order time scheme (Newmark, generalized-α, BDF2) is added later,
the macroscopic $\bar F$ tracking must use the *same* multi-stage scheme as
the bulk equilibrium time integration; otherwise the constraint RHS is
inconsistent with the bulk's intermediate-stage configurations.

**Diagnostic**: stress/strain hysteresis at converged steady-state; growing
$|\langle F \rangle - \bar F^{(n)}|$ over many time steps.

**Mitigation in Phase 5**: not relevant — first-order matches first-order.
Documented for forward extensibility.

#### Gotcha 6 (low): Non-conforming face matching on severely deformed reference geometry

**Failure mode**: The Phase 4.4 BVH and Sutherland-Hodgman clip topology
are built once on the reference faces. If the reference geometry has
degenerate or near-degenerate face elements (e.g., a Neper-generated mesh
with extreme aspect ratios near grain boundaries), the clipping topology
may have quality issues that aren't apparent until the simulation deforms
significantly.

**Diagnostic**: $C v$ residual doesn't go to zero at converged steps; large
condition number on `SaddlePointSolver`.

**Mitigation in Phase 5**: standard mesh-quality sanity checks at classifier
construction (boundary face-element aspect ratio, minimum area). If a face
element's reference area is below a tolerance, emit a warning; suggest mesh
quality improvement.

#### Gotcha 7 (low): The constraint primal must match the bulk primal

**Failure mode**: If for any reason the bulk equilibrium ends up using a
displacement-form residual (e.g., because the user accidentally sets a
displacement-style essential_vel BC), the constraint $C v = g$ is operating
on the wrong primal. The mortar constraint enforces velocity periodicity
when the bulk is enforcing displacement equilibrium — internally inconsistent.

**Diagnostic**: Newton fails to converge from the first iteration; residual
doesn't decrease at all.

**Mitigation in Phase 5**: assert at `MortarPbcManager` construction that
the mech_operator's primal field is velocity (it is —
`m_sim_state->GetPrimalField()` is always the velocity TDOF vector in
ExaConstit). This is structurally guaranteed by ExaConstit's design but
worth asserting defensively.

#### Gotcha 8 (theoretical): Non-axis-aligned RVEs

**Failure mode**: For non-axis-aligned reference geometry (e.g., a
hexagonal RVE for hcp polycrystals, or a curvilinear-boundary RVE for
additive manufacturing), $|J^{E_0}_\text{face}|$ is not constant per
element — it varies even on the reference. The constant-$|J|$ argument in
§P5.8.3 fails; biorthogonality is approximate even before any deformation.

**Diagnostic**: Phase 4.4 patch test fails on a non-axis-aligned RVE — this
is detectable at the unit-test level before the full simulation runs.

**Mitigation in Phase 5**: out of scope. Phase 5 supports axis-aligned RVEs
only, validated by the Phase 4.1 patch test suite. Curvilinear RVEs are an
explicit non-goal (architecture doc §13.3 long-term, with Tribol as the
likely matching backend).

### §P5.8.10 Potential fallbacks if the assumption fails

| Failure mode | Recovery path | Cost |
|---|---|---|
| Gotcha 1 (severe distortion) | Out of scope for mortar PBC. Switch to domain decomposition or adaptive remeshing. | High (different formulation) |
| Gotcha 2 ($\bar F$ inconsistency) | User-side: refine $\bar L$ time history. | Low |
| Gotcha 3 ($X_0 \ne 0$) | Pre-translate mesh, OR extend `essential_vel_grad` for reference-origin interpretation. | Low (translate) or Medium (BC extension) |
| Gotcha 4 ($p \ge 2$ biorthogonality) | Phase 6 LOR. | Medium (Phase 6 work) |
| Gotcha 5 (multi-stage time integration) | Match macroscopic $\bar F$ stepping to bulk scheme. | Low at higher-order time scheme implementation |
| Gotcha 6 (mesh quality) | Mesh quality improvement on user side. | Low |
| Gotcha 7 (primal mismatch) | Assert defensively; should never trigger. | Trivial |
| Gotcha 8 (non-axis-aligned RVE) | Out of scope for Phase 5. Phase 5+ direction: Tribol-based matching, or Popp-Wohlmuth consistent biorthogonalization on the reference. | High (new infrastructure) |

The most consequential fallback is for Gotcha 8 if non-axis-aligned RVEs
become a priority: **switch to consistent biorthogonalization on the
reference**. This is the same machinery contact mortar codes use, applied to
reference geometry instead of current. The construction is:

1. Per face element, compute $\hat M^{E_0}_{ij} = \int_{\hat E} N_i N_j |J^{E_0}|\, d\hat\xi$
   on the *reference* geometry (which has non-constant $|J^{E_0}|$ for
   non-axis-aligned RVEs but is still time-invariant).
2. Solve $\hat M^{E_0} A^{E_0} = \mathrm{diag}(\hat \ell^{E_0})$ for the
   per-element coefficient matrix $A^{E_0}_{ij}$.
3. Use $A^{E_0}_{ij}$ instead of the closed-form $A^{\hat E}_{ij}$ in the
   dual basis evaluation.
4. Build $D$ on the reference using these element-specific coefficients —
   $D$ becomes diagonal again.

This is computed once at preprocessing (not per Newton iteration as in
contact mortar) because the reference geometry is time-invariant. The
overhead is per-element linear-solve cost at preprocessing; runtime is
unchanged.

This Gotcha 8 fallback is "Phase 5+ infrastructure" — flag in the doc, do
not implement in Phase 5. If a user demands curvilinear RVEs, the fallback
is well-defined and the implementation cost is bounded.

### §P5.8.11 Validation strategy for the UL adaptation

Phase 4 patch tests validate the technique at small deformation (single
load step, ~5% strain). Phase 5 needs explicit validation that the build-once
approach extends correctly to UL multi-step finite deformation.

#### Test 1: large total strain via multi-step monotonic loading (new)

**Setup**: a Q1 hex RVE, single-grain Neohookean material, simple shear
loading at $\bar L_{xy} = 0.01\,/\text{s}$ for 100 time steps with
$\Delta t = 0.5\,\text{s}$. Total accumulated strain at end: ~50%.

**PASS criteria**:
- $\langle F \rangle^{(n)}$ matches tracked $\bar F^{(n)}$ at every step to
  FP precision (Hill–Mandel power balance).
- $\tilde v$ visualization is small in magnitude relative to $v_\text{lin}$
  at every step (indicating clean fluctuation periodicity throughout).
- Newton convergence rate is asymptotically quadratic at every step
  (indicating saddle-point system is well-conditioned and consistent).

If all three pass, the build-once-on-reference approach is empirically
validated for large-deformation single-grain UL.

#### Test 2: heterogeneous polycrystal (existing, extend)

The existing Phase 4 strip-split and checkerboard tests at single load step
validate matrix construction. Extend to 50 time steps of monotonic loading
(same $\bar L_{xy}$ ramp), with periodically-saved checkpoints for restart
testing.

**PASS criteria**: same as Test 1, plus matched homogenized stress
$\langle \sigma \rangle$ at intermediate checkpoints to FP precision (no
drift between save / restore cycles indicates state consistency).

#### Test 3: cross-validation against a TL prototype (optional but valuable)

If we have access to a TL displacement-primal mortar PBC implementation
(Lopes' code, or a Python prototype written for cross-validation), run the
same problem in both and compare:

- Homogenized stress $\langle \sigma \rangle$ at every step
- $\tilde u$ field at every step
- $\lambda$ values at every step

PASS: agreement to FP precision (modulo time integration order — TL with
direct $\bar F$ specification vs UL with $\bar F^{(n)}$ derived from
$\bar L^{(n)}$ time integration may have $O(\Delta t)$ differences which
are expected).

This test, if it can be constructed, is the strongest validation that the
UL adaptation is correct: the same physics solved by two different
formulations should agree.

#### Test 4: Hill-Mandel theorem at every step

For any converged state, the Hill-Mandel condition states

$$
\bar P : \dot{\bar F} = \frac{1}{|\Omega_0|} \int_{\Omega_0} P : \dot F\, d\Omega_0,
\tag{P5.8.11.a}
$$

i.e., the macroscopic stress power equals the volume-average of the
microscopic stress power. This must hold to FP precision at any converged
state, regardless of step count or deformation magnitude.

**Implementation**: register Hill-Mandel power balance as a per-step
diagnostic, computed by `PostProcessingDriver` from the macroscopic
$(\bar P, \dot{\bar F})$ and the volume-averaged microscopic
$(\langle P \rangle, \langle \dot F \rangle)$. Print residual; CI flag if
> 1e-10 relative.

### §P5.8.12 References

- Belytschko, T., Liu, W.K., & Moran, B. *Nonlinear Finite Elements for
  Continua and Structures*. Wiley, 2014.
- Benzi, M., Golub, G.H., & Liesen, J. "Numerical solution of saddle point
  problems." *Acta Numerica* 14 (2005): 1–137.
- Bernardi, C., Maday, Y., & Patera, A.T. "A new nonconforming approach to
  domain decomposition: the mortar element method." *IMA J. Numer. Anal.*
  14 (1994): 1–13.
- Bonet, J., & Wood, R.D. *Nonlinear Continuum Mechanics for Finite Element
  Analysis*. Cambridge UP, 2008.
- Hill, R. "Elastic properties of reinforced solids: some theoretical
  principles." *J. Mech. Phys. Solids* 11 (1963): 357–372. (Hill-Mandel
  theorem.)
- Lopes, I.A.R., Ferreira, B.P., & Pires, F.M.A. "On the efficient
  enforcement of uniform traction and mortar periodic boundary conditions
  in computational homogenisation." *CMAME* 384 (2021): 113930.
- Mandel, J. "Contribution théorique à l'étude de l'écrouissage et des lois
  de l'écoulement plastique." *Proc. 11th Int. Congress on Applied
  Mechanics*, Munich (1965).
- Nervi, J.E. & Idiart, M.I. *International Journal of Plasticity* 65
  (2015): 30–52.
- Pazner, W. & Kolev, T. (LOR; see Phase 6 references.)
- Popp, A., Gee, M.W., & Wall, W.A. "A finite deformation mortar contact
  formulation using a primal–dual active set strategy." *IJNME* 79 (2009):
  1354–1391.
- Popp, A., Wohlmuth, B.I., Gee, M.W., & Wall, W.A. "Dual quadratic mortar
  finite element methods for 3D finite deformation contact." *SIAM
  J. Sci. Comput.* 34 (2012): B421–B446.
- Puso, M.A. "A 3D mortar method for solid mechanics." *IJNME* 59 (2004):
  315–336.
- Puso, M.A. & Laursen, T.A. "A mortar segment-to-segment contact method for
  large deformation solid mechanics." *CMAME* 193 (2004): 601–629.
- Reis, F.J.P. & Pires, F.M.A. "A mortar based approach for the enforcement
  of periodic boundary conditions on arbitrarily generated meshes." *CMAME*
  274 (2014): 168–191.
- Wohlmuth, B.I. "A mortar finite element method using dual spaces for the
  Lagrange multiplier." *SIAM J. Numer. Anal.* 38 (2000): 989–1012.
- Wohlmuth, B.I. *Discretization Techniques and Iterative Solvers Based on
  Domain Decomposition*. Springer LNCSE 17, 2001.
- Wohlmuth, B.I. & Popp, A. Chapter in *Mortar Methods for Single- and
  Multi-Field Applications*, Springer (2014).

---

## §P5.9 Multi-region, assembly mode, and GPU compatibility

These are properties of the Phase 4 stack, not of the integration:

- **Multi-region**: the constraint depends on geometric topology only, not
  material. ExaCMech / UMAT / MultiExaModel work without change. Phase 4
  strip-split and checkerboard tests prove this on the linear-elastic side;
  behavior carries through to nonlinear K.
- **Assembly**: Day 1 default is EA for the constraint (matches PA K for
  GPU). HypreParMatrix path retained for debug / direct-solver use. Both
  validated bit-tight in Phase 4.
- **GPU**: Phase 4.3.B status applies — forward `Mult` is GPU-clean;
  atomic-add `MultTranspose` is in flight. The integration inherits whatever
  GPU support Phase 4.3.B delivers.

---

## §P5.10 Output and post-processing

`PostProcessingDriver` already computes $\langle F \rangle$ and
$\langle \sigma \rangle$ — Phase 5 does *not* duplicate these. The mortar
PBC adds three specific outputs:

### §P5.10.1 Per-step diagnostic: $\langle L \rangle$

A new per-step quantity printed alongside the existing volume-averaged
stress / strain rate: the volume-averaged spatial velocity gradient
$\langle L \rangle = \langle \nabla v \rangle$. For a correctly converged
mortar PBC step, $\langle L \rangle$ should equal the prescribed
$L_\text{macro}$ to FP precision. Drift is the canonical diagnostic for a
misbehaving constraint.

Implementation: extend `PostProcessingDriver` with a
`RegisterVelocityGradientAverage` call analogous to the existing stress /
strain rate registrations. Computed via the existing volume-averaging
machinery on the velocity gradient quadrature function.

### §P5.10.2 ParaView field: $\tilde v = v - v_\text{lin}$

For visualization of the heterogeneous fluctuation. New `ParGridFunction`
field "v_tilde" registered with the existing `PostProcessingDriver`'s
ParaView pipeline. Computation is one
`MortarPbcManager::ComputeFluctuationField` call per output step, using
current mesh nodes to compute $v_\text{lin} = \bar L^{(n)} \cdot (x^{(n)} - x_0)$
in spatial form.

This is the analogue of $\tilde u$ in the test drivers, adapted for the
velocity primal. Useful for spotting localization, identifying whether the
fluctuation is small enough to justify a coarser RVE, and validating that
periodicity holds visually.

### §P5.10.3 Hill-Mandel power balance

Per §P5.8.11 Test 4: a per-step diagnostic computing the relative residual

$$
r^{(n)}_\text{HM} = \frac{|\bar P^{(n)} : \dot{\bar F}^{(n)} - \langle P^{(n)} : \dot F^{(n)} \rangle|}{|\bar P^{(n)} : \dot{\bar F}^{(n)}|}.
$$

Should be $< 10^{-10}$ at every converged step. CI failure flag if it
exceeds this threshold.

Implementation: `MortarPbcManager::ComputeHillMandelPowerBalance()` (already
declared in the public API, §P5.4.1). Macroscopic $\bar P^{(n)}$ from the
assembled $\lambda$ via Lopes et al. eq. 37; macroscopic $\dot{\bar F}^{(n)}$
already cached in `m_macro_Fdot`. Volume-average $\langle P : \dot F \rangle$
from the existing `PostProcessingDriver` machinery.

**Implementation note** (~30 LOC of new code in `MortarPbcManager`): Lopes
et al. eq. 37 expresses the macroscopic first PK stress as a rank-1 sum
over Lagrange-multiplier rows weighted by reference homologous-pair
offsets:

$$
\bar P^{(n)} = \frac{1}{|\Omega_0|} \sum_i \lambda_i^{(n)} \otimes (X_i^+ - X_i^-)
$$

where the sum is over all LM rows in the constraint, $\lambda_i^{(n)}$
is the converged multiplier for row $i$ at step $n$, and $(X_i^+ - X_i^-)$
is the reference homologous-pair offset for that row. Both are
already cached: $\lambda$ is `MortarPbcManager::m_lambda`, and the
offsets are inside `m_rhs_geometric_factors` (which stores
$\hat \ell^{\hat E_0}_i \cdot (X^+ - X^-)_{\text{pair}(i)}$ — divide
out the lumped row-sum to recover the offset). The implementation is
a single MPI-reduce loop:

```cpp
double ComputeHillMandelPowerBalance() const {
    // 1) Build local rank-1 sum into a 3×3 dense matrix.
    mfem::DenseMatrix P_bar_local(3); P_bar_local = 0.0;
    const auto LAMBDA  = m_lambda.HostRead();
    const auto FACTORS = m_rhs_geometric_factors.HostRead();
    const int  nrows   = m_builder->NumLocalRows();
    for (int i = 0; i < nrows; ++i) {
        // factors[i] = lumped_row_sum * (X+-X-); recover offset.
        const double inv_lump = 1.0 / m_classifier->LumpedRowSum(i);
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                P_bar_local(k, l) += LAMBDA[i] * FACTORS[i*3 + l] * inv_lump;
            }
        }
    }
    // 2) MPI_Allreduce to global P_bar.
    mfem::DenseMatrix P_bar(3);
    MPI_Allreduce(P_bar_local.Data(), P_bar.Data(), 9,
                  MPI_DOUBLE, MPI_SUM, m_pmesh.GetComm());
    P_bar *= 1.0 / m_volume_omega_0;  // cached at construction
    // 3) Compute the residual.
    const double bar_power = MatrixDoubleContraction(P_bar, m_macro_Fdot);
    const double micro_power = m_post_proc->ComputeVolumeAvgPDotF();
    return std::abs(bar_power - micro_power) /
           std::max(std::abs(bar_power), 1.0e-30);
}
```

The classifier's `LumpedRowSum(i)` accessor for individual LM rows is
existing functionality in `BoundaryClassifier3D`. The `m_volume_omega_0`
cached value comes from a single `pmesh.GetGlobalNE()`-weighted sum at
construction.

### §P5.10.4 Why no new output category

These pieces fit cleanly into the existing `[PostProcessing.Projections]`
and `[Visualizations]` infrastructure as additional projection names:

```toml
[PostProcessing.Projections]
enabled_projections = ["stress", "von_mises", "velocity_gradient_avg",
                       "v_tilde", "hill_mandel"]
```

No new output table; the user just enables the projections by name.

---

## §P5.11 CMake / build system

`src/mortar_pbc/` becomes a new subdirectory linked into `exaconstit_mech`.
Axom is conditionally enabled for non-conforming support. Minimal changes
to the existing build-system patterns:

```cmake
# src/CMakeLists.txt — add the subdirectory.
add_subdirectory(mortar_pbc)

# src/mortar_pbc/CMakeLists.txt
set(MORTAR_PBC_SOURCES
    boundary_classifier_3d.cpp
    constraint_builder_3d.cpp
    face_mortar_assembler_3d.cpp
    face_mortar_assembler_clipped_3d.cpp
    mortar_assembler_2d.cpp
    mortar_constraint_operator.cpp
    saddle_point_solver.cpp
    tile_partition_3d.cpp
    boundary_helpers_3d.cpp
    mortar_pbc_manager.cpp           # new (Phase 5)
)

add_library(exaconstit_mortar_pbc STATIC ${MORTAR_PBC_SOURCES})
target_link_libraries(exaconstit_mortar_pbc
    PUBLIC mfem MPI::MPI_CXX
    PRIVATE caliper)

if (MORTAR_PBC_HAS_AXOM)
    target_compile_definitions(exaconstit_mortar_pbc PUBLIC MORTAR_PBC_HAS_AXOM)
    target_link_libraries(exaconstit_mortar_pbc PUBLIC axom::core axom::primal axom::spin)
endif()

target_include_directories(exaconstit_mortar_pbc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
```

The `exaconstit_mech` library links against `exaconstit_mortar_pbc`. The
`system_driver` and `option_parser_v2` translation units include
`mortar_pbc/mortar_pbc_manager.hpp` as needed.

When `ENABLE_AXOM=OFF`, the non-conforming code path is disabled at compile
time. Phase 4 sandbox already validates that the mortar code compiles
clean both with and without Axom.

---

## §P5.12 Testing strategy

Three layers, integrating §P5.8.11:

### §P5.12.1 Unit tests (new)

In `test/mortar_pbc/`:

- `test_mortar_pbc_manager_construct` — basic construction / destruction
  cycle, memory-leak-clean under valgrind on a small mesh.
- `test_mortar_pbc_manager_corner_tdofs` — verify the 24 corner TDOFs are
  correctly identified and consistent across np=1, 4, 7.
- `test_mortar_pbc_manager_macro_F_update` — verify
  `UpdateMacroscopicF` produces correct $\bar F^{(n)}$ values for a
  multi-step monotonic loading sequence (compare against analytic matrix
  exponential or trapezoidal rule integration).
- `test_mortar_pbc_manager_constraint_rhs` — verify `UpdateConstraintRHS`
  produces a constant-per-pair RHS matching (P5.8.6.d) on a deformed mesh.

### §P5.12.2 End-to-end small-problem tests (new)

In `test/data/` with corresponding `.toml` files:

- `mortar_pbc_linear_elastic.toml` — homogeneous linear-elastic 4×4×4 hex
  RVE under simple shear, single time step. Reference: $\langle F \rangle = F_\text{macro}$
  and Hill-Mandel residual below 1e-12.
- `mortar_pbc_neohookean_50pct.toml` — single-grain Neohookean RVE, 100
  time steps simple shear ramping to ~50% strain (Test 1 of §P5.8.11).
  Hill-Mandel residual below 1e-10 at every step; $\tilde v$ stays small;
  Newton converges quadratically.
- `mortar_pbc_voce_polycrystal.toml` — small (~50 grain) polycrystal with
  ExaCMech Voce hardening, 50 time steps under tension. Reference:
  homogenized stress matches a known-good baseline; Hill-Mandel below 1e-10.
- `mortar_pbc_heterogeneous_strip_multistep.toml` — Phase 4 strip-split
  promoted to 50 time steps (Test 2 of §P5.8.11).
- `mortar_pbc_checkerboard_multistep.toml` — Phase 4 checkerboard promoted
  to 50 time steps.
- `mortar_pbc_restart.toml` — checkpoint at step 25, restart at step 26,
  verify $\bar F^{(n)}$ and $\lambda$ are restored cleanly.

Run all six at np=1, 4, 7. Add to CI.

### §P5.12.3 Regression / performance benchmarks

- A 32³ hex polycrystal with 200 grains, ExaCMech FCC, 50 time steps under
  tension — compare wall-time and memory between mortar PBC enabled and
  disabled (with appropriately matched conventional Dirichlet for the
  disabled case). Goal: mortar PBC overhead < 30% on CPU, < 50% on GPU.
- A scaling study at np=1, 4, 16, 64, 256 (up to whatever Phase 4.2
  supports) — the saddle-point solver should be the dominant cost; classifier
  setup should be < 10% of total time; constraint matrix construction
  should happen once at startup and be < 5% of total time.

### §P5.12.4 Test the "feature off" path

Critical regression check: every existing `test/data/*.toml` test that
doesn't enable mortar PBC must produce bit-identical results before and
after Phase 5. This validates the inert-when-disabled invariant. Run as
part of CI on every PR.

---

## §P5.13 Phasing

Each batch lands focused, locally-testable work; the test suite stays green
at every step (including the "feature off" regression check).

```
Phase 5.0 — MortarSaddlePointSystem constraint-RHS extension
├── 5.0.A  Add `SetConstraintRHS(const Vector& g)` and
│           `ClearConstraintRHS()` methods to
│           `test/mortar_pbc/mortar_saddle_point_system.{hpp,cpp}`.
│           Add private `m_g_rhs` member (mfem::Vector); store a
│           shallow copy via `MakeRef` to avoid extra allocation.
│           Modify `Mult(x_block, r_block)` so that, when an RHS is
│           installed, the constraint-side residual becomes
│              r_C_block = C * u - g
│           instead of the homogeneous form. Default state (no RHS
│           set, or after `ClearConstraintRHS()`) keeps the existing
│           homogeneous behavior.
├── 5.0.B  Add a unit test in
│           `test/mortar_pbc/test_mortar_saddle_point_system.cpp`
│           that exercises both code paths:
│              (i)  no RHS set → r_C = C·u (existing behavior).
│              (ii) RHS set    → r_C = C·u - g (new behavior).
│           Verify on a small synthetic constraint that the residual
│           vanishes when u is constructed to satisfy C·u = g exactly.
├── 5.0.C  Verify all existing Phase 4.3 tests still pass — the
│           default behavior is unchanged, so existing tests should
│           be untouched. Cross-check with `--constraint-storage=ea`
│           patch test on np = {1, 4, 7}.

         ↓ (gate: existing tests green; new RHS-mode test passes;
              code-review sign-off on the small Phase 4.3 modification)

Phase 5.1 — Promote test/mortar_pbc/ → src/mortar_pbc/
├── 5.1.A  Move files; update CMake; verify existing tests still pass.
├── 5.1.B  Linking from main exaconstit_mech library; verify the main
│           binary builds with mortar PBC code present but unused.

         ↓ (gate: existing test/mortar_pbc/ tests still green)

Phase 5.2 — Options: snap_tol, lor_depth, SaddlePointSolverOptions
├── 5.2.A  Add to MeshOptions / SolverOptions; parse [Mesh] +
│           [Solvers.SaddlePoint] tables; validate.
├── 5.2.B  Verify TOML without mortar enabled parses identically before/after.

         ↓ (gate: option parsing test green; existing TOMLs unchanged)

Phase 5.3 — MortarPbcManager class (depends on Phase 5.0)
├── 5.3.A  Class skeleton: constructor takes K residual / K Jacobian
│           closures (per §P5.4.1) and delegates to existing Phase 4
│           classifier/builder/operator/solver setup; F̄ initialized to I.
│           Constructor builds the MortarSaddlePointSystem internally
│           from the closures and the EA constraint operator.
├── 5.3.B  Corner-TDOF identification: walk the BoundaryClassifier3D
│           CornerInfo3D records, build the 24-element Array<int>.
├── 5.3.C  UpdateMacroscopicF + UpdateConstraintRHS implementation.
│           UpdateConstraintRHS calls `m_saddle_system->SetConstraintRHS`
│           (the Phase 5.0 method). Unit test against analytic values
│           for a multi-step monotonic loading.
├── 5.3.D  ComputeFluctuationField + ComputeHillMandelPowerBalance.
│           Unit test on a homogeneous problem (Hill-Mandel ≡ 0).
├── 5.3.E  λ accumulation API: GetLambda, ResetLambda, warm-start
│           between steps. Unit test: λ persists across multiple
│           saddle-point step calls.

         ↓ (gate: MortarPbcManager standalone tests green)

Phase 5.4 — NonlinearMechOperator::UpdateEssTDofsCornerSubset
├── 5.4.A  Add UpdateEssTDofsCornerSubset(const Array<int>&) method that
│           bypasses attribute expansion.
├── 5.4.B  Verify ParNonlinearForm::SetEssentialTrueDofs handles a
│           24-TDOF list correctly. Smoke test.

         ↓ (gate: mech_operator with corner-only ess TDOFs gives
             expected F and K)

Phase 5.5 — SystemDriver wiring
├── 5.5.A  Constructor: detect mortar PBC enabled, build manager, override
│           ess TDOFs, route saddle-point Operator to Newton solver.
├── 5.5.B  Solve() / SolveInit() / UpdateVelocity() / UpdateEssBdr()
│           mortar branches.
├── 5.5.C  Per-step UpdateMacroscopicF + UpdateConstraintRHS hook in
│           mechanics_driver main loop.

         ↓ (gate: end-to-end mortar PBC simulations run through
             SystemDriver correctly on linear-elastic test)

Phase 5.6 — Newton solver compatibility
├── 5.6.A  ExaNewtonSolver: linear-elastic patch test through MortarSaddle-
│           PointSystem. Convergence in ~1 Newton iter for linear case.
├── 5.6.B  ExaNewtonLSSolver: Neohookean RVE patch test under moderate
│           strain (10%); converges in < 20 Newton iters.
├── 5.6.C  ExaTrustRegionSolver: Neohookean RVE under severe strain (50%
│           shear) to validate TRDOG retains its advantage.

         ↓ (gate: all three Newton variants converge with mortar PBC)

Phase 5.7 — Validation suite (the six small-problem tests)
├── 5.7.A  mortar_pbc_linear_elastic.toml
│          Also: promote the `MortarPbcManager::DiagnoseConstraintConsistency`
│          consistency check (||C·v_aff - g||_inf < tol for v_aff = F̄·x)
│          to a ctest assertion at n ∈ {2³, 3³, 4³}. This is the precise
│          test that catches the perm-reorder trap of §P5.14.10 and any
│          structurally similar future regressions in C assembly.
├── 5.7.B  mortar_pbc_neohookean_50pct.toml (Test 1 of §P5.8.11)
├── 5.7.C  mortar_pbc_voce_polycrystal.toml
├── 5.7.D  mortar_pbc_heterogeneous_strip_multistep.toml (Test 2)
├── 5.7.E  mortar_pbc_checkerboard_multistep.toml (Test 2)
├── 5.7.F  mortar_pbc_restart.toml
           Each at np=1, 4, 7. Add all to CI.

         ↓ (gate: all six validation tests green at np=1, 4, 7)

Phase 5.8 — Output: ⟨L⟩, v_tilde, Hill-Mandel diagnostic
├── 5.8.A  Register ⟨L⟩ projection in PostProcessingDriver.
├── 5.8.B  Register v_tilde ParGridFunction; ParaView pipeline updated.
├── 5.8.C  Register Hill-Mandel diagnostic per §P5.10.3.

         ↓ (gate: outputs reach files cleanly; Hill-Mandel < 1e-10
             on all validation tests)

Phase 5.9 — Component-restricted PBC (spec-driven constraint filter)
├── 5.9.A  Option layer: PeriodicBC struct (essential_ids +
│           essential_comps); BoundaryOptions::periodic_bcs +
│           periodic_bc_entry_per_step (sparse step→entry map).
│           PeriodicBC::validate() catches non-positive attrs and
│           out-of-range comps; pair-completeness deferred to
│           manager (requires classifier).
├── 5.9.B  Classifier extensions: LabelForMeshAttribute /
│           MeshAttributeForLabel / PairPartnerLabel / ArePaired /
│           CornersOnFaceAttribute / IsBoundaryFaceAttribute /
│           AnchorCornerTDofs. All cache-only (O(1) lookups against
│           pre-built tables).
├── 5.9.C  ConstraintBuilder3D filtered overloads of Build,
│           BuildHypreParMatrix, EmitRowFactors, NumLocalRows,
│           NumConstraints accepting (active_pair_labels, comp_mask).
│           Parameter-less variants forward with {all pairs, all
│           comps} for back-compat. Filter rules: face pair active
│           iff axis in active_axes; edge group active iff BOTH
│           perpendicular axes in active_axes.
├── 5.9.D  MortarConstraintOperator::Reset(active_pair_labels,
│           comp_mask). LOCAL, no MPI. Re-walks flat row arrays;
│           updates m_n_comps_active, m_local_c[3], m_row_lambda_off,
│           Height. Does NOT rebuild edge_pair blocks (all 9 kept)
│           or off-rank import/export topology (over-imports under
│           reduced filter — correct, bounded waste). Native
│           mfem::forall kernels for Mult / MultTranspose /
│           ComputeInvDiagSchur honor m_local_c.
├── 5.9.E  Manager integration:
│           - RebuildForActiveSpec(essential_ids, essential_comps):
│             validates pairs, calls Reset, refreshes saddle system,
│             recomputes corner ess TDOFs, resizes m_lambda / m_g_rhs,
│             re-emits per-row factors.
│           - SynthesizeDefaultPbcSpec(classifier) static helper.
│           - ComputeCornerEssTDofsFromSpec free function: anchor
│             unconditional + incident-face gate + comp_mask.
│           - MortarSaddlePointSystem::Refresh method + invocation
│             from RebuildForActiveSpec (stale-cache hotfix).
├── 5.9.F  SystemDriver hook:
│           - SyncMortarPbcForStep(step_idx) with full state machine
│             (default-synth, idempotence, sparse update_steps,
│             config-error abort).
│           - Two private members: m_pbc_initialized,
│             m_pbc_active_entry_idx (with -1 sentinel for default).
│           - Ctor wiring: SyncMortarPbcForStep(1) replaces inline
│             UpdateEssTDofsCornerSubset call.
│           - m_x_saddle reallocation + newton_solver re-SetOperator
│             on transitions.
├── 5.9.G  mechanics_driver.cpp: add SyncMortarPbcForStep(ti) call
│           inside the BCManager::GetUpdateStep transition block,
│           immediately before UpdateEssBdr.
├── 5.9.H  Tests:
│           - test_constraint_builder_3d.cpp: three new filter cases
│             (X-only, X-face-pair-only, empty) plus the
│             period_signed_per_row signature fix from §P5.7.A.
│           - test_mortar_pbc_manager_filter.cpp (new file): five
│             cases of ComputeCornerEssTDofsFromSpec (full-XYZ,
│             X-only-single-pair, XY-two-pairs, anchor-only,
│             round-trip XYZ→X→XYZ).
├── 5.9.I  Integration validation: linear-elastic uniaxial-X patch
│           test runs end-to-end with correct Newton convergence
│           and Width() = x_block.Size() = 429 on a 4×4×4 mesh.
│           Multi-entry production test (full-XYZ step 1 → X-only
│           step 5) deferred to follow-on validation work.

         ↓ (gate: linear-elastic patch test green; existing fully-
              periodic regression suite still green; test_mortar_
              pbc_manager_filter and test_constraint_builder_3d all
              cases pass)

Phase 5.10 — Performance benchmarks + GPU validation + documentation
├── 5.10.A  CPU performance benchmark (32³ polycrystal, ExaCMech FCC).
├── 5.10.B  GPU validation on MI300A and/or A100.
├── 5.10.C  Multi-rank scaling study.
├── 5.10.D  Documentation: developers_guide.md update; README.md update;
│           tutorial example.

         ↓ (gate: performance acceptable; GPU runs validate; docs reviewed)

Phase 5.11 — Saddle-system residual scaling
├── 5.11.A  Options layer: `SaddleScalingOptions` table under
│           `[Solvers.SaddlePoint.Scaler]` (enabled, per_subblock,
│           partition, floor, range_cap). Parsed in option_parser_v2;
│           decoded into a `mortar_pbc::SaddleResidualScalerConfig`
│           at the `MortarPbcManager` boundary, mirroring the
│           `SaddlePointSolverOptions` → `SaddlePointSolverConfig`
│           pattern (separation of options-side and mortar-pbc-side
│           headers).
├── 5.11.B  Partition machinery: `mortar_pbc::SubblockPartition` enum
│           (`FaceEdge`, `PerPair`) added in `constraint_builder_3d.hpp`,
│           kept distinct from the options-side `::SubblockPartition`
│           to avoid pulling option_parser_v2.hpp into mortar_pbc
│           headers. `ConstraintBuilder3D::GetRowSubblockIds` populates
│           the per-row sub-block index array + the per-sub-block
│           label vector used downstream.
├── 5.11.C  `SaddleResidualScaler` class: holds the current scaling
│           state (`m_d_u`, per-row `m_d_lambda`, per-sub-block
│           `m_subblock_factor`); `Choose(r_u_norm, subblock_norms)`
│           applies Rule A unit-balance with floor / range_cap guards;
│           in-place `ApplyToResidual`, `UnapplyToIncrement`,
│           `ApplyToIncrement` for use by the wrappers and the
│           Newton-side convergence test. All operations local — no
│           MPI; caller is responsible for the residual-norm
│           reductions.
├── 5.11.D  Operator / solver / preconditioner wrappers (`ScaledSaddle-
│           Operator`, `ScaledSaddleSolver`, `ScaledSaddlePreconditioner`).
│           Always constructable around the underlying Phase 4.3
│           saddle stack; act as math no-ops when the scaler's
│           `IsEnabled()` is false. The operator wrapper sees
│           Newton's `oper->Mult` and `oper->GetGradient` paths and
│           applies `D^-1` on residuals + wraps the Jacobian as
│           `ScaledJacobianOperator` (which evaluates `D^-1 J D` on
│           the fly with no matrix-storage cost).
├── 5.11.E  Manager-side wiring: `MortarPbcManager` owns the scaler;
│           `ChooseScalingForStep` runs at the top of each step,
│           gathering the initial residual block norms (one
│           Allreduce) and calling `Choose`. Re-built on Phase-5.9
│           spec transitions so the partition tracks the active
│           filter.
├── 5.11.F  Newton-side diagnostic callback API: `NewtonDiagnosticSink`
│           = `function<void(const NewtonIterDiagnostic&)>` on
│           `ExaNewtonSolver` and `ExaNewtonLSSolver`. Invoked at the
│           top of each Newton iter AFTER the new residual norm is
│           computed and BEFORE the convergence-check break. The
│           diagnostic struct carries iter / norm / norm0 / norm_max
│           / converged_now plus non-owning pointers to the current
│           residual and solution iterate.
├── 5.11.G  TRDOG integration: `ExaTrustRegionSolver::SetScaler` lets
│           the dogleg body convert between solver-coord (`c` from
│           `prec_mech->Mult`) and physical-coord directions when
│           interpolating against gradients evaluated by
│           `ScaledJacobianOperator::MultTranspose`. Math: with
│           asymmetric `D^-1 J D`, the natural-output coord of the
│           inner solver is solver-coords; gradient is naturally
│           scaled-coords; reconciling those for the dogleg
│           interpolation needs one `ApplyToIncrement` per step.
├── 5.11.H  SystemDriver wraps Newton+J_solver+J_prec with the
│           scaling stack when the scaler is enabled. The unwrapped
│           saddle-stack path is preserved bit-for-bit when the
│           scaler is disabled (identity short-circuit in the
│           wrappers + gated install in SystemDriver).
├── 5.11.I  Per-iter diagnostic CSV (rolled into 5.11.H's delivery
│           bundle): rank-0-only file write, columns for
│           `step / iter / norm / norm0 / norm_max / converged_now /
│           scaler_enabled`.
├── 5.11.J  Rich diagnostic logger
│           (`SaddleNewtonDiagnosticLogger`): per-block residual
│           decomposition into `res_K`, `res_lam`, and per-sub-block
│           `res_lam_<label>` columns, plus per-sub-block scaling
│           factor evolution `d_u`, `d_lam_<label>`. Python analyzer
│           `analyze_newton_log_v2.py` produces per-step summary
│           tables and pathology detection (stalled residuals,
│           non-converged steps, sub-block-specific scaling
│           failures).
├── 5.11.K  Two-run diff diagnostic infrastructure:
│           `InspectingIterativeSolver` wraps the Newton-visible
│           J-solver to capture post-solve telemetry (inner Krylov
│           iter count, final norm, b, dx). The logger gains a
│           `MakePostSolveSink()` returning a callback that pairs
│           post-solve data with the buffered pre-solve row and
│           flushes a combined CSV row. The new `analyze_newton_log_v3.py
│           --diff` mode aligns two CSVs by `(step, iter)` and
│           reports the first column with `|base - compare| > eps`,
│           with a diagnostic-hint mapping from column name to the
│           suspect wrapper layer. Workflow: run with
│           `scaler.floor = 1.0e+30` (clamps D to I via floor-guard
│           branch in `Choose`) AND with scaler disabled; bit-equal
│           CSVs confirm wrapper transparency, divergent column
│           localizes the bug.

         ↓ (gate: validation suite still green with scaling enabled;
              two-run diff at `floor = 1.0e+30` shows bit-equal CSVs
              across the eligible columns; documented hazards from
              §P5.14.15 and §P5.14.16 surfaced and resolved)

Phase 5 complete. Mortar PBC is a first-class ExaConstit feature.
```

---

## §P5.14 Hazards and traps

The Phase 4 trap list (architecture doc §12, Phase 4 plan §P4.8) applies;
the integration adds the following new traps. The v2 traps about per-step
constraint rebuild are obsolete and removed.

### §P5.14.1 $\bar F$ tracking restart consistency

**Failure mode**: If ExaConstit checkpoints/restarts, $\bar F^{(n)}$ must
be saved/restored alongside the rest of the simulation state. If only the
mesh and $v$ field are saved, restart will reinitialize $\bar F^{(n)}$ to
the identity, giving wrong constraint RHS values from the first restarted
step.

**Mitigation**: register `m_macro_F` as part of `MortarPbcManager`'s
serializable state. The I/O layer must include this 9-double matrix in the
checkpoint format. CI test: `mortar_pbc_restart.toml` (§P5.12.2) explicitly
checkpoints mid-simulation and verifies post-restart $\bar F$ matches the
pre-restart value.

### §P5.14.2 The corner-pin signs and origin convention

**Failure mode**: $\bar L$ is row-major or column-major in the flat
9-vector? The existing `essential_vel_grad` parser determines this; mortar
PBC must follow the same convention. If a future refactor changes the
parser convention, mortar PBC will silently apply wrong corner pin values.

The `vgrad_origin` convention (defaults to mesh minimum if not specified;
honored as a fixed spatial point if specified) is also a footgun for
$X_0 \ne 0$ users (see §P5.8.9 Gotcha 3).

**Mitigation**: a unit test that exercises the corner-pin computation with
several (vgrad, origin, corner_coord) inputs and asserts the expected
output. The test serves as a regression contract for the ordering
convention.

### §P5.14.3 The Newton convergence criterion

**Failure mode**: `MortarSaddlePointSystem::Mult` must return the combined
residual $\sqrt{\|F_\text{int} + C^T \lambda\|^2 + \|Cv - g\|^2}$, not just
$\|F_\text{int}\|$. If the bug ever creeps in (e.g., in a future
optimization that bypasses the constraint residual contribution), Newton
will appear to converge when the constraint is unsatisfied.

**Mitigation**: CI test that asserts post-converged $\|Cv - g\|$ is below
the same tolerance as the bulk residual. Architecture doc §12.1 Trap 3 is
the reference.

### §P5.14.4 Lambda warm-starting between time steps

**Failure mode**: At the start of each new time step, $\lambda$ from the
previous step's converged solution is *almost* right (the constraint
geometry hasn't changed, but the macroscopic loading has advanced).
Initial-iteration residual is *not* zero with $\lambda$-warm-start as it
would be with $\lambda = 0$ — but the warm-start is closer to the new
equilibrium than zero, so it's a useful initial guess.

**Decision**: warm-start $\lambda$ by default. Add `MortarPbcManager::ResetLambda()`
for the user to call if they want to disable warm-starting for debugging.

### §P5.14.5 The MortarSaddlePointSystem K closure capture

**Failure mode**: `SystemDriver` passes K closures to the adapter that
capture `mech_operator` by raw pointer. If `mech_operator` is rebuilt
mid-simulation (it shouldn't be, but adaptive remeshing in the future
might), the capture goes stale.

**Mitigation**: assert at construction that `mech_operator` lifetime
exceeds the `MortarPbcManager`'s. Document that adaptive remeshing requires
manager reconstruction.

### §P5.14.6 Hill-Mandel drift detection

**Failure mode**: Any of the §P5.8.9 Gotchas 1–5 (severe distortion,
$\bar F$ inconsistency, $X_0 \ne 0$, $p \ge 2$, time-integrator mismatch)
manifests as Hill-Mandel power balance drift. Without monitoring, the
drift can accumulate silently and corrupt long-running simulations.

**Mitigation**: CI requires Hill-Mandel residual < 1e-10 on all validation
tests. Production runs print the residual to log; a runtime warning if it
exceeds 1e-8 alerts the user. Documentation in `developers_guide.md` and
the tutorial directs users to interpret Hill-Mandel as the canonical
sanity check.

### §P5.14.7 The off-equilibrium RHS subtlety

**Failure mode**: §P5.8.6 derives that the constraint RHS should use the
macroscopic-rate form (P5.8.6.d), not the current-spatial-offset form
(P5.8.6.a), to avoid feeding unconverged fluctuation back into the Newton
iteration. If a future implementation accidentally uses the
current-spatial-offset form, Newton convergence will degrade (chasing a
moving target) without an obvious correctness signal.

**Mitigation**: implement only the macroscopic-rate form; the spatial-offset
form is not a code path that exists. Add a unit test asserting that the
RHS for a given macroscopic state is computed deterministically from
$\bar F^{(n)}$ and $\bar L^{(n)}$ alone, with no dependence on the current
boundary $v$ or $\tilde u$ field.

### §P5.14.8 Saddle-point residual must use the set constraint RHS, not the homogeneous default

**Failure mode**: `MortarSaddlePointSystem::Mult` defaults to the
homogeneous form $r_C = C \cdot u$ when no constraint RHS is installed
(this is the default behavior the Phase 4.3 code already exhibits, kept
unchanged after the Phase 5.0 extension for backward compatibility). If
the implementation calls `UpdateConstraintRHS()` correctly each step but
the manager forgets to forward $g$ to `m_saddle_system->SetConstraintRHS(g)`
— or if a refactor accidentally removes that forwarding call — the Newton
iteration will converge to the wrong fluctuation field. Specifically, it
will solve $C v = 0$ instead of $C v = g$, which produces an interior
field consistent with periodic fluctuation under no macroscopic loading
(zero overall deformation rate at the boundary), regardless of what the
corners are pinned to. The corner DOFs themselves still satisfy
$v_\text{corner} = \bar L \cdot x_\text{corner}$ (correct, via the
Dirichlet pin), but the rest of the boundary will satisfy a
homogeneous-fluctuation periodicity that is *not* what the macroscopic
loading requires. Symptoms: $v_\tilde$ has visible discontinuity across
non-corner boundary DOFs in ParaView; $\langle L \rangle \ne \bar L$;
Hill-Mandel drift may be small but nonzero.

**Mitigation**: in `MortarPbcManager::UpdateConstraintRHS`, call
`m_saddle_system->SetConstraintRHS(m_g_rhs)` *unconditionally* at the
end. Add a unit test that initializes the saddle system, calls
`UpdateConstraintRHS` with a known nontrivial $\bar L$, and verifies via
`Mult` on a probe vector that $r_C$ contains the expected $-g$
contribution. The test should use a probe vector $u$ such that
$C u = 0$ and check $\| r_C + g \|_\infty < \epsilon$.

### §P5.14.9 Refactoring the corner-pin projection may regress non-mortar tests

**Failure mode**: §P5.5.4 proposes factoring the existing
`UpdateVelocity` essential_vel_grad projection into a free function
`ProjectVelocityGradientToCornerTDofs` so it can be called with a
TDOF subset. The non-mortar path is intended to use the same helper
with `corner_tdofs = ess_tdofs` (full set), making the mortar branch a
clean substitution. If the refactor changes the iteration order, the
sentinel handling for already-set TDOFs, or the mesh-node-vs-original-
coord lookup, the existing non-mortar tests (`test_patch_*`,
`test_neohookean_*`, etc.) may regress with results that look almost
right but are subtly off — the kind of failure mode that's hard to
debug because the mortar feature itself is "off."

**Mitigation**: do the refactor in a *separate* batch (Phase 5.5.A
before any mortar wiring) with the only change being mechanical
extraction — same loop, same arithmetic, same memory layout, just
moved to a free function. CI runs the full non-mortar test suite at
each step. Bisect-friendly: revert this batch if any non-mortar test
fails. If the factoring turns out to be hard to do safely (e.g.,
because the existing code has hidden coupling to other UpdateVelocity
state), abandon the factoring and instead implement the mortar
corner-only projection as a *separate* parallel function, accepting
the code duplication.

### §P5.14.10 The conforming-face-mortar permutation-reorder trap

**Failure mode**: The Wohlmuth row-sum identity $\sum_{\ell \in \text{kept}}
A^m_{i\ell} = D_i$ (which makes $C \cdot u_\text{aff} = g$ hold exactly for
affine displacement fields — see §P5.8.4) silently breaks on opposite-normal
face pairs whenever the mortar shape evaluation is double-permuted in the
conforming-face-mortar assembler's inner loop. The 2³ unit test passes —
because the geometric configuration there is blind to the bug — but the
constraint-consistency diagnostic at 3³ and finer meshes shows
$\|C \cdot v_\text{aff} - g\|_\infty \approx 0.5\, D_\text{max}\, \|\bar L\|$,
with the argmax row landing on a corner-adjacent face-interior node and
showing $Cv/g = 1.5$ exactly. Newton fails to converge or converges to a
wrong answer; $\bar F^{xx}$ at the corner pin is correct (e.g., 1.001) but
the actual deformation comes out wrong (e.g., 0.9995) — the constraint is
enforcing the wrong thing.

**The geometric setup that triggers it**: On a cubic RVE with periodic
BCs, the $-x$ face and $+x$ face have outward normals pointing in opposite
directions ($-\hat e_x$ and $+\hat e_x$). Standard face element
orientation conventions order local nodes CCW *from outside the volume*. For
the $-x$ face this is CCW in $(y, z)$; for the $+x$ face this is CW in
$(y, z)$ — because "from outside" for the $+x$ face means looking in
$-\hat e_x$ direction, which flips handedness.

When the conforming matcher (`MatchConformingFacePairs`) pairs corresponding
nm and mortar elements by in-plane physical centroid and then computes
the node-correspondence permutation via `NodePermByCoordMatch`, the
resulting `mortar_node_perm` is **NOT the identity** for axis-aligned cube
periodic faces. It is the involution $(0, 3, 2, 1)$ for quads (the
"anti-diagonal mirror" in local-node indexing) and $(0, 2, 1)$ for tris.

This is a *geometric fact* of opposite-normal face pairs — not a bug, not a
coincidence, not a mesh-generator quirk. Any boundary classifier that
orders face elements CCW-from-outward-normal — which is the standard
convention in essentially all FE libraries (MFEM, deal.II, libMesh, etc.) —
will produce this perm on opposite face pairs. **Mortar PBC implementers
should expect non-identity perms on every periodic face pair, full stop.**

> **Documentation note**: The existing docstring on `MatchConformingFacePairs`
> in `face_mortar_assembler_3d.hpp` claims *"For axis-aligned meshes this
> permutation is always the identity (0, 1, 2, 3)."* This claim is
> **incorrect**. The docstring should be updated to reflect the geometric
> reality described above.

**The correct algorithm**: The inner assembly loop integrates
$\int_E M^\text{nm}_k(\xi^\text{nm}) \cdot N^\text{m}_\ell(\xi^\text{m})\, dA$
over the nonmortar element's reference quad, with $\xi^\text{nm}$ the nm
Gauss point coordinate. Given the non-identity perm, this needs three
steps:

1. **Map the nm reference Gauss point to the mortar's *own* reference
   frame** via the perm-defined affine map. For the $(0, 3, 2, 1)$
   involution this gives $\xi^\text{m} = (\eta^\text{nm}, \xi^\text{nm})$ —
   a swap of the local axes. The helper
   `QuadFaceMortarAssembler::MortarRefFromPermutation` implements this
   correctly.

2. **Evaluate `NQuad4` in the mortar's own reference frame** at $\xi^\text{m}$.
   The result `NQuad4(pt_mortar)[j]` is *already* mortar local node $j$'s
   shape function value at the physical Gauss point. No further reorder is
   needed: the standard $Q_1$ basis function $N_j$ is — by definition — 1
   at its own reference vertex and 0 at the other three, so evaluating it
   at $\xi^\text{m}$ gives the correct value at the physical position
   corresponding to that mortar local node.

3. **Scatter** $A_\text{loc}[k][\ell_\text{loc}]$ to
   $A^m[k_\text{global}, \texttt{mortar\_col\_of[m.gtdofs[}\ell_\text{loc}\texttt{]]}]$.
   The mortar local index $\ell_\text{loc}$ used for both the integrand and
   the scatter column must be the same — the integrand was for "mortar
   local $\ell_\text{loc}$'s shape function," and it must land in the
   column for "mortar local $\ell_\text{loc}$'s gtdof."

**The trap is to apply additional perm indirection on the shape values**.
A naive (and wrong) implementation reasons: "the perm maps nm local $i$ to
mortar local $\text{perm}[i]$; so to align mortar shape values with nm
local indices, I should reorder them by the perm." This *double-permutes*:
step 1 above already accounts for the perm by mapping the parametric
coordinate to the right frame; a second reorder then re-permutes the shape
values, mismatching the mortar local index used in the scatter from the
mortar local index whose shape was actually integrated. The result is
that `A_loc[k][l_loc]` (the integral against mortar local $\text{perm}[\ell_\text{loc}]$'s
shape) gets scattered to the column for mortar local $\ell_\text{loc}$'s
gtdof — different mortar global nodes whenever the perm is non-identity.

**Why 2³ doesn't catch it**: For a 2³ RVE every face has exactly 1
face-interior (kept) node and 4 face elements, ALL of which are
corner-tagged (each element touches exactly one face corner). For
corner-tagged elements the Wohlmuth-modified dual basis collapses to
$M_\text{kept} = 1$ (constant on the element); see §P5.8.3 for the
Wohlmuth modification rule. With $M = 1$, the inner integral
$\int M \cdot N_\ell\, dA = \int N_\ell\, dA = J$ for every $\ell$. Hence
$A_\text{loc}[k_\text{kept}] = (J, J, J, J)$ regardless of how the four
shape values are permuted among themselves — the bug is **invisible** on
this geometry.

At 3³ and finer, "edge"-tagged elements appear (those touching exactly one
face edge — one Wohlmuth modification axis instead of two). Their
$A_\text{loc}$ rows are NOT uniform: for an `edge-eta-low` element with
$M_2 = (1 + 3\xi)/2$, $A_\text{loc}[2] = (0, J, J, 0)$ — two zero entries
and two $J$ entries via biorthogonality with $N_1$ and $N_2$ while
orthogonal to $N_0$ and $N_3$. Under a wrong shape-value permutation, the
zero entries and $J$ entries get reshuffled into the wrong scatter
columns, producing extra mass on row sums that shouldn't get it.

**The 1.5 arithmetic** (exact reproduction): For an axis-aligned RVE at
any mesh resolution $n^3$ with $n \geq 3$, the corner-adjacent face-interior
nodes (4 per periodic-face pair) sit at the junction of:

| Element type touching node | D contribution | Row-sum (correct) | Row-sum (buggy) |
|---|---|---|---|
| 1 × corner-tag ($M = 1$ const)  | $J$         | $J$  (drop-restricted)         | $J$  (bug invisible)              |
| 2 × edge-tag ($M = (1 \pm 3\xi)/2$) | $2J$  | $2J$                            | $4J$  ($+J$ per element)          |
| 1 × none-tag ($M$ std dual)     | $J$         | $J$  (bi-orthogonality)         | $J$  (bug invisible)              |
| **Total**                       | **$4J$**    | **$4J$**                        | **$6J$**                          |

Buggy ratio $= 6J / 4J = 1.5$, **independent of $n$**. This 1.5 ratio is a
geometric invariant of the bug — independent of mesh resolution, element
size, or applied $\bar L$. The absolute error scales with the row's $D$
value ($= h^2$ for corner-adjacent nodes in a uniform Cartesian RVE) and
linearly with $\|\bar L\|$. So at higher mesh resolution the error per
element shrinks (because $h^2$ shrinks), but the *ratio* persists exactly.

**Why edges (the 1D mortar) are structurally immune**: The 1D edge mortar
(`MortarAssembler2D::IntegrateOverlapSegment`) uses *physical-coord*
parameterization for the overlap segment — it places Gauss points on the
physical 1D overlap interval and then maps each one back to the parent
elements' local $\xi$ via $\xi = (\phi - \phi_\text{mid})/\phi_\text{half}$.
The mortar shape value $N^-$ is evaluated directly at $\xi^-$ on the mortar
parent. There is no perm on the mortar side at all — the parameterization
in physical coords sidesteps the orientation issue entirely. Hence the
structural analog of the trap doesn't exist in 1D. *Edge rows in the
diagnostic stay clean throughout, which is what initially misled
investigation toward face-specific causes.*

**Detection**: The standalone diagnostic
`MortarPbcManager::DiagnoseConstraintConsistency` computes
$\|C \cdot v_\text{aff} - g\|_\infty$, where $v_\text{aff} = \bar F \cdot x$
is the affine field that — for any valid $\bar F$ — should make the
constraint exact (per §P5.8.4). The $L_\infty$ norm at floating-point
roundoff (typically $\sim 10^{-14}$ on a unit-cube RVE) signals the
row-sum identity holds globally. Any value $O(D_\text{max} \cdot \|\bar L\|)$
signals a row-sum violation.

The companion diagnostic `[constraint_diag_argmax_diff]` reports the
specific argmax row's period vector, component, ell ($= D[\text{row}]$),
$g$, $Cv$, and diff. The $Cv/g$ ratio at the argmax row is the bug's
signature: **1.5 → this trap (conforming face-mortar reorder)**;
**other non-unity ratios → other distinct bugs** (e.g., 1.25 ≈
corner-sentinel-only drops without edge-sentinel drops; 2.25 ≈ no
mortar-side drops at all; etc.). Tracking *which exact ratio* a bug
produces narrows the diagnostic search dramatically — implementers who
hit a non-unity ratio should compute what the various plausible failure
modes predict and look for the match.

**Lessons for future implementers / for porting this scheme to other
codes**:

1. **The "natural" mortar shape evaluation should match the non-conforming
   (clipped) path's pattern**: compute the mortar reference coordinate
   (via perm-defined affine map for conforming pairs, or via inverse
   iso-map for clipped pairs), then evaluate the standard reference basis
   (`NQuad4` or `NTri3`) *once* in the mortar's own reference frame. The
   output is *already* indexed by mortar local node. **No further reorder
   is ever needed.**

2. **Always assume non-identity perm on opposite-normal face pairs** —
   which is essentially every PBC face pair, since periodic faces are by
   definition opposite-normal. Test this case explicitly. **The simplest
   case (2³ or any single-element-thick periodic geometry) is structurally
   blind to this class of bug**: every face element is corner-tagged, the
   $M = 1$ collapse hides any shape-reorder error.

3. **The Wohlmuth row-sum identity is brittle but self-diagnosing**: any
   per-element error scales by element count, but the consistency check
   $C \cdot v_\text{aff} = g$ uses only built operators (no separate
   reference implementation needed) and pinpoints the offending row.
   **The argmax-of-diff reporter is essential**: $L_\infty$-norm-only
   reporting gives no actionable signal about where in the global $C$
   the error lives.

4. **Sweep mesh resolutions in the consistency test**: write the test to
   cover at least $\{2^3, 3^3, 4^3\}$, not just the smallest. Bugs that
   are blind to one resolution but visible at the next are common in
   mortar work because of the way Wohlmuth modification patterns change
   with mesh density (corner-only at $n = 2$; corner + edge + interior at
   $n \geq 3$).

5. **The clipped path's design is the reference**: when implementing the
   conforming path in a new code, use the clipped path as the template —
   compute mortar parametric coords directly (via inverse iso-map or
   equivalent), evaluate the standard basis in the mortar's own ref frame,
   scatter against `m.gtdofs[l_loc]` directly. If your conforming and
   clipped paths produce numerically different $A^m$ matrices on a
   conforming-overlap test problem (matched element pairs, no clipping
   needed), one of them has this class of bug.

**Mitigation in the Phase 5.7.A bundle**: removed the spurious
`ReorderMortarShape` call from both
`QuadFaceMortarAssembler::AssemblePairConforming` and
`TriFaceMortarAssembler::AssemblePairConforming`; kept
`MortarRefFromPermutation` and `MortarBaryFromPermutation` (these are
correct and handle the parametric-axis swap). The `ReorderMortarShape`
helpers and their declarations should be deleted entirely; they are no
longer called by anything.

**Cross-references**: §P5.8.3 (Wohlmuth biorthogonality on reference
faces — the discrete identity that the trap breaks); §P5.8.4
($C \cdot u_\text{aff} = g$ as the constraint-correctness test); §P5.8.5
(geometric setup of opposite-normal periodic faces and how the classifier
caches them).

### §P5.14.11 Stale size caches in `MortarSaddlePointSystem` and `SystemDriver::m_x_saddle`

**Failure mode**: `MortarSaddlePointSystem`'s ctor caches `m_n_u`,
`m_n_lam`, `height`, `width`, and `m_block_offsets` from the
constraint operator's ctor-time dimensions. If the operator's filter
spec changes mid-construction (Phase 5.9 routinely does this in
`SyncMortarPbcForStep(1)` from the SystemDriver ctor) or mid-run
(multi-entry transitions), the cached sizes go stale and downstream
size verifications fail. The diagnostic fingerprint is
`MortarSaddlePointSystem::Mult: x_block size X != Width() Y` where
`Y` is the unfiltered total and `X` is the post-filter total —
exactly the abort message produced by the X-only linear-elastic
patch test before the §P5.18.7 refresh cascade was in place.

The same failure mode applies to `SystemDriver::m_x_saddle`
(`std::unique_ptr<BlockVector>`) on mid-run transitions: it is sized
once at ctor time from `NumLocalConstraints()` and stays that size
unless explicitly reallocated.

**Mitigation**: `MortarSaddlePointSystem::Refresh()` is called from
`MortarPbcManager::RebuildForActiveSpec` immediately after
`m_C_op.Reset(...)`. `SystemDriver::SyncMortarPbcForStep`
reallocates `m_x_saddle` (`std::make_unique<BlockVector>(new_offsets)`)
on every transition and re-calls
`newton_solver->SetOperator(GetSaddleSystem())` to invalidate any
Newton-side cached sizes. The `if (m_x_saddle)` guard at the
reallocation site handles the ctor's first call (where `m_x_saddle`
is still null and the saddle-prec block, running a few lines later,
will size it correctly from the already-refreshed
`NumLocalConstraints()`).

Audit list for future stale-cache risks: any class that takes an
operator by reference at construction and queries `Width()`/`Height()`
at any time other than per-call. Specifically:
`MortarSaddlePointSystem` (fixed), `SystemDriver::m_x_saddle`
(fixed), `ExaNewtonSolver` (re-`SetOperator` covers it),
`MortarSaddlePreconditioner` (safe — re-`SetOperator`'d per Newton
iter and queries `m_C_op` through reference), `SaddlePointSolver`
(safe as observed — uses operators by reference at call time).

### §P5.14.12 Constructor ordering for the SystemDriver mortar block

**Failure mode**: The SystemDriver mortar ctor block has a strict
ordering requirement post-Phase 5.9:

```cpp
m_mortar_pbc = std::make_shared<MortarPbcManager>(...);   // (1)
m_mortar_enabled = true;                                   // (2) — before (3)
SyncMortarPbcForStep(1);                                   // (3) — needs (2) true
// ... saddle prec construction ...
m_x_saddle = make_unique<BlockVector>(m_saddle_offsets);   // (4) — sized from
                                                           //       updated count
newton_solver->SetOperator(saddle_system);                 // (5)
```

If `m_mortar_enabled = true` is moved *after* `SyncMortarPbcForStep(1)`,
the Sync early-returns without installing anything (its first guard
is `if (!m_mortar_enabled) return`), leaving the manager in its
unfiltered-default state. The user-supplied spec is silently ignored
until the first time-step's `Sync` call eventually applies it — at
which point `m_x_saddle` is already sized from the stale unfiltered
count, triggering the §P5.14.11 stale-size abort.

If `m_x_saddle` is allocated *before* `SyncMortarPbcForStep(1)`, it
is sized from the unfiltered 162-row state, and the same abort
emerges on the first step.

**Mitigation**: keep the constructor sequence as documented. Add a
comment at each of the three ordering-critical lines explaining the
constraint. A unit test that simulates the spec being applied at
ctor time and verifies `m_x_saddle.Size() == n_K +
NumLocalConstraints()` would catch any future re-ordering
regression.

### §P5.14.13 Sub-XYZ specs may leave rotation rigid-body modes unconstrained

**Failure mode**: §P5.18.5's anchor-pinning rule removes the three
translation modes unconditionally. Under full-XYZ specs, the
multi-corner pinning incidentally constrains rotations as well; under
sub-XYZ specs (X-only, XY, etc.), rotations about the un-pinned axes
may remain. If $K$ is rank-deficient on those modes, the Newton
saddle solve fails to converge or produces non-unique solutions.
The symptom is non-convergence in Newton (Krylov stagnation or
spurious unbounded iterates) rather than a clean abort.

**Mitigation**: `ComputeCornerEssTDofsFromSpec`'s docstring
documents the caveat. Users running sub-XYZ specs and encountering
Newton non-convergence should add corner Dirichlet BCs via the
standard BC machinery (e.g. pin one transverse component of a
non-anchor corner) to remove the residual rotation mode. Phase
5.9.I's integration validation covers full-XYZ and uniaxial X with
free Y/Z; the latter is rotation-RBM-free for axis-aligned tension.
Auto-detection of unconstrained rotation modes is logged as future
work in §P5.18.12.

### §P5.14.14 `Reset` is collective by convention but non-blocking

**Failure mode**: `MortarConstraintOperator::Reset` and the
downstream `MortarPbcManager::RebuildForActiveSpec` are **local —
no MPI calls.** However, the unchanged off-rank import/export
topology in the operator is symmetric across ranks; correctness
requires that every rank sees the same filter spec post-`Reset`.
Both sides of every off-rank exchange must agree on what bytes get
sent/received. If a rank-divergent call to `Reset` occurs (rank A
applies filter $F_A$, rank B applies $F_B$), no immediate abort
fires; the next constraint matvec produces silently corrupted
output.

This is the same "collective by convention" pattern that, e.g.,
`MPI_Allreduce`'s op and datatype arguments enforce implicitly:
the runtime doesn't check that ranks agree, but they must.

**Mitigation**: the spec source on every rank is the same TOML file
parsed by the same code into the same `BoundaryOptions::periodic_bcs`
vector, so divergence is impossible by construction in the
production flow. `SystemDriver::SyncMortarPbcForStep` is the only
caller of `RebuildForActiveSpec` in production; its lookup
`periodic_bc_entry_per_step.at(step_idx)` is deterministic given
the same step_idx, which all ranks pass uniformly.

A defensive check (MPI_Allreduce a hash of `active_pair_labels` +
`comp_mask` for `MPI_BAND` equality) would catch divergence but
makes `Reset` collective; the trade-off was decided against in
favor of MFEM's general "caller responsibility" convention. Document
the convention in the `Reset` and `RebuildForActiveSpec` doxygen
comments.

### §P5.14.15 `iterative_mode` flag forwarding in `mfem::Solver` wrappers

**Failure mode**: Surfaced during the Phase 5.11.K
`InspectingIterativeSolver` integration. Any class derived from
`mfem::Solver` inherits a public `iterative_mode` bool. The
`ExaNewtonSolver` family sets this on `prec_mech` once before the
Newton loop starts:

```cpp
prec_mech->iterative_mode = false;   // tell J-solver to zero x
```

When `prec_mech` is a wrapper (the inspector, the scaled saddle
solver, or any future `mfem::Solver`-derived decorator), this
assignment lands on the WRAPPER's `iterative_mode` field. The inner
solver consults its own `iterative_mode` field to decide whether
to use `x` as an initial guess inside `Mult`. If the wrapper's
`Mult` body just forwards `m_inner->Mult(b, x)` without first
syncing `iterative_mode`, the inner solver reads whatever flag
state it last had (typically the MFEM default `true`), treats `x`
as a non-zero initial guess, and seeds its Krylov from whatever
bytes happen to live in `x`'s buffer.

The symptom is non-deterministic Newton behavior across runs of
the same problem with the same binary: the inner solver starts
from buffer noise that varies run-to-run depending on the
allocator's reuse pattern. Diagnoses that fit this pattern but
turn out wrong: (a) GPU device-memory validity flags, (b) Vector
copy-constructor device-awareness bugs, (c) BlockVector::Update
lifetime mistakes, (d) MPI reduction-order non-determinism (ruled
out trivially on single-rank). All are red herrings — the actual
trigger is the un-forwarded flag.

**Mitigation**: at the top of every wrapper's `Mult`, forward the
flag before delegating:

```cpp
void Mult(const Vector& b, Vector& x) const override {
    m_inner->iterative_mode = iterative_mode;
    m_inner->Mult(b, x);
    // ... post-solve hook, etc ...
}
```

`m_inner` is held as `shared_ptr<Solver>`; mutating the pointee in
a const method is legal because the pointer itself is unchanged.
The cost is one bool assignment per `Mult` call — negligible.

This same pattern applies to `ScaledSaddleSolver`,
`ScaledSaddlePreconditioner`, and any future
solver-wrapping-solver class. Code review should treat absence of
this line as a defect.

### §P5.14.16 `SetSolver` overload selection: shared_ptr vs reference

**Failure mode**: Surfaced during Phase 5.11.K's SystemDriver
wiring. `ExaNewtonSolver` (and the base `mfem::IterativeSolver`)
expose multiple `SetSolver` overloads:

```cpp
// On ExaNewtonSolver — sets the shared_ptr-typed `prec_mech`:
void SetSolver(std::shared_ptr<mfem::Solver> prec);

// On the mfem::IterativeSolver base — sets only the base class's
// raw `prec` pointer:
void SetSolver(Solver& solver);
```

Newton's `Mult` body uses `prec_mech`, not the base class's
`prec`. The shared_ptr-overload is the one that wires Newton's
J-solve correctly.

A wiring snippet that dereferences a shared_ptr before passing it:

```cpp
std::shared_ptr<mfem::Solver> newton_visible_solver = /* ... */;
newton_solver->SetSolver(*newton_visible_solver);   // wrong overload
```

resolves to the base class `SetSolver(Solver&)` — which sets the
raw `prec` pointer that Newton doesn't use. `prec_mech` retains
whatever value the previous shared_ptr-overload call left it at,
silently bypassing the new solver. Compiles cleanly, runs
cleanly, gives wrong answers (Newton uses the stale `prec_mech`).

The symptom is subtle: removing or rearranging a `SetSolver` call
elsewhere in the constructor changes Newton's behavior because the
"last shared_ptr `SetSolver` call wins" stale-state pattern is
fragile to reordering.

**Mitigation**: pass `shared_ptr` directly, never dereferenced:

```cpp
newton_solver->SetSolver(newton_visible_solver);   // correct
```

`newton_visible_solver` here is `std::shared_ptr<mfem::Solver>`;
the call binds to the shared_ptr overload and updates `prec_mech`.

For future wiring code in SystemDriver, all `SetSolver` calls on
Newton should pass shared_ptr by value. Adding a deleted
`SetSolver(Solver&)` overload on `ExaNewtonSolver` (delegating to
the shared_ptr version after a `shared_from_this`-style fixup) is
an alternative that would surface the bug at compile time, but is
intrusive enough that documentation + code review is the
preferred mitigation for now.

---

## §P5.15 Open design questions

1. **Should `MortarPbcManager` be a singleton (like `BCManager`) or a member
   of `SystemDriver`?**

   **Recommendation**: member of `SystemDriver`. Singletons are awkward
   when multiple `SystemDriver` instances exist (which doesn't happen in
   current ExaConstit but might in FE² future work).

2. **Should the saddle-point solver be a runtime-selected variant of the
   existing nonlinear solver, or a separate solver subclass?**

   **Recommendation**: runtime-selected via the `MortarSaddlePointSystem`
   adapter pattern. No solver subclass explosion.

3. **What's the relationship between `BCManager` and `MortarPbcManager`?**

   **Recommendation**: `BCManager` continues to own all BC data —
   essential_vel_grad, essential_velocity, etc. `MortarPbcManager` reads
   the relevant BC data through `BCManager` and applies its corner-only
   override. The override is the only "new BC behavior."

4. **Where does $\bar F$ tracking live?**

   **Recommendation**: in `MortarPbcManager`. It's intrinsically tied to
   the mortar PBC formulation — not a general-purpose homogenization
   quantity that BCManager or SimulationState should own. If FE² coupling
   ever wants to track $\bar F$ at the macro scale, that's a separate
   external state.

5. **What goes into the ParaView output by default?**

   **Recommendation**: nothing new from mortar by default. The user's
   existing ParaView config is preserved. New projections (`v_tilde`,
   `velocity_gradient_avg`, `hill_mandel`) are opt-in via
   `[PostProcessing.Projections] enabled_projections`.

---

## §P5.16 Cross-references

- Architecture doc §3, §4, §6 — mortar method, dual basis, saddle-point.
- Architecture doc §11.7, §12 — classifier and trap list.
- Architecture doc §13.3 — original BCManager / SystemDriver integration
  sketch (this v3 doc supersedes that sketch).
- Phase 4 plan §P4.4.6 — EA path, `MortarSaddlePointSystem`.
- Phase 4 plan §P4.4.6.6 — adapter design (Batch R).
- Phase 6 plan — higher-order via LOR (next phase after Phase 5).

---

## §P5.17 Done criteria

Phase 5 is **done** when ALL of the following hold:

- [ ] All six small-problem validation tests (§P5.12.2) pass at np=1, 4, 7
      in CI.
- [ ] The "feature off" regression check (§P5.12.4) passes — every existing
      non-mortar test produces bit-identical results before and after Phase 5.
- [ ] Hill-Mandel power balance < 1e-10 on all validation tests at every
      converged step.
- [ ] Performance benchmarks (§P5.12.3) show acceptable overhead (< 30%
      CPU, < 50% GPU) for representative crystal-plasticity problems.
- [ ] All three Newton solver variants (NR, NRLS, TRDOG) work with mortar
      PBC on at least one nontrivial problem each.
- [ ] All three assembly modes (PA, EA, FULL) work with mortar PBC on at
      least the linear-elastic homogeneous test.
- [ ] GPU validation passes on at least one of MI300A or A100 (or their
      CUDA / HIP equivalents).
- [ ] Restart test passes — checkpoint mid-simulation, restart, verify
      $\bar F^{(n)}$ and $\lambda$ are restored.
- [ ] User-facing documentation updated.
- [ ] No `// TODO` markers in production code paths (only in hooks
      explicitly deferred to Phase 6 / Phase 7).
- [ ] Doxygen-complete public API for `MortarPbcManager` and the modified
      `SystemDriver` / `ExaNewtonSolver` methods.

When done, the next logical step is Phase 6 (higher-order primal via LOR
— see `PHASE6_HIGHER_ORDER_LOR.md`), or any of the long-term items in
architecture doc §14.3 (Tribol integration, UT BCs, FE² coupling).

---

## §P5.18 Component-restricted PBC: per-pair and per-component constraint filtering

The Phase 5.0–5.7 batches deliver mortar PBC with all three face
pairs constrained in all three spatial components. This is what
"fully periodic" means in classical RVE terms and is the right
default for homogenization studies of fully-periodic
microstructures. It is not the right default for **patch tests**
(uniaxial tension with free transverse contraction), **partially-
periodic protocols** (e.g. layered laminates periodic in-plane but
free in the through-thickness direction), or **load schedules** that
need to *change* the periodicity status mid-simulation (e.g.
relaxation of a constraint after a critical step).

§P5.18 extends the manager-and-operator stack to accept a
user-supplied filter specifying which face pairs and which spatial
components are active, while preserving the fully-periodic default
bit-for-bit when no spec is given.

### §P5.18.1 The `PeriodicBC` spec — TOML interface and option-struct decoding

A `PeriodicBC` entry has two fields:

```toml
[[BCs.periodic_bcs]]
    essential_ids   = [1, 2]   # face attributes
    essential_comps = 1        # X only (decoded via BCData::GetComponents)
```

- `essential_ids` (`std::vector<int>`): boundary face attributes the
  periodic constraint applies to. Must list **both halves** of every
  participating face pair (e.g. `{1, 2}` for the x-pair, not just
  `{1}`); the pair-completeness check in
  `MortarPbcManager::RebuildForActiveSpec` aborts otherwise.
- `essential_comps` (`int`, 1..7): packed code that decodes to a
  spatial-component mask via the same `BCData::GetComponents`
  convention `VelocityGradientBC::essential_comps` uses:

| Code | Components | `comp_mask`      |
|------|------------|------------------|
| 1    | X          | `{T, F, F}`      |
| 2    | Y          | `{F, T, F}`      |
| 3    | Z          | `{F, F, T}`      |
| 4    | XY         | `{T, T, F}`      |
| 5    | XZ         | `{T, F, T}`      |
| 6    | YZ         | `{F, T, T}`      |
| 7    | XYZ        | `{T, T, T}`      |

The option layer additions live in `option_parser_v2.{hpp,cpp}`:

```cpp
struct PeriodicBC {
    std::vector<int> essential_ids;
    int              essential_comps = 7;
    void validate() const;
};

struct BoundaryOptions {
    // ...existing members...
    std::vector<PeriodicBC>      periodic_bcs;
    std::unordered_map<int, int> periodic_bc_entry_per_step;
};
```

`periodic_bc_entry_per_step` is a **sparse** step→entry map. Only
steps at which the spec transitions appear as keys. For a TOML with
`BCs.update_steps = [1, 5]` and two `[[BCs.periodic_bcs]]` blocks,
the map contains `{1 → 0, 5 → 1}`; steps 2–4 and 6+ inherit their
last installed spec via the state machine in §P5.18.6.

**The empty-`periodic_bcs` case (default fallback).** When
`periodic_bcs.empty()`, the manager synthesizes
`({all 6 face attributes}, essential_comps = 7)` at construction
time. This reproduces the pre-§P5.18 fully-constrained behavior
**bit-for-bit** — see §P5.18.11 — so existing TOMLs require no
edits.

### §P5.18.2 Two-axis filtering: pair filter and component filter

The spec resolves into two orthogonal mechanisms applied sequentially
through the row-emission pipeline:

**Pair filter** (`active_pair_labels`, derived from `essential_ids`):

- A face mortar pair is *active* iff its axis appears in
  `active_axes`. The axis map is left/right→x, bottom/top→y,
  front/back→z (matches the classifier's labeling convention).
- An edge mortar group is *active* iff **both** perpendicular axes
  appear in `active_axes`. For an x-parallel edge (perpendicular
  axes y and z), both the y-pair and the z-pair must be active.

**Component filter** (`comp_mask`, from `essential_comps`):

- Within an active pair, the c-component constraint row is emitted
  iff `comp_mask[c] == true`.

The pair filter operates on whole geometric groups; the component
filter is finer-grained and drops rows within otherwise-active
pairs. Both are applied jointly: a row is emitted iff its pair and
its component both pass.

**Why the edge filter is conservative.** Edge mortars exist to
handle the codimension-2 case where two face pairs interact along an
edge. The geometric match is between two 1-D nonmortar/mortar
intervals offset in two perpendicular directions; the existing
classifier and builder code assemble this match assuming *both*
perpendicular directions correspond to active periodicity. A more
permissive filter (single perpendicular axis active) would require
new "partial-edge" code paths that do not exist. The current
behavior is **drop the entire edge group** if either perpendicular
axis is inactive. For axis-aligned RVE test cases (the present
scope) this is correct; for irregular boundaries see §P5.18.12.

### §P5.18.3 EA operator row layout under filter

The `MortarConstraintOperator` (EA path) applies $C u$ and
$C^T \lambda$ matrix-free via per-pair block scatters into flat
arrays. Under the filter, the row layout changes from "vdim rows
per node" to "`m_n_comps_active` rows per node":

| Quantity                       | Pre-§P5.18 (vdim = 3)             | Post-§P5.18                              |
|--------------------------------|-----------------------------------|------------------------------------------|
| Rows per kept node             | 3                                 | `m_n_comps_active = popcount(comp_mask)` |
| `m_row_lambda_off[i]`          | `i * 3`                           | `i * m_n_comps_active`                   |
| Lambda slot for component `c`  | `lam_off + c`                     | `lam_off + m_local_c[c]` (skip if `< 0`) |
| `m_local_c[c]`                 | `c`                               | `LocalRowOfComp(comp_mask, c)`           |
| `Height()`                     | `m_n_active_rows * 3`             | `m_n_active_rows * m_n_comps_active`     |

`LocalRowOfComp(mask, c)` returns the position of `c` in the
subsequence of `true` entries of `mask`, or `-1` if `mask[c] ==
false`. For `mask = {T, F, T}`: `LocalRowOfComp` is `{0, -1, 1}`.
The table `m_local_c[3]` is pre-computed in `Reset` and captured by
value in the matvec kernels.

**`MortarConstraintOperator::Reset(active_pair_labels, comp_mask)`**
is the new entry point that re-walks pair blocks under a new filter
spec without destroying the operator. It:

1. Replaces `m_active_pair_labels` and `m_comp_mask`; recomputes
   `m_n_comps_active` and `m_local_c[3]`.
2. Calls `BuildFlatRowArrays()` (same routine the ctor uses) which
   re-emits `m_row_D`, `m_row_g_n_local`, `m_row_csr_off`,
   `m_csr_A`, `m_csr_g_m_local`, `m_csr_g_m_recv`, `m_row_lambda_off`,
   `m_n_active_rows` under the new filter.
3. Updates `height = m_n_active_rows * m_n_comps_active`.

**What `Reset` does *not* rebuild.** Three caches stay at their
ctor-time values across `Reset` calls:

- `m_local_edge_pairs` — all 9 edge mortar groups are assembled at
  ctor and kept regardless of filter. Filter applies at flat-array
  build time. Cheap; assembly cost dominated by face mortars anyway.
- `m_gtdof_lookup` — filter-independent.
- The off-rank import/export topology
  (`m_import_off_rank_gtdofs`, `m_export_local_gtdofs`,
  `m_import_recv_counts`, `m_import_displs`). The topology was
  sized at ctor for the **unfiltered union** of gtdofs referenced by
  any pair on this rank. Under reduced filter, this is a strict
  superset of what's actually exchanged at matvec time — we
  over-import some off-rank mortar values that the filtered kernel
  then doesn't read. This is correct (no missing data) and the
  waste is bounded by the original topology size, which is already a
  small fraction of matvec cost. Rebuilding the topology after
  `Reset` would require collective `MPI_Alltoall + MPI_Alltoallv`
  on every spec change — a contract change for `Reset` that the
  current design avoids. See §P5.18.12 for future work.

**MPI scope: local.** `Reset` is non-collective. However, **all
ranks must call `Reset` with identical arguments** — collective-by-
convention, the same pattern `MPI_Allreduce`'s op + datatype enforce
implicitly. The unchanged import/export topology is symmetric across
ranks, so as long as both sides agree on the new filter (which they
do, since both sides parse the same TOML), every exchange remains
self-consistent without an explicit collective.

**Kernel changes.** `Mult` (`mfem::forall` device kernel),
`MultTranspose` (host walk), and `ComputeInvDiagSchur` (host walk
over pair blocks) all take the filter into account:

```cpp
// Mult kernel sketch (device-side):
const int lc0 = m_local_c[0], lc1 = m_local_c[1], lc2 = m_local_c[2];
mfem::forall(n_rows, [=] MFEM_HOST_DEVICE (int i) {
    // ... scatter from u into per-node u_block ...
    const int lam_off = d_row_lambda_off[i];
    if (lc0 >= 0) d_y[lam_off + lc0] = /* component-0 dot product */;
    if (lc1 >= 0) d_y[lam_off + lc1] = /* component-1 dot product */;
    if (lc2 >= 0) d_y[lam_off + lc2] = /* component-2 dot product */;
});
```

The per-component branch is branch-free in the sense that the
predicates `lcN >= 0` are loop-invariant — they compile to either
the dot-product code or nothing, per thread.

### §P5.18.4 Manager-level orchestration: `RebuildForActiveSpec`

`MortarPbcManager::RebuildForActiveSpec` coordinates the cross-
component state update. It is called by SystemDriver's
`SyncMortarPbcForStep` (§P5.18.6) whenever the active periodic-BC
entry changes:

```cpp
void MortarPbcManager::RebuildForActiveSpec(
    const std::vector<int>& essential_ids,
    int                     essential_comps)
{
    // 1. essential_comps -> comp_mask via switch.
    const auto comp_mask = CompMaskFromInt(essential_comps);

    // 2. Pair-completeness validation + canonical active pair labels.
    const auto active_pair_labels =
        ValidateAndDeriveActivePairLabels(m_classifier, essential_ids);

    // 3. EA operator: re-walk flat row arrays under new filter.
    m_C_op.Reset(active_pair_labels, comp_mask);

    // 4. Saddle system: re-read cached size members (see §P5.18.7).
    m_saddle_system->Refresh();

    // 5. Corner ess TDOFs: spec-aware derivation.
    m_corner_ess_tdofs = ComputeCornerEssTDofsFromSpec(
        m_classifier, *m_fes, essential_ids, comp_mask);

    // 6. Lambda + g RHS buffers: resize in place.
    m_lambda.SetSize(m_C_op.Height());   m_lambda = 0.0;
    m_g_rhs .SetSize(m_C_op.Height());   m_g_rhs  = 0.0;

    // 7. Per-row metadata caches: re-emit under new filter.
    m_builder.EmitRowFactors(active_pair_labels, comp_mask,
                             m_period_signed_per_row,
                             m_component_per_row,
                             m_ell_hat_per_row);
}
```

**Pair-completeness validation.** For every `attr` in
`essential_ids`, the classifier resolves the partner attribute. If
the partner is not present in `essential_ids`, the call aborts with
a message naming both the present attr/label and the missing one:

```
MortarPbcManager::RebuildForActiveSpec: periodic BC entry references
face attribute 4 (label 'top') but its required pair partner
attribute 3 (label 'bottom') is missing from essential_ids. Both
halves of every pair must be listed.
```

`MortarPbcManager::SynthesizeDefaultPbcSpec(classifier)` is a static
helper that returns `(all face attrs, essential_comps = 7)` — the
spec equivalent to "fully constrained on every pair." Used by
SystemDriver's `SyncMortarPbcForStep` when the user's
`periodic_bcs` is empty.

**Vector-lifecycle invariant.** `m_lambda` and `m_g_rhs` are
`mfem::Vector` (not `BlockVector` or owning containers).
`SetSize(N)` reallocates the internal data buffer but **preserves
the `Vector` object's identity** (the same C++ object at the same
address). The saddle system's pointer to `m_g_rhs` (installed once
via `SetConstraintRHS` at manager construction, per §P5.0) stays
valid through all subsequent `RebuildForActiveSpec` calls. This
contrasts with `m_x_saddle` in SystemDriver (§P5.18.7), which is a
`std::unique_ptr<BlockVector>` and is **reallocated** on each
transition.

**Per-row metadata.** `m_period_signed_per_row` (the row-major
3-doubles-per-row period vector cache from §P5.7.A),
`m_component_per_row`, and `m_ell_hat_per_row` are all sized to the
post-`Reset` row count. They feed `UpdateConstraintRHS`'s
`mfem::forall` kernel that computes
$g_i = \dot{\bar F}_{c,k} \, L_k \, \hat{\ell}_i$ per row.

### §P5.18.5 Corner essential TDOFs under the spec

The corner pin set is derived by a free function in
`mortar_pbc_manager.cpp`:

```cpp
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D&        classifier,
    const mfem::ParFiniteElementSpace& fes,
    const std::vector<int>&            essential_ids,
    const std::array<bool, 3>&         comp_mask);
```

Two layered rules:

1. **Anchor unconditional pinning.** The classifier's "blf"
   (bottom-left-front, min in all three coordinates) corner is
   pinned in **all three components**, independent of the spec. This
   removes the three translation rigid-body modes. The function calls
   `classifier.AnchorCornerTDofs(fes)` (added in §P5.18's
   companion classifier extensions) to obtain the rank-local TDOFs.

2. **Spec-gated pinning for the other seven corners.** A non-anchor
   corner is eligible iff it is incident on at least one face listed
   in `essential_ids`, where incidence is resolved via
   `classifier.CornersOnFaceAttribute(attr)`. Eligible corners
   contribute the c-component TDOF iff `comp_mask[c] == true`.

**Incident-face gate vacuity on a 6-face axis-aligned RVE.** Every
corner of such an RVE sits at the extremum (min or max) of every
axis, so every corner is incident on exactly three of the six faces
— one face per axis. The pair-completeness validation in
`RebuildForActiveSpec` ensures that every active axis has both its
faces in `essential_ids`. Therefore, on any standard RVE, every
corner is incident on at least one listed face for any valid spec.
The incident-face gate is **provably vacuous** on this geometry.

The gate is nevertheless implemented explicitly because (a) it
matches the spec docstring literally, (b) it generalizes to
non-axis-aligned RVE boundaries where the equivalence is broken
(see §P5.18.12), and (c) the cost is one set lookup per non-anchor
corner — negligible.

**Rank-summed pin counts for representative specs (on any 6-face
axis-aligned RVE):**

| Spec                                                  | Anchor | Non-anchor (×7) | Total |
|-------------------------------------------------------|--------|-----------------|-------|
| all 6 attrs, comps = 7 (XYZ)                          | 3      | 21              | 24    |
| `{left, right}`, comps = 1 (X)                        | 3      | 7               | 10    |
| `{left, right, top, bottom}`, comps = 4 (XY)          | 3      | 14              | 17    |

**Rotation rigid-body modes are not auto-handled.** Anchor pinning
removes the three translation modes only. Under full XYZ specs, the
multi-corner pinning incidentally constrains rotations as well; under
sub-XYZ specs (X-only, XY, etc.), rotations about the unpinned axes
may remain. If $K$ is rank-deficient on those modes, the saddle
solver will fail to converge or produce non-unique solutions. The
current implementation does not attempt to detect or fix this;
users running sub-XYZ specs are responsible for adding rotation-
pinning corner Dirichlet BCs via the standard BC machinery if Newton
fails to converge in a rotational mode. See §P5.18.12 for the
sketch of an auto-handling extension.

### §P5.18.6 SystemDriver per-step hook

`SystemDriver::SyncMortarPbcForStep(int step_idx)` is the bridge
between the user TOML (`[[BCs.periodic_bcs]]` +
`BCs.update_steps`) and `MortarPbcManager::RebuildForActiveSpec`.
The intended call sequence in `mechanics_driver.cpp`:

```cpp
for (int ti = 1; ti <= total_steps; ++ti) {
    if (BCManager::GetInstance().GetUpdateStep(ti)) {
        oper.SyncMortarPbcForStep(ti);     // <-- NEW (this section)
        oper.UpdateEssBdr();
        oper.UpdateVelocity();
        oper.SolveInit();
    }
    if (m_mortar_enabled) {
        m_mortar_pbc->UpdateMacroscopicF(ess_velocity_gradient,
                                         GetCurrentDt());
        m_mortar_pbc->UpdateConstraintRHS();
    }
    oper.UpdateModel();
    oper.Solve();
    sim_state->FinishCycle();
    ti++;
}
```

`SyncMortarPbcForStep` precedes `UpdateEssBdr` because the latter
re-pushes the corner subset to `mech_operator`; if `Sync` has
transitioned to a new spec, the corners reflect the new spec when
`UpdateEssBdr` runs.

**State machine:**

| Condition                                                   | Action                                                                   |
|-------------------------------------------------------------|--------------------------------------------------------------------------|
| `!m_mortar_enabled`                                         | return (no-op)                                                           |
| `periodic_bcs.empty()` + first call                         | synthesize default, install, set `m_pbc_initialized = true`              |
| `periodic_bcs.empty()` + already initialized                | return (default is step-invariant)                                       |
| step in `entry_per_step_map` + entry == cached              | return (idempotent)                                                      |
| step in `entry_per_step_map` + entry ≠ cached               | `RebuildForActiveSpec`; resize `m_x_saddle`; re-`SetOperator` newton     |
| step NOT in map + already initialized                       | return (sparse `update_steps`; intermediate steps inherit)               |
| step NOT in map + first call                                | `MFEM_ABORT` (configuration error — start step must be in the map)       |

Two private members track state:

```cpp
bool m_pbc_initialized       = false;
int  m_pbc_active_entry_idx  = -1;   // -1 sentinel = synthesized default
```

The `-1` sentinel distinguishes "synthesized default installed"
from "user-supplied entry 0 installed." When the user's first entry
is semantically equivalent to the default, the first
`SyncMortarPbcForStep(1)` still triggers a (redundant but
correctness-preserving) rebuild because `-1 ≠ 0`.

**Constructor wiring change.** The §P5.5.A ctor pattern was:

```cpp
m_mortar_pbc = std::make_shared<MortarPbcManager>(...);
mech_operator->UpdateEssTDofsCornerSubset(
    m_mortar_pbc->GetCornerEssTDofs());
m_mortar_enabled = true;
```

Post-§P5.18 it becomes:

```cpp
m_mortar_pbc = std::make_shared<MortarPbcManager>(...);
m_mortar_enabled = true;             // MUST be set before Sync
SyncMortarPbcForStep(1);             // installs initial spec + corners
```

The `m_mortar_enabled = true` line **must precede**
`SyncMortarPbcForStep` because Sync early-returns when
`!m_mortar_enabled`. `SyncMortarPbcForStep(1)` itself calls
`mech_operator->UpdateEssTDofsCornerSubset` internally, so the
explicit pre-§P5.18 call is removed.

### §P5.18.7 Stale-cache refresh cascade

This is the gotcha that bit during X-only validation and warrants
a section of its own.

**Problem.** Several pieces of state cache the constraint-row count
at *construction* time. After `m_C_op.Reset(...)` mutates
`m_C_op.Height()`, those caches go stale:

| Cached state                                                  | Owner                       | Cached at         | Refresh mechanism                    |
|---------------------------------------------------------------|-----------------------------|-------------------|--------------------------------------|
| `m_n_u`, `m_n_lam`, `m_block_offsets`, `height`, `width`     | `MortarSaddlePointSystem`   | system ctor       | new `Refresh()` method (this batch)  |
| `m_x_saddle` BlockVector + `m_saddle_offsets`                | `SystemDriver`              | sys-driver ctor   | reallocation in `SyncMortarPbcForStep` |
| Newton solver internal vectors (if cached at `SetOperator`)  | `ExaNewtonSolver`           | `SetOperator` time | re-call `SetOperator` in `Sync`     |
| `m_C_op` reference internals in `MortarSaddlePreconditioner` | `MortarSaddlePreconditioner`| ctor              | none needed — re-`SetOperator`'d per Newton iter |
| `m_g_rhs` pointer in `MortarSaddlePointSystem`               | `MortarSaddlePointSystem`   | `SetConstraintRHS` | none needed — `Vector::SetSize` preserves object address |

**The diagnostic fingerprint.** Without the refresh cascade, the
linear-elastic X-only patch test aborts at the first Newton
iteration with:

```
Verification failed: (x_block.Size() == Width()) is false:
 --> MortarSaddlePointSystem::Mult: x_block size 429 != Width() 537
```

Decomposing on a 4×4×4 mesh: `375 = TrueVSize`; `537 − 375 = 162`
is the unfiltered XYZ lambda count; `429 − 375 = 54` is the
"all-pair-active, X-comp-only" count (9 edges × 3 × 1 + 3 faces ×
9 × 1). The saddle system reports its cached unfiltered `Width()`;
the caller's `x_block` is correctly sized to the post-`Reset` count
because it was built from `NumLocalConstraints()`. Mismatch.

**The fix has three pieces:**

1. **`MortarSaddlePointSystem::Refresh()`** (new public method):

```cpp
void MortarSaddlePointSystem::Refresh() {
    m_n_u   = m_C_op.Width();
    m_n_lam = m_C_op.Height();
    m_block_offsets[0] = 0;
    m_block_offsets[1] = m_n_u;
    m_block_offsets[2] = m_n_u + m_n_lam;
    height = m_n_u + m_n_lam;
    width  = m_n_u + m_n_lam;
}
```

Called from `MortarPbcManager::RebuildForActiveSpec` immediately
after `m_C_op.Reset(...)`. Local — no MPI. Idempotent.

2. **`m_x_saddle` reallocation** in `SyncMortarPbcForStep`:

```cpp
if (m_x_saddle) {     // null on the ctor's first call to Sync
    const int n_K   = mech_operator->Width();
    const int n_lam = m_mortar_pbc->NumLocalConstraints();
    m_saddle_offsets[1] = n_K;
    m_saddle_offsets[2] = n_K + n_lam;
    m_x_saddle = std::make_unique<mfem::BlockVector>(m_saddle_offsets);
    *m_x_saddle = 0.0;
    newton_solver->SetOperator(m_mortar_pbc->GetSaddleSystem());
}
```

The `if (m_x_saddle)` guard handles the ctor-time first call to
`Sync`: `m_x_saddle` is allocated *after* `Sync` returns (a few
lines later in the ctor, by the saddle-prec construction block),
sized correctly the first time from the already-updated
`NumLocalConstraints()`. For mid-run transitions, the guard
triggers and reallocates.

3. **`newton_solver->SetOperator`** is re-called after the BlockVector
   reallocation. The saddle system is the same `shared_ptr<Operator>`
   before and after `Refresh`, but some Newton implementations cache
   `Height()`/`Width()` at `SetOperator` time. Re-calling
   `SetOperator` forces any such cache to refill against the
   refreshed saddle system.

**Why `MortarSaddlePreconditioner` needs no refresh.** It holds
`m_C_op` by reference and queries `m_C_op.Height()`/`.Width()` at
matvec call time. Additionally, MortarSaddlePreconditioner is re-
`SetOperator`'d per Newton iteration. Both effects mean it picks
up the new sizes naturally.

**Why `m_g_rhs` (the saddle system's constraint-RHS pointer) needs no
refresh.** `mfem::Vector::SetSize` reallocates the internal data
buffer but preserves the `Vector` object's address. The saddle
system's pointer to `m_g_rhs` (installed once via
`SetConstraintRHS` at manager construction) is to the *object*, not
the data buffer; it remains valid across resizes. The saddle system
reads the buffer through the pointer at each `Mult` call, picking up
new contents automatically.

### §P5.18.8 Worked sizing example: X-only on a 4×4×4 mesh

Tracing the full data flow for the case that exercised the refresh
cascade:

**TOML:**
```toml
[[BCs.periodic_bcs]]
    essential_ids   = [1, 2, 3, 4, 5, 6]
    essential_comps = 1
```

(All 6 faces listed; only X component active. This produces "all
pairs active, X-only emitted" — 54 lambda rows.)

**Pre-§P5.18 / unfiltered total:** 9 edges × 3 × 3 + 3 faces ×
9 × 3 = 162.

**Post-§P5.18 filtered total:** 9 edges × 3 × 1 + 3 faces × 9 × 1 =
54 (pair filter keeps everything; comp filter drops 2/3).

**Construction sequence (`SystemDriver` ctor):**

| #   | Event                                                       | `m_C_op.Height()` | `saddle.Width()` | `m_x_saddle.Size()` |
|-----|-------------------------------------------------------------|-------------------|------------------|---------------------|
| 1   | Manager ctor; classifier + builder built                    | —                 | —                | —                   |
| 2   | `m_C_op` ctor sets `Height = 162` (unfiltered default)      | 162               | —                | —                   |
| 3   | `MortarSaddlePointSystem` ctor; caches `m_n_lam = 162`     | 162               | 537              | —                   |
| 4   | Manager ctor returns                                        | 162               | 537              | —                   |
| 5   | `m_mortar_enabled = true`                                   | 162               | 537              | —                   |
| 6a  | `Sync(1)` → `Reset({"right","top","back"}, {T,F,F})`        | **54**            | 537              | —                   |
| 6b  | `m_saddle_system->Refresh()`                                | 54                | **429**          | —                   |
| 6c  | Corner ess TDOFs recomputed (24 → 10 rank-summed)          | 54                | 429              | —                   |
| 6d  | `m_lambda`, `m_g_rhs` `SetSize(54)`                         | 54                | 429              | —                   |
| 6e  | `EmitRowFactors` re-emits per-row meta                      | 54                | 429              | —                   |
| 6f  | `mech_operator->UpdateEssTDofsCornerSubset(corners)`        | 54                | 429              | —                   |
| 7   | Saddle prec block: `m_x_saddle = make_unique<BV>(...)`      | 54                | 429              | **429**             |
| 8   | `newton_solver->SetOperator(saddle_system)`                 | 54                | 429              | 429                 |

**Step 1:**

`SyncMortarPbcForStep(1)` is called by `mechanics_driver`. Target
entry = 0; cached entry = 0; idempotent — no-op. `UpdateEssBdr`,
`UpdateVelocity`, `SolveInit`, `Solve` proceed. Newton's
`saddle_system->Mult(x_block, r_block)` sees `x_block.Size() = 429`
and `Width() = 429`; verification passes; Newton converges.

### §P5.18.9 TOML configuration examples

**Backward-compatible (no TOML edits needed).** Existing simulations
with `[Mesh] periodicity = true` and no `[[BCs.periodic_bcs]]`
blocks get the synthesized default at SystemDriver ctor — same
24-corner pinning, same 162-row constraint matrix, bit-for-bit
identical observable behavior to pre-§P5.18.

**Single-entry X-only PBC (the validation case):**

```toml
[Mesh]
    periodicity = true

[[BCs.velocity_gradient]]
    essential_ids = [1, 2, 3, 4, 5, 6]
    L_bar = [[0.01, 0, 0], [0, 0, 0], [0, 0, 0]]

[[BCs.periodic_bcs]]
    essential_ids   = [1, 2, 3, 4, 5, 6]
    essential_comps = 1                   # X only

[BCs]
    update_steps = [1]
```

**Multi-entry (XYZ for steps 1–4, X-only starting step 5):**

```toml
[[BCs.periodic_bcs]]
    essential_ids   = [1, 2, 3, 4, 5, 6]
    essential_comps = 7                   # XYZ

[[BCs.periodic_bcs]]
    essential_ids   = [1, 2, 3, 4, 5, 6]
    essential_comps = 1                   # X-only

[BCs]
    update_steps = [1, 5]
```

At step 1: `Sync(1)` finds entry 0 → installs XYZ. Steps 2–4: no
entries in map → no-op (inherit). Step 5: `Sync(5)` finds entry 1
→ rebuilds for X-only, resizes `m_x_saddle`. Steps 6+: no entries
→ no-op (inherit X-only).

**Single-pair X-only (true uniaxial-X PBC):**

```toml
[[BCs.periodic_bcs]]
    essential_ids   = [1, 2]              # only left/right
    essential_comps = 1
```

Pair-completeness check passes (1 and 2 are a pair). Pair filter
keeps the x-pair only; 0 edge groups active (no perp-axis pair has
both axes in active_axes). Comp filter keeps X only. Total: 1 face
× 9 interior × 1 comp = **9 lambda rows**. 10 corner TDOFs.

### §P5.18.10 Test coverage

**Existing `test_constraint_builder_3d.cpp`** updated with three
new filter cases (in addition to fixing the §P5.7.A
`EmitRowFactors` signature: `Vector& period_signed_per_row`
replaces `Array<int>& axis_idx`):

| Test                                             | Coverage                                        |
|--------------------------------------------------|-------------------------------------------------|
| `test_filter_x_only_2x2x2`                       | comp_mask = X-only → 12 rows (= 36/3)           |
| `test_filter_x_face_pair_only_2x2x2`             | only 1 face pair in active_pair_labels → 3 rows |
| `test_filter_empty_2x2x2`                        | empty pairs or empty comp_mask → 0 rows         |

**New `test_mortar_pbc_manager_filter.cpp`** exercises
`ComputeCornerEssTDofsFromSpec` directly (no full-manager
construction):

| Test                                               | Expected rank-summed TDOFs |
|----------------------------------------------------|----------------------------|
| `test_full_xyz`                                    | 24 (matches pre-§P5.18)    |
| `test_x_only_single_pair`                          | 10                         |
| `test_xy_two_pairs`                                | 17                         |
| `test_anchor_only_empty_essential_ids`             | 3 (gate drops 7 non-anchor)|
| `test_round_trip_xyz_xonly_xyz`                    | 24, 10, 24                 |

**Integration testing** of `RebuildForActiveSpec` + `SyncMortarPbcForStep`
end-to-end is validated through production simulation: the
linear-elastic uniaxial-X patch test that exposed the refresh-cascade
hotfix now passes with `Width() = x_block.Size() = 429`. A formal
multi-entry production test (2-step load with full-XYZ at step 1 and
X-only at step 5, verifying the transition triggers a rebuild and
the resulting velocity field exhibits correct lateral Poisson
contraction) is future work — see Phase 5.9 batch E in §P5.13 and
§P5.18.12.

### §P5.18.11 Backward-compatibility invariants

These invariants are guaranteed by §P5.18 and should be verified
after any future change touching the filter machinery:

1. **Manager defaults (no `RebuildForActiveSpec` call).** After
   `MortarPbcManager(...)` construction and before any
   `SyncMortarPbcForStep` / `RebuildForActiveSpec` call:
   `NumLocalConstraints()` = unfiltered local row count;
   `GetCornerEssTDofs().Size()` rank-sums to 24;
   `m_n_comps_active = 3`, `m_local_c = {0, 1, 2}`,
   `m_row_lambda_off[i] = i * 3`.

2. **Synthesize-and-rebuild is a no-op for fully-constrained.**
   Calling `RebuildForActiveSpec` with
   `SynthesizeDefaultPbcSpec(classifier)`'s output produces flat
   row arrays bit-for-bit identical to the pre-§P5.18 ctor state.

3. **Default-fallback runs match pre-§P5.18 observable behavior.**
   With `BoundaryOptions::periodic_bcs.empty()`, the ctor's
   `SyncMortarPbcForStep(1)` synthesizes the default and applies
   it; `Mult` / `MultTranspose` / `ComputeInvDiagSchur` produce
   identical outputs to the pre-§P5.18 path; the saddle system's
   `Width() / Height()` equals the unfiltered sum; 24-corner pinning
   is in effect.

4. **`SyncMortarPbcForStep` idempotence.** Calling
   `SyncMortarPbcForStep(N)` for any `N` that doesn't transition to
   a new entry is a complete no-op: zero work, zero state mutation.
   Verifiable by reading `m_pbc_active_entry_idx` before/after —
   same value.

### §P5.18.12 Future work

1. **Non-axis-aligned RVE meshes.** The current edge filter rule
   ("both perpendicular axes active") and anchor convention
   ("blf" = min in all 3 coordinates) are tied to the axis-aligned
   assumption. The incident-face gate
   (`CornersOnFaceAttribute`) is implemented literally so it
   generalizes, but the other two need revisitation for irregular
   boundaries. Likely entry points:
   - `EdgePerpendicularAxes` becomes problem-specific.
   - Anchor selection needs a generalized rule (e.g. "the corner
     with smallest sum-of-coordinates in a canonical projection").
   - Possibly per-edge filter rather than per-edge-group filter.

2. **Tighter off-rank import/export topology under reduced filter.**
   `MortarConstraintOperator::Reset` does not rebuild the
   import/export topology (over-imports under reduced filter; see
   §P5.18.3). The waste is bounded, but for very large filter
   reductions on very large problems, rebuilding could be
   worthwhile. The cost is that `Reset` becomes collective
   (`MPI_Alltoall + MPI_Alltoallv`). Worth doing only if profiling
   identifies the over-import as a hot spot.

3. **Rotation rigid-body auto-handling for sub-XYZ specs.** As noted
   in §P5.18.5, sub-XYZ specs may leave rotation modes
   unconstrained. A future enhancement:
   - Detect which rotation modes are unconstrained for the active
     spec.
   - Add the minimum set of corner-component pins to remove them.
   - Expose a TOML flag to opt in/out.

   Non-trivial: minimum-pin selection is a discrete optimization
   over the null-space basis. Likely worth doing only after specific
   user demand.

4. **Per-entry early validation.** `PeriodicBC::validate()` currently
   checks `essential_ids` non-empty + positive +
   `essential_comps ∈ {1..7}`. Pair-completeness is deferred to
   `RebuildForActiveSpec` because it requires the classifier. A
   future enhancement could move pair-completeness into early
   validation if the classifier (or a "boundary topology" struct)
   becomes available at TOML-parse time.

5. **Formal multi-entry integration test.** The manager-free unit
   test (§P5.18.10) covers corner-derivation logic. A multi-entry
   production test exercising mid-run spec transitions would catch:
   `m_x_saddle` reallocation correctness; Newton behavior across
   the transition (does `dlam` reset semantically?); volume-average
   diagnostic consistency. Heavy SimulationState setup; deferred to
   Phase 5.9.E.

---

## §P5.19 Saddle-system residual scaling (Phase 5.11)

The Phase 5.0–5.10 stack delivers mortar PBC with Newton iterating on
the unscaled saddle system

$$
A x = \begin{bmatrix} K & C^\top \\ C & 0 \end{bmatrix}
      \begin{bmatrix} u \\ \lambda \end{bmatrix}
    = \begin{bmatrix} f \\ g \end{bmatrix}.
$$

The block magnitudes are mismatched by construction: $\|K\|$ scales
with the mesh-discretized stiffness (typically $\mathcal{O}(10^4)$
to $\mathcal{O}(10^9)$ for stiff polycrystals), while
$\|C\| \approx 1$ from the dimensionless biorthogonal projector. The
saddle-point Krylov solver (MINRES + block-diagonal preconditioner)
inherits this disparity in its residual sequence: the $\lambda$-block
residual is many orders of magnitude smaller than the $u$-block, the
shared convergence test compares both against a single tolerance, and
Newton's outer convergence ends up gated almost entirely on the
$u$-block residual. The $\lambda$ block is effectively unconstrained
at convergence time — sometimes acceptable, sometimes the source of
quiet Hill-Mandel violations and slow asymptotic convergence on
plasticity problems with active constraints.

§P5.19 introduces a per-step block-diagonal scaling that rebalances
the residual blocks to unit magnitude at Newton iteration 0, so the
shared convergence test exerts uniform pressure on every block.
Everything else about the Phase 5 architecture is preserved: the
`MortarPbcManager`, the saddle system, the Newton family, the
constraint filter from §P5.18, all unchanged when scaling is
disabled.

### §P5.19.1 The asymmetric block-diagonal scaling

Define a block-diagonal scaling matrix

$$
D = \begin{bmatrix} d_u\, I_u & 0 \\ 0 & D_\lambda \end{bmatrix},
$$

where $d_u$ is a single scalar applied to the $u$-block and
$D_\lambda$ is piecewise-constant diagonal — one factor per
sub-block (see §P5.19.3 for the partition). The scaled saddle
system is

$$
\tilde{A} = D^{-1} A D^{-1},
\qquad
\tilde{r} = D^{-1} r,
\qquad
\Delta x_\text{phys} = D\, \Delta x_\text{solver}.
$$

The scaling is **asymmetric in the math sense** — the Jacobian is
conjugated by $D^{-1}$ on both sides (so $\tilde{A}$ stays
symmetric for symmetric $A$), but the residual is scaled by
$D^{-1}$ on one side only and the increment by $D$. This is the
standard symmetric-block-diagonal-scaling formulation; the
"asymmetric" label refers to the fact that residual scaling
(`D^-1 r`) and increment unscaling (`D dx`) use $D$ in different
directions, NOT to any breakdown of symmetry in the operator.

Useful identities (used throughout the wrapper implementations):

| Object         | Solver-coord form | Physical-coord form           |
|----------------|-------------------|-------------------------------|
| residual       | $\tilde r$        | $r = D \tilde r$              |
| Jacobian       | $\tilde J$        | $J = D \tilde J D$            |
| Jacobian-T     | $\tilde J^\top$   | $J^\top = D \tilde J^\top D$  |
| increment      | $\widetilde{\Delta x}$ | $\Delta x = D \widetilde{\Delta x}$ |
| inner product  | $\tilde u^\top \tilde v$ | $u^\top D^{-2} v$       |

Newton iterates in solver-coords. The wrappers transparently
translate at the boundaries: residual `Mult` produces $\tilde r$;
J-solve produces $\widetilde{\Delta x}$ then un-applies $D$ to
return $\Delta x$ to Newton's outer arithmetic.

### §P5.19.2 Rule A: unit-balance with floor/cap guards

The per-step scaling factor selector ("`Choose`") implements
**Rule A unit-balance**: choose each factor to make its block's
scaled residual norm equal to 1 at iteration 0.

For a block with initial norm $\|r_k\|$:

$$
d_k = \text{ScaleFromNorm}(\|r_k\|; \text{floor}, \text{cap})
    = \begin{cases}
        1 & \text{if } \|r_k\| < \text{floor} \\
        \min(\|r_k\|, \text{cap}) & \text{otherwise}.
      \end{cases}
$$

- **floor** (default $10^{-12}$). Block norms below this are treated
  as zero — divide-by-near-zero would otherwise inflate the factor
  catastrophically. The block's scaling defaults to $d_k = 1$
  (identity), which is the right behavior when the block has
  nothing to contribute.
- **cap** (default $10^{12}$). Block norms above this are clipped.
  Prevents extreme values amplifying floating-point error in the
  Krylov.

Both guards are pure safety; the typical operating regime has
$\|r_k\|$ between $10^{-4}$ and $10^{6}$ and the guards never fire.

`Choose` is called once per time step, BEFORE Newton's iter 0,
using initial residual block norms (MPI-reduced by the manager).
The scaling stays fixed through the step. Phase 5.9 spec
transitions trigger a `RebuildPartition` which resets all factors
to 1 — the next step's `Choose` repopulates them.

The choice not to re-`Choose` mid-step is deliberate: changing $D$
mid-Newton would make $\tilde r^{(k+1)}$ incomparable to
$\tilde r^{(k)}$, breaking the convergence-rate logic in NRLS
backtracking and the trust-region update rule in TRDOG. Per-step
constancy keeps the scaled problem well-defined as a single
nonlinear system.

### §P5.19.3 Sub-block partition for the $\lambda$ block

$D_\lambda$ is **piecewise-constant per sub-block** — one factor
$d_\lambda^{(k)}$ shared across all rows in sub-block $k$. Two
partition schemes are supported via `mortar_pbc::SubblockPartition`
(declared in `constraint_builder_3d.hpp`, deliberately kept
separate from the options-side enum of the same name):

- **`FaceEdge`** (default): 2 sub-blocks. Sub-block 0 = all rows
  from active edge mortar groups; sub-block 1 = all rows from
  active face mortar pairs. The coarsest physically meaningful
  partition; always exposes 2 labels regardless of filter state
  (an empty sub-block under a restrictive §P5.18 spec is allowed).
- **`PerPair`**: one sub-block per active mortar pair, in walk
  order — edges from `m_classifier.EdgePairs()` first, then faces
  from `m_classifier.FacePairs()`. Sub-block count varies with the
  filter spec; full-XYZ unfiltered yields $9 + 3 = 12$
  sub-blocks; X-only yields 1 (the x-face pair, edges all
  dropped).

The partition is built once at manager construction (or each
`RebuildForActiveSpec` call) by walking `ConstraintBuilder3D::
GetRowSubblockIds`, which produces both the per-row sub-block index
array and the label vector consumed downstream by the diagnostic
logger.

**Why per-sub-block instead of a single $d_\lambda$**: heterogeneous
loading often saturates one constraint direction (e.g. the load
axis) while leaving the others slack. A single scalar averages the
block norms and ends up too small for the saturated direction and
too large for the slack ones. Per-sub-block factors handle this
naturally — the saturated direction gets its own normalization, the
slack ones stay near identity.

The `cfg.per_subblock = false` mode (config option) reverts to the
single-scalar formulation as a degenerate case (all
$d_\lambda^{(k)}$ tied to the joint norm). No separate code path.

### §P5.19.4 Wrappers: operator, solver, preconditioner

Three thin decorators sit between the Newton family and the Phase
4.3 saddle stack:

```
                +-------------------+
   Newton  -->  | ScaledSaddleOp    |  -- residual: D^-1 (A x - b)
   (oper)       | (wraps saddle op) |     Jacobian: ScaledJacobianOperator
                +-------------------+

                +-------------------+
   Newton  -->  | ScaledSaddleSolver|  -- in:  scales rhs by D^-1
   (prec)       | (wraps J_solver)  |     out: scales increment by D
                +-------------------+

                +-------------------+
   J_solver --> | ScaledSaddlePrec  |  -- conjugates inner prec by D^-1
   (prec)       | (wraps J_prec)    |     so the Krylov sees a coherent
                +-------------------+     scaled preconditioner
```

All three are constructed unconditionally by `SystemDriver`'s
mortar block (regardless of the scaler's enabled state). When
`cfg.enabled == false`, the wrappers act as math no-ops via an
identity short-circuit on the `IsEnabled()` test inside their
`Mult` methods — bit-for-bit equivalent to the unwrapped saddle
stack. When `cfg.enabled == true`, the wrappers are installed onto
the Newton solver in the SystemDriver mortar-setup block (§P5.5)
via three `SetOperator` / `SetSolver` / `SetPreconditioner` calls,
gated on `IsEnabled()`.

The two non-obvious math details:

- **`ScaledJacobianOperator::Mult` and `MultTranspose` evaluate
  $D^{-1} J D$ and $D J^\top D^{-1}$ on the fly** — no $D J D$
  matrix is ever assembled. Each call is `ApplyToIncrement`
  (multiply by $D$), `inner_J->Mult`, then `ApplyToResidual`
  (multiply by $D^{-1}$). Cost per `Mult`: two element-wise
  vector passes on top of the inner Jacobian apply.
- **`ScaledSaddleSolver::Mult` does pre-scale + delegate +
  un-scale**: input `b` (which Newton hands in as the scaled
  residual already, from `ScaledSaddleOperator::Mult`) goes
  straight to the inner Krylov; output `x` is in solver-coords
  and gets `ApplyToIncrement`'d (i.e. multiplied by $D$) before
  return so Newton sees $\Delta x_\text{phys}$.

A critical implementation detail surfaced in §P5.14.15: each
wrapper must forward Newton's `iterative_mode` flag to its inner
solver inside `Mult`. Without that forward, the inner uses its own
stale flag and may read uninitialized buffer bytes — a classic
non-determinism trap.

### §P5.19.5 Newton-family integration (NR, NRLS, TRDOG)

**NR and NRLS** absorb the scaling without any solver-side change.
Newton's outer arithmetic is in solver-coords:

```
r̃   <- oper_mech->Mult(x, r̃);              // ScaledSaddleOp
Δx̃  <- prec_mech->Mult(r̃, Δx̃);             // ScaledSaddleSolver, returns Δx_phys
                                            //   = D · Δx̃
x   <- x - α · Δx_phys                      // x is in physical coords
```

The trick is that `ScaledSaddleSolver` does the `Δx̃ → Δx_phys`
unwrap internally before return, so Newton sees physical increments
even though the residual it reads is scaled. The NRLS merit
function (norm of the scaled residual) is monotone on the scaled
problem by construction, so the line search works without
modification.

**TRDOG** needs more care because the dogleg body interpolates
between two directions — the Newton step (from the J-solve) and
the steepest-descent step (from $J^\top r$ via
`MultTranspose`). These come back in different coords:

- The Newton step from `prec_mech->Mult` is $\Delta x_\text{phys}$
  (post-unwrap).
- The gradient from `MultTranspose` is naturally $\tilde J^\top \tilde r$
  in solver-coords.

To interpolate consistently, TRDOG converts the Newton step back
to solver-coords with one `ApplyToIncrement` call (divide by $D$)
before doing the dogleg interpolation, then `UnapplyToIncrement`s
the final dogleg result before applying it to $x$. The scaler is
passed in via `ExaTrustRegionSolver::SetScaler(scaler, offsets)` at
SystemDriver setup time when scaling is enabled.

### §P5.19.6 Diagnostic logger

The `SaddleNewtonDiagnosticLogger` writes a CSV row per Newton
iter, capturing:

```
step, iter, norm, norm0, norm_max, converged_now, scaler_enabled,
res_K, res_lam, res_lam_<sub-block>...,
d_u, d_lam_<sub-block>...,
inner_iter, inner_norm,
dx_u_norm, dx_lam_norm_<sub-block>...
```

Pre-solve fields (norm down through `d_lam_<sub-block>`) come from
the `NewtonDiagnosticSink` callback (§P5.13 Phase 5.11.F), invoked
at the top of each Newton iter after the new residual norm is
computed. Post-solve fields (`inner_iter` through
`dx_lam_norm_<sub-block>`) come from the `InspectingIterativeSolver`'s
post-solve callback (§P5.19.7). The logger buffers a row at
pre-solve and flushes when the post-solve fires (or immediately on
Newton convergence, since the converged-now iter doesn't have a
subsequent J-solve).

Norms are physical — the logger reads the residual as-is from
Newton's iterate pointer. Per-block decomposition uses the
scaler's `SubblockOfRow` partition table to split the
$\lambda$-block by sub-block. The Python analyzer
`scripts/analyze_newton_log_v2.py` produces per-step convergence
summaries and detects three pathology classes: scaling-factor
drift between steps, K-block-vs-$\lambda$-block asymmetric
convergence, and per-sub-block-specific stalls.

`scaler_enabled` is included as a column so a single CSV is
self-describing — runs with scaling off and on can be archived
together without ambiguity about which is which.

### §P5.19.7 `InspectingIterativeSolver` and two-run diff workflow

`mortar_pbc::InspectingIterativeSolver` is a `mfem::Solver` wrapper
that fires a callback after each `Mult` invocation, supplying the
inner Krylov's iteration count + final norm + the $(b, x)$ pair.
When installed between Newton and the J-solver, it lets the
diagnostic logger pair pre-solve and post-solve telemetry into a
single CSV row. The inner Krylov reference (for `GetNumIterations`
and `GetFinalNorm`) is bound via `std::function` captures at
SystemDriver setup time — the deepest layer (typically MINRES,
underneath the `ScaledSaddleSolver` wrapper), not the wrapper
chain, since `GetNumIterations` lives on `mfem::IterativeSolver`
not on the wrapper's `mfem::Solver` base.

**Two-run diff workflow**: with `InspectingIterativeSolver` in
place, wrapper-transparency tests become a CSV diff:

1. Run with `cfg.enabled = true` and `cfg.floor = 1.0e+30`. The
   floor-guard branch in `Choose` fires on every block at every
   step, returning 1.0 for every factor → $D = I$ exactly. Logger
   writes `newton_iters_scaled.csv`.
2. Run with `cfg.enabled = false`. No wrappers installed. Logger
   writes `newton_iters_unscaled.csv`.
3. Diff:

   ```sh
   python3 scripts/analyze_newton_log_v3.py \
       --diff newton_iters_unscaled.csv newton_iters_scaled.csv
   ```

The analyzer aligns rows by `(step, iter)`, filters out
`scaler_enabled` (expected difference), and reports the first
column with `|base - compare| > eps` (default 0.0 — exact match).

At $D = I$ the two runs should produce bit-equal CSVs (modulo
`scaler_enabled`). Any divergent column at the smallest
$(step, iter)$ localizes the wrapper that broke transparency. The
column → suspect mapping the analyzer prints:

| First divergent column | Suspect wrapper                                          |
|------------------------|----------------------------------------------------------|
| `norm`, `norm0`        | `ScaledSaddleOperator::Mult` residual eval               |
| `res_K`                | u-block path through `ScaledSaddleOperator` or scaler    |
| `res_lam`              | lambda-block path                                        |
| `res_lam_<label>`      | Specific sub-block in the scaler partition               |
| `inner_iter`           | Wrapped operator/precond driving MINRES off course       |
| `inner_norm`           | Same — Krylov trajectory diverges                        |
| `dx_u_norm`            | `ScaledSaddleSolver` post-solve unwrap on u-block        |
| `dx_lam_norm_<label>`  | Same, on lambda sub-block                                |

The ordering reflects the Newton compute order: residual evaluation
is BEFORE the J-solve, so a `norm` divergence implicates the
residual evaluator; if `norm` matches but `inner_iter` differs the
Lanczos trajectory is being perturbed by the wrapped operator or
preconditioner; if `inner_iter` matches but `dx_*` differs, the
post-solve unwrap step is the suspect.

This is a divide-and-conquer aid, not a root-cause finder — it
points at the suspect wrapper, but the actual buggy line inside
that wrapper still needs inspection. Once the wrappers pass the
transparency test at $D = I$, the diff workflow can be repurposed
(with `eps > 0`) to compare different scaling configurations
against each other for tuning.

### §P5.19.8 File manifest

```
src/mortar_pbc/saddle_residual_scaler.{hpp,cpp}        — Phase 5.11.C
src/mortar_pbc/saddle_scaling_wrappers.{hpp,cpp}       — Phase 5.11.D
src/mortar_pbc/saddle_newton_diagnostic_logger.{hpp,cpp} — Phase 5.11.J
src/mortar_pbc/inspecting_iterative_solver.{hpp,cpp}   — Phase 5.11.K
scripts/analyze_newton_log_v2.py                       — Phase 5.11.J
scripts/analyze_newton_log_v3.py                       — Phase 5.11.K
```

Plus per-batch additions in:

```
src/options/option_parser_v2.{hpp,cpp}                 — 5.11.A
src/mortar_pbc/constraint_builder_3d.{hpp,cpp}         — 5.11.B
src/mortar_pbc/mortar_pbc_manager.{hpp,cpp}            — 5.11.E
src/solvers/mechanics_solver.{hpp,cpp}                 — 5.11.F/G
src/system_driver.{hpp,cpp}                            — 5.11.H/I/K
```

---

End of `PHASE5_EXACONSTIT_INTEGRATION_v7.md`.
