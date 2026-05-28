# Phase 6 (v2) — Higher-Order Primal Field via LOR Projection: Unified Architecture

> **This document is v2 and supersedes v1 (`PHASE6_HIGHER_ORDER_LOR.md`) in
> its entirety.** v1 framed the LOR support as an additive wrapper layer
> (`HighOrderSetup`, `HighOrderMortarConstraintOperator`,
> `SurfaceLORProjector`) sitting on top of an unchanged Phase 4 pipeline,
> with a runtime composition between the wrapper and the underlying
> `MortarConstraintOperator`. v2 abandons that wrapper approach in favor of
> a **unified projector architecture** in which the constraint operator
> consumes a `SurfaceProjector` at construction time and uses it to
> translate classifier-side TDOF indices to parent-FES TDOF indices once,
> baking the translation into the flat-row arrays. The runtime matvec is
> identical in structure to the pre-Phase-6 implementation; the LOR
> machinery becomes a construction-time concern, not a per-iteration
> concern.
>
> **What v2 changes architecturally**
>
> 1. **`HighOrderSetup` is eliminated.** Its responsibilities split between
>    `SimulationState` (which becomes the canonical owner of all shared
>    mesh and FES infrastructure, including the LOR-refined boundary
>    submesh and its FES) and `MortarPbcManager` (which holds the
>    projector and the classifier as direct shared-pointer members).
>
> 2. **`HighOrderMortarConstraintOperator` is eliminated.** The existing
>    `MortarConstraintOperator` is refactored to accept a `SurfaceProjector`
>    at construction. There is no wrapper class; there is one
>    constraint-operator class that operates uniformly in direct
>    ($p = 1$) and LOR-projected ($p \ge 2$) modes.
>
> 3. **`BoundaryClassifier3D` is decomposed (option B from the design
>    review)**: the classifier now always operates on a pre-built
>    boundary `ParSubMesh` and an FES *defined on that submesh* (rather
>    than on the parent volume mesh). The boundary-submesh extraction
>    that was previously inside the classifier moves up to
>    `SimulationState` where it lives alongside the other shared mesh
>    infrastructure. A convenience free function
>    `MakeBoundaryClassifierFromParent` is provided for test code that
>    builds classifiers directly from parent meshes.
>
> 4. **Smart-pointer migration across the mortar-PBC stack.** Components
>    that were previously held by reference or raw pointer (the
>    classifier inside the constraint operator, the operator inside the
>    saddle system and the preconditioner, the constraint builder
>    inside the classifier) are migrated to `std::shared_ptr<T>` (or
>    `std::shared_ptr<const T>` when the holder genuinely does not
>    mutate). Ownership becomes explicit at every interface.
>
> 5. **The "Option γ" commitment.** In direct-path ($p = 1$) operation,
>    the constraint operator goes through the same projector-based
>    indexing as the LOR path; at $p = 1$ the projector is a degenerate
>    permutation between the parent FES's boundary Lagrange nodes and
>    the (un-refined) boundary submesh's FES TDOFs. The pre-Phase-6
>    "direct-with-`GetParentVertexIDMap`" path is retired. The
>    architecture is uniform; there is no special direct-mode code path.
>
> **What v2 preserves from v1**
>
> The Phase 6.1 theoretical foundation (the Pazner-Kolev coincidence
> property, the linear dual basis on the refined submesh, the
> recommendation to defer the Popp-Wohlmuth-Gee-Wall basis-transformation
> route until forced, the deferred Barbosa-Hughes stabilization, tet
> $p=2$ as the Day 1 target with hex $p=2$ later) all carry over
> unchanged. The phasing structure is similar but reflects the
> consolidation: v1's Phase 6.1-6.5 collapses into v2's Phase 6.0-6.2,
> with Phase 6.0 (foundational refactors) carrying the bulk of the
> implementation cost and Phase 6.1-6.2 being mostly enablement and
> validation.
>
> **Companion documents**:
> - `MORTAR_PBC_ARCHITECTURE.md` — the main architecture doc. §4.11 and
>   §4.12 cover the LOR theory and the recommendation for ExaConstit.
>   §5 covers the Wohlmuth modifications; these continue to apply
>   on the LOR submesh without change.
> - `PHASE4_CPP_PORT_PLAN.md` — Phase 4 implementation plan. §P4.4
>   describes the classifier, constraint builder, and operator. v2
>   refactors these.
> - `PHASE5_EXACONSTIT_INTEGRATION_v7.md` — Phase 5 integration plan.
>   §P5.4 (`MortarPbcManager`), §P5.8 (UL constraint), §P5.18
>   (component-restricted PBC), §P5.19 (saddle-system residual
>   scaling, implemented in Phase 5.11 batch) all interoperate with
>   the v2 architecture.
> - `PHASE6_HIGHER_ORDER_LOR.md` — v1 of this document; **superseded**.
>
> **Cross-references**: §X.Y refers to the architecture doc; §P4.X.Y
> to the Phase 4 plan; §P5.X.Y to the Phase 5 plan v7; §P6.X.Y to
> this v2 document. The Phase 5 plan distinguishes `§P5.X` (sections
> in the doc) from `Phase 5.X` (batches in the §P5.13 phasing); these
> are independent numberings (e.g. §P5.19 is the section documenting
> the work done in Phase 5.11 the batch).
>
> **Loading this document into a fresh conversation**: Pair this file
> with `MORTAR_PBC_ARCHITECTURE.md` (especially §4.10-§4.12, §5),
> `PHASE5_EXACONSTIT_INTEGRATION_v7.md`, and the headers of the four
> core mortar-PBC classes (`BoundaryClassifier3D`, `ConstraintBuilder3D`,
> `MortarConstraintOperator`, `MortarPbcManager`) plus
> `SimulationState`. Together they are sufficient context to resume
> Phase 6 from any phase boundary.

---

## Table of Contents

- §P6.1 Goals and non-goals
  - §P6.1.1 Goals
  - §P6.1.2 Non-goals (explicitly deferred)
- §P6.2 Why higher order matters now (or soon)
- §P6.3 Theory: LOR projection on refined boundary submeshes
  - §P6.3.1 The Pazner-Kolev coincidence property
  - §P6.3.2 The linear dual basis transports to LOR without modification
  - §P6.3.3 Stability and convergence; why we tolerate the p / 1 pairing
  - §P6.3.4 Why LOR and not Popp-Wohlmuth-Gee-Wall
  - §P6.3.5 The Hill-Mandel consistency check at higher order
- §P6.4 Architectural overview: the unified projector architecture
  - §P6.4.1 The three architectural options considered
  - §P6.4.2 Why option γ wins
  - §P6.4.3 The runtime data flow
  - §P6.4.4 Component responsibility matrix
- §P6.5 SimulationState as the home for shared mesh infrastructure
  - §P6.5.1 Existing infrastructure (recap)
  - §P6.5.2 New accessors
  - §P6.5.3 The aliasing pattern at lor_depth = 1
  - §P6.5.4 Implementation sketch
  - §P6.5.5 Gotchas
- §P6.6 BoundaryClassifier3D refactor
  - §P6.6.1 What changes and why
  - §P6.6.2 The new constructor signature
  - §P6.6.3 Internal algorithm: from parent-FES gtdofs to submesh-FES gtdofs
  - §P6.6.4 Shared-pointer migration of members
  - §P6.6.5 MakeBoundaryClassifierFromParent convenience helper
  - §P6.6.6 Gotchas
- §P6.7 The SurfaceProjector class
  - §P6.7.1 Responsibility and lifetime
  - §P6.7.2 Mathematical formulation
  - §P6.7.3 The TDofMap data structure
  - §P6.7.4 Vertex matching algorithm
  - §P6.7.5 Cross-rank topology and Alltoallv
  - §P6.7.6 Mult and MultTranspose at runtime
  - §P6.7.7 Gotchas
- §P6.8 MortarConstraintOperator under the projector
  - §P6.8.1 Constructor and member layout
  - §P6.8.2 BuildFlatRowArrays with projector-mediated TDOF translation
  - §P6.8.3 Width and Height semantics
  - §P6.8.4 Mult and MultTranspose are unchanged at runtime
  - §P6.8.5 ComputeInvDiagSchur with projector mediation
  - §P6.8.6 Reset under the new architecture
  - §P6.8.7 Bit-equivalence considerations
- §P6.9 ConstraintBuilder3D parallel changes
  - §P6.9.1 The HypreParMatrix path
  - §P6.9.2 EmitConstraintTriples with projection
  - §P6.9.3 EmitRowFactors invariants
- §P6.10 MortarPbcManager simplification
  - §P6.10.1 Single classifier
  - §P6.10.2 Construction sequence
  - §P6.10.3 RebuildForActiveSpec under uniform architecture
  - §P6.10.4 Corner pinning through the projector
- §P6.11 Smart-pointer migration strategy
  - §P6.11.1 Migration targets and rationale
  - §P6.11.2 const vs non-const policy
  - §P6.11.3 What stays raw / by value
  - §P6.11.4 Cycle considerations
- §P6.12 Algorithmic changes summary
  - §P6.12.1 BuildFlatRowArrays
  - §P6.12.2 ComputeInvDiagSchur
  - §P6.12.3 ComputeCornerEssTDofs / ComputeCornerEssTDofsFromSpec
  - §P6.12.4 EmitConstraintTriples (HypreParMatrix path)
  - §P6.12.5 BuildLocalPairBlocks invariants
  - §P6.12.6 Cost analysis
- §P6.13 Tet p=2 — the Day 1 algorithmic target
  - §P6.13.1 What the mesh looks like
  - §P6.13.2 SetCurvature(2) for Neper-generated meshes
  - §P6.13.3 What stays the same at p=2
  - §P6.13.4 BBar interaction
  - §P6.13.5 PA / EA / FA assembly mode considerations
- §P6.14 Hex p=2 — deferred but architecturally ready
- §P6.15 Optional Barbosa-Hughes residual stabilization
  - §P6.15.1 The convergence rate question
  - §P6.15.2 Implementation sketch
  - §P6.15.3 When to enable
- §P6.16 Phasing: the Phase 6 batch sequence
  - §P6.16.1 Phase 6.0 — Foundational refactors
  - §P6.16.2 Phase 6.1 — Enable lor_depth = 2
  - §P6.16.3 Phase 6.2 — P2 tet polycrystal validation
  - §P6.16.4 Phase 6.3 — Hex p=2 (deferred)
  - §P6.16.5 Phase 6.4 — Barbosa-Hughes (deferred)
- §P6.17 Validation strategy
  - §P6.17.1 Regression at lor_depth = 1 (Phase 6.0 gate)
  - §P6.17.2 Linear-reproduction (Phase 6.0 / 6.1)
  - §P6.17.3 P2 tet polycrystal vs P1 reference (Phase 6.2)
  - §P6.17.4 Hill-Mandel diagnostic at p=2
  - §P6.17.5 Component-restricted PBC at p=2
  - §P6.17.6 Multi-rank A/B comparison
- §P6.18 Hazards and traps
  - §P6.18.1 The byNODES vs byVDIM trap, again
  - §P6.18.2 Snap-coordinate tolerance scaling
  - §P6.18.3 Cross-rank LOR-vs-parent partition mismatch
  - §P6.18.4 Corner-Dirichlet consistency
  - §P6.18.5 BBar + p ≥ 2 rejection
  - §P6.18.6 LOR refinement of curved surfaces
  - §P6.18.7 essential_vel_grad projection at p ≥ 2
  - §P6.18.8 ParSubMesh::UniformRefinement aliasing
  - §P6.18.9 ConstraintRHS sizing under projector
  - §P6.18.10 Per-row reference geometric factors under LOR
  - §P6.18.11 The classifier's `Fes()` accessor change
  - §P6.18.12 Test infrastructure shifts
  - §P6.18.13 The classifier's `GtdofOwnerRank` under shared-ptr migration
  - §P6.18.14 Refresh cascade through smart pointers
- §P6.19 Open questions and forward plan
  - §P6.19.1 Should `SurfaceProjector` ever be reused outside mortar PBC?
  - §P6.19.2 Should the boundary submesh FES live in `SimulationState` even when mortar PBC is disabled?
  - §P6.19.3 Should the LOR refinement depth be per-boundary-attribute?
  - §P6.19.4 What about $p \ge 3$?
  - §P6.19.5 What happens at curved boundaries beyond axis-aligned?
  - §P6.19.6 Forward plan summary
- §P6.20 Cross-references to other planning docs
  - §P6.20.1 Architecture doc references
  - §P6.20.2 Phase 4 plan references
  - §P6.20.3 Phase 5 plan (v7) references
  - §P6.20.4 v1 references (superseded)
- §P6.21 Done criteria

---

## §P6.1 Goals and non-goals

### §P6.1.1 Goals

1. **Support $p \ge 2$ primal fields in mortar PBC**, with the
   constraint enforced through a low-order auxiliary space on a
   refined boundary `ParSubMesh`. The volume problem stays at high
   order; the constraint problem reduces to a Q1 / P1 problem on the
   refined boundary mesh. Theoretical foundation: architecture doc
   §4.11.

2. **Unify the direct ($p = 1$) and LOR ($p \ge 2$) code paths.**
   The pre-Phase-6 implementation had the classifier internally
   extract a boundary `ParSubMesh` and use `GetParentVertexIDMap` to
   back-translate to parent-FES TDOFs. Phase 6 v2 retires that
   per-vertex back-translation and replaces it with a uniform
   projector-based TDOF translation that works the same way at
   $p = 1$ and at $p \ge 2$. There is exactly one constraint-operator
   code path at runtime.

3. **Tetrahedral $p = 2$ as the Day 1 target.** Crystal plasticity
   tet meshes routinely benefit from quadratic elements (mid-edge
   nodes capture bending modes of slip-dominated deformation that
   linear tets struggle with); Neper-generated polycrystal meshes
   are most accurate at $p = 2$. See §P6.2 for full motivation.
   Hex $p \ge 2$ is feature-complete in theory but not a Day 1
   priority (§P6.14).

4. **Reuse the existing Phase 4 / Phase 5 stack verbatim at runtime.**
   The classifier's wirebasket decomposition, the constraint builder,
   the EA operator's flat-array matvec, the saddle-point solver, the
   Wohlmuth corner / edge modifications, the Phase 5.9
   component-restricted PBC filter machinery (per §P5.18 spec), and
   the Phase 5.11 residual-scaling wrappers (per §P5.19) — all of
   these continue to operate without per-class modification beyond
   the projector-aware TDOF translation at construction time.

5. **Tribol-aligned design.** The LOR-on-`ParSubMesh` approach is
   exactly what Tribol uses for high-order contact in MFEM
   [Chin, MFEM Workshop 2023]. Re-using their established mechanics
   keeps Phase 6 idiomatic to the LLNL/MFEM ecosystem.

6. **`SimulationState`-centric mesh and FES ownership.** Following
   the existing pattern of `GetBoundarySubMesh()`, the LOR boundary
   submesh and its FES (and the un-refined submesh's FES) become
   methods on `SimulationState`. Future consumers — visualization,
   post-processing, full-mesh LOR for solver preconditioning, FE²
   coupling — share the same infrastructure.

7. **Smart-pointer ownership semantics.** Components that previously
   held references or raw pointers (the classifier inside the
   constraint operator, the operator inside the saddle system, etc.)
   migrate to `std::shared_ptr<T>` / `std::shared_ptr<const T>`.
   Ownership becomes explicit at every interface.

8. **Optional Barbosa-Hughes residual stabilisation** for optimal
   $H^1$ rates at higher order, deferred to a follow-on phase (§P6.15).

### §P6.1.2 Non-goals (explicitly deferred)

- **Hex $p \ge 2$ at Day 1.** Phase 6.0-6.2 deliver tet $p = 2$;
  hex $p = 2$ comes in Phase 6.3 as a small additive extension. The
  hex case is structurally easier than tet (tensor-product LOR is
  simpler than barycentric subdivision), so the deferral is purely
  about prioritization.
- **$p \ge 3$.** Possible by deeper LOR refinement
  (`refine_levels = p - 1` for tet, $2^{p-1}$ levels for hex), but
  no current ExaConstit use case demands $p > 2$. The TOML
  validation in Phase 6.0 limits `lor_depth ∈ {1, 2}`; lifting this
  is a one-line change when needed, but the projector's matching
  algorithm also needs extension for higher-order edge interior
  Lagrange nodes (§P6.19.4).
- **The full Popp-Wohlmuth-Gee-Wall higher-order dual basis**
  (architecture doc §4.10). This is the *theoretically* optimal
  higher-order mortar but requires per-element-type
  basis-transformation derivations that we do not undertake. See
  §P6.3.4 for the trade-off argument.
- **Mixed-order meshes** (some elements at $p = 1$, some at $p = 2$).
  ExaConstit does not support these in production; not a Phase 6
  concern.
- **GPU-resident projector construction.** The projector is
  constructed once at simulation startup; its matching algorithm
  is pure host code. Phase 6 does not GPU-port the projector setup.
  The *runtime* matvec is GPU-portable (the projector is consulted
  only at construction; runtime flat-array indices are already
  device-resident).
- **`SurfaceProjector::Mult` performance optimization beyond
  correctness.** The projector's runtime use is for inv-diag
  projection in `ComputeInvDiagSchur` and for diagnostic
  computations; it is not on the inner Krylov loop. Setup-style
  performance is fine.
- **Curved boundary surfaces.** Day 1 targets axis-aligned RVE
  geometries (flat faces). Uniform refinement of a flat-face
  submesh is exact at the $P_p$ Lagrange nodes; curved surfaces
  introduce a curvilinear-vs-Lagrange-node distinction that is out
  of scope for Phase 6. See architecture doc §13.3 for the
  long-term path (likely via Tribol).
- **Non-conforming face matching at higher order.** Conforming face
  matching is the Phase 6 scope. Non-conforming + higher-order is
  the combination of Phase 4.4 (Sutherland-Hodgman polygon clipping,
  deferred per Phase 4 plan §P4.1) and Phase 6; both must ship
  before that combination is available. The Phase 6 architecture
  supports it without modification once Phase 4.4 lands.
- **BBar at $p \ge 2$.** The current BBar integrator's
  element-volume-averaging formulation does not generalize cleanly
  to higher-order primal. Phase 6 hard-rejects this combination at
  build time (§P6.13.4, §P6.18.5). Re-enabling would require a
  different BBar formulation (e.g., $L_2$ projection onto a
  piecewise-constant pressure space), which is its own workstream.

---

## §P6.2 Why higher order matters now (or soon)

ExaConstit's current production focus is $p = 1$ hex / linear tet for
crystal plasticity. The pressure to support $p \ge 2$ comes from two
directions:

**Tetrahedral meshes from Neper, the canonical polycrystal mesh
generator, are most accurate at $p = 2$ for crystal plasticity.** The
mid-edge nodes capture the bending and twisting modes that arise at
grain boundaries. Linear tets can produce stress concentrations that
are mesh-dependent in ways that quadratic tets resolve. For RVE
homogenization of polycrystals — ExaConstit's bread and butter — this
distinction shows up as effective tangent moduli that converge in
the wrong direction with mesh refinement for $p = 1$ tets, an artifact
that $p = 2$ tets clean up cleanly.

**Higher-fidelity ensembles for FE² and parameter optimization.**
Calibrating constitutive models against experimental data is sensitive
to micro-scale fidelity; quadratic tet RVEs reduce the mesh-dependence
noise floor that obscures the constitutive signal. As the
`workflows/optimization/` driver matures, the ability to drive higher-
order RVEs becomes a calibration-quality lever.

Hex $p = 2$ is rarely needed in practice: hex meshes already have
natural compatibility with tensor-product crystal structures, and for
axis-aligned RVE problems (which is most of what ExaConstit handles),
$p = 1$ hex captures the relevant deformation modes well enough. The
Day 1 priority is therefore tet $p = 2$; hex $p = 2$ is a small
follow-on (§P6.14).

---

## §P6.3 Theory: LOR projection on refined boundary submeshes

This section recapitulates the theoretical foundation from architecture
doc §4.11-§4.12 and adds the Phase 6 v2-specific framing. Readers
already familiar with the LOR theory can skip to §P6.4.

### §P6.3.1 The Pazner-Kolev coincidence property

For an order-$p$ primal field $u_h$ on a 3D mesh $\mathcal{T}_h$, consider
the boundary `ParSubMesh` $\partial \mathcal{T}_h$ — a 2D surface mesh
inheriting boundary-element geometry and attributes from the parent's
boundary faces. Apply uniform refinement $p - 1$ times to this submesh.
The resulting mesh $\partial \mathcal{T}_{h/p}$ has the following
geometric property [Pazner & Kolev 2021, Theorem 2.1]:

$$
\bigl\{ \text{Lagrange nodes of } P_p \text{ on } \partial \mathcal{T}_h \bigr\}
\;=\;
\bigl\{ \text{vertices of } \partial \mathcal{T}_{h/p} \bigr\}.
\qquad (P6.3.1)
$$

The set equality is pointwise (exact coordinate match for straight-
sided elements; match to machine epsilon for curvilinear elements).
For tet face elements at $p = 2$: the 6 Lagrange nodes (3 corners +
3 mid-edges) coincide *exactly* with the 6 vertices of the 4-sub-
triangle LOR refinement. For hex face elements at $p = 2$: the 9
Lagrange nodes (4 corners + 4 mid-edges + 1 centroid) coincide with
the 9 vertices of the 4-sub-quad LOR refinement.

**Consequence for the constraint problem.** Any continuous $P_p$
field $u_h$ on the boundary admits a unique continuous *piecewise-
linear* representation $u_h^{\text{LOR}}$ on the refined boundary
mesh, with **identical nodal values** —
$u_h(x_\alpha) = u_h^{\text{LOR}}(x_\alpha)$ for every Lagrange node
$x_\alpha$. The mapping is a trivial bijection of coefficient
vectors. **It is a permutation matrix, not an interpolation.** This
is the load-bearing fact that makes LOR-based mortar viable: no
information is lost in the parent-to-LOR direction (since the LOR
representation captures the full nodal information of $u_h$ at the
Lagrange nodes), and reconstruction is exact in the reverse
direction at the Lagrange nodes.

### §P6.3.2 The linear dual basis transports to LOR without modification

The mortar constraint on a periodic boundary pair $(\Gamma^-, \Gamma^+)$
in weak form is (architecture doc eq. 3.4):

$$
\langle \mu_i, [u_h \circ \Pi - u_h]\rangle_{\Gamma^-} \;=\; 0
\quad \forall \mu_i \in \Lambda_h,
\qquad (P6.3.2)
$$

where $\Pi$ is the periodic translation between the homologous faces
and $\Lambda_h$ is the Lagrange multiplier space on $\Gamma^-$.

For the linear case ($p = 1$ primal), $\Lambda_h$ is the standard
linear dual basis $M_i$ satisfying the bi-orthogonality property
(architecture doc eq. 4.1):

$$
\int_{E} M_i \, N_j \, ds \;=\; \delta_{ij} \int_E N_j \, ds,
\qquad (P6.3.3)
$$

with $N_j$ the standard nodal Lagrange basis on the surface element
$E$. Architecture doc §4.2 (line-2, 1D simplex, used for 2D mortar
problems), §4.3 (quad-4, 2D hypercube), and §4.4 (tri-3, 2D simplex)
derive explicit formulas for the dual basis on the reference elements
relevant to Phase 6 surface mortar. (§4.5 derives the tet-4 dual
basis for completeness; that is for *volume* mortar — e.g. multi-block
coupling — and is not used in the Phase 6 surface pipeline.) These
formulas live in `face_mortar_assembler_3d.{hpp,cpp}` and
`mortar_assembler_2d.{hpp,cpp}`.

**The LOR adaptation is purely geometric.** The LOR pipeline takes the
periodic boundary `ParSubMesh`, refines it $p - 1$ times, and runs the
Phase 4 mortar pipeline on the refined surface mesh **at order 1**.
The dual basis used is precisely the standard linear dual basis from
§4.2-§4.4 — no new derivations, no per-element-type adjustments. The
refined sub-elements (sub-triangles for tet faces, sub-quads for hex
faces) are themselves tri-3 / quad-4 elements; the dual basis is
defined on them directly.

### §P6.3.3 Stability and convergence; why we tolerate the p / 1 pairing

The non-trivial theoretical point is that pairing a $P_p$ displacement
field (in the volume) with a $P_1$ Lagrange multiplier (on the LOR
surface) is **not automatically inf-sup stable**
[Brivadis, Buffa, Wohlmuth & Wunderlich 2015, CMAME 284]: the unmodified
$p / (p-1)$ pairing exhibits cross-point oscillations and a non-uniform
inf-sup constant, leading to suboptimal saddle-point convergence rates:

$$
\| u - u_h \|_{H^1} \;\le\; C \cdot \varepsilon_{\text{primal}} + C \cdot \varepsilon_{\text{LM}}
\;\approx\; O(h^p) + O(h^{3/2})
\qquad (P6.3.4)
$$

at $p \ge 2$. The $H^1$ error degrades from the optimal $O(h^p)$ rate
to $O(h^{3/2})$ because of the LM space's coarser order.

Three remediations are available (architecture doc §4.11.3):

- **(R1)** Belgacem cross-point modification — restores inf-sup
  stability via vertex zero-out + barycentric redistribution at
  cross-points. Recovers the $p / (p-1)$ pairing.
- **(R2)** Use the $p / (p-2)$ pairing (e.g., $P_2 / P_0$ at $p = 2$).
  Provably stable but suboptimal in $\lambda$ approximation;
  generally unsuitable for elasticity due to volumetric locking
  concerns.
- **(R3)** Barbosa-Hughes residual stabilization — adds a
  stabilization term to the saddle-point system that recovers
  quasi-optimal convergence for any $L^2$-conforming LM space
  [Gustafsson, Råback & Videman 2022, CMAME 404].

**For RVE-PBC homogenization, the inf-sup degradation is benign for
the engineering quantities of interest.** Homogenized stress and
effective tangent moduli depend on the *integral* behavior of the
fluctuation field, not on its pointwise $H^1$ error. Numerical
experiments in the homogenization literature (cited in architecture
doc §4.11.3) report effective tangent moduli converging at the bulk
rate even with mismatched-order LM, provided the saddle point is
well-posed (i.e., cross-point modification is in place).

The Phase 6 v2 default is therefore **unstabilized LOR with cross-
point modifications**, with Barbosa-Hughes available as an opt-in for
academic convergence-rate studies (§P6.15). This matches Tribol's
design choice and produces the right answers for ExaConstit's
production use cases.

### §P6.3.4 Why LOR and not Popp-Wohlmuth-Gee-Wall

Architecture doc §4.10 describes the Popp-Wohlmuth-Gee-Wall basis-
transformation procedure [Popp et al. 2012, SISC 34], which constructs
strictly bi-orthogonal *higher-order* dual bases on tri-6, quad-8,
quad-9, hex-20, hex-27 reference elements. This is the
**theoretically optimal** higher-order mortar approach: full $O(h^p)$
$H^1$ rates, no stabilization needed, no cross-point modification
beyond the higher-order analogue of the Wohlmuth modification.

We do *not* take this route. The reasons:

1. **Per-element-type derivation cost.** Each element type (tri-6,
   quad-8, quad-9, hex-20, hex-27) requires its own basis-
   transformation matrix $A_e$ and its own boundary modifications
   (architecture doc eqs. 4.34-4.36). The derivations are
   non-trivial and the resulting code is element-type-specific in
   ways that the linear dual basis is not.
2. **No precedent in MFEM.** Tribol uses the LOR-on-`ParSubMesh`
   approach precisely because the per-element-type higher-order
   dual basis route is too expensive for the LLNL ecosystem. We
   align with this precedent.
3. **Engineering quantities of interest tolerate the $H^1$
   degradation.** Per §P6.3.3, the inf-sup issue is benign for
   homogenized stress / tangent moduli. We do not need optimal
   $H^1$ rates for our use cases.
4. **Future-compatibility.** If a use case eventually demands
   optimal $H^1$ rates, the basis-transformation route is
   additive — it can be implemented as a different
   `face_mortar_assembler` variant, swapped in via runtime flag,
   without touching the rest of the pipeline. The LOR machinery
   does not preclude it.

### §P6.3.5 The Hill-Mandel consistency check at higher order

Architecture doc §8 derives the Hill-Mandel theorem as the
consistency check for homogenization:

$$
\langle \sigma : \dot{\varepsilon} \rangle_\Omega \;=\; \overline{\sigma} : \overline{\dot{\varepsilon}}
\qquad (P6.3.5)
$$

This consistency must hold at $p \ge 2$ as well. The Hill-Mandel
diagnostic (§P5.10.3 in the Phase 5 plan; registered in
post-processing via Phase 5.8.C) is implemented in
`MortarPbcManager::ComputeHillMandelPowerBalance` and computes the
relative residual:

$$
\eta_{\text{HM}} \;=\; \frac{\bigl|\, \langle P : \dot{F} \rangle_\Omega - \overline{P} : \overline{\dot{F}} \,\bigr|}{\bigl|\, \overline{P} : \overline{\dot{F}} \,\bigr|}
\qquad (P6.3.6)
$$

and asserts $\eta_{\text{HM}} < 10^{-10}$ on validation runs.

**At $p \ge 2$, the volume integration in (P6.3.5) uses higher-order
quadrature** — the integration order in `SimulationState` is set as
`int_order = 2 * options.mesh.order + 1`, which is automatic. The
$\overline{P} : \overline{\dot{F}}$ side computes via volume averages,
which also use the higher-order quadrature. So Hill-Mandel should
hold with the same numerical tightness at $p = 2$ as at $p = 1$,
provided:

1. The volume quadrature is correctly higher-order (verified by
   ExaConstit's existing infrastructure).
2. The constraint $C v = g$ is enforced to Krylov tolerance (held
   by the saddle-point solver, projector-aware or not).
3. The boundary traction $\lambda$ contribution to the Hill-Mandel
   power balance is computed consistently with the LOR pipeline.
   This is the place where Phase 6 needs explicit verification —
   the lambda contribution is computed in
   `MortarPbcManager::ComputeHillMandelPowerBalance` from the
   accumulated $\lambda$ and the boundary geometry; under LOR,
   $\lambda$ lives on the refined submesh and the geometry is
   the refined submesh's. The accounting must be consistent.

The Hill-Mandel diagnostic is therefore a primary validation gate
for Phase 6 — see §P6.17.4.

---

## §P6.4 Architectural overview: the unified projector architecture

### §P6.4.1 The three architectural options considered

During the v1-to-v2 design review, three architectural options were
considered for supporting $p \ge 2$:

**Option α — Wrapper approach (v1's original design).** Build a new
`HighOrderMortarConstraintOperator` class that holds an underlying
$p = 1$ `MortarConstraintOperator` (on the LOR submesh) and a
`SurfaceLORProjector`, composing them at every matvec.

Pros:
- Minimal change to existing $p = 1$ code.
- Direct-path performance untouched.
- Clear separation of concerns.

Cons:
- Two operator classes maintained in parallel; bug fixes in matvec
  logic, off-rank topology, filter spec, GPU port must be
  propagated to both.
- Phase 5.9 component-restricted PBC filter machinery exists twice
  (once in the underlying operator, once forwarded by the wrapper).
- Phase 5.11 residual-scaling wrappers (§P5.19) need to be made aware
  of the wrapper class.
- The wrapper is a decorator with no independent state; the
  abstraction is real but its sole purpose is to factor a small
  index-translation step.

**Option β — Direct path with fast path; LOR path with wrapper.** Keep
the existing direct-path `MortarConstraintOperator` exactly as is;
add a separate `HighOrderMortarConstraintOperator` only when
`lor_depth > 1`. The manager branches on `lor_depth` and constructs
one of the two operators.

Pros:
- Direct-path zero-cost guarantee preserved.
- The split is at a single point (manager construction).

Cons:
- Two operator classes maintained in parallel.
- The Phase 5.9 filter machinery still exists twice.
- The runtime saddle system and scaling layer must be polymorphic
  over the two operator types (either via virtual dispatch or
  via separate code paths).

**Option γ — Unified projector architecture (v2's choice).** Refactor
`MortarConstraintOperator` to consume a `SurfaceProjector` at
construction. The classifier always operates on a boundary submesh
and a FES on that submesh. The projector translates submesh-FES
gtdofs to parent-FES gtdofs *once* at construction (and at every
`Reset` under Phase 5.9 filter changes). The runtime matvec uses
flat arrays indexed in parent-FES TDOFs — identical to the pre-
Phase-6 implementation. There is one operator class.

At $p = 1$ (`lor_depth = 1`), the projector is a degenerate
permutation between the un-refined boundary submesh's FES and the
parent FES's boundary Lagrange nodes. The flat-array contents are
*structurally identical* to what the pre-Phase-6 implementation
produced (they store parent-FES local TDOF indices), so the runtime
matvec is bit-equivalent.

At $p = 2$ (`lor_depth = 2`), the projector additionally maps
mid-edge / mid-face nodes on the refined submesh to mid-edge /
centroid Lagrange nodes on the parent FES. The flat-array contents
are larger (more rows = more LOR vertices) but the runtime structure
is the same.

Pros:
- One operator class. All Phase 4 / Phase 5 machinery extends
  uniformly to $p \ge 2$ with no per-mode branching.
- Phase 5.9 filter, Phase 5.11 residual scaling (§P5.19), GPU
  matvec — all of these are independent of $p$.
- Future LOR-for-other-purposes (visualization, full-mesh LOR
  preconditioning) reuses the same `SurfaceProjector` infrastructure.
- The classifier loses its "I know about the parent ParMesh" coupling,
  becoming a pure boundary-submesh classifier — the option B
  refactor from Design 2 lands here naturally.

Cons:
- The refactor touches more code than option β: the classifier's
  TDOF accessors change, the constraint builder's
  `EmitConstraintTriples` changes, the operator's `BuildFlatRowArrays`
  changes, the corner-pinning helper changes. All in lockstep, in
  Phase 6.0.
- Direct-path matvec performance at runtime is bit-equivalent (no
  cost), but construction cost incurs one additional Alltoallv (for
  the projector's cross-rank topology) at simulation startup. This
  is dominated by K assembly in practice.
- The classifier's existing constructor (taking parent ParMesh +
  parent FES) is retired from production code. A free function
  `MakeBoundaryClassifierFromParent` is provided for test
  convenience (§P6.6.5).

### §P6.4.2 Why option γ wins

The maintainability calculus, expanded:

1. **Bug fixes propagate once.** Every modification to the matvec
   path, off-rank topology, filter spec, or GPU kernel applies to
   *the* operator, not to a primary and a wrapper. Phase 4.3.B's
   ongoing GPU port for `MultTranspose` does not need a parallel
   wrapper-side implementation. Phase 5.9's filter machinery lives
   in one place. The Phase 5.11 residual-scaling stack (§P5.19)
   remains agnostic.

2. **The runtime cost is zero at $p = 1$.** The flat-array contents
   are unchanged in structure; the matvec walks them identically.
   The "extra Alltoallv at construction" cost is paid once and is
   bounded by the size of the projector's import topology, which is
   on the order of (boundary surface area) / (number of ranks). For
   a 100³ RVE on 1000 ranks, this is on the order of 100 doubles per
   rank per setup — negligible.

3. **The runtime cost at $p \ge 2$ is the same as the matvec at
   $p = 1$, plus the additional rows from LOR refinement.** No
   wrapper overhead, no per-iteration projector composition. The
   GPU port (Phase 4.3.B and beyond) extends to $p \ge 2$ for free.

4. **The classifier becomes more self-contained.** It no longer
   reaches into the parent ParMesh for `GetParentVertexIDMap`-based
   back-translation. It operates on its inputs (submesh + FES on
   submesh) without external coupling. This is the option B
   commitment from Design 2 — biting the bullet now in exchange for
   a long-term cleaner abstraction.

5. **Future use cases are absorbed.** A LOR-of-the-full-mesh for
   visualization or preconditioning would use the same
   `SurfaceProjector` mechanism (extended to a `VolumeProjector` if
   needed). The architectural primitive is reusable.

### §P6.4.3 The runtime data flow

The data flow at runtime, in option γ:

```
+---------------------------------------------------------+
|  MortarConstraintOperator::Mult(u_parent, lambda)      |
|                                                         |
|  u_parent : parent-FES TDOF vector (Width = Pp boundary)|
|  lambda   : LM vector (Height = LOR rows)              |
|                                                         |
|  for each row i in m_n_active_rows:                    |
|      g_n_loc = m_row_g_n_local[i]   // parent-FES local |
|      lambda[m_row_lambda_off[i]+c] =                   |
|          D[i] * u_parent[g_n_loc]                      |
|          - sum_e (A_e * u_m)        // u_m from local  |
|                                     //  or off-rank    |
+---------------------------------------------------------+
```

The flat arrays `m_row_g_n_local`, `m_csr_g_m_local`, `m_csr_g_m_recv`
hold **parent-FES local TDOF indices** (or -1 for off-rank / sentinel
slots). The projector does not appear at runtime in `Mult` or
`MultTranspose`. The projector is consulted only at:

1. `BuildFlatRowArrays` (construction time and `Reset` time): to
   translate classifier-side (submesh-FES) gtdofs to parent-FES
   local TDOF indices for the flat arrays.
2. `ComputeInvDiagSchur`: to project `diag(K)^{-1}` from parent-FES
   space to submesh-FES space before computing the Schur diagonal.
3. Corner-pinning helpers in `MortarPbcManager`: to translate the
   classifier's corner records (submesh-FES) to parent-FES local
   TDOFs.

### §P6.4.4 Component responsibility matrix

| Component | Owns | References (shared_ptr) | Phase 6 v2 changes |
|---|---|---|---|
| `SimulationState` | parent `ParMesh`, parent FES, un-refined boundary `ParSubMesh`, un-refined submesh's FES, LOR `ParSubMesh`, LOR submesh's FES | (none) | Add `GetBoundarySubMeshFes`, `GetLorBoundarySubMesh`, `GetLorBoundarySubMeshFes` |
| `BoundaryClassifier3D` | wirebasket topology, per-pair blocks, boundary subcomm | submesh, submesh's FES | Drop the parent-ParMesh constructor (move to free function for tests); accept shared_ptr inputs; algorithm refactored to read TDOFs from submesh-FES directly |
| `SurfaceProjector` | TDOF translation table, cross-rank topology | parent FES, submesh FES, submesh | New class |
| `ConstraintBuilder3D` | row partition logic, COO accumulation buffers | classifier, projector | `Emit*` methods consume a projector for parent-FES translation |
| `MortarConstraintOperator` | flat arrays, off-rank topology, filter spec | classifier, projector, parent FES | Construction takes projector; flat-array build does projector-mediated translation; matvec unchanged at runtime |
| `MortarSaddlePointSystem` | block offsets, K closures | constraint operator | No semantic change; smart-pointer migration |
| `MortarPbcManager` | classifier, projector, operator, saddle system, scaler, lambda accumulator, corner ess TDOFs | sim state | Single classifier (no parent + LOR split); projector always constructed (degenerate at `lor_depth = 1`, non-trivial at `lor_depth > 1`); corner-pinning helpers use projector |

The reference graph stays a DAG; no cycles arise from the
shared_ptr migration.

---

## §P6.5 SimulationState as the home for shared mesh infrastructure

### §P6.5.1 Existing infrastructure (recap)

`SimulationState` already owns and exposes:

```cpp
std::shared_ptr<mfem::ParMesh>               GetMesh();
std::shared_ptr<mfem::ParSubMesh>            GetBoundarySubMesh();   // lazy
std::shared_ptr<mfem::ParFiniteElementSpace> GetMeshParFiniteElementSpace();
```

`GetBoundarySubMesh()` is lazily built on first call via
`ParSubMesh::CreateFromBoundary(*m_mesh, bdr_attrs)` where
`bdr_attrs = m_mesh->bdr_attributes` (the full attribute set). The
result is cached as `m_bdr_submesh` and shared with consumers via
`shared_ptr`. The author's existing comment foreshadows Phase 6:

> "We might eventually need to make this a map or have a LOR version
> if we decide to map our quadrature function data from a HOR set to
> a LOR version to make visualizations easier..."

Phase 6 v2 honors this foresight. The new accessors follow the same
lazy-construction-with-shared_ptr-caching pattern.

### §P6.5.2 New accessors

```cpp
class SimulationState
{
public:
    // ... existing methods ...

    /// FES on the un-refined boundary `ParSubMesh`.
    /// H1, vdim = 3, order = 1, byNODES. Lazily built; cached.
    /// This is the FES the Phase 6 classifier uses in the direct
    /// (lor_depth = 1) path.
    std::shared_ptr<mfem::ParFiniteElementSpace> GetBoundarySubMeshFes();

    /// LOR-refined boundary `ParSubMesh`.
    /// For lor_depth = 1, aliases GetBoundarySubMesh() directly.
    /// For lor_depth > 1, constructs a fresh boundary submesh
    /// (NOT the one held by m_bdr_submesh — UniformRefinement is
    /// in-place, so aliasing would corrupt the un-refined consumer)
    /// and applies UniformRefinement lor_depth - 1 times.
    std::shared_ptr<mfem::ParSubMesh> GetLorBoundarySubMesh();

    /// FES on the LOR submesh. H1, vdim = 3, order = 1, byNODES.
    /// Aliases GetBoundarySubMeshFes() when lor_depth = 1.
    std::shared_ptr<mfem::ParFiniteElementSpace> GetLorBoundarySubMeshFes();

private:
    std::shared_ptr<mfem::ParFiniteElementSpace> m_bdr_submesh_fes;
    std::shared_ptr<mfem::ParSubMesh>            m_lor_bdr_submesh;
    std::shared_ptr<mfem::ParFiniteElementSpace> m_lor_bdr_submesh_fes;
};
```

### §P6.5.3 The aliasing pattern at lor_depth = 1

At `lor_depth = 1`:

- `GetLorBoundarySubMesh()` returns the same `shared_ptr` as
  `GetBoundarySubMesh()`.
- `GetLorBoundarySubMeshFes()` returns the same `shared_ptr` as
  `GetBoundarySubMeshFes()`.

The two pairs of accessors are deliberately separate API surfaces so
that downstream code can ask for "the LOR mesh and FES" without
branching on `lor_depth`. At depth 1 it gets the un-refined ones; at
depth 2 it gets the refined ones. The manager construction code
becomes branch-free at the FES query.

**The aliasing has a side-effect that requires discipline.** Any code
calling `UniformRefinement` on a cached submesh would corrupt all
aliased consumers. The convention is therefore: **cached submeshes
returned by `SimulationState` are immutable from the consumer's
perspective.** This is documented in the docstrings; a future
refactor could enforce it at the type level via `shared_ptr<const T>`,
but that requires `SimulationState` itself to use mutable handles
internally during setup, which is a separate cleanup.

The defensive alternative — having `GetLorBoundarySubMesh()` always
return a fresh extraction at depth 1 — costs a collective and some
memory per call without buying real safety (consumers should not be
mutating cached objects regardless). We commit to the aliasing.

### §P6.5.4 Implementation sketch

```cpp
std::shared_ptr<mfem::ParSubMesh>
SimulationState::GetLorBoundarySubMesh()
{
    if (m_lor_bdr_submesh) { return m_lor_bdr_submesh; }

    const int depth = m_options.mesh.lor_depth;
    if (depth <= 1) {
        m_lor_bdr_submesh = GetBoundarySubMesh();  // alias
        return m_lor_bdr_submesh;
    }

    // Build a *fresh* extraction so we don't mutate the cached
    // un-refined submesh held by m_bdr_submesh. UniformRefinement
    // is in-place; aliasing here would corrupt all m_bdr_submesh
    // consumers.
    mfem::Array<int> bdr_attrs(m_mesh->bdr_attributes);
    m_lor_bdr_submesh = std::make_shared<mfem::ParSubMesh>(
        mfem::ParSubMesh::CreateFromBoundary(*m_mesh, bdr_attrs));

    for (int r = 0; r < depth - 1; ++r) {
        m_lor_bdr_submesh->UniformRefinement();
    }
    return m_lor_bdr_submesh;
}
```

The FEC is cached in `m_map_fec` following the existing pattern
(the parent FEC is similarly cached in the existing constructor;
see `SimulationState::SimulationState`).

```cpp
std::shared_ptr<mfem::ParFiniteElementSpace>
SimulationState::GetLorBoundarySubMeshFes()
{
    if (m_lor_bdr_submesh_fes) { return m_lor_bdr_submesh_fes; }

    const int depth = m_options.mesh.lor_depth;
    if (depth <= 1) {
        m_lor_bdr_submesh_fes = GetBoundarySubMeshFes();  // alias
        return m_lor_bdr_submesh_fes;
    }

    auto lor_submesh = GetLorBoundarySubMesh();
    const int space_dim = lor_submesh->SpaceDimension();
    const std::string fec_key =
        "H1_" + std::to_string(space_dim) + "D_P1";
    if (m_map_fec.find(fec_key) == m_map_fec.end()) {
        m_map_fec[fec_key] = std::make_shared<mfem::H1_FECollection>(
            /*order=*/1, /*dim=*/space_dim);
    }
    m_lor_bdr_submesh_fes =
        std::make_shared<mfem::ParFiniteElementSpace>(
            lor_submesh.get(),
            m_map_fec[fec_key].get(),
            /*vdim=*/3,
            mfem::Ordering::byNODES);
    return m_lor_bdr_submesh_fes;
}
```

`GetBoundarySubMeshFes()` follows the same pattern but uses the
un-refined `GetBoundarySubMesh()` as its mesh and lives at the
direct-path equivalent.

### §P6.5.5 Gotchas

**(G1) `UniformRefinement` aliasing.** Per §P6.5.3 — never call
`UniformRefinement` on a `SimulationState`-cached submesh. The
convention is enforced by documentation, not by `const`-ness (since
`UniformRefinement` is a non-const method on `Mesh`). A future
refactor could expose only `shared_ptr<const ParSubMesh>` to enforce
this at the type level; we defer that change because internal
`SimulationState` code may need to mutate the cached submeshes
during initialization, and the cleaner fix is a separate
"make `SimulationState` own const-views" effort.

**(G2) MPI partition mismatch between submesh and parent.** MFEM's
`ParSubMesh::CreateFromBoundary` may produce a submesh whose MPI
partition does not match the parent ParMesh's partition. This is
the standard MFEM behavior — submesh partitioning is driven by
which parent boundary elements are owned by which rank, which need
not align with the parent's partition of its volume elements.
Consequences:

- The submesh's FES has TDOF offsets independent of the parent
  FES's TDOF offsets.
- A parent-FES TDOF on rank $r$ may correspond to a submesh-FES
  TDOF on rank $r' \ne r$.
- The `SurfaceProjector` handles this via Alltoallv (§P6.7.5).

This is not new in Phase 6 — the existing Phase 4 / Phase 5
constraint operator already handles cross-rank TDOF references via
its off-rank import/export topology. The projector layers another
similar topology on top.

**(G3) Memory pressure at high `lor_depth`.** At `lor_depth = 2` on
tet, each face element produces 4 sub-faces; the total LOR submesh
vertex count is approximately 4× the un-refined one. For a 100³
polycrystal RVE with ~60k boundary faces, the LOR submesh has
~240k faces and ~360k vertices, which translates to ~1M TDOFs at
vdim = 3. This is comparable to the volume FES TDOF count for the
same mesh, so the constraint problem becomes a meaningful memory
contributor. At `lor_depth = 3` (would support $p = 3$) the LOR
submesh would have 9× the elements of the un-refined one; we limit
`lor_depth ∈ {1, 2}` in Phase 6.

**(G4) The FEC reuse across FES instances.** `H1_FECollection` is
allocated once per (dim, order) combination and shared across all
FES that use the same FE family. The existing pattern in
`SimulationState` uses `m_map_fec` for this. The new accessors
honor the same pattern. The parent FES uses an order-$p$ FEC; the
LOR FES uses an order-1 FEC; these are separate entries in
`m_map_fec`.

**(G5) `bdr_attributes` propagation through `CreateFromBoundary`.**
A `ParSubMesh` created from boundary inherits the parent boundary
elements' `bdr_attribute` values as its *element* attributes (not
its `bdr_attributes`). This is the standard MFEM semantic and is
what the classifier's `DiscoverFaceLabelByAttr` expects. The LOR
refinement preserves these element attributes through sub-element
creation (verified at Phase 6.0.A unit test).

---

## §P6.6 BoundaryClassifier3D refactor

### §P6.6.1 What changes and why

Pre-Phase-6 v2, the classifier:

1. Took a parent `ParMesh&` and a parent `ParFiniteElementSpace&`
   (which is the volume FES, vdim = 3, defined on the parent ParMesh).
2. Internally built a boundary `ParSubMesh` via
   `ParSubMesh::CreateFromBoundary`.
3. Used `ParSubMesh::GetParentVertexIDMap` to translate submesh-
   vertex IDs back to parent ParMesh vertex IDs.
4. Queried the parent FES with the parent vertex IDs to obtain
   per-vertex parent-FES TDOFs (the gtdofs that the classifier's
   `CornerInfo3D`, `EdgeInfo3D`, `FaceInfo3D` records expose).

Phase 6 v2 changes this in two coordinated ways:

**(A) The classifier accepts a pre-built `ParSubMesh` and a FES on
that submesh.** It no longer extracts a submesh from a parent or
references a parent FES. This is the option B refactor from Design 2.

**(B) The classifier reads TDOFs from the submesh-FES directly.** It
no longer uses `GetParentVertexIDMap`. A submesh vertex's gtdofs come
from the submesh-FES's `GetVertexVDofs` walk.

The motivation for both:

- The classifier becomes a pure "given a 2D boundary mesh + an FES on
  it, decompose it into corners / edges / faces" utility, with no
  hidden coupling to a parent ParMesh.
- The submesh-FES holds TDOFs that are sized to the surface DOF
  count, not to the volume DOF count. At $p = 1$ these are the same
  TDOFs as the parent FES would have on the boundary; at $p = 2$
  they include additional mid-edge / centroid TDOFs that the parent
  FES has on the boundary but that the un-refined submesh-FES does
  not.
- The translation from submesh-FES gtdofs to parent-FES gtdofs lives
  in `SurfaceProjector` (a dedicated class), not implicitly inside
  the classifier.

### §P6.6.2 The new constructor signature

```cpp
class BoundaryClassifier3D
{
public:
    /// Construct from a pre-built boundary submesh and an FES
    /// defined on it.
    /// Mesh requirements:
    ///   - 2D submesh embedded in 3D (result of CreateFromBoundary
    ///     on an axis-aligned 3D box RVE).
    ///   - Vertices form an axis-aligned bounding box.
    ///   - Six face-attribute groups, each corresponding to one
    ///     extreme of one axis.
    /// FES requirements:
    ///   - H1, vdim = 3, order = 1 (asserted), Ordering::byNODES.
    ///   - fes_on_submesh->GetParMesh() == bdr_submesh.get().
    /// MPI scope: collective on bdr_submesh->GetComm().
    BoundaryClassifier3D(
        std::shared_ptr<mfem::ParSubMesh>            bdr_submesh,
        std::shared_ptr<mfem::ParFiniteElementSpace> fes_on_submesh,
        double tol_rel           = 1e-9,
        double pair_match_tol_rel = 1e-9);
};
```

The existing parent-ParMesh-taking constructor is **retired from
production code**. A free function `MakeBoundaryClassifierFromParent`
(§P6.6.5) preserves test-side convenience.

### §P6.6.3 Internal algorithm: from parent-FES gtdofs to submesh-FES gtdofs

The classifier's existing algorithm has the following major steps
(from `boundary_classifier_3d.cpp`):

1. `DiscoverFaceLabelByAttr` — inspect boundary-element coordinates
   to map MFEM attribute → canonical face label.
2. `BuildBoundarySubmesh` — extract `ParSubMesh::CreateFromBoundary`.
3. `GatherBoundaryRecords` — per-rank vertex records, AllGather,
   dedup by snap-coord keys.
4. `ClassifyVertices` — corner / edge / face by attribute-set
   cardinality.
5. `BuildCorners`, `BuildEdges`, `BuildFaces` — populate the
   classifier's main accessors.
6. `BuildLocalPairBlocks` — per-rank pair block assembly via tile
   shuffle on the boundary subcomm.

Step 2 disappears in v2 (the submesh is provided). Steps 1, 3-6
remain but read TDOFs from the submesh-FES rather than from the
parent FES.

**Concrete change for step 3.** In the current code, a per-rank
vertex record carries the parent-FES gtdofs `(gtdof_x, gtdof_y,
gtdof_z)`. These are obtained as:

```cpp
// Pre-Phase-6 v2:
const int parent_vid = bdr_submesh->GetParentVertexIDMap()[submesh_vid];
mfem::Array<int> vdofs;
parent_fes.GetVertexVDofs(parent_vid, vdofs);
const int gtdof_x = parent_fes.GetLocalTDofNumber(vdofs[0]);  // or sentinel
```

Post-Phase-6 v2:

```cpp
// Phase 6 v2:
mfem::Array<int> vdofs;
fes_on_submesh.GetVertexVDofs(submesh_vid, vdofs);
const int gtdof_x = fes_on_submesh.GetLocalTDofNumber(vdofs[0]);
```

The query goes to the submesh-FES, not the parent FES. The returned
gtdofs are **submesh-FES gtdofs** — they index into the submesh-FES's
TDOF space, not the parent FES's. Downstream code (constraint
builder, operator, corner-pinning helper) is updated in lockstep to
recognize this (§P6.8, §P6.9, §P6.10.4).

**Note on the boundary subcomm.** The boundary subcomm
(`m_boundary_comm`) is created via
`MPI_Comm_split(submesh->GetComm(), has_local_boundary_elements,
my_rank, &m_boundary_comm)`. The submesh has elements that *are*
boundary elements (vs. parent ParMesh, whose elements are volume),
so `has_local_boundary_elements` is "does this rank own any submesh
elements". Functionally equivalent to the pre-v2 derivation; the
rank set is the same.

### §P6.6.4 Shared-pointer migration of members

The classifier currently has members like:

```cpp
mfem::ParMesh&                m_pmesh;        // reference, non-owning
mfem::ParFiniteElementSpace&  m_fes;          // reference, non-owning
std::unique_ptr<mfem::ParSubMesh> m_bdr_submesh;  // owned
```

Post-v2:

```cpp
std::shared_ptr<mfem::ParSubMesh>            m_bdr_submesh;
std::shared_ptr<mfem::ParFiniteElementSpace> m_fes;
```

The classifier:

- Does **not** mutate the submesh (after construction completes; the
  refinement happens in `SimulationState` before the submesh is
  passed in).
- Does **not** mutate the FES.

So both could be `shared_ptr<const T>`. However, per the user's
policy:

> "Make use of `shared_ptr<const T>` only where it makes sense and if
> there might be a possibility of an object needing to be mutated in
> the future then we should probably err on the side of caution and
> go with a non-const version for it."

The submesh is potentially future-mutated (adaptive remeshing) and
the FES is potentially future-rebuilt (adaptive refinement). We err
on non-const for both.

**Other classifier members** to consider:
- The boundary subcomm `MPI_Comm m_boundary_comm` stays raw — POD
  handle, managed via destructor RAII.
- `m_tile_partition` (unique_ptr) stays unique_ptr — single-owner
  semantics are correct.
- The corners / edges / faces dicts stay owned by value — they're
  small `std::map`s of concrete data, no sharing benefit.

### §P6.6.5 MakeBoundaryClassifierFromParent convenience helper

For test code that builds classifiers directly from parent meshes,
a free function in `boundary_classifier_3d.hpp`:

```cpp
namespace mortar_pbc {

/// Construct a BoundaryClassifier3D from a parent ParMesh and a
/// parent vdim=3 FES, for test convenience.
///
/// Performs the three setup steps the production code now does
/// inside SimulationState:
///   1. Extract a boundary ParSubMesh via CreateFromBoundary over
///      all boundary attributes.
///   2. Build an H1, order = 1, vdim = 3, byNODES FES on the submesh.
///   3. Construct the classifier from the submesh + FES.
///
/// Production code should use SimulationState::GetBoundarySubMesh()
/// + GetBoundarySubMeshFes() to obtain shared, cached instances.
/// This free function exists for unit and integration tests where
/// constructing a full SimulationState is overkill.
std::shared_ptr<BoundaryClassifier3D> MakeBoundaryClassifierFromParent(
    mfem::ParMesh& parent_pmesh,
    const mfem::ParFiniteElementSpace& parent_fes,
    double tol_rel = 1e-9);

}  // namespace mortar_pbc
```

Test code that previously did `BoundaryClassifier3D cl(pmesh, fes)`
now does `auto cl = MakeBoundaryClassifierFromParent(pmesh, fes)`.
The semantic remains "build a classifier on the boundary of this
parent mesh", just with explicit (and shared) submesh ownership.

### §P6.6.6 Gotchas

**(G1) The submesh's `bdr_attributes` vs. its elements' `attribute`.**
A `ParSubMesh` created from boundary has its *elements* (which were
the parent's boundary elements) carrying the parent's `bdr_attribute`
values. The submesh's own `bdr_attributes` (which would describe the
submesh's own boundary — its wirebasket) is a different array. The
classifier's `DiscoverFaceLabelByAttr` must use the submesh's
*element* `attribute` member, not its `bdr_attributes`.

**(G2) `GetVertexVDofs` semantics on a vector FES.** For a vdim = 3
FES with `Ordering::byNODES`, the three vdofs for a vertex are at
positions `node_idx`, `node_idx + n_nodes`, `node_idx + 2*n_nodes`.
The classifier must use `GetVertexVDofs` which abstracts over the
ordering. The existing code uses this; we preserve it.

**(G3) The FES on the LOR submesh must be order = 1.** The
classifier asserts this in the constructor:
`MFEM_VERIFY(fes->GetMaxElementOrder() == 1, ...)`. This is a
correctness requirement, not a stylistic preference — the Phase 4
dual basis derivations assume linear primal on the boundary;
violating this gives wrong constraints silently.

**(G4) The submesh's communicator must match the parent ParMesh's
communicator.** `ParSubMesh::CreateFromBoundary` produces a submesh
on the same `MPI_Comm` as the parent. The boundary subcomm is then
split off this comm. The free function
`MakeBoundaryClassifierFromParent` forecloses misuse; direct callers
of the constructor must respect the contract.

**(G5) The `GetParentVertexIDMap` removal.** Existing classifier code
that uses `GetParentVertexIDMap` for downstream consumers (e.g.,
post-processing that wants to back-map to parent vertices) is now
broken. Mitigation: the `SurfaceProjector` exposes the translation
table; consumers needing parent-side indices can ask the projector.
We audit all consumers of `GetParentVertexIDMap` in the Phase 6.0
work.

---

## §P6.7 The SurfaceProjector class

### §P6.7.1 Responsibility and lifetime

`SurfaceProjector` translates **submesh-FES TDOF indices to parent-FES
TDOF indices** (and vice versa). It provides:

- A **TDofMap** consulted at constraint-operator construction and at
  `Reset` time. The map gives, for each submesh-FES local TDOF, the
  corresponding parent-FES local TDOF (if owned by this rank) or an
  off-rank slot index into a per-matvec import buffer.
- **`Mult` and `MultTranspose`** operators in the standard
  `mfem::Operator` API, for code that wants to apply the projection
  as a linear operator (e.g., projecting `inv_diag_K` from parent-
  FES space to submesh-FES space, projecting fluctuation fields for
  visualization, etc.).

The projector holds (`shared_ptr`):
- The parent FES.
- The submesh FES.
- The submesh (for vertex-position lookups during matching).

Its lifetime is tied to the manager (it's a manager member). At
runtime, the constraint operator's `Mult` does **not** consult the
projector — the index translation is baked into the flat arrays at
construction.

### §P6.7.2 Mathematical formulation

The projector represents the discrete operator

$$
R : V_h^{(\text{parent})} \;\to\; V_h^{(\text{submesh})}
\qquad (P6.7.1)
$$

defined by **trace-and-evaluate**: for $u \in V_h^{(\text{parent})}$,
$(R u)$ is the function in $V_h^{(\text{submesh})}$ that agrees with
$u$'s trace at every submesh-FES Lagrange node.

By the Pazner-Kolev coincidence property (§P6.3.1), every submesh-FES
Lagrange node is also a parent-FES boundary Lagrange node. The
representation of $R$ in the Lagrange basis is therefore a
**permutation matrix** — its rows are the submesh-FES TDOF indices,
its columns are the parent-FES TDOF indices, and each row has exactly
one nonzero (= 1). The transpose $R^T$ is also a permutation.

At $p = 1$ (un-refined submesh), $R$ is the restriction-to-boundary
operator: a permutation that picks out the parent-FES boundary TDOFs.

At $p = 2$ (once-refined submesh), $R$ is the restriction-to-boundary
operator extended to mid-edge / centroid Lagrange nodes: still a
permutation, but with more rows.

### §P6.7.3 The TDofMap data structure

```cpp
class SurfaceProjector : public mfem::Operator
{
public:
    /// Per-submesh-FES-local-TDOF translation to parent-FES gtdof.
    struct TDofMap
    {
        /// Parent-FES gtdof for each submesh-FES local TDOF.
        /// Size: submesh_fes->GetTrueVSize(). Always positive.
        mfem::Array<int> submesh_to_parent_gtdof;

        /// Parent-FES local TDOF index when the gtdof is owned by
        /// this rank; -1 otherwise.
        mfem::Array<int> submesh_to_parent_local_or_minus1;

        /// For each off-rank parent gtdof referenced, the owning
        /// parent-FES rank.
        mfem::Array<int> off_rank_parent_owner_rank;

        /// Per-rank counts and displacements for the Alltoallv
        /// import (parent -> submesh) and export (submesh -> parent).
        std::vector<int> import_recv_counts;
        std::vector<int> import_displs;
        std::vector<int> export_send_counts;
        std::vector<int> export_displs;
    };

    const TDofMap& Map() const { return m_map; }

    // mfem::Operator interface...
};
```

The map provides:

- `submesh_to_parent_gtdof[i]` — for submesh-FES local TDOF `i`, the
  global parent-FES TDOF. Used to determine ownership.
- `submesh_to_parent_local_or_minus1[i]` — local parent-FES TDOF
  (when owned) or -1 (when off-rank). Used directly in flat arrays
  for the rank-local case.
- The Alltoallv exchange tables — used by `Mult` / `MultTranspose`
  at runtime, and by the constraint operator's `BuildFlatRowArrays`
  to size its own off-rank topology to be compatible with the
  projector's.

### §P6.7.4 Vertex matching algorithm

The matching algorithm operates on physical coordinates and snap
tolerance:

```
Input:
  parent_fes  — parent FES on parent ParMesh
  submesh_fes — submesh FES on (possibly LOR-refined) submesh
  submesh     — the submesh (provides vertex positions)
  snap_tol    — absolute tolerance for coordinate matching

Algorithm:
  1. Enumerate parent_fes's boundary Lagrange nodes.
     For each boundary face element of parent_pmesh:
       - Get its Pp Lagrange node positions (via
         FiniteElementSpace::GetElementVDofs and the FE's
         GetNodes interface).
       - For each Lagrange node, hash its snapped coordinates
         (key = round(coord_i / snap_tol)) into a global
         map: snap_coord_key -> parent_fes_gtdof[3].

  2. Enumerate submesh_fes's local TDOFs.
     For each local submesh TDOF i:
       - Get its physical coordinate.
       - Hash to snap_coord_key.
       - Look up parent_fes_gtdof[3] from the map.
       - Cross-rank: if the parent-FES gtdof is owned by a
         different rank, exchange ownership info via Alltoallv.

  3. Build the per-rank Alltoallv topology from the resolved
     ownership information.
```

The hash-based lookup avoids the naive
$O(N_{\text{LOR}} \times N_{\text{parent\_boundary}})$ cost; matching
is $O(N)$ total with a hash lookup constant.

**Cross-rank complications.** In step 1, the parent-FES boundary
Lagrange nodes are distributed across ranks. The hash needs to be
*global*: every rank must know which parent-FES gtdofs are at which
snap coordinates. We do this via a one-time AllGather of the
parent-FES boundary Lagrange node table, sized at
$O(\text{boundary\_DOF\_count})$. For a 100³ RVE with ~10⁴ boundary
DOFs, this is ~10⁴ doubles per rank — small.

### §P6.7.5 Cross-rank topology and Alltoallv

The projector's runtime `Mult` does:

```
Mult(x_parent_local, y_submesh_local):
  1. Pack send buffer of parent-FES values that off-rank submesh
     TDOFs need.
  2. MPI_Alltoallv exchange.
  3. For each submesh-FES local TDOF i with a local parent
     counterpart (map.submesh_to_parent_local_or_minus1[i] >= 0):
       y_submesh_local[i] = x_parent_local[map[i]]
  4. For each submesh-FES local TDOF i with an off-rank parent
     counterpart, read from the receive buffer.
```

**MultTranspose** reverses the direction:
```
MultTranspose(x_submesh_local, y_parent_local):
  1. Zero y_parent_local.
  2. For each submesh-FES local TDOF i with a local parent
     counterpart, ADD x_submesh_local[i] to y_parent_local[map[i]].
  3. For each off-rank counterpart, pack into an export buffer,
     Alltoallv, and on the receiving side ADD to y_parent_local.
```

### §P6.7.6 Mult and MultTranspose at runtime

The projector's `Mult` is consulted at runtime by:

- `MortarConstraintOperator::ComputeInvDiagSchur` — to project
  `inv_diag_K` (parent-FES sized) to submesh-FES space before
  computing the Schur diagonal.
- Post-processing code that projects parent-FES fields to surface
  visualizations.

`MultTranspose` is consulted at runtime by:

- Reverse-direction post-processing.
- Diagnostic code that wants to evaluate $C^T \lambda$ on the parent
  FES — although this can be done more directly through
  `MortarConstraintOperator::MultTranspose`, which already produces
  parent-FES output.

The runtime cost is one Alltoallv plus a permutation. For
`ComputeInvDiagSchur`, this is paid once per saddle-point
preconditioner setup (so on each Newton step at most, often less);
not on the inner Krylov loop.

### §P6.7.7 Gotchas

**(G1) Snap tolerance sensitivity.** The `snap_tol` default of 1e-10
is suitable for unit-cube RVEs with element sizes ≥ 1e-3. For meshes
with element sizes < snap_tol × 100, the snap-coord keys may collide
or fail to match. Mitigation: scale `snap_tol` by element size at
projector construction.

**(G2) Non-axis-aligned RVEs.** If the parent ParMesh is not axis-
aligned, the boundary Lagrange node positions still match the
submesh vertex positions (the geometric coincidence is intrinsic to
$P_p$ FE refinement), but the snap-coord keying may be sensitive to
rotation. Phase 6 v2 targets axis-aligned RVEs (Day 1).

**(G3) Higher-order Lagrange nodes at face interiors.** For hex
$p = 2$, parent-FES Lagrange nodes include face centroids. The
submesh-FES at the corresponding refined mesh has vertex-DOFs at
those centroid positions. **For tet $p = 2$, there are no face
centroid Lagrange nodes** (tri-6 has 6 nodes). The algorithm handles
both uniformly.

**(G4) The parent-FES gtdofs in the snap-coord hash must be unique.**
A parent-FES gtdof is owned by exactly one rank (MFEM contract); the
global hash dedup must respect this. Phase 6.0 validation includes
a check that the hash size equals the parent-FES boundary gtdof
count post-dedup.

---

## §P6.8 MortarConstraintOperator under the projector

### §P6.8.1 Constructor and member layout

```cpp
class MortarConstraintOperator : public mfem::Operator
{
public:
    /**
     * @brief Construct with classifier, projector, and parent FES.
     *
     * @param classifier  Fully-built `BoundaryClassifier3D` on the
     *                    submesh side.
     * @param projector   `SurfaceProjector` translating
     *                    submesh-FES gtdofs to parent-FES gtdofs.
     * @param parent_fes  Parent FES — the FES the constraint operator
     *                    operates against at runtime (its `Width()`
     *                    equals `parent_fes->GetTrueVSize()`).
     */
    MortarConstraintOperator(
        std::shared_ptr<const BoundaryClassifier3D>     classifier,
        std::shared_ptr<const SurfaceProjector>         projector,
        std::shared_ptr<const mfem::ParFiniteElementSpace> parent_fes);

private:
    std::shared_ptr<const BoundaryClassifier3D>     m_classifier;
    std::shared_ptr<const SurfaceProjector>         m_projector;
    std::shared_ptr<const mfem::ParFiniteElementSpace> m_parent_fes;

    // Flat arrays (unchanged structure from Phase 4.3.B / Batch X):
    int m_n_active_rows;
    mfem::Array<int> m_row_lambda_off;
    mfem::Vector     m_row_D;
    mfem::Array<int> m_row_g_n_local;   // PARENT-FES local TDOFs
    mfem::Array<int> m_row_csr_off;
    mfem::Vector     m_csr_A;
    mfem::Array<int> m_csr_g_m_local;   // PARENT-FES local TDOFs
    mfem::Array<int> m_csr_g_m_recv;    // off-rank import slots

    // Off-rank import / export topology — same structure as before;
    // now keyed by PARENT-FES gtdofs, derived from the projector's
    // translation of classifier-side off-rank references.
    std::vector<int> m_import_off_rank_gtdofs;  // parent-FES gtdofs
    std::vector<int> m_export_local_gtdofs;     // parent-FES gtdofs

    // Phase 5.9 filter state (unchanged):
    std::vector<std::string> m_active_pair_labels;
    std::array<bool, 3>      m_comp_mask;
    int                      m_n_comps_active;
    std::array<int, 3>       m_local_c;

    void BuildFlatRowArrays();
    void BuildOffRankTopology();
};
```

The operator's `Width()` returns `parent_fes->GetTrueVSize()`. The
`Height()` returns `m_n_active_rows * m_n_comps_active`, as in
Phase 5.9.

### §P6.8.2 BuildFlatRowArrays with projector-mediated TDOF translation

The flat-row arrays are built in two passes (per the existing
implementation):

1. **Pass 1 — count active rows and total CSR entries.** Walks edge
   mortar groups and face mortar pairs, applies the filter, counts
   rows.
2. **Pass 2 — populate the flat arrays.** Walks the same data
   structures, fills `m_row_D`, `m_row_g_n_local`, `m_csr_A`, etc.

**The Phase 6 v2 change is in pass 2.** Where the existing code reads
classifier-side gtdofs (which were parent-FES gtdofs pre-refactor)
and converts directly to local parent-FES TDOFs:

```cpp
// Pre-Phase-6 v2:
const int classifier_gtdof = nonmortar_node.gtdof_x;  // parent-FES gtdof
const int parent_local_or_offrank_slot =
    ClassifyOwnership(classifier_gtdof);
m_row_g_n_local[i * 3 + 0] = parent_local_or_offrank_slot;
```

Post-Phase-6 v2:

```cpp
// Phase 6 v2:
const int classifier_gtdof = nonmortar_node.gtdof_x;  // SUBMESH-FES gtdof
const auto& projector_map = m_projector->Map();
const int parent_local_or_offrank_slot =
    projector_map.submesh_to_parent_local_or_minus1[classifier_gtdof];
m_row_g_n_local[i * 3 + 0] = parent_local_or_offrank_slot;
```

The translation is a O(1) lookup in the projector's pre-computed map.
Total cost of `BuildFlatRowArrays` is unchanged asymptotically; it
gains one indirection per row.

**Off-rank topology construction** parallels this: the operator's
off-rank import / export topology is sized from the *parent-FES*
gtdofs referenced by all pair blocks on this rank. Pre-v2, the
classifier emitted parent-FES gtdofs directly; post-v2, the
classifier emits submesh-FES gtdofs and the projector translates
them. The resulting topology is correct (same parent-FES gtdofs, just
arrived at via a different path).

### §P6.8.3 Width and Height semantics

- `Width()` is `m_parent_fes->GetTrueVSize()` — the parent FES's
  rank-local TDOF count. This is the dimensionality of the input
  vector to `Mult` and the output vector of `MultTranspose`.
- `Height()` is `m_n_active_rows * m_n_comps_active` — the rank-local
  count of active constraint rows times the count of active
  components per row (Phase 5.9 filter).

At $p = 1$ (`lor_depth = 1`): `Width()` = parent FES TDOFs (volume
FES); `Height()` = same as pre-v2.

At $p = 2$ (`lor_depth = 2`): `Width()` = parent FES TDOFs (volume
FES at $p = 2$); `Height()` = constraint rows on the LOR-refined
boundary (larger than $p = 1$).

### §P6.8.4 Mult and MultTranspose are unchanged at runtime

The runtime matvec walks the flat arrays exactly as in Phase 4.3.B:

```cpp
void MortarConstraintOperator::Mult(const Vector& x, Vector& y) const
{
    // x : parent FES TDOF vector
    // y : LM vector

    // ... off-rank import (Alltoallv on parent-FES gtdofs) ...

    mfem::forall(m_n_active_rows, [=] MFEM_HOST_DEVICE (int i) {
        const double D_kk = d_row_D[i];
        const int    csr_a = d_csr_off[i];
        const int    csr_b = d_csr_off[i + 1];
        const int    lam_off = d_lam_off[i];

        for (int c = 0; c < kVDim; ++c) {
            const int lr = local_c[c];
            if (lr < 0) { continue; }

            const int gn_loc = d_g_n_loc[i * kVDim + c];
            if (gn_loc < 0) { continue; }

            // d_x is parent-FES local TDOF vector — same as before.
            double y_c = D_kk * d_x[gn_loc];
            for (int e = csr_a; e < csr_b; ++e) {
                const int gm_loc  = d_g_m_loc [e * kVDim + c];
                const int gm_recv = d_g_m_recv[e * kVDim + c];
                double u_m;
                if (gm_loc >= 0)        { u_m = d_x[gm_loc]; }
                else if (gm_recv >= 0)  { u_m = d_recv[gm_recv]; }
                else                    { continue; }
                y_c -= d_csr_A[e] * u_m;
            }
            d_y[lam_off + lr] = y_c;
        }
    });
}
```

The kernel reads `d_x[gn_loc]` where `gn_loc` is a parent-FES local
TDOF index — exactly the same semantics as in Phase 5.9. The
projector does not appear at runtime.

This is the central reason option γ wins: GPU port, filter spec,
off-rank topology, scaling layer, preconditioner — **all unchanged**
because the runtime sees only parent-FES indices.

### §P6.8.5 ComputeInvDiagSchur with projector mediation

The Schur complement diagonal is:

$$
\text{diag}(C \, \text{diag}(K)^{-1} \, C^T)
\qquad (P6.8.1)
$$

where $C$ is the constraint operator and $K$ is the bulk stiffness.
The block-Jacobi preconditioner uses this to scale the LM block.

The existing API is:

```cpp
mfem::Vector ComputeInvDiagSchur(const mfem::Solver& K_jacobi_prec) const;
```

**Phase 6 v2 change.** Internally, the operator now:

```cpp
mfem::Vector
MortarConstraintOperator::ComputeInvDiagSchur(const mfem::Solver& K_jacobi_prec) const
{
    // Step 1: probe diag(K)^{-1} on the parent FES side.
    mfem::Vector ones_parent(Width());
    ones_parent = 1.0;
    mfem::Vector inv_diag_K_parent(Width());
    K_jacobi_prec.Mult(ones_parent, inv_diag_K_parent);

    // Step 2: project inv_diag_K from parent FES to submesh FES.
    mfem::Vector inv_diag_K_submesh(m_projector->Height());
    m_projector->Mult(inv_diag_K_parent, inv_diag_K_submesh);

    // Step 3: forward to the Vector-taking overload.
    return ComputeInvDiagSchur(inv_diag_K_submesh);
}

mfem::Vector
MortarConstraintOperator::ComputeInvDiagSchur(const mfem::Vector& inv_diag_K_submesh) const
{
    // ... per-pair-block walk over m_classifier.PairBlocks() ...
    //     using inv_diag_K_submesh indexed by classifier-side gtdofs.
}
```

At $p = 1$ (with the un-refined submesh), the projection step is a
trivial permutation. At $p = 2$, the projection picks out parent-FES
TDOFs at boundary-and-mid-edge positions for the LOR submesh's
vertices.

**Splitting into two overloads** is a small refactor (the Vector
version is the existing body, renamed; the Solver version probes then
forwards). It improves testability — unit tests can pass pre-projected
inv_diag without mocking a K-Jacobi preconditioner.

### §P6.8.6 Reset under the new architecture

`MortarConstraintOperator::Reset(active_pair_labels, comp_mask)` is the
Phase 5.9 entry point for spec changes. Internally:

```cpp
void MortarConstraintOperator::Reset(
    const std::vector<std::string>& active_pair_labels,
    const std::array<bool, 3>&      comp_mask)
{
    m_active_pair_labels = active_pair_labels;
    m_comp_mask          = comp_mask;
    m_n_comps_active     = CountActiveComps(m_comp_mask);
    m_local_c[0]         = LocalRowOfComp(m_comp_mask, 0);
    m_local_c[1]         = LocalRowOfComp(m_comp_mask, 1);
    m_local_c[2]         = LocalRowOfComp(m_comp_mask, 2);

    BuildFlatRowArrays();
    height = m_n_active_rows * m_n_comps_active;
}
```

The Phase 6 v2 change is internal: `BuildFlatRowArrays` consults the
projector for TDOF translation, as in §P6.8.2. The off-rank topology
is unchanged (sized at construction for the unfiltered union), per
Phase 5.9's existing semantics.

The projector itself is *not* rebuilt during `Reset` — its TDOF map
is filter-independent, depending only on the submesh-FES ↔ parent-FES
geometry.

### §P6.8.7 Bit-equivalence considerations

A critical Phase 6.0 validation requirement: at `lor_depth = 1`, the
operator's runtime matvec output must be bit-equivalent to the
pre-Phase-6 implementation.

This is *not* automatic. The flat-array contents are produced via a
different code path (classifier emits submesh-FES gtdofs, projector
translates to parent-FES local TDOFs, flat arrays populated). The
parent-FES local TDOF indices in the flat arrays *should* be the
same set of values, but the order in which they're written depends
on the classifier's iteration order, which in turn depends on the
submesh-FES TDOF numbering.

**Two outcomes are possible:**

(a) Same iteration order, identical flat arrays bit-for-bit. Matvec
output is bit-equivalent.

(b) Different iteration order, different flat array layouts, but
matvec output is still bit-equivalent because the row partition and
the per-row values are the same modulo reordering.

Either outcome is acceptable from a correctness perspective. The
Phase 6.0 validation tests (§P6.17.1) check matvec output, not flat
array layout.

If the test fails (matvec output differs), the failure modes are:
- The classifier's submesh-FES gtdof ordering produces a different
  row partition than the previous parent-FES gtdof ordering, and the
  per-row dual coefficients are not invariant under the reordering.
- The projector's translation has a bug (off-by-one, wrong dimension,
  etc.).

Debug strategy: compare row-by-row classifier emission against the
pre-Phase-6 emission, identifying the first row where they diverge.

---

## §P6.9 ConstraintBuilder3D parallel changes

The constraint builder (`ConstraintBuilder3D`) provides the
HypreParMatrix path for $C$, used for A/B validation against the EA
operator. It must be updated in lockstep with the operator (§P6.8).

### §P6.9.1 The HypreParMatrix path

The builder's main method, `BuildHypreParMatrix`, produces a sparse
$C$ as a `HypreParMatrix` indexed by **parent-FES gtdofs** in its
columns. The HypreParMatrix is then used in saddle-point solves
when the EA path is disabled (e.g., for debugging, or for users who
prefer the assembled-matrix workflow).

Internally, the builder walks the classifier's pair blocks and
emits constraint triples `(row, col, value)`. The `col` value is a
*parent-FES gtdof* (because that's what the HypreParMatrix is indexed
by).

Phase 6 v2: the classifier emits submesh-FES gtdofs; the builder
translates to parent-FES gtdofs via the projector. This parallels
the operator's translation in §P6.8.2.

### §P6.9.2 EmitConstraintTriples with projection

```cpp
class ConstraintBuilder3D
{
public:
    ConstraintBuilder3D(
        std::shared_ptr<const BoundaryClassifier3D> classifier,
        std::shared_ptr<const SurfaceProjector>     projector);

    /// Emit constraint triples for HypreParMatrix assembly.
    /// Triples (row, col, value) where col is a PARENT-FES gtdof
    /// (via projector translation).
    void EmitConstraintTriples(
        const std::vector<std::string>& active_pair_labels,
        const std::array<bool, 3>&      comp_mask,
        std::vector<int>&               rows,
        std::vector<int>&               cols,
        std::vector<double>&            values) const;

private:
    std::shared_ptr<const BoundaryClassifier3D> m_classifier;
    std::shared_ptr<const SurfaceProjector>     m_projector;
};
```

The projector is consulted only in `EmitConstraintTriples` (and
optionally in `EmitRowFactors`, §P6.9.3); the builder is otherwise
stateless.

### §P6.9.3 EmitRowFactors invariants

`EmitRowFactors` produces per-row reference geometric factors used
by `MortarPbcManager` for the constraint-RHS update (§P5.8.6 — the
right-hand side update under UL).
The factors are:
- Signed periodic shift vector per row (3 components).
- Component index per row (the c in 0/1/2).
- Wohlmuth lumped-row factor per row.

These are **geometric** quantities — they depend on corner positions,
edge directions, and the lumped-row factor (architecture doc §5).
They do **not** depend on the FES choice. The row ordering matches
the constraint emission order, so the indexing is consistent between
the operator's flat-array layout and the manager's per-row factor
buffers.

Phase 6 v2 change: minimal. The factors are emitted in the same order
as before; the projector is not consulted (since the factors don't
involve TDOF lookups, only geometry). The row count is whatever
`NumLocalRows(active_pair_labels, comp_mask)` returns, consistent
with the operator's `Height()`.

---

## §P6.10 MortarPbcManager simplification

### §P6.10.1 Single classifier

Pre-v2 (in v1's design): the manager would have built two classifiers
— a parent classifier (for corner pinning) and an LOR classifier
(for the constraint pipeline).

Post-v2 (option γ): the manager builds **one classifier**, on the
LOR submesh + LOR FES. At `lor_depth = 1`, this classifier is on the
un-refined submesh — same effect as a "parent classifier" would have
had, but expressed through the unified SimulationState accessors.

The classifier emits corner records in submesh-FES gtdofs. The
corner pinning helper (§P6.10.4) translates these to parent-FES
TDOFs via the projector.

### §P6.10.2 Construction sequence

```cpp
class MortarPbcManager
{
private:
    std::shared_ptr<SimulationState> m_sim_state;

    // The boundary classifier — on the LOR submesh + LOR FES
    // (at lor_depth = 1 these are the un-refined submesh + FES).
    std::shared_ptr<BoundaryClassifier3D> m_classifier;

    // The projector — translates submesh-FES gtdofs to parent-FES
    // gtdofs. At lor_depth = 1, a degenerate permutation.
    std::shared_ptr<SurfaceProjector> m_projector;

    // The constraint operator — receives classifier + projector.
    std::shared_ptr<MortarConstraintOperator> m_C_op;

    // The constraint builder — also receives classifier + projector
    // (for the HypreParMatrix path).
    std::shared_ptr<ConstraintBuilder3D> m_builder;

    // Rest unchanged from Phase 5:
    std::shared_ptr<SaddlePointSolver> m_saddle_solver;
    std::shared_ptr<MortarSaddlePointSystem> m_saddle_system;
    // ... scaler, lambda accumulator, corner ess TDOFs, ...
};
```

Construction:

```cpp
MortarPbcManager::MortarPbcManager(
    std::shared_ptr<SimulationState> sim_state,
    KResidualFn k_residual,
    KJacobianFn k_jacobian)
    : m_sim_state(std::move(sim_state))
{
    // Build classifier on LOR submesh + LOR FES.
    // (At lor_depth = 1, these alias the un-refined submesh + FES.)
    m_classifier = std::make_shared<BoundaryClassifier3D>(
        m_sim_state->GetLorBoundarySubMesh(),
        m_sim_state->GetLorBoundarySubMeshFes(),
        m_sim_state->GetOptions().mesh.snap_tol);

    // Build projector — translates classifier-side (submesh-FES)
    // gtdofs to parent-FES gtdofs.
    m_projector = std::make_shared<SurfaceProjector>(
        m_sim_state->GetMeshParFiniteElementSpace(),
        m_sim_state->GetLorBoundarySubMeshFes(),
        m_sim_state->GetLorBoundarySubMesh(),
        m_sim_state->GetOptions().mesh.snap_tol);

    // Build builder + operator with classifier + projector.
    m_builder = std::make_shared<ConstraintBuilder3D>(m_classifier, m_projector);
    m_C_op    = std::make_shared<MortarConstraintOperator>(
        m_classifier, m_projector,
        m_sim_state->GetMeshParFiniteElementSpace());

    // Saddle system / scaler / etc. take m_C_op via shared_ptr.
    // ...

    BuildCornerEssTDofs();
    BuildReferenceGeometricFactors();
}
```

There is no `if (lor_depth > 1)` branch in this sequence. The
SimulationState accessors return the right thing in both modes
(aliased at `lor_depth = 1`, refined at `lor_depth = 2`).

### §P6.10.3 RebuildForActiveSpec under uniform architecture

Phase 5.9's `RebuildForActiveSpec` orchestrates spec changes:

```cpp
void MortarPbcManager::RebuildForActiveSpec(
    const std::vector<int>& essential_ids,
    int essential_comps)
{
    const auto comp_mask = CompMaskFromInt(essential_comps);
    const auto active_pair_labels =
        ValidateAndDeriveActivePairLabels(*m_classifier, essential_ids);

    m_C_op->Reset(active_pair_labels, comp_mask);
    m_saddle_system->Refresh();

    // Phase 6 v2: pass projector for parent-FES translation.
    m_corner_ess_tdofs = ComputeCornerEssTDofsFromSpec(
        *m_classifier,
        *m_projector,                                          // NEW
        *m_sim_state->GetMeshParFiniteElementSpace(),
        essential_ids,
        comp_mask);

    m_lambda.SetSize(m_C_op->Height());
    m_g_rhs .SetSize(m_C_op->Height());
    m_lambda = 0.0;
    m_g_rhs  = 0.0;

    BuildReferenceGeometricFactors(active_pair_labels, comp_mask);
}
```

The only difference from Phase 5.9 is that
`ComputeCornerEssTDofsFromSpec` now takes a projector argument. The
spec-change mechanics are otherwise identical.

### §P6.10.4 Corner pinning through the projector

`ComputeCornerEssTDofsFromSpec` is the helper that builds the rank-
local corner-pinned TDOFs:

```cpp
mfem::Array<int> ComputeCornerEssTDofsFromSpec(
    const BoundaryClassifier3D&        classifier,
    const SurfaceProjector&            projector,            // NEW
    const mfem::ParFiniteElementSpace& parent_fes,
    const std::vector<int>&            essential_ids,
    const std::array<bool, 3>&         comp_mask)
{
    const int my_rank = classifier.Rank();
    const HYPRE_BigInt parent_offset = parent_fes.GetMyTDofOffset();

    // Step 1: anchor "blf" corner — 3 components unconditionally.
    mfem::Array<int> out = AnchorCornerTDofsFromSpec(
        classifier, projector, parent_fes);

    // Step 2: incident-face gate.
    std::set<std::string> incident_labels;
    for (int attr : essential_ids) {
        const auto labels = classifier.CornersOnFaceAttribute(attr);
        incident_labels.insert(labels.begin(), labels.end());
    }

    // Step 3: 7 non-anchor corners.
    const auto& projector_map = projector.Map();
    for (const auto& kv : classifier.Corners()) {
        const CornerInfo3D& c = kv.second;
        if (c.label == "blf") continue;
        if (incident_labels.find(c.label) == incident_labels.end()) continue;

        // c.gtdof_x, c.gtdof_y, c.gtdof_z are SUBMESH-FES gtdofs now.
        // Translate to PARENT-FES gtdofs via the projector.
        const std::array<int, 3> submesh_gtdofs = {
            c.gtdof_x, c.gtdof_y, c.gtdof_z};

        for (int comp = 0; comp < 3; ++comp) {
            if (!comp_mask[comp]) continue;

            const int parent_gtdof =
                projector_map.submesh_to_parent_gtdof[submesh_gtdofs[comp]];

            const int owner_rank = ParentGtdofOwnerRank(parent_fes, parent_gtdof);
            if (owner_rank == my_rank) {
                const int parent_local =
                    static_cast<int>(
                        static_cast<HYPRE_BigInt>(parent_gtdof) - parent_offset);
                out.Append(parent_local);
            }
        }
    }

    return out;
}
```

The projector's TDOF map is consulted once per corner-component
candidate. The output is a list of **parent-FES local TDOFs** — the
same kind of list the mech operator expects via
`UpdateEssTDofsCornerSubset`.

The `AnchorCornerTDofsFromSpec` helper similarly translates the
anchor's three submesh-FES gtdofs to parent-FES local TDOFs through
the projector.

---

## §P6.11 Smart-pointer migration strategy

### §P6.11.1 Migration targets and rationale

The smart-pointer migration applies to interfaces where one component
**holds** another component for the duration of its lifetime, but
ownership is currently expressed as a raw reference or pointer.

| Holder | Held thing | Current | Post-v2 | Rationale |
|---|---|---|---|---|
| `MortarConstraintOperator` | `BoundaryClassifier3D` | `const X&` | `shared_ptr<const X>` | Explicit ownership |
| `MortarConstraintOperator` | `SurfaceProjector` | (new in v2) | `shared_ptr<const X>` | Same |
| `MortarConstraintOperator` | parent FES | (new in v2) | `shared_ptr<const X>` | Already shared_ptr in SimulationState |
| `ConstraintBuilder3D` | `BoundaryClassifier3D` | `const X&` | `shared_ptr<const X>` | Consistency |
| `ConstraintBuilder3D` | `SurfaceProjector` | (new in v2) | `shared_ptr<const X>` | Same |
| `MortarSaddlePointSystem` | `MortarConstraintOperator` | `const X&` | `shared_ptr<X>` | Calls `Reset` indirectly through manager; non-const |
| `MortarSaddlePreconditioner` | `MortarConstraintOperator` | `const X&` | `shared_ptr<const X>` | Read-only access pattern |
| `MortarPbcManager` | classifier, projector, builder, operator, saddle solver, saddle system | by value / unique_ptr | `shared_ptr<T>` | Multiple downstream consumers share these |

### §P6.11.2 const vs non-const policy

Per the user's policy:

> "Make use of `shared_ptr<const T>` only where it makes sense and if
> there might be a possibility of an object needing to be mutated in
> the future then we should probably err on the side of caution and
> go with a non-const version for it."

Application:

- **`shared_ptr<const T>` candidates**: The classifier (its
  post-construction state is immutable; corners / edges / faces are
  set once). The projector (its TDOF map is set once; the operator
  interface is logically `const`). The parent FES (mutated only via
  Phase 5.2 setup pathways outside the manager).
- **`shared_ptr<T>` (non-const) candidates**: The constraint operator
  (its filter spec is mutated via `Reset`). The saddle system (its
  block offsets are mutated via `Refresh`). The constraint builder
  (its current implementation is stateless but future versions might
  cache, so we hedge non-const).
- **By-value or unique_ptr (no sharing)**: lambda accumulator, per-
  row factor buffers, corner ess TDOFs — these are owned exclusively
  by the manager.

### §P6.11.3 What stays raw / by value

- **`MPI_Comm` handles** — POD-style, managed via RAII destructor.
- **`mfem::Vector` and `mfem::Array<int>` data members** — owned by
  value, copied / moved as needed by the holding class. Smart-pointer
  wrapping adds no clarity.
- **`int`, `double`, `bool`, etc.** — primitive types.
- **Function-local variables** — stack-owned.

### §P6.11.4 Cycle considerations

A potential cycle risk: if the manager holds `shared_ptr` to the
constraint operator, and the constraint operator's classifier holds
`shared_ptr` to the submesh which is owned by `SimulationState`, and
the manager holds `shared_ptr` to `SimulationState`, is there a
cycle?

No. The reference graph is:
- Manager → SimulationState → ParMesh, submesh, parent FES,
  submesh-FES.
- Manager → operator → classifier → submesh + submesh-FES.
- Manager → operator → projector → parent FES + submesh-FES +
  submesh.
- Manager → operator → parent FES.

Every edge points from the manager outward. Nothing points back. No
cycle.

If a future feature requires bidirectional ownership (e.g., a
post-processing handle that wants to "subscribe" to mortar updates),
the standard solution is `weak_ptr` on the upstream end.

---
## §P6.12 Algorithmic changes summary

This section consolidates the algorithmic changes scattered across
§P6.6-§P6.10 into a single reference table.

### §P6.12.1 BuildFlatRowArrays

**Pre-v2:** Classifier emits parent-FES gtdofs in pair blocks. The
operator's `BuildFlatRowArrays` reads them directly and uses
`ClassifyOwnership(parent_gtdof)` to determine local-vs-off-rank
storage.

**Post-v2:** Classifier emits submesh-FES gtdofs. The operator's
`BuildFlatRowArrays` consults the projector's TDOF map to translate
to parent-FES local TDOFs (or off-rank slot indices).

**Cost:** One additional O(1) lookup per flat-array entry. Total
construction cost dominated by the existing per-pair-block walk;
the lookup is in the noise.

### §P6.12.2 ComputeInvDiagSchur

**Pre-v2:** Probe `K_jacobi_prec.Mult(ones, inv_diag_K)`, get
`inv_diag_K` as parent-FES vector. Compute Schur diagonal from
per-pair blocks indexed by parent-FES gtdofs.

**Post-v2:** Probe `K_jacobi_prec.Mult(ones, inv_diag_K)` (same).
Project `inv_diag_K` via the projector to submesh-FES space. Compute
Schur diagonal from per-pair blocks indexed by submesh-FES gtdofs.

**Cost:** One additional Alltoallv (the projector's runtime
exchange). Paid once per `ComputeInvDiagSchur` call — typically
once per Newton step (or less, since the preconditioner can be
reused).

### §P6.12.3 ComputeCornerEssTDofs / ComputeCornerEssTDofsFromSpec

**Pre-v2:** Classifier's corner records hold parent-FES gtdofs.
Direct lookup in parent-FES partition to get local TDOFs.

**Post-v2:** Classifier's corner records hold submesh-FES gtdofs.
Translate via projector to parent-FES gtdofs, then look up in
parent-FES partition.

**Cost:** Three lookups per corner (one per component). Negligible
(8 corners × 3 components = 24 lookups total).

### §P6.12.4 EmitConstraintTriples (HypreParMatrix path)

**Pre-v2:** Builder emits triples `(row, col, value)` where `col` is
the classifier's parent-FES gtdof.

**Post-v2:** Builder emits triples `(row, col, value)` where `col`
is the projector's translation of the classifier's submesh-FES
gtdof to a parent-FES gtdof.

**Cost:** Same as §P6.12.1 — O(1) lookup per emitted triple.

### §P6.12.5 BuildLocalPairBlocks invariants

The classifier's `BuildLocalPairBlocks` performs the tile-
partitioned face-pair matching on the boundary subcomm (Phase 4.2).
The geometric algorithm (centroid matching, in-plane parametric
matching) operates on **physical coordinates**, not TDOF indices.
**Unchanged in Phase 6 v2.**

The block contents (D and A_m matrices) reference vertex indices,
not TDOFs. The submesh-FES gtdof lookup happens *after* block
matching, during the row-emission step. **Unchanged.**

### §P6.12.6 Cost analysis

Phase 6 v2 adds:

- **Construction-time cost:** One additional collective AllGather
  (or Allgatherv) in the projector's matching algorithm. Volume:
  O(boundary_DOF_count). For a 100³ RVE: ~10⁴ doubles + ~10⁴ ints.
- **Per-Reset cost:** One projector-map-lookup pass during
  `BuildFlatRowArrays`. Volume: O(active_flat_array_entries).
  O(1) per entry.
- **Per-Mult cost:** Zero. The runtime matvec is unchanged.
- **Per-MultTranspose cost:** Zero.
- **Per-ComputeInvDiagSchur cost:** One projector Alltoallv. Paid
  once per Newton step at most.

The dominant cost remains the per-Krylov-iteration matvec, which
is identical in structure (and bit-equivalent at `lor_depth = 1`)
to the pre-Phase-6 implementation.

---

## §P6.13 Tet p=2 — the Day 1 algorithmic target

### §P6.13.1 What the mesh looks like

A $p = 2$ tetrahedral mesh in MFEM has:

- **Volume elements**: 4-node tetrahedra (the geometry) with 10
  Lagrange nodes per element (4 corners + 6 mid-edges).
- **Boundary face elements**: 3-node triangles (geometry) with 6
  Lagrange nodes per face (3 corners + 3 mid-edges).
- **Surface FE space**: $P_2$ trace space on each triangular face;
  6 nodal DOFs per face.

The LOR refinement of a tri-3 face produces 4 sub-triangles by
connecting the 3 mid-edge points. The 6 vertices of these sub-
triangles are precisely the 3 corner + 3 mid-edge $P_2$ Lagrange
nodes. Pazner-Kolev coincidence holds exactly for straight-sided
tets.

### §P6.13.2 SetCurvature(2) for Neper-generated meshes

Neper produces $P_1$ tet meshes by default. To get a $P_2$ tet mesh
suitable for Phase 6:

```cpp
mfem::Mesh mesh(neper_mesh_file);
mesh.SetCurvature(/*order=*/2);
```

This places mid-edge nodes at the **midpoints of straight edges**
(linear interpolation), giving a $P_2$ mesh with planar faces.

For axis-aligned RVEs (Phase 6 Day 1 scope), this is sufficient —
the boundary faces are flat, and the mid-edge nodes are at exact
half-edge positions which match the LOR submesh's sub-triangle
vertices to machine precision.

For curved-boundary geometries, `SetCurvature` would need to project
mid-edge nodes onto the curved surface — out of scope for Phase 6.

### §P6.13.3 What stays the same at p=2

- The Phase 4 mortar pipeline runs unchanged on the LOR submesh.
- The Phase 5.9 component-restricted PBC machinery runs unchanged
  (the filter spec applies to face pairs, not to mesh order).
- The Phase 5.11 residual-scaling stack (§P5.19) runs unchanged.
- The Newton solver family (NR, NRLS, TRDOG) runs unchanged.
- The Hill-Mandel diagnostic runs unchanged (§P6.3.5).

### §P6.13.4 BBar interaction

ExaConstit's BBar integrator [Hughes 1980, architecture doc §4]
operates on volumetric pressure averaging at $p = 1$. At $p \ge 2$,
the BBar averaging would require a different formulation (typically
$\bar{B}$ from $L_2$ projection onto a piecewise-constant pressure
space, rather than the element-volume average used at $p = 1$).

**Phase 6 v2 does NOT support BBar at $p \ge 2$.** This is a hard
limitation enforced via:

```cpp
MFEM_VERIFY(!(options.mesh.order >= 2 && options.mech.use_bbar),
            "BBar volumetric stabilization is not yet supported at "
            "mesh order >= 2. Disable BBar or use order = 1.");
```

The assertion lives in `SystemDriver` construction or in
`mech_operator` construction (whichever is more natural for the
existing BBar code path). v1 documented this restriction as a
caveat; v2 hardens it to a build-time `MFEM_VERIFY` so accidental
misuse fails loudly rather than producing silently wrong stress.

### §P6.13.5 PA / EA / FA assembly mode considerations

ExaConstit supports three assembly modes:

- **PA (partial assembly)**: matrix-free; the volume operator is
  applied per-element without forming a sparse matrix. Current
  ExaConstit PA support is $p = 1$-specific for some integrators
  (notably the BBar integrator and parts of the crystal-plasticity
  integrators).
- **EA (element assembly)**: per-element sparse blocks, applied
  via element-by-element matvec.
- **FA (full assembly)**: globally assembled sparse
  `HypreParMatrix`.

At $p = 2$, PA may not be available for all integrators. The Day 1
TOML recommendation is `assembly = "EA"` or `assembly = "FA"`. PA at
$p \ge 2$ is a separate workstream.

The constraint operator side is unaffected — the EA path
(`MortarConstraintOperator`) and the HypreParMatrix path
(`ConstraintBuilder3D::BuildHypreParMatrix` consumed by the saddle
system) both work at any $p$.

---

## §P6.14 Hex p=2 — deferred but architecturally ready

The hex case is structurally simpler than tet (tensor-product LOR
refinement, no barycentric subdivision). The architecture from
Phase 6.0-6.2 covers hex without modification:

- **Quad-9 face elements** ($P_2$ hex boundary faces): 9 Lagrange
  nodes (4 corners + 4 mid-edges + 1 face centroid). LOR refinement
  produces 4 quad-4 sub-quads with 9 vertices. Pazner-Kolev
  coincidence holds.
- **The projector's matching algorithm** handles the additional
  face-centroid Lagrange node uniformly — same hash-based lookup,
  no special case.
- **The Wohlmuth modification** applies to the LOR submesh's
  wirebasket structure; for hex faces this is the same as the tet
  case (only the face element type differs).

Phase 6 v2 enables hex $p = 2$ trivially: add hex polycrystal
validation meshes to the test suite, exercise the same code paths.
No code changes needed beyond test-suite expansion.

We defer hex $p = 2$ to Phase 6.3 (later) to keep Phase 6.0-6.2
focused on tet, where the engineering pressure is.

---

## §P6.15 Optional Barbosa-Hughes residual stabilization

### §P6.15.1 The convergence rate question

Per §P6.3.3, the LOR pipeline at $p \ge 2$ has a stability gap that
manifests as suboptimal $H^1$ convergence rates ($O(h)$ instead of
$O(h^p)$). For engineering QoIs (homogenized stress, effective
tangent), the gap is benign. For academic convergence-rate studies,
the gap matters.

### §P6.15.2 Implementation sketch

Barbosa-Hughes stabilization adds a residual term to the saddle-
point bilinear form:

$$
+ \gamma_\beta \sum_E h_E \, \int_E (\lambda - \Pi_h(E_b u))(\mu - \Pi_h(E_b v)) \, dA
\qquad (P6.15.1)
$$

where $\Pi_h(E_b \cdot)$ is the trace of the elasticity edge-flux
onto the boundary, $\gamma_\beta = O(1/(\lambda + 2\mu))$ is a
material-dependent stabilization parameter, and $h_E$ is the local
element size.

Implementation: a new integrator class
`BarbosaHughesStabilizationIntegrator` that:

1. Assembles a contribution to the (2,2) block of the saddle-point
   system (currently zero).
2. Uses the projector to evaluate $\Pi_h(E_b u)$ — same projector,
   different use.

The saddle-point solver and preconditioner gain support for a
non-zero (2,2) block (most Krylov solvers handle this naturally;
the block-Jacobi preconditioner needs a small extension).

### §P6.15.3 When to enable

- **Default: off.** Engineering QoIs do not need it.
- **Opt-in via TOML:** `[Solvers.SaddlePoint].stabilization_gamma`,
  default 0 (off). Positive values enable.
- **Required for academic convergence-rate studies** at $p \ge 2$.

Phase 6.4 is the optional batch that delivers this.

---
## §P6.16 Phasing: the Phase 6 batch sequence

### §P6.16.1 Phase 6.0 — Foundational refactors

The heavy-lift batch. Touches the production code path; gated by
full regression.

```
Phase 6.0 — Foundational refactors

├── 6.0.A  SimulationState: GetBoundarySubMeshFes + GetLorBoundarySubMesh
│           + GetLorBoundarySubMeshFes. Lazy, shared_ptr cached, aliasing at
│           lor_depth = 1. Unit test: depth = 1 returns the same handle as
│           GetBoundarySubMesh; depth = 2 returns a 4x-refined submesh
│           with vertex coincidence at Pp Lagrange nodes to snap_tol = 1e-12.

├── 6.0.B  MeshOptions::validate allows lor_depth ∈ {1, 2}; requires
│           lor_depth == mesh.order when lor_depth > 1. TOML smoke test.

├── 6.0.C  BoundaryClassifier3D refactor:
│           - New constructor taking shared_ptr<ParSubMesh> +
│             shared_ptr<ParFiniteElementSpace>.
│           - Internal algorithm reads TDOFs from submesh-FES directly
│             (no GetParentVertexIDMap usage).
│           - Existing parent-ParMesh constructor moves to free function
│             MakeBoundaryClassifierFromParent (in
│             boundary_classifier_3d.hpp) for test convenience.
│           - Members migrate to shared_ptr<ParSubMesh> +
│             shared_ptr<ParFiniteElementSpace>.
│           - Unit tests (in test/mortar_pbc/) updated to use the free
│             function; behavioral parity expected.

├── 6.0.D  SurfaceProjector class:
│           - New file src/mortar_pbc/surface_projector.{hpp,cpp}.
│           - TDofMap data structure.
│           - Matching algorithm (snap-coord hash + Alltoallv).
│           - Mult / MultTranspose implementation.
│           - Unit tests: identity-on-constant, linear reproduction,
│             cross-rank topology at np = 1, 4, 7.

├── 6.0.E  MortarConstraintOperator refactor:
│           - New constructor taking shared_ptr<classifier> +
│             shared_ptr<projector> + shared_ptr<parent_fes>.
│           - BuildFlatRowArrays uses projector's TDOF map for
│             classifier-to-parent translation.
│           - BuildOffRankTopology uses parent-FES gtdofs (translated
│             via projector).
│           - ComputeInvDiagSchur projects inv_diag_K via projector
│             before Schur arithmetic.
│           - Split into ComputeInvDiagSchur(Solver&) and
│             ComputeInvDiagSchur(Vector&) overloads; the Solver version
│             probes then forwards to the Vector version.

├── 6.0.F  ConstraintBuilder3D refactor:
│           - Constructor takes shared_ptr<classifier> +
│             shared_ptr<projector>.
│           - EmitConstraintTriples uses projector to translate cols
│             from submesh-FES to parent-FES gtdofs.
│           - EmitRowFactors unchanged (geometric, no TDOF lookup).

├── 6.0.G  ComputeCornerEssTDofs / ComputeCornerEssTDofsFromSpec /
│           AnchorCornerTDofsFromSpec:
│           - Free functions in mortar_pbc_manager.{hpp,cpp}.
│           - All three gain a projector argument.
│           - Translate classifier corner records (submesh-FES) to
│             parent-FES local TDOFs via the projector.

├── 6.0.H  MortarPbcManager refactor:
│           - Single classifier (not parent + LOR pair).
│           - Construction sequence: classifier -> projector -> builder
│             -> operator -> saddle solver -> saddle system.
│           - All members migrate to shared_ptr.
│           - RebuildForActiveSpec passes projector to corner-pinning
│             helpers.

├── 6.0.I  Smart-pointer migration for saddle system and preconditioner:
│           - MortarSaddlePointSystem holds
│             shared_ptr<MortarConstraintOperator>.
│           - MortarSaddlePreconditioner holds
│             shared_ptr<const MortarConstraintOperator>.
│           - Verify the Phase 5.11 residual-scaling wrappers
│             (ScaledSaddleOperator, ScaledJacobianOperator,
│             ScaledSaddleSolver, ScaledSaddlePreconditioner per
│             §P5.19) consume the new shared_ptr handles cleanly
│             and that their internal Refresh paths still work
│             under spec changes (see §P6.18.14).

├── 6.0.J  Regression: full Phase 4 + Phase 5 test suite green at
│           np = 1, 4, 7. Bit-equivalence at lor_depth = 1 verified
│           on:
│           - test_mortar_constraint_operator (matvec)
│           - test_patch_3d_pbc (homogeneous)
│           - test_patch_3d_pbc_ea_compare (HypreParMatrix vs EA)
│           - test_mortar_pbc_manager (corner TDOFs)
│           - test_mortar_pbc_manager_filter (Phase 5.9 filter)
│           - mortar_pbc_linear_elastic.toml (Phase 5.7 driver)
│           - mortar_pbc_neohookean_50pct.toml
│           - mortar_pbc_voce_polycrystal.toml
│           - mortar_pbc_heterogeneous_strip_multistep.toml
│           - mortar_pbc_checkerboard_multistep.toml
│           - mortar_pbc_restart.toml

         ↓ (gate: lor_depth = 1 path is bit-equivalent to pre-Phase-6;
             new SurfaceProjector unit tests pass; classifier free
             function works for test code)
```

The Phase 6.0 gate is the **strongest** in the Phase 6 sequence: if
bit-equivalence at `lor_depth = 1` doesn't hold, the refactor is
broken and must be fixed before Phase 6.1 starts.

### §P6.16.2 Phase 6.1 — Enable lor_depth = 2

```
Phase 6.1 — Enable lor_depth = 2 for tet p=2

├── 6.1.A  Update MeshOptions::validate to accept lor_depth = 2 with
│           mesh.order = 2. Verify the validation catches:
│           - lor_depth = 2 with mesh.order = 1 (mismatch — reject).
│           - lor_depth = 2 with mesh.order = 3 (mismatch — reject).
│           - lor_depth = 1 with mesh.order = 2 (mismatch — reject;
│             p=2 primal without LOR is the unstable p/(p-1) case).

├── 6.1.B  P2 tet smoke test: small 4x4x4 P2 tet mesh
│           (via Mesh::MakeCartesian3D + SetCurvature(2)) under
│           homogeneous PBC. Manufactured-solution patch test:
│           v(x) = bar_L * x for an axis-aligned bar_L. PASS criterion:
│           u_tilde = 0 to Krylov tolerance after one Newton step.

├── 6.1.C  Cross-rank validation: same smoke test at np = 1, 4, 7.

├── 6.1.D  Phase 5.9 component-restricted test at p=2: X-only PBC on
│           the same 4x4x4 P2 mesh. Verify the filter cascades correctly
│           through the refactored stack.

         ↓ (gate: P2 tet smoke + component-restricted at multiple np)
```

### §P6.16.3 Phase 6.2 — P2 tet polycrystal validation

```
Phase 6.2 — Day-1 validation: P2 tet polycrystal

├── 6.2.A  Build the validation mesh: a 50-grain Neper polycrystal,
│           loaded via mesh.type = "file" and promoted to P2 via
│           SetCurvature(2). Crystal plasticity material (ExaCMech
│           Voce hardening). Simple shear macroscopic loading.

├── 6.2.B  Generate a P1 reference at matched DOF count: same
│           microstructure refined uniformly so total volume TDOF
│           count matches the P2 mesh.

├── 6.2.C  Run both at the same load schedule (5 time steps, simple
│           shear). Record per-step volume-averaged Cauchy stress.

├── 6.2.D  PASS criterion: ||sigma_homog_P2 - sigma_homog_P1ref|| /
│           ||sigma_homog_P1ref|| < 0.05 across all 5 steps and all
│           6 Voigt-stress components.

├── 6.2.E  Hill-Mandel diagnostic eta_HM < 1e-10 at every step (per
│           §P6.3.5).

├── 6.2.F  Run at np = 1, 4 (np = 7 optional if mesh size allows).

         ↓ (gate: P2 tet polycrystal validation green at np = 1, 4;
             Hill-Mandel < 1e-10 throughout)
```

### §P6.16.4 Phase 6.3 — Hex p=2 (deferred)

```
Phase 6.3 — Hex p=2 extension (mirror of Phase 6.1-6.2 with hex)

├── 6.3.A  Validation mesh: hex polycrystal with SetCurvature(2).
├── 6.3.B  Same smoke + polycrystal validation as Phase 6.1-6.2.
├── 6.3.C  PASS criterion same as Phase 6.2.D.
```

### §P6.16.5 Phase 6.4 — Barbosa-Hughes (deferred)

```
Phase 6.4 — Optional Barbosa-Hughes stabilization

├── 6.4.A  BarbosaHughesStabilizationIntegrator class.
├── 6.4.B  Saddle solver / preconditioner extension for non-zero
│           (2,2) block.
├── 6.4.C  TOML option [Solvers.SaddlePoint].stabilization_gamma.
├── 6.4.D  Convergence-rate study: P2 tet RVE with mesh refinement,
│           with and without stabilization. Verify O(h^p) recovery
│           when stabilization is enabled.

         ↓ (gate: stabilization works; opt-in default = off)

Phase 6 complete. Higher-order primal is supported in mortar PBC.
```

---

## §P6.17 Validation strategy

### §P6.17.1 Regression at lor_depth = 1 (Phase 6.0 gate)

The single most important validation: at `lor_depth = 1`, after the
Phase 6.0 refactor, **all Phase 4 + Phase 5 tests still pass and
matvec outputs are bit-equivalent to pre-refactor**.

Bit-equivalence means: for the same input vector $u$, the output
$Cu$ has the same numerical value at every entry, modulo Krylov-
tolerance differences. The flat array contents may differ (different
emission order, see §P6.8.7), but the matvec result is the same.

Tests to verify:
- `test_mortar_constraint_operator` (matvec correctness, np = 1, 4, 7).
- `test_patch_3d_pbc_ea_compare` (HypreParMatrix vs EA matvec
  equivalence to FP precision).
- All six end-to-end Phase 5.7 validation drivers
  (linear elastic, neohookean, polycrystal, multi-step heterogeneous,
  checkerboard, restart) at np = 1, 4, 7.

### §P6.17.2 Linear-reproduction (Phase 6.0 / 6.1)

`SurfaceProjector` unit tests:

1. **Identity on constant.** For a constant submesh-FES vector,
   `Mult` reproduces it as a constant parent-FES boundary vector,
   and vice versa.
2. **Linear reproduction.** For $u(x, y, z) = a x + b y + c z$
   projected onto parent FES, then applying `Mult` to the submesh-FES
   reconstruction must give the same coefficient vector at the LOR
   nodes (Pazner-Kolev coincidence).
3. **Cross-rank consistency.** Same tests at np = 1, 4, 7; the
   parallel projection must give the same result as the serial.

### §P6.17.3 P2 tet polycrystal vs P1 reference (Phase 6.2)

The Day-1 validation. A 50-grain Neper polycrystal mesh promoted to
$P_2$ via `SetCurvature(2)` is run under simple-shear loading with
ExaCMech Voce hardening for 5 time steps. A $P_1$ reference of the
same microstructure is generated at matched volume-TDOF count (the
$P_1$ reference is uniformly refined until its TDOF count equals
the $P_2$ mesh's). Both runs produce a per-step volume-averaged
Cauchy stress trajectory.

**PASS criterion:** for every step $n \in \{1, \ldots, 5\}$ and
every Voigt-stress component $c \in \{xx, yy, zz, xy, yz, xz\}$:

$$
\frac{\bigl|\, \sigma^{P2}_{n,c} - \sigma^{P1\text{ref}}_{n,c} \,\bigr|}
     {\,\| \sigma^{P1\text{ref}}_n \|\,} < 0.05.
\qquad (P6.17.1)
$$

The denominator uses the Frobenius norm of the full $P_1$ reference
stress tensor to avoid spurious large relative errors when an
individual component is near zero.

**Failure-mode hypotheses if the test fails:**

1. The LOR-side classifier emits constraint rows in an order that
   produces inconsistent per-row reference factors compared to the
   pre-Phase-6 emission order — manifests as the constraint not
   being satisfied to Krylov tolerance, which propagates to
   homogenized stress error.
2. The projector's parent-FES Lagrange-node enumeration misses
   mid-edge nodes on tri-6 boundary faces — manifests as a sparse
   constraint matrix with rank deficiency, breaking the saddle
   solve.
3. The volume quadrature order in ExaCMech does not match
   `mesh.order = 2`. Check `int_order = 2 * mesh.order + 1` is
   actually being used by the integrator.

Run at np = 1 (sanity) and np = 4 (parallel). np = 7 optional if
mesh size permits balanced partitioning.

### §P6.17.4 Hill-Mandel diagnostic at p=2

The Hill-Mandel residual $\eta_{\text{HM}}$ from (P6.3.6) is computed
at every load step by `MortarPbcManager::ComputeHillMandelPowerBalance`
(§P5.10.3 in the Phase 5 plan). At $p = 2$, the diagnostic must hold
to the same tightness as at $p = 1$ (typically < 1e-10).

If $\eta_{\text{HM}}$ degrades at $p = 2$, the failure modes (in
order of likelihood) are:
1. The lambda accumulation is being projected through the wrong
   side. The lambda lives in submesh-FES space; the
   $\overline{P} : \overline{\dot{F}}$ side uses parent-FES quadrature.
   The two sides must be consistently accounted for.
2. The volume quadrature order doesn't match the $p = 2$ primal.
   ExaConstit's `int_order = 2 * order + 1` should handle this
   automatically, but verify.
3. The constraint is not enforced tightly enough (Krylov tolerance
   issue, not a Phase 6 issue per se).

### §P6.17.5 Component-restricted PBC at p=2

Run the Phase 5.9 X-only test on a $P_2$ tet mesh. The corner
pinning, the filter spec rebuild via `RebuildForActiveSpec`, the
saddle system `Refresh` cascade — all must work under LOR.

PASS criterion: same as Phase 5.9 (rank-summed corner TDOF counts
match expected values; matvec result has correct dimensionality).

### §P6.17.6 Multi-rank A/B comparison

The HypreParMatrix path (`ConstraintBuilder3D::BuildHypreParMatrix`)
and the EA path (`MortarConstraintOperator`) produce mathematically
equivalent constraint enforcement. At $p = 2$, run both on the same
problem and compare outputs to Krylov tolerance.

PASS criterion: same matvec output to 1e-12 (FP precision); same
final $u$ solution to 1e-7 (Krylov tolerance accumulated over
iterations).

---

## §P6.18 Hazards and traps

The Phase 4 / Phase 5 trap list (architecture doc §12, Phase 4 plan
§P4.8, Phase 5 plan §P5.14) applies in its entirety. Phase 6 v2 adds:

### §P6.18.1 The byNODES vs byVDIM trap, again

The parent FES and the LOR-FES (and the un-refined-submesh FES) must
all use `Ordering::byNODES`. The projector's matching algorithm
assumes byNODES — the vdim-strided layout is incompatible with the
permutation-style TDOF lookup.

**Mitigation:** assert byNODES at projector construction. If any FES
is byVDIM, abort with a clear message. (Phase 5.9 already imposes
this on the parent FES through the constraint operator; Phase 6.0
extends to the new LOR-FES.)

### §P6.18.2 Snap-coordinate tolerance scaling

The default `snap_tol = 1e-10` is safe for unit-cube RVEs with
element sizes ≥ 1e-3. For very fine meshes or non-unit-scale RVEs,
this tolerance may be too tight (mismatches due to FP arithmetic) or
too loose (collisions in the snap-coord hash).

**Mitigation:** at projector construction, scale `snap_tol` by the
minimum element size in the submesh: `effective_tol = max(snap_tol,
1e-3 * min_edge_length)`. Add to the projector constructor's
diagnostic output.

### §P6.18.3 Cross-rank LOR-vs-parent partition mismatch

The LOR submesh and the parent ParMesh may have different MPI
partitions. The projector handles this via Alltoallv (§P6.7.5), but
the topology setup is non-trivial.

**Mitigation:** validate at np = 4, 7 with a deliberately
non-uniform partition (e.g., one rank holds most of one face).

### §P6.18.4 Corner-Dirichlet consistency

The 24 corner TDOFs (8 corners × 3 components) at full PBC are
parent-FES local TDOFs. The classifier emits corner records in
submesh-FES gtdofs. The translation via the projector must produce
exactly 24 distinct parent-FES TDOFs (rank-summed).

**Mitigation:** `MortarPbcManager::BuildCornerEssTDofs` includes a
rank-sum sanity check (already present in Phase 5.3.B). At Phase 6.0,
extend the check to verify that the projector-translated TDOFs match
the expected count.

### §P6.18.5 BBar + p ≥ 2 rejection

ExaConstit's BBar integrator does not support $p \ge 2$ (§P6.13.4).
Enabling both leads to silently wrong volumetric stabilization.

**Mitigation:** hard `MFEM_VERIFY` at SystemDriver / mech_operator
construction. Message: "BBar stabilization is not supported at
mesh.order ≥ 2; disable BBar or use mesh.order = 1."

### §P6.18.6 LOR refinement of curved surfaces

For curved $P_2$ boundaries (e.g., a cylindrical RVE), the LOR
submesh's mid-edge vertices may not lie on the curved surface — they
follow MFEM's mid-side-vertex placement strategy, which is linear
interpolation by default.

**Mitigation:** Phase 6 v2 targets axis-aligned (flat-face) RVEs.
For curved geometries, defer to a future phase (architecture doc
§13.3, likely via Tribol).

### §P6.18.7 essential_vel_grad projection at p ≥ 2

The corner-pin BC value computation projects $v(x) = \bar{L} \cdot
(x - x_0)$ onto the parent FES. At $p = 2$, this projection is onto
a $P_2$ space.

**Mitigation:** verify MFEM's `VectorFunctionCoefficient::Project`
handles vdim = 3 at $p = 2$ correctly. Smoke test in Phase 6.0.

### §P6.18.8 ParSubMesh::UniformRefinement aliasing

Mutating the cached LOR submesh via `UniformRefinement` would corrupt
any aliasing at `lor_depth = 1` (§P6.5.5).

**Mitigation:** documented discipline in `SimulationState` docstrings:
cached submeshes are immutable from the consumer's perspective. No
type-level enforcement (would require a separate `const`-ness
cleanup in SimulationState).

### §P6.18.9 ConstraintRHS sizing under projector

The constraint-RHS buffer `m_g_rhs` in `MortarPbcManager` must be
sized to `m_C_op->Height()`. At `lor_depth = 2`, this is the LOR
constraint row count, which differs from the un-refined value. The
manager's `Refresh`-cascade handling already does this correctly
(via the `RebuildForActiveSpec` mechanics implemented in Phase 5.9.E),
but the path is exercised more aggressively at $p = 2$.

**Mitigation:** unit test in Phase 6.1: verify `m_g_rhs.Size() ==
m_C_op->Height()` after both initial construction and a
`RebuildForActiveSpec` call, under `lor_depth = 2`.

### §P6.18.10 Per-row reference geometric factors under LOR

The per-row reference factors emitted by `EmitRowFactors` are
geometric (signed periodic shift, component, lumped-row factor).
Under LOR, the row count is larger (more constraint rows on the
refined submesh), but each row's geometric data is determined by the
classifier's pair-block walk on the LOR submesh. The geometric data
should be consistent — the LOR submesh's vertex positions are at the
Pp Lagrange nodes, and the bbox / corner geometry is preserved by
refinement.

**Mitigation:** verify in Phase 6.1.B that the per-row factors emitted
under `lor_depth = 2` produce a constraint that is satisfied by the
manufactured affine solution $v(x) = \bar{L} x$. If the factors are
wrong, the affine solution will not satisfy $C v = g$ to Krylov
tolerance.

### §P6.18.11 The classifier's `Fes()` accessor change

Pre-v2: `classifier.Fes()` returns the parent FES.
Post-v2: `classifier.Fes()` returns the submesh-FES.

This is an API-level change. Any code calling `classifier.Fes()` and
expecting the parent FES is broken.

**Mitigation:** audit all callers of `classifier.Fes()` in Phase 6.0;
update them to use `m_parent_fes` (held separately by the
manager / operator) or the projector if they need parent-FES TDOFs.

### §P6.18.12 Test infrastructure shifts

Test code that builds classifiers via the old `BoundaryClassifier3D
(pmesh, fes)` constructor is broken. The free function
`MakeBoundaryClassifierFromParent` provides the migration path.

**Mitigation:** update all `test/mortar_pbc/test_*.cpp` files in
Phase 6.0.C. The change is mechanical (one line per test); the test
behavior is preserved.

### §P6.18.13 The classifier's `GtdofOwnerRank` under shared-ptr migration

Pre-v2, `BoundaryClassifier3D::GtdofOwnerRank(gtdof)` returned the
parent-FES owner rank for a parent-FES gtdof. Under v2, the
classifier holds the submesh FES, so `GtdofOwnerRank` returns the
submesh-FES owner rank for a submesh-FES gtdof.

**This semantic change is intentional** — the classifier's gtdofs
are now submesh-FES gtdofs, so the owner rank query is consistent.
Consumers that need parent-FES owner ranks must use the projector
instead.

**Mitigation:** audit all `GtdofOwnerRank` call sites in Phase 6.0.
Replace with projector-mediated parent-FES owner queries where
appropriate. Specific call sites to audit:
- `MortarPbcManager::BuildCornerEssTDofs` (Phase 5.3.B).
- `MortarConstraintOperator::BuildOffRankTopology`.
- `ConstraintBuilder3D::BuildHypreParMatrix` (for column-partition
  setup).

### §P6.18.14 Refresh cascade through smart pointers

The Phase 5.11 residual-scaling wrappers (§P5.19) introduce a stack
of four classes around the saddle solve:
`ScaledSaddleOperator` (residual wrap + Jacobian-wrap),
`ScaledJacobianOperator` (evaluates $D^{-1} J D$ on the fly),
`ScaledSaddleSolver` (pre-scales RHS, post-unscales increment), and
`ScaledSaddlePreconditioner` (conjugates inner prec by $D^{-1}$).

Under v2's `shared_ptr` migration, the non-const
`shared_ptr<MortarConstraintOperator>` in the saddle system needs
to propagate size updates through this stack. The chain
`MortarPbcManager::RebuildForActiveSpec` →
`m_saddle_system->Refresh()` → `m_C_op->Height()` is the v2 path;
the scaling wrappers re-read sizes through the shared handle and
their internal Refresh paths (per §P5.19) re-derive the scaling
partition for the new active-spec block structure.

**Hazard.** The scaling wrappers (per §P5.19) consume the
underlying saddle stack through `mfem::Operator` and `mfem::Solver`
references and through `Refresh` calls that re-read sizes from the
constraint operator. The shared-pointer migration is intended to be
backward-compatible — `.get()` on a `shared_ptr` yields the same
raw pointer the wrappers historically held — but the
`Refresh`-cascade semantics under spec change at `lor_depth = 2`
must be verified, not assumed.

**Mitigation:** explicit unit test exercising `RebuildForActiveSpec`
under `lor_depth = 2` with the scaler enabled and a non-trivial
sub-block partition (e.g. `partition = PerPair`, X-only spec).
Assert:
- All four wrappers' sizes reflect the new operator height after
  `Refresh`.
- The scaler's `m_d_lambda` / `m_subblock_factor` arrays are
  re-sized correctly when the active sub-block count changes
  under filter.
- The two-run diff infrastructure (Phase 5.11.K with
  `floor = 1.0e+30`) produces bit-equal CSVs against the
  pre-Phase-6 implementation at `lor_depth = 1`.

Test at np = 1, 4 minimum.

---

## §P6.19 Open questions and forward plan

### §P6.19.1 Should `SurfaceProjector` ever be reused outside mortar PBC?

The projector machinery (parent-FES TDOFs ↔ submesh-FES TDOFs via
Pazner-Kolev coincidence) is general — it works for any high-order
boundary trace problem, not just mortar PBC. Potential future
consumers:

- **Visualization mapping.** Project high-order primal fields onto
  the LOR submesh for ParaView output without rasterization
  artifacts. The user's `SimulationState` comment foreshadows this.
- **Surface integral diagnostics.** Compute surface tractions on a
  refined visualization mesh rather than at parent FES quadrature
  points.
- **Tribol-style contact interfaces.** If ExaConstit grows a contact
  capability, the same projector mechanics apply.

For Phase 6, the projector lives in `src/mortar_pbc/` and serves only
the mortar pipeline. A future refactor could move it to
`src/sim_state/` or `src/utilities/` alongside other shared
infrastructure, with mortar PBC becoming one of multiple consumers.
We defer this decision.

### §P6.19.2 Should the boundary submesh FES live in `SimulationState` even when mortar PBC is disabled?

Currently, `SimulationState::GetBoundarySubMesh()` is lazily built on
first call. The new `GetBoundarySubMeshFes()` follows the same
pattern. If no consumer asks for them, they are never built — zero
overhead.

This is the right default. Future post-processing or visualization
code that wants a boundary FES uses the same accessor; no
mortar-PBC-specific gating.

### §P6.19.3 Should the LOR refinement depth be per-boundary-attribute?

Phase 6 v2 uses a single `lor_depth` for all boundary face
attributes. For RVE problems with axis-aligned boundaries, all six
faces have the same primal order, so a single depth is correct.

If a future problem has mixed-order boundaries (e.g., one face at
$p = 1$ from a coarser parent mesh region, another at $p = 2$ from
a refined region), per-attribute depth might be needed. We defer
until a concrete use case emerges.

### §P6.19.4 What about $p \ge 3$?

`lor_depth = 3` (refine twice) would support $p = 3$ tet via 16
sub-triangles per face. The projector algorithm extends naturally;
the FES, classifier, and operator are all unchanged. The only
blocker is performance: at $p = 3$ on a 100³ mesh, the LOR submesh
has 9× the elements of the un-refined one, ~5× the TDOFs of the
$p = 2$ case.

Phase 6 limits `lor_depth ∈ {1, 2}` for two reasons:

1. No current ExaConstit use case demands $p = 3$.
2. The Pazner-Kolev coincidence holds exactly for tensor-product /
   barycentric refinement at $p = 2$; at $p = 3$ there are
   higher-order edge interior Lagrange nodes whose positions must
   match the twice-refined LOR submesh's mid-mid-edge vertices.
   The geometric coincidence still holds, but the projector's
   matching algorithm needs extension (more Lagrange node types
   per element).

Lifting `lor_depth` to 3 is a future phase.

### §P6.19.5 What happens at curved boundaries beyond axis-aligned?

Per §P6.18.6: out of scope for Phase 6. The Pazner-Kolev coincidence
becomes approximate (machine-epsilon, not exact) when boundary faces
are curvilinear. The projector's snap-tolerance handles this in
principle; in practice, curved boundaries introduce additional
sources of mismatch (mid-edge node placement on the curved surface
vs. on the chord between corners). Defer to a future phase, likely
with Tribol or a Mesh::SetCurvature-on-the-LOR-submesh strategy.

### §P6.19.6 Forward plan summary

- **Phase 6.0 (foundational refactor)** lands first. Gates: full
  Phase 4 + Phase 5 test suite green at `lor_depth = 1`.
- **Phase 6.1 (enable lor_depth = 2)** is then a small enablement
  batch. Gates: P2 tet smoke + component-restricted at np = 1, 4, 7.
- **Phase 6.2 (P2 tet polycrystal validation)** is the Day-1 win.
- **Phase 6.3 (hex p=2)** is a small additive extension.
- **Phase 6.4 (Barbosa-Hughes)** is optional, defaulted off.

Open questions feed into the long-term roadmap (architecture doc
§13, §14) but do not block Phase 6 completion.

---

## §P6.20 Cross-references to other planning docs

### §P6.20.1 Architecture doc references

- **§4.10**: Popp-Wohlmuth-Gee-Wall basis-transformation procedure;
  Phase 6 v2 explicitly defers this in favor of LOR + Barbosa-Hughes
  (§P6.3.4).
- **§4.11**: LOR theory (Pazner-Kolev, linear dual on refined
  submesh, stability considerations). Phase 6 v2 §P6.3 recapitulates
  the core results and refers to §4.11 for full derivations.
- **§4.12**: Recommendation for ExaConstit higher-order PBC. Phase 6
  v2 is the implementation of this recommendation.
- **§5**: Wohlmuth crosspoint modifications. These continue to apply
  unmodified on the LOR submesh (§P6.8 — the modifications are
  classifier-level operations and the refactored classifier
  performs them identically on a refined submesh).
- **§6**: Saddle-point system and Krylov solver. Phase 6 v2 does not
  change this — the operator is consumed by the saddle system via
  the same `mfem::Operator` interface.
- **§7**: Warm-start theory. Phase 5.8 (`UpdateConstraintRHS`) is
  the implementation; Phase 6 v2 does not modify it. The
  per-row reference geometric factors continue to drive the RHS
  computation correctly.
- **§8**: Hill-Mandel diagnostic. §P5.10.3 in the Phase 5 plan is
  the implementation; Phase 6 v2 validates that it holds at
  `lor_depth = 2` to the same numerical tightness as at
  `lor_depth = 1` (§P6.3.5, §P6.17.4).
- **§9**: Total Lagrangian / Updated Lagrangian discipline. Phase 5.8
  is the UL adaptation; Phase 6 v2 does not modify it.
- **§11.7**: 3D mesh + boundary classifier. Phase 6 v2 §P6.6 is the
  classifier refactor.
- **§13**: C++ port pathway. Phase 6 v2 is the implementation of
  the LOR path described there.
- **§14**: Open questions and forward plan; Phase 6.6+ items
  reside there.

### §P6.20.2 Phase 4 plan references

- **§P4.4.4**: Phase 4.2 distributed-pair matching. The boundary
  subcomm and tile-partitioning logic is preserved through the
  refactor.
- **§P4.4.6**: Phase 4.3 EA path. Phase 6 v2's runtime matvec is
  the same EA implementation.
- **§P4.4.6.9**: Phase 4.3.B GPU port (still in flight). The Phase 6
  v2 refactor preserves the GPU-compatible flat-array structure
  (`mfem::forall`, typed memory accessors).
- **§P4.13**: Phase 4 done criteria. Phase 6 v2 does not change these.

### §P6.20.3 Phase 5 plan (v7) references

Note on naming convention: in the Phase 5 v7 plan, `§P5.X` refers to a
*section* and `Phase 5.X.Y` refers to a *batch* in the phasing
(§P5.13). The section and batch numberings are independent —
e.g. `§P5.9` (the section) is "Multi-region, assembly mode, and GPU
compatibility" while `Phase 5.9` (the batch) is the
"Component-restricted PBC" implementation work. Similarly, `§P5.19`
(the section) documents saddle-system residual scaling while
`Phase 5.11` (the batch) is its implementation. Phase 6 v2
references both as needed.

- **§P5.4**: `MortarPbcManager` class. Phase 6 v2 §P6.10
  simplifies its construction (single classifier instead of
  parent + LOR pair) and adds projector-aware corner pinning.
- **§P5.5**: Corner-Dirichlet via the existing BCManager /
  `mono_def_flag` pattern. Unchanged in v2.
- **§P5.6**: SystemDriver `Solve()` / `SolveInit()` / per-step
  lifecycle. Unchanged.
- **§P5.7**: Newton-solver integration via `MortarSaddlePointSystem`
  (NRLS in §P5.7.3, TRDOG in §P5.7.4). Unchanged.
- **§P5.8**: The mortar PBC constraint under Updated Lagrangian —
  theoretical justification and the constraint-RHS update under UL
  (§P5.8.6). Unchanged.
- **§P5.9**: Multi-region, assembly mode, and GPU compatibility.
  Unchanged. (Component-restricted PBC content lives in §P5.18.)
- **§P5.10**: Output and post-processing, including the Hill-Mandel
  power balance diagnostic (§P5.10.3). Phase 6 v2 validates
  Hill-Mandel at $p = 2$ to the same tightness as at $p = 1$
  (§P6.3.5, §P6.17.4).
- **§P5.11**: CMake / build system. Phase 6 v2 adds the
  `surface_projector.{hpp,cpp}` files to the `src/mortar_pbc/`
  CMake target.
- **§P5.13**: Phasing. Phase 6 v2 §P6.16 follows the same
  ASCII-tree style. The Phase 5.9 batch (in §P5.13) is the
  implementation of component-restricted PBC; the Phase 5.11 batch
  is the implementation of saddle-system residual scaling
  (§P5.19); Phase 6 v2 verifies both continue to work through the
  projector-mediated operator (§P6.10.3, §P6.17.5, §P6.18.14).
- **§P5.14**: Hazards and traps. Phase 6 v2 §P6.18 extends this list
  with traps specific to the LOR / projector machinery.
- **§P5.18**: Component-restricted PBC — the `PeriodicBC` spec,
  two-axis filtering, EA operator row layout under filter, and
  `RebuildForActiveSpec`. Phase 6 v2 §P6.10.3 honors all the
  invariants documented there.
- **§P5.19**: Saddle-system residual scaling. The four wrappers
  (`ScaledSaddleOperator`, `ScaledJacobianOperator`,
  `ScaledSaddleSolver`, `ScaledSaddlePreconditioner`) and the
  `SaddleResidualScaler` consume the constraint operator through
  the same `mfem::Operator` API; they are transparent to the
  projector-mediated TDOF translation introduced in Phase 6 v2.
  The smart-pointer migration in §P6.11 ensures the operator
  handles flow through this stack cleanly; §P6.18.14 captures the
  `Refresh`-cascade trap that the shared-handle change interacts
  with.

### §P6.20.4 v1 references (superseded)

The v1 doc (`PHASE6_HIGHER_ORDER_LOR.md`) is **superseded** by this
v2 document. References to v1 §P6.X.Y in code comments or other
planning material should be updated to v2 references using the
following correspondence:

| v1 reference | v2 reference |
|---|---|
| §P6.4 (Architectural overview: wrapper pattern) | §P6.4 (option γ unified architecture) |
| §P6.5 (LOR ParSubMesh construction) | §P6.5 (SimulationState as home for shared infrastructure) |
| §P6.6 (`SurfaceLORProjector`) | §P6.7 (renamed to `SurfaceProjector`) |
| §P6.7 (`HighOrderMortarConstraintOperator`) | §P6.8 (folded into `MortarConstraintOperator`) |
| §P6.8 (Wohlmuth on LOR) | implicit in §P6.6, §P6.8 |
| §P6.9 (non-conforming on LOR) | mentioned in §P6.1.2 non-goals (deferred with Phase 4.4) |
| §P6.10 (Tet p=2 Day 1) | §P6.13 (Tet p=2 algorithmic target) |
| §P6.11 (Hex deferred) | §P6.14 |
| §P6.12 (Barbosa-Hughes optional) | §P6.15 |
| §P6.13 (Phasing) | §P6.16 |
| §P6.14 (Hazards) | §P6.18 |

---

## §P6.21 Done criteria

Phase 6 v2 is **done** (for the tet $p = 2$ Day 1 scope) when ALL
of the following hold:

- [ ] **Phase 6.0 foundational refactor lands cleanly.**
  - `SimulationState` exposes `GetBoundarySubMeshFes`,
    `GetLorBoundarySubMesh`, `GetLorBoundarySubMeshFes`.
  - `BoundaryClassifier3D` accepts a pre-built submesh + FES on
    that submesh; old parent-mesh constructor moved to
    `MakeBoundaryClassifierFromParent` free function.
  - `SurfaceProjector` class exists with documented matching
    algorithm + cross-rank Alltoallv topology.
  - `MortarConstraintOperator`, `ConstraintBuilder3D`,
    `MortarPbcManager` refactored as described in §P6.8-§P6.10.
  - Smart-pointer migration complete per §P6.11.

- [ ] **Phase 6.0 regression gate green.** Full Phase 4 + Phase 5
  test suite at np = 1, 4, 7. Bit-equivalence at `lor_depth = 1`
  on all matvec-output-checking tests.

- [ ] **Phase 6.1 enables `lor_depth = 2`.** P2 tet smoke test
  (patch test on 4³ mesh) passes at np = 1, 4, 7. Phase 5.9
  component-restricted spec change verified at p=2.

- [ ] **Phase 6.2 P2 tet polycrystal validation green.** PASS
  criterion 6.2.D (§P6.16.3): relative stress error < 5% across 5
  steps × 6 components vs. P1 reference at matched DOF count.

- [ ] **Hill-Mandel diagnostic** $\eta_{\text{HM}} < 10^{-10}$ at
  every step of the Phase 6.2 validation, at $p = 2$.

- [ ] **No `// TODO` markers in production code paths** (only
  acceptable in code explicitly deferred to Phase 6.3 / 6.4).

- [ ] **Doxygen-complete public API** for `SurfaceProjector`,
  refactored `BoundaryClassifier3D`, refactored
  `MortarConstraintOperator`, and the SimulationState additions.

- [ ] **BBar + p ≥ 2 hard rejection** with clear error message
  (§P6.13.4, §P6.18.5).

- [ ] **TOML validation correctly rejects** `lor_depth = 2` with
  `mesh.order = 1` and vice versa (Phase 6.0.B sub-batch in
  §P6.16.1; Phase 6.1.A sub-batch in §P6.16.2).

- [ ] **Hex $p = 2$ explicitly deferred** to Phase 6.3 (clearly
  marked as not Day 1).

- [ ] **Barbosa-Hughes stabilization explicitly deferred** to
  Phase 6.4 (clearly marked as not Day 1, opt-in default off).

When all these hold, Phase 6 v2 is complete and ExaConstit supports
higher-order primal fields ($p = 2$ tet) in mortar PBC. The next
logical step is Phase 6.3 (hex $p = 2$) or Phase 6.4
(Barbosa-Hughes), or the long-term items in architecture doc §14.3
(Tribol integration for non-axis-aligned RVEs, FE² coupling, $p \ge 3$).

---

End of `PHASE6_HIGHER_ORDER_LOR_v2.md`.

This document should be re-read at the start of each major work
session within Phase 6. When new bugs are encountered, add them to
§P6.18. When new architectural decisions are made, update §P6.4
(if architectural) or §P6.19 (if forward-looking). When a question
in §P6.19 is answered, move it to a "decided" subsection or remove
it.
