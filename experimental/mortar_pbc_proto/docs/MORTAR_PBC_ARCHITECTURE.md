# Mortar Periodic Boundary Conditions for Computational Homogenization
## Theory, Practice, and a Roadmap from 2D to 3D, ExaConstit-Bound

> **Living architecture document.** Read this once before touching the code; refer
> back to it when designing new pieces. Anyone joining the project — whether they
> already know FEM but not mortar methods, or vice versa — should leave this doc
> understanding *why* every architectural choice was made and *how* the pieces
> interlock to form a single homogenization driver.

---

## Document scope and audience

This document is the all-guiding reference for the mortar non-conforming periodic
boundary conditions (PBC) prototype, developed in pyMFEM as a precursor to
production C++ integration into ExaConstit (LLNL crystal-plasticity FE code,
MFEM/RAJA-based, partial-assembly / GPU). It captures:

1. **The math**: enough computational mechanics and mortar-method theory that a
   reader with a normal FEM background but no specialised PBC / mortar exposure
   can follow every algorithmic decision.
2. **The current code**: what each module does and why; how the saddle-point,
   constraint-builder, and warm-start pieces fit together.
3. **The hard-won lessons**: the bugs we hit, the half-formulations that nearly
   worked, and the diagnostics that finally caught the problem. Future-Claude (or
   future-anyone) should not re-discover these.
4. **The 3D extension plan**: the hierarchical wirebasket structure, the dual-basis
   modifications, the staging, the open design questions. Treat this section as
   the working contract for what Phase 3 means and how it stages into ExaConstit.

The total length is intentional. A short doc would force readers back to the
2021 Lopes paper and our six prior session transcripts; this doc is a single
self-contained source of truth.

> If you are reading this to start work, the recommended first pass is:
> §0 (vocabulary), §1 (high-level mental model), §2 (Method C vs D), §10 (status
> at this checkpoint), §11 (Phase 3 plan). The remaining sections are reference.

---

## Table of Contents

- §0. Vocabulary and notation
- §1. The big picture: what computational homogenization needs from PBC
- §2. Two formulations: Method C vs Method D, and why we use D
- §3. The mortar method — variational form, discrete construction, algorithm
- §4. The dual basis: derivation, simplex unification, and explicit formulas
    - §4.0 Derivation from the bi-orthogonality requirement
    - §4.1 Simplex unification: line-2, tri-3, tet-4 (M_i = (d+2) N_i − 1)
    - §4.2 Line-2 (1D simplex)
    - §4.3 Quad-4 (2D hypercube tensor product)
    - §4.4 Tri-3 (2D simplex; tet-mesh face element)
    - §4.5 Tet-4 (3D simplex; for volume mortar)
    - §4.6 Hypercubes vs simplices
    - §4.7 Why bi-orthogonal: condition number and Schur complement
    - §4.8 Higher-order: the line-3 dual basis (1D, p = 2)
    - §4.9 The bi-orthogonality obstruction at p ≥ 2 on simplices and serendipity (with general predictive criterion)
    - §4.10 The Popp-Wohlmuth-Gee-Wall basis-transformation procedure
    - §4.11 The lower-order projection (LOR) fallback
    - §4.12 Recommendation for ExaConstit higher-order PBC
- §5. Hierarchical crosspoint structure and the Wohlmuth modification
    - §5.1 The 2D problem and the line-2 modification
    - §5.2 The triangle (tri-3) modification (3D face mortar on tet meshes)
    - §5.3 The quad-4 modification (3D face mortar on hex meshes)
    - §5.4 The 3D wirebasket hierarchy
    - §5.5 Hex meshes vs tet meshes: same hierarchy, different elements
    - §5.6 Why this matters for correctness
- §6. The saddle-point system and how we solve it
- §7. Warm-start theory: from ExaConstit's `SolveInit` to multi-step F ramping
    - §7.4 Derivation of the projection equation (eq. 7.4)
- §8. Diagnostics: volume-averaged F as the consistency check
    - §8.1 Hill-Mandel theorem with explicit divergence-theorem derivation
- §9. Visualisation and the total-Lagrangian discipline
- §10. Status at the Phase-2 ↔ Phase-3 boundary
- §11. Extending to 3D: the wirebasket framework
    - §11.1 The hierarchy and what changes from 2D
    - §11.2 Hex track: hex-8 volumes with quad-4 face mortar
    - §11.3 Tet track: tet-4 volumes with tri-3 face mortar
    - §11.4 Mixed hex-tet meshes
    - §11.5 The 3D edge mortar
    - §11.6 The face mortar geometric-matching algorithm
    - §11.7 The 3D mesh + boundary classifier
    - §11.8 The phasing plan for Phase 3
    - §11.9 Open Phase-3 design questions
- §12. Hard-won lessons (the trap list)
- §13. C++ port pathway into ExaConstit
- §14. Open questions and forward plan
- §15. References

---

# §0. Vocabulary and notation

This section is for readers with a regular FEM background who have not worked
on mortar methods or RVE homogenization before. Skim it; come back when an
unfamiliar term appears.

| Symbol / term | Meaning |
|---|---|
| **RVE** | Representative Volume Element. The microscale domain Ω over which we solve a boundary-value problem and from which we read back homogenized stress / tangent. For us, Ω is a square (2D) or cube (3D); call its side length L and its volume V. |
| **F**, **F_macro** | The (prescribed) macroscopic deformation gradient. A 2×2 (resp. 3×3) tensor that drives the homogenization. |
| **u(X)** | Total displacement field on the RVE. Reference coordinates X. |
| **u_lin(X)** | The affine part: u_lin = (F − I) X. By construction this gives ∇u_lin = F − I, a constant field that reproduces F exactly. |
| **ũ(X), u_tilde** | The fluctuation: ũ = u − u_lin. Required to be Ω-periodic so that ⟨F⟩_Ω = F_macro by the average theorem. |
| **nonmortar / mortar** *(or **−** / **+**, equivalently B / A)* | The two sides of a mortar coupling. The Lagrange-multiplier rows live on the **nonmortar** ("−", "B") side; the **mortar** ("+", "A") side provides the values that feed the constraint. Naming follows the Wohlmuth-mortar literature and the `D^{nm}` / `A^m` matrix names: the "nm" superscript on D refers to the nonmortar-side mass; the "m" superscript on A refers to the mortar-side trace. The dual basis lives on the nonmortar side. **Pre-existing convention note:** the Python prototype's docstrings (e.g. `mortar_pbc/mortar_2d.py`, citing the Lopes 2021 paper) use the opposite "+"/"−" mapping ("+" = nonmortar, "−" = mortar). The mapping to "nonmortar"/"mortar" is unambiguous; the +/− symbols are a recurring source of cross-paper notational disagreement. |
| **C** | The constraint matrix: rows index Lagrange multipliers (one per nonmortar-side periodic DOF, per spatial component); columns index displacement TDOFs. C·u = 0 is the discrete periodicity condition. |
| **λ** | Lagrange multipliers, one per row of C. Physically: the periodic-traction reactions on the nonmortar side. |
| **TDOF** | True degree of freedom. In MFEM parlance, the global, uniquely-owned (after parallel partition) displacement components. Distinct from local LDOFs that include shared/ghost copies. |
| **K** | The tangent stiffness operator. Linear elastic in our prototype; nonlinear (e.g. crystal plasticity) in the eventual ExaConstit deployment. We treat K strictly as an `mfem::Operator` — never gathered to CSR for the actual solve, never assumed to be a `HypreParMatrix`. |
| **Saddle-point system** | The block linear system [[K, Cᵀ], [C, 0]] [u; λ] = [b; 0] (or its Newton-step version). Indefinite — that's why CG is rejected; we use MINRES / GMRES / BiCGStab. |
| **Patch test** | The minimal correctness criterion: a homogeneous RVE under uniform F must produce ũ = 0 to machine precision. If any version of the code fails the patch test, that's a hard fail (not a "pretty close" — exactly zero). |
| **Mortar method** | A weak-coupling FE technique for joining non-matching meshes across an interface. Originally developed for domain decomposition (Bernardi-Maday-Patera), extended to dual basis (Wohlmuth 2000, 2001) for diagonal Schur complement. We use it to enforce ũ(X⁺) = ũ(X⁻) at periodic boundary pairs without requiring the meshes on opposite faces to align. |
| **Wirebasket** | In 3D, the union of edges (the "wires") of the RVE. In a hierarchical PBC formulation, edges are coupled separately from faces and corners are pinned separately from edges, so that each level's constraint complements the next. |
| **Crosspoint** | A geometric point where an edge meets a corner (2D) or a face meets an edge or corner (3D). The dual-basis support of the nonmortar-side mortar Lagrange multipliers must be modified at crosspoints (Wohlmuth's modification, Lopes Eq. C.2 and §4.4.2). |
| **Method C, Method D** | Two different ways to assemble the mortar PBC system. See §2. We use Method D for the prototype. |
| **Total Lagrangian** | A kinematic framework where every operation (FE assembly, gradient evaluation, integration, projection) happens with respect to the *reference* (undeformed) configuration. This is what we use everywhere except visualisation. |
| **Updated Lagrangian** | An alternative where the reference configuration *resets* to the current configuration at each load step. ExaConstit is updated-Lagrangian at the *macroscopic* time-step level: at the end of each step the converged kinematic state becomes the new "reference" for the next step's stress evaluation. Conceptually distinct from the discretization; relevant when planning the C++ port. |

Notational convention used throughout:
- Bold lower-case for vectors (**u**, **F**), bold upper-case for tensors / matrices when no ambiguity.
- Subscripts c / u distinguish *constrained* / *unconstrained* DOFs (essential / free in the FE-jargon sense).
- Superscripts n, n+1 index load steps.
- "Step" without further qualification means *load step*. "Iteration" means *Newton iteration* within a load step.

---

# §1. The big picture: what computational homogenization needs from PBC

A computational homogenization scheme handles a multiscale solid mechanics
problem by replacing a real, microscopically-heterogeneous material with an
*effective* macroscopic one, whose constitutive behaviour is queried by solving
a microscale BVP on a *Representative Volume Element* (RVE) at every macroscopic
quadrature point.

Consider the macro problem at a single Gauss point. The macro solver hands us a
deformation gradient **F**. We must:

1. **Apply F to the RVE.** Specifically, drive the RVE's displacement field so
   that the volume-averaged deformation gradient equals F.
2. **Solve equilibrium on the RVE.** Equilibrium under whatever constitutive
   law lives in the RVE (linear elastic, neo-Hookean, crystal plasticity, …).
3. **Read back homogenized stress.** ⟨P⟩_Ω = (1/V) ∫_Ω P dV gives the macro
   first Piola-Kirchhoff stress to send back to the macro solver.
4. **Read back homogenized tangent.** ⟨∂P/∂F⟩_Ω. Required for Newton at the
   macro level.

Step 1 is where PBC enters. Three requirements pin down what "apply F" means:

- **Average theorem.** ⟨F⟩_Ω = F_macro. By Hill-Mandel, this requires either
  (a) prescribed displacement u = F·X on ∂Ω, or
  (b) prescribed traction t = F^{-T}·N on ∂Ω, or
  (c) Ω-periodic boundary conditions where u(X⁺) − u(X⁻) = (F − I)·(X⁺ − X⁻).
- **Periodicity is the canonical choice.** It minimizes the geometric stiffness
  artefact of the boundary, gives physically meaningful effective properties,
  and is the choice both Lopes (2021) and Miehe (2003) advocate.
- **Decomposition.** Write u = u_lin + ũ where u_lin = (F − I)X. By
  construction, periodicity of ũ — i.e. ũ(X⁺) = ũ(X⁻) — is equivalent to
  the periodic jump condition on u above.

The fluctuation ũ is what the FE solver actually computes. The art is in
discretizing the periodicity constraint on ũ, especially when the meshes on
opposite faces do not match. **That's what the mortar method buys us.**

Why non-matching meshes matter:

- For axis-aligned hex/quad meshes that we generate ourselves, opposite faces
  match by construction, and "node-coupled PBC" works (literally identify TDOFs
  on opposite-face node pairs).
- But for any geometry generated by a meshing tool (NETGEN, gmsh, Tetgen) on a
  general RVE, the face meshes won't match. A naive PBC implementation fails
  silently (or worse: it accepts the mismatch as a valid pair and produces
  wrong answers).
- Mortar methods enforce the coupling *integrally*: ∫_Γ ψ ⊗ (ũ⁺ − ũ⁻) ds = 0
  for all test functions ψ in some space. The space of choice is a *dual basis*
  (Wohlmuth) — see §4.

A working PBC implementation must:

1. Identify the periodic boundary pairs (corner/edge/face geometric structure).
2. Build a constraint matrix C such that C·u_total = 0 enforces ũ
   periodicity, with appropriate handling of crosspoints.
3. Pin enough modes to remove rigid-body translation (4 corners × 2 components
   in 2D = 8 essential TDOFs; 8 × 3 = 24 in 3D).
4. Embed C·u = 0 into the BVP — typically as a Lagrange-multiplier saddle-point
   system.
5. Pass the patch test exactly.
6. Reproduce ⟨F⟩ = F_macro to machine precision (volume-averaged-F
   diagnostic).
7. Solve scalably, not just on toy meshes.

The prototype satisfies (1)-(6) in 2D for both conforming and intentionally
non-matching meshes, with linear elasticity. (7) is in scope for the C++ port.

---

# §2. Two formulations: Method C vs Method D, and why we use D

This is the most-misunderstood point in the literature, where carelessness
during implementation produces silent errors that *only* show up as ⟨F⟩
deviating from F_macro by some O(1) amount. Both methods are well-defined and
mathematically valid; they differ in *which displacement field is the unknown*
and consequently in *what the Dirichlet and constraint conditions look like*.
Lopes (2021) §3.3 enumerates them as Methods A through D; we summarize C and D
because those are the only two relevant for our prototype.

## §2.1 Method C: solve for the fluctuation directly

**Primal:** ũ (the periodic fluctuation).

**System:**

- Unknown: ũ on Ω.
- Equilibrium (linear-elastic case for clarity):
  K_uu·ũ + K_uc·ũ_c = − K_uu·u_lin − K_uc·u_lin,c   on free DOFs
- Essential BC: ũ_c = 0 at the chosen pinning corners.
- Constraint: C·ũ = 0 (mortar periodicity of the fluctuation).

After solving, total displacement is u = u_lin + ũ.

In Method C the corner Dirichlet is "ũ = 0 at corners" — *not* u = u_lin at
corners. The affine field u_lin is a known offset that's never an unknown.

**When Method C is convenient:** when the FE infrastructure naturally treats
ũ as the field (e.g. if the user wrote a separate FE assembly that takes u_lin
as a fixed body-force-like contribution and solves only for ũ).

**When Method C is awkward:** standard FE codes (MFEM, libMesh, deal.II) work
on the *total* displacement field. Method C requires special handling to avoid
double-counting u_lin.

## §2.2 Method D: solve for the total displacement, with corners pinned at u_lin[corner]

**Primal:** u (the total displacement).

**System:**

- Unknown: u on Ω.
- Equilibrium: K·u = 0  (no body force in our setting).
- Essential BC: u_c = u_lin[corner] = (F − I)·X_corner at the chosen pinning corners.
- Constraint: a periodicity condition that, after corner BC, produces the
  correct ũ-periodic answer.

In Method D the corner Dirichlet *is* the affine-corner-displacement: when we
say "corners pinned", we mean u(X_corner) = (F − I) X_corner exactly.

**Initial iterate:** ũ⁰ = 0, so u⁰ = u_lin everywhere. The Newton step solves
for du = u_tilde with C·du = 0 (a fluctuation-periodicity reading) and total u = u_lin + du.

This is the convention Lopes uses (his Remark 1, line 342: "The linear
displacement part is applied to the entire RVE domain in the first stage as an
initial guess"). It maps cleanly to ExaConstit's formulation, where the primal
is the full kinematic state and Dirichlet BCs are applied at their full
prescribed values, not as deltas.

## §2.3 Why we picked Method D (and what's subtle about it)

Method D is what works inside MFEM's `ParBilinearForm` / `ParNonlinearForm`
infrastructure without painful workarounds. The total field is the natural
unknown; standard `EliminateRowsCols` handles the corner Dirichlet; the
constraint matrix C couples *fluctuation* DOFs (which after corner elimination
are the only thing the constraint sees).

The subtlety:

1. **C operates on the fluctuation, but the primal is the total.** This sounds
   trivial but caused a real bug. When we compute the right-hand side of the
   linear solve, we want `r1 = K·u_lin` (with corner entries zeroed). After
   corner elimination, the eliminated K has zero columns at the corner
   positions, so `K_eliminated·u_lin` *loses* the K_uc·u_lin[corner] term that
   couples free rows to corner displacements. **Use the full (un-eliminated) K
   to compute r1, then zero corner entries of r1.** See §6.4 and the §12 trap
   list. Forgetting this gives the patch test the appearance of working
   (Krylov converges, constraint residual is small, SciPy direct cross-checks
   match — but they all match the *wrong* answer, with free DOFs collapsing
   toward zero instead of following u_lin).

2. **The constraint as seen by the saddle-point solve has corners zeroed
   out.** The corner cols of C are zeroed by `apply_dirichlet_zero_to_C`,
   because the corner DOFs are essential and shouldn't appear in the
   constraint. (After corner elimination from K, those columns of the saddle-
   point top block would be zero anyway; we zero C's cols defensively.) This
   places us in a Method-C reading at the constraint level — `C·du = 0` —
   while the primal-level interpretation is Method D — `u_total = u_lin + du`.
   The two readings are equivalent modulo the affine offset; the implementation
   is consistent as long as both halves agree on the sign convention.

3. **What changes between load steps.** In a multi-step ramp F^{n+1} ≠ F^n,
   the *corner displacements* change because u_lin = (F−I)X changes. The
   prescribed-Dirichlet values for the corners thus shift step-to-step. Hence
   the warm-start projection (§7) has to handle a "Δu at the essential
   corners" injection — which is exactly the pattern ExaConstit's `SolveInit`
   handles for velocity primal; we translate it to displacement primal.

## §2.4 What killed the wrong RHS in the multi-step driver

The first multi-step driver implementation used `K_eliminated·u_lin` as the RHS
inside the driver class because the eliminated K was the only K the driver had
been handed. This produced answers where, in heterogeneous RVEs, free DOFs
appeared to be moving in the *opposite* direction of u_lin (the user spotted
the symptom in ParaView). The fix was to pass two K-handles into the driver:
`K_full` (un-eliminated, used for the RHS) and `K_eliminated` (used as the
saddle-point's top block). See §6.4 for the full derivation and §12 trap 11
for the bug description.

---

# §3. The mortar method — variational form, discrete construction, algorithm

The mortar method is the canonical weak-coupling FE technique for joining
non-matching meshes across an interface. We give the *minute version* first
(for orientation), then the continuous variational form (with citations
[Bernardi et al. 1994; Wohlmuth 2000, 2001]), then the discrete construction
that produces the rows of our constraint matrix C, and finally the explicit
geometric-matching algorithm in pseudocode.

## §3.1 The minute version

You have two interfaces Γ⁺ and Γ⁻ that should be identified periodically. Their
meshes don't match. You want a constraint that says *the displacement fields
agree on the interface in a weak sense*. Mortar method:

1. Pick the nonmortar (B, "−") side.
2. Choose a Lagrange-multiplier space Λ_h on the nonmortar side. Each basis
   function μ_i ∈ Λ_h corresponds to one row of the constraint matrix C.
3. Build C row-by-row by computing ∫_{Γ⁻} μ_i · (u⁺ − u⁻) ds, expressed in
   terms of mortar / nonmortar FE shape functions.
4. The whole interface then gets one row per nonmortar-side multiplier DOF per
   spatial component. C has (#LM rows) columns equal to (#displacement TDOFs)
   and a sparsity pattern that's local to each nonmortar-side element plus its
   mortar-side image.

After C is built, embed the constraint into the BVP via Lagrange multipliers:
[[K, Cᵀ], [C, 0]] [u; λ] = [b; 0]. (See §6.)

## §3.2 The continuous variational form

Let Ω be the RVE domain with boundary ∂Ω. Periodicity identifies pairs of
opposite parts of ∂Ω; for each pair, denote the two halves by Γ⁺ (mortar /
"plus" side) and Γ⁻ (nonmortar / "minus" side). The periodic mapping
Π : Γ⁻ → Γ⁺ relates the geometric image of each nonmortar point to its mortar
counterpart. For an axis-aligned cube of side L, Π is a pure translation by
±L along the appropriate coordinate axis.

The continuous fluctuation-periodicity condition reads, in strong form,

    ũ(X) = ũ(Π(X)),    X ∈ Γ⁻.                                (3.1)

This is what we want to enforce, but it is too strong to hold pointwise on a
mesh whose Γ⁻ and Γ⁺ traces don't match. The mortar method weakens (3.1) by
testing it against a Lagrange-multiplier space Λ ⊂ [L²(Γ⁻)]^d (one component
per spatial dimension d). The weak form is

    ∫_{Γ⁻} μ · ( ũ ∘ Π − ũ|_{Γ⁻} ) ds = 0    ∀ μ ∈ Λ.          (3.2)

When (3.2) holds for every μ in a sufficiently rich Λ, the difference
ũ ∘ Π − ũ|_{Γ⁻} is L²(Γ⁻)-orthogonal to Λ. The discrete choice of Λ_h ⊂ Λ
determines exactly *which* discrete projection of (3.1) is enforced; this
choice is the methodological lever the mortar method gives us.

The full RVE BVP, in mixed Lagrange-multiplier form, is then [Lopes et al.
2021, §3.2]:

> Find (u, λ) ∈ V × Λ such that
>
>     a(u, v) − ⟨λ, [v]⟩_{Γ⁻}  = ⟨f, v⟩      ∀ v ∈ V          (3.3a)
>     ⟨μ, [u]⟩_{Γ⁻}            = 0           ∀ μ ∈ Λ          (3.3b)
>
> where:
>
> - V is the FE space (with corner Dirichlet BCs imposed strongly),
> - a(u, v) is the bilinear form of the elasticity problem
>   (a(u, v) = ∫_Ω σ(u) : ε(v) dV in the linear-elastic case),
> - [v] := v ∘ Π − v|_{Γ⁻} is the periodic jump on Γ⁻,
> - ⟨·,·⟩_{Γ⁻} is the L²(Γ⁻) duality pairing.

Equation (3.3a) is the equilibrium with the constraint reaction Cᵀλ
appearing on the LHS. Equation (3.3b) is the (weak) periodicity. Together
they give the saddle-point system [[K, Cᵀ], [C, 0]] of §6.

## §3.3 The discrete formulation: deriving the rows of C

Discretize V with the standard FE space V_h (continuous H¹ piecewise
polynomials, vector-valued, vdim = d). On Γ⁻ the trace of V_h has shape
functions {N_j^⁻}; on Γ⁺ the trace has {N_k^⁺}. Choose Λ_h spanned by
multiplier basis functions {μ_i} on Γ⁻ — for the dual-basis mortar method
these are the *dual* of {N_j^⁻} (see §4 for the explicit construction).

Substituting u_h = ∑ N_j^⁻ u_j^⁻ + ∑ N_k^⁺ u_k^⁺ + (interior-only DOFs) into
(3.3b):

    ⟨μ_i, u_h ∘ Π − u_h|_{Γ⁻}⟩
    = ∑_k ( ∫_{Γ⁻} μ_i (N_k^⁺ ∘ Π) ds ) u_k^⁺
    − ∑_j ( ∫_{Γ⁻} μ_i N_j^⁻ ds ) u_j^⁻
    = 0.                                                       (3.4)

Define two element-level matrices:

    D_{ij} := ∫_{Γ⁻} μ_i N_j^⁻ ds                              (3.5a)
    A^m_{ik} := ∫_{Γ⁻} μ_i (N_k^⁺ ∘ Π) ds                      (3.5b)

D is the *nonmortar-side mass matrix* against the multiplier basis. A^m
("mortar matrix") is the mortar-side coupling: it integrates the
multiplier μ_i (defined on Γ⁻) against the mortar shape function N_k^⁺
evaluated at Π(X) (the periodic image of the nonmortar point X).

The discrete form of (3.3b) is then, in matrix-vector notation,

    A^m · u^⁺ − D · u^⁻ = 0,                                   (3.6)

per spatial component. Each component (x, y, z) gets its own copy of
(3.6); the constraint for a vector-valued field stacks them block-
diagonally.

The full constraint matrix C is built by assembling the contributions from
all nonmortar-side elements:

    C = [ −D | A^m | 0 | … ]                                   (3.7)

where the columns are organized as [nonmortar-side DOFs | mortar-side DOFs |
interior DOFs]. The interior DOFs have zero entries (the constraint
involves only boundary values). The signed structure says: the constraint
row enforces (mortar-side LM-weighted) = (nonmortar-side LM-weighted), i.e.
A^m u^⁺ = D u^⁻ from (3.6).

**Why dual basis matters here.** If we choose the multiplier space
Λ_h = trace(V_h) — the standard mortar method [Bernardi et al. 1994] —
then μ_i = N_i^⁻, and D becomes the nonmortar-side FE mass matrix (full,
banded, not diagonal). The Schur complement C diag(K)⁻¹ Cᵀ is then
dense within the nonmortar-side support. If we instead choose Λ_h to be
*biorthogonal* to {N_j^⁻} on Γ⁻ — Wohlmuth's dual mortar approach
[Wohlmuth 2000] — then by construction D is diagonal, and inversion in
(3.6) (or condensation of λ from the saddle-point system in §6) becomes
element-local. This is the architectural payoff for the dual basis.

## §3.4 Standard mortar vs dual-basis mortar

Two flavours:

- **Standard mortar** [Bernardi, Maday & Patera 1994]: Λ_h = trace(V_h)
  modulo boundary conditions. The matching condition (3.4) becomes a
  global linear system involving the nonmortar-side FE mass matrix D. Optimal
  a priori error estimates O(h^{p+1}) for p-th order FE. Schur complement
  is dense and ill-conditioned in 3D.

- **Dual-basis mortar** [Wohlmuth 2000, 2001]: Λ_h is the dual basis,
  bi-orthogonal to {N_j^⁻} on Γ⁻, supported in only a few elements. D is
  diagonal. C·M⁻¹·Cᵀ becomes sparse and banded, with bandwidth equal to
  the multiplier-mortar coupling support. Same a priori error estimates as
  standard mortar [Wohlmuth 2000, Theorem 4.1].

We use dual-basis mortar throughout. The dual basis is what makes the
multiplier-block elimination tractable in 3D and is the right starting point
for the eventual ExaConstit production solver. The construction generalises
to triangles and tetrahedra (see §4.4–§4.5) and to higher-order elements
[Lamichhane & Wohlmuth 2002; Popp et al. 2012].

## §3.5 Geometric matching: nonmortar quadrature → mortar interpolation

The hardest geometric piece is the realisation of the integral in (3.5b).
For each nonmortar-side element (line segment in 2D, quad-4 or tri-3 face in
3D), the basic algorithm is:

```
for each nonmortar-side element S in Γ⁻:
    fe_S = nonmortar element shape data (N_j^⁻, dual basis μ_i, parametric domain)
    place a Gauss quadrature rule {(ξ_q, w_q)} on S's reference domain
    for each Gauss point q:
        x_q = nonmortar element transformation T_S(ξ_q)            # physical point
        x_mortar = Π(x_q)                                       # periodic image
        find mortar element M containing x_mortar
        compute ξ_mortar = inverse transformation T_M⁻¹(x_mortar)
        evaluate nonmortar dual basis μ_i(ξ_q) for i in nonmortar-LM DOFs
        evaluate nonmortar shape N_j^⁻(ξ_q)        for j in nonmortar DOFs
        evaluate mortar shape N_k^⁺(ξ_mortar)  for k in mortar DOFs
        |J_S| = element Jacobian determinant at ξ_q
        for i, j: D_local[i,j]   += w_q · |J_S| · μ_i(ξ_q) · N_j^⁻(ξ_q)
        for i, k: A^m_local[i,k] += w_q · |J_S| · μ_i(ξ_q) · N_k^⁺(ξ_mortar)
    assemble D_local into D (global, with appropriate row/column TDOF maps)
    assemble A^m_local into A^m
```

Two key properties of this algorithm:

1. **Quadrature is on the nonmortar element's reference domain.** All FE
   shape and dual-basis values are evaluated at nonmortar-element parametric
   points. The mortar is *evaluated* at the projected point, not
   integrated against.

2. **The integration domain is the nonmortar element**, not its intersection
   with the mortar. The variational form (3.4) integrates over Γ⁻ in its
   entirety; even if a nonmortar element overlaps multiple mortar elements
   (non-conforming case), each Gauss point is processed individually with
   its own mortar-element lookup. We do *not* need polygon-clipping in
   the algorithm above — quadrature on the nonmortar reference suffices for
   any non-conforming pair, conforming or otherwise.

   *Caveat for sub-element accuracy:* if a nonmortar element is much larger
   than the mortar elements it overlaps, a single Gauss rule on the
   nonmortar may not resolve the mortar-side discontinuities (jumps in
   ∇N_k^⁺) at element boundaries. In that case the integration must be
   *sub-divided* at the mortar-element boundaries — this is where
   Sutherland-Hodgman polygon clipping enters (§3.7). For our 2D
   prototype we use a sufficient-order quadrature on the un-clipped
   nonmortar element, which is acceptable when the meshes have comparable
   refinement; for production 3D this will need clipping.

   *The D-vs-A^m domain split (important).* When we do sub-divide for
   the non-conforming case, the integration domain depends on which
   matrix entry we're computing:

   - **D contributions (`D_kk = ∫_Γ⁻ μ_k N_k⁻ dA`)** are accumulated PER
     NONMORTAR ELEMENT, with the integration domain being the FULL
     nonmortar element. They depend only on nonmortar-element shape data
     — there is no mortar-side input, hence no need to know which sub-
     polygon any quadrature point falls into. Computing D directly on
     the full element (`D_k = ∫_E N_k dA`, exploiting the dual-basis
     biorthogonality identity that lumps μ_k against N_k) avoids
     compounding rounding error and is computationally cheaper.
   - **A^m contributions (`A^m_kl = ∫_Γ⁻ μ_k (N_l⁺ ∘ Π) dA`)** are
     accumulated PER CLIPPED OVERLAP, with the integration domain being
     the OVERLAP polygon (a sub-region of the nonmortar element). They
     require evaluating the mortar-side shape function `N_l⁺` at the
     projected point, which only makes sense within a specific mortar
     element. Each overlap polygon is fan-triangulated and quadratured
     per sub-triangle.

   Why this split is correct: Wohlmuth's biorthogonality identity
   `∫_E μ_i N_j dE = δ_ij ∫_E N_i dE` holds when integrated over the
   FULL nonmortar element E, NOT segment-wise. So we compute D directly
   as `∫_E N_i` (a cheap element-local quadrature) rather than as
   `∑_segments ∫ μ_i N_i` (which would compound rounding error and
   requires summing all overlapping segments correctly).

   The 2D code in `mortar_pbc/mortar_2d.py` implements this split (D
   per full nonmortar segment, A^m per overlap segment) and the C++
   port in `mortar_assembler_2d.cpp` mirrors it. The 3D non-conforming
   port (Phase 3.5 / Phase 4.4) extends the same pattern.

For axis-aligned periodic boundaries (our case), the geometric matching
simplifies dramatically:

- **2D**: a nonmortar point at (x, 0) maps via Π to (x, L). Local search on
  the mortar is a 1D parameter-space search along the y = L edge.
- **3D**: a nonmortar point on the y = 0 face at (x, 0, z) maps to (x, L, z).
  Two-parameter (ξ, η) search on the mortar quad face (or barycentric
  search on a mortar triangle face).

The current 2D code (`mortar_pbc/mortar_2d.py`) handles step 4 of the
algorithm via direct 1D parameter search. The 3D code (Phase 3.2–3.3)
needs the 2D analog. For *conforming* meshes in 3D, the mortar-element
lookup is by direct geometric indexing; for *non-conforming* (Phase 3.5)
it requires the AABB-tree-or-similar lookup plus the clipping subroutine.

## §3.6 The conforming "free-pass" case

When the nonmortar and mortar meshes match node-for-node on the periodic
interface, every nonmortar Gauss point lands on a mortar element such that
ξ_mortar = ξ_nonmortar (modulo the orientation of the parametric coordinate
on opposite faces). Then evaluating mortar shape functions N_k^⁺ at
ξ_mortar gives the same values as evaluating nonmortar shape functions
N_j^⁻ at ξ_nonmortar (same FE family, same parametric coordinate). For dual
basis with bi-orthogonality:

    D_{ii} = ∫_{Γ⁻} μ_i N_i^⁻ ds = (∫_{Γ⁻} N_i^⁻ ds)            (3.8a)
    A^m_{ik} = ∫_{Γ⁻} μ_i (N_k^⁺ ∘ Π) ds = (∫_{Γ⁻} N_i^⁻ ds) δ_{ik}  (3.8b)

(see §4.2 for why the bi-orthogonality gives a row-sum-equal-to-N-integral
structure). Hence after the row-scaling D⁻¹ implicit in (3.6), the
constraint reduces to

    A^m_{normalized} u^⁺ − u^⁻ = 0,    A^m_{normalized} = identity-with-sign-on-pair

i.e. one row per nonmortar DOF, with +1 on the nonmortar-DOF column and −1 on the
mortar-DOF column. This is the "lumped" or "node-coupled" PBC — the same
answer a hand-crafted node-pair-identification PBC would give.

The conforming case is therefore a useful *correctness baseline*: build a
trivially conforming RVE, check that C is exactly the signed-identity
structure (modulo Wohlmuth corner mods, §5), run the patch test.

The 2D `test_conforming_pair_recovers_lumping` unit test exists for
exactly this purpose. Phase 3.2 will need the 3D analog (one for
quad-face conforming pairs, one for tri-face conforming pairs).

## §3.7 Aside: Sutherland-Hodgman polygon clipping (Phase 3.5 preview)

For non-conforming face pairs in 3D where nonmortar-element / mortar-element
overlap is non-trivial, the integral (3.5b) must be sub-divided to capture
mortar-side basis discontinuities. Sutherland-Hodgman [Sutherland & Hodgman
1974] gives a robust convex-on-convex clipping algorithm, applicable to
quad-on-quad and tri-on-tri (and mixed) face overlaps:

```
function sutherland_hodgman_clip(subject_polygon, clip_polygon):
    # subject_polygon: vertices of the nonmortar element (in mortar-local coords)
    # clip_polygon  : vertices of one mortar element (assumed convex)
    output = subject_polygon
    for each edge (e1, e2) of clip_polygon:
        input = output
        output = []
        for each pair of consecutive vertices (s, p) in input:
            if p is inside_halfplane(e1, e2):
                if s is not inside_halfplane(e1, e2):
                    output.append(intersection(s, p, e1, e2))
                output.append(p)
            else:
                if s is inside_halfplane(e1, e2):
                    output.append(intersection(s, p, e1, e2))
        if output is empty: return []   # no overlap
    return output
```

The clipped polygon is then triangulated (fan-triangulation works for the
convex case) and Gauss quadrature is placed on each sub-triangle. The
mortar-element basis is evaluated at the projected sub-triangle Gauss
points, the nonmortar-element basis at the inverse-projected points. The
contributions accumulate into the same D and A^m as before.

This algorithm handles:
- **Quad nonmortar on quad mortar**: 4-on-4, both convex.
- **Tri nonmortar on tri mortar**: 3-on-3, both convex.
- **Mixed**: clip the nonmortar (3 or 4 vertices) by each mortar in turn.

Hand-rolling Sutherland-Hodgman for these cases is straightforward and
avoids the heavy `shapely` dependency. We defer the implementation to
Phase 3.5; conforming-mesh testing in Phases 3.1–3.4 doesn't need it.

---

# §4. The dual basis: derivation, simplex unification, and explicit formulas

The dual basis is the algebraic core of Wohlmuth's mortar method
[Wohlmuth 2000, §4.1]. This section derives it from first principles, then
gives the explicit formulas for the four element types we need:

| Element | Geometry | Volume / Face element of | Citation |
|---|---|---|---|
| **line-2** | 1D segment, 2 nodes | quad-4 / tri-3 (edge); 3D edge mortar | [Wohlmuth 2000; Lopes et al. 2021, Eq. C.1] |
| **tri-3** | 2D triangle, 3 nodes | tet-4 (face); also 2D simplex mesh | [Wohlmuth 2000, §4.1] |
| **quad-4** | 2D bilinear quadrilateral, 4 nodes | hex-8 (face) | [Lopes et al. 2021, Eq. C.3] |
| **tet-4** | 3D tetrahedron, 4 nodes | tet mesh (volume) | [Lamichhane & Wohlmuth 2007] |

ExaConstit users may run hex meshes (whose periodic faces are quad-4) or tet
meshes (whose periodic faces are tri-3); a single PBC implementation must
support both. Mixed meshes (some hex, some tet) are also allowed in MFEM and
the formulation must accommodate them on a face-by-face basis.

## §4.0 Derivation from the bi-orthogonality requirement

The defining property of the dual basis [Wohlmuth 2000, eq. 4.1]:

    ∫_E M_i N_j dE = δ_ij ∫_E N_j dE,    i, j = 1, …, n_loc          (4.1)

where E is a single boundary element (line in 2D, tri or quad in 3D) on the
nonmortar side, {N_j} are the standard FE shape functions, and {M_i} is the dual
basis we are constructing. The right-hand side is the *standard FE shape
function integral*, not the FE mass matrix entry — this is what makes the
dual basis "biorthogonal to N with respect to a diagonal target".

Constructive ansatz: write each M_i as a linear combination of the same
shape functions,

    M_i = ∑_j A_ij N_j,                                              (4.2)

where A is an n_loc × n_loc matrix to be determined. Substituting (4.2)
into (4.1):

    ∑_k A_ik ∫_E N_k N_j dE = δ_ij ∫_E N_j dE                         (4.3)

Define the **standard FE mass matrix** M^FE on E and the **shape integral
vector** s:

    M^FE_kj := ∫_E N_k N_j dE,    s_j := ∫_E N_j dE                   (4.4)

Then (4.3) becomes the matrix equation

    A · M^FE = diag(s),    so    A = diag(s) · (M^FE)⁻¹.              (4.5)

This is the algebraic core. Once we know M^FE and s for a given reference
element, we get A explicitly by inverting M^FE and right-multiplying by
diag(s). The dual basis is then just (4.2): each M_i is a linear combination
of the FE shape functions on the same element.

**Local support.** Each M_i is supported on exactly the same elements as
N_i — element-local, just like the FE basis [Wohlmuth 2000, Theorem 4.2].
This is why the discrete D matrix becomes diagonal: D_{ii} = s_i ≠ 0 by
(4.1), and D_{ij} = 0 for j ≠ i.

**Partition of unity.** A direct consequence of (4.1) and ∑_j N_j = 1 is:

    ∑_i M_i(x) = 1     ∀ x ∈ E.                                      (4.6)

Proof: at any x ∈ E, write the constant function 1 = ∑_j N_j(x). Then
∫_E (∑_i M_i) N_j dE = ∑_i ∫_E M_i N_j dE = s_j (one term, i = j survives by
(4.1)) = ∫_E N_j dE = ∫_E 1 · N_j dE. Since the {N_j} span all polynomials
of total degree 1 on simplices (or bilinear functions on hypercubes), and
since ∑_i M_i is in the same span, the equality of integrals against every
N_j forces ∑_i M_i = 1 pointwise. ∎

This partition-of-unity property is what guarantees *constant reproduction*
across non-conforming pairs: if ũ⁻ ≡ const on Γ⁻ and ũ⁺ ≡ const on Γ⁺, then
the constraint row ∫ μ_i (u⁺ ∘ Π − u⁻) ds = 0 is satisfied automatically.

## §4.1 Simplex unification: line-2, tri-3, tet-4

For a *d-dimensional simplex* (d=1: line; d=2: triangle; d=3: tetrahedron),
the standard P1 shape functions are the barycentric coordinates λ_1, …,
λ_{d+1}. The integrals (4.4) on the reference simplex of measure |E| are
[Strang & Fix 1973, §3.2]:

    ∫_E λ_i dE     = |E| / (d+1)                                      (4.7a)
    ∫_E λ_i² dE    = 2 |E| / [(d+1)(d+2)]                             (4.7b)
    ∫_E λ_i λ_j dE = |E| / [(d+1)(d+2)],   i ≠ j                      (4.7c)

So M^FE has the structure (M^FE)_ij = α + β δ_ij where

    α = |E| / [(d+1)(d+2)],     β = |E| / [(d+1)(d+2)].

That is, M^FE = α (1_(d+1) 1_(d+1)ᵀ + I), which has rank-1 plus identity
structure. Its inverse is computed by the Sherman-Morrison identity:

    (M^FE)⁻¹ = (1/α) · [I − (1/(d+2)) 1 1ᵀ].                          (4.8)

Combining with diag(s) = (|E| / (d+1)) I:

    A = diag(s) · (M^FE)⁻¹
      = [|E|/(d+1)] · (1/α) · [I − 1 1ᵀ / (d+2)]
      = (d+2) · [I − 1 1ᵀ / (d+2)]
      = (d+2) I − 1 1ᵀ                                                (4.9)

Therefore A_ii = d+1 (diagonal) and A_ij = −1 (off-diagonal). Substituting
back into (4.2):

    M_i = (d+1) N_i − ∑_{j≠i} N_j = (d+1) N_i − (1 − N_i) = **(d+2) N_i − 1**
                                                                      (4.10)

This single closed form covers all three simplex cases:

| d | Element | Formula | Verified at |
|---|---|---|---|
| 1 | line-2 | M_i = 3 N_i − 1 | §4.2 |
| 2 | tri-3 | M_i = 4 λ_i − 1 = 4 N_i − 1 | §4.4 |
| 3 | tet-4 | M_i = 5 λ_i − 1 = 5 N_i − 1 | §4.5 |

Equation (4.10) is much cleaner than the mixed forms in [Lopes et al. 2021]
and matches [Lamichhane & Wohlmuth 2007, eq. 3.4] for the linear simplex
case. The tensor product for hypercubes (line-2 ⊗ line-2 = quad-4, etc.)
does not collapse to (4.10); it is its own structure (§4.6).

## §4.2 The line-2 dual basis (1D simplex, d=1)

Reference element: ξ ∈ [−1, +1], measure |E| = 2.

Standard shape functions:

    N_1(ξ) = (1 − ξ) / 2,       N_2(ξ) = (1 + ξ) / 2                 (4.11)

By (4.10) with d=1:

    M_i(ξ) = 3 N_i(ξ) − 1                                            (4.12)

which gives explicitly

    M_1(ξ) = 3 · (1−ξ)/2 − 1 = (3 − 3ξ − 2) / 2 = (1 − 3ξ) / 2       (4.13a)
    M_2(ξ) = 3 · (1+ξ)/2 − 1 = (1 + 3ξ) / 2                          (4.13b)

This matches [Lopes et al. 2021, Eq. C.1] exactly. Verification by direct
integration (no factor of 1/2 mistakes — the line measure on [−1,1] is dξ):

    ∫_{−1}^{+1} M_1 N_1 dξ = ∫_{−1}^{+1} (1 − 3ξ)(1 − ξ) / 4 dξ
                           = (1/4) ∫_{−1}^{+1} (1 − 4ξ + 3ξ²) dξ
                           = (1/4) [2 − 0 + 2] = 1                   (4.14a)

    ∫_{−1}^{+1} M_1 N_2 dξ = (1/4) ∫_{−1}^{+1} (1 − 3ξ)(1 + ξ) dξ
                           = (1/4) ∫_{−1}^{+1} (1 − 2ξ − 3ξ²) dξ
                           = (1/4) [2 − 0 − 2] = 0                   (4.14b)

And ∫_{−1}^{+1} N_1 dξ = ∫_{−1}^{+1} (1−ξ)/2 dξ = 1, so ∫ M_1 N_1 = ∫ N_1
holds — the diagonal target value is the shape integral, as (4.1) requires.
Symmetric calculations confirm M_2.

The implementation in `mortar_pbc/mortar_2d.py`:

```python
def N_line2(xi: float) -> tuple[float, float]:
    """Standard line-2 shape functions on [-1, +1]."""
    return ((1.0 - xi) * 0.5, (1.0 + xi) * 0.5)

def M_line2_dual(xi: float) -> tuple[float, float]:
    """Lopes Eq. C.1 / Wohlmuth (2000) line-2 dual basis."""
    return ((1.0 - 3.0 * xi) * 0.5, (1.0 + 3.0 * xi) * 0.5)
```

Verified by `test_dual_basis_biorthogonality` to machine precision.

## §4.3 The quad-4 dual basis (2D hypercube, d=2 tensor product)

Reference element: ξ, η ∈ [−1, +1]², measure |E| = 4.

Standard shape functions (tensor product of line-2):

    N_1(ξ,η) = (1−ξ)/2 · (1−η)/2     (corner (−1,−1))                (4.15a)
    N_2(ξ,η) = (1+ξ)/2 · (1−η)/2     (corner (+1,−1))                (4.15b)
    N_3(ξ,η) = (1+ξ)/2 · (1+η)/2     (corner (+1,+1))                (4.15c)
    N_4(ξ,η) = (1−ξ)/2 · (1+η)/2     (corner (−1,+1))                (4.15d)

Tensor product dual basis [Lopes et al. 2021, Eq. C.3]:

    M_quad4_i(ξ,η) = M_line2_p(ξ) · M_line2_q(η)                     (4.16)

where (p, q) ∈ {(1,1), (2,1), (2,2), (1,2)} for i = 1, 2, 3, 4 respectively.

Bi-orthogonality follows from the 1D bi-orthogonality and Fubini's theorem:

    ∫∫ M_quad4_i N_quad4_j dξ dη
        = (∫ M_line2_p(ξ) N_line2_p'(ξ) dξ) · (∫ M_line2_q(η) N_line2_q'(η) dη)
        = δ_pp' · δ_qq'                                              (4.17)

where (p', q') indexes node j the same way (p, q) indexes node i. The
identity is δ_ij = δ_pp' δ_qq' modulo the corner-numbering convention.

Partition of unity: M_1 + M_2 + M_3 + M_4 = (M_1^line2(ξ) + M_2^line2(ξ)) ·
(M_1^line2(η) + M_2^line2(η)) = 1 · 1 = 1. ✓

Explicit form, expanding (4.16) for node 1:

    M_quad4_1(ξ,η) = ((1−3ξ)/2) · ((1−3η)/2)
                   = (1 − 3ξ − 3η + 9ξη) / 4                         (4.18)

The other three follow by sign changes.

## §4.4 The tri-3 dual basis (2D simplex, d=2)

Reference element: standard triangle in barycentric coordinates with
λ_1 + λ_2 + λ_3 = 1, measure |E| (= 1/2 on the unit triangle, but the
formula is element-area-normalised).

Standard shape functions: N_i = λ_i (i = 1, 2, 3).

By (4.10) with d=2:

    M_i(λ_1, λ_2, λ_3) = 4 λ_i − 1                                   (4.19)

Bi-orthogonality verification using (4.7):

    ∫_E M_1 N_1 dE = ∫_E (4 λ_1 − 1) λ_1 dE
                   = 4 ∫_E λ_1² dE − ∫_E λ_1 dE
                   = 4 · 2|E|/(3·4) − |E|/3
                   = 4 · |E|/6 − |E|/3
                   = 2|E|/3 − |E|/3 = |E|/3                          (4.20a)

And ∫_E N_1 = |E|/3 by (4.7a). Match: ∫ M_1 N_1 = ∫ N_1. ✓

    ∫_E M_1 N_2 dE = ∫_E (4 λ_1 − 1) λ_2 dE
                   = 4 ∫_E λ_1 λ_2 dE − ∫_E λ_2 dE
                   = 4 · |E|/[(3·4)] − |E|/3
                   = |E|/3 − |E|/3 = 0                               (4.20b)

✓ Symmetric for the other entries.

Partition of unity: M_1 + M_2 + M_3 = 4(λ_1 + λ_2 + λ_3) − 3 = 4 − 3 = 1. ✓

The implementation, planned for `mortar_pbc/mortar_3d.py` in Phase 3.2:

```python
def N_tri3(lam: tuple[float, float, float]) -> tuple[float, float, float]:
    """Standard tri-3 shape functions = barycentric coordinates."""
    return (lam[0], lam[1], lam[2])

def M_tri3_dual(lam: tuple[float, float, float]) -> tuple[float, float, float]:
    """Tri-3 dual basis: M_i = 4 N_i - 1.
    
    Reference: Wohlmuth (2000) Section 4.1; Lamichhane & Wohlmuth (2007) eq. 3.4.
    Cite: derived in MORTAR_PBC_ARCHITECTURE.md §4.4.
    """
    return (4.0 * lam[0] - 1.0, 4.0 * lam[1] - 1.0, 4.0 * lam[2] - 1.0)
```

## §4.5 The tet-4 dual basis (3D simplex, d=3)

Reference element: standard tetrahedron in barycentric coordinates with
λ_1 + λ_2 + λ_3 + λ_4 = 1.

Standard shape functions: N_i = λ_i (i = 1, 2, 3, 4).

By (4.10) with d=3:

    M_i(λ_1, …, λ_4) = 5 λ_i − 1                                     (4.21)

Bi-orthogonality verification using (4.7) with d=3, |E| = volume:

    ∫_E λ_i dE     = |E| / 4
    ∫_E λ_i² dE    = 2|E| / 20 = |E| / 10
    ∫_E λ_i λ_j dE = |E| / 20,   i ≠ j

So:

    ∫_E M_1 N_1 dE = 5 · |E|/10 − |E|/4 = |E|/2 − |E|/4 = |E|/4 = ∫ N_1  ✓
    ∫_E M_1 N_2 dE = 5 · |E|/20 − |E|/4 = |E|/4 − |E|/4 = 0              ✓

Partition of unity: M_1 + M_2 + M_3 + M_4 = 5(λ_1+λ_2+λ_3+λ_4) − 4 = 1. ✓

Match: [Lamichhane & Wohlmuth 2007, eq. 3.4] for the linear tet case.

This is the dual basis for **3D edge / face mortar on tet meshes**. A tet
volume element has 4 triangular faces; for face mortar between periodic
faces of a tet RVE, each nonmortar face is a tri-3 element and uses the §4.4
dual basis (`M_tri3_dual`). The tet-4 dual itself (4.21) is needed only
for *volume* mortar (e.g. cross-mesh patch coupling, not our PBC use case).
We document it here for completeness because it slots into the same
unified simplex formula, and because future ExaConstit features (e.g.
multi-block coupling on internal interfaces) may use it.

## §4.6 Hypercubes vs simplices: structural differences

| Property | Simplex (line-2 / tri-3 / tet-4) | Hypercube (quad-4 / hex-8) |
|---|---|---|
| Dual basis shape | M_i = (d+2) N_i − 1 | Tensor product M_line2 ⊗ … |
| Polynomial degree | Total degree 1 in λ_i | Multi-linear (degree 1 in each ξ_k) |
| Bi-orthogonality structure | Eq. (4.10) closed form | Eq. (4.16) tensor structure |
| Partition of unity | (4.6) by direct calculation | Tensor product of 1D version |
| 3D face element ↔ volume element | Tri-3 face ↔ tet-4 volume | Quad-4 face ↔ hex-8 volume |

For mixed meshes (some hex elements with quad-4 faces, some tet elements
with tri-3 faces), the dual basis is selected per-face: each face inherits
its dual basis from the face element type, not from the volume element.
The mortar assembler must therefore dispatch on `face.geom_type` and apply
the appropriate `M_*_dual` function. This polymorphism is straightforward
to encode in C++ via virtual function dispatch on `mfem::Element::Type`.

## §4.7 Why bi-orthogonal matters: condition number and Schur complement

The dual basis is more than algebraic decoration. The diagonality of D
in (3.5a) gives:

- **D⁻¹** is trivially the diagonal of reciprocals: D_{ii}⁻¹ = 1 / s_i.
- **C M^{−1} Cᵀ ≈ A^m D⁻¹ (A^m)ᵀ** structure: the Schur complement of the
  constraint block has a sparsity pattern dictated by A^m alone, not by
  D. Each LM row's nonzero pattern is its own A^m row's nonzero pattern.
- **Static condensation** of λ becomes a sparse operation: solving D λ =
  rhs is element-local, no global matrix-matrix multiplication.

For our prototype's saddle-point Krylov path, this matters less directly
(we keep λ as an unknown in the saddle-point system), but the diagonal
block-Jacobi preconditioner on the multiplier block exploits exactly this
structure: diag(C diag(K)⁻¹ Cᵀ) is computed via `WeightedRowSqSum` on the
C operator (see §6.3), which is parallel-safe and works because of the
predictable sparsity that the dual basis induces.

For the eventual production solver, especially at 3D scale and especially
under mesh refinement, dual-basis mortar is the only practical choice.
Standard mortar [Bernardi et al. 1994] gives a non-diagonal D and a much
denser Schur complement, which scales poorly. See [Wohlmuth 2000, §5;
Wohlmuth 2001, Ch. 1] for detailed condition-number analyses.

## §4.8 Higher-order: the line-3 dual basis (1D, p = 2)

In one dimension, the strict bi-orthogonal dual basis exists *at all
orders* p ≥ 1, and is given by an explicit closed form. We work out the
quadratic case (line-3) explicitly because (a) it's the foundational 1D
piece needed by 2D quad-9 / serendipity quad-8 face mortar via tensor
product, (b) it shows the construction (4.5) generalising cleanly when
the lumped diagonal is positive, and (c) it sets up the 2D obstruction
in §4.9 by contrast.

Reference element: ξ ∈ [−1, +1], measure |E| = 2.

Standard Lagrange shape functions for the 3-node line element
(corner nodes at ξ = ∓1, mid-node at ξ = 0):

    N_1(ξ) = ½ ξ (ξ − 1)         (left corner)                       (4.22a)
    N_2(ξ) = ½ ξ (ξ + 1)         (right corner)                      (4.22b)
    N_3(ξ) = 1 − ξ²              (mid-node)                          (4.22c)

The shape integrals over [−1, +1] (these are the `s` vector of (4.4)):

    s_1 = ∫_{−1}^{+1} N_1 dξ = 1/3      (positive)                   (4.23a)
    s_2 = ∫_{−1}^{+1} N_2 dξ = 1/3      (positive)                   (4.23b)
    s_3 = ∫_{−1}^{+1} N_3 dξ = 4/3      (positive)                   (4.23c)

The fact that *all* three are positive is what makes the strict
bi-orthogonal dual exist — see §4.9 for why. The FE mass matrix:

    M^FE = (1/15) · ⎡ 4  −1   2 ⎤
                   ⎢−1   4   2 ⎥                                     (4.24)
                   ⎣ 2   2  16 ⎦

By (4.5), A = diag(s) · (M^FE)⁻¹. Computing (M^FE)⁻¹ and the product
[Lamichhane & Wohlmuth 2002, eq. 3.1]:

    Φ_1(ξ) = (5/24)(5ξ² − 2ξ − 1)    (peak at left corner)           (4.25a)
    Φ_2(ξ) = (5/24)(5ξ² + 2ξ − 1)    (peak at right corner)          (4.25b)
    Φ_3(ξ) = (5/12)(3 − 5ξ²)         (peak at mid-node)              (4.25c)

**Verification.** ∫ Φ_1 N_1 dξ = ∫ (5/24)(5ξ² − 2ξ − 1) · ½ ξ(ξ − 1) dξ
expanding and integrating term-by-term over [−1, +1] yields exactly 1/3
= s_1, and ∫ Φ_1 N_2 dξ = 0 = ∫ Φ_1 N_3 dξ. Symmetric for Φ_2, Φ_3.
Strict bi-orthogonality, no relaxation. ✓

Partition of unity: Φ_1 + Φ_2 + Φ_3 = (5/24)(5ξ² − 2ξ − 1)
+ (5/24)(5ξ² + 2ξ − 1) + (5/12)(3 − 5ξ²) = (5/24)(10ξ² − 2)
+ (5/12)(3 − 5ξ²) = (50/24)ξ² − 10/24 + 15/12 − (25/12)ξ²
= (25/12)ξ² − (25/12)ξ² + (15 − 5)/12 = 1. ✓

A subtlety not visible in the linear case: **the dual basis Φ_i is
discontinuous across element boundaries** [Lamichhane & Wohlmuth 2002,
Remark 3.2]. The basis is locally supported (one element of support per
basis function) but its values at element-end nodes from adjacent
elements differ. This is harmless for the mortar saddle-point system —
the LM is an L² object on the nonmortar interface, not an H¹ object — but
it forecloses some smoothness-based stabilisation strategies. To recover
*continuity* without sacrificing strict bi-orthogonality, one applies a
quartic `g(t) ∈ P_4([0,1])` correction satisfying g(t) = −g(1−t),
g(1) = 1, ∫₀¹ g · p dt = 0 ∀ p ∈ P_2 [Lamichhane & Wohlmuth 2002,
Lemma 3.5]. This `g` is one degree higher than the cubic correction
needed for P_1 elements precisely because we now require P_2
reproduction.

Tensor-product extension to 2D / 3D:

    Φ^{quad9}_{(i,j)}(ξ, η) = Φ^{line3}_i(ξ) · Φ^{line3}_j(η)        (4.26)
    Φ^{hex27}_{(i,j,k)}(ξ, η, ζ) = Φ^{line3}_i(ξ) · Φ^{line3}_j(η) · Φ^{line3}_k(ζ)
                                                                     (4.27)

These are the **closed-form, strictly bi-orthogonal** dual bases for
biquadratic and triquadratic Lagrangian tensor-product elements. They
slot into the same `M_*_dual` polymorphic dispatch as the linear cases,
with the only architectural change being `M_quad9_dual` returning a
9-tuple and `M_hex27_dual` returning a 27-tuple.

## §4.9 The bi-orthogonality obstruction at p ≥ 2 on simplices and serendipity elements

The construction (4.5) `A = diag(s) · (M^FE)⁻¹` *fails* for nodal P_p
Lagrange elements on simplices at p ≥ 2 and for Q^p serendipity elements.
The failure is algebraic, not numerical, and admits a clean general
statement.

### §4.9.1 The lumped-integral positivity criterion

**Proposition (lumped positivity).** *The strict bi-orthogonal,
locally-supported dual basis (4.5) exists iff the lumped diagonal
s_j = ∫_E N_j dE is nonzero for every shape function N_j.*

**Proof sketch.** Equation (4.1) reads ∫ M_j N_j = δ_jj · s_j = s_j on
the diagonal. If s_j = 0, the construction would force ∫ M_j N_j = 0,
which combined with the partition-of-unity ∑_i M_i = 1 yields a
contradiction: integrating the partition of unity against N_j gives
s_j on one side and ∑_i (∫ M_i N_j) = ∫ M_j N_j = 0 on the other (using
bi-orthogonality of off-diagonal terms). The two sides must agree, but
0 ≠ s_j unless we relax bi-orthogonality. Conversely, if all s_j > 0
(or uniformly nonzero with consistent sign), `diag(s) · (M^FE)⁻¹` is
well-defined and the resulting A has rows that integrate to 1. ∎

The lumped diagonal s_j is therefore the diagnostic: **compute s_j for
every shape function N_j on the reference element; if any vanishes,
strict bi-orthogonality with locally supported basis is impossible**.

### §4.9.2 What goes wrong on tri-6 (and tet-10, quad-8, hex-20)

For the **tri-6** element with corner shape function
N_1 = λ_1 (2λ_1 − 1) (Lagrange interpolant of degree 2, equal to 1 at
vertex 1 and 0 at the other 2 vertices and 3 mid-edges):

    s_1 = ∫_T λ_1 (2λ_1 − 1) dA
        = 2 ∫_T λ_1² dA − ∫_T λ_1 dA
        = 2 · (2|T|/12) − |T|/3        (using simplex integrals 4.7)
        = |T|/3 − |T|/3 = **0**                                       (4.28)

The corner-node lumped weight vanishes identically [Popp et al. 2012,
§3.2]. The obstruction is a topological-and-degree fact: the function
λ(2λ − 1) is symmetric about λ = ½ (the boundary midpoint between vertex
and opposite edge in the barycentric simplex), and its integral over
the half-simplex λ ≥ ½ exactly cancels its integral over λ < ½.

The same calculation gives, for **higher-dimensional simplices**, a
*dimension-dependent* result that we verify here in detail because the
quantitative pattern is different from what one might naively expect:

For a P_2 corner on a d-simplex (|T| = 1/d!):

    s_corner = 2 ∫ λ² − ∫ λ
             = 2 · (2!/(d+2)!) · d! · |T| − (1!/(d+1)!) · d! · |T|
             = ((4 / (d+2)!) − (1 / (d+1)!)) · d! · |T|
             = (4 − (d+2)) / (d+2)! · d! · |T|
             = (2 − d) / ((d+1)(d+2)) · d! · |T|/(d!)   wait, simplifying:
             = (2 − d) / ((d+1)(d+2)) · |T|   [after cleaning up]    (4.28b)

Plugging in d:
- **d=1 (line-3 corner)**: s = (2−1)/(2·3) · 2 = 1/6 · 2 = 1/3 > 0
  (matches §4.8 eq. 4.23a; the strict bi-orthogonal dual exists)
- **d=2 (tri-6 corner)**: s = (2−2)/(3·4) · |T| = 0
  (the boundary case; exactly on the threshold)
- **d=3 (tet-10 corner)**: s = (2−3)/(4·5) · |T| = −|T|/20 = **−1/120**
  (genuinely *negative*, not zero — the 2D claim above does not
  generalize to 3D)
- **d=4 and higher**: s = (2−d)/((d+1)(d+2)) · |T|, increasingly
  negative as d grows.

The 2D simplex therefore sits exactly on a knife-edge between the
1D-positive and 3D-negative regimes. This is sharper than the
classical "the higher-order simplex dual fails" statement: the sign
of the failure is dimension-dependent, and only in 2D does the corner
integral *vanish* exactly. In 3D it crosses to negative — making
tet-10 structurally similar to the serendipity case (next bullet),
not to the tri-6 case.

The other failing element types continue:

- **quad-8 (serendipity)** corner: ∫ N_corner = −|E|/12 [Lamichhane &
  Wohlmuth 2004, §3]. The serendipity basis has *no* central bubble
  to absorb the corrections, leaving each corner with a negative
  lumped diagonal that breaks bi-orthogonality more severely than the
  zero-valued tri-6 case.
- **hex-20 (serendipity)** corner: ∫ N_corner < 0 (same mechanism).

**Why does it not fail on the tensor-product full-Lagrangian
quad-9 / hex-27?** Because the central bubble (and edge-mid bubbles)
absorb mass that would otherwise leave the corner integrals zero or
negative. In barycentric language: the bilinear-times-bilinear
construction of quad-9 has corner shape function
N_1 = ¼ ξ(ξ−1) η(η−1), with ∫_{[-1,+1]²} = (1/3)(1/3) = 1/9 > 0, and
all 9 lumped weights positive. The full-tensor product *retains*
positivity per direction; serendipity loses it by removing the bubble.

### §4.9.3 The general pattern

Combining §4.9.1 with the explicit cases:

| Element type | Strict biorthogonal dual exists? | Why |
|---|---|---|
| **Q^p tensor-product** at any p (line-{p+1}, quad-{(p+1)²}, hex-{(p+1)³}, full-Lagrangian, including NURBS / B-splines) | **Yes** (closed-form via tensor product of 1D dual) | All s_j > 0; tensor structure preserves positivity |
| **P_1 simplex** (line-2, tri-3, tet-4) | **Yes** (eq. 4.10) | s_j = |E|/(d+1) > 0 |
| **P_p simplex at p ≥ 2 in 1D** (line-3, line-4, …) | **Yes** | All s_j > 0 always; line-3 explicit eq. 4.23 has s = (1/3, 1/3, 4/3) |
| **P_2 simplex in 2D** (tri-6) | **Boundary case: no** | s_corner = 0 *exactly* (eq. 4.28); the 2D simplex sits on the knife-edge between 1D-positive and 3D-negative regimes |
| **P_2 simplex in 3D** (tet-10) | **No** | s_corner = −|T|/20 = −1/120 (eq. 4.28b with d=3); negative, similar to serendipity rather than to tri-6 |
| **Q^p serendipity** (quad-8, hex-20) | **No** | Corner s_j < 0 (s_corner_quad8 = −|E|/12; s_corner_hex20 < 0 similarly) |
| **B-spline of degree p ≥ 1** | **Yes** when refined; non-trivial geometric mappings need parametric integration [Wunderlich et al. 2019, arXiv:1806.11535] | Knot-span structure preserves positivity |

The **dimension-dependent simplex pattern** for P_2 corner shapes
(eq. 4.28b) is:

    s_corner_P2 = (2 − d) / ((d+1)(d+2)) · |T|

with sign ∈ {+, 0, −} for d ∈ {1, 2, ≥3} respectively. This is sharper
than the textbook "higher-order simplices fail bi-orthogonality": only
the 2D simplex fails by *vanishing*; in 3D it fails by *flipping
sign*, making tet-10 quantitatively similar to the serendipity case
even though the barycentric-Lagrange shape functions have very
different structure.

This is the predictive rule: **check the lumped integrals s_j. If any
vanishes (P_2 simplex in 2D corners) or is negative (P_2 simplex in
3D+ corners; serendipity corners), strict bi-orthogonality fails and
a relaxation is required**.

The Lamichhane-Wohlmuth optimal-rate theorem [Lamichhane & Wohlmuth
2007, *Math. Comp.* 76, doi:10.1090/S0025-5718-06-01907-7] gives a
sharper sufficient condition for **polynomial-reproducing** (P_{p−1} ⊂
M_h) bi-orthogonal duals: the FE nodes must be **Gauss-Lobatto** spaced.
Equispaced Lagrange nodes (the default for tri-6, tet-10) give a
bi-orthogonal dual that loses one order of consistency; for quadratic
this is often invisible in practice but degrades for cubic+. See
[Oswald & Wohlmuth 2001].

### §4.9.4 Two relaxations: feasible and quasi-dual

When the strict construction fails, two well-developed relaxations
recover bi-orthogonality on a *modified* basis:

**Feasible dual basis** [Lamichhane & Wohlmuth 2007, §3].
The LM space M_h has **the same dimension** as the trace space
W_{0,h}, and strict bi-orthogonality holds between {M_i} and a
*modified* primal basis {Ñ_j} obtained by local element-wise
re-coupling. Polynomial reproduction (P_p ⊂ M_h) is preserved by
construction. Support enlargement is bounded (≤ 2p+1 elements in 1D
patches). This is the construction behind the Popp et al. 2012
basis-transformation procedure (§4.10).

**Quasi-dual basis** [Lamichhane, Stevenson & Wohlmuth 2005, *Numer.
Math.* 102, doi:10.1007/s00211-005-0636-z]. The LM dimension is
*relaxed*: dim M_h < dim W_{0,h}, with strict bi-orthogonality holding
only on a smaller index set I_h^δ ⊂ I_h. The polynomial reproduction
condition is preserved, the mortar coupling matrix D remains diagonal
on the active LM block (so static condensation works), but the loss
of dimension matching means some primal modes are not directly
constrained — the construction relies on a continuous-mortar argument
to ensure the missing modes are controlled by the active ones. This is
the natural relaxation for cubic+ tetrahedra and serendipity hex where
even the feasible construction would require unmanageable support
enlargements.

The user's project is well-served by the feasible variant for tri-6,
quad-8, quad-9; the quasi-dual is reserved for cubic+ tetrahedra (a
Phase-6+ scope item).

## §4.10 The Popp-Wohlmuth-Gee-Wall basis-transformation procedure

The most practical implementation of feasible higher-order dual bases —
used in BACI/4C, MOOSE, and the broader contact-mechanics literature —
is the **basis transformation** of [Popp, Wohlmuth, Gee & Wall 2012,
*SIAM J. Sci. Comput.* 34, B421–B446, doi:10.1137/110848190].

### §4.10.1 The recipe

For each nonmortar-side element with FE shape vector N (size n_loc), define
a per-element transformation T_e ∈ ℝ^{n_loc × n_loc} such that
Ñ = T_e · N has positive lumped integral at every node:

    s̃_j = ∫_E Ñ_j dE > 0     for all j.                              (4.29)

Then build the *feasible dual* on Ñ via the standard recipe (4.5):

    Ã_e = diag(s̃) · (M̃^FE)⁻¹    where M̃^FE_{ij} = ∫_E Ñ_i Ñ_j dE   (4.30)
    Φ_i = ∑_j Ã_{ij} Ñ_j                                              (4.31)

The full element-level transformation [Popp et al. 2012, eq. 37]:

    Φ = Ã_e · T_e · N = D̃_e · (T_e · M^FE · T_e^T)⁻¹ · T_e · N      (4.32)

This is "biorthogonal on Ñ but not on the original N" — which is what
*feasible* means.

### §4.10.2 Explicit transformation matrices

For each element type, Popp et al. 2012 specifies the transformation T_e
explicitly. The pattern is **redistribute mid-edge weight into the
adjacent corner nodes**, which in barycentric language is:

For **tri-6** [Popp et al. 2012, eq. 38]:

    Ñ_i^corner = N_i^corner + ½ ∑_{k ∈ E(i)} N_k^edge   (i = 1, 2, 3)
    Ñ_k^edge   = ½ N_k^edge                              (k = 4, 5, 6)
                                                                     (4.33)

where E(i) is the set of two edges adjacent to corner i. The
transformation matrix is then:

    T^tri6 = ⎡ 1   0   0   ½   0   ½ ⎤      ← corner 1 absorbs ½ of edges 4,6
             ⎢ 0   1   0   ½   ½   0 ⎥      ← corner 2 absorbs ½ of edges 4,5
             ⎢ 0   0   1   0   ½   ½ ⎥      ← corner 3 absorbs ½ of edges 5,6
             ⎢ 0   0   0   ½   0   0 ⎥      ← edge 4 keeps ½
             ⎢ 0   0   0   0   ½   0 ⎥      ← edge 5 keeps ½
             ⎣ 0   0   0   0   0   ½ ⎦      ← edge 6 keeps ½         (4.34)

After applying (4.30)–(4.31), the resulting feasible dual coefficient
matrix on Ñ is [Popp et al. 2012, eq. 39]:

    Ã^tri6 = ⎡ 3   0   0   0  −½  −½ ⎤
              ⎢ 0   3   0  −½   0  −½ ⎥
              ⎢ 0   0   3  −½  −½   0 ⎥
              ⎢ 0   0   0   1   0   0 ⎥                              (4.35)
              ⎢ 0   0   0   0   1   0 ⎥
              ⎣ 0   0   0   0   0   1 ⎦

Row-sums = 1 (partition of unity preserved). Bi-orthogonality:
∫ Φ_i Ñ_j = δ_ij · s̃_j on the modified basis. P_1 reproduction holds
(sufficient for optimal H¹ rate on quadratic elements).

For **quad-8 (serendipity)** [Popp et al. 2012, eq. 40], the pattern
is similar — each corner absorbs ¼ of each adjacent mid-edge — giving
the 8×8 transformation:

    Ã^quad8 = ⎡ 9/4   0    0    0   −¾   0    0   −¾ ⎤
               ⎢  0   9/4   0    0   −¾  −¾   0    0 ⎥
               ⎢  0    0   9/4   0    0  −¾  −¾    0 ⎥
               ⎢  0    0    0   9/4   0    0  −¾  −¾ ⎥                (4.36)
               ⎢  0    0    0    0    1    0   0    0 ⎥
               ⎢  0    0    0    0    0    1   0    0 ⎥
               ⎢  0    0    0    0    0    0   1    0 ⎥
               ⎣  0    0    0    0    0    0   0    1 ⎦

The corner row coefficient 9/4 (vs 3 for tri-6) reflects the different
weight distribution; the −¾ couples each corner to its two adjacent
mid-edges.

For **quad-9 (full Lagrangian)**, no transformation is required — the
dual basis is the strict tensor product (4.26) of the line-3 dual.

For **hex-20** (serendipity), the construction parallels quad-8 with
each corner absorbing ¼ of each of the three adjacent mid-edges; the
explicit 20×20 matrix is in [Popp et al. 2012, eq. 41].

For **hex-27** (full Lagrangian), tensor product (4.27) — strict
bi-orthogonality.

For **tet-10**, the dual basis lives on the tri-6 *face elements* of
the nonmortar-side surface, so the construction reduces to (4.34)–(4.35).

### §4.10.3 The crosspoint / wirebasket modification at higher order

The 1D Wohlmuth corner modification (§5.1) was "M_corner = 0, M_neighbor
= 1 on the end element". The higher-order generalisation is *more
delicate* because there are multiple boundary-adjacent shape functions
per element (corner + edge-midnodes) and partition-of-unity must be
preserved with **polynomial reproduction up to P_{p−1}**, not just
constants [Lamichhane, Stevenson & Wohlmuth 2005, §3.2].

For each boundary node n on the wirebasket ∂γ, the modification picks
an interior triangle Δ̃ ⊂ E with vertices ℓ_1^n, ℓ_2^n, ℓ_3^n at distance
comparable to diam(Δ̃), and computes the **barycentric coordinates**
σ_r^n of n with respect to Δ̃ (the unique solution of
∑_r σ_r^n p(ℓ_r^n) = p(n) for all p ∈ P_1). The modification is then:

    M_{ℓ_r}^mod ← M_{ℓ_r} + σ_r^n · M_n,    M_n^mod ← 0               (4.37)

Naive copy-paste of the linear-case formula (assigning weight 1 to a
single neighbor) loses the P_1 reproduction and degrades to suboptimal
rates — the barycentric weighting (4.37) is essential. This generalises
the §5.1 line-2 recipe (where there's only one "neighbor" so its
barycentric weight is trivially 1).

For **edge midnodes adjacent to face boundaries**, [Flemisch & Wohlmuth
2007] and [Popp et al. 2012, §3.3] specify an additional consistent
absorption: when an edge midnode lies on the wirebasket, its multiplier
weight folds into the *opposite* interior corner/edge node within the
same face element, with weights determined by the same P_{p−1}
reproduction condition. **Each element type / order combination
requires its own table of modifications**: the engineering literature
maintains explicit per-type code paths.

### §4.10.4 Convergence rates

For p-th order primal Lagrange FEs and the feasible dual mortar of
[Popp et al. 2012, Wohlmuth, Popp, Gee & Wall 2012, *Comput. Mech.* 49,
doi:10.1007/s00466-012-0704-z]:

| Quantity | Rate |
|---|---|
| Energy norm ‖u − u_h‖_{H¹(Ω)} | O(h^p) |
| L² norm ‖u − u_h‖_{L²(Ω)} | O(h^{p+1}) |
| LM in (H^{1/2}_{00})' norm | O(h^p) |

These match the standard mortar [Bernardi, Maday & Patera 1994]
rates — the dual relaxation costs no consistency. Quadrature must be
exact for at least degree 2p+1 to preserve the L² superconvergence;
segment-based integration (Puso-Laursen 2004) with 7-point Gauss on
triangles is standard for quadratic 3D contact.

## §4.11 The lower-order projection (LOR) fallback

For environments where implementing the §4.10 basis-transformation per
element type is too costly — and especially for the LLNL/MFEM
ecosystem, where this is the Tribol design choice — an attractive
alternative is to **build the constraint matrix at order 1 on a refined
boundary submesh**, leaving the volume problem at higher order. This is
the *lower-order refinement* (LOR) approach.

### §4.11.1 The geometric setup

Given a primal FE space V_h^{(p)} of order p ≥ 2 on a mesh T_h, the
**lower-order-refined boundary submesh** is constructed as follows:

```
function build_lor_boundary_submesh(pmesh, fes_p, periodic_attr):
    # Step 1: extract boundary submesh of periodic faces.
    psub = ParSubMesh.CreateFromBoundary(pmesh, periodic_attr)
    
    # Step 2: uniformly refine psub by p (= polynomial order of fes_p).
    # After refinement, the vertices of psub_lor coincide *exactly* with
    # the Lagrange nodes of order-p elements on the original boundary.
    psub_lor = psub.UniformRefinement(times=log2(p))   # symbolic; use p sub-divisions
    
    # Step 3: build order-1 LM space on the refined submesh.
    fec_lam = H1_FECollection(order=1, dim=psub_lor.Dimension())
    fes_lam = ParFiniteElementSpace(psub_lor, fec_lam, vdim=dim)
    
    return psub_lor, fes_lam
```

The crucial geometric property [Pazner & Kolev 2021, MFEM LOR docs]:

    {Lagrange nodes of P_p on T_h} = {vertices of T_{h/p} (uniform refine ×p)}
                                                                     (4.38)

For p = 2: a P2 line element has 3 nodes (corners + 1 midpoint), and
once-refined linear sub-elements have those same 3 vertices. A P2 quad
has 9 nodes (4 corners + 4 mid-edges + 1 centroid), and a 2×2-refined
quad has those same 9 vertices. A P2 hex has 27 nodes; a 2×2×2-refined
hex has those same 27 vertices. The Lagrange basis is *interpolatory*
at exactly the refinement vertices.

Consequence: any continuous P_p field u_h on the original boundary
admits a unique continuous *piecewise-linear* representation u_h^{LOR}
on the refined boundary mesh, with **identical nodal values** —
u_h(x_α) = u_h^{LOR}(x_α) for every Lagrange node x_α. The mapping is a
trivial bijection of coefficient vectors.

### §4.11.2 The constraint matrix on LOR

With V_h^{(p)} restricted to the periodic boundary giving u_h on Γ⁻
(the nonmortar side), and the LOR multiplier space Λ_h^{(1)} of order-1
piecewise-linears on T_{h/p}, the mortar form (3.4) becomes:

    ⟨μ_i, [u_h ∘ Π − u_h]⟩_{Γ⁻}
    = ∑_k (∫_{Γ⁻} μ_i (N_k^{+,(p)} ∘ Π) ds) u_k^+
    − ∑_j (∫_{Γ⁻} μ_i N_j^{−,(p)} ds) u_j^−
    = 0     ∀ μ_i ∈ Λ_h^{(1)}                                        (4.39)

The integrals are computed *exactly* (or to high quadrature order) on
the LOR refined mesh, with μ_i piecewise linear and N_k^{(p)} piecewise
of order p. The element-level matrices D and A^m have the same form as
(3.5) but with mixed-order shape functions.

The LM space is constructed using the **§4 linear dual basis** on the
refined LOR mesh — line-2, tri-3, or quad-4 dual depending on face
element type. **No higher-order dual derivation is needed.** The
linear bi-orthogonal dual on T_{h/p} satisfies (4.1) on each refined
sub-element:

    ∫_{E_{LOR}} M_i^{(1)} N_j^{(1),LOR} ds = δ_ij ∫_{E_{LOR}} N_j^{(1),LOR} ds
                                                                     (4.40)

where N_j^{(1),LOR} is the order-1 hat function on T_{h/p}. The
constraint matrix C is then assembled exactly as in §3, with the
nonmortar-side LM rows numbered by LOR-vertex and the displacement
columns numbered by P_p TDOFs of the original V_h^{(p)}.

### §4.11.3 Stability and convergence under LOR

The non-trivial point: pairing P_p displacement with P_1 multiplier
(the "p / 1" pairing) is **not automatically inf-sup stable**.
[Brivadis, Buffa, Wohlmuth & Wunderlich 2015, *CMAME* 284,
doi:10.1016/j.cma.2014.09.012]: "the p/(p−1) pairing is numerically
shown to be unstable" in the unmodified mortar formulation. The
instability manifests as cross-point oscillations in λ and a non-uniform
inf-sup constant, leading to suboptimal saddle-point errors:

    ‖u − u_h‖_{H¹} ≤ C · ε_primal + C · ε_LM
                  ≈ O(h^p) + O(h^{3/2})  (loses optimal rate at p ≥ 2)
                                                                     (4.41)

Three remediations exist in the literature, each with a different
trade-off:

**(R1) Stay with p / (p−1) but apply Belgacem-style cross-point
modification.** Zero out vertex shape functions and redistribute via
barycentric weights (the §4.10.3 generalisation). This recovers
inf-sup stability for the strict p/(p−1) pairing but keeps the LM at
order p−1, which for p=2 gives a P1 LM — the same order as our LOR
choice. Belgacem mod is geometric on the original mesh; LOR is geometric
on the refined mesh. Algebraically related, distinct in practice.

**(R2) Use the p / (p−2) pairing.** For elasticity p=2 this gives P2/P0
constant LM, provably inf-sup stable but suboptimal in λ approximation.
Generally unsuitable for elasticity due to volumetric locking concerns.

**(R3) Add a Barbosa-Hughes-type residual stabilisation term to the
saddle-point block.** [Acharya & Patel 2019, arXiv:1705.10519;
Gustafsson, Råback & Videman 2022, arXiv:2209.02418,
"Mortaring for linear elasticity using mixed and stabilised finite
elements"]. The stabilised mortar form replaces (3.3a)–(3.3b) with:

    a(u, v) − ⟨λ, [v]⟩ + γ_β ∑_E h_E ⟨λ − Π_h(E_b u), μ − Π_h(E_b v)⟩_E = ⟨f, v⟩
                                                                     (4.42a)
    ⟨μ, [u]⟩ + γ_β ∑_E h_E ⟨…⟩ = 0                                   (4.42b)

with a stabilisation parameter γ_β = O(1/(λ + 2μ)) (mesh-independent;
material-dependent), h_E the local element size, and Π_h(E_b ·) a
projection of the elasticity edge-flux. The added bilinear term gives
an additional "penalty-like" coupling that restores inf-sup stability
for *any* L²-conforming multiplier including P1 LM on P2 displacement.
**For RVE-PBC homogenisation, where the jump-error dominates the
quantities of interest (effective tangent moduli), route R3 is the most
pragmatic** — it adds one new integrator to the existing assembly
pipeline and recovers quasi-optimal convergence.

For the LOR pairing in particular, the LOR refinement *also* improves
the inf-sup constant by reducing the "LM space too coarse" effect: the
LM on T_{h/p} has more DOFs than the LM on T_h would have at the same
order. For p=2 the LOR LM has the *same* DOF count as a P_2 LM on T_h
— LOR is "P1 on a refined mesh" not "P1 on the original". The cross-
point issue is genuinely there but is locally bounded; published
homogenisation studies report effective tangent moduli converging at
the bulk rate even with mismatched-order LM, provided the saddle point
is well-posed (i.e. the cross-point modification or stabilisation is
in place).

### §4.11.4 The MFEM mechanics

A single ParMesh can carry both a P2 displacement FES and a P1 LM FES on
a refined ParSubMesh — polynomial order is a property of the FES, not
the Mesh [MFEM `fem/fe_coll.hpp`]:

```cpp
// Volume FES at order 2.
auto *fec_u = new H1_FECollection(2, dim);
auto *fes_u = new ParFiniteElementSpace(&pmesh, fec_u, dim,
                                          Ordering::byVDIM);

// LOR boundary submesh + order-1 LM FES.
ParSubMesh psub = ParSubMesh::CreateFromBoundary(pmesh, periodic_bdr_attr);
psub.UniformRefinement();   // refine once for p=2; twice for p=3 (= p subdivisions)
auto *fec_lam = new H1_FECollection(1, psub.Dimension());
auto *fes_lam = new ParFiniteElementSpace(&psub, fec_lam, dim);

// Mixed-order constraint matrix.
ParMixedBilinearForm Cmat(fes_u, fes_lam);
Cmat.AddTraceFaceIntegrator(new MortarConstraintIntegrator(M_line2_dual));
Cmat.Assemble();
```

The crucial properties:

- `H1_Trace_FECollection` is **not** required — ParSubMesh handles the
  trace geometry directly.
- The constraint matrix C is built with `ParMixedBilinearForm` whose
  trial space is the high-order displacement FES and test space is the
  low-order LM FES on the refined submesh. Quadrature rule is selected
  for the higher of the two orders.
- **Partial / element / full assembly is per-bilinear-form**. Keep K at
  PA on GPU; assemble C at FULL (sparse HypreParMatrix). The block
  saddle-point operator `[[K_op, Cᵀ_op], [C_op, 0]]` mixes a matrix-free
  K with a sparse C — exactly the abstraction the §6 prototype already
  uses. **Constraint construction remains agnostic to the volume
  assembly choice (PA / EA / FA)**, as designed.
- AMG on K under PA requires `ParLORDiscretization` for the AMG
  setup; this is a separate concern from LOR mortar and orthogonal to
  the constraint design.

### §4.11.5 Implementation cost vs higher-order dual

| Approach | Engineering cost | Per element-type proliferation | MFEM availability |
|---|---|---|---|
| Higher-order standard P_p LM with Belgacem cross-point modification | Medium | Low (vertex zero-out + barycentric redistribution) | Doable with stock APIs |
| Higher-order **dual** (Popp 2012 basis transformation) | **High** | **Per element type**: tri-6, quad-8, quad-9, hex-20, hex-27 each need own A_e and own boundary modifications | Not in stock MFEM; requires custom FECollections + integrators |
| **LOR + linear dual + Barbosa-Hughes stabilisation** (recommended) | **Low** | None (re-uses §4.2–§4.5 linear dual) | Out-of-the-box with one extra integrator |
| Tribol-style LOR projection | Low | None | Available in MFEM 4.7+ via Tribol miniapp |
| Penalty (no LM) | Trivial | None | Trivial; conditioning issues |

## §4.12 Recommendation for ExaConstit higher-order PBC

ExaConstit's primary FE order for crystal plasticity is p = 1 (linear
hex / linear tet); higher-order is **not** on the immediate roadmap.
However, when it eventually is, the recommended path is:

1. **Stay with the current §4.2–§4.5 linear dual basis machinery.**
2. **Build an order-1 LM space on a uniformly-refined ParSubMesh** of
   the periodic boundary, per (4.38) and the §4.11.4 mechanics.
3. **Add a Barbosa-Hughes residual stabilisation integrator** (4.42)
   to the saddle-point block; γ_β tuned per material.
4. **Validate with manufactured-solution h-refinement** to confirm
   near-optimal H¹ rates O(h^p) on the displacement.
5. **Reach for the §4.10 Popp 2012 basis-transformation only if a
   homogenisation use case demonstrates measurable accuracy degradation
   at the engineering quantities of interest** (effective tangent
   moduli, stress homogenisation). Existing CPFEM-homogenisation
   literature has *no* precedent for higher-order mortar PBC and
   suggests this is unlikely to be needed.

This recommendation aligns with Tribol's design philosophy
[Chin, MFEM Workshop 2023, "Contact constraint enforcement using the
Tribol interface"] and avoids the proliferation of per-element-type
dual basis derivations and Wohlmuth modifications. The
**assembly-agnostic constraint construction** that has been a design
invariant since Phase 1A is preserved: C is a sparse HypreParMatrix
built from linear duals, K is consumed via Operator interface at any
PA/EA/FA setting, and the saddle-point solver in §6 doesn't care.

We flag higher-order extensions as a Phase-6+ scope item in §14.3.

---

# §5. Hierarchical crosspoint structure and the Wohlmuth modification

The crosspoint problem arises because the standard dual basis (§4) places
nonzero multiplier weight at *every* nonmortar-side node, including those that
are essentially constrained (corners) or already constrained at a lower
hierarchy level (edges in 3D). The constraint becomes redundant or
inconsistent. **Wohlmuth's modification** [Wohlmuth 2000, §5;
Wohlmuth 2001, §1.3.4] adjusts the dual basis on nonmortar-side elements
adjacent to such crosspoints so that:

1. The multiplier rows for "redundant" DOFs are removed (M_redundant ≡ 0
   on the affected element).
2. **Partition of unity** (§4.0, eq. 4.6) is preserved on the modified
   element, ensuring constant-reproduction across the interface.
3. **Local biorthogonality is relaxed in a controlled way**: the modified
   M_i is no longer pointwise dual to N_j on the modified element, but the
   *quasi-dual* property [Lamichhane & Wohlmuth 2007, §3.2] holds — the
   constraint enforces the right physics in the modified region.

This section derives the modification explicitly for line-2 (used in 2D
edge mortar and 3D edge mortar), tri-3 (used in 3D face mortar on tet
meshes), and quad-4 (used in 3D face mortar on hex meshes). The 1D case
is the foundation; the 2D cases generalize it to tensor-product (quad)
and barycentric (triangle) settings.

## §5.1 The 2D problem and the line-2 modification

Take a square RVE with the 4 corners and 4 edges. The PBC story:

- **Corners**: pin all 4 corners to remove rigid-body translation and
  rotation. 4 corners × 2 components = 8 essential TDOFs. In Method D,
  corner *displacement values* are u_lin[corner] = (F − I) X_corner; in
  Method C they are zero (essential ũ at corners). Reference: [Lopes
  et al. 2021, §3.4, lines 1034–1035].
- **Edges**: couple opposite-edge pairs (right ↔ left, top ↔ bottom) via
  the line-2 mortar method (§3, §4.2). Each edge has interior nodes plus
  two end nodes. The end nodes ARE the corners — they overlap with the
  essential set.

### §5.1.1 The crosspoint over-constraint

Without modification, the nonmortar-side line-2 mortar would assemble an LM
row for *every* nonmortar DOF, including the corner DOFs at the edge endpoints.
Combined with the corner essential BC, this produces:

| DOF | Essential BC | Mortar LM row | Result |
|---|---|---|---|
| Corner | u = u_lin[corner] | row in C with corner column nonzero | over-constrained |
| Edge interior | none | row in C with column nonzero | correctly constrained |

The "over-constraint" comes through: the constraint matrix C now has rows
that mention the essential corner DOFs in their column structure. After
applying corner Dirichlet (which zeroes those columns of C — see
`apply_dirichlet_zero_to_C`), the LM rows for the corner DOFs become
*zero rows*: 0 = 0 trivially, but they consume LM unknowns. The system
has redundant constraints; the C·diag(K)⁻¹·Cᵀ Schur complement has a zero
diagonal entry corresponding to the corner-LM row, which makes the
saddle-point preconditioner ill-defined.

### §5.1.2 The modification: M_i on the corner-end element

Let the nonmortar-side end element be a line-2 with nodes labeled 1 (the corner
endpoint, ξ = −1) and 2 (the interior neighbor, ξ = +1). The
*standard* dual basis (eq. 4.13):

    M_1(ξ) = (1 − 3ξ) / 2    (corner side)                            (5.1a)
    M_2(ξ) = (1 + 3ξ) / 2    (neighbor side)                          (5.1b)

The Wohlmuth-modified dual basis on this end element [Wohlmuth 2000, §5;
Lopes et al. 2021, Eq. C.2]:

    M_1^mod(ξ) ≡ 0           (corner row dropped)                     (5.2a)
    M_2^mod(ξ) ≡ 1           (neighbor takes constant value)          (5.2b)

This says: on the corner-end element, do not assemble a constraint row for
the corner DOF. The neighbor DOF's multiplier is identically 1 — a
*constant* over this element.

**Partition of unity preserved.** M_1^mod(ξ) + M_2^mod(ξ) = 0 + 1 = 1
for all ξ ∈ [−1, +1]. ✓

**Constant reproduction preserved.** A constant ũ ≡ c integrated against
M_2^mod on this element gives ∫ M_2^mod · c dξ = c · 2 (segment length on
[−1,+1]), which is the same value the standard linear-N integration would
give: ∫ N_1 c + ∫ N_2 c = c · 1 + c · 1 = 2c. So the modified basis
reproduces constants correctly across the modified end-segment.

**Biorthogonality is relaxed.** ∫ M_2^mod N_2 dξ = ∫ 1 · (1+ξ)/2 dξ = 1
(matches the standard target ∫ N_2 = 1). But ∫ M_2^mod N_1 dξ = ∫ 1 ·
(1−ξ)/2 dξ = 1 ≠ 0. The off-diagonal "leak" is intentional: it routes the
corner-DOF coupling into the neighbor's row, which is what removes the
redundancy with the corner Dirichlet [Wohlmuth 2000, eq. 5.4].

### §5.1.3 Why this fixes the over-constraint

After modification:

- The **corner LM row is gone** (M_corner^mod = 0 means no constraint
  contribution from this element to the corner row, and dropping the
  corner row entirely from the LM space removes the redundancy).
- The **neighbor LM row** still constrains the neighbor DOF, but now
  through M_2^mod = 1, which integrates against both N_1 and N_2 on the
  end element.

The constraint then enforces the right physics: the neighbor's
fluctuation periodicity, while letting the corner be free to satisfy its
Dirichlet BC without LM interference.

The implementation in `mortar_pbc/mortar_2d.py`:

```python
def M_line2_dual_modified(xi: float, side: str) -> tuple[float, float]:
    """Lopes Eq. C.2 / Wohlmuth (2000) corner-modified dual basis.

    side == 'left'  : the left node (ξ=-1, "node 1") is the Dirichlet corner.
                      M_1 = 0; M_2 = 1.
    side == 'right' : the right node (ξ=+1, "node 2") is the Dirichlet corner.
                      M_1 = 1; M_2 = 0.
    side == 'none'  : interior element, use standard dual basis.
    """
    if side == "left":
        return (0.0, 1.0)
    elif side == "right":
        return (1.0, 0.0)
    else:
        return M_line2_dual(xi)
```

Verified by `test_wohlmuth_crosspoint_modification` (partition of unity,
corner-side-zero, neighbor-side-integrals).

## §5.2 The triangle (tri-3) modification (3D face mortar on tet meshes)

For a tet-mesh RVE, periodic faces are tri-3 elements. The face boundary
has *three edges* and *three corners*. The Wohlmuth modification on a
triangle adjacent to a face-boundary edge (or corner) generalises the 1D
recipe.

### §5.2.1 Triangle classification by face-boundary adjacency

Let a tri-3 face element have vertices labeled 1, 2, 3 with barycentric
coordinates (1,0,0), (0,1,0), (0,0,1). The face boundary is a 2D loop;
each tri-3 face element belongs to one of:

- **Interior** — none of the 3 vertices is on the face boundary.
  Standard dual basis (eq. 4.19): M_i = 4 λ_i − 1.
- **Edge-adjacent** — exactly one vertex is on the face boundary, OR
  one whole edge of the triangle lies on the face boundary. Modify
  the dual basis at that vertex/edge.
- **Corner-adjacent** — two vertices are on face-boundary edges (i.e.,
  the triangle touches a face *corner*). Modify two vertices.

(A tri-3 face element cannot have *all three* vertices on the face
boundary unless the tri-3 *is* a face corner triangle, which is a
degenerate case for a coarse mesh — possible but rare. We handle it as
the degenerate limit of the corner-adjacent case.)

### §5.2.2 Edge-adjacent modification (one vertex dropped)

Suppose vertex 1 (with shape function N_1 = λ_1) is on a face-boundary
edge. The modified dual basis sets M_1^mod = 0 and re-distributes the
weight across M_2 and M_3:

    M_1^mod(λ) = 0                                                    (5.3a)
    M_2^mod(λ) = a + b λ_2 + c λ_3                                    (5.3b)
    M_3^mod(λ) = a + c λ_2 + b λ_3   (by symmetry)                    (5.3c)

We require partition of unity: M_2^mod + M_3^mod = 1, i.e.

    2a + (b+c)(λ_2 + λ_3) = 1     for all (λ_2, λ_3) with λ_1 = 1 − λ_2 − λ_3

This must hold for all admissible (λ_2, λ_3), so:
- coefficient of (λ_2 + λ_3): b + c = 0 → c = −b
- constant term: 2a = 1 → a = 1/2

We additionally require the standard target integrals:

    ∫_E M_2^mod N_2 dE = ∫_E N_2 dE = |E|/3                           (5.4)

Computing with (5.3b) and (4.7):

    ∫_E (1/2 + b λ_2 − b λ_3) λ_2 dE
    = (1/2) ∫ λ_2 dE + b ∫ λ_2² dE − b ∫ λ_2 λ_3 dE
    = (1/2)(|E|/3) + b(|E|/6) − b(|E|/12)
    = |E|/6 + b|E|/12

Set equal to |E|/3 = 4|E|/12:

    |E|/6 + b|E|/12 = 4|E|/12
    2|E|/12 + b|E|/12 = 4|E|/12
    b = 2

So:

    M_2^mod(λ) = 1/2 + 2 λ_2 − 2 λ_3                                  (5.5a)
    M_3^mod(λ) = 1/2 − 2 λ_2 + 2 λ_3                                  (5.5b)
    M_1^mod(λ) = 0                                                    (5.5c)

**Verification.** Partition of unity:
M_2 + M_3 = 1 + 0 + 0 = 1. (M_1 = 0 contributes nothing.)
Including the dropped corner: M_1 + M_2 + M_3 = 0 + 1 = 1. ✓

Bi-orthogonality (target value):
- ∫ M_2 N_2 = (1/2)(|E|/3) + 2(|E|/6) − 2(|E|/12) = |E|/6 + |E|/3 − |E|/6 = |E|/3 ✓
- ∫ M_2 N_3 = (1/2)(|E|/3) + 2(|E|/12) − 2(|E|/6) = |E|/6 + |E|/6 − |E|/3 = 0 ✓
- ∫ M_2 N_1 (the *dropped* row's column): (1/2)(|E|/3) + 2(|E|/12) − 2(|E|/12) = |E|/6 ≠ 0

The last entry is the "leak" — a controlled non-orthogonality between the
modified M_2 and the dropped node's N_1, identical in spirit to the 1D
case (§5.1.2). The corner DOF is essentially constrained, so the leak
into N_1's column is harmless after corner-column zeroing of C.

### §5.2.3 Corner-adjacent modification (two vertices dropped)

Suppose vertices 1 and 2 are both on face-boundary edges (so the tri-3
touches a face corner where two boundary edges meet). The modification
sets both M_1^mod = M_2^mod = 0, and the third vertex's M_3^mod must
satisfy the partition-of-unity and constant-reproduction targets alone.

By symmetry of the construction, M_3^mod(λ) = a + b λ_3. Partition of
unity (only M_3^mod is nonzero among the three):

    M_3^mod(λ) = 1     ∀ λ ∈ E       (i.e. a = 1, b = 0)              (5.6)

This is the direct 2D analog of (5.2): on a corner-adjacent triangle, the
single non-dropped multiplier is identically 1.

**Verification.**

- Partition of unity: 0 + 0 + 1 = 1 ✓
- Constant reproduction: ∫ 1 · c dE = c · |E|, matches ∫(N_1+N_2+N_3) c dE
  = ∫ 1 · c dE = c · |E| ✓
- ∫ M_3 N_3 = ∫ 1 · λ_3 dE = |E|/3 = ∫ N_3 ✓ (target met)
- ∫ M_3 N_1 = ∫ 1 · λ_1 dE = |E|/3 ≠ 0 (leak, harmless after corner-col zero)
- ∫ M_3 N_2 = |E|/3 (leak)

### §5.2.4 Implementation outline (Phase 3.2)

```python
def M_tri3_dual_modified(
    lam: tuple[float, float, float],
    boundary_nodes: tuple[bool, bool, bool],
) -> tuple[float, float, float]:
    """Wohlmuth-modified dual basis on a tri-3 face element.

    boundary_nodes[i] = True if vertex i is on a face-boundary feature
                       (edge or corner of the parent face) and therefore
                       the corresponding LM row should be dropped.

    Cases:
      0 boundary nodes: standard tri-3 dual (M_i = 4 λ_i − 1).
      1 boundary node: edge-adjacent modification (eq. 5.5).
      2 boundary nodes: corner-adjacent modification (eq. 5.6 — the
                       remaining vertex's multiplier is identically 1).
      3 boundary nodes: degenerate; multiplier identically 0 on this
                       element (no constraint contribution).
    """
    n_dropped = sum(boundary_nodes)
    if n_dropped == 0:
        return M_tri3_dual(lam)
    elif n_dropped == 1:
        # Identify which vertex is dropped, apply (5.5) accordingly.
        idx_dropped = boundary_nodes.index(True)
        # ... permute (5.5) so that the dropped vertex gets M = 0
        ...
    elif n_dropped == 2:
        # Identify which vertex is *not* dropped; its M = 1, others = 0.
        idx_kept = boundary_nodes.index(False)
        result = [0.0, 0.0, 0.0]
        result[idx_kept] = 1.0
        return tuple(result)
    else:  # n_dropped == 3
        return (0.0, 0.0, 0.0)
```

Verification target for Phase 3.2 unit test
`test_wohlmuth_tri3_modification`:

- Bi-orthogonality at non-dropped vertices: ∫ M_i^mod N_i = ∫ N_i = |E|/3.
- Off-diagonal between two non-dropped vertices: 0.
- Partition of unity over non-dropped vertices: 1.
- Off-diagonal into dropped vertices: |E|/3 (harmless leak).

## §5.3 The quad-4 modification (3D face mortar on hex meshes)

For a hex-mesh RVE, periodic faces are quad-4 elements. The face boundary
has *four edges* and *four corners*. The Wohlmuth modification generalises
the 1D recipe via tensor product.

### §5.3.1 Quad classification

Let a quad-4 face element have nodes labeled 1, 2, 3, 4 at parametric
corners (−1,−1), (+1,−1), (+1,+1), (−1,+1). Each face element is one of:

- **Interior** — none of the 4 vertices is on the face boundary.
  Standard quad-4 dual basis (eq. 4.16).
- **Edge-adjacent** — exactly one edge of the quad-4 (so 2 of its 4
  vertices) is on a face-boundary edge. Modify the dual basis in *one*
  parametric direction.
- **Corner-adjacent** — exactly one vertex is on a face corner (and 2 of
  its 4 vertices are on face-boundary edges). Modify in *both*
  parametric directions.

### §5.3.2 Edge-adjacent: one parametric direction modified

Suppose the η = −1 edge of the quad-4 is on a face-boundary edge. Then
nodes 1 and 2 (η-coordinate = −1) are dropped; nodes 3 and 4 (η-coordinate
= +1) are kept.

The 1D modified dual basis in η (with side="left", since η = −1 is the
"left" of [−1,+1]):

    M_line2_mod(η, "left") = (0, 1)     (M(η=-1)=0, M(η=+1)=1)        (5.7)

Tensor product with the standard 1D dual in ξ:

    M_quad4_1^mod(ξ,η) = M_line2(ξ, p=1) · 0 = 0                      (5.8a)
    M_quad4_2^mod(ξ,η) = M_line2(ξ, p=2) · 0 = 0                      (5.8b)
    M_quad4_3^mod(ξ,η) = M_line2(ξ, p=2) · 1 = (1+3ξ)/2               (5.8c)
    M_quad4_4^mod(ξ,η) = M_line2(ξ, p=1) · 1 = (1−3ξ)/2               (5.8d)

So nodes 1 and 2 (the dropped edge) have M ≡ 0; nodes 3 and 4 (the
neighboring edge) have M = 1D-dual-in-ξ × 1.

Partition of unity in (ξ, η) on this element:

    ∑_i M_i^mod = 0 + 0 + (1+3ξ)/2 + (1−3ξ)/2 = 1     ∀ (ξ,η)         (5.9)

✓ The 1D partition-of-unity in ξ carries through.

Symmetric for the other three boundary-edge orientations (η=+1, ξ=±1).

### §5.3.3 Corner-adjacent: both parametric directions modified

Suppose node 1 (parametric corner (−1,−1)) is on a face corner. Then both
the ξ = −1 edge AND the η = −1 edge of the quad-4 are face-boundary
edges. The 1D modification applies in *both* ξ and η directions, giving
(side_ξ, side_η) = ("left", "left"):

    M_line2_mod(ξ, "left") = (0, 1)
    M_line2_mod(η, "left") = (0, 1)

Tensor product:

    M_quad4_1^mod(ξ,η) = 0 · 0 = 0     (the corner)                   (5.10a)
    M_quad4_2^mod(ξ,η) = 1 · 0 = 0     (corner-adjacent in η)         (5.10b)
    M_quad4_3^mod(ξ,η) = 1 · 1 = 1     (diagonally opposite)          (5.10c)
    M_quad4_4^mod(ξ,η) = 0 · 1 = 0     (corner-adjacent in ξ)         (5.10d)

Only the **diagonally opposite** vertex has a non-zero (and constant)
multiplier on this corner-adjacent quad. Partition of unity: 0 + 0 + 1 +
0 = 1 ✓.

This is the direct 2D analog of (5.6) — same structure as the
corner-adjacent triangle case, where the single non-dropped multiplier is
identically 1.

### §5.3.4 Implementation outline (Phase 3.2)

```python
def M_quad4_dual_modified(
    xi: float, eta: float,
    side_xi: str = "none",   # "none" | "left" | "right"
    side_eta: str = "none",  # "none" | "bottom" | "top"
) -> tuple[float, float, float, float]:
    """Wohlmuth-modified dual basis on a quad-4 face element via tensor product.

    side_xi  modification: "left" drops node-side ξ=-1; "right" drops ξ=+1.
    side_eta modification: "bottom" drops node-side η=-1; "top" drops η=+1.

    Edge-adjacent: exactly one of (side_xi, side_eta) is non-"none".
    Corner-adjacent: both are non-"none" (diagonal-opposite node retains M=1).
    """
    M_xi = M_line2_dual_modified(xi, side_xi)   # tuple of 2
    M_eta = M_line2_dual_modified(eta, side_eta)  # tuple of 2
    return (
        M_xi[0] * M_eta[0],    # node 1 at (-1,-1)
        M_xi[1] * M_eta[0],    # node 2 at (+1,-1)
        M_xi[1] * M_eta[1],    # node 3 at (+1,+1)
        M_xi[0] * M_eta[1],    # node 4 at (-1,+1)
    )
```

Verification target for Phase 3.2 unit test
`test_wohlmuth_quad4_modification`:

- Edge-adjacent: nodes on the modified edge have M ≡ 0; partition of
  unity preserved.
- Corner-adjacent: only the diagonal-opposite node has M ≡ 1; partition
  of unity preserved.
- Bi-orthogonality (target): ∫ M_i^mod N_i = ∫ N_i (|E|/4 for the 4-node
  quad with the standard mass-integral target).

### §5.3.5 The 3-sentinel corner-of-face quad (subtle but ubiquitous)

When the boundary classifier (§11.8 Phase 3.3.B) walks face elements
and stamps sentinel values on per-vertex DOFs, a single quad-4
element can carry **three** sentinels at once: one corner-of-the-RVE
DOF (sentinel `-1`) plus two box-edge-interior DOFs (sentinel `-2`)
on the two element edges meeting at that RVE corner. The remaining
fourth node — diagonally opposite the RVE corner — is the only kept
face-interior DOF.

This 3-sentinel pattern is **the most common boundary-adjacent quad
configuration on an axis-aligned RVE**: every box face has 4 such
quads at its 4 corners. On a 4×4×4 hex mesh, that's 24 such quads
(4 per face × 6 faces). They are NOT degenerate cases — they're
the bulk of the wirebasket-modified work.

The right Wohlmuth tag for this configuration is one of `corner-LL`,
`corner-LR`, `corner-UR`, `corner-UL`, picked so the dropped sides
match the {ξ, η} extents of the sentinel cluster. The naming
convention is **side-coverage, not corner-of-kept-node**: the tag
names which two element sides are dropped, NOT which corner the
kept node is at. Mapping (where the kept node is the only
non-sentinel local node):

| kept local node | kept-node corner | dropped sides | tag |
|---|---|---|---|
| 0 | (xi=−1, eta=−1) "LL" | xi-high + eta-high | `corner-UR` |
| 1 | (xi=+1, eta=−1) "LR" | xi-low  + eta-high | `corner-UL` |
| 2 | (xi=+1, eta=+1) "UR" | xi-low  + eta-low  | `corner-LL` |
| 3 | (xi=−1, eta=+1) "UL" | xi-high + eta-low  | `corner-LR` |

(Yes, the tag for "kept node 2 = UR corner" is `corner-LL` —
because side_xi="left" and side_eta="bottom" are what's dropped.
The tag is named after the dropped sides; this is the convention
used by `M_quad4_dual_modified(side_xi="left", side_eta="bottom")`.)

**Why the modification matters for correctness here.** If the
3-sentinel quad were tagged `'none'` and the assembler used the
standard (unmodified) dual basis for the kept row, the constraint
matrix would *almost* be right: the constraint builder zeros the
corner/edge columns by sentinel logic anyway. But the kept (face-
interior, face-interior) entry of A_m would carry a small leak
from the standard-vs-modified dual basis difference. That leak
manifests as a small constraint residual at convergence (not a
catastrophic failure, but a real correctness issue). The modified
dual basis fixes the kept-row entries to the right values. The
fix is implemented in
``BoundaryClassifier3D._classify_quad_boundary_tag`` which dispatches
all 16 sentinel-pattern cases (0/1/2/3/4 sentinels with all
geometric arrangements).

The analogous 2-vertex-dropped tri-3 case (§5.2.3) handles the
corresponding tet-mesh configuration cleanly — the
``M_tri3_dual_modified`` machinery accepts `boundary_nodes = (T, T, F)`
to drop two vertices simultaneously, with the kept vertex's dual
becoming a constant 1 (per eq. 5.6).

## §5.4 The 3D wirebasket hierarchy

In 3D the geometric hierarchy is one level deeper than 2D:

| Feature | Dim | Count (cube RVE) | Constraint role | LM rows |
|---|---|---|---|---|
| **Corner** | 0 | 8 | Essential Dirichlet (u_corner = (F−I)X_corner) | None |
| **Edge** (wirebasket) | 1 | 12 | Mortar, with 1D Wohlmuth at corner endpoints | Corners dropped |
| **Face** | 2 | 6 | Mortar, with 2D Wohlmuth (tri or quad) along edge boundary | Edges dropped |

The cascade ensures non-redundancy: each level constrains exactly the
DOFs that aren't already covered by a higher level [Wohlmuth 2001,
§1.3.4; Lamichhane & Wohlmuth 2007, §3.3].

Three levels of constraint, three modifications:

1. **Corner Dirichlet**: 24 essential TDOFs (8 corners × 3 components).
   Method D applies u_corner = (F − I) X_corner; the 8 corners are pinned
   exactly. No LM rows.
2. **Edge mortar with corner crosspoint mod**: each pair of periodic
   edges gets one mortar block. Wohlmuth modification at corner
   endpoints (eq. 5.2) removes corner-LM rows. The cube has 12 edges
   total, partitioned into 3 groups of 4 (by axis parallelism); within
   each group, pick one as mortar and assemble 3 mortar-nonmortar mortar
   blocks. Total: 3 directions × 3 = 9 edge mortar blocks.
3. **Face mortar with edge crosspoint mod**: each pair of opposite faces
   gets one mortar block. Wohlmuth modification along edge boundaries
   (eq. 5.5 / 5.6 for triangles, eq. 5.8 / 5.10 for quads) removes
   edge-LM rows. There are 3 face pairs (one per axis direction).

## §5.5 Hex meshes vs tet meshes: same hierarchy, different elements

The hierarchy in §5.4 is independent of element type. What differs is
the *element class* used at each level:

| Mesh type | Volume element | Face element | Edge element |
|---|---|---|---|
| **Hex** | hex-8 | quad-4 | line-2 |
| **Tet** | tet-4 | tri-3 | line-2 |
| **Mixed** | hex-8 + tet-4 | quad-4 + tri-3 | line-2 |

In all three cases:

- Edge mortar uses the **line-2** dual basis with the 1D Wohlmuth
  modification (§5.1). The element class is the same regardless of
  whether the parent volume is hex or tet.
- Face mortar uses **quad-4** (hex parent) or **tri-3** (tet parent),
  with the corresponding 2D Wohlmuth modification (§5.2 for tri-3, §5.3
  for quad-4).
- Mixed meshes: each face dispatches on its element type. A
  quad-4-face from a hex element next to a tri-3-face from a tet
  element on the same periodic boundary is allowed; the constraint
  rows assemble per-face with the appropriate `M_*_dual_modified`
  function.

The architectural implication: the C++ port must dispatch on
`mfem::Element::Type` (or equivalent) when assembling face mortar,
selecting the dual basis polymorphically. This polymorphism slots
naturally into a `MortarFaceAssembler` class with virtual `Assemble`
implementations for `QuadFaceAssembler` and `TriFaceAssembler`.

ExaConstit currently supports both hex and tet meshes for crystal
plasticity, with users routinely choosing between them based on grain
geometry complexity. PBC support must therefore handle both natively
[ExaConstit issue #8 commentary; ExaConstit user guide §3].

## §5.6 Why this matters for correctness

If you skip the Wohlmuth modification:

- **2D**: the patch test still passes for some macroscopic F (e.g.
  uniform uniaxial), but fails for shear F or any F that places the
  corner-LM redundancy into a numerical contradiction. The discrete
  constraint becomes inconsistent at the corner; the saddle-point
  Schur complement has zero diagonal entries; the block-Jacobi
  preconditioner produces NaN or infinite scalers.
- **3D**: the situation is worse. Without the edge-level modification,
  every face mortar is over-constrained at all 12 edges. Without the
  corner-level modification on edges, every edge mortar is
  over-constrained at all 8 corners. The redundant constraints don't
  just produce slightly-wrong answers; they produce a singular
  C·diag(K)⁻¹·Cᵀ Schur complement.

So the modification is not optional [Wohlmuth 2000, Theorem 5.1]. The
unit tests verify the modification *at the dual-basis level*
(independent of the FE assembly), making the correctness easy to
localise when something downstream breaks.

The 2D unit test `test_wohlmuth_crosspoint_modification` validates
properties (5.2). Phase 3.2 will add `test_wohlmuth_tri3_modification`
(eqs. 5.5, 5.6) and `test_wohlmuth_quad4_modification` (eqs. 5.8, 5.10)
as 3D analogs.

---

# §6. The saddle-point system and how we solve it

## §6.1 The continuous problem

For Method D with linear elasticity (the prototype's solving regime), the
strong form is:

- ∇·σ = 0 in Ω
- σ = C·ε, ε = (∇u + ∇uᵀ)/2  (linear elastic)
- u = u_lin = (F−I)X on essential corner set
- ⟨ũ⟩-periodic on opposite faces (mortar weak periodicity)

Lagrangian for the constrained equilibrium:

L(u, λ) = (1/2) uᵀ K u − λᵀ C u

(no body force in our setup; the corner displacement enters as a Dirichlet
BC, not via L).

Stationary: K u + Cᵀ λ = 0; C u = 0.

The discretized form is:

[[K, Cᵀ], [C, 0]] [u; λ] = [b; 0]

where b absorbs whatever right-hand side comes from the corner Dirichlet
elimination (it's K_eliminated u_lin shifted to the RHS, with corner entries
forced to satisfy u = u_lin[corner]).

## §6.2 Indefiniteness — why CG is rejected

The saddle-point matrix has signature (+, −) — symmetric but not positive
definite. CG diverges (or worse, gives garbage). Three valid Krylov choices:

- **MINRES**: optimal for symmetric indefinite. Default for our linear-elastic
  symmetric K.
- **GMRES**: works for any matrix; needed when K is non-symmetric (some
  constitutive models give non-symmetric tangent — crystal plasticity
  *can*).
- **BiCGStab**: a non-symmetric option with shorter recurrences than GMRES.

The `SaddlePointSolver` class supports all three at runtime via a
`solver=` parameter. CG is explicitly forbidden in the API.

## §6.3 The block-Jacobi preconditioner

The 2-block diagonal preconditioner:

P = [diag(K), 0; 0, diag(C diag(K)⁻¹ Cᵀ)]

implemented as:

- Block (0,0): apply diag(K)⁻¹. Computed via `Operator.AssembleDiagonal()`,
  which works uniformly on PA, EA, FA, and HypreParMatrix forms of K. We
  *never* call `K.As<HypreParMatrix>()` or anything like that — diagonal
  extraction is the right level of abstraction.
- Block (1,1): apply diag(C diag(K)⁻¹ Cᵀ)⁻¹. Computed *without* forming
  C diag(K)⁻¹ Cᵀ explicitly — instead the C operator exposes a method
  `WeightedRowSqSum(weights, out)` that returns out[i] = Σ_j C[i,j]² · w[j]
  for owned rows. With w = diag(K)⁻¹ this gives exactly the row-diagonal of
  C diag(K)⁻¹ Cᵀ, the missing piece.

In production we'll replace block-Jacobi-on-K with HypreBoomerAMG (when K is
fully assembled) or a multigrid-on-PA-K (when K is matrix-free). The
prototype's block-Jacobi is a stepping stone.

## §6.4 The RHS construction (the bug-prone part)

Given the linear system:

[[K_e, Cᵀ], [C, 0]] [du, dλ] = [−r1, 0]

where:

- K_e = K with corner rows/cols zeroed and replaced by identity-on-diagonal.
- r1 = K_full · u_lin (the full, un-eliminated K applied to u_lin), with
  corner entries of r1 zeroed afterward.

**Why r1 must use K_full and not K_e:**

For homogeneous material under uniform F, the affine field u_lin IS the
equilibrium solution. That means K_full · u_lin = 0 at *free* rows
(Σ_col K_full[free_row, col] · u_lin[col] = 0). At corner rows it gives the
nontrivial corner reaction force, but those rows of r1 are zeroed.

If instead you compute r1 = K_e · u_lin, the K_uc column has been zeroed by
the elimination, so K_e · u_lin at free rows gives K_uu · u_lin[free] only —
which is *NOT* zero in general (the affine field requires the K_uc · u_lin[corner]
contribution to balance K_uu · u_lin[free] for the affine to be the solution).
The result is r1 has spurious nonzero values at free rows, and the saddle-
point solve produces a `du` that drives free DOFs *away* from u_lin to "fix"
the spurious residual.

Symptom in 2D heterogeneous case: in ParaView, free DOFs appear to move in
the *opposite* direction from u_lin while corners stay correct. This was the
multi-step driver bug from session 6. The fix: pass *both* K_full and K_e
into the driver, use K_full for r1 computation, K_e for the saddle-point top
block.

In 2D Phase-2 single-step working code, K was assembled, then `K.Mult(u_lin,
f)` happened, *then* corner elimination was applied to K and to f
simultaneously (`apply_dirichlet_to_distributed_K`). Order of operations
saved us. The multi-step driver moved corner elimination outside the driver,
breaking the implicit assumption.

## §6.5 The Newton residual (when nonlinear)

For nonlinear K (= ∂F_int/∂u from a nonlinear material), the Newton residual
at iterate (u^k, λ^k) is:

r1^k = F_int(u^k) + Cᵀ · λ^k         (force balance)
r2^k = C · u^k − g                   (constraint residual; g=0 for fluctuation periodicity)

The Newton step solves [[K^k, Cᵀ], [C, 0]] [du, dλ] = [−r1^k, −r2^k].

Critical: r1 includes the +Cᵀ · λ^k term. Naively using F_int(u^k) alone
gives a residual that doesn't go to zero at convergence — it stagnates at the
natural force scale of the problem because at equilibrium F_int = −Cᵀλ, not
zero. See the §12 trap list.

For the linear-elastic prototype with one Newton iteration, F_int(u) = K·u,
λ⁰ = 0, so r1 = K·u_lin (computed via K_full as discussed in §6.4).

## §6.6 Sign conventions in the saddle-point API

To eliminate sign-error bugs we converged on this API for `SaddlePointSolver.solve_step`:

```python
def solve_step(self, *, K_op, C_op, CT_op, r1_local, r2_local):
    """Solve the constrained Newton step.
    
    The system solved is
        [[K  C^T] [du  ]   [-r1_local]
         [C   0 ]] [dλ ] = [-r2_local]
    
    Caller assembles the FULL Newton residuals r1, r2 (including any C^T λ
    contribution).  Solver simply negates them.
    """
```

The solver internally negates `r1_local` and `r2_local` to form the RHS. This
removes ambiguity: the caller computes the residual *as written in the
literature* (∇L, including the Cᵀλ term in r1 and the constraint mismatch in
r2), and the solver always produces the correct (du, dλ) update.

## §6.7 SetIterativeMode(False) on the inner Krylov

This is a defensive pattern. The inner Krylov solves for *increment* (du, dλ),
which has no relationship to the previous Newton iteration's increment. If
`SetIterativeMode(True)` is set, the Krylov solver treats the incoming du as
an initial guess — but we always pass zero, so it's a no-op…

Except for CG specifically, an iterative-mode initial guess that's been
zeroed but is passed through a `BlockVector` of mixed zero-and-nonzero blocks
*can* trigger Lanczos breakdowns or poor convergence. Even though we use
MINRES/GMRES/BiCGStab and not CG, the false negative is cheap to avoid.
Set `SetIterativeMode(False)` always.

The Newton outer loop *does* warm-start at the outer level: u and λ accumulate
across Newton iterations. That's correct; the inner Krylov is something
different.

---

# §7. Warm-start theory: from ExaConstit's `SolveInit` to multi-step F ramping

## §7.1 The problem warm-starts solve

In a multi-step load history, each step n+1 inherits the converged kinematic
state at step n. If between steps n and n+1 the boundary conditions change
(e.g. the prescribed displacement at the corners shifts because F_macro
shifted), then the previous-step state is *no longer in equilibrium with the
new boundary*: free DOFs are still at their step-n values while corner DOFs
must jump to their step-n+1 values.

Starting Newton from this misaligned state is risky:

- **Mild case**: Newton converges in extra iterations, with the first iterate
  showing a large residual that just reflects the BC mismatch.
- **Severe case**: the first Newton iterate puts the material into a state
  that's outside the basin of convergence — for hyperelastic models, this can
  mean elements with `det(F) ≤ 0`, which can return NaN or otherwise crash
  the integrator.
- **Crystal-plasticity-specific**: for rate-dependent models, the prior
  velocity field is a state the integrator depends on. A bad initial iterate
  leads to non-physical guesses for the slip-system rates.

The ExaConstit-style warm-start projects the BC change through the
*previous-step tangent* to produce a sensible initial iterate that has the
new corner displacements applied AND has the free DOFs adjusted by a single
linear solve to be approximately consistent with those new corner values.

## §7.2 ExaConstit's `SystemDriver::SolveInit` (the reference)

Sources:
- `src/system_driver.cpp:441-478` (`SolveInit`)
- `src/fem_operators/mechanics_operator.cpp:295-331` (`GetUpdateBCsAction`)

The pattern is, in pseudo-code:

```cpp
// Before Newton step n+1.
// State: x_n (converged), v_n (converged), prescribed_v at step n+1 known.

deltaF = 0;                                               // size: n_TDOF
deltaF[essential_TDOFs] = prescribed_v[ess] - v_n[ess];   // change in BC

// Build a special operator that:
//   1. Computes b = K_full @ deltaF on FREE rows (the K_uc · Δv_c term).
//   2. Adds the residual at the previous-converged state (= 0 at convergence,
//      nonzero if step n didn't quite converge — captures leftover imbalance).
//   3. Combines: y = K_uc · Δv_c + R^n on free rows.
oper = mech_operator->GetUpdateBCsAction(v_n, deltaF, b);

// Solve the eliminated system K_eliminated @ Δv = -b for Δv on free rows.
// CG (this is a positive-definite system; no constraints involved here).
CG_solve(K_eliminated, -b, Δv);

// Initial iterate for Newton step n+1 is:
//   v_initial = v_n + deltaF + Δv
//   = v_n  on free DOFs (Δv ≈ 0 if v_n was good) + (correction)
//   = prescribed_v[ess] on essential DOFs (deltaF puts them there exactly)
//   = v_n + Δv elsewhere (the projected correction)
v_initial = v_n + deltaF + Δv;

// Now run Newton from v_initial.
Newton_from(v_initial);
```

Two key insights:

1. **`deltaF` is nonzero ONLY at essential DOFs.** It captures the change in
   corner displacement (or velocity, for ExaConstit's velocity primal). At
   non-essential DOFs deltaF = 0.
2. **`K_full @ deltaF` extracts the K_uc · Δv_c contribution.** Because deltaF
   has nonzero values only at essential cols (= corners), `K_full @ deltaF`
   at free rows equals K_uc · deltaF[ess] — exactly the change in residual at
   free rows caused by the BC change.

   The `K_eliminated` version would give zero (K_uc cols zeroed by
   elimination). So `GetUpdateBCsAction` must use the un-eliminated K — same
   K_full vs K_eliminated distinction we already saw in §6.4.

`GetUpdateBCsAction` implements this by temporarily setting the essential
TDOF list to *empty* on the local Jacobian (so the action of K is computed
as the full operator), then calling `local_jacobian.Mult(deltaF, y)`, then
restoring the original essential TDOF list. The previous-state residual is
added, and corner entries of the result are zeroed (so the inner CG solve
doesn't try to "fix" the essential rows, which are already correct).

## §7.3 Translation to displacement primal (our setting)

Our prototype's primal is u (displacement), not v (velocity). The translation:

| ExaConstit | Mortar PBC prototype |
|---|---|
| v_n converged at step n | u_n converged at step n |
| prescribed_v[ess] at step n+1 | u_lin[corner] at step n+1 = (F^{n+1} − I)·X[corner] |
| deltaF = prescribed_v[ess] − v_n[ess] at corners | deltaF[corner] = u_lin^{n+1}[corner] − u_n[corner] = (F^{n+1} − F^n)·X[corner] |
| K_n = local Jacobian at v_n | K_n = K = ElasticityIntegrator(λ, μ) — independent of u for linear elastic |
| ΔR_u = -K_uc · Δv_c | ΔR_u = -K_uc · deltaF |
| Solve K_e Δv = -(R^n + ΔR_u) | Solve [[K_e, Cᵀ], [C, 0]] [Δv, Δλ] = [-(R^n + ΔR_u), -C·deltaF] |
| v_initial = v_n + deltaF + Δv | u_initial = u_n + deltaF + Δv |

Two key differences:

1. **The constraint coupling**: ExaConstit's `SolveInit` is a *bare* CG solve,
   no Lagrange multipliers. Our setting has the mortar constraint, so the
   warm-start projection is itself a saddle-point solve (using the same
   `SaddlePointSolver` we use for the main Newton step). This ensures the
   projected initial state is *also* mortar-periodic.

2. **R^n is zero in linear elastic**: for our prototype, the previous step
   converged to machine precision (linear system), so R^n = 0. The R^n term
   is included for nonlinear / sub-converged future use.

## §7.4 Derivation of the projection equation

We now derive the projection equation explicitly. Suppose at step n the
state (u^n, λ^n) satisfies, after corner BC are applied:

    K(u^n) · u^n + Cᵀ λ^n = 0     (force balance on free DOFs)        (7.1a)
    C · u^n               = 0     (mortar periodicity)                (7.1b)

with corner DOFs already at u_lin^n[corner].

At step n+1, prescribe new corner values: u^{n+1}[corner] =
u_lin^{n+1}[corner]. The free DOFs and λ are unknown. We seek an *initial
iterate* u^{n+1, 0} = u^n + Δu that:

(i) Has the new corner values exactly: u^{n+1, 0}[corner] =
    u_lin^{n+1}[corner].
(ii) Approximately satisfies (7.1a) with K linearised at u^n.
(iii) Exactly satisfies (7.1b) for the new state.

From (i): Δu[corner] = u_lin^{n+1}[corner] − u^n[corner] =
u_lin^{n+1}[corner] − u_lin^n[corner] = (F^{n+1} − F^n) · X[corner], let's
call this **deltaF**.

So we decompose Δu = deltaF + Δv, where deltaF has nonzero entries only
at corners, and Δv has zero corner entries (free-DOF correction).

Linearise (7.1a) about u^n:

    K(u^n) · (u^n + Δu) + Cᵀ (λ^n + Δλ) = 0
    K(u^n) · u^n + K(u^n) · Δu + Cᵀ λ^n + Cᵀ Δλ = 0
    R^n + K(u^n) · Δu + Cᵀ Δλ = 0                                     (7.2)

where R^n := K(u^n) · u^n + Cᵀ λ^n is the residual at step n (zero at
clean convergence; nonzero if step n didn't quite converge — we capture
this term for robustness).

Substitute Δu = deltaF + Δv into (7.2):

    R^n + K · (deltaF + Δv) + Cᵀ Δλ = 0
    K · Δv + Cᵀ Δλ = − R^n − K · deltaF                               (7.3a)

Linearise (7.1b):

    C · (u^n + Δu) = 0
    C · u^n + C · Δu = 0
    0 + C · (deltaF + Δv) = 0
    C · Δv = − C · deltaF                                             (7.3b)

Stack (7.3a) and (7.3b) into the saddle-point form:

    ┌ K_e   Cᵀ ┐ ┌ Δv ┐   ┌ −(R^n + K_full · deltaF) ┐
    │          │ │    │ = │                           │              (7.4)
    └ C     0  ┘ └ Δλ ┘   └       − C · deltaF        ┘

with corner rows handled as in §6.4: K_e (eliminated K) is used in the
saddle-point top block (with corner Dirichlet built in via the identity
rows), but `K_full · deltaF` is computed using the FULL un-eliminated
K because deltaF is nonzero at corners (the K_uc · deltaF[corner] term
matters — see §6.4 trap 1).

After solving (7.4), the warm-start initial iterate is:

    u^{n+1, 0} = u^n + deltaF + Δv                                    (7.5)

with corners at u_lin^{n+1}[corner] (because deltaF supplies the change
exactly at corners and Δv has zero corner entries). λ^{n+1, 0} =
λ^n + Δλ.

**For linear K**, (7.4) IS the exact Newton step from u^n + deltaF (which
already has correct corners but wrong free-DOF values), and Δv brings
the free DOFs to the new equilibrium in one solve. Newton has nothing
left to do at step n+1 — see §7.5.

**For nonlinear K**, (7.4) gives an *initial iterate* in Newton's basin
of attraction; Newton then converges in 2-3 iterations rather than
5-10 if started cold from u^n + deltaF (which has corner-induced
imbalance) or even more iterations if started from u^n (where corners
are wrong).

## §7.5 Why warm-start is degenerate for linear elastic

For a fully-linear problem, each step is independent: the answer at step n+1
is determined entirely by F^{n+1} and the geometry/material; it does *not*
depend on the step-n state at all. The "warm-start projection" with linear K
gives the *exact* answer in one solve — there's nothing left for Newton to do.

So in the linear-elastic prototype:

- `solve_first_step(F_1)`: builds u_lin^1, solves saddle-point for du,
  forms u^1 = u_lin^1 + du. This is an *independent* solve.
- `solve_next_step(F_2)`: in principle, applies the warm-start recipe and
  finds u_initial that's already at the new equilibrium. *In practice for
  linear elastic, this reduces to "solve fresh"* — same answer. We
  implement it as a re-invocation of `_solve_independently(F_2)` and
  document why.

The architecture is in place for the eventual nonlinear extension:

- `MortarPbcDriver2D` carries `K_op_full`, `K_op` (eliminated), `C_op`, `CT_op`,
  state `u_par`, `lam_par`, `F_prev`.
- `solve_next_step` for nonlinear materials would:
  1. Compute deltaF: zero everywhere, fill corners with `(F^{n+1} − F^n)·X[corner]`.
  2. Compute b = K_full · deltaF, zero corner entries.
  3. Add R^n if available (zero at clean convergence).
  4. Solve saddle-point for (Δv, Δλ) per (7.4).
  5. u_initial = u_n + deltaF + Δv. Set Newton's initial iterate.
  6. Run Newton from u_initial.

This recipe is documented in `MortarPbcDriver2D.solve_next_step` for direct
translation when the Newton outer loop is added back (after pyMFEM's
NeoHookean integrator is fixed or replaced).

## §7.6 Subtlety: "prev-state mesh-coordinate corruption"

A trap we hit: the visualization writer was warping the mesh nodes after each
solve and *not* restoring them to reference. Subsequent calls to
`apply_linear_part(fes, F^{n+1})` projected `(F^{n+1} − I) X` against the *deformed*
mesh nodes, giving u_lin values that grew with each step (the affine field
was being applied to already-displaced X coordinates).

Symptoms:
- u_lin at step k looked "more stretched" than it should be by a factor of (1 + cumulative-strain).
- The volume-averaged-F diagnostic *still showed* ⟨F⟩ = F_macro to
  machine precision — because both `apply_linear_part` and `compute_volume_averaged_F`
  used the same deformed mesh. They were internally consistent with each other,
  consistent with the wrong reference.
- The SciPy direct cross-check failed by ~6%, because the K matrices were
  *static* (assembled at start, never touched), so they corresponded to the
  reference mesh, but the gathered u_lin at the verification block was
  computed against the deformed-from-step-3 mesh. Two different reference
  frames in the same linear system.

The fix: `PbcVisualizationWriter.write_step` now resets the mesh to the
reference snapshot *after* saving each cycle. The writer is side-effect-free
with respect to the mesh; every operation outside the writer always sees the
reference configuration.

This is the **total-Lagrangian discipline** in code form. See §9 for the
broader framing.

---

# §8. Diagnostics: volume-averaged F as the consistency check

## §8.1 The Hill-Mandel average theorem

[Hill 1972; Mandel 1972] establish that for a heterogeneous body Ω in a
homogenisation context, the macroscopic stress-strain pair must derive
from a microscale BVP whose volume-averaged kinematics equal the
prescribed macroscale F. We verify this for the periodic case explicitly.

Decompose u = u_lin + ũ on Ω, with u_lin = (F_macro − I) X and ũ
periodic on opposite faces of ∂Ω.

The deformation gradient F = I + ∇u = I + ∇u_lin + ∇ũ. Its volume
average is:

    ⟨F⟩_Ω = (1/V_Ω) ∫_Ω F dV
          = (1/V_Ω) ∫_Ω (I + ∇u_lin + ∇ũ) dV
          = I + (1/V_Ω) ∫_Ω ∇u_lin dV + (1/V_Ω) ∫_Ω ∇ũ dV             (8.1)

The first integral evaluates to:

    (1/V_Ω) ∫_Ω ∇u_lin dV = (1/V_Ω) ∫_Ω (F_macro − I) dV
                          = F_macro − I                                (8.2)

since (F_macro − I) is constant. The second integral is the key — we
claim it vanishes for periodic ũ.

**Proposition** (Hill-Mandel for periodic boundary):

    ∫_Ω ∇ũ dV = 0     for ũ Ω-periodic.                                (8.3)

**Proof.** Apply the divergence theorem (Gauss's theorem) componentwise.
The (i,j) component of ∇ũ is ∂ũ_i / ∂X_j, so:

    ∫_Ω (∇ũ)_{ij} dV = ∫_Ω ∂ũ_i / ∂X_j dV = ∮_{∂Ω} ũ_i N_j dA          (8.4)

In tensor form: ∫_Ω ∇ũ dV = ∮_{∂Ω} ũ ⊗ N dA.

Partition ∂Ω into pairs of opposite faces (Γ_k^+, Γ_k^-) for k = 1, …, d.
On the pair (Γ_k^+, Γ_k^-) the outward unit normals are N^+ = +e_k and
N^- = −e_k respectively (axis-aligned cube; the argument generalises by
periodic identification for arbitrary periodic shapes).

Periodicity says ũ takes the same value at points X ∈ Γ_k^- and Π(X) ∈
Γ_k^+ where Π is the periodic mapping. So on the pair:

    ∫_{Γ_k^+} ũ ⊗ N^+ dA + ∫_{Γ_k^-} ũ ⊗ N^- dA
    = ∫_{Γ_k^+} ũ ⊗ (+e_k) dA + ∫_{Γ_k^-} ũ ⊗ (−e_k) dA
    = (∫_{Γ_k^+} ũ dA − ∫_{Γ_k^-} ũ dA) ⊗ e_k                          (8.5)

By periodicity of ũ and the area-preserving mapping Π:

    ∫_{Γ_k^+} ũ dA = ∫_{Γ_k^-} ũ dA                                    (8.6)

so (8.5) is zero. Summing over all d pairs of opposite faces:

    ∮_{∂Ω} ũ ⊗ N dA = 0    ⟹    ∫_Ω ∇ũ dV = 0.    ∎

Substituting (8.2) and (8.3) into (8.1):

    ⟨F⟩_Ω = I + (F_macro − I) + 0 = F_macro.                           (8.7)

**Implication.** ⟨F⟩_Ω = F_macro **independent of any internal
heterogeneity, mesh refinement, or constitutive law**. The result holds
whenever ũ is *exactly* periodic. It's a property of the kinematic
constraint, not of the elastic problem.

This makes the volume-averaged F the *single most important consistency
check* on any PBC implementation:

- If ⟨F⟩ = F_macro to machine precision: the discrete periodicity is
  right AND the displacement field is correct (modulo the reference-
  frame caveat — see §8.3).
- If ⟨F⟩ ≠ F_macro: something is wrong. Either the constraint isn't
  enforcing periodicity correctly, or the corner Dirichlet isn't right,
  or the post-processing is using the wrong mesh state, or the
  integration is subtly off.

## §8.2 Implementation

`mortar_pbc.compute_volume_averaged_F(pmesh, fes, u_par)`:

```python
for each local element e:
    eltrans = fes.GetElementTransformation(e)
    ir = mfem.IntRules.Get(fe.GetGeomType(), 2*order+1)
    for each Gauss point q:
        eltrans.SetIntPoint(q)
        w = q.weight * eltrans.Weight()
        gf_u.GetVectorGradient(eltrans, grad_u_at_qp)
        accumulate w * grad_u_at_qp into grad_u_acc
        accumulate w into vol_acc
allreduce(grad_u_acc, vol_acc)
return I + grad_u_acc / vol_acc
```

This is dimension-agnostic — works in 2D and 3D unchanged. The integrand
`grad_u_at_qp` is dim×dim. In 3D we Allreduce 9 doubles instead of 4.

## §8.3 What ⟨F⟩ catches

The diagnostic catches:

- Constraint matrix C built incorrectly (e.g. wrong dual basis, missing
  Wohlmuth modification, wrong nonmortar/mortar pairing).
- Corner Dirichlet applied at the wrong values.
- Mesh-state-corruption in post-processing (the "deformed mesh as reference"
  bug from §7.6).
- Integration order too low (would produce small-but-nonzero error).

The diagnostic does *not* catch:

- Bugs internal to the FE assembly (e.g. wrong material tensor) — those
  show up as wrong stress, not wrong ⟨F⟩.
- Sub-converged Newton (the diagnostic measures ⟨F⟩ for whatever u_par was
  passed; if u_par is sub-converged, ⟨F⟩ may still match F_macro because
  the constraint is satisfied even if equilibrium isn't).

## §8.4 PASS criterion threshold

For our 2D prototype: `|⟨F⟩ − F_macro|_max < 1e-9`. Linear elastic with
direct-quality Krylov convergence, this should typically be `< 1e-13` —
machine precision. The 1e-9 threshold is loose enough to allow for some
preconditioner-quality slack while still being orders of magnitude below
"physically correct" tolerances.

For 3D, the threshold should hold (1e-9 or tighter). The integral is
direction-symmetric, so 3D doesn't change the precision target.

---

# §9. Visualisation and the total-Lagrangian discipline

## §9.1 The discipline

All operations on the FE mesh — assembly, projection, gradient evaluation,
integration, residual computation, K computation — happen on the **reference
configuration**. The deformed mesh is purely a visualisation artefact. We
never compute against the deformed mesh.

This is the **total-Lagrangian** convention. ExaConstit, despite using
"updated-Lagrangian" terminology at the macroscopic time-step level, uses
total-Lagrangian within each load step's solve: the integrator references
the reference configuration to evaluate F, σ, K. ExaConstit's "updated"
aspect is that *between* load steps, the converged state propagates as the
new initial state — but the reference geometry doesn't actually change. (This
is a mild abuse of terminology in the field; the distinction matters less
than the practice.)

## §9.2 Why this matters in code

Two specific places where the reference-vs-deformed distinction got us into
trouble:

1. **`apply_linear_part(fes, F)`**. Internally calls
   `gf.ProjectCoefficient(coef)` where `coef.EvalValue(x)` returns
   `(F − I) · x`. The "x" here is whatever the *current* mesh's nodal
   coordinates are. If the mesh has been warped to deformed, `x = X + u_prev`,
   and `apply_linear_part` returns `(F − I) (X + u_prev)` — a function of the
   accumulated displacement, not the reference position. This silently
   produces wrong u_lin values.

2. **`compute_volume_averaged_F(pmesh, fes, u_par)`**. Calls
   `gf_u.GetVectorGradient(eltrans, grad_u_at_qp)`. The `eltrans` is built
   from the mesh's current nodal coordinates. ∇u in the deformed
   configuration ≠ ∇u in the reference configuration (they differ by the
   deformation gradient itself, which is the very thing we're trying to
   compute). If the mesh is deformed, ⟨F⟩ from this routine is wrong.

The fix is in `PbcVisualizationWriter`: on every `write_step`, *reset* the
mesh to the reference configuration *after* saving the deformed cycle. The
writer is the only piece of code that ever touches the mesh nodes; every
other operation sees the reference.

## §9.3 The mesh-node update mechanics

To "reset to reference" requires:

1. Snapshot the reference node coordinates at `PbcVisualizationWriter`
   construction time, before any solve runs.
2. To warp: read the reference snapshot, add the displacement, write back.
3. To reset: read the reference snapshot, write back unchanged.
4. After every reset/warp, call `pmesh.NodesUpdated()` to invalidate cached
   geometric factors (otherwise MFEM will use stale `eltrans` from before the
   nodes changed).

The MFEM API for this:

```python
nodes_gf = pmesh.GetNodes()                     # ParGridFunction of node coords
ref_tdofs = mfem.Vector()
nodes_gf.GetTrueDofs(ref_tdofs)                 # snapshot at ctor time
ref_snapshot = np.array(ref_tdofs.GetDataArray(), copy=True)

# Later: reset to reference
for i in range(ref_tdofs.Size()):
    ref_tdofs[i] = float(ref_snapshot[i])
nodes_gf.SetFromTrueDofs(ref_tdofs)
pmesh.NodesUpdated()
```

## §9.4 The byNODES vs byVDIM ordering trap

A subtle MFEM-default trap: when you build a vector FE space via
`ParFiniteElementSpace(pmesh, fec, vdim=dim)`, the default ordering is
**Ordering::byNODES**. When you call `pmesh.SetCurvature(order)`, the default
ordering of the resulting nodal grid function is **Ordering::byVDIM**.

These are different layouts:
- `byNODES`: TDOFs listed as `[u_x(0), u_x(1), ..., u_x(N), u_y(0), ..., u_y(N), ...]`
- `byVDIM`: TDOFs listed as `[u_x(0), u_y(0), u_x(1), u_y(1), ...]`

If your displacement FES is byNODES and your mesh-nodes FES is byVDIM,
`for i in range(n_tdof): nodes[i] += u_par[i]` silently swaps x and y
components, producing a 90°-rotated warp.

The fix: explicitly pass the desired ordering to `SetCurvature`:

```python
pmesh.SetCurvature(order=1, discont=False, space_dim=-1, ordering=fes.GetOrdering())
```

Now the nodal grid function shares the displacement FES's ordering. The unit
test `_ensure_nodal_with_matching_ordering` handles this defensively, and
`_warp_mesh_by` asserts the orderings match before mutating.

---

# §10. Status at the Phase-2 ↔ Phase-3 boundary

## §10.1 Verified-passing as of this commit

| Test | Verified |
|---|---|
| Unit tests, 2D suite (6 tests) | PASS on np=1; pure-Python, no MPI |
| Unit tests, 3D Phase 3.2.A suite (25 tests) | PASS on np=1; pure-Python, no MPI |
| Unit tests, 3D Phase 3.2.B suite (11 tests) | PASS on np=1; pure-Python, no MPI |
| Unit tests, 3D Phase 3.3.A suite (4 tests) | PASS on np=1; verifies `MortarAssembler2D` reuse on `EdgeInfo3D` (axis-generic dispatch, x/y/z symmetry) |
| Unit tests, 3D Phase 3.3.B helpers (8 tests) | PASS on np=1; pure-Python helpers in `BoundaryClassifier3D` (boundary-tag dispatch incl. 3-sentinel quad, axis inference, face-bounding edges, CCW reordering, end-to-end sentinel-tagged assembler dispatch) |
| Unit tests, 3D Phase 3.3.C suite (5 tests) | PASS on np=1; pure-Python with synthetic 2×2×2 mock classifier (row count, constant-field nullspace, affine-field jump, linearity, sparsity / face-row column targeting) |
| `examples/patch_test_2d.py` (Phase 1B linear-elastic baseline) | PASS np = 1, 2, 4, 8 |
| `examples/patch_test_2d_heterogeneous.py` (5× strip-split, multi-step) | PASS np = 1, 2, 4, 8 with `--F=uniaxial`, `--F=shear`, `--F=mild-shear`, `--steps=1..N` |
| `examples/patch_test_2d_checkerboard.py` (5× 4-quadrant XOR, multi-step) | PASS np = 1, 2, 4, 8, all F choices |
| `examples/patch_test_3d_homogeneous.py` (Phase 3.1 hex+tet, full-∂Ω Dirichlet) | PASS np = 1, 2, 4, 8 with `--mesh-type hex` and `--mesh-type tet`; `--paraview` validates visually |
| `examples/probe_boundary_classifier_3d.py` (Phase 3.3.B integration smoke-test) | PASS np = 1, 4 with `--mesh-type hex` and `--mesh-type tet` |
| `examples/probe_constraint_builder_3d.py` (Phase 3.3.D integration smoke-test) | Pending Robert's macOS validation; sandbox lacks pyMFEM |

The 3D Phase 3.2.A unit suite (`tests/test_mortar_3d_unit.py`) verifies:

- Lumped-positivity precondition (§4.9.1) for all 9 element types in
  scope, with correct sign pattern: line-2 / line-3 / tri-3 / quad-4 /
  quad-9 / tet-4 all-positive (PASS list); tri-6 corner = 0; quad-8
  corner < 0; tet-10 corner < 0 (FAIL list, see §4.9.2 for the
  dimension-dependent simplex pattern).
- Bi-orthogonality of M_tri3_dual, M_quad4_dual, M_tet4_dual on
  reference elements to ~1e-16 precision.
- Partition of unity of all standard FE shape functions and the
  implemented dual bases.
- Wohlmuth modifications (eqs. 5.5, 5.6, 5.8, 5.10): tri-3 with 0/1/2/3
  vertices dropped; quad-4 edge-adjacent and corner-adjacent.
- Conforming-pair lumping recovery (eq. 3.8) on the *kernel* level
  (single-element bi-orthogonality verification).

The 3D Phase 3.2.B unit suite (`tests/test_face_mortar_3d.py`) verifies
the face-mortar *assembler* (the pure-Python LOOP layer that consumes
QuadFaceElement / TriFaceElement data and produces FaceMortarPairBlock):

- Lumped-positivity construction guard: `QuadFaceMortarAssembler()` /
  `TriFaceMortarAssembler()` instantiate cleanly; a hypothetical
  tri-6-style broken-basis subclass raises `RuntimeError` at __init__.
- Single-element conforming-pair recovery for quad-4 and tri-3:
  D = A_m = (face_area / n_nodes) · I_n to ~1e-13 precision.
- 2×2 grid quad-4 conforming pair: D pattern = (1, 2, 1, 2, 4, 2, 1,
  2, 1) · 0.25 (matches per-node sub-element-count weighting); A_m =
  diag(D).
- Sentinel-row drop on quad-4 with `gtdofs = (0, -1, 1, 2)`: the
  corresponding row is absent from D and A_m; off-diagonal mortar-col
  zero-pattern matches the kept (3, 4) block.
- Wohlmuth corner-LL modification on quad-4: corner row dropped via
  sentinel; D rows unchanged from unmodified case (D uses standard N,
  not modified M); A_m row sums DIFFER (modification active);
  modified dual partition-of-unity preserved at every Gauss point.
- Wohlmuth tri-3 v0 (one-vertex-dropped, edge-adjacent): kept (2, 3)
  block; cols (1, 2) = I_2 ((|T|/3) per diagonal); col 0 leak = 0.5
  (non-zero, consistent with eq. 5.5 verification — the "harmless
  leak" into the dropped corner column).
- `match_conforming_face_pairs` helper: 9-element grid pairs with
  identity perm; shuffled-mortar order recovered correctly;
  non-conforming 2×2 vs 3×3 raises `RuntimeError`.

PASS criteria, unified across drivers:

- Krylov converges (`sps.last_converged == True`).
- `||C u_tilde||_2 < 1e-8` (constraint residual, machine precision typical).
- `||u_tilde||_inf > 1e-12` (heterogeneous must produce non-trivial fluctuation).
- `||du_krylov − du_direct||_inf < 1e-6` (Krylov vs. SciPy direct
  cross-check; typically ~1e-13 in practice).
- `|⟨F⟩ − F_macro|_max < 1e-9` (homogenization consistency; typically ~1e-15).

**Doc correction surfaced during Phase 3.2 implementation.** The
original §4.9.2/§4.9.3 claimed tet-10 corner s = 0 by analogy with
tri-6. Direct numerical evaluation (matching the closed-form
arithmetic) gives s_corner = −|T|/20 = −1/120 instead. The §4.9
section now contains the corrected dimension-dependent simplex
formula (eq. 4.28b): s_corner_P2 = (2−d)/((d+1)(d+2)) · |T|, which
is positive for d=1, zero only at d=2, and negative for d≥3. This
sharpens the predictive lumped-positivity rule and is exactly the
kind of correction the unit-test suite was designed to surface.

**Doc correction surfaced during Phase 3.1 macOS validation.** The
original §11.8 Phase 3.1 design pinned only the 8 corners at u_lin
and predicted u = u_lin elsewhere "because the affine field is the
exact solution." This is incorrect: with corner-only Dirichlet, the
rest of ∂Ω carries the natural BC σ·n = 0, which is incompatible
with the constant stress σ = C : sym(F-I) of the affine field.
Robert's macOS run produced ‖K · u_lin‖_∞ ≈ 589 (the integrated
boundary traction σ·n, NOT noise) and ‖du‖_∞ ≈ 7e-2 (a non-affine
minimum-energy field that satisfies σ·n = 0 on the free boundary).
The correction in §11.8 promotes Phase 3.1 to FULL Dirichlet on all
6 boundary faces at u_lin, which makes interior DOFs the only free
ones and recovers (K · u_lin)_i = 0 for all interior i (∫∇N_i dV = 0
for compactly-supported N_i). This is the standard linear-elasticity
patch test; the role of mortar PBC at Phase 3.4 is precisely to
*replace* the missing free-Neumann boundary tractions with periodic
nonmortar-mortar coupling, restoring well-posedness with only 8 corner
Dirichlets.

**MPI deadlock surfaced during Phase 3.1 np > 1 validation.** The
3D driver originally had `n_global_elements = pmesh.GetGlobalNE()`
inside an `if rank == 0:` block. `ParMesh::GetGlobalNE()` is a
COLLECTIVE in MFEM (it does an internal `MPI_Allreduce` summing
per-rank element counts across the ParMesh communicator); calling it
only on rank 0 strands rank 0 inside the Allreduce while ranks 1..N-1
fly past and reach the next collective (`ParFiniteElementSpace`)
alone. Symptom: clean execution at np = 1, hang after the first
collective at np ≥ 2. The fix — call collectives on ALL ranks, then
guard only the print with `if rank == 0` — was already documented
in §11.7 but missed in the 3D driver. The same trap was warned
about explicitly in `examples/patch_test_2d.py` lines 649-654; we
now have a matching warning comment in the 3D driver and a §10.4
"distributed-driver invariants" subsection summarising the rule.

## §10.2 What the prototype currently provides

Capabilities:
1. 2D mortar PBC for non-conforming RVE meshes (rectangular geometry).
2. Linear elastic constitutive model via `ElasticityIntegrator` +
   `PWConstCoefficient` for piecewise-constant Lamé parameters.
3. Method D (total-displacement primal) with corner Dirichlet at u_lin[corner]
   and mortar fluctuation periodicity.
4. Wohlmuth-modified dual basis at corner crosspoints (Lopes Eq. C.2),
   verified by unit test.
5. Distributed Krylov saddle-point solver (GMRES + block-Jacobi prec).
6. Multi-step driver with ExaConstit-style warm-start architecture (degenerate
   for linear elastic; ready for nonlinear extension).
7. Volume-averaged F homogenization diagnostic.
8. ParaView visualization (multi-cycle, mesh-node-warped, byNODES/byVDIM
   robust).
9. SciPy direct cross-check on rank 0 for verification.

Code structure:

```
mortar_pbc_proto/
├── README.md                                        # Quickstart
├── PROJECT_STATUS.md                                # Pre-Phase-3 status
├── docs/
│   └── MORTAR_PBC_ARCHITECTURE.md                   # This document
├── mortar_pbc/                                       # Pure-Python package
│   ├── __init__.py                                  # Lazy-loaded public API
│   ├── types_2d.py                                  # EdgeNodes2D, CornerInfo
│   ├── boundary_2d.py                               # BoundaryClassifier2D
│   ├── mortar_2d.py                                 # Dual basis + MortarAssembler2D
│   ├── constraint_builder.py                        # ConstraintBuilder2D
│   ├── constraint_assembler.py                      # ABC + stack_constraints
│   ├── saddle_point.py                              # SaddlePointSolver, prec
│   ├── multistep_driver.py                          # MortarPbcDriver2D + ⟨F⟩ diagnostic
│   ├── visualization.py                             # PbcVisualizationWriter
│   ├── diagnostics.py                               # General diagnostic helpers
│   └── _verify_solver.py                            # SciPy direct (quarantined)
├── examples/
│   ├── patch_test_2d.py                             # Phase 1B baseline
│   ├── patch_test_2d_heterogeneous.py               # Strip-split, multi-step
│   ├── patch_test_2d_checkerboard.py                # 4-quadrant XOR, multi-step
│   └── diag_neohookean_2x2.py                       # NeoHookean NaN diagnostic
└── tests/
    └── test_mortar_2d_unit.py                        # 6 unit tests
```

## §10.3 What the prototype doesn't do (and why)

1. **NeoHookean / nonlinear material**: pyMFEM's `NeoHookeanModel` produces NaN
   at u=0 across all constructor variants tested in this build (uniaxial F,
   single-material, multi-material, scalar-coefficient, Coefficient-coefficient).
   We pivoted to linear elastic for the prototype. Diagnostic preserved in
   `examples/diag_neohookean_2x2.py`. Replacement strategies for the production
   ExaConstit port: (a) write a custom `HyperelasticModel` subclass that's
   numerically robust at u=0; (b) use a different MFEM build; (c) skip
   NeoHookean and go straight to crystal plasticity (which is the actual
   target). Linear elasticity is sufficient for prototyping the mortar PBC
   machinery itself.

2. **Newton iteration**: with linear elastic K, each step converges in one
   solve. The `MortarPbcDriver2D.solve_next_step` documents the warm-start
   recipe but for linear elastic implements it as a single fresh solve per
   step. Phase-2's earlier neo-Hookean Newton outer loop is preserved in
   transcript form for re-introduction when the integrator is fixed.

3. **Tribol integration for general non-conforming geometry**: deferred. We
   built our own mortar machinery to (a) understand the method, (b) own the
   integration into ExaConstit's PA path. Tribol may be revisited as an
   alternative dual-basis / non-conforming geometry-matching backend; current
   prototype handles axis-aligned 2D directly.

4. **3D**: nothing yet. That's Phase 3, the subject of §11.

5. **Uniform Traction (UT) BCs**: deferred but architectural hook is in place
   (`ConstraintAssembler` ABC + `stack_constraints` helper). Adding UT later
   is a matter of writing one new `UniformTractionConstraintAssembler` and
   stacking it.

6. **C++ ExaConstit port**: planned for Phase 5. See §13 for design.

## §10.4 Distributed-driver invariants (the rank-asymmetric-collective trap)

This rule has bitten the codebase twice — once in 2D (where it's
explicitly warned against in `examples/patch_test_2d.py` lines
649-654) and once in 3D (Phase 3.1, surfaced during Robert's macOS
np = 4 validation). It deserves a centralised statement.

**Rule.** A function that internally uses MPI collectives must be
called by ALL ranks at the same point in program order. Wrapping
such a call in `if rank == 0:` causes rank 0 to enter the collective
alone and block waiting for ranks 1..N-1, who fly past and reach the
NEXT collective alone, who block waiting for rank 0. Deadlock.

**Three-line failure pattern (illustrative).**

```python
# WRONG — deadlocks at np > 1:
if rank == 0:
    n = pmesh.GetGlobalNE()        # collective: MPI_Allreduce inside
    print(f"global elements = {n}")

# RIGHT:
n = pmesh.GetGlobalNE()             # collective on all ranks
if rank == 0:                        # rank-0-only print is fine
    print(f"global elements = {n}")
```

**Known collectives in MFEM that look like local accessors.** Most
of these run inside `if rank == 0:` blocks "by mistake" because
their names suggest a property query rather than a communication:

- `Mesh::GetGlobalNE()` (when `*this` is a ParMesh) → MPI_Allreduce
- `Mesh::GetGlobalNV()` (when ParMesh) → MPI_Allreduce
- `ParGridFunction::ComputeL2Error(...)` → MPI_Allreduce
- `ParGridFunction::Norml2()` / `Norml1()` / `Normlinf()` → MPI_Allreduce
- `ParBilinearForm::Assemble()` and `ParallelAssemble()` → MPI internal
- `ParFiniteElementSpace::GetEssentialTrueDofs(...)` → has a parallel
  fix-up step; at minimum participates in any later assembly fence
- The constructors `ParMesh(comm, mesh)`, `ParFiniteElementSpace(...)`,
  `HypreBoomerAMG(K_par)`, `HypreParMatrix::ParAdd(...)`, etc. —
  collective by definition.

**Known collectives in mpi4py that DEFINITELY require all ranks.**

- `comm.Allreduce(...)`, `comm.Allgather(...)`, `comm.Bcast(...)`,
  `comm.Barrier()`, `comm.Reduce(...)` — but `Reduce` on root only is
  fine if all ranks call it; the asymmetry is in WHICH ranks call,
  not what they pass.

**Robust pattern for diagnostic prints.** When the value to print is
the result of a collective:

```python
# Compute on all ranks (collective participates everywhere).
val = some_collective_call(...)

# Print on rank 0 only (no further collective implied).
if rank == 0:
    print(f"  diagnostic: {val}")
```

When the value is a per-rank quantity that needs to be summed for the
print (e.g., per-rank TDOF counts → global TDOF count):

```python
# Allreduce on all ranks (collective).
local = compute_local(...)
total = comm.allreduce(local, op=MPI.SUM)

# Print on rank 0 only.
if rank == 0:
    print(f"  global total: {total}")
```

**When in doubt, instrument.** A `comm.Barrier()` call right before a
suspicious `if rank == 0:` block will surface the deadlock immediately:
the Barrier requires all ranks. If rank 0 enters the Barrier and the
others reach it from the next collective, they all unstick and the
program continues to the actual deadlock site, making it diagnosable.

This is purely an interface-discipline problem; there's no clever
runtime detection in MPI. Audit drivers against the pattern above
before declaring an np > 1 run "working".

**Rank-local vs. global indices in cross-rank dedup.** A related
trap surfaced during Phase 3.3.B macOS validation: ``ParMesh``
vertex indices, element indices, and boundary-element indices are
ALL rank-local. Vertex 27 on rank 0 is unrelated to vertex 27 on
rank 1 — they're indices into each rank's own local arrays. When
AllGather'ing per-rank records that need cross-rank deduplication
(e.g., merging boundary-vertex attribute sets across ranks), keying
the merge dictionary by the rank-local vertex index causes silent
data collisions: the rank-1 record overwrites the rank-0 record
under the same dictionary key, even though they refer to physically
different vertices.

**The fix is to use a globally-meaningful key.** Two patterns work:

1. **Snapped physical coordinates** (used by ``boundary_2d`` and
   ``boundary_3d``): ``key = round(coord / tol)`` as a tuple. Stable
   across ranks because every rank computes the same key from the
   same physical position. Requires the parent mesh to use the same
   coordinate values across ranks (true for serial-mesh-then-
   ParMesh-partition; would need extra care for distributed mesh
   readers with curved boundaries).

2. **Global TDOF numbers** (used in ``ConstraintBuilder2D``): when
   the records being merged correspond to FE DOFs, ``GetGlobalTDofNumber``
   returns the same global index from any rank that knows the DOF.
   This is preferable when available because it sidesteps coordinate-
   precision concerns entirely.

The general lesson: **never use a rank-local index as a key in a
data structure shared across ranks**. The ``parent_vertex_id`` field
on ``_VertexRecord`` was renamed to ``pvid`` (a synthetic global
counter) once this was understood, to make it a positive cue not to
confuse it with the rank-local parent-vertex index it was originally
populated from.

## §10.5 MFEM API conventions for attribute arrays (a foot-gun)

Two MFEM APIs that both take an `Array<int>` of "attributes" use
**different conventions** for what the array contents mean. This
caused a complete classification failure in Phase 3.3.B that
produced "found 0 corners" with no other diagnostic. Documenting
the distinction here so it doesn't bite again.

**Boolean-mask convention** (used by `GetEssentialTrueDofs` and most
solver-level APIs):

- Array length = `bdr_attributes.Max()`.
- Entry `i` = 1 selects attribute `i + 1`; entry `i` = 0 deselects.
- Standard usage:
  ```python
  ess_bdr = mfem.intArray(n_bdr_attrs)
  ess_bdr.Assign(1)                  # select all
  fes.GetEssentialTrueDofs(ess_bdr, list)
  ```

**Attribute-list convention** (used by `SubMesh::CreateFromBoundary`,
`SubMesh::CreateFromDomain`, and similar mesh-derivation APIs):

- Array length = number of attributes you want to select.
- Each entry IS the attribute integer, listed once per selection.
- Correct usage to select all 6 boundary faces:
  ```python
  attrs = mfem.intArray(6)
  for i in range(6):
      attrs[i] = i + 1               # values [1, 2, 3, 4, 5, 6]
  ParSubMesh.CreateFromBoundary(parent, attrs)
  ```
- Passing `[1, 1, 1, 1, 1, 1]` as a "boolean mask" instead returns a
  submesh of just attribute 1, repeated six times = one face's worth.
  No error message — the call silently succeeds with a partial
  result. Symptom in our Phase 3.3.B run: classifier produced 25
  vertices on a 4×4×4 hex (the bottom-face vertex count) instead of
  the expected 98 boundary vertices.

**Rule of thumb when adding a new MFEM call that takes an `Array<int>`
of attributes:** check the MFEM source. If the function name suggests
selecting/extracting (CreateFromX, ExtractX, RestrictTo), it almost
certainly takes the attribute-list convention. If the function name
suggests configuring or marking essential/Dirichlet conditions,
it probably takes the boolean-mask convention. When in doubt, write
a 5-line probe with debug output that exercises both cases on a
small mesh and inspect the resulting submesh / DOF-list size.

---

# §11. Extending to 3D: the wirebasket framework

This is the road map for Phase 3. It exists in this document so that whoever
picks up the work — in this conversation or a future one — has a fully-stated
plan with all the math and architectural decisions called out. Don't start
coding without reading this section.

## §11.1 The hierarchy and what changes from 2D

The 2D RVE has 4 corners + 4 edges + (no faces because 2D). The 3D RVE has
8 corners + 12 edges + 6 faces. The constraint structure becomes
*hierarchical* in 3D:

- **Level 0 (Corners)**: essential Dirichlet, 8 corners × 3 components = 24
  TDOFs. No LM rows; no constraint participation.
- **Level 1 (Edges)**: mortar coupling, with corner LMs dropped. Each pair of
  periodic edges gets one constraint group. Wohlmuth modification at corner
  endpoints uses the existing 1D recipe.
- **Level 2 (Faces)**: mortar coupling, with edge LMs dropped. Each pair of
  periodic faces gets one constraint group. Wohlmuth modification at edge
  *boundary strips* — a 2D extension of the 1D corner modification.

The cascade ensures non-redundancy: each level constrains exactly the DOFs
that aren't already covered by a higher level.

The full constraint matrix C is then a vertical stack of three blocks:

```
C = [ C_edges_x ]   ←  3 mortar-coupled edge groups in x direction
    [ C_edges_y ]   ←  3 mortar-coupled edge groups in y direction
    [ C_edges_z ]   ←  3 mortar-coupled edge groups in z direction
    [ C_faces_yz ]  ←  3 face mortar pair (perpendicular to x)
    [ C_faces_xz ]  ←  3 face mortar pair (perpendicular to y)
    [ C_faces_xy ]  ←  3 face mortar pair (perpendicular to z)
```

(The actual organization may differ slightly — by face/edge group rather than
direction — but the overall stacking is what matters.)

This stacking is exactly the use case our existing `stack_constraints`
machinery (in `mortar_pbc/constraint_assembler.py`) was designed for. Each
level is a separate `ConstraintAssembler`, and `stack_constraints([...])`
produces the unified C.

## §11.2 The hex mesh track: hex-8 volumes with quad-4 face mortar

For hex-mesh RVEs, the periodic boundary structure uses:

| Level | Element class | Dual basis | Wohlmuth modification |
|---|---|---|---|
| 0 (corners) | hex-8 vertices | (none — essential) | (none) |
| 1 (edges) | line-2 (hex edge) | §4.2 (eq. 4.13) | §5.1 (eq. 5.2) |
| 2 (faces) | quad-4 (hex face) | §4.3 (eq. 4.16) | §5.3 (eq. 5.8 / 5.10) |

The full algorithmic recipe per face pair, hex-mesh case:

```
for each pair of opposite hex-faces (mortar_face, nonmortar_face):
    for each quad element Q in nonmortar_face:
        classify Q against face boundary:
            side_xi = "left" | "right" | "none"
            side_eta = "bottom" | "top" | "none"
        select dual basis: M_quad4_dual_modified(ξ, η, side_xi, side_eta)
        place 2D Gauss quadrature on Q's reference (ξ, η) ∈ [-1,+1]²
        for each Gauss point:
            x_q = T_Q(ξ, η)                          # physical point on nonmortar face
            x_m = Π(x_q)                             # periodic image on mortar face
            (ξ_m, η_m, mortar_quad_id) = locate(x_m, mortar_face)
            evaluate nonmortar M^mod at (ξ, η)
            evaluate mortar N at (ξ_m, η_m)
            accumulate D_local, A_m_local
        assemble into global D, A^m blocks
```

Reference for the formulation: [Lopes et al. 2021, §4.4.2; Wohlmuth 2001,
§1.3.4].

## §11.3 The tet mesh track: tet-4 volumes with tri-3 face mortar

For tet-mesh RVEs, the periodic boundary structure uses:

| Level | Element class | Dual basis | Wohlmuth modification |
|---|---|---|---|
| 0 (corners) | tet-4 vertices | (none — essential) | (none) |
| 1 (edges) | line-2 (tet edge) | §4.2 (eq. 4.13) | §5.1 (eq. 5.2) |
| 2 (faces) | tri-3 (tet face) | §4.4 (eq. 4.19) | §5.2 (eq. 5.5 / 5.6) |

The hierarchy (level 0 / 1 / 2 of §5.4) is identical; only the level-2
element class differs. Phase 3.2 must therefore implement BOTH dual bases
and dispatch on face element type.

The algorithmic recipe per face pair, tet-mesh case:

```
for each pair of opposite tet-faces (mortar_face, nonmortar_face):
    for each triangle element T in nonmortar_face:
        classify T against face boundary:
            boundary_nodes = (b1, b2, b3)  # per-vertex bool: on face boundary?
        select dual basis: M_tri3_dual_modified(λ, boundary_nodes)
        place 2D Gauss quadrature on T's reference simplex (barycentric)
        for each Gauss point (in barycentric coords):
            x_q = T_T(λ_1, λ_2, λ_3)                 # physical point on nonmortar face
            x_m = Π(x_q)                             # periodic image on mortar face
            (λ_m, mortar_tri_id) = locate(x_m, mortar_face)
            evaluate nonmortar M^mod at λ
            evaluate mortar N at λ_m
            accumulate D_local, A_m_local
        assemble into global D, A^m blocks
```

The differences from the hex case are mechanical:

- **Quadrature rule**: Dunavant rules [Dunavant 1985] for triangles instead
  of tensor-product Gauss for quads.
- **Geometric matching `locate`**: barycentric inverse via affine triangle
  transformation (more straightforward than inverse bilinear quad map,
  which requires a Newton iteration in the non-axis-aligned case).
- **Boundary classification**: per-vertex booleans (3 bits) vs.
  per-edge sides (4 sides on a quad, only relevant if the entire edge
  lies on the face boundary).

A subtle point: a tri-3 face element can have **3 boundary configurations
not present in the quad-4 case**:

1. **Single vertex on face boundary, no edge on face boundary**: only
   one vertex is "on" but the two adjacent edges of the triangle leave
   the boundary into the face interior. This is the typical case for a
   well-refined triangulated face and uses (5.5).
2. **One edge on face boundary**: two consecutive vertices are "on";
   the corresponding triangle edge lies along the face boundary. The
   edge-adjacent modification (eq. 5.5) applies twice — once per "on"
   vertex — but care must be taken that they aren't applied
   independently. The cleaner formulation: drop both vertices' rows;
   the third vertex's M ≡ 1 (this is the §5.2.3 corner-adjacent case
   structurally, even though geometrically the triangle is edge-adjacent
   not corner-adjacent).
3. **Two edges of triangle on face boundary** (i.e. the triangle is at
   a face corner): all three vertices are "on" *or* two are on and one
   is interior. The interior vertex's M ≡ 1; this is the (5.6) case.

Implementation note: pass `boundary_nodes` as the per-vertex bool tuple
and let the `M_tri3_dual_modified` function dispatch on the count
(§5.2.4). This gives the right behavior for all configurations
without case-by-case sign management.

## §11.4 Mixed hex-tet meshes

MFEM allows mixed-element meshes where some volume elements are hex-8
and others are tet-4 in the same `ParMesh`. ExaConstit users may build
such meshes for crystal-plasticity RVEs to mix structured grain
interiors (hex) with topology-conforming grain boundaries (tet).

Implications for PBC face mortar:

- **Each periodic face pair may have mixed face elements**. A periodic
  face on the y = 0 boundary may consist of some quad-4 faces (from hex
  elements bordering this face) and some tri-3 faces (from tet
  elements). The opposite y = L face has the *same* mix structurally —
  but possibly with different topology because the mesh on each face is
  generated independently.
- **Face mortar dispatches per-face**. Each nonmortar-side face element
  selects its dual basis (`M_quad4_dual_modified` or
  `M_tri3_dual_modified`) based on `face.geom_type`. The mortar-side
  face element, accessed via the geometric matching (§3.5), provides
  its own shape functions (`N_quad4` or `N_tri3`) and these are
  evaluated at the projected (ξ_m, η_m, ...) coordinates regardless of
  the nonmortar's element type.
- **Sub-element accuracy** for non-conforming pairs (Phase 3.5): the
  Sutherland-Hodgman clipping operates on convex polygons, indifferent
  to whether the polygon was a quad or a triangle. Cross-class clipping
  (quad nonmortar on tri mortar, or tri nonmortar on quad mortar) is the same
  algorithm.

The architecture: `MortarFaceAssembler` is a virtual base class with
concrete `QuadFaceAssembler` and `TriFaceAssembler` derivatives. The
`ConstraintBuilder3D` walks each face pair and dispatches the
appropriate assembler per nonmortar-side face element.

For Phase 3.4 (conforming-mesh first), we test:

- Pure hex RVE (all face elements are quad-4).
- Pure tet RVE (all face elements are tri-3).
- Mixed RVE (some hex, some tet on the same periodic face).

The mixed test is the hardest correctness check because it exercises
the polymorphic dispatch and the cross-element-class face matching.

## §11.5 The 3D edge mortar (line-2, common to hex and tet meshes)

3D edge mortar is element-class-independent: edges of hex-8 and tet-4
volumes are both line-2 [Lopes et al. 2021, §4.4.1]. The 2D edge mortar
infrastructure (`MortarAssembler2D`) carries forward; we re-use it.

Two complications versus 2D:

1. **Each edge has two corner endpoints** (1D corners), and the Wohlmuth
   modification (eq. 5.2) applies at both ends. The 1D recipe in
   `M_line2_dual_modified` already handles "left" and "right"; an
   edge-element adjacent to one corner uses one modification, adjacent
   to the other corner uses the other. The implementation works by
   passing `side ∈ {"left", "right", "none"}` per edge element.

2. **Each set of 4 parallel edges forms a periodic group**, not just a
   pair. The cube's 12 edges partition into 3 groups of 4 (one group
   per axis direction). Within each group, all 4 edges are periodic
   equivalents. The mortar coupling per group is:

   - Pick edge e₁ as mortar.
   - Couple e₂ ↔ e₁, e₃ ↔ e₁, e₄ ↔ e₁ via 3 line-2 mortar blocks.
   - Stack the LM rows: if each edge has n_int interior DOFs after
     dropping corners, the group's edge mortar produces 3 × n_int LM
     rows per spatial component (one per nonmortar-edge LM DOF, three
     nonmortar edges).

The constraint pseudocode for one direction's edge group:

```
for direction d in {x, y, z}:
    (mortar_edge, nonmortar_edges[3]) = group_parallel_edges(d)
    for each nonmortar edge e in nonmortar_edges:
        for each line-2 element L in e:
            classify L: side ∈ {"left", "right", "none"}
            select dual: M_line2_dual_modified(ξ, side)
            place 1D Gauss quadrature on L
            for each Gauss point ξ_q:
                x_q = T_L(ξ_q)
                x_m = Π_d(x_q)                  # axis-d periodic translation
                (ξ_m, mortar_line_id) = locate(x_m, mortar_edge)
                evaluate nonmortar M^mod at ξ_q
                evaluate mortar N at ξ_m
                accumulate D, A^m
```

For axis-aligned cubes, `Π_d` is a pure translation by L along axis d
(or − L for the opposite edge). The `locate` step is a 1D parameter
search along the mortar edge.

## §11.6 The face mortar geometric-matching algorithm

For each pair of opposite faces (3 pairs in 3D), the face mortar is a
2D mortar over a 2D interface. The algorithm parallels §3.5 with the
following 3D-specific structure:

```
function assemble_face_mortar_3d(nonmortar_face, mortar_face, axis):
    # axis ∈ {x, y, z}: the periodic translation direction
    Π = (x → x ± L * e_axis)             # axial translation operator
    for each nonmortar face element S in nonmortar_face:
        # S may be quad-4 or tri-3 depending on volume element
        face_class = classify_against_face_boundary(S, nonmortar_face.boundary)
        M_dual = (M_quad4_dual_modified if S.is_quad else
                  M_tri3_dual_modified)
        N_nonmortar = (N_quad4 if S.is_quad else N_tri3)
        ir = quadrature_rule(S.geom_type, order=2*p+1)  # p = polynomial order
        for q in ir.points:
            x_q = T_S(q.local_coord)
            x_m = Π(x_q)
            # Locate mortar element containing x_m
            (mortar_elem, m_local_coord) = locate_mortar(x_m, mortar_face)
            N_mortar_at_m = (N_quad4(m_local_coord) if mortar_elem.is_quad else
                             N_tri3(m_local_coord))
            M_at_q = M_dual(q.local_coord, face_class)
            w_q = q.weight * |det(J_T_S)|
            for i in nonmortar_LM_DOFs:
                for j in nonmortar_DOFs:
                    D_local[i,j] += w_q * M_at_q[i] * N_nonmortar[j](q.local_coord)
                for k in mortar_DOFs:
                    A_m_local[i,k] += w_q * M_at_q[i] * N_mortar_at_m[k]
        assemble_block(D_local, A_m_local, S.dofs, mortar_elem.dofs)
```

For axis-aligned periodic faces (our case), the `locate_mortar` step
collapses to a 2D parametric search:

- **Conforming meshes**: `locate_mortar` is direct geometric indexing
  (each nonmortar Gauss-point image lies in exactly one mortar element,
  identifiable by spatial sort).
- **Non-conforming meshes** (Phase 3.5): the nonmortar-element / mortar-
  element overlap may span multiple mortar elements. The integral must
  be sub-divided at mortar-element boundaries via Sutherland-Hodgman
  clipping (§3.7). Each sub-polygon contributes its own quadrature, and
  the contributions accumulate into the same D and A^m.

For axis-aligned cubes, `locate_mortar` for conforming meshes is:

```python
def locate_mortar(x_mortar, mortar_face_axis):
    # Drop the axis-d coordinate (it's redundant — both faces have the same
    # axis-d value modulo periodic translation).
    plane_coords = drop_axis(x_mortar, mortar_face_axis)
    # Find which mortar element contains plane_coords.
    elem_id = mortar_face.spatial_index.locate(plane_coords)
    # Compute local coordinates within that element.
    local = mortar_face.elements[elem_id].inverse_map(plane_coords)
    return (elem_id, local)
```

For quad-4 the inverse map requires a Newton iteration in the
general case; for axis-aligned grids, it reduces to two scalar
divisions. For tri-3, the inverse map is an affine 2x2 solve.

## §11.7 The 3D mesh + boundary classifier

`BoundaryClassifier3D` is the 3D analog of our 2D classifier. Given an
arbitrary mesh (hex, tet, or mixed) with nodal coordinates and boundary
attributes:

```
Input:  pmesh, fes
Output: 8 corners (each: TDOF index, X coordinate, attribute)
        12 edges (each: list of TDOF indices interior to the edge,
                   2 corner endpoints, parallel direction)
        6 faces  (each: list of face-element handles, organised by
                   face-element type (quad-4 or tri-3),
                   list of edges bounding the face,
                   perpendicular direction)
```

Geometric classification is independent of element type — it operates on
nodal coordinates only:

- **Corner**: a node at a vertex of the cube (where 3 boundary
  attributes meet, or where 3 face-planes intersect).
- **Edge**: a node on exactly one boundary edge (where 2 boundary
  attributes meet), not a corner.
- **Face**: a node on exactly one boundary face (single boundary
  attribute), not on any edge.

For axis-aligned cubes, this reduces to coordinate checks against the
6 face planes:

```python
def classify_node_3d(coords, eps=1e-12, L=1.0):
    """Classify a node into corner / edge / face / interior."""
    on_x_min = abs(coords[0]) < eps
    on_x_max = abs(coords[0] - L) < eps
    on_y_min = abs(coords[1]) < eps
    on_y_max = abs(coords[1] - L) < eps
    on_z_min = abs(coords[2]) < eps
    on_z_max = abs(coords[2] - L) < eps
    n_boundary = sum([on_x_min, on_x_max, on_y_min, on_y_max,
                      on_z_min, on_z_max])
    if n_boundary >= 3: return "corner"
    elif n_boundary == 2: return "edge"
    elif n_boundary == 1: return "face"
    else:                 return "interior"
```

The `BoundaryClassifier3D` then groups TDOFs by feature, with attention
to MPI distribution:

- A corner TDOF is owned by exactly one rank (the one that owns the
  underlying vertex).
- An edge TDOF is owned by one rank, but several ranks may need to
  know about the edge for constraint assembly (analogous to ghost
  faces in 2D).
- A face TDOF is owned by one rank.

For mixed-element meshes, the classifier must additionally:

- Group face elements by element type (quad vs tri) within each face.
- Ensure that each face-element's geometric vertices have been
  classified as corner / edge / face appropriately.
- Propagate the classification to per-face-element boundary
  configurations (e.g., for a tri-3 face element, the per-vertex boolean
  array `boundary_nodes` of §5.2.4).

Each rank's `BoundaryClassifier3D` reports the corners / edges / faces
it owns plus the face-element-level data needed to assemble the
constraint matrix block-by-block.

### §11.7.1 Cross-rank keying: snap-coord global identity

A subtle but load-bearing implementation detail surfaced during Phase
3.3.B macOS validation: when AllGather'ing per-rank vertex / element
records for cross-rank deduplication, **the dedup key MUST be globally
meaningful**. The two patterns that work in this codebase:

1. **Snapped physical coordinates** (used by `BoundaryClassifier2D`
   and `BoundaryClassifier3D`):
   ```python
   def snap_key(xyz):
       return (round(xyz[0] / tol),
               round(xyz[1] / tol),
               round(xyz[2] / tol))
   ```
   Stable across ranks because every rank computes the same key from
   the same physical position. Requires the parent mesh to have
   identical coordinate values on shared vertices across ranks (true
   for the `ParMesh(comm, serial_mesh)` partitioning we use).

2. **Global TDOF numbers** (used in `ConstraintBuilder2D`): when the
   records being merged correspond to FE DOFs, `GetGlobalTDofNumber`
   returns the same global index from any rank that knows the DOF.
   Preferable when applicable because it sidesteps coordinate-
   precision concerns.

What does **not** work as a dedup key:

- `parent_vertex_id` from `ParMesh.GetVertices()` or the
  `parent_vmap` of a `ParSubMesh`. These are RANK-LOCAL indices.
  Vertex 27 on rank 0 is unrelated to vertex 27 on rank 1 — they
  index into each rank's own local vertex array. Keying a merge
  dictionary by these causes silent data collisions: the rank-1
  record overwrites the rank-0 record under the same key, even
  though they refer to physically different vertices.

The original Phase 3.3.B implementation made this mistake. The
symptom at np > 1 was "1 or 2 boundary vertices missing a TDOF
component" — vertices on rank-boundary regions where the collision
left their gtdof tuple incomplete. The fix was to switch the dedup
key to snapped coords; the `_VertexRecord.parent_vertex_id` field
became `pvid` (a synthetic global counter assigned at merge time),
explicitly NOT the rank-local parent vertex index it was originally
populated from. This pattern is cross-referenced in §10.4
"distributed-driver invariants".

### §11.7.2 Runtime discovery of attribute → label mapping

Another implementation detail from Phase 3.3.C macOS validation:
the mapping from MFEM boundary-attribute integers to face labels
(bottom, top, front, back, left, right) **must be discovered at
runtime, not hardcoded**. MFEM's ``MakeCartesian3D`` boundary-
attribute ordering is not part of the documented API contract —
it varies between MFEM versions and between hex vs. tet element
types.

The bug it caused
-----------------
Phase 3.3.B initially hardcoded:

```python
_FACE_LABEL_BY_ATTR = {
    1: "bottom",  # I assumed y_min
    2: "front",   # I assumed z_min
    3: "right",   # x_max — correct
    4: "back",    # I assumed z_max
    5: "left",    # x_min — correct
    6: "top",     # I assumed y_max
}
```

But on the actual MFEM build under test (4.6+ via pyMFEM commit
7e99b925), attribute 1 corresponds to z_min (front in our
naming), not y_min. The classifier built `FaceInfo3D` records
where ``face_label="bottom"`` (claiming perp=y) was populated
with face elements whose vertices all had **z=0 invariant** —
i.e., quads from the actual front face (z=0).

Phase 3.3.B's topology checks didn't catch this — the **count**
of corners/edges/faces was correct (8/12/6), and the per-face
quad count was correct (16/face for hex). Only when Phase 3.3.C
called ``match_conforming_face_pairs`` between what was labelled
"bottom" (perp=y) and "top" (also a swapped label) did the
geometric mismatch surface: nonmortar centroid at (0.125, 0.0) in the
(x, z) plane has z_mean=0, which can only happen if all 4 z-coords
are 0 — a degenerate quad on the bottom face, which is impossible.

The fix
-------
``BoundaryClassifier3D._discover_face_label_by_attr`` is called
at __init__ time. For each boundary attribute present on the
mesh, it inspects one parent boundary element with that
attribute, determines which axis is invariant (zero spread) and
at which extreme (matching ``bbox_min`` or ``bbox_max``), and
maps (axis, extreme) to the canonical label via
``_AXIS_EXTREME_TO_LABEL``. The discovered mapping is stored as
``self._face_label_by_attr`` and used by all downstream methods.

Detection guarantees
--------------------
- If the mesh isn't axis-aligned (no axis is invariant within
  ``self.tol``), discovery raises explicitly.
- If two attributes map to the same label (e.g., both attribute
  1 and attribute 4 land on ``y_min``), discovery raises.
- If discovery doesn't find an element for every attribute in
  ``[1, n_attrs]``, discovery raises.

Lesson generalised
------------------
**Don't hardcode index-to-meaning mappings that depend on FE
library internals.** MFEM's element-type ordering (e.g., which
local face is "face 0" for a hex), boundary attribute ordering,
and DOF orderings (byNODES vs byVDIM) are all conventions that
shift between versions and configurations. Discover the mapping
from actual mesh data when correctness depends on it. The cost
is one extra setup pass at init time; the benefit is robustness
to upstream changes that would otherwise produce silent
correctness bugs (face elements assigned to wrong faces but
right counts, etc.).

### §11.7.3 What is (and isn't) in C's nullspace

A subtle question that surfaced during Phase 3.3.C macOS validation
and is worth pinning down: **the constant displacement field is
NOT in C's nullspace** (in the wirebasket-hierarchy formulation we
use), even though "u_nonmortar = u_mortar at every matched pair" is
trivially satisfied by a constant.

Why constants leak
------------------
The mortar block partition-of-unity `D[k] = Σ_l A_m[k, l]` holds
when both sides are summed over **all** mortar nodes — corner +
edge + interior. But the constraint matrix C is built with **corner
and box-edge mortars dropped via sentinels** (the wirebasket
hierarchy of §5.4). The dropped contributions don't appear in the
A_m sum, but they DO appear in D[k] (which is computed from the
nonmortar measure alone, independent of mortar sentinels). So:

    D[k] - Σ_kept A_m[k, l] = ∫ M_k · N_dropped_mortar ≠ 0

For a nonmortar node k near a box corner, the corner mortar node's N
function has support there, and the corresponding A_m entry that
"would have been" at column corner_mortar is dropped by the
sentinel filter. Result: row k has a partition-of-unity defect of
order J/2 (half the corner-element Jacobian).

Why this is correct
-------------------
The defect is exactly compensated in the saddle-point system by
the **explicit Dirichlet prescription on corner DOFs**. Phase 1B's
2D driver (and the upcoming Phase 3.4 3D driver) prescribes:

    u_corner = u_lin(X_corner) = (F-I) X_corner  (locked)

When the saddle-point right-hand side is built as
``b_constraint = -C_corner · u_corner_prescribed``, the
partition-of-unity defect becomes a constraint forcing term that
correctly drives the nonmortar DOFs to track the mortar modulo the
imposed corner values. A constant field has u_corner = constant,
which IS what the constraint enforces — but only if you account
for the corner column contribution explicitly in the RHS, NOT by
asking C·u_const = 0.

What IS in C's nullspace
------------------------
**Periodic fluctuations that vanish at corners.** A function like
``sin(2π X/L) sin(2π Y/L) sin(2π Z/L)`` (or any product where each
factor vanishes at X=0 and X=L) is:

  1. zero at every box corner / box edge / box face boundary
     (so all sentinel-affected DOFs are zero anyway), and
  2. periodic with period L, so u(nonmortar_X) = u(mortar_X) for any
     matched mortar-nonmortar pair on the same axis.

Both conditions together mean C · u = 0 exactly. This is the right
"nullspace probe" for testing C: build a periodic-vanishing-at-
corners field, multiply by C, expect machine-zero residual.

Lesson for Phase 3.4 driver implementation
-------------------------------------------
The 3D end-to-end driver must compute the constraint RHS as the
**non-zero macroscopic-jump term** including corner contributions.
A naive `b = 0` would converge u_tilde to a wrong solution (one
where corners have arbitrary values) rather than to u_lin =
(F-I)·X. The 2D Phase 1B code already does this correctly via
``apply_linear_part`` + corner-prescribed Dirichlet; the 3D driver
mirrors the structure.

## §11.8 The phasing plan for Phase 3

The plan is staged so each phase is locally testable. Hex and tet tracks
develop in parallel where convenient; some phases are element-type
agnostic.

**Phase 3.1 — 3D mesh + linear-elastic patch test, NO mortar.**

Hex mesh built via `mfem.Mesh.MakeCartesian3D`, OR tet mesh via
`MakeCartesian3D` with `Element.TETRAHEDRON`. **Full Dirichlet** on
all 6 boundary faces at u_lin = (F-I)X. NO periodic constraint, NO
traction. Solve linear elastic K · u = 0 with the prescribed Dirichlet
boundary; for homogeneous material, the unique solution is u = u_lin.

**Why full-boundary Dirichlet, not corner-only.** The naïve "8 corners
pinned at u_lin, free elsewhere" formulation does NOT have u_lin as
its solution. For homogeneous linear elasticity:
- div σ(u_lin) = 0 in Ω      (constant stress ⇒ zero divergence)
- σ · n ≠ 0    on ∂Ω         (constant stress hits surface normal)

Pinning corners only leaves ∂Ω\corners with the natural BC σ · n = 0,
which is incompatible with the constant-stress field. The minimum-
energy solver then returns a non-affine field that satisfies σ · n =
0 on the free boundary; ‖du‖_∞ comes back at the percent level, not
machine precision. The free-Neumann mismatch is exactly the boundary
load the production-stage *mortar PBC* (Phase 3.4) supplies via
periodic nonmortar-mortar coupling — there's nothing to validate here at
Phase 3.1 about that mechanism, so we sidestep it by clamping all of
∂Ω.

With full-boundary Dirichlet at u_lin, only interior DOFs are free,
and ∫∇N_i dV = 0 for compactly-supported interior basis functions, so
(K · u_lin)_i = 0 for all interior i. The solver drives du = 0 to
machine precision. This validates the K assembly + Dirichlet
elimination + CG-AMG solve infrastructure end-to-end, without mortar.

This phase establishes:
- 3D mesh handling for both hex and tet.
- 3D FES (vdim = 3, byNODES ordering — see §9.4 trap).
- Boundary-TDOF discovery via `fes.GetEssentialTrueDofs(ess_bdr_all,
  list)` and conversion to global TDOFs (helper:
  `find_all_boundary_tdofs`).
- Full-boundary Dirichlet via `EliminateRowsCols`.
- 3D ParaView visualization (mesh-node-warped, byNODES/byVDIM robust).
- 3D `compute_volume_averaged_F` (just a dim = 3 generalisation of
  the 2D one — element-type-agnostic).

PASS criterion: ‖u − u_lin‖_∞ < 1e-10 for homogeneous uniform F on
both hex and tet RVE meshes.

**Phase 3.2 — Dual basis + Wohlmuth modification + face-mortar assembler, pure-Python tests.**

This phase is split into two sub-phases that develop on the same pure-
Python layer (no MFEM dependency, fully unit-testable from synthetic
data):

**Phase 3.2.A — Dual bases and Wohlmuth modifications.**

Build:
- `M_line2_dual` already in place (`mortar_pbc/mortar_2d.py`).
- `M_tri3_dual(λ)` — eq. 4.19.
- `M_quad4_dual(ξ, η)` — eq. 4.16.
- `M_tet4_dual(λ)` — eq. 4.21 (volume mortar; not used for face mortar
  but documented for completeness).
- `M_tri3_dual_modified(λ, boundary_nodes)` — eqs. 5.5, 5.6.
- `M_quad4_dual_modified(ξ, η, side_ξ, side_η)` — eqs. 5.8, 5.10.

Unit tests, 3D analogs of the 2D suite (one per dual basis kind):

- `test_lumped_positivity_*`: **precondition test** — for each element
  type's standard FE shape functions {N_j}, verify s_j = ∫_E N_j > 0
  by direct quadrature on the reference element (one test per type:
  line-2, line-3, tri-3, tri-6, quad-4, quad-8, quad-9, tet-4). Per
  the §4.9.1 lumped-positivity criterion, this is the O(1) acceptance
  test for whether strict bi-orthogonality is even attemptable on the
  element. Expected outcome: PASS for line-2, line-3, tri-3, tet-4,
  quad-4, quad-9; FAIL with s_corner = 0 for tri-6, tet-10; FAIL with
  s_corner < 0 for quad-8, hex-20. The failing cases route to §4.10
  (basis-transformation) or §4.11 (LOR) at higher-order roadmap time.
  At Phase 3.2 we only implement the PASS-list dual bases, but this
  test guards against silently shipping a broken dual when a new
  element type is added later.
- `test_dual_basis_biorthogonality_*`: ∫ M_i N_j = δ_ij ∫ N_j (one
  test per element type currently in scope).
- `test_dual_basis_partition_of_unity_*`: ∑_i M_i = 1 (one test per
  type).
- `test_wohlmuth_quad4_modification`: edge-adjacent and corner-adjacent
  modifications preserve partition of unity.
- `test_wohlmuth_tri3_modification`: 1- and 2-vertex-dropped
  modifications preserve partition of unity.

**Status: COMPLETE.** `mortar_pbc/mortar_3d.py` ships all of the
above; `tests/test_mortar_3d_unit.py` covers all listed tests; all
pass.

**Phase 3.2.B — Face-mortar assembler for conforming face pairs.**

Bridge layer between the per-element dual bases of 3.2.A and the
global constraint matrix C built in Phase 3.3. The 3D analog of
`MortarAssembler2D` — operates on pure-Python face-element data
classes (no MFEM dependency), so unit-testable with synthetic
face meshes.

Architectural decisions, locked here so 3.3 can plug in:

1. **`MortarFaceAssembler` ABC + concrete subclasses
   `QuadFaceMortarAssembler` and `TriFaceMortarAssembler`** per §11.9
   Q7. The base class carries the assembly LOOP (nonmortar-element
   iteration, quadrature, accumulation into D and A^m); subclasses
   provide element-type-specific kernels (`_eval_nonmortar_dual`,
   `_eval_nonmortar_shape`, `_eval_mortar_shape`, `_quadrature_pts_wts`,
   `_nonmortar_jacobian`).

2. **Element data classes** `QuadFaceElement` and `TriFaceElement`
   (in `mortar_pbc/types_3d.py`) hold:
   - `coords`: (n_nodes, 3) physical coords of face-element corners
     in CCW order viewed from the *outward* normal of the nonmortar face.
   - `gtdofs`: list of n_nodes ints — global TDOFs of the *primary*
     spatial component, with sentinel **−1 for corner DOFs** and
     **−2 for edge DOFs** (these rows are dropped by the wirebasket
     hierarchy of §5.4). Vector-valued constraint construction in 3.3
     expands `gtdofs[i]` to per-component TDOFs via the FES ordering.
   - `parametric_axes`: tuple of two axis labels ("x"/"y"/"z") that
     parametrize the face plane.
   - `perpendicular_axis`: axis label of the face normal.
   - `boundary_tag`: per-edge classification of the element ("none",
     "edge-X", "corner-XY", …) used by the assembler to choose the
     correct Wohlmuth-modified dual.

3. **Conforming-pair path is the only Phase 3.2.B scope.** The
   assembler accepts a list of pre-matched `(nonmortar_elem_idx,
   mortar_elem_idx, mortar_node_perm)` tuples plus the nonmortar/mortar
   element lists. Mortar-node-permutation handles the case where the
   mortar-side face-element's local node ordering is shifted/reflected
   relative to nonmortar-side; for axis-aligned `MakeCartesian3D` meshes
   the permutation is the identity, but the API supports general
   conforming pairings to keep Phase 3.5 a drop-in extension.

4. **`match_conforming_face_pairs(nonmortar_elems, mortar_elems,
   perpendicular_axis, period)`** helper, pure-Python, uses
   parametric centroids + a tolerance-based KD-tree-style spatial
   index to pair up nonmortar/mortar elements. Returns the
   `(nonmortar_idx, mortar_idx, mortar_node_perm)` list. For axis-aligned
   `MakeCartesian3D` it's a single-pass match; for misaligned but
   conforming meshes it handles permutations.

5. **Sentinel-row drop policy.** Rows of D and A^m corresponding to
   nonmortar-side gtdofs −1 (corner) or −2 (edge) are dropped *during*
   assembly: the assembler simply doesn't accumulate into those rows.
   This matches the 2D pattern (`MortarAssembler2D` drops rows for
   corner sentinels) and the §5.4 wirebasket hierarchy.

Unit tests, validating the above on synthetic data (no MFEM):

- `test_face_mortar_quad_single_elem_conforming`: one quad-4 nonmortar
  paired with one quad-4 mortar, no boundary modification. Verify
  D = A^m = (|E|/4) · I_4 (eq. 3.8 conforming-pair lumping).
- `test_face_mortar_quad_2x2_grid_conforming`: 2×2 quad grid on each
  face. Verify D and A^m are 4×4 diagonal with correct per-node
  Jacobian-weighted lumping.
- `test_face_mortar_tri_single_elem_conforming`: tri-3 nonmortar/mortar
  pair, no modification. Verify D = A^m = (|T|/3) · I_3.
- `test_face_mortar_quad_with_edge_sentinel_drop`: nonmortar with one
  edge-sentinel gtdof = −2. Verify the corresponding row of D and
  A^m is absent / zero (depending on sentinel-drop policy chosen).
- `test_face_mortar_quad_with_corner_modification`: nonmortar element
  adjacent to a face corner uses `M_quad4_dual_modified` with
  appropriate `corner-XY` tag. Verify A^m off-diagonal coupling
  emerges and partition-of-unity row sums (∑_l A^m[k,l] over
  *non-sentinel* mortar nodes) match the modified dual's expected
  integrals.
- `test_face_mortar_tri_with_one_vertex_dropped`: equivalent for
  tri-3.
- `test_lumped_positivity_guard`: the assembler's __init__ runs
  `lumped_positivity()` against its own `_eval_nonmortar_shape` on the
  reference element and raises if any s_j ≤ 0. Verify this catches a
  hypothetical mis-instantiation with a tri-6 dual basis.

The test file is `tests/test_face_mortar_3d.py`; it runs in the
sandbox without MFEM.

**Phase 3.3 — `BoundaryClassifier3D` + `ConstraintBuilder3D`.**

This phase is split into four sub-phases. 3.3.A is a small dim-
genericity refactor that lets the existing 2D edge-mortar machinery
be reused for 3D edge pairs; 3.3.B builds the boundary classifier
on a single ParSubMesh primitive; 3.3.C composes the per-element-
type and per-feature blocks into the global constraint matrix; 3.3.D
is the first integration test (sparsity-only; full patch test is 3.4).

**Phase 3.3.A — Generalise `MortarAssembler2D` for 3D edge coordinates.**

The 2D edge-mortar math (1D parametric integration with line-2 dual
basis and Wohlmuth corner modification) is dimension-agnostic. The
only 2D-specific code is the axis-lookup in `_param_endpoints`:

```python
axis = 0 if edge.parametric_axis == "x" else 1   # 2D-only
```

The fix is a one-line dictionary lookup that supports `"z"` too:

```python
axis = {"x": 0, "y": 1, "z": 2}[edge.parametric_axis]
```

After this change, `MortarAssembler2D._assemble_pair` operates on
any duck-typed edge with `parametric_axis ∈ {"x", "y", "z"}`,
`edge_min`/`edge_max`, `coords[node_idx, axis]`, and an `elements`
list of `(node1, node2)` tuples with corner sentinels. `EdgeInfo3D`
satisfies all of these. The downstream `gtdofs` plumbing differs
between 2D and 3D, but the assembler doesn't touch gtdofs — only
the constraint builder consumes them.

Verification target: a unit test that takes a synthetic `EdgeInfo3D`
pair (along the z-axis at fixed x, y), runs `MortarAssembler2D
._assemble_pair`, and verifies the lumping recovery (D = A_m =
diag(per-segment Jacobian) on a conforming pair).

**Phase 3.3.B — `BoundaryClassifier3D` via a single boundary ParSubMesh.**

Architectural decision (locked): one `ParSubMesh` of the entire
boundary, not one per face attribute. Rationale:

1. **Unified back-mapping.** A single submesh-to-parent mapping
   covers face-elements, edges, and corners. We don't manage 6
   separate face-submeshes plus 12 edge-data structures plus
   8 corner records, each with its own parent-mapping concern.
2. **Wirebasket classification falls out structurally.** On an
   axis-aligned box:
     - submesh vertex touches **3** distinct parent boundary
       attributes ⇒ corner (8 of them)
     - submesh edge has **2** distinct parent attributes adjacent ⇒
       box edge (12 of them, 4 per direction)
     - submesh element has **1** parent boundary attribute ⇒ face
       interior element (6 face groups)
   The classification is one walk over submesh elements, accumulating
   per-vertex sets of parent boundary attributes.
3. **Forward-compatible with the §4.11 LOR fallback.** A single
   refined submesh suffices for higher-order LM construction; we
   don't re-architect for that future at Phase 6+.

ParSubMesh-to-parent API used:

- `mfem.ParSubMesh.CreateFromBoundary(parent_pmesh, attrs_array)` —
  builds the submesh.
- `submesh.GetParentElementIDMap()` — `Array<int>` of parent
  boundary-element indices per submesh element.
- `submesh.GetParentVertexIDMap()` — `Array<int>` of parent vertex
  indices per submesh vertex.
- `pmesh.GetBdrAttribute(parent_bdr_id)` — face-attribute lookup on
  the parent boundary element.
- `parent_fes.GetVertexDofs(parent_vert_id)` and the standard
  `local_dof → global_tdof` chain — for getting parent TDOFs at any
  submesh vertex.

For order-1 H1 (Phase 3 scope), DOFs live at vertices, so the
vertex-id map is sufficient for full TDOF back-mapping. Higher-order
(Phase 6+) requires walking edge/face interior DOFs too; the §4.11
LOR fallback obviates that for our use case.

The classifier output:
- `corners: Dict[str, CornerInfo3D]` — 8 corner records with parent
  global TDOFs.
- `edges: List[EdgeInfo3D]` — 12 edges, each with parent global
  TDOFs and the line-2 connectivity needed by `MortarAssembler2D`.
- `faces: List[FaceInfo3D]` — 6 faces, each with a list of
  `QuadFaceElement` or `TriFaceElement` (or both, for mixed
  hex+tet meshes — the boundary submesh's `GetGeometryType()`
  per element discriminates).

The classifier interface is cleanly separable from the underlying
MFEM ParSubMesh: it produces pure-Python data classes that
downstream `ConstraintBuilder3D` and the existing Phase 3.2.B
assemblers can consume without holding a ParSubMesh reference.

**Phase 3.3.C — `ConstraintBuilder3D`.**

Takes the classifier output and produces global C as a CSR matrix
(replicated, scipy-style, mirroring 2D `ConstraintBuilder2D`).
For each periodic group:

- **Edge mortar blocks (9 total)**: 3 directions × 3 mortar-nonmortar
  pairs each (1 mortar + 3 parallel nonmortars per direction). Each
  block built via the Phase-3.3.A-generalised `MortarAssembler2D
  ._assemble_pair(mortar_edge, nonmortar_edge)`. Wohlmuth corner
  modification handled by the existing `_corner_side` mechanism;
  corner-DOF rows dropped via the existing sentinel pattern.
- **Face mortar blocks (3 total)**: 3 mortar-nonmortar face pairs.
  Each face-element list passed to the appropriate Phase-3.2.B
  assembler (`QuadFaceMortarAssembler` or `TriFaceMortarAssembler`,
  dispatched per face element via geometry type; mixed-element
  faces accumulate from both assemblers and row-stack). Wohlmuth
  modification via `boundary_tag` on each face element; corner-
  and edge-DOF rows dropped via the sentinel pattern.

All blocks stacked via the existing `stack_constraints` machinery
into one CSR C. The constraint builder is a pure-Python
orchestrator — no MFEM dependency beyond what the classifier
already brought in. This keeps the C-assembly side of the saddle
point cleanly portable to a custom C++ class for ExaConstit
(important because MFEM has no `MixedNonlinearForm` analogue to
its `MixedBilinearForm`, so the C++ port will assemble C directly
into a `HypreParMatrix` rather than via MFEM's mixed-form
machinery).

**Phase 3.3.D — Sparsity-only integration test.**

Build the full pipeline (classifier → assemblers → C) on an
axis-aligned `MakeCartesian3D` hex RVE and a tet RVE, both 4×4×4.
Verify:
- C has the expected row count: (n_edge_DOFs × 3 components) +
  (n_face_DOFs × 3 components), with corner / edge crosspoints
  removed by the wirebasket hierarchy.
- C·u = 0 for an affine field u = (F-I)X (constraint is satisfied
  exactly by any field that's affine across the periodic boundary;
  this is the linear-field reproduction property of the dual basis).
- Symmetry of mortar coupling under mortar/nonmortar swap (sanity
  check; mortar formulation is asymmetric by design but the
  swap should produce a valid block too).

This phase does NOT solve the saddle-point system — that's 3.4.
This phase verifies C alone.

**Phase 3.4 — End-to-end 3D patch test driver.**

Hex AND tet RVE with conforming mesh on opposite faces, linear elastic
Method-D plus mortar PBC, multi-step ramp, ParaView output, ⟨F⟩
diagnostic, SciPy direct cross-check. PASS criteria identical to 2D:
Krylov converges, constraint residual at machine precision, Krylov vs.
direct match, ⟨F⟩ = F_macro to ~1e-13, fluctuation non-trivial in
heterogeneous case.

Test layouts:
- Homogeneous hex cube (sanity, both element types): u_tilde = 0.
- 3D analog of strip-split (hex track): half x ≤ L/2 stiff, half compliant.
- 3D analog of strip-split (tet track): same, on a tet mesh.
- 3D analog of checkerboard (hex track): 8-octant XOR pattern.
- 3D analog of checkerboard (tet track): same on tet mesh.
- **Mixed-element test (highest correctness bar)**: half hex, half tet.

**Phase 3.5 — Non-conforming face pairs.**

Add the geometric face-to-face polygon clipping (Sutherland-Hodgman, see
§3.7 pseudocode). Mesh different refinements on opposite faces: e.g.,
y=0 face has 4×4 quads, y=L face has 6×6 quads of slightly rotated
orientation. Re-run the patch test suite. Since the linear-elastic /
mortar formulation doesn't change, this is purely a geometric
extension of the nonmortar-quadrature-to-mortar-coordinate matching.

This is the phase where Tribol [LLNL Tribol] *might* become attractive
as an alternative backend for the polygon-clipping piece. Defer
evaluation until 3.4 is solid; hand-rolling Sutherland-Hodgman for
convex-on-convex (our case for quad-on-quad axis-aligned faces, also
fine for tri-on-tri and mixed cases) is straightforward and
dependency-free.

## §11.9 Open Phase-3 design questions

These are decisions that need an answer (or are at least flagged) before
Phase 3.3 starts. The recommendations are mine; finalise after a pass
through this doc.

1. **Constraint storage layout.** In 2D, C is replicated on every rank. In
   3D for moderate RVE sizes the same approach works:

   - 64×64×64 cube RVE: 6 faces × ~64×64 face-DOFs/face = ~24k face LM rows.
     Plus 12 edges × ~64 edge-DOFs/edge = ~770 edge LM rows. Per spatial
     component (×3): ~74k total rows. NNZ per row is ≤ 8 (nonmortar + mortar 4-node-quad
     coupling). Storage: 74k × 8 × 8 bytes = 4.7 MB per rank. **Replicated
     across ranks at this scale is fine.**
   
   - For larger RVEs (256×256×256 or above) we'd want distributed C. The
     existing operator-only design supports it — just need a distributed
     row-partition aware version of `WeightedRowSqSum`.
   
   **Recommendation: stay replicated for Phase 3, migrate later if needed.**

2. **Reference vs spatial configuration for mortar integration.** For our
   total-Lagrangian convention (§9), all assembly uses the reference
   configuration. ExaConstit's "updated-Lagrangian-at-load-step" model
   doesn't change the per-step kinematics: the reference geometry doesn't
   actually move. Mortar C is built once per mesh-change event. For nonlinear
   materials with K = ∂F_int/∂u, K changes per Newton iterate but C does not.

   **Recommendation: build C once, on the reference configuration, when the
   mesh and material are set. Re-build only on mesh adaptation events. Confirmed.**

3. **Dual basis integration order.** The integrand depends on element
   class:

   - **quad-4 unmodified**: the dual basis is bilinear in (ξ, η), the FE
     basis is bilinear, and ∫ M_i N_j is biquadratic — order 2
     Gauss-Legendre quadrature (4 points = 2×2) handles it exactly.
   - **quad-4 corner-modified** (eq. 5.10): the dual basis is constant
     (= 1) on the modified element. Integration against bilinear N is
     trivially bilinear; 1×1 quadrature suffices.
   - **tri-3 unmodified**: dual basis (eq. 4.19) is linear in λ_i; FE
     basis is linear. ∫ M_i N_j is quadratic in barycentric
     coordinates. Dunavant's 3-point rule [Dunavant 1985] of degree 2
     is exact.
   - **tri-3 edge-adjacent modified** (eq. 5.5): dual basis is linear
     (constant + linear); ∫ M^mod N is still quadratic. 3-point
     Dunavant.
   - **tri-3 corner-adjacent modified** (eq. 5.6): dual basis is
     constant. ∫ const N is linear; 1-point centroid rule suffices.
   - **line-2 unmodified**: integrand is quadratic; 2-point Gauss
     suffices.
   - **line-2 modified**: integrand is linear; 1-point suffices.

   **Recommendation: use a uniform "safe" rule per element type
   (4-point Gauss for quad, 3-point Dunavant for tri, 2-point Gauss for
   line-2) across all elements regardless of modification status. The
   theoretical reduction of order on modified elements gives at most a
   ~20% speedup that doesn't matter at prototype scale and is fragile
   (a missed corner case integrates wrong). Optimise only if
   profiling shows it matters.**

4. **Polygon clipping for non-conforming face pairs (Phase 3.5).**
   Sutherland-Hodgman [Sutherland & Hodgman 1974] is simple enough to
   hand-roll for convex-on-convex polygons:

   - **Quad-on-quad** (axis-aligned hex pairs): trivial, 4-on-4.
   - **Tri-on-tri** (axis-aligned tet pairs): same algorithm, 3-on-3.
   - **Mixed** (quad nonmortar on tri mortar, or vice versa): same
     algorithm; clip the nonmortar (3 or 4 vertices) against the mortar
     (3 or 4 vertices).

   `shapely` has the algorithm but is a heavy dependency. Tribol [LLNL
   Tribol] has industrial-strength clipping for contact mechanics; we
   may evaluate Tribol's API in Phase 3.5 as an alternative.

   **Recommendation: hand-roll Sutherland-Hodgman in Phase 3.5
   (~150 lines of Python, dependency-free); defer non-conforming
   testing until conforming Phase 3.4 is solid. Re-evaluate Tribol
   only if hand-rolled clipping proves unstable for skewed faces.**

5. **3D mesh source.** Five mesh types in scope:
   - (a) Pure hex via `mfem.Mesh.MakeCartesian3D`.
   - (b) Pure tet via `MakeCartesian3D` + `Mesh::ConvertToTets()`,
     OR by reading a tet `.mesh` file.
   - (c) Mixed hex + tet (read from external mesh files; MFEM
     supports mixed-element meshes natively).
   - (d) Non-conforming hex (independent face refinement; build via a
     `build_nonconforming_cube` analog of the existing
     `build_nonconforming_square`).
   - (e) Non-conforming tet (analogous).

   **Recommendation: (a) and (b) for phases 3.1–3.4, plus (c) for the
   mixed-element correctness test in 3.4. (d) and (e) for phase 3.5.
   Defer non-conforming until conforming is solid.**

6. **Edge LM grouping.** Per-direction (4 edges per direction, 3 mortar
   pairs per direction → 9 total mortar groups) versus per-edge-pair?
   The latter means 12 separate mortar groups (each pair of
   "topologically equivalent" edges). The implementation can go either
   way.

   **Recommendation: per-direction grouping. Each direction has 4
   parallel edges; pick one mortar, couple the other 3.
   3 directions × 1 mortar × 3 nonmortar-couplings = 9 sub-blocks; stack
   them into one C block per direction.**

7. **Element-type dispatch for face mortar.** The polymorphic
   `MortarFaceAssembler` interface (§11.4) handles quad-4 and tri-3
   uniformly. The C++ port will use virtual dispatch on
   `mfem::Element::Type`. For Python, dispatch on
   `element.GetGeometryType()` returning `mfem.Geometry.SQUARE` vs
   `mfem.Geometry.TRIANGLE`.

   **Recommendation: dispatch on `element.GetGeometryType()`. Build
   `QuadFaceMortarAssembler` and `TriFaceMortarAssembler` as concrete
   subclasses of a common `MortarFaceAssembler` ABC; let
   `ConstraintBuilder3D` dispatch per face element.**

8. **Higher-order primal field.** ExaConstit's primary FE order is
   p = 1 for crystal plasticity, but if/when p ≥ 2 enters the roadmap,
   the design question is: implement the §4.10 Popp-Wohlmuth-Gee-Wall
   higher-order dual basis from scratch (per element type), or use the
   §4.11 lower-order projection (LOR) fallback?

   **Recommendation: defer to Phase 6+; when needed, use LOR + linear
   dual + Barbosa-Hughes stabilisation per §4.12.** This re-uses the
   §4.2–§4.5 linear dual machinery, requires only a uniformly-refined
   ParSubMesh and one new stabilisation integrator, and matches Tribol's
   established design philosophy. The full higher-order dual basis is
   a multi-month effort with no precedent in the CPFEM-homogenisation
   literature; LOR is the pragmatic middle ground.

---

# §12. Hard-won lessons (the trap list)

This is the most important section of the document. Each trap below cost
real time. Future work should re-read this list before each new feature.

## §12.1 Discrete-correctness traps

**Trap 1. Use K_full to compute RHS in Method D, not K_eliminated.**

Symptom: free DOFs move in the *opposite* direction of u_lin in the
visualization. Corners are correct.

Diagnosis: `K_eliminated · u_lin` zeros out the K_uc · u_lin[corner] term at
free rows, but for the affine field to be the equilibrium under affine-corner
BC, that term must be present (it's the K_uu · u_lin[free] balancer). Without
it, the saddle-point solve drives u toward something ≠ u_lin to "fix" a
spurious residual.

Solution: assemble K twice (`K_full`, `K_eliminated`); use `K_full` for the
RHS computation `f = K_full · u_lin`; zero corner entries of `f` by hand;
use `K_eliminated` for the saddle-point top block.

In code: `MortarPbcDriver2D.__init__` takes both `K_op` (eliminated) and
`K_op_full` (un-eliminated). `_solve_independently` uses `K_op_full.Mult` for
the RHS. SciPy direct cross-check uses `K_full_global_csr` for its RHS too.

Per MFEM issue #793: `a.ParallelAssemble()` may share `SparseMatrix` data
with the `ParBilinearForm`. To get truly independent K_full and K_eliminated,
build *two independent* `ParBilinearForm` objects and assemble each
separately.

**Trap 2. The Wohlmuth corner modification is not optional.**

Symptom: in 2D, the patch test fails for shear F or any F that places the
corner-LM redundancy into a numerical contradiction. Krylov may diverge or
the constraint residual may stagnate.

Diagnosis: without dual-basis modification at corner-adjacent nonmortar segments,
the corner LM rows are redundant with the corner Dirichlet BCs. The
discrete C is rank-deficient.

Solution: implement `M_line2_dual_modified(xi, side)` per Lopes Eq. C.2,
drop corner-LM rows from the constraint block during assembly, and verify
via a unit test (`test_wohlmuth_crosspoint_modification`).

In 3D, this generalizes: corners dropped from edges (1D Wohlmuth), edges
dropped from faces (2D Wohlmuth on quad-4). See §11.

**Trap 3. The Newton residual must include the C^T · λ contribution.**

Symptom: ||F_int||_2 stagnates at the natural force scale of the problem
(e.g. ~1e5 for our 5× contrast neo-Hookean test) regardless of how
converged the actual equilibrium is. Newton appears to fail.

Diagnosis: at equilibrium, F_int = −Cᵀλ, not zero. ||F_int||_2 is *NOT* the
right convergence measure. ||F_int + Cᵀλ||_2 is.

Solution: in the Newton loop, after solving for du and dλ, accumulate
λ += dλ, and compute the next iteration's residual as
`r1 = nlf.Mult(u) + Cᵀ · λ`. Pass `r1` to the saddle-point solver AND use
`||r1||_2` as the convergence criterion.

The verification gather block must mirror this. Naively recomputing
`nlf.Mult(x, residual)` after Newton converges and reporting that as "final
residual" is misleading — it's F_int alone, not F_int + Cᵀλ.

**Trap 4. ParNonlinearForm handles essential DOFs internally.**

Symptom: applying `apply_dirichlet_to_distributed_K` *after*
`nlf.GetGradient(x)` corrupts K (double-elimination).

Diagnosis: `ParNonlinearForm.SetEssentialTrueDofs(...)` makes nlf:
- `nlf.Mult(x, residual)` returns residual with essential DOFs already zeroed.
- `nlf.GetGradient(x)` returns the tangent with essential rows/cols already
  eliminated.

Solution: only the *linear-elastic* manual driver path applies
`apply_dirichlet_to_distributed_K`. Nonlinear drivers must NOT.

**Trap 5. Krylov stagnation from a tiny RHS.**

Symptom: Newton declares failure, but the trace shows residual at noise
floor before max_iter. Newton "couldn't improve."

Diagnosis: when Newton has effectively converged but the outer loop hasn't
recognised it, the next Krylov call sees a tiny RHS, exits with 0 iterations,
returns du = 0. The outer loop sees no improvement and concludes failure.

Solution: include `||du||_2 < du_floor` as a convergence path in the Newton
outer loop, in addition to relative residual + constraint criteria.

**Trap 6. Absolute Newton tolerance ignores problem scale.**

Symptom: setting atol = 1e-10 is physically meaningless when the natural
force scale is 1e5. Either Newton "converges" prematurely on tolerance that
nothing physical needs to satisfy, or it never reaches that tolerance because
the noise floor is at 1e-7.

Solution: relative-drop convergence with absolute floor as safety net for
trivially-tiny problems. `||r1||_2 < max(rtol · r0, atol)`. Choose rtol per
problem class (1e-8 typical), atol per noise floor (1e-12 conservative).

## §12.2 MFEM / pyMFEM API traps

**Trap 7. byNODES vs byVDIM ordering mismatch.**

Symptom: visualization shows a 90° rotation of the deformed mesh.

Diagnosis: `ParFiniteElementSpace(pmesh, fec, vdim=dim)` defaults to
`Ordering::byNODES`. `pmesh.SetCurvature(order)` defaults to `Ordering::byVDIM`.
Adding a byNODES displacement TDOF vector elementwise to a byVDIM mesh-node
TDOF vector silently swaps x/y components.

Solution: explicitly pass `fes.GetOrdering()` to `SetCurvature`:

```python
pmesh.SetCurvature(1, False, -1, fes.GetOrdering())
```

The visualization helper handles this defensively now.

**Trap 8. `nlf.GetGradient` returns `mfem::Operator&` (base class).**

Symptom: trying to call `as_HypreParMatrix` on the return value of
`nlf.GetGradient(x)` gives an attribute error.

Diagnosis: pyMFEM exposes only the base. The dynamic type is normally
`HypreParMatrix`, but pyMFEM's SWIG wrapper doesn't downcast automatically.

Solution: use `mfem.Opr2HypreParMat` (the explicit downcast helper) or
duck-type-check `hasattr(op, "MergeDiagAndOffd")`. For verification gather
paths only — the actual saddle-point solve doesn't care about the dynamic
type, since it consumes K via `Mult` only.

**Trap 9. `GetDataArray()` view-vs-copy ambiguity.**

Symptom: writing into a numpy view of an `mfem.Vector` mysteriously fails to
update the underlying vector.

Diagnosis: on some pyMFEM builds `mfem.Vector.GetDataArray()` returns a
view; on others it's a copy. The behavior depends on SWIG flags at build
time.

Solution: use element-wise assignment via `__setitem__`:

```python
for i in range(vec.Size()):
    vec[i] = float(arr[i])
```

This always works, on every pyMFEM build, on every type of vector.

**Trap 10. `ParallelAssemble` may share data.**

Symptom: calling `EliminateRowsCols` on a "second" HypreParMatrix corrupts
the "first" one too.

Diagnosis: `a.ParallelAssemble()` returns a HypreParMatrix that may share
the underlying SparseMatrix with the ParBilinearForm. Calling it twice on
the same `a` is *not* guaranteed to give independent matrices.

Solution: build two independent `ParBilinearForm` objects (with the same
integrators and FES), `Assemble()` each, `ParallelAssemble()` each. Pay the
small cost of the extra local-assembly step in exchange for guaranteed
independence.

**Trap 11. BlockDiagonalPreconditioner doesn't own its diagonal blocks.**

Symptom: Krylov solve produces NaN or random garbage. Stack trace shows
something about freed memory.

Diagnosis: `mfem.BlockDiagonalPreconditioner` does NOT own the
`Operator` objects passed to `SetDiagonalBlock(i, op)`. Python GC will
collect them mid-Krylov-solve unless explicit references are kept alive
*outside* the function scope.

Solution: `SaddlePointSolver._build_block_jacobi_prec` returns a `keepalive`
list that the caller stashes on `self._last_prec_refs`. This holds Python
references to the diagonal block objects for the duration of the solve.

**Trap 12. NeoHookean integrator NaN at u=0.**

Symptom: `nlf.Mult(zero_par, residual)` returns NaN throughout (except at
essential DOFs which are 0).

Diagnosis: pyMFEM's `NeoHookeanModel(mu_coef, K_coef)` constructor (and all
variants tested) has a numerical issue at u=0 in this build of pyMFEM.
We pivoted to linear-elastic for the prototype.

Solution: linear-elastic `ElasticityIntegrator` works fine. For the eventual
production port, write a custom integrator subclass or use a different MFEM
build. Diagnostic preserved at `examples/diag_neohookean_2x2.py`.

## §12.3 MPI traps

**Trap 13. Every collective must run on every rank.**

Symptom: deadlocks at np > 1, especially after rank-0-only print blocks.

Diagnosis: a `comm.allreduce`, `C_op.Mult`, or `BoundaryClassifier2D`
construction inside a `if rank == 0:` block (or under any rank-asymmetric
guard like `if n_lam_local > 0:`) means rank 0 enters the collective and
other ranks don't, deadlocking.

Solution: never wrap a collective in a rank-asymmetric guard. If you need
a print-only block, separate the collective from the print:

```python
# WRONG:
if rank == 0:
    val = comm.allreduce(local, op=MPI.SUM)  # deadlock
    print(val)

# RIGHT:
val = comm.allreduce(local, op=MPI.SUM)      # everyone enters
if rank == 0:
    print(val)
```

**Trap 14. MPI gather requires consistent vector sizes.**

Symptom: rank 0 receives a flat-array but its content is misaligned to the
contributing ranks' partitions.

Diagnosis: `comm.Gatherv` uses `counts` and `displs` arrays. If the per-rank
vector sizes were computed with a different convention than the gather
expects, the displacement array will be wrong.

Solution: always gather sizes via an `allgather(my_size)` first, then
compute displs via `cumsum(counts[:-1])` *with `prepend=0`*. Don't try to
infer counts from the FES partition — use what the actual local data
provides.

## §12.4 Visualization / total-Lagrangian discipline traps

**Trap 15. Mesh-node mutation persists across visualisation calls.**

Symptom: in multi-step driver, step k's u_lin is "more stretched" than
expected by ~1% or more (depending on step and k). The cross-check fails
by similar magnitude.

Diagnosis: the visualization writer warps the mesh to deformed configuration
and saves; without restoring to reference, the next call to
`apply_linear_part(fes, F^{n+1})` evaluates `(F^{n+1} − I)·X` against the
*deformed* nodes, not the reference. This compounds over multiple steps.

Solution: `PbcVisualizationWriter.write_step` resets the mesh to the
reference snapshot *after* saving each cycle. The writer is now side-effect-
free with respect to the mesh; every operation outside the writer always
sees the reference. See §9.

This is the **total-Lagrangian discipline** — implementations are responsible
for keeping the mesh on the reference configuration unless visualisation is
explicitly active.

**Trap 16. ⟨F⟩ matches F_macro for the wrong reason.**

Symptom: even when the implementation has Trap-15-style bugs (deformed
reference frame), the ⟨F⟩ diagnostic reports F_macro to machine precision.

Diagnosis: when both `apply_linear_part` and `compute_volume_averaged_F`
read from the *same* deformed mesh state, they are mutually consistent —
the homogenization average theorem still says ⟨∇ũ⟩ = 0 because that's a
*property of periodicity*, not of the particular reference frame. The
diagnostic measures internal consistency, not correctness against the
reference frame.

Solution: enforce reference-frame discipline (see Trap 15); separately
verify via SciPy direct cross-check on rank 0 using ALL operators from the
reference-frame state. The cross-check catches reference-frame mismatch
*if and only if* the K matrices in it are reference-frame and the gathered
u_lin is also reference-frame.

In our prototype: K is assembled once at init (reference-frame), and after
applying Trap-15 fix, all subsequent operations use reference-frame
quantities. Verification block now succeeds at machine precision.

## §12.5 Process / debugging traps

**Trap 17. Trust the unit tests; don't trust the patch test.**

The unit tests verify *math properties* of pieces (dual basis bi-orthogonality,
partition of unity, Wohlmuth modification correctness). They are direct
statements about isolated math.

The patch test (homogeneous RVE → ũ = 0) is a *derived consequence* of:
- Correct math → correct mortar assembly → correct constraint → correct
  saddle-point system → correct linear solve → patch test passes.

If a unit test fails, you know exactly where the bug is. If the patch test
fails, you only know *something* in that chain is wrong.

When debugging, fix the unit tests first. When developing a new piece, write
the unit test first.

**Trap 18. Verify on conforming AND non-conforming.**

A conforming-only test passes even if your A_m matrix has a sign error,
because the diagonality of D papers over the issue. Non-conforming exposes
the asymmetry of the dual basis.

The 2D unit test `test_nonconforming_pair_consistency` exists for this. The
3D extension will need a `test_nonconforming_face_pair_consistency` that
linear-projects against the standard dual / N basis.

**Trap 19. Verify on heterogeneous AND homogeneous.**

A homogeneous-only test passes even if your constraint matrix has a sign error,
because ũ = 0 and the constraint is trivially satisfied. Heterogeneous
material guarantees a non-trivial fluctuation that the constraint actually
needs to enforce.

The 2D heterogeneous strip-split and checkerboard layouts are this check.
The 3D test suite needs a 3D analog (heterogeneous octant pattern, see
§11.7 Phase 3.4).

---

# §13. C++ port pathway into ExaConstit

This is the production target. The 2D prototype, the in-progress 3D extension,
and eventually the C++ rewrite all go into ExaConstit's framework. This
section tells future readers what the port looks like.

> **For the actual implementation plan, see `PHASE4_CPP_PORT_PLAN.md`.**
> This section provides the high-level class sketch and the integration-
> with-ExaConstit-internals story (§13.3, §13.4, §13.5). The companion
> doc `PHASE4_CPP_PORT_PLAN.md` provides the per-component implementation
> specifics, phasing, hazards, and done criteria — i.e. it's the working
> document for the port itself. This section stays as the conceptual
> overview; the companion doc is the project plan.

## §13.1 What pyMFEM has taught us about MFEM C++

The translation table:

| pyMFEM (prototype) | MFEM C++ (port) |
|---|---|
| `mfem.par.ParFiniteElementSpace` | `mfem::ParFiniteElementSpace` |
| `mfem.par.ParBilinearForm` | `mfem::ParBilinearForm` |
| `mfem.par.HypreParMatrix` | `mfem::HypreParMatrix` |
| `mfem.par.GMRESSolver` | `mfem::GMRESSolver` |
| `mfem.par.BlockOperator` | `mfem::BlockOperator` |
| `mfem.par.BlockDiagonalPreconditioner` | `mfem::BlockDiagonalPreconditioner` |
| `mfem.par.IntegrationRules.Get(...)` | `mfem::IntegrationRules::Get(...)` |
| Python `PyOperatorBase` subclass | C++ `mfem::Operator` subclass |
| Python ABC `ConstraintAssembler` | C++ pure-virtual interface |

The pyMFEM API is essentially a 1:1 wrapper of MFEM C++, so the prototype's
class structures translate directly. The places where pyMFEM-specific quirks
needed defensive coding (Trap 9, Trap 10) collapse to non-issues in C++.

## §13.2 The class design in C++

Following Lopes' and our prototype's structure, the C++ port has:

```cpp
namespace exaconstit { namespace mortar_pbc {

// 2D and 3D variants of the boundary classifier.
class BoundaryClassifier2D { ... };
class BoundaryClassifier3D { ... };

// Pure-virtual constraint assembler interface.
class ConstraintAssembler {
public:
    virtual void Assemble(...) = 0;
    virtual int NumLocalRows() const = 0;
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const = 0;
    virtual void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const = 0;
    virtual ~ConstraintAssembler() = default;
};

// Concrete subclass for mortar PBC.
class MortarPbcConstraintAssembler : public ConstraintAssembler { ... };

// (Future) Concrete subclass for uniform traction.
// class UniformTractionConstraintAssembler : public ConstraintAssembler { ... };

// Stack multiple assemblers into one combined constraint operator.
std::unique_ptr<ConstraintAssembler> StackConstraints(
    std::vector<std::unique_ptr<ConstraintAssembler>> assemblers);

// Saddle-point solver.  Subclass of mfem::ConstrainedSolver.
class MortarPbcSchurSolver : public mfem::ConstrainedSolver { ... };

// Multi-step driver, mirrors MortarPbcDriver2D.
class MortarPbcDriver { ... };

}}
```

The `MortarPbcSchurSolver` class is a candidate **upstream MFEM contribution**:
MFEM's `mfem/linalg/constraints.hpp` already provides
`SchurConstrainedHypreSolver`, `EliminationCGSolver`, and
`PenaltyConstrainedSolver`, but all three require an assembled
`HypreParMatrix` K. None handle the matrix-free / PA-K / GPU-friendly case.
Our `MortarPbcSchurSolver` *is* that variant. After ExaConstit integration is
solid, propose upstream as a fourth subclass.

## §13.3 Hooks into existing ExaConstit infrastructure

ExaConstit's existing framework provides:

- `BCManager`: handles essential BCs by attribute. PBC is constraint-based,
  not essential-BC-based, so we either extend BCManager with a constraint-aware
  variant or add a sibling `ConstraintManager` class. Recommendation: sibling.

- `mech_operator`: ExaConstit's wrapper around `ParNonlinearForm` (or its
  PA-friendly equivalent). Provides the K-as-Operator that our saddle-point
  solver consumes. No changes needed — already PA-friendly.

- `SystemDriver::SolveInit`: the warm-start projection. Already implements
  the "linear projection of BC change through previous-step tangent" pattern
  (§7). Needs extension to handle PBC's saddle-point version (the projection
  is itself a saddle-point solve when constraints are active).

- `BCManager::ComputeBCDelta`: the place that computes the change in essential
  values between steps. For displacement-driven PBC, this becomes
  `(F^{n+1} − F^n)·X[corner]`. Needs adapter.

The `MortarPbcDriver2D` (and eventually 3D) maps to a new ExaConstit class,
say `MortarPbcSystemDriver`, that wraps `SystemDriver` and adds the
constraint-assembly + saddle-point-solve responsibilities.

## §13.4 The PA path requirement

Critical architectural constraint, baked in since Phase 1A:

- **K is always treated as `mfem::Operator` only.** Never `tocsr()`, never
  `As<HypreParMatrix>()`, never gathered.
- The block-Jacobi preconditioner uses only `Operator::AssembleDiagonal`,
  which works uniformly across PA, EA, FA, and HypreParMatrix forms.

This is the GPU-portability requirement: in PA mode, K is matrix-free, lives
on GPU, and never produces a CSR. Anything that requires CSR access is a
no-go for the production solver. The block-Jacobi + Krylov path is correct
for any K-form; HypreBoomerAMG (a more sophisticated prec) is FA-only and
would need replacement with a matrix-free multigrid in PA mode.

For the prototype's saddle-point solver, the C operator is built as a Python
wrapper around a scipy CSR (replicated per rank). This is fine for
prototype-scale. In C++ we'll re-implement C as a true `mfem::Operator` that
applies the mortar coupling matrix-free or via a small distributed CSR.

## §13.5 What goes upstream and what stays in ExaConstit

**Goes upstream (potential MFEM contribution):**
- `MortarPbcSchurSolver`: a fourth `ConstrainedSolver` subclass, matrix-free
  K-friendly, block-Jacobi prec.

**Stays in ExaConstit:**
- `MortarPbcConstraintAssembler` and the surrounding `ConstraintAssembler`
  ABC: domain-specific to the RVE-PBC application. Fine in `exaconstit::mortar_pbc::`.
- `BoundaryClassifier2D/3D`: similar, fine in ExaConstit.
- `MortarPbcDriver`: a thin orchestration layer; ExaConstit-specific.

The rule of thumb: if it's reusable across applications (not just RVE
homogenization), it goes upstream. If it's RVE-specific, it stays.

---

# §14. Open questions and forward plan

This section is the working agenda. Items are tagged by priority.

## §14.1 Immediate (Phase 3, in priority order)

- [ ] **Phase 3.1**: 3D linear-elastic patch test, NO mortar. Establish 3D
      mesh / FES / Dirichlet / visualization scaffolding.
- [ ] **Phase 3.2**: Quad-4 dual basis + Wohlmuth modification, pure-Python
      unit tests. ~5 new unit tests. No MFEM coupling required.
- [ ] **Phase 3.3**: `BoundaryClassifier3D` + `ConstraintBuilder3D`. Integrates
      Phase 3.2 output into the constraint-assembly machinery. Conforming
      meshes only.
- [ ] **Phase 3.4**: End-to-end 3D patch test driver. PASS criteria identical
      to 2D, plus three new test layouts (homogeneous, octant strip-split,
      octant 8-XOR).
- [ ] **Phase 3.5**: Non-conforming face pairs via Sutherland-Hodgman.

## §14.2 Medium-term (Phase 4-5)

- [ ] **Phase 4 — C++ port (standalone in `tests/mortar_pbc/`)**:
      Detailed plan in `PHASE4_CPP_PORT_PLAN.md`. Three rounds:
      Phase 4.1 initial port with AllGather + HypreParMatrix C;
      Phase 4.2 distributed-hash matching to scale beyond ~500 ranks;
      Phase 4.3 element-assembly C operator for GPU portability.
      Validation against the validated Python prototype's three test
      drivers (homogeneous, heterogeneous strip-split, checkerboard
      octant-XOR). Does NOT touch ExaConstit production code paths;
      lives entirely in `tests/mortar_pbc/`.
- [ ] **Phase 5 — ExaConstit integration**: Once Phase 4 is green and
      promoted to `src/mortar_pbc/`, integrate with `BCManager`,
      `SystemDriver::SolveInit`, the velocity-primal switch (§7.1
      and §13.3 cover the interface points). This is a separate
      planning conversation.
- [ ] **Upstream MFEM contribution**: propose `MortarPbcSchurSolver` (or a
      more general matrix-free constrained solver) as a fourth
      `ConstrainedSolver` subclass. After Phase 4.3 is solid (the EA
      path is what makes it matrix-free).

## §14.3 Long-term (Phase 6+)

- [ ] **Multi-step driver with proper warm-start handling for nonlinear K**:
      the `MortarPbcDriver2D.solve_next_step` recipe is documented; needs
      Newton outer loop reactivation when nonlinear material is available.
- [ ] **Velocity-based primal formulation**: rate-dependent crystal plasticity
      wants this. Maps cleanly to ExaConstit's existing primal.
- [ ] **Tribol integration as an alternative `ConstraintAssembler`**: for
      contact and general non-conforming geometry beyond axis-aligned RVEs.
- [ ] **Uniform Traction (UT) BCs as a second `ConstraintAssembler`**: UT
      was the original motivation for the ConstraintAssembler ABC; now it's
      a matter of writing one new subclass and stacking it.
- [ ] **Higher-order primal field (p ≥ 2)**: see §4.8–§4.12 for the dual
      basis theory and the recommended LOR + linear dual + Barbosa-Hughes
      stabilisation pathway. Triggered if/when ExaConstit adopts p = 2 hex
      / quad-9 / tri-6 / tet-10 elements for crystal plasticity. Tribol's
      LOR mechanics (§4.11.4) provides the precedent in the LLNL/MFEM
      ecosystem.

## §14.4 Open design questions (require explicit answers)

These are flagged in §11.9 with recommendations; finalise them before Phase
3.3 starts.

1. Constraint storage: replicated per-rank in 3D? **Recommendation: yes,
   migrate to distributed only if memory pressures require it.**
2. Reference vs spatial mortar integration? **Recommendation: reference,
   build C once per mesh-change.**
3. Dual basis integration order? **Recommendation: 2nd-order Gauss
   quadrature (4 points/quad), reduce to 1st-order on Wohlmuth-modified
   elements only if profiling shows the savings matter.**
4. Polygon clipping library or hand-roll for non-conforming faces?
   **Recommendation: hand-roll Sutherland-Hodgman in Phase 3.5.**
5. 3D mesh source? **Recommendation: `MakeCartesian3D` + face-independent
   refinement extension (`build_nonconforming_cube`) for testing;
   conforming-only for Phases 3.1-3.4.**
6. Edge LM grouping per-direction or per-pair? **Recommendation:
   per-direction (3 sub-blocks per direction, mortar + 3 nonmortars; total 9
   edge-mortar sub-blocks).**
7. Element-type dispatch for face mortar? **Recommendation: dispatch on
   `element.GetGeometryType()`; `QuadFaceMortarAssembler` and
   `TriFaceMortarAssembler` as concrete subclasses.**
8. Higher-order primal field handling (p ≥ 2)?
   **Recommendation: defer to Phase 6+; when needed, use LOR + linear
   dual + Barbosa-Hughes stabilisation per §4.12.** Avoid the per-element-
   type basis-transformation route unless homogenisation accuracy
   demands it.

---

# §15. References

## §15.1 Primary references

1. **Lopes, I. A. R.; Ferreira, B. P.; Andrade Pires, F. M.** (2021). *On the
   efficient enforcement of uniform traction and mortar periodic boundary
   conditions in computational homogenisation.* Computer Methods in Applied
   Mechanics and Engineering, **384**, 113930. DOI: 10.1016/j.cma.2021.113930.
   
   Primary reference for our formulation. Method D (line 342, Remark 1),
   corner essentials (lines 1034–1035), Wohlmuth crosspoint modification
   (Appendix C, equations C.1–C.3). Local copy:
   `/mnt/user-data/uploads/1-s2_0-S004578252100267X-main.pdf` (in original
   conversation environment).

2. **Wohlmuth, B. I.** (2000). *A mortar finite element method using dual
   spaces for the Lagrange multiplier.* SIAM Journal on Numerical Analysis,
   **38**(3), 989–1012.

   Foundation paper for the dual-basis mortar method. Crosspoint
   modification originally from this paper.

3. **Wohlmuth, B. I.** (2001). *Discretization Methods and Iterative
   Solvers Based on Domain Decomposition.* Lecture Notes in Computational
   Science and Engineering, vol. 17. Springer.

   Book-length development of the mortar / dual-basis method.

## §15.2 Computational homogenization references

4. **Miehe, C.** (2003). *Computational micro-to-macro transitions for
   discretized micro-structures of heterogeneous materials at finite
   strains based on the minimization of averaged incremental energy.*
   Computer Methods in Applied Mechanics and Engineering, **192**, 559–591.

   Canonical reference for displacement-fluctuation-based PBC formulation;
   the "Lopes/Miehe school" of PBC. Method D in our terminology corresponds
   to Miehe's formulation.

5. **Geers, M. G. D.; Kouznetsova, V. G.; Brekelmans, W. A. M.** (2010).
   *Multi-scale computational homogenization: Trends and challenges.*
   Journal of Computational and Applied Mathematics, **234**, 2175–2182.

   Survey paper. Useful for context on the broader homogenization
   landscape.

## §15.3 ExaConstit and tooling

6. **ExaConstit GitHub**: https://github.com/llnl/ExaConstit
   - `src/system_driver.cpp:441-478` (`SolveInit`).
   - `src/fem_operators/mechanics_operator.cpp:295-331` (`GetUpdateBCsAction`).
   - Issue #8: discussion of time-evolving BCs and the warm-start rationale.

7. **MFEM**: https://github.com/mfem/mfem
   - `mfem/linalg/constraints.hpp`: `ConstrainedSolver` ABC and three
     existing subclasses (Schur/Elim/Penalty).
   - Issue #793: shared-data behavior of `ParBilinearForm::ParallelAssemble`
     (relevant to Trap 10).

8. **pyMFEM**: https://github.com/mfem/pyMFEM
   - Commit pinned to `7e99b925cfcbec002c9e21230b3c561cb19436a6`
     (MFEM 4.9 build fixes).

9. **Tribol**: https://github.com/llnl/Tribol
   - LLNL contact / mortar library. May be relevant as backend for Phase 3.5
     non-conforming geometric matching.

## §15.4 Related supporting references

10. **Sutherland, I. E.; Hodgman, G. W.** (1974). *Reentrant polygon clipping.*
    Communications of the ACM, **17**(1), 32–42.
    DOI: 10.1145/360767.360802.

    Basic polygon clipping algorithm; relevant for Phase 3.5 face mortar
    geometric matching. Cited in §3.7 and §11.9.

11. **Bernardi, C.; Maday, Y.; Patera, A. T.** (1994). *A new
    nonconforming approach to domain decomposition: The mortar element
    method.* In: Brezis, H.; Lions, J.-L. (eds.) Nonlinear Partial
    Differential Equations and their Applications. Collège de France
    Seminar, Vol. XI. Pitman, pp. 13–51.

    Original (standard, non-dual) mortar method. Cited in §3.4 and §4.7.

12. **Hill, R.** (1972). *On constitutive macro-variables for
    heterogeneous solids at finite strain.* Proceedings of the Royal
    Society A, **326**(1565), 131–147.
    DOI: 10.1098/rspa.1972.0001.

    Hill-Mandel principle, average theorem. Cited in §8.1.

13. **Mandel, J.** (1972). *Plasticité Classique et Viscoplasticité.*
    CISM Courses and Lectures No. 97. Springer, Wien.

    Companion of [Hill 1972] for the macro-micro stress-strain
    averaging theorem in finite-strain plasticity. Cited in §8.1.

14. **Lamichhane, B. P.; Wohlmuth, B. I.** (2007). *Higher order mortar
    finite element methods in 3D with dual Lagrange multiplier bases.*
    Numerische Mathematik, **107**(1), 151–170.
    DOI: 10.1007/s00211-005-0636-z.

    Provides dual Lagrange multiplier bases for higher-order tetrahedral
    and serendipity-hexahedral elements; the linear-tet formula M_i =
    5 λ_i − 1 (eq. 4.21 in this doc) appears as their Theorem 3.4
    special case. Cited in §4.4, §4.5, §4.8, §5.

15. **Popp, A.; Wohlmuth, B. I.; Gee, M. W.; Wall, W. A.** (2012).
    *Dual quadratic mortar finite element methods for 3D finite
    deformation contact.* SIAM Journal on Scientific Computing,
    **34**(4), B421–B446.
    DOI: 10.1137/110848190.

    Construction of feasible dual Lagrange multiplier spaces for
    higher-order interface elements (6-node tri, 8/9-node quad). Source
    of the basis-transformation procedure for higher-order biorthogonal
    bases. Cited in §4.8.

16. **Strang, G.; Fix, G. J.** (1973). *An Analysis of the Finite
    Element Method.* Prentice-Hall.

    Standard FE textbook; source for simplex integration formulas
    (eqs. 4.7a–c in this doc). Cited in §4.1.

17. **Dunavant, D. A.** (1985). *High degree efficient symmetrical
    Gaussian quadrature rules for the triangle.* International Journal
    for Numerical Methods in Engineering, **21**(6), 1129–1148.
    DOI: 10.1002/nme.1620210612.

    Triangle quadrature rules used in the tri-3 face mortar
    integration (§11.3). The 3-point degree-2 rule is the default for
    Phase 3.2. Cited in §11.3 and §11.9.

18. **Flemisch, B.; Wohlmuth, B. I.** (2007). *Stable Lagrange
    multipliers for quadrilateral meshes of curved interfaces in 3D.*
    Computer Methods in Applied Mechanics and Engineering, **196**(8),
    1589–1602.

    Detailed treatment of dual basis on 3D curved interfaces; relevant
    for future extensions beyond axis-aligned cubes.

## §15.5 Higher-order dual mortar references

19. **Lamichhane, B. P.; Wohlmuth, B. I.** (2002). *Higher order dual
    Lagrange multiplier spaces for mortar finite element
    discretizations.* Calcolo, **39**(4), 219–237.
    DOI: 10.1007/s100920200010.

    Original construction of strict bi-orthogonal dual basis for
    quadratic line elements (line-3, eq. 4.25 in this doc) and the
    quartic correction for continuity at crosspoints. Cited in §4.8.

20. **Popp, A.; Wohlmuth, B. I.; Gee, M. W.; Wall, W. A.** (2012).
    *Dual quadratic mortar finite element methods for 3D finite
    deformation contact.* SIAM Journal on Scientific Computing,
    **34**(4), B421–B446. DOI: 10.1137/110848190.

    The basis-transformation procedure for tri-6, quad-8, quad-9, hex-20.
    Eqs. 4.34–4.36 in this doc reproduce the explicit transformation
    matrices. Production reference for BACI/4C, MOOSE.
    Cited in §4.10. (Also listed as #15 above for §4.8 historical
    citation; this entry is the canonical reference for the
    transformation procedure.)

21. **Wohlmuth, B. I.; Popp, A.; Gee, M. W.; Wall, W. A.** (2012).
    *An abstract framework for a priori estimates for contact
    problems in 3D with quadratic finite elements.* Computational
    Mechanics, **49**, 735–747. DOI: 10.1007/s00466-012-0704-z.

    Convergence theory for the §4.10 basis-transformation construction;
    proves O(h^p) energy / O(h^{p+1}) L² rates for quadratic dual
    mortar. Cited in §4.10.4.

22. **Lamichhane, B. P.; Stevenson, R. P.; Wohlmuth, B. I.** (2005).
    *Higher order mortar finite element methods in 3D with dual
    Lagrange multiplier bases.* Numerische Mathematik, **102**(1),
    93–121. DOI: 10.1007/s00211-005-0636-z.

    The "quasi-dual" relaxation: dim M_h < dim W_{0,h} construction for
    cubic+ tetrahedra and serendipity hex where even the feasible
    construction of [Popp et al. 2012] is impractical. Cited in §4.9.4.
    (Note: this is the same DOI as ref #14, which is the publication of
    the same work — distinct citations because the LSW05 framework
    proper is the *prelimiary* technical machinery developed in the
    full Numer. Math. paper. We cite the LSW05 form when discussing
    the quasi-dual relaxation, the LW07 form when discussing higher-
    order tet/hex feasible duals.)

23. **Lamichhane, B. P.; Wohlmuth, B. I.** (2004). *A quasi-dual
    Lagrange multiplier space for serendipity mortar finite elements
    in 3D.* M2AN: Mathematical Modelling and Numerical Analysis,
    **38**(1), 73–92. DOI: 10.1051/m2an:2004004.

    Treats the quad-8 / hex-20 serendipity case where corner lumped
    integrals are *negative*. Cited in §4.9.2.

24. **Oswald, P.; Wohlmuth, B. I.** (2001). *On polynomial
    reproduction of dual FE bases.* Proc. Domain Decomposition
    Methods 13, pp. 85–96.

    The Gauss-Lobatto theorem: full P_{p−1} polynomial reproduction
    of dual basis on tensor-product elements holds *iff* nodes are
    Gauss-Lobatto-spaced. Cited in §4.9.3.

25. **Brivadis, E.; Buffa, A.; Wohlmuth, B. I.; Wunderlich, L.**
    (2015). *Isogeometric mortar methods.* Computer Methods in
    Applied Mechanics and Engineering, **284**, 292–319.
    DOI: 10.1016/j.cma.2014.09.012.

    Establishes that "the p/(p−1) pairing is numerically unstable"
    in the unmodified mortar formulation, motivating either Belgacem
    cross-point modification, or LOR + stabilisation. Cited in §4.11.3.

26. **Wunderlich, L.; Seitz, A.; Alaydin, M. D.; Wohlmuth, B. I.;
    Popp, A.** (2019). *Biorthogonal splines for optimal weak
    patch-coupling in isogeometric analysis with applications to
    finite deformation elasticity.* Computer Methods in Applied
    Mechanics and Engineering, **346**, 197–224.
    arXiv:1806.11535.

    IGA dual mortar with B-splines; relevant for the parametric-
    integration treatment of curvilinear interfaces. Cited in §4.9.3.

27. **Acharya, B. S.; Patel, A.** (2019). *Convergence results with
    natural norms: Stabilized Lagrange multiplier method for elliptic
    interface problems.* arXiv:1705.10519.

    Barbosa-Hughes-type stabilisation that recovers quasi-optimal
    rates for non-stable LM pairings (including LOR). Cited in §4.11.3.

28. **Gustafsson, T.; Råback, P.; Videman, J.** (2022). *Mortaring
    for linear elasticity using mixed and stabilized finite elements.*
    Computer Methods in Applied Mechanics and Engineering, **404**,
    115795. DOI: 10.1016/j.cma.2022.115795. arXiv:2209.02418.

    Modern treatment of Barbosa-Hughes stabilised mortar applied to
    elasticity; closest to the LOR + stabilisation construction
    recommended in §4.11.3 / §4.12 for ExaConstit higher-order PBC.

29. **Pazner, W.; Kolev, T.** (2021). *Low-order preconditioning of
    high-order finite element problems.* SIAM Journal on Scientific
    Computing, **43**(6), A4032–A4055. DOI: 10.1137/20M1364643.

    Theory of LOR (low-order refinement); the geometric property
    (4.38) — Lagrange-node / refinement-vertex coincidence — is
    Theorem 2.1 of this paper. Foundation for the §4.11.1
    construction.

30. **Chin, E.** (2023). *Contact constraint enforcement using the
    Tribol interface physics library.* MFEM Workshop 2023,
    https://mfem.org/pdf/workshop23/19_Chin_Tribol.pdf.

    Documents Tribol's design choice to project high-order primal
    fields onto a low-order-refined contact mesh — the precedent in
    the LLNL/MFEM ecosystem cited in §4.12.

---

End of MORTAR_PBC_ARCHITECTURE.md.

This document should be re-read at the start of each major work session.
When new bugs are encountered, add them to §12. When new architectural
decisions are made, add them to §11 or §13. When a question in §14 is
answered, move it to a "decided" subsection or remove it.

