# Adapting AMG-with-Filtering to a Dual-Basis Mortar Periodic Boundary Condition Saddle System in ExaConstit

A technical analysis of how to transplant the AMG-with-Filtering (AMGF) preconditioner of Petrides et al. [Petrides 2025] from its origin in interior-point contact mechanics into the dual-basis mortar periodic-boundary-condition saddle system implemented in `src/mortar_pbc`. The document develops the linear algebra from first principles, derives the augmented Lagrangian formulation rigorously, analyzes ExaConstit's HypreBoomerAMG configuration in detail, and lays out a phased implementation roadmap with concrete experimental milestones.

---

## Executive Summary

**Problem.** ExaConstit's mortar periodic-boundary-condition saddle system uses a block-diagonal preconditioner whose Schur block is a crude diagonal lumped approximation and whose K-block does not address constraint-induced near-null modes localized at the periodicity boundary $\Gamma$. On heterogeneous polycrystal problems, observed Krylov iteration counts can run into the hundreds, with $\|r_K\| \gg \|r_\lambda\|$ residual asymmetries of one to six orders of magnitude.

**Proposed approach.** Apply AMG-with-Filtering (AMGF) — a subspace-correction preconditioner that augments a standard BoomerAMG V-cycle with an exact direct solve on a small problematic subspace — to the K-block of the saddle preconditioner. For polycrystal RVE problems the problematic subspace is the set of displacement DOFs adjacent to $\Gamma$: small (typically $1\text{--}5\%$ of $n_u$), geometrically defined, and accessible to a sparse direct solver via MUMPS or CPardiso. The Petrides convergence theorem then bounds the preconditioned condition number by $2(\beta + 3)$, where $\beta$ depends only on the AMG quality on the bulk operator with the boundary subspace filtered out.

**Recommendation.** Phased implementation, beginning with the lowest-risk modification:

- **Phase 1: Path A** — AMGF wrapping the existing BoomerAMG K-block preconditioner. Minimal code change; tests whether the K-block is the conditioning bottleneck. ~1 week.
- **Phase 2: Path D** — Powell–Hestenes augmented Lagrangian reformulation with AMGF on the augmented (1,1) block $K_\gamma = K + \gamma C^T C$. Conditional on Phase 1 leaving residual issues. Simultaneously addresses K-block conditioning, Schur-block conditioning, and the residual imbalance. ~2–3 weeks.
- **Phase 3: Production benchmarks** of the Path A + Path D combination across the full ExaConstit envelope (large polycrystals, non-associated flow, GPU residency, sub-XYZ periodicity). ~Month 2.
- **Phase 4: Path B**, only if Phase 3 reveals corner-modified multiplier conditioning as a remaining bottleneck. AMGF on the dual Schur complement. Highest implementation cost, least mature theory.
- **Phase 5: Alternative AMG** (PCBDDC, MueLu, GenEO), only if AMGF-based approaches prove inadequate on the most extreme problems.

**The deployed BoomerAMG configuration is the correct baseline and should not be modified.** In particular, `HypreBoomerAMG::SetElasticityOptions` is deliberately not enabled: while it improves linear convergence on pure-elasticity benchmarks, empirical testing on ExaConstit's nonlinear crystal-plasticity workloads has consistently shown it to cause outer Newton divergence shortly after plastic flow initiates. The structural reasons — rigid-body modes are computed once at AMG setup and become unrepresentative as plastic deformation accumulates, and the interpolation built on elastic RBMs is a poor model of the plastic operator's near-null space — are documented in §4.5. AMGF wraps the deployed configuration; it does not modify it.

**How to read this document.**

- *Implementer who needs the action items:* skim §0.13 (the existing preconditioner), then read §2 (integration paths) and §7 (implementation roadmap), using §9 as the experimental playbook.
- *Reviewer assessing the structural argument:* read §1 (why subspace correction is the right tool), §2 (where to attach it), and §5 (augmented-Lagrangian theory for Path D).
- *Researcher who wants the full theoretical development:* read sequentially.
- *Practitioner deciding whether this approach fits their problem:* read this summary, §1, §8 (comparison with alternatives), and §10 (caveats and open questions).

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Section 0 — Mathematical Preliminaries](#section-0--mathematical-preliminaries)
- [Section 1 — The Structural Argument for Subspace Correction](#section-1--the-structural-argument-for-subspace-correction)
- [Section 2 — Integration Paths into the ExaConstit Pipeline](#section-2--integration-paths-into-the-exaconstit-pipeline)
- [Section 3 — Explicit Construction of the AMGF Preconditioner](#section-3--explicit-construction-of-the-amgf-preconditioner)
- [Section 4 — The $K^{-1}$ Surrogate Problem under ExaConstit's Actual AMG Configuration](#section-4--the-k-1-surrogate-problem-under-exaconstits-actual-amg-configuration)
- [Section 5 — Augmented Lagrangian: Derivation, Conditioning, and Saddle Structure](#section-5--augmented-lagrangian-derivation-conditioning-and-saddle-structure)
- [Section 6 — The Residual Imbalance Problem and Its Consequences](#section-6--the-residual-imbalance-problem-and-its-consequences)
- [Section 7 — Implementation Roadmap in MFEM/ExaConstit](#section-7--implementation-roadmap-in-mfemexaconstit)
- [Section 8 — Comparison with Alternative Saddle-AMG Approaches](#section-8--comparison-with-alternative-saddle-amg-approaches)
- [Section 9 — Recommended Sequenced Experiments](#section-9--recommended-sequenced-experiments)
- [Section 10 — Caveats and Open Questions](#section-10--caveats-and-open-questions)
- [Section 11 — Bibliography](#section-11--bibliography)

---

## Section 0 — Mathematical Preliminaries

This section defines every linear-algebra object that appears later, building from finite-element discretization up through Schur complements, $A$-orthogonal projections, Galerkin coarse-grid reductions, indefinite Krylov methods, and subspace corrections. It is intentionally thorough because the AMGF construction stacks these concepts on top of each other; missing any one of them makes the rest opaque.

Readers who want the implementation recommendation without the full theoretical development can read the Executive Summary above and skip directly to §2 (integration paths) and §7 (implementation roadmap), returning here for definitions when needed. Readers comfortable with the linear algebra may skim to §0.13 for the existing preconditioner deployed in ExaConstit.

### 0.1 Finite element setting

ExaConstit discretizes the quasi-static balance of linear momentum on a representative volume element $\Omega \subset \mathbb{R}^3$ using continuous piecewise-polynomial (typically $H^1$-conforming, $Q_1$ or $Q_2$ vector-valued) finite elements. Let $V_h \subset [H^1(\Omega)]^3$ denote the discrete displacement space with basis $\{\phi_i\}_{i=1}^{n_u}$; let $M_h$ denote the discrete Lagrange multiplier space supported on the union $\Gamma$ of the three pairs of opposite periodicity faces, with basis $\{\psi_k\}_{k=1}^{n_\lambda}$.

The Bubnov–Galerkin formulation of the linearized boundary-value problem (one Newton step) reads: find $(u_h, \lambda_h) \in V_h \times M_h$ such that

$$a(u_h, v_h) + b(v_h, \lambda_h) = \ell(v_h) \quad \forall v_h \in V_h$$

$$b(u_h, \mu_h) = g(\mu_h) \quad \forall \mu_h \in M_h$$

where $a(\cdot,\cdot)$ is the (symmetric or non-symmetric) tangent stiffness bilinear form arising from elastic-plastic linearization, $b(v_h, \mu_h) = \int_\Gamma v_h \cdot \mu_h \, ds$ is the standard constraint bilinear form, $\ell$ is the residual linear form, and $g$ encodes the macroscopic-strain-driven boundary data. Equivalently, in matrix form, this is the block system in §0.3.

### 0.2 Vector and matrix sizes; indexing conventions

Throughout, $n_u$ denotes the number of "live" displacement DOFs (Dirichlet-eliminated, typically corners pinned to remove rigid-body translations and at least one rotation), and $n_\lambda$ denotes the number of Lagrange-multiplier DOFs. Typical sizes for ExaConstit polycrystal RVE runs are $n_u \sim 10^5$ to $10^8$ with $n_\lambda \sim n_u^{2/3}$ scaling, i.e. roughly $1$–$5\%$ of $n_u$ at moderate resolution, decreasing under uniform refinement.

A generic real matrix is $A \in \mathbb{R}^{m \times n}$; a vector $v \in \mathbb{R}^n$. The transpose is $A^T$; the inverse $A^{-1}$. For symmetric $A = A^T$ we write $A \succ 0$ ("$A$ is symmetric positive definite," SPD) if $v^T A v > 0$ for all $v \neq 0$, and $A \succeq 0$ ("symmetric positive semidefinite") if $\geq 0$. The notation $A \succ B$ means $A - B \succ 0$.

### 0.3 The ExaConstit saddle-point block system

After mortar PBC assembly, each Newton step solves

$$\begin{bmatrix} K & C^T \\ C & 0 \end{bmatrix} \begin{bmatrix} du \\ d\lambda \end{bmatrix} = -\begin{bmatrix} r_K \\ r_\lambda \end{bmatrix}$$

where:

- $K \in \mathbb{R}^{n_u \times n_u}$ is the elastic-plastic tangent stiffness, assembled from element-level contributions through ExaConstit's `mechanics_integrators` and `mechanics_operator` infrastructure. Mostly symmetric; non-symmetric for non-associated flow with kinematic hardening cross-terms.
- $C \in \mathbb{R}^{n_\lambda \times n_u}$ is the dual-basis mortar constraint matrix, assembled by `MortarConstraintOperator` in `src/mortar_pbc`.
- The zero block reflects the fact that periodic constraints carry no compliance term, in contrast to stabilized mortar contact formulations which would have a small $-\epsilon I$ in the (2,2) block.

Following the Lopes–Ferreira–Andrade Pires construction [Lopes 2021], the dual basis is chosen so that — after partitioning RVE-boundary DOFs into nonmortar-side (one face of each opposite pair) and mortar-side (the partner face) groups — the matrix $C$ has the structure $C = \begin{bmatrix} D & -A^m \end{bmatrix}$ with $D \in \mathbb{R}^{n_\lambda \times n_\lambda}$ **diagonal** (or block-diagonal after Wohlmuth corner modifications) and $A^m \in \mathbb{R}^{n_\lambda \times n_m}$ a sparse rectangular coupling matrix between the two faces. The diagonality of $D$ is the central algebraic feature of the Wohlmuth dual basis [Wohlmuth 2000]: biorthogonality

$$\int_\sigma \phi_i \psi_j \, ds = \delta_{ij} \int_\sigma \phi_i \, ds$$

makes the nonmortar-side mortar mass matrix diagonal, so that — in Wohlmuth's own words — *"the mortar map can be represented by a diagonal matrix; in the standard mortar method a linear system of equations must be solved."* This is precisely what allows static condensation of $u^+$ in terms of $u^-$ via $u^+ = D^{-1} A^m u^-$, when condensation is desired (cf. §8.2 and §10.6).

### 0.4 Restriction and prolongation matrices

A **restriction matrix** $R \in \mathbb{R}^{n_c \times n}$ with $n_c \leq n$ is a 0/1 matrix with at most one $1$ per row, indexed by an index set $\mathcal{I} \subset \{1, \ldots, n\}$ of size $n_c$. Its action $Rv$ on a vector $v \in \mathbb{R}^n$ extracts the subvector indexed by $\mathcal{I}$. Its transpose $R^T \in \mathbb{R}^{n \times n_c}$ is a **prolongation by zero-padding**: $(R^T w)_i = w_j$ if $i$ is the $j$-th element of $\mathcal{I}$, and $0$ otherwise.

The composition $R^T R \in \mathbb{R}^{n \times n}$ is the **diagonal orthogonal projector** onto the coordinate axes indexed by $\mathcal{I}$: it is diagonal with ones at positions $\mathcal{I}$ and zeros elsewhere. The composition $R R^T \in \mathbb{R}^{n_c \times n_c}$ is the identity matrix $I_{n_c}$. The product $A_c := R A R^T$ is the **principal submatrix** of $A$ indexed by $\mathcal{I}$.

**Why this matters and where it shows up in FEM.** Although the formalism is abstract, the underlying operation is one every FEM practitioner has used. The restriction matrix is the algebraic object that does *"pick out the boundary DOFs from the full vector"* (or "pick out the interior DOFs," or "pick out the DOFs of a specific element," etc.) — the bookkeeping that happens whenever you want to apply a boundary condition, eliminate a constrained DOF, perform static condensation on an element's internal DOFs, or do a Schwarz overlap. The principal submatrix $R A R^T$ is what you get when you write down the stiffness contribution from just those DOFs, ignoring (zeroing out) all coupling to the excluded ones. In our context, $\mathcal{I}$ will be the set of *periodicity-boundary-adjacent displacement DOFs* — the few percent of the mesh's DOFs that sit on the periodicity faces — and $R A R^T$ will be the principal submatrix of the tangent stiffness on just those nodes.

These structural facts will be the workhorses of §3, where the AMGF subspace basis matrix $P$ plays the role of $R^T$ above.

**Concrete example.** Let $n = 6$ and $\mathcal{I} = \{2, 5\}$ (in 1-indexed notation). Then

$$R = \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \end{bmatrix}, \quad R^T = \begin{bmatrix} 0 & 0 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$$

For any $v \in \mathbb{R}^6$, $Rv = (v_2, v_5)^T \in \mathbb{R}^2$, and $R^T R v = (0, v_2, 0, 0, v_5, 0)^T$. Given any symmetric $A \in \mathbb{R}^{6 \times 6}$, the matrix $R A R^T \in \mathbb{R}^{2 \times 2}$ has entries $(A_{22}, A_{25}; A_{52}, A_{55})$ — exactly the principal $2 \times 2$ submatrix at rows/columns 2 and 5.

### 0.5 Projections

A square matrix $\Pi$ is a **projection** if $\Pi^2 = \Pi$ (applying it twice gives the same result as applying it once — geometrically, "I've already moved the vector onto its target subspace; doing it again changes nothing"). A projection is **orthogonal with respect to a particular inner product** if it is also self-adjoint in that inner product. The two cases relevant here:

**Euclidean orthogonal projection.** Onto the column space of a full-column-rank matrix $U \in \mathbb{R}^{n \times k}$:

$$\Pi_U = U (U^T U)^{-1} U^T$$

Satisfies $\Pi_U^2 = \Pi_U$ and $\Pi_U^T = \Pi_U$. If $U = R^T$ for a Boolean restriction $R$ as in §0.4 (so $U$ is the prolongation, tall-and-skinny), then $U^T U = R R^T = I$, so $\Pi_U = R^T R$ — the coordinate-axis projector that zeros out every component except those indexed by $\mathcal{I}$.

**A-orthogonal projection** ($A$-symmetric, where $A \succ 0$). Onto the column space of $U$, with respect to the inner product $(x, y)_A := y^T A x$:

$$\Pi_A^U = U (U^T A U)^{-1} U^T A$$

Satisfies $(\Pi_A^U)^2 = \Pi_A^U$ and is $A$-self-adjoint: $\Pi_A^U$ is symmetric with respect to $(\cdot, \cdot)_A$ even though it is not Euclidean-symmetric. The complementary projector $I - \Pi_A^U$ projects onto the $A$-orthogonal complement of the column space of $U$.

**Why "$A$-orthogonal" matters physically.** For solid-mechanics readers, the cleanest interpretation is via energy. When $A$ is a stiffness matrix, the inner product $(x, y)_A = y^T A x$ is twice the *strain energy* stored when the displacement field $x$ does work against the elastic force generated by $y$ (or vice versa, by symmetry). Two displacement fields $x$ and $y$ are **$A$-orthogonal** when this strain-energy coupling is zero — i.e., they are *energetically independent*: a perturbation of the structure in one direction stores no energy that couples to the other. This is exactly the orthogonality used in modal analysis: the natural mode shapes $\phi_i$ of a structure are $K$-orthogonal (and $M$-orthogonal) precisely because each mode is an independent degree of freedom in the energy decomposition. The $A$-orthogonal projection $\Pi_A^U$ in subspace-correction theory generalizes this: it splits any displacement field into "the part that lives in our chosen subspace" plus "the part that is energetically independent of it." This is the operator that appears in subspace correction theory; see §3.1.

**Oblique projections.** If we project along a direction other than the orthogonal complement, we get an oblique projector, e.g. $\Pi = U V^T$ with $V^T U = I$ but $V \neq U$. Constraint-preconditioning theory [Keller 2000] makes heavy use of oblique projectors related to the kernel of $C$, though we will not use them directly here.

### 0.6 Galerkin coarse-grid projection

Given a fine-grid matrix $A \in \mathbb{R}^{n \times n}$ and a prolongation matrix $P \in \mathbb{R}^{n \times n_c}$ (full column rank), the **Galerkin coarse operator** is

$$A_c := P^T A P \in \mathbb{R}^{n_c \times n_c}$$

If $A$ is SPD and $P$ has full column rank, then $A_c$ is SPD. If $P = R^T$ for a Boolean restriction $R$ as in §0.4 (so $P$ is the prolongation), then $A_c$ is the principal submatrix of $A$ indexed by $\mathcal{I}$. **This last case — Boolean prolongation built from a geometrically defined index set $\mathcal{I}$ — is exactly the AMGF "filtered subspace operator" $P^T A P$ that §3 will use.** The three terms — "principal submatrix" (§0.4), "Galerkin coarse operator" (§0.6 above), and "filtered subspace operator" (§3+) — all refer to the same matrix. Throughout the document we use "filtered subspace operator" when the AMGF context is foregrounded and "principal submatrix" or "Galerkin coarse operator" when the connection to §0.4 or §0.6 is the point.

The Galerkin coarse operator inherits structural properties from $A$ (symmetry, positive-definiteness when $P$ has full rank) but loses sparsity at distance $> 1$ in the coupling graph induced by $P$. For Boolean $P$ with $\mathcal{I}$ a *contiguous* index set, $A_c$ remains sparse. For $\mathcal{I}$ a *geometrically defined* subset (such as RVE-boundary DOFs), $A_c$ retains the surface-coupling structure of $A$ restricted to $\mathcal{I}$ — typically a banded, narrow-bandwidth matrix even when the underlying mesh is 3D.

### 0.7 Schur complements

Given a partitioned matrix

$$\mathcal{A} = \begin{bmatrix} K & C^T \\ C & E \end{bmatrix}$$

with $K \in \mathbb{R}^{n_u \times n_u}$ invertible, the **(lower) Schur complement** is $S := E - C K^{-1} C^T \in \mathbb{R}^{n_\lambda \times n_\lambda}$. (The symbol $\mathcal{A}$ for the saddle matrix follows the convention of [Benzi 2005]; the calligraphic font distinguishes it from generic scalar matrices $A$ and from the candidate smoother $B$ that will appear in §3.1. The symbol $E$ here denotes the (2,2) block of the *generic* partitioned matrix; it should not be confused with the Wohlmuth diagonal $D$ of §0.3, which is internal to the constraint matrix $C$, not the (2,2) block of the saddle system. In the ExaConstit case $E = 0$.) The block-LDU decomposition

$$\begin{bmatrix} K & C^T \\ C & E \end{bmatrix} = \begin{bmatrix} I & 0 \\ C K^{-1} & I \end{bmatrix} \begin{bmatrix} K & 0 \\ 0 & S \end{bmatrix} \begin{bmatrix} I & K^{-1} C^T \\ 0 & I \end{bmatrix}$$

is the foundational identity of saddle-point theory [Benzi 2005, Eq. 3.1]. From it follows the inverse formula

$$\mathcal{A}^{-1} = \begin{bmatrix} I & -K^{-1} C^T \\ 0 & I \end{bmatrix} \begin{bmatrix} K^{-1} & 0 \\ 0 & S^{-1} \end{bmatrix} \begin{bmatrix} I & 0 \\ -C K^{-1} & I \end{bmatrix}$$

which can be expanded out to

$$\mathcal{A}^{-1} = \begin{bmatrix} K^{-1} + K^{-1} C^T S^{-1} C K^{-1} & -K^{-1} C^T S^{-1} \\ -S^{-1} C K^{-1} & S^{-1} \end{bmatrix}$$

In the ExaConstit case $E = 0$, so $S = -C K^{-1} C^T$, which is the negative of a symmetric positive (semi)definite matrix whenever $K \succ 0$ and $C$ has full row rank.

**Physical interpretation of the Schur complement.** For solid-mechanics readers the algebra here is dense, but the physical content of $\hat{S} = C K^{-1} C^T$ is simple and worth stating directly. $K^{-1}$ is the *compliance* operator of the bulk elastic-plastic problem: applied to a force vector, it produces the displacement field that force generates. $C$ is the operator that picks out the constraint-mode component of a displacement field (think: "given a displacement, what's the resulting periodicity mismatch?"). So $C K^{-1} C^T$ has a chain interpretation: start with a unit Lagrange multiplier (i.e., a unit "periodicity-restoring force" applied along the constraint mode); $C^T$ converts it to a body force on the displacement field; $K^{-1}$ computes the resulting displacement; $C$ reads off the resulting periodicity mismatch. The Schur complement is therefore the **constraint-mode compliance**: how much periodicity mismatch you get per unit applied multiplier, after letting the elastic body respond. In contact problems, the equivalent object is the contact-surface compliance — how much penetration you get per unit applied contact force, with the bulk free to respond. This is why $\hat{S}$ is sometimes called the *dual operator*: it lives on the multiplier side of the problem and inherits its character (well-conditioned or ill-conditioned, smooth or singular) from how the bulk operator $K^{-1}$ acts on the constraint subspace $C^T$.

**Sign convention.** Many references work with the negative-definite $S = -C K^{-1} C^T$; others define the Schur complement after a sign flip on the constraint equation to obtain $\hat{S} = C K^{-1} C^T \succ 0$. We use $\hat{S}$ when we want SPD properties (Krylov method analysis, AMG application) and $-\hat{S}$ when we want algebraic consistency with the saddle factorization. This is purely a sign convention and does not affect the iterations.

**Spectral significance.** The spectrum of the saddle matrix $\mathcal{A}$ (with $E = 0$) consists of $n_u$ positive eigenvalues and $n_\lambda$ negative eigenvalues, with magnitudes determined by $K$ and $\hat{S}$ jointly. Specifically [Murphy 2000], the eigenvalues $\mu$ of $\mathcal{A}^{-1} \cdot \text{diag}(K, -\hat{S})$ are roots of the polynomial $\mu^2 - \mu - 1 = 0$, namely $\mu \in \{1, (1 \pm \sqrt{5})/2\}$, so MINRES with the "ideal" block-diagonal Schur preconditioner converges in at most three iterations on any well-posed saddle system with $E = 0$. The corresponding result for the block-triangular preconditioner gives convergence in two iterations. These are existence results; in practice the inexactness of $\hat{S}^{-1}$ degrades the result.

**What this means for preconditioner design.** The Murphy–Golub–Wathen result tells you that *if* you could invert $\hat{S}$ exactly, three (or two) MINRES iterations would suffice regardless of problem size. The practical preconditioning question is therefore reducible to: *how well can we approximate $\hat{S}^{-1}$?* In the existing ExaConstit pipeline, $\hat{S}^{-1}$ is approximated by $\text{diag}(\hat{S})^{-1}$ — the inverse of just the diagonal of the constraint-mode compliance matrix. This throws away all off-diagonal coupling between distinct multiplier modes, which is a crude approximation when constraint modes interact (corner-modified Wohlmuth dual basis, heterogeneous polycrystals near the periodicity boundary). Path B targets this directly; Path D sidesteps it by reformulating the saddle system so $\hat{S}$ never appears.

### 0.8 Matrix norms and condition numbers

For a square matrix $A$, the **spectral norm** (induced 2-norm) is

$$\|A\|_2 := \sup_{v \neq 0} \frac{\|Av\|_2}{\|v\|_2} = \sigma_{\max}(A)$$

the largest singular value. For SPD $A$, $\|A\|_2 = \lambda_{\max}(A)$.

For an SPD matrix $M$, the **M-norm** is $\|v\|_M := \sqrt{v^T M v}$. The **M-induced operator norm** is $\|A\|_M := \sup_v \|Av\|_M / \|v\|_M$.

The **$\kappa_2$ condition number** of an SPD matrix is

$$\kappa_2(A) := \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)} = \|A\|_2 \|A^{-1}\|_2$$

For a non-symmetric matrix, $\kappa_2(A) = \sigma_{\max}(A)/\sigma_{\min}(A)$. The **M-condition number** is defined analogously using the M-induced operator norm.

A **preconditioner** is a matrix $M \approx A$ (in a sense to be made precise) that is cheaper to invert than $A$ and that reduces the effective condition number of the system. The relevant quantity governing Krylov convergence is $\kappa(M^{-1} A)$ (left preconditioning) or $\kappa(A M^{-1})$ (right preconditioning), not $\kappa(A)$ itself.

### 0.9 Spectral equivalence

Two SPD matrices $A$ and $B$ are **spectrally equivalent on a subspace $V$** if there exist constants $0 < \alpha \leq \beta < \infty$ such that

$$\alpha \, v^T B v \leq v^T A v \leq \beta \, v^T B v \quad \forall v \in V$$

The ratio $\beta/\alpha$ is the **spectral equivalence constant**. If $A$ and $B$ are spectrally equivalent globally with constants $\alpha, \beta$, then $\kappa(B^{-1} A) \leq \beta/\alpha$. This is the central quantity in subspace correction theory: a preconditioner $B$ is a good preconditioner for $A$ if $\beta/\alpha$ is moderate (say, $\leq 100$) and mesh-independent.

**Physical reading.** When $A$ and $B$ are stiffness-like operators, $v^T A v$ and $v^T B v$ are *energy measures*: each is twice the strain energy stored in $v$ under the action of $A$ or $B$. Spectral equivalence says: for every admissible displacement $v$, the energies measured by $A$ and $B$ are within a fixed multiplicative factor of each other, *uniformly* — the factor doesn't blow up for special modes or as the mesh refines. So $B$ doesn't have to be close to $A$ entry-by-entry; it has to give the same answer (to within a bounded factor) to the question "how much energy is in this displacement field?" When this energy-norm equivalence holds, $B$ inverts well enough that preconditioned Krylov converges in iteration counts independent of mesh size.

The notion is critical for AMGF analysis. Petrides et al. [Petrides 2025] assume that the AMG preconditioner is spectrally equivalent to the inverse of an "elasticity-without-contact" operator on a subspace; the AMGF bound then transfers that quality to the full operator at the cost of an additional factor of 2 plus a constant.

### 0.10 Eigendecompositions and the spectrum of indefinite matrices

An SPD matrix $A$ admits a real eigendecomposition $A = U \Lambda U^T$ with $U$ orthogonal ($U^T U = I$) and $\Lambda = \text{diag}(\lambda_i) \succ 0$. The eigenvectors $u_i$ are orthonormal and form a basis. The condition number is $\lambda_{\max}/\lambda_{\min}$, and CG converges at the rate

$$\|e_k\|_A \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|e_0\|_A$$

i.e. logarithmic in the tolerance and proportional to $\sqrt{\kappa}$.

A symmetric **indefinite** matrix (such as the saddle-point matrix $\mathcal{A}$ of §0.7 with $E = 0$) also has a real eigendecomposition with real eigenvalues, but they straddle zero. The relevant Krylov method is MINRES, which converges at the rate [Trefethen 1997, Lecture 38]

$$\|r_k\| \leq 2 \left(\frac{\sqrt{\kappa_{\text{eff}}} - 1}{\sqrt{\kappa_{\text{eff}}} + 1}\right)^{\lfloor k/2 \rfloor} \|r_0\|$$

where the *effective* condition number $\kappa_{\text{eff}}$ is determined by the spread of *positive* and *negative* eigenvalue clusters separately. Specifically, if positive eigenvalues lie in $[\lambda_1^+, \lambda_2^+]$ and negative eigenvalues lie in $[-\lambda_2^-, -\lambda_1^-]$, then

$$\kappa_{\text{eff}} = \frac{\lambda_2^+ \lambda_2^-}{\lambda_1^+ \lambda_1^-}$$

This is *not* the same as the condition number $\kappa = \max|\lambda|/\min|\lambda|$, and the difference matters: tight clustering of each sign group beats wide overall range. This is why block-diagonal preconditioning of saddle systems is effective even though it does nothing to remove the indefiniteness.

For a non-symmetric matrix, no real eigendecomposition exists in general, but a Schur decomposition $A = U T U^*$ (with $U$ unitary and $T$ upper triangular) does. GMRES convergence depends on the eigenvalue distribution of $A$ in the complex plane and the conditioning of $U$.

### 0.11 Krylov subspace methods, briefly

The **Krylov subspace** of order $k$ generated by $A$ and $r_0$ is

$$\mathcal{K}_k(A, r_0) := \text{span}\{r_0, A r_0, A^2 r_0, \ldots, A^{k-1} r_0\}$$

A Krylov method seeks an approximate solution $x_k \in x_0 + \mathcal{K}_k$ minimizing some norm of the residual or error. The relevant methods for our setting are:

- **Conjugate Gradients (CG)** [Saad 2003, Ch. 6]. SPD systems only. Minimizes $\|e_k\|_A$ where $e_k = x - x_k$. Short three-term recurrence; one matrix-vector product per iteration; two inner products. Optimal for SPD.
- **MINRES** [Saad 2003, §6.7]. Symmetric (possibly indefinite). Minimizes $\|r_k\|_2$. Short three-term recurrence. Optimal for symmetric.
- **GMRES** [Saad 2003, Ch. 6]. Non-symmetric. Minimizes $\|r_k\|_2$. Long recurrence (or restarted); storage grows with iteration count. The standard choice for non-symmetric systems.
- **BiCGStab** [Saad 2003, §7.4.2]. Non-symmetric. Bounded storage. Less robust than GMRES but cheaper per iteration.

For the saddle system with the block-diagonal Schur preconditioner, MINRES is the natural choice when $K$ is symmetric. For non-symmetric $K$ or for block-triangular preconditioners, GMRES is required.

### 0.12 Preconditioning conventions

Given a linear system $A x = b$, a **left preconditioner** $M$ transforms the system to $M^{-1} A x = M^{-1} b$; a **right preconditioner** $M$ transforms it to $A M^{-1} y = b$ with $x = M^{-1} y$; a **split preconditioner** $M = M_L M_R$ transforms to $M_L^{-1} A M_R^{-1} y = M_L^{-1} b$ with $x = M_R^{-1} y$. The right and split forms preserve the original residual $b - Ax$, which can be important for stopping criteria.

For symmetric problems, one uses a symmetric preconditioner (i.e. $M$ SPD or symmetric indefinite with appropriate block structure). The "split" form $M^{1/2}$ is conceptually useful but in practice one applies $M^{-1}$ as a whole.

### 0.13 The existing preconditioner in ExaConstit

**Plain-English summary first.** ExaConstit's existing preconditioner for the saddle system is the textbook block-diagonal Schur form: precondition the displacement block with whatever K-block preconditioner is configured (typically BoomerAMG), and precondition the multiplier block with a *crude* diagonal lumped approximation of the constraint-mode compliance. The latter is what §0.7 introduced as $\hat{S} = C K^{-1} C^T$ — the operator describing how Lagrange multipliers couple to constraint mismatches through the bulk's compliance. The diagonal lumped version of $\hat{S}^{-1}$ throws away all the off-diagonal entries, treating each constraint mode as if it interacted only with itself. This is a reasonable first approximation but breaks down exactly where the structure is most subtle (corners, material interfaces, mesh anisotropy). The two weaknesses listed below are direct consequences of that crude lumping plus the unrelated fact that bulk AMG doesn't know about the constraint structure.

The existing saddle-point preconditioner, defined in `src/system_driver.cpp` and `src/mortar_pbc/mortar_saddle_preconditioner.hpp`, is

$$P^{-1}_{\text{current}} = \begin{bmatrix} P_K^{-1} & 0 \\ 0 & \text{diag}\big(C \, \text{diag}(K)^{-1} \, C^T\big)^{-1} \end{bmatrix}$$

where $P_K$ is the configured K-block preconditioner from the existing `J_prec` infrastructure — typically `HypreBoomerAMG`, but also one of `HypreEuclid` (ILU), `HypreSmoother::l1GS`, `HypreSmoother::Chebyshev`, or `HypreSmoother::l1Jacobi`. The Schur block is a diagonal lumped approximation of $\hat{S} = C K^{-1} C^T$, computed via `MortarConstraintOperator::ComputeInvDiagSchur` by probing a Jacobi-style preconditioner with the ones vector.

This is the canonical "ideal" block-diagonal Schur preconditioner [Murphy 2000] with both diagonal blocks approximated. It is a reasonable starting point but has two structural weaknesses worth understanding before extending it:

1. **The Schur block approximation is crude.** Replacing $\hat{S}^{-1}$ with $\text{diag}(\hat{S})^{-1}$ loses all off-diagonal coupling in the Schur complement, which matters when the dual-basis biorthogonality is broken at corners (Wohlmuth modifications), when the polycrystal heterogeneity creates anisotropic compliance on the boundary, or when the mesh near $\Gamma$ is non-uniform.
2. **The K-block preconditioner does not directly address constraint-induced near-null modes.** The deployed BoomerAMG configuration is robust for the bulk elastic-plastic operator (see §4 for detailed analysis), but it does not have explicit knowledge of the mortar constraint structure. Near-null modes induced by the constraint geometry — specifically, displacement modes localized near $\Gamma$ that interact poorly with the constraint matrix $C$ — fall outside what the bulk AMG can capture on its own.

AMGF is targeted exactly at the first weakness when applied to the Schur block (Path B, see §2.3), or at the second weakness when applied to the K block (Path A, see §2.2). The augmented Lagrangian reformulation (Path D, see §2.5 and §5) attacks both simultaneously by changing the operator structure. The four integration paths are formally introduced in §2.

---

## Section 1 — The Structural Argument for Subspace Correction

This section is the conceptual core of the document. It explains *why* AMGF is the structurally right tool for the ExaConstit mortar PBC problem — not merely a preconditioner that might happen to work, but the natural response to a specific failure mode of classical AMG on constraint-augmented operators. The argument has three steps: (i) classical AMG fails on operators with constraint-induced near-null subspaces (§1.1), (ii) those near-null subspaces can be characterized geometrically rather than algebraically (§1.2), and (iii) the same structural pattern that motivated AMGF in the contact-IP setting of Petrides et al. is present, with tight analogy, in the mortar PBC setting (§1.3 and §1.4). Together these establish that AMGF's success in the interior-point contact application is not accidental and should carry over to the present problem.

### 1.1 Why plain AMG fails on constrained operators

**A motivating scenario.** Picture a 3D polycrystal RVE under macroscopic strain, with mortar periodic BCs enforcing $u^+ - u^- = \bar{\varepsilon} \cdot (x^+ - x^-)$ on opposite face pairs. The bulk material is elastic-plastic with grain-to-grain heterogeneity — some grains carry load, others have already yielded and shed it. Three things are happening at once that AMG was not designed to handle simultaneously:

1. The bulk operator $K$ has heterogeneous coefficients (grain stiffness contrast). Standard AMG handles this with a high strength threshold $\theta$, which prevents coarsening from accidentally bridging across grain interfaces.
2. The constraint operator $C$ ties together DOFs on opposite faces that are physically far apart in the mesh. From AMG's strength-graph point of view, these are "unconnected" — yet the saddle structure couples them implicitly through the multiplier.
3. The Wohlmuth dual basis with corner modifications breaks biorthogonality at the periodicity-face corners. The resulting Schur complement $\hat{S}$ has localized ill-conditioning at exactly those corner multipliers.

Each of these features creates **mode shapes that the bulk AMG cannot see**: displacement fields that are nearly zero-energy in the matrix entries that AMG examines but whose true energy is governed by the constraint structure. The rest of this section makes that precise.

**How classical AMG works.** Classical AMG [Henson 2002, Brezina 2005] builds a hierarchy of coarse problems based on a **strength-of-connection** graph derived from the matrix entries of $A$: two DOFs $i, j$ are "strongly connected" if $|A_{ij}|$ exceeds a threshold (typically $\theta \cdot \max_k |A_{ik}|$ with $\theta \in [0.25, 0.9]$). Coarse DOFs are chosen to be sparse independent sets in the strength graph; interpolation operators are built from the strong-connection structure. The construction is purely algebraic — AMG looks only at the matrix entries and never at the underlying geometry.

This works beautifully for diagonally dominant or strongly elliptic operators where the strength graph reflects physical coupling and the **algebraically smooth** error modes (those eigenvectors of $A$ associated with small eigenvalues, i.e., the "soft modes" that store little energy when excited) are approximately constant on aggregates of strongly-connected DOFs. For pure linear elasticity in 3D, the algebraically smooth modes are the six rigid-body modes (RBMs) — three translations and three rotations. The translations are perfectly captured by constant-per-aggregate coarsening; the rotations are *not* (they vary linearly across an aggregate), and AMG needs explicit help to represent them. The standard remedy in the literature is `HypreBoomerAMG::SetElasticityOptions`, which injects RBM coordinates and modifies the interpolation via the LS (least-squares) procedure of [Baker 2010] to preserve them exactly across grid levels. For *nonlinear* elasto-plasticity, however, this approach interacts poorly with the evolving plastic operator, and §4.5 documents why ExaConstit does not use it and how its omission is compensated for. The relevant point for the present argument is the structural one: AMG's coarsening sees only the matrix entries, and any near-null subspace that is not captured by the strength graph (whether RBM-induced or constraint-induced) requires explicit augmentation.

**What "constraint-induced near-null subspace" means physically.** A near-null mode of an operator $A$ is a vector $v$ such that $v^T A v$ is much smaller than $v^T v$ — i.e., a direction in DOF-space along which the operator stores very little energy per unit norm. For a free elastic body, the near-null modes are the rigid-body translations and rotations: the body can move in those directions without storing any strain energy. *Adding a constraint changes which modes are soft*. Concretely:

- For the **dual Schur** $\hat{S} = C K^{-1} C^T$: this is the constraint-mode compliance (§0.7). A near-null mode of $\hat{S}$ is a Lagrange multiplier direction $\lambda$ for which $\lambda^T \hat{S} \lambda$ is small — meaning that applying a unit "constraint-restoring force" in that direction generates only a tiny resulting constraint mismatch, because the body is very stiff against that particular constraint-restoring loading. Two physical sources produce this:
  - Multiplier directions $\lambda$ for which $C^T \lambda$ is nearly in the kernel of $K^{-1}$, i.e. for which the constraint is nearly redundant in the metric induced by $K$ (the bulk can satisfy the constraint at almost no energy cost).
  - Multiplier directions concentrated at geometric features where the dual-basis approximation degenerates: corners, wirebasket edges, regions where Wohlmuth corner modifications break biorthogonality. At these locations the multiplier-to-displacement coupling $C^T$ has a very localized, almost-collinear support, and the resulting Schur entries are pathological.

  These modes are *localized* on the constraint surface $\Gamma$ and do not extend into the bulk. AMG applied directly to $\hat{S}$ cannot coarsen them effectively because the surface stencil of $\hat{S}$ is dense and short-ranged; standard strength-of-connection fails to identify the right aggregates.

- For the **augmented operator** $K_\gamma = K + \gamma C^T C$ that appears in Path D (see §5): this operator has near-null modes that are *bulk displacement fields satisfying the constraint exactly*. Physically these are the deformations the body would "naturally" undergo under macroscopic strain — periodic displacement fields with no periodicity mismatch. They store full bulk strain energy (so AMG sees them as not particularly soft) but they are also exactly in $\ker(C)$, so the $\gamma C^T C$ augmentation term contributes nothing extra. AMG's strength-of-connection, which sees only the matrix entries $K + \gamma C^T C$, cannot tell which entries come from the bulk and which from the rank-$n_\lambda$ augmentation; the augmentation perturbs the coarsening graph in ways that the algorithm has no principled way to undo.

In both cases, the geometry of the problematic subspace is *known a priori*: it lives on $\Gamma$ (or its bulk-neighborhood). This is the structural opportunity AMGF exploits — instead of asking AMG to discover the problematic modes algebraically, AMGF tells it directly where to look.

### 1.2 The "problematic subspace" concept, formalized

Let $A$ be SPD and let $B$ be a candidate preconditioner. We say a subspace $W \subset \mathbb{R}^n$ is **problematic for $B$** if $B$'s effective spectrum is much worse on $W$ than on its $A$-orthogonal complement $V$. Concretely, define the *effective spectrum* of the preconditioned operator $BA$ on any subspace $U \subset \mathbb{R}^n$ as

$$\kappa(BA \big|_U) := \sup_{v \in U \setminus 0} \frac{v^T A v}{v^T B^{-1} v} \cdot \sup_{v \in U \setminus 0} \frac{v^T B^{-1} v}{v^T A v}$$

Then $W$ is $\delta$-**problematic for $B$** if there exists a complementary subspace $V$ with $\kappa(BA \big|_V) \leq \bar{\kappa}$ and $\kappa(BA \big|_W) \geq (1 + \delta) \bar{\kappa}$ for some $\delta > 0$ and some $\bar{\kappa}$. Informally: $W$ is the set of directions on which $B$ is a *bad* preconditioner; $V$ is the set on which $B$ is *good*. Throughout the rest of the document, $W$ denotes the problematic subspace and $V$ denotes its $A$-orthogonal complement; this is the convention adopted by [Petrides 2025] and propagated to the Petrides theorem of §3.10.

The Xu–Zikatanov subspace correction theorem [Xu 2002], specialized to this setting, states: if $\dim W = n_c$ is small (say, $n_c \leq c_* \, n^{2/3}$ for some constant $c_*$ and a geometrically defined surface subspace), then enriching $B$ with an *exact* solve on $W$ produces a new preconditioner $\tilde{B}$ with $\kappa(\tilde{B} A) \leq \kappa_V + O(1)$ where the $O(1)$ term is a fixed small constant and $\kappa_V$ is the conditioning of $B$ on the complement. The exact form of this bound, with all the constants, is the content of the AMGF analysis [Petrides 2025, Theorem 5.10] (cf. §3.10 below).

For ExaConstit's mortar PBC, the problematic subspace is exactly the set of DOFs on the periodicity boundary $\Gamma$ (Path A applied to $K$, see §2.2) or a subset of the multiplier space (Path B applied to $\hat{S}$, see §2.3) or the boundary-adjacent displacement DOFs again (Path D applied to the augmented $K_\gamma$, see §2.5 and §5). All three problematic subspaces are small relative to $n_u$, geometrically defined, and accessible to a sparse direct solver.

**A concrete picture.** For an FEM reader, the cleanest mental model is this: consider a polycrystal RVE with $n_u \sim 10^7$ displacement DOFs, of which maybe $5 \times 10^5$ ($\sim 5\%$) live on or adjacent to the periodicity faces. The bulk preconditioner $B$ (BoomerAMG) handles 95% of the DOFs effectively — interior grains, smooth deformation fields, the kind of elastic-plastic behavior multigrid was designed for. But on the boundary 5%, the constraint $C$ introduces couplings that AMG's strength graph cannot detect (the two opposite faces are tied together through the mortar map, but they're separated by the entire body in the mesh). Mode shapes localized on those boundary DOFs that satisfy the constraint approximately are nearly invisible to AMG: they store very little "AMG-detected energy" but their actual physical energy depends on the constraint coupling, which AMG doesn't see. Those mode shapes are the problematic subspace $W$. The strategy of AMGF is to fix this *not* by making AMG smarter — that's hard and brittle — but by augmenting AMG with an exact solve on those specific boundary DOFs, sidestepping the discovery problem altogether.

### 1.3 Structural analogy with contact-IP

**What is an interior-point method, briefly.** Solid-mechanics readers who haven't worked with optimization-style contact algorithms may not have encountered the interior-point (IP) approach in detail. An IP method handles inequality constraints (e.g., non-penetration: gap $\geq 0$) by replacing the hard inequality with a *barrier function* — a smooth penalty that becomes infinitely steep as the constraint is approached from the feasible side. The barrier is scaled by a parameter $\mu > 0$, and as $\mu$ is driven toward zero through a sequence of inner solves, the iterates converge to the true constrained solution. Each inner solve (at a fixed $\mu$) linearizes around the current iterate and solves a saddle system in which the slack variables and dual multipliers appear. As $\mu \to 0$, that saddle system becomes increasingly ill-conditioned because the barrier-induced diagonal terms approach zero on active constraints and blow up on inactive ones. The Petrides–Hartland–Kolev paper developed AMGF precisely to deal with this controlled-ill-conditioning regime.

**The contact-IP linearized saddle system.** The Petrides–Hartland–Kolev paper [Petrides 2025] developed AMGF for the interior-point (IP) approach to frictionless contact, where each IP barrier iteration solves a regularized saddle system of the form

$$\begin{bmatrix} K + J^T \Sigma J & J^T \\ J & -\Sigma^{-1} \end{bmatrix} \begin{bmatrix} du \\ d\lambda \end{bmatrix} = \cdots$$

where $J$ is the gap Jacobian and $\Sigma = \text{diag}(z_i / s_i)$ is the diagonal scaling matrix arising from slack variables $s_i$ and dual multipliers $z_i$ at active contact constraints. (We use $\Sigma$ here, rather than the letter $D$ that the original paper sometimes employs, to avoid collision with the Wohlmuth diagonal $D$ of §0.3.) As IP convergence proceeds, $\Sigma$ becomes increasingly ill-conditioned (some entries go to zero on inactive constraints, others to infinity on active ones), creating ill-conditioned near-null modes in the primal Schur complement $K + J^T \Sigma J$ that are localized on the contact interface.

The mapping to mortar PBC is:

| Concept | Contact-IP [Petrides 2025] | Mortar PBC (ExaConstit) |
|---|---|---|
| Bulk operator | $K$ (elasticity) | $K$ (elastic-plastic tangent) |
| Constraint operator | $J$ (gap Jacobian) | $C$ (dual-basis mortar) |
| Constraint metric | $\Sigma$ (IP diagonal, ill-conditioned) | none (zero (2,2) block) |
| Primal operator solved | $K + J^T \Sigma J$ (SPD) | $K$ alone, or $K + \gamma C^T C$ in augmented form |
| Problematic subspace | Contact-active DOFs | RVE-boundary DOFs |
| Subspace size | $\mathcal{O}(n^{2/3})$ | $\mathcal{O}(n^{2/3})$ |
| Source of ill-conditioning | IP barrier parameter $\mu \to 0$ | Polycrystal heterogeneity + corner-modified Wohlmuth dual basis |
| Filtered-subspace solver | Sparse direct (MUMPS/CPardiso) | Sparse direct (MUMPS/CPardiso) |

The analogy is structural, not literal. The contact-IP setting has $\Sigma$ as a *parameter-dependent* ill-conditioner (it depends on the IP iterate), whereas mortar PBC has *geometry-dependent* ill-conditioning (it depends on the corner modifications and material heterogeneity). The AMGF machinery is agnostic to the source of the ill-conditioning; it only requires that the problematic subspace be small and identifiable.

### 1.4 Why the analogy is tight rather than loose

A subtle point worth emphasizing: in contact-IP, the contact constraint is *inequality-constrained* and the IP barrier introduces a parameter $\mu$ that biases the solution. The unmodified KKT residual is recovered only in the limit $\mu \to 0$. The "saddle system" being solved at each IP iteration is therefore a *regularized* saddle, with the regularization carrying physical meaning.

In mortar PBC, the constraint is *equality-constrained* and there is no regularization parameter. The saddle system is *unregularized* — the (2,2) block is exactly zero.

This sounds like a disanalogy, but it isn't. The AMGF construction does not depend on the regularization; it depends only on the fact that there is an SPD operator (the primal Schur in contact-IP, the augmented operator $K_\gamma$ or the dual Schur $\hat{S}$ in mortar PBC) on which AMG plus a small-subspace direct solve produces uniform convergence. The geometric setting in mortar PBC is, if anything, *cleaner* than in contact-IP, because the problematic subspace is fixed once the periodicity mesh is built, whereas in contact-IP it depends on the active-set decision and changes between iterations.

---

## Section 2 — Integration Paths into the ExaConstit Pipeline

The AMGF preconditioner is a general construction for symmetric positive-definite operators: given an SPD matrix $A$, a subspace $W \subset \mathbb{R}^n$ identified as problematic for a candidate preconditioner $B$, and a smoother that handles the complement of $W$ well, AMGF produces an effective preconditioner via subspace correction. The mortar PBC saddle system in ExaConstit is, however, *not* a single SPD operator — it is a $2 \times 2$ block indefinite system. Applying AMGF therefore requires an architectural decision: *which SPD operator inside or derived from the saddle structure will AMGF act on?*

Four natural answers exist. This section enumerates them, lays out the structural rationale for each, and explains why the recommended sequencing is Path A first, Path D second, with Path B held in reserve and Path C deprioritized. The mathematical construction of the AMGF preconditioner itself — the explicit definition of the subspace basis $P$, the filtered subspace operator $P^T A P$, the iteration pseudocode, and the convergence theorem — is deferred to §3, which can then reference the paths defined here without forward references.

### 2.1 Overview of the four integration paths

The existing `MortarSaddlePreconditioner` infrastructure in `src/mortar_pbc` is structured around a `BlockDiagonalPreconditioner` with a K-block and a Schur-block component. Four natural attachment points for AMGF exist within or as modifications of this framework:

- **Path A** — AMGF replaces the K-block preconditioner inside the existing block-diagonal Schur preconditioner. The Schur-block approximation $\text{diag}(\hat{S})^{-1}$ is retained.
- **Path B** — AMGF replaces the Schur-block preconditioner. The K-block is unchanged (BoomerAMG on $K$).
- **Path C** — The block-diagonal preconditioner is replaced by a block-triangular variant; AMGF is applied to the K-block within that triangular structure.
- **Path D** — The saddle system itself is reformulated via the Powell–Hestenes augmented Lagrangian; AMGF is applied to the augmented (1,1) block $K_\gamma = K + \gamma C^T C$, and the Schur-block preconditioner becomes the trivial scaling $\gamma I$.

The four paths differ along three structural axes: (i) which block of the preconditioner is targeted by AMGF, (ii) whether the saddle system itself is modified, and (iii) whether the saddle preconditioner remains block-diagonal or becomes block-triangular. Paths A and B operate on the original saddle system with the original block-diagonal preconditioner; Path C changes the preconditioner structure to block-triangular; Path D changes the saddle system itself.

The four paths are not mutually exclusive — Paths A and D can in principle be combined (AMGF on $K_\gamma$ inside an augmented saddle), and one could imagine combining Paths B and D as well. The discussion below treats each path independently for clarity, with combinations briefly addressed in §2.6.

### 2.2 Path A — AMGF on the K-block of the existing block-diagonal preconditioner

**Description.** Replace the K-block preconditioner inside the existing block-diagonal Schur preconditioner with an AMGF wrapper around the same `HypreBoomerAMG` instance. The Schur block of the saddle preconditioner is unchanged — still the diagonal lumped approximation $\text{diag}(\hat{S})^{-1}$ computed via `MortarConstraintOperator::ComputeInvDiagSchur`. The AMGF subspace basis $P$ restricts to displacement DOFs whose support intersects the periodicity boundary $\Gamma$.

**Mathematical structure.** The saddle preconditioner becomes

$$P^{-1}_A = \begin{bmatrix} M_{\text{AMGF}}^{-1}(K) & 0 \\ 0 & \text{diag}(\hat{S})^{-1} \end{bmatrix}$$

where $M_{\text{AMGF}}^{-1}(K)$ denotes the AMGF preconditioner applied to $K$; the full construction is given in §3.

**Why this is attractive.** The existing block-diagonal saddle preconditioner has two structural weaknesses: a crude diagonal Schur approximation, and a K-block preconditioner whose effectiveness depends on the BoomerAMG quality on the bulk elastic-plastic operator. Path A addresses the second weakness specifically, and does so at the point where AMGF is *structurally* the right tool: the near-null modes of $K$ on a heterogeneous polycrystal with mortar constraints localize at material interfaces and at $\Gamma$. AMGF's subspace correction at boundary-adjacent DOFs captures the boundary-localized modes exactly. The interface-localized modes are partly captured by the existing BoomerAMG configuration's high strength threshold (cf. §4.2), so the combination is complementary.

**Limitations.** Path A leaves the Schur-block approximation untouched. If the dominant source of slow Krylov convergence is poor Schur conditioning rather than K-block conditioning, Path A produces no improvement.

**Diagnostic to determine whether Path A is appropriate.** Run the existing preconditioner on a representative problem and log $\|r_K\|$ and $\|r_\lambda\|$ separately versus Krylov iteration. If $\|r_K\|$ stagnates while $\|r_\lambda\|$ decays rapidly, the K-block is the bottleneck and Path A is the appropriate choice. If $\|r_\lambda\|$ stagnates while $\|r_K\|$ decays, the Schur block is the bottleneck and Path B or Path D is needed instead. If both stagnate, the issue is more fundamental and Path D is likely the right intervention.

**Implementation cost.** Minimal. The MFEM 4.9 `AMGFSolver` class is designed for exactly this kind of wrapping: one wraps the existing `HypreBoomerAMG` instance, supplies a `HypreParMatrix` for $P$ encoding the boundary-adjacent DOFs, and supplies a direct subspace solver (MUMPS or CPardiso). No changes to the outer Krylov method or to the Schur-block preconditioner are required.

### 2.3 Path B — AMGF on the dual Schur complement $\hat{S}$

**Description.** Leave the K-block preconditioner unchanged (BoomerAMG on $K$). Replace the diagonal lumped Schur approximation with an AMGF preconditioner targeting the dual Schur complement $\hat{S} = C K^{-1} C^T$. The AMGF subspace basis $P$ now lives in $\mathbb{R}^{n_\lambda}$ and restricts to multiplier DOFs corresponding to corner-modified constraints or to constraints adjacent to high-contrast material interfaces.

**Mathematical structure.** The saddle preconditioner becomes

$$P^{-1}_B = \begin{bmatrix} V_{\text{AMG}}(K) & 0 \\ 0 & M_{\text{AMGF}}^{-1}(\hat{S}) \end{bmatrix}$$

where $V_{\text{AMG}}(K)$ denotes one application of the existing BoomerAMG V-cycle as a preconditioner.

**Why this is attractive.** When the bottleneck is in the multiplier block, Path B directly addresses it. The Wohlmuth corner modifications [Wohlmuth 2001] necessarily break biorthogonality at corner multiplier rows in order to recover partition-of-unity, and the resulting Schur complement has localized ill-conditioning at exactly those corner multipliers — precisely the kind of localized problematic subspace AMGF was designed to correct.

**Why this is operationally difficult.** AMGF requires assembly of the filtered subspace operator $P^T \hat{S} P$. For Path B, $\hat{S} = C K^{-1} C^T$ is not assembled explicitly because the $K^{-1}$ factor would require a full direct factorization of $K$, infeasible at production scale. The MFEM 4.9 `AMGFSolver` API expects the operator $A$ as a `HypreParMatrix` from which $P^T A P$ can be formed by sparse matrix multiplication — it does not natively support the $A = C K^{-1} C^T$ composition. Realizing Path B therefore requires one of (i) explicit assembly of $\hat{S}$ (infeasible), (ii) approximation of $K^{-1}$ via per-column BoomerAMG V-cycles, building $Z^T V_{\text{AMG}}(K) Z$ as a surrogate subspace operator with $Z := C^T P$ (the topic of §3.6), or (iii) an inner iterative solve for each column of $Z$. Options (ii) and (iii) introduce inexactness that is not directly covered by the Petrides convergence theorem of §3.10.

**Limitations.** The non-trivial implementation cost is the primary barrier. Additionally, identifying the correct multiplier subset for $P$ requires exposing internal information from the `MortarConstraintOperator` class (which multiplier rows are corner-modified, which are adjacent to high-contrast interfaces) that is not currently part of its public interface.

### 2.4 Path C — AMGF inside a block-triangular preconditioner

**Description.** Replace the block-diagonal saddle preconditioner with a block-triangular form, and apply AMGF to the K-block within that triangular structure. The block-triangular preconditioner and its inverse are

$$P_C = \begin{bmatrix} K & C^T \\ 0 & -\hat{S} \end{bmatrix}, \quad P^{-1}_C = \begin{bmatrix} K^{-1} & K^{-1} C^T \hat{S}^{-1} \\ 0 & -\hat{S}^{-1} \end{bmatrix}$$

with $M_{\text{AMGF}}^{-1}(K)$ substituted for $K^{-1}$ and a Schur approximation substituted for $\hat{S}^{-1}$.

**Why this is theoretically attractive.** The Murphy–Golub–Wathen result [Murphy 2000] establishes that an exact block-triangular Schur preconditioner produces GMRES convergence in 2 iterations, versus 3 iterations for the exact block-diagonal form. Path C therefore has a slight theoretical edge in the idealized regime.

**Why Path C is deprioritized.** Four practical considerations weigh against Path C in the current setting:

1. **Loss of symmetric Krylov methods.** The block-triangular preconditioner is non-symmetric even when both $K$ and $\hat{S}$ are SPD. This forces use of GMRES rather than MINRES for symmetric $K$, forfeiting the short-recurrence advantages of MINRES — bounded storage, predictable per-iteration cost, no restart logic.
2. **Additional matrix-vector product per iteration.** Each preconditioner application requires an extra matvec with $C^T$ (or $C$) to compute the off-diagonal coupling term $K^{-1} C^T \hat{S}^{-1}$, increasing per-iteration cost relative to Paths A and B.
3. **The theoretical advantage degrades in the inexact-Schur regime.** The 2-versus-3 iteration count advantage applies only with exact $\hat{S}^{-1}$. With a diagonal lumped Schur approximation or with an AMGF-on-$\hat{S}$ surrogate (i.e., the realistic regime), the advantage diminishes substantially; in practice Paths A and C require similar Krylov iteration counts.
4. **Implementation complexity outweighs benefit.** A block-triangular preconditioner is more invasive to implement than a block-diagonal one, requiring custom block-application logic and careful handling of the off-diagonal coupling term in parallel. The cost-benefit ratio is unfavorable compared with Path A or Path D.

Path C is therefore not pursued in the recommended roadmap. It is documented here for completeness and to make clear that the choice not to pursue it is deliberate rather than an oversight.

### 2.5 Path D — Augmented Lagrangian reformulation, AMGF on $K_\gamma$

**Description.** Reformulate the saddle system using the Powell–Hestenes augmented Lagrangian (full derivation in §5). The (1,1) block of the augmented saddle system becomes $K_\gamma = K + \gamma C^T C$, which is SPD when $K$ is SPD on $\ker(C)$. Apply AMGF to $K_\gamma$. The Schur-block preconditioner becomes the trivial scaling $\gamma I$, which §5.8 shows to be spectrally equivalent to $\hat{S}_\gamma^{-1}$ up to $O(1/\gamma)$ corrections.

**Mathematical structure.** The augmented saddle system is

$$\begin{bmatrix} K_\gamma & C^T \\ C & 0 \end{bmatrix} \begin{bmatrix} du \\ d\lambda \end{bmatrix} = -\begin{bmatrix} r_K + \gamma C^T r_\lambda \\ r_\lambda \end{bmatrix}$$

and the preconditioner is

$$P^{-1}_D = \begin{bmatrix} M_{\text{AMGF}}^{-1}(K_\gamma) & 0 \\ 0 & \gamma I \end{bmatrix}$$

**Why this is structurally favorable.** Path D simultaneously addresses both weaknesses of the existing preconditioner. The augmented (1,1) block is SPD and amenable to AMG, with AMGF correcting the boundary subspace. The multiplier block requires no special preconditioning beyond a scalar multiply. The augmentation also automatically rebalances the displacement and multiplier residual scales (cf. §6), addressing the orders-of-magnitude residual asymmetry observed in the existing pipeline.

**Limitations.**

1. **Assembly of $K_\gamma$.** One must compute and store $K + \gamma C^T C$. This is one `HypreParMatrix::Add` plus a $C^T C$ product, both standard MFEM operations. The hierarchical AMG must be built on $K_\gamma$ rather than $K$ — same setup-cost order, but rebuilt at every Newton step alongside $K$ itself.
2. **Choice of $\gamma$.** Too small: the augmentation does not help, and the system behaves like the unaugmented saddle. Too large: $\kappa(K_\gamma)$ grows linearly in $\gamma$, degrading AMG performance. The natural scaling $\gamma \sim \|K\|_F / \|C^T C\|_F$ centers the parameter in the productive range; the sensitivity analysis in §5.10 shows robustness to factor-of-ten perturbations from this natural scale.
3. **Interaction with the existing nonlinear solver.** The augmented form changes the linearized Newton system; it does not change the underlying nonlinear problem. No outer Powell–Hestenes multiplier update is required at the Newton-linearization level (§5.9 explains why), so the existing Newton and trust-region machinery is unaffected.

### 2.6 Side-by-side comparison

| Property | Path A | Path B | Path C | Path D |
|---|---|---|---|---|
| AMGF applied to | $K$ | $\hat{S}$ | $K$ inside triangular | $K_\gamma = K + \gamma C^T C$ |
| Saddle system modified | No | No | No | Yes (augmented) |
| Preconditioner structure | Block-diagonal | Block-diagonal | Block-triangular | Block-diagonal |
| Outer Krylov | MINRES / GMRES | MINRES / GMRES | GMRES | MINRES / GMRES |
| Schur-block preconditioner | diag lumped (existing) | AMGF on $\hat{S}$ | diag lumped | $\gamma I$ (trivial) |
| Addresses K-block conditioning | Yes | No | Yes | Yes |
| Addresses Schur-block conditioning | No | Yes | No | Yes (by construction) |
| Addresses residual imbalance | No | No | No | Yes |
| Petrides theorem applies cleanly | Yes (SPD $K$) | Approximation-dependent | Yes (SPD $K$) | Yes (SPD $K_\gamma$) |
| Implementation effort | 1 week | 3–4 weeks | 2 weeks | 2–3 weeks |
| Code intrusiveness | Minimal | Moderate | Moderate | Moderate |
| Parameter tuning required | None | None | None | $\gamma$ |
| Risk of regression | Low | Moderate | Moderate | Low |

A combination Path A + Path D is structurally consistent and would in principle stack the benefits of both, but the practical recommendation is to test them sequentially rather than simultaneously: the data from Path A informs whether Path D is needed, and conflating the two changes makes attribution of any improvement (or regression) difficult.

### 2.7 Recommended sequencing

The recommended sequencing reflects an explicit risk-management strategy: start with the cheapest, lowest-risk modification, and escalate to more invasive changes only when the data justifies them.

**Phase 1 — Path A.** Lowest implementation cost. Directly tests whether the K-block is the conditioning bottleneck in the existing pipeline. If iteration counts improve substantially, the hypothesis is confirmed and a productive solution is in hand for the current generation of problems.

**Phase 2 — Path D, conditional on the Phase 1 outcome.** If Path A shows modest improvement but the Schur-block conditioning remains a bottleneck, Path D is the recommended next step. The augmented Lagrangian reformulation simultaneously addresses the Schur conditioning, the residual imbalance, and provides the cleanest mathematical structure for future analysis.

**Phase 3 — Production benchmarks of the combined Phases 1+2 approach.** Before considering further AMGF extensions, the Path A (+ Path D, if implemented) approach is validated across the full ExaConstit production envelope: large-scale polycrystal RVEs, non-associated flow, GPU-resident runs, sub-XYZ periodicity. Detailed task list and diagnostic measurements are in §9.

**Phase 4 — Path B, only if the production benchmarks of Phase 3 reveal multiplier-block conditioning as a remaining bottleneck.** Path B has the highest implementation cost and the least mature theoretical guarantee due to the $K^{-1}$ surrogate inexactness. It is appropriate only if Phases 1–3 leave specific failure cases (e.g., problems with severe corner-modification-induced ill-conditioning that Path D does not naturally resolve) inadequately preconditioned.

**Path C is not pursued** for the reasons given in §2.4.

With the integration paths now defined, §3 develops the mathematical construction of the AMGF preconditioner itself. That construction is essentially the same across all paths; only the operator it acts on and the choice of subspace basis differ.

---

## Section 3 — Explicit Construction of the AMGF Preconditioner

This section defines the AMGF preconditioner mathematically: the subspace basis $P$, the filtered subspace operator $P^T A P$, the iteration pseudocode, and the convergence theorem. The construction is general — it produces a preconditioner for any SPD operator $A$ once a problematic subspace and a candidate smoother are specified — and §3.5 explains how it specializes to each of the integration paths defined in §2.

Worked examples are sized to be verifiable by hand. The full machinery rests on three pieces of theory: the Xu–Zikatanov subspace correction framework (§3.1), the principal-submatrix structure of $P^T A P$ when $P$ is a Boolean prolongation (§3.4), and the Sherman–Morrison-style decomposition underlying the Petrides convergence bound (§3.10).

### 3.1 The Xu–Zikatanov subspace correction framework

**Plain-English statement of what subspace correction does.** A subspace correction preconditioner is built from two ingredients: (i) a "global" preconditioner $B$ that works well on most of the problem but is bad on a small problematic subspace, and (ii) an *exact* solve on that small subspace. The construction combines them so that the resulting preconditioner inherits $B$'s effectiveness everywhere $B$ is already good, *and* gets exact action on the modes where $B$ fails. Concretely, the bulk preconditioner $B$ (in our case, a single BoomerAMG V-cycle on the elastic-plastic tangent stiffness) handles all the bulk displacement modes — but is blind to the constraint-induced near-null modes on the periodicity boundary. Subspace correction tacks an exact direct solve on those boundary-DOF degrees of freedom directly onto $B$'s action. The result is a preconditioner that converges in iteration counts independent of mesh size and constraint conditioning.

There are two ways to combine the two ingredients — *additive* (apply both and add the corrections) or *multiplicative* (apply them in sequence with proper interleaving). The multiplicative form is more expensive per application but produces tighter convergence guarantees, and is what AMGF actually uses.

**The algebra.** Let $A$ be SPD on $\mathbb{R}^n$. The **additive subspace correction preconditioner** with smoother $B$ and subspace basis $P \in \mathbb{R}^{n \times n_c}$ is

$$M_{\text{add}}^{-1} := B + P (P^T A P)^{-1} P^T$$

Reading this from right to left in its action on a residual $r$: the second term restricts $r$ to the small subspace ($P^T r$), solves the small problem exactly ($(P^T A P)^{-1}$ — a sparse direct factorization of the boundary-DOF principal submatrix), and prolongates back into the global space ($P \cdot \text{result}$). The first term $B \cdot r$ is the bulk preconditioner action. The additive form sums these two corrections.

The **multiplicative subspace correction preconditioner** is defined by its error propagation operator:

$$I - M_{\text{mult}}^{-1} A = (I - B A)(I - \Pi_A^P)(I - B A)$$

where $\Pi_A^P := P (P^T A P)^{-1} P^T A$ is the $A$-orthogonal projection onto $\text{range}(P)$ (cf. §0.5). The structure is: pre-smooth with $B$, then correct exactly on $\text{range}(P)$, then post-smooth with $B$. The pre/post-smoothing makes $M_{\text{mult}}$ symmetric when $B$ is symmetric.

The Xu–Zikatanov theorem [Xu 2002] states: for the multiplicative form,

$$\kappa(M_{\text{mult}}^{-1} A) \leq \frac{1 + 2 c_1^2 c_0}{(2 - c_1)^2 (1 - \omega_B)}$$

where $c_0, c_1$ are the **stable decomposition** and **strengthened Cauchy–Schwarz** constants relating $A$, $B$, and the decomposition $\mathbb{R}^n = \text{range}(P) \oplus \text{range}(I - \Pi_A^P)$, and $\omega_B$ is a smoother-quality constant. (The letters $c_0, c_1$ are used here, rather than the $K_0, K_1$ of Xu and Zikatanov's original notation, to avoid collision with the tangent stiffness $K$ of §0.3.) The precise statement (which is the foundation of [Petrides 2025, Theorem 5.10]) gives explicit bounds in terms of the spectral equivalence constants of $B$ on the complementary subspace.

For our purposes, the practical content is: if $B$ is a "good" preconditioner on the complementary subspace (i.e. effective on bulk DOFs away from $\Gamma$) and the subspace solve is exact (which it is, by direct factorization), then $\kappa(M_{\text{mult}}^{-1} A) \leq C_*$ for a constant $C_*$ that depends only on the AMG quality on the complement and *not* on the conditioning of $A$ on $\text{range}(P)$. (The subscripted $C_*$ here is a generic constant, written this way to avoid collision with the mortar constraint matrix $C$ of §0.3.)

### 3.2 Additive vs. multiplicative variants

The additive form $M_{\text{add}}^{-1} = B + P (P^T A P)^{-1} P^T$ is conceptually simpler and naturally symmetric whenever $B$ is. Its convergence theory is also slightly looser; the multiplicative form gives sharper constants but at the cost of additional smoother applications.

In the [Petrides 2025] implementation distributed with MFEM 4.9, AMGF uses the **multiplicative form** (two AMG V-cycles plus one subspace solve per preconditioner application), with the smoother $B$ being a *single V-cycle of HypreBoomerAMG* — not a simple Jacobi or Gauss–Seidel smoother. This is the key practical choice: the inner V-cycle handles the bulk operator effectively, and the subspace solve corrects what the V-cycle alone cannot capture.

A common simplification seen in introductory expositions is to write the additive form with the simplest possible smoother $B = \text{diag}(A)^{-1}$. This is *much* weaker than what the Petrides implementation actually uses; it should be regarded as a sketch of the structure rather than a serviceable preconditioner. The functional AMGF preconditioner replaces the diagonal smoother with a BoomerAMG V-cycle, so a useful conceptual shorthand for the additive variant is

$$M_{\text{add}}^{-1} = V_{\text{AMG}}(A) + P (P^T A P)^{-1} P^T$$

where $V_{\text{AMG}}(A)$ denotes a single application of the BoomerAMG V-cycle on $A$ as a preconditioner (not a solve). The multiplicative form pre- and post-applies $V_{\text{AMG}}(A)$ around the subspace correction, per the error-propagation equation in §3.1.

**Notation convention used throughout the remainder of the document.** We write $M_{\text{AMGF}}(A)$ for the multiplicative AMGF preconditioner applied to a generic SPD operator $A$, with the operator made explicit in the argument to avoid confusion when AMGF is applied to different operators in different paths. Specifically:

- $M_{\text{AMGF}}(K)$ — AMGF applied to the tangent stiffness $K$ (Paths A and C).
- $M_{\text{AMGF}}(K_\gamma)$ — AMGF applied to the augmented stiffness $K_\gamma = K + \gamma C^T C$ (Path D).
- $M_{\text{AMGF}}(\hat{S})$ — AMGF applied to the dual Schur complement $\hat{S} = C K^{-1} C^T$ (Path B).

When the operator is unambiguous, we drop the argument and write simply $M_{\text{AMGF}}$, or — in the abstract setting of the Petrides convergence theorem in §3.10 — bare $M$.

### 3.3 What is $P$, explicitly?

$P \in \mathbb{R}^{n \times n_c}$ is the **tall, sparse, Boolean** matrix whose columns are the standard basis vectors $e_i$ of $\mathbb{R}^n$ for $i$ in the problematic index set $\mathcal{I} \subset \{1, \ldots, n\}$, arranged in column order.

In the notation of the preliminaries, $P$ here is exactly the prolongation matrix $R^T$ of §0.4 — the transpose of the Boolean restriction $R$ defined there, equivalently the prolongation that appeared in the Galerkin coarse-operator construction of §0.6. We use the same letter $P$ throughout the document for this object; the path-specific preconditioners $P_K$, $P^{-1}_A$, $P^{-1}_D$, etc. introduced in §2 are *subscripted* and refer to preconditioners rather than to the prolongation. The AMGF construction can therefore be read as Galerkin coarse-graining on a *geometrically chosen* small subspace, paired with a candidate preconditioner $B$ (the smoother in subspace-correction language) that handles the complement of that subspace.

In pseudocode, given `idx[]` a list of length $n_c$ of zero-based indices into the global vector:

```python
P = scipy.sparse.csr_matrix((n, n_c))
for j in range(n_c):
    P[idx[j], j] = 1.0
```

In MFEM, $P$ is constructed as a `HypreParMatrix` from row/column index pairs, one nonzero per column. The MFEM 4.9 `AMGFSolver::SetFilteredSubspaceTransferOperator` API accepts $P$ as a `HypreParMatrix*`.

**For Path A** (AMGF on the K-block of the saddle preconditioner), $\mathcal{I}$ is the set of displacement DOF indices adjacent to the periodicity boundary $\Gamma$:

$$\mathcal{I}_K = \{ i : \text{DOF } i \text{ has support intersecting } \Gamma \}$$

**For Path C** (AMGF on $K$ inside a block-triangular preconditioner), $\mathcal{I}$ is identical to Path A's $\mathcal{I}_K$ — the AMGF subspace depends only on which SPD operator is preconditioned, not on the surrounding block structure.

**For Path D** (AMGF on the augmented $K_\gamma = K + \gamma C^T C$), $\mathcal{I}$ is again the boundary-adjacent displacement DOFs $\mathcal{I}_K$. The same $P$ is used as in Path A; only the operator $A$ that AMG sees changes from $K$ to $K_\gamma$.

**For Path B** (AMGF on the dual Schur $\hat{S}$), $\mathcal{I}$ is a *subset of multiplier indices* $\mathcal{I}^\lambda \subset \{1, \ldots, n_\lambda\}$ — specifically those for which the dual basis is corner-modified or the polycrystal phase contrast is sharp. This is a subtler choice; see §2.3 for the structural discussion and §3.6 below for the operational difficulties it introduces.

### 3.4 What does $P^T A P$ look like, with worked examples?

Because $P$ is a Boolean prolongation (the transpose of a Boolean restriction, in the language of §0.4), the matrix $P^T A P$ is the **principal submatrix** of $A$ indexed by $\mathcal{I}$:

$$(P^T A P)_{jk} = A_{i_j, i_k}, \quad i_j, i_k \in \mathcal{I}$$

This is *exactly* the construction in §0.4–§0.6 — $P^T A P$ is the Galerkin coarse operator $A_c$ of §0.6 specialized to a Boolean prolongation — and §0.6 already noted that the three names ("principal submatrix," "Galerkin coarse operator," "filtered subspace operator") refer to the same object.

**Toy example 1.** Let $n = 5$ and let $A$ be the standard 1D-Laplacian-on-uniform-mesh stiffness:

$$A = \begin{bmatrix} 2 & -1 & 0 & 0 & 0 \\ -1 & 2 & -1 & 0 & 0 \\ 0 & -1 & 2 & -1 & 0 \\ 0 & 0 & -1 & 2 & -1 \\ 0 & 0 & 0 & -1 & 2 \end{bmatrix}$$

Take $\mathcal{I} = \{1, 5\}$ (the two boundary nodes). Then

$$P = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 1 \end{bmatrix}, \quad P^T A P = \begin{bmatrix} A_{11} & A_{15} \\ A_{51} & A_{55} \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

The boundary nodes are not directly coupled in this stencil, so $P^T A P$ is diagonal. In a 3D mesh with $\mathcal{I}$ being the entire boundary surface, $P^T A P$ would have full surface-coupling structure: each boundary node coupled to its boundary neighbors.

**Toy example 2.** Same $A$ but take $\mathcal{I} = \{1, 2, 5\}$ — boundary nodes plus one adjacent interior node. Then

$$P^T A P = \begin{bmatrix} A_{11} & A_{12} & A_{15} \\ A_{21} & A_{22} & A_{25} \\ A_{51} & A_{52} & A_{55} \end{bmatrix} = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & 0 \\ 0 & 0 & 2 \end{bmatrix}$$

Now node 2's coupling to node 1 is preserved, but its coupling to node 3 (which is interior and excluded from $\mathcal{I}$) is lost. This is the *Galerkin restriction* effect: information about the boundary's interaction with the bulk is severed in $P^T A P$. The AMGF preconditioner compensates for this by combining the subspace solve with a full V-cycle on $A$ that captures bulk-to-bulk coupling.

**Realistic example.** For an ExaConstit polycrystal RVE at $n_u = 10^7$ DOFs with $\mathcal{I}_K = 5 \cdot 10^5$ (5% of total), $P^T K P$ is a $5 \cdot 10^5 \times 5 \cdot 10^5$ sparse SPD matrix with the surface-stencil sparsity pattern: each row has $\sim 30$ nonzeros (corresponding to the boundary node's neighbors that are also in $\mathcal{I}_K$). The MUMPS direct factorization of such a matrix has approximate cost $\mathcal{O}(n_c^{1.5}) \sim 10^{8}$ floating-point operations, comparable to a single coarse-grid solve in the AMG hierarchy. The factorization is *reusable* across all Krylov iterations within a single Newton step, so its amortized cost is small.

### 3.5 Specialization of the construction to each integration path

The integration paths defined in §2 differ in *which* SPD operator the AMGF correction is applied to. The construction above is general, but it is useful to record explicitly what $P$ and $P^T A P$ look like in each case — particularly the dimensions, the assembly cost, and (for Path B) the operational difficulty introduced by the $K^{-1}$ factor.

**Path A — AMGF on $K$.** The AMGF preconditioner is applied to the (1,1) block of the saddle preconditioner, namely $K$ itself. The subspace lives in $\mathbb{R}^{n_u}$, so $P \in \mathbb{R}^{n_u \times n_c}$ with $n_c = |\mathcal{I}_K|$, the count of displacement DOFs adjacent to $\Gamma$. The filtered subspace operator is $P^T K P \in \mathbb{R}^{n_c \times n_c}$, assembled by a single sparse matrix-matrix multiplication. No composition with $C$ is involved at the AMGF level; the constraint matrix $C$ enters only through the outer Schur preconditioner of the saddle system.

**Path C — AMGF on $K$ inside a block-triangular preconditioner.** The AMGF construction itself is identical to Path A: same $P$, same $P^T K P$, same subspace solve. Path C differs only in how the resulting K-block preconditioner is combined with the Schur preconditioner inside the outer saddle preconditioner (block-triangular instead of block-diagonal). Because the AMGF preconditioner depends only on $K$ and on $\mathcal{I}_K$, not on the surrounding block structure, the construction below applies unchanged.

**Path D — AMGF on $K_\gamma = K + \gamma C^T C$.** The AMGF preconditioner is applied to the augmented (1,1) block. The subspace structure is identical to Path A: $P \in \mathbb{R}^{n_u \times n_c}$, $P^T K_\gamma P \in \mathbb{R}^{n_c \times n_c}$. The filtered subspace operator is

$$P^T K_\gamma P = P^T K P + \gamma (C P)^T (C P)$$

which requires the auxiliary matrix $C P \in \mathbb{R}^{n_\lambda \times n_c}$ — a wide-and-short sparse matrix obtained by extracting the columns of $C$ corresponding to $\mathcal{I}_K$. Assembly is two sparse matrix-matrix multiplications plus one sparse matrix addition, all standard operations in MFEM's `HypreParMatrix` API.

**Path B — AMGF on the dual Schur $\hat{S} = C K^{-1} C^T$.** Here the AMGF preconditioner is applied to the (2,2) block of the (Schur-complement-based) saddle preconditioner. The subspace lives in $\mathbb{R}^{n_\lambda}$, so $P \in \mathbb{R}^{n_\lambda \times n_c^\lambda}$ with $n_c^\lambda$ the count of multiplier DOFs identified as problematic (corner-modified multipliers, or those adjacent to sharp polycrystal interfaces). The filtered subspace operator is

$$P^T \hat{S} P = P^T C K^{-1} C^T P = (C^T P)^T K^{-1} (C^T P) = Z^T K^{-1} Z$$

where $Z := C^T P \in \mathbb{R}^{n_u \times n_c^\lambda}$ is a sparse tall-and-thin matrix whose columns are linear combinations of columns of $C^T$. (We use the letter $Z$ here to avoid collision with $W$, which throughout the document denotes the *problematic subspace* in the Petrides framework — cf. §1.2 and §3.10.) This is where the computational difficulty of Path B enters: computing $Z^T K^{-1} Z$ requires applying $K^{-1}$ to each of the $n_c^\lambda$ columns of $Z$, which is the central topic of §3.6 below.

The auxiliary matrix $Z := C^T P$ appears only in the Path B treatment. For Paths A, C, and D, no $Z$ matrix is needed.

### 3.6 Computing $Z^T K^{-1} Z$ in Path B

For Path B, applying $(P^T \hat{S} P)^{-1}$ requires forming the small dense matrix $Z^T K^{-1} Z$ with $Z = C^T P$. This in turn requires applying $K^{-1}$ to each column of $Z$. Three options:

**Exact direct solve.** Use MUMPS or CPardiso to factor $K$, then forward/back-substitute on each column of $Z$. Cost: one factorization plus $n_c^\lambda$ back-substitutions per Newton step. Memory: $\mathcal{O}(n_u^{4/3})$ for the factor. Tractable for $n_u \lesssim 10^7$, infeasible above.

**Approximate via BoomerAMG V-cycle.** Replace $K^{-1}$ with $V_{\text{AMG}}(K)$, the action of a single V-cycle. Cost: $n_c^\lambda$ V-cycle applications per Newton step. Memory: same as the existing AMG hierarchy. The resulting $Z^T V_{\text{AMG}}(K) Z$ is no longer exactly $Z^T K^{-1} Z$ but is a spectrally equivalent surrogate with constants depending on the AMG quality.

**Approximate via PCG to loose tolerance.** Use BoomerAMG-preconditioned CG to solve $K x_i = z_i$ for each column $z_i$ of $Z$, to a moderate tolerance (say $10^{-3}$). Cost: $\sim 10$–$20$ V-cycles per column. More accurate than option 2, more expensive.

The MFEM 4.9 `AMGFSolver` implementation uses option 1 by default (sparse direct solver on the subspace operator), constructing the subspace operator algebraically as $P^T A P$ without any reference to $K^{-1}$. This is feasible because in Petrides et al.'s setting, $A$ is the *primal Schur* $K + J^T \Sigma J$ rather than $K^{-1}$; the inversion of $K$ is not required. For our Path B, where AMGF is applied to $\hat{S} = C K^{-1} C^T$, we would need either to assemble $\hat{S}$ explicitly (infeasible) or to approximate it via one of the options above. **This makes Path B operationally harder than Path A or Path D in the existing MFEM API**, and is a major reason to prefer Path A or Path D in the initial implementation.

### 3.7 AMGF pseudocode

For Path A applied to $K$ as the (1,1) block of the saddle preconditioner:

```
# Setup phase (once per Newton step, when K is refreshed):
#   1. Build P ∈ R^{n_u × n_c} from list of boundary-adjacent DOFs
#   2. Form K_filt = P^T K P   (sparse SpMM, O(nnz(K)) work)
#   3. Factor K_filt via MUMPS/CPardiso, store factor
#   4. Build BoomerAMG hierarchy for K (existing setup)

# Application phase (per outer Krylov iteration, applying M^{-1} to vector r):
function apply_amgf(r):
    # Pre-smooth: one BoomerAMG V-cycle on r
    y1 = V_AMG(K) · r

    # Subspace correction: project r onto P-subspace, solve, prolongate
    r_filt = P^T · (r - K · y1)              # restrict residual to subspace
    e_filt = K_filt^{-1} · r_filt            # MUMPS solve, small
    y2 = y1 + P · e_filt                     # add subspace correction

    # Post-smooth: one BoomerAMG V-cycle on updated residual
    y_final = y2 + V_AMG(K) · (r - K · y2)

    return y_final
```

This is the standard multiplicative subspace correction with V-cycle pre/post-smoothing. The pre/post structure makes the overall preconditioner symmetric when the V-cycle is symmetric (which is satisfied by the symmetric smoothers (l1-Jacobi, l1-symmetric Gauss-Seidel) configured in the existing setup).

For Path D applied to $K_\gamma = K + \gamma C^T C$, the only change is that $V_{\text{AMG}}$ is built on $K_\gamma$ rather than $K$, and the subspace operator becomes $P^T K P + \gamma (CP)^T (CP)$.

### 3.8 Cost analysis

Per Newton step:

- **Setup:** Build $P$ ($\mathcal{O}(n_c)$); form $P^T K P$ ($\mathcal{O}(\text{nnz}(K))$ for the SpMM); factor via MUMPS ($\mathcal{O}(n_c^{1.5})$ for typical surface sparsity in 3D); build BoomerAMG hierarchy ($\mathcal{O}(\text{nnz}(K))$ setup cost — same as existing).

- **Per outer Krylov iteration:** Two V-cycle applications ($2 \cdot \mathcal{O}(\text{nnz}(K))$ — same as a current single V-cycle plus one extra); one $P^T \cdot$ and one $P \cdot$ ($\mathcal{O}(n_c)$ each — negligible); one MUMPS forward/back-substitution ($\mathcal{O}(n_c \log n_c)$ to $\mathcal{O}(n_c^{4/3})$ depending on sparsity). For $n_c \sim n_u^{2/3}$, the MUMPS substitution cost is $\mathcal{O}(n_u^{8/9})$, asymptotically dominated by the V-cycle cost $\mathcal{O}(n_u)$.

The bottom line: AMGF roughly *doubles* the per-V-cycle cost compared to a single AMG V-cycle, plus adds a setup cost that amortizes across all Krylov iterations within a Newton step. If AMGF reduces outer Krylov iteration count by more than $2\times$ (which it typically does, often $5\text{--}10\times$ for the kind of problematic-subspace pathology we're targeting), it wins overall.

### 3.9 Memory analysis

- **MUMPS factor for $P^T K P$:** approximately $\mathcal{O}(n_c \log n_c)$ to $\mathcal{O}(n_c^{4/3})$ nonzero entries. For $n_c = 5 \cdot 10^5$ this is roughly $10^7$–$10^8$ entries, i.e. tens of megabytes to a few hundred megabytes per MPI rank.
- **AMG hierarchy memory:** unchanged from existing setup.
- **$P$ matrix storage:** $\mathcal{O}(n_c)$, negligible.

The dominant memory cost is the MUMPS factor. For GPU-resident ExaConstit runs, this factor lives on the CPU side; the data exchange per Krylov iteration is a single vector of length $n_c$. At $n_c = 5 \cdot 10^5$ floats, this is 2 MB per iteration — manageable but worth profiling.

### 3.10 The Petrides convergence theorem, in plain language

[Petrides 2025, Theorem 5.10] gives the central convergence result for AMGF. In our notation:

> **Theorem (Petrides–Hartland–Kolev et al. 2025).** Let $A$ be SPD on $\mathbb{R}^n$. Let $W = \text{range}(P)$ for a Boolean $P \in \mathbb{R}^{n \times n_c}$, and let $V$ be the $A$-orthogonal complement of $W$. Let $B$ be a preconditioner for $A$ that is spectrally equivalent to $A^{-1}$ on $V$ with constant $\beta$:
>
> $$v^T A v \leq \beta \, v^T B^{-1} v \quad \forall v \in V$$
>
> and similarly $v^T B^{-1} v \leq v^T A v$ on $V$ (which is the natural lower bound for a reasonable preconditioner). Then the AMGF multiplicative preconditioner $M$ satisfies
>
> $$\kappa(M^{-1} A) \leq 2(\beta + 3)$$

The constant 2 comes from the pre/post smoothing structure; the $+3$ is a fixed overhead from the subspace projection geometry. The constant $\beta$ is the *only* problem-dependent quantity, and it represents the quality of the V-cycle (or other smoother) on the *bulk* subspace, *with the boundary modes filtered out*.

**Why this theorem matters more than it might look.** Two features of the bound $2(\beta + 3)$ are worth pausing on:

1. **It's $\beta + 3$, not $\beta^2$ or $\beta \cdot \kappa_W$.** A naive worry would be that combining a bulk preconditioner (quality $\beta$) with a subspace solve (handling some other conditioning $\kappa_W$) could give a combined bound that multiplies the two — meaning your overall preconditioner is only as good as the *worse* of the two. The Xu–Zikatanov framework, and Petrides's specialization of it, instead give an *additive* bound: the cost of the subspace correction is a fixed constant (the "+3"), not a multiplicative factor that depends on how bad the problematic subspace was. This is the structural mathematical content of the theorem and the reason AMGF works at all: the problematic subspace can be arbitrarily ill-conditioned (any $\kappa_W$ up to $10^{12}$ for a degenerate corner-modified Wohlmuth case), and the AMGF bound is *unchanged*. The exact direct solve on the subspace absorbs all of that conditioning into a one-time factorization cost.

2. **Only $\beta$ — the AMG quality on the bulk — appears in the bound.** The whole job of designing AMGF is therefore to make sure $\beta$ is small (i.e., AMG works well on bulk operators where the problematic modes have been filtered out), and then *not worry* about the conditioning of the constraint modes. This is exactly the inverse of the usual preconditioner-design effort, which spends most of its time trying to handle the bad-conditioning modes; AMGF says, "just solve them exactly, the cost is bounded."

This is a powerful result. It says: AMGF transforms a preconditioner that may be very bad on the full space (because of the boundary modes) into one that is uniformly good, provided the preconditioner is reasonable on the complement of the boundary modes. The reduction in iteration count is roughly $\sqrt{\kappa_{\text{full}} / \kappa_M} = \sqrt{\kappa_{\text{full}} / (2(\beta + 3))}$ for CG convergence.

In practice, for ExaConstit's polycrystal problems, $\beta$ is expected to be in the range $\beta \in [5, 50]$ depending on AMG configuration and polycrystal heterogeneity (see §4 for detailed estimates). With $\beta = 20$, the bound gives $\kappa(M^{-1} A) \leq 46$, and PCG converges in $\sqrt{46} \log(1/\epsilon) \approx 7 \log(1/\epsilon)$ iterations — perhaps 50–100 iterations for $\epsilon = 10^{-8}$. Compared to a problematic-subspace-induced $\kappa \sim 10^6$ scenario (which is realistic for polycrystals with sharp coefficient jumps), this represents a $\sim 100\times$ iteration count reduction.

---


## Section 4 — The $K^{-1}$ Surrogate Problem under ExaConstit's Actual AMG Configuration

The Petrides convergence theorem (§3.10) hinges on the AMG preconditioner being a good preconditioner for the bulk operator on the complementary subspace. The bound $\kappa(M^{-1} A) \leq 2(\beta + 3)$ depends linearly on the AMG quality $\beta$. This section dissects the BoomerAMG configuration deployed in ExaConstit to estimate $\beta$ and discusses whether changes to that configuration are warranted.

### 4.1 The BoomerAMG configuration deployed in ExaConstit

The BoomerAMG instance in `src/system_driver.cpp` is configured with the following parameters. There are two versions in the codebase, an older and a newer.

**Older configuration (lines that use direct Hypre API calls):**
- `HYPRE_BoomerAMGSetMaxLevels(h_amg, 30)` — up to 30 multigrid levels
- `HYPRE_BoomerAMGSetCoarsenType(h_amg, 0)` — CLJP coarsening (Cleary–Luby–Jones–Plassmann)
- `HYPRE_BoomerAMGSetMeasureType(h_amg, 0)` — local measure for parallel coarsening
- `HYPRE_BoomerAMGSetStrongThreshold(h_amg, 0.90)` — strong threshold $\theta = 0.90$ (very high)
- `HYPRE_BoomerAMGSetNumSweeps(h_amg, 3)` — three smoother sweeps per level
- `HYPRE_BoomerAMGSetRelaxType(h_amg, 8)` — l1-symmetric Gauss–Seidel relaxation
- `HYPRE_BoomerAMGSetNumFunctions(h_amg, 3)` — three "functions" (treats x, y, z displacement components as separate unknowns)
- `HYPRE_BoomerAMGSetSmoothType(h_amg, 6)` — Schwarz smoother on coarsest levels
- `HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3)` — apply Schwarz smoother on first 3 levels
- `HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3)` — three Schwarz sweeps per smoother application
- `HYPRE_BoomerAMGSetVariant(h_amg, 0)`, `HYPRE_BoomerAMGSetOverlap(h_amg, 0)`, `HYPRE_BoomerAMGSetDomainType(h_amg, 1)` — Schwarz configuration: non-overlapping subdomains, one DOF per subdomain (so this is essentially a block-Jacobi smoother on Schwarz blocks)

**Newer configuration (the `SetSystemsOptions` path):**
- `prec_amg->SetSystemsOptions(problem_dim=3, order_bynodes)` — MFEM's wrapper for the same systems-AMG setup, with explicit DOF ordering passed
- All other parameters at MFEM/Hypre defaults

**The configuration deliberately does not call `SetElasticityOptions`.** That method is the standard mechanism for injecting rigid-body modes into the BoomerAMG interpolation, and on pure-elasticity benchmarks it produces a $\sim 4\times$ improvement in convergence rate. For nonlinear elasto-plasticity, however, empirical testing has shown it to be counterproductive — the underlying reasons are documented in §4.5. Without `SetElasticityOptions`, BoomerAMG treats the displacement field as a generic 3-component system (via `SetSystemsOptions(3)`) and does not attempt to encode rotational rigid-body modes in its interpolation operators. The compensating mechanism is the conservative high strength threshold $\theta = 0.90$ combined with the strong smoother choice; together these produce a multigrid hierarchy that converges robustly on heterogeneous polycrystal problems in the plastic regime.

### 4.2 What CLJP coarsening does

CLJP (Cleary–Luby–Jones–Plassmann) coarsening [Henson 2002, §3.1] is a parallel coarsening algorithm that selects coarse points by a randomized graph-coloring procedure on the strength-of-connection graph. The strength graph has an edge $(i,j)$ when $|A_{ij}| \geq \theta \max_{k \neq i} |A_{ik}|$. CLJP coarsens by selecting a maximal independent set of "first-pass" coarse points, then a second pass refines to ensure the coarse problem is well-defined.

For $\theta = 0.90$, only the very strongest connections count as "strong" — typically only the nearest neighbors in a regular mesh with isotropic coefficients, and only the cross-interface coupling in problems with sharp coefficient jumps. This is an empirical tuning targeting heterogeneous polycrystal problems: high $\theta$ prevents the coarsening from crossing material interfaces inappropriately, at the cost of slower coarsening rates and more multigrid levels.

The trade-off: high $\theta$ is robust to heterogeneity but produces a *less aggressive* coarsening, meaning coarse problems shrink more slowly and the AMG hierarchy is deeper (up to 30 levels per the configured `MaxLevels`). Deeper hierarchies mean longer setup and per-V-cycle costs.

### 4.3 Smoother choices: l1-symmetric Gauss–Seidel and Schwarz

The `RelaxType=8` setting is l1-symmetric Gauss–Seidel, a relaxation method that uses a diagonal scaling based on $\ell_1$-row norms of the off-diagonal entries:

$$x_i^{(k+1)} = x_i^{(k)} + \omega \frac{r_i^{(k)}}{|A_{ii}| + \sum_{j \neq i, A_{ij} \text{ strong}} |A_{ij}|}$$

This is a robust smoother for problems where the diagonal alone is a poor scaling (e.g. anisotropic or jumping coefficients), which is exactly the polycrystal regime. The "symmetric" qualifier means forward and backward sweeps are alternated, preserving symmetry of the V-cycle.

The `SmoothType=6` adds an additional Schwarz smoother on the first 3 levels. With `DomainType=1` and `Overlap=0`, this is a non-overlapping block-Jacobi on Schwarz domains — essentially a stronger local smoother on the fine and near-fine levels, where the polycrystal coefficient jumps are most pronounced.

Together, these smoother choices represent careful tuning for heterogeneous elasticity. They are *not* the BoomerAMG defaults, and they reflect substantial domain knowledge.

### 4.4 `SetSystemsOptions`: what it does

`HypreBoomerAMG::SetSystemsOptions(dim, ordering)` configures BoomerAMG to treat the unknowns as a "PDE system" with `dim` components per node. It sets:
- `NumFunctions = dim`
- DOF function indices so that BoomerAMG knows which DOFs correspond to which component
- "Hybrid" interpolation that respects the system structure

What this *does* do: prevent the coarsening from aggregating, say, an $x$-displacement DOF with a $y$-displacement DOF at the same spatial node. It maintains the component structure on coarse levels.

What this **does not** do: inject rigid-body modes into the interpolation. The coarse-level operators are still constructed by Galerkin projection of the fine-level operators, and the interpolation operators do not preserve rigid-body rotations.

For a homogeneous, isotropic linear elasticity problem with no Poisson-ratio incompressibility complications, `SetSystemsOptions` alone produces decent convergence — typically $\rho \approx 0.2\text{--}0.3$ per V-cycle. For heterogeneous polycrystals with sharp grain-boundary coefficient jumps, the convergence rate degrades to $\rho \approx 0.4\text{--}0.6$, depending on the contrast ratio.

### 4.5 `SetElasticityOptions`: what it does, and why ExaConstit does not use it

`HypreBoomerAMG::SetElasticityOptions(fespace)` does what `SetSystemsOptions` does, **plus** it:

- Reads nodal coordinates from the FE space.
- Constructs the six rigid-body mode vectors (three translations, three rotations).
- Modifies the interpolation operator using the LS (least-squares) interpolation of [Baker 2010] so that rigid-body modes are preserved exactly across grid levels.

The LS interpolation is a non-trivial modification: it changes how fine-grid DOFs are interpolated from coarse-grid DOFs by requiring that, when the coarse representation is a rigid-body translation or rotation, the fine representation reproduces it exactly. This is the AMG analog of imposing rigid-body partition of unity in geometric multigrid.

**On pure elasticity, the literature reports significant improvements.** The Baker–Kolev–Yang paper [Baker 2010] reports verbatim results: convergence rate $\rho = 0.07$ for a standard 3D elasticity test with `SetElasticityOptions`, versus $\rho = 0.31$ without — a $4\times$ improvement equivalent to halving the per-Krylov-iteration count for a fixed residual reduction target.

**On nonlinear elasto-plasticity, however, empirical evidence shows the opposite.** Testing of `SetElasticityOptions` on ExaConstit's actual crystal-plasticity workloads has consistently shown that enabling it causes the outer Newton iteration to diverge as soon as plastic flow initiates. By contrast, BoomerAMG configured *without* elasticity options — but with `SetSystemsOptions(3)` for the displacement-component structure, the high strength threshold $\theta = 0.90$, and the l1-symmetric Gauss–Seidel and Schwarz smoothers described above — converges robustly across the full plastic deformation history and actually accelerates outer Newton convergence as well.

The structural reasons for this discrepancy are several, and consistent with broader experience injecting near-null-space information into algebraic preconditioners for nonlinear constitutive models:

1. **The rigid-body-mode assumption is specific to linear elasticity.** The LS interpolation in [Baker 2010] is built on the premise that the small-eigenvalue (algebraically smooth) modes of $K$ are well approximated by the six rigid-body vectors derived from undeformed nodal coordinates. For the linear elastic tangent stiffness this is approximately true. For an elastic-plastic tangent stiffness undergoing plastic redistribution, the algebraically smooth modes are *not* the original rigid-body modes — they are perturbed by the evolving plastic strain field, which redirects energy into different soft directions. Interpolation operators that exactly preserve elastic RBMs at every level then misrepresent the actual near-null space of the plastic operator, and the Galerkin coarse-grid operators inherit that misrepresentation. The result is a multigrid hierarchy that fails to coarsen the operator effectively in the plastic regime.

2. **The coordinate vector is computed once at AMG setup and then frozen.** The rigid-body mode vectors injected by `SetElasticityOptions` come from undeformed nodal coordinates — they do not update as the body deforms. For finite-deformation crystal plasticity in particular, this means the assumed "rigid-body rotation" mode at AMG setup time progressively diverges from the actual rotation kinematics of plastically deformed grains. The interpolation operator becomes less and less representative of the actual operator's near-null space as deformation accumulates.

3. **Node-ordering assumptions interact poorly.** ExaConstit's broader infrastructure has multiple implicit assumptions about how nodes are ordered (`byNodes` vs `byVDIM`, MPI partitioning, parallel renumbering by `Hypre`). `SetElasticityOptions` expects a specific coordinate-vector ordering that must match these other assumptions exactly; any mismatch causes the LS interpolation to be built on wrong coordinate data, with effects ranging from suboptimal preconditioning to outright divergence.

4. **The reported speedups are for linear solves only.** Even where `SetElasticityOptions` produces faster linear convergence on a single tangent system, the value of that speedup is moot if the outer nonlinear iteration diverges or if the multigrid convergence rate is brittle to small changes in the operator structure between Newton iterations.

**Conclusion for the implementation roadmap.** The BoomerAMG configuration currently deployed in `src/system_driver.cpp` — `SetSystemsOptions(3)`, high `StrongThreshold`, l1-symmetric Gauss–Seidel relaxation, Schwarz smoothing on the upper levels — is the correct baseline. It is the result of substantial empirical tuning for the actual nonlinear-plasticity regime ExaConstit targets, and it produces both robust V-cycle convergence and well-behaved outer Newton convergence. AMGF is applied *on top of* this configuration; `SetElasticityOptions` is not enabled.

### 4.6 Puantitative estimate of $\beta$ for the ExaConstit setup

Putting this together, the Petrides bound parameter $\beta$ for the ExaConstit BoomerAMG configuration on a heterogeneous polycrystal can be estimated from the V-cycle convergence rate $\rho$:

$$\beta \approx \frac{1 + \rho}{1 - \rho}$$

(this is the standard conversion from contraction rate to spectral condition number for the preconditioned operator).

For a representative polycrystal RVE with contrast ratio $\sim 5\text{--}10\times$ between hard and soft grains:
- **Optimistic estimate** ($\rho = 0.3$): $\beta \approx 1.86$, AMGF bound $\kappa(M^{-1}A) \leq 9.7$ — outstanding.
- **Realistic estimate** ($\rho = 0.5$): $\beta \approx 3$, AMGF bound $\leq 12$ — excellent.
- **Pessimistic estimate** ($\rho = 0.7$, severe heterogeneity): $\beta \approx 5.7$, AMGF bound $\leq 17.4$ — still good.

PCG with $\kappa \leq 20$ converges in roughly $\sqrt{20} \log(1/\epsilon) \approx 4.5 \log(1/\epsilon)$ iterations, so about 40 iterations for $\epsilon = 10^{-8}$. This is *much* better than the iteration counts typically observed in problematic-subspace pathology, which can easily be in the 200–500 range.

For more severe heterogeneity (porosity, $\rho_{\max}/\rho_{\min} \sim 100\times$), $\rho$ could exceed 0.8, giving $\beta \approx 9$ and AMGF bound $\leq 24$. Still tractable.

These estimates apply to the ExaConstit baseline AMG configuration as it stands; the bound $\beta + 3$ does not require any modification of the underlying preconditioner. AMGF's role is precisely to take an underlying AMG that may have moderate-quality $\beta \in [3, 10]$ on the bulk and lift it to uniform convergence regardless of the constraint-induced ill-conditioning.

### 4.7 Implications for the implementation roadmap

The combination of three facts — that AMGF requires only a moderate-quality AMG on the bulk to produce uniform convergence, that the deployed BoomerAMG configuration delivers exactly this quality, and that `SetElasticityOptions` is empirically counterproductive for nonlinear elasto-plasticity — has a clean implication: **the bulk AMG component should not be touched as part of the AMGF implementation.** AMGF wraps it; it does not modify it.

This is a significant simplification of the implementation effort:

1. No need to modify `src/system_driver.cpp`'s AMG setup logic.
2. No need to thread node-coordinate data through the preconditioner construction.
3. No need to manage memory growth from RBM-aware coarsening.
4. No risk of regressing the Newton convergence behavior in the plastic regime.

Instead, the entire AMGF intervention happens at the level of the saddle preconditioner construction in `src/mortar_pbc/mortar_saddle_preconditioner.hpp`, with the existing `HypreBoomerAMG` instance used unmodified as the smoother inside the AMGF preconditioner.

### 4.8 Alternatives to BoomerAMG

For severely heterogeneous problems where the deployed BoomerAMG configuration plus AMGF still leaves large iteration counts, the literature offers alternatives. None is a drop-in replacement for ExaConstit's existing infrastructure, but all are worth knowing about.

**BDDC (Balancing Domain Decomposition by Constraints) [Mandel 1993].** A two-level domain decomposition method that explicitly handles rigid-body modes and constraint-style coarse spaces. Production-grade implementation in PETSc as `PCBDDC`. Strengths: provably robust for heterogeneous elasticity; handles incompressibility well; well-suited for moderate parallelism. Weaknesses: requires non-trivial subdomain decomposition; not natively GPU-resident; combining with matrix-free RAJA partial assembly is non-obvious.

**FETI-DP (Finite Element Tearing and Interconnecting, Dual-Primal) [Farhat 2001].** Dual formulation of BDDC. Same robustness, same implementation complexity. Slightly different parallel-decomposition assumptions.

**GenEO (Generalized Eigenproblems in the Overlaps) [Spillane 2014].** Builds robust two-level preconditioners by solving local generalized eigenvalue problems on subdomain overlaps. The abstract states verbatim: *"We achieve this by solving local generalized eigenvalue problems in the overlaps of subdomains that isolate the terms responsible for slow convergence. We prove a general theoretical result that rigorously establishes the robustness of the new coarse space."* Provably robust for heterogeneous PDEs, including elasticity systems with high contrast. One-time setup cost is heavy; runtime is comparable to AMG. Available as part of FreeFem++ and PETSc.

**Adaptive Smoothed Aggregation ($\alpha$SA) [Brezina 2005].** Discovers near-nullspace modes via an adaptive bootstrap process. Robust for problems where rigid-body modes are insufficient (e.g. when material anisotropy creates additional smooth modes). Available in Trilinos/MueLu. Heavy setup, comparable runtime.

**MueLu smoothed aggregation with explicit RBMs [Wiesner 2021].** A saddle-preserving AMG hierarchy with explicit rigid-body-mode injection. The Wiesner et al. paper applies this approach to mortar contact. Discussed in detail in §8. The rigid-body-mode handling here is structurally different from `SetElasticityOptions` — MueLu's smoothed aggregation builds aggregates that respect the RBMs by construction rather than imposing them through LS interpolation — and the combination with saddle structure preservation is what gives the approach its robustness. Whether it would also encounter the nonlinear-plasticity divergence issues observed with `SetElasticityOptions` is an open question.

For ExaConstit's near-term roadmap, the recommended order of investigation is:
1. AMGF on top of the deployed BoomerAMG configuration (Path A — cheap, immediate gain). This is the primary recommendation.
2. If Path A is insufficient, the augmented Lagrangian reformulation of Path D, with AMGF on $K_\gamma$. This addresses both the K-block and the Schur-block weaknesses simultaneously.
3. Only if Paths A and D both prove inadequate on the most heterogeneous problems, evaluate `PCBDDC` (PETSc) or MueLu (Trilinos) — but with awareness that these introduce substantial dependencies and code restructurings, and that any near-null-space-aware AMG variant should be empirically vetted for the same nonlinear-plasticity divergence behavior that motivates avoiding `SetElasticityOptions` in the BoomerAMG case.

GenEO and $\alpha$SA are likely overkill for the polycrystal regime ExaConstit targets and add substantial complexity.

---

## Section 5 — Augmented Lagrangian: Derivation, Conditioning, and Saddle Structure

A common point of confusion in augmented-Lagrangian preconditioning is the question: *why is the augmented system still 2×2 rather than 1×1?* The short answer is that the augmented Lagrangian preserves the Lagrange multiplier as a primary unknown; pure penalty discards it. The long answer requires deriving both approaches from first principles and comparing their conditioning.

### 5.1 Setting: equality-constrained quadratic minimization

Consider the equality-constrained quadratic program

$$\min_{u \in \mathbb{R}^{n_u}} f(u) := \tfrac{1}{2} u^T K u - b^T u \quad \text{s.t.} \quad C u = g$$

where $K \in \mathbb{R}^{n_u \times n_u}$ is symmetric and positive definite on $\ker(C)$ (otherwise no minimum exists), $C \in \mathbb{R}^{n_\lambda \times n_u}$ has full row rank, and $g \in \mathbb{R}^{n_\lambda}$. This is the canonical form of one Newton step of the ExaConstit linearized boundary-value problem with mortar PBC.

The constraint set $\{u : Cu = g\}$ is an affine subspace of $\mathbb{R}^{n_u}$, with associated linear subspace $\ker(C) = \{v : Cv = 0\}$. The minimum is unique under the standing assumptions.

### 5.2 The standard Lagrangian and the saddle system

Form the Lagrangian

$$L(u, \lambda) := f(u) + \lambda^T (C u - g) = \tfrac{1}{2} u^T K u - b^T u + \lambda^T (C u - g)$$

A standard result of convex duality is: $(u^*, \lambda^*)$ is a saddle point of $L$ (minimizer in $u$, maximizer in $\lambda$) if and only if $u^*$ solves the original constrained problem and $\lambda^*$ is the corresponding Lagrange multiplier.

Stationarity conditions:

$$\nabla_u L = K u^* - b + C^T \lambda^* = 0$$

$$\nabla_\lambda L = C u^* - g = 0$$

In matrix form:

$$\begin{bmatrix} K & C^T \\ C & 0 \end{bmatrix} \begin{bmatrix} u^* \\ \lambda^* \end{bmatrix} = \begin{bmatrix} b \\ g \end{bmatrix}$$

This recovers the existing ExaConstit system with $b = -r_K$ and $g = -r_\lambda$ on the right-hand side. The (2,2) block is exactly $0$ because there is no compliance term.

### 5.3 The pure quadratic penalty method

The pure penalty approach replaces the hard constraint with a quadratic penalty term. Define the penalized objective

$$f_\gamma(u) := f(u) + \tfrac{\gamma}{2} \|C u - g\|^2_2 = \tfrac{1}{2} u^T (K + \gamma C^T C) u - (b + \gamma C^T g)^T u + \tfrac{\gamma}{2} g^T g$$

for $\gamma > 0$. The minimizer $u_\gamma$ satisfies $\nabla f_\gamma = 0$:

$$(K + \gamma C^T C) u_\gamma = b + \gamma C^T g$$

This is a **1×1 SPD system** (provided $K \succ 0$ on $\ker(C)$ and $\gamma > 0$). No Lagrange multiplier appears.

**Properties of pure penalty:**

- **Constraint error:** $u_\gamma$ satisfies $\|C u_\gamma - g\| = O(1/\gamma)$ as $\gamma \to \infty$. The exact constraint is recovered only in the limit. Concretely, one can show $C u_\gamma - g = -\gamma^{-1} (C K^{-1} C^T)^{-1} \cdot [C K^{-1} b - g - \gamma^{-1} (\cdots)]$, with the constraint violation decaying linearly in $1/\gamma$.

- **Conditioning of the operator:** $\kappa(K + \gamma C^T C) = O(\gamma)$ as $\gamma \to \infty$. Specifically, the eigenvalues of $K + \gamma C^T C$ on $\ker(C)$ are unchanged (the constraint contributes nothing there), while the eigenvalues on $\text{range}(C^T)$ scale like $\gamma$. So $\lambda_{\max}/\lambda_{\min} \to \infty$ linearly with $\gamma$.

- **The multiplier is lost.** Pure penalty discards $\lambda$. It can be recovered approximately as $\lambda \approx \gamma (C u_\gamma - g)$, but this recovery is itself $O(1/\gamma)$-inaccurate, and the recovery formula is numerically unstable when $\gamma$ is large.

**Why pure penalty is unsuitable for ExaConstit:**

1. The mortar PBC enforcement must be *exact* (or very nearly so) for accurate computational homogenization. The downstream effective-stress calculation depends on satisfying the periodicity exactly to recover the right-hand side of the macroscopic constitutive law. A constraint violation of $10^{-3}$ is unacceptable; getting to $10^{-8}$ requires $\gamma \sim 10^8$, at which point the linear system is severely ill-conditioned.

2. Many post-processing quantities — specifically the effective stress and the reaction tractions at periodic faces — *are* the Lagrange multipliers. Pure penalty discards them.

3. The conditioning blow-up ruins AMG performance. AMG on $K + 10^8 C^T C$ would require either drastically tuned strong-thresholds or would simply fail to converge.

For these reasons, pure penalty is not used in practice for mortar PBC in computational homogenization. It is presented here only as a contrast to augmented Lagrangian, to make clear what the augmentation is *fixing*.

### 5.4 The Powell–Hestenes–Rockafellar augmented Lagrangian

The augmented Lagrangian function is

$$L_\gamma(u, \lambda) := f(u) + \lambda^T (C u - g) + \tfrac{\gamma}{2} \|C u - g\|^2_2$$

This is *simultaneously* a Lagrangian (because of the $\lambda^T (Cu - g)$ term) and a penalty function (because of the $\|Cu - g\|^2$ term). The key insight, due to Hestenes and Powell independently in 1969 [Hestenes 1969, Powell 1969], is that **the multiplier $\lambda$ is kept as a primary unknown, not eliminated**.

The crucial property: at the exact optimum $(u^*, \lambda^*)$ of the original constrained problem,

$$L_\gamma(u^*, \lambda^*) = f(u^*) + \lambda^{*T} \cdot 0 + \tfrac{\gamma}{2} \cdot 0 = f(u^*) = L(u^*, \lambda^*)$$

So the augmented and unaugmented Lagrangians *agree* at the optimum, *independently of $\gamma$*. The augmentation adds curvature in the constraint direction without biasing the solution.

This is the property that pure penalty lacks: pure penalty's minimum is *not* the original problem's minimum unless $\gamma \to \infty$. Augmented Lagrangian's saddle point *is* the original problem's saddle point for any $\gamma \geq 0$.

### 5.5 Derivation of the augmented saddle system

Step-by-step derivation of the KKT conditions for $L_\gamma$:

**Step 1.** Original problem:
$$\min_u \tfrac{1}{2} u^T K u - b^T u \quad \text{s.t.} \quad C u = g$$

**Step 2.** Standard Lagrangian:
$$L(u, \lambda) = \tfrac{1}{2} u^T K u - b^T u + \lambda^T (C u - g)$$

**Step 3.** KKT conditions of standard Lagrangian (gradient zero):
$$K u^* - b + C^T \lambda^* = 0, \quad C u^* - g = 0$$

**Step 4.** Augmented Lagrangian — add a quadratic penalty term that vanishes at the optimum:
$$L_\gamma(u, \lambda) := L(u, \lambda) + \tfrac{\gamma}{2} \|C u - g\|^2$$

Since $C u^* - g = 0$ at the optimum, $L_\gamma(u^*, \lambda^*) = L(u^*, \lambda^*)$.

**Step 5.** KKT conditions of augmented Lagrangian:
$$\nabla_u L_\gamma = K u - b + C^T \lambda + \gamma C^T (C u - g) = 0$$
$$\nabla_\lambda L_\gamma = C u - g = 0$$

**Step 6.** Rearranging the first equation:
$$(K + \gamma C^T C) u + C^T \lambda = b + \gamma C^T g$$

**Step 7.** Matrix form (the augmented saddle system):
$$\begin{bmatrix} K + \gamma C^T C & C^T \\ C & 0 \end{bmatrix} \begin{bmatrix} u \\ \lambda \end{bmatrix} = \begin{bmatrix} b + \gamma C^T g \\ g \end{bmatrix}$$

The structural insight is that **only the (1,1) block changed**. The off-diagonal $C^T$ and $C$ are unchanged. The (2,2) zero is unchanged. The multiplier $\lambda$ is still a primary unknown. The right-hand side gains a $\gamma C^T g$ contribution in the displacement block.

In the Newton-iteration context, the right-hand side terms $b$ and $g$ are replaced by negative residuals $-r_K$ and $-r_\lambda$:

$$\begin{bmatrix} K + \gamma C^T C & C^T \\ C & 0 \end{bmatrix} \begin{bmatrix} du \\ d\lambda \end{bmatrix} = -\begin{bmatrix} r_K + \gamma C^T r_\lambda \\ r_\lambda \end{bmatrix}$$

### 5.6 Why the system is still 2×2: the structural answer

The structural question — *"why isn't the augmented form just a 1×1 block?"* — has the following answer:

Pure penalty *eliminates* $\lambda$ entirely. The system becomes 1×1 in $u$ only, and the constraint is enforced approximately (with error $O(1/\gamma)$). The multiplier is unrecoverable except by lossy post-processing.

Augmented Lagrangian *keeps* $\lambda$ as an unknown. The system remains 2×2. The constraint is enforced exactly. The multiplier is a primary output.

In one sentence: **pure penalty trades exactness for system size; augmented Lagrangian gives you both at the cost of structural complexity in the preconditioner.**

The structural complexity is, however, *favorable*: the augmented (1,1) block is SPD and AMG-friendly (with AMGF), and the augmented Schur complement is bounded by $\gamma^{-1}$-times-identity, making the multiplier-block preconditioner trivial. The full preconditioner is

$$P^{-1}_\gamma = \begin{bmatrix} M_{\text{AMGF}}^{-1}(K_\gamma) & 0 \\ 0 & \gamma I \end{bmatrix}$$

with $K_\gamma = K + \gamma C^T C$. Compare to the un-augmented preconditioner

$$P^{-1}_0 = \begin{bmatrix} M^{-1}(K) & 0 \\ 0 & \hat{S}^{-1} \end{bmatrix}$$

with $\hat{S} = C K^{-1} C^T$ requiring approximation. The augmented form replaces the hard problem of approximating $\hat{S}^{-1}$ with the trivial problem of using $\gamma I$, in exchange for solving on a slightly modified (1,1) block.

### 5.7 Conditioning of the augmented system

**Mechanical interpretation of the augmentation.** Before diving into spectra, the structural picture of what $K + \gamma C^T C$ does is worth stating directly: **augmentation is the addition of springs with stiffness $\gamma$ that resist constraint violation**. The bulk stiffness $K$ acts on all displacement modes as usual. The added term $\gamma C^T C$ activates only on displacements that violate the periodicity constraint — formally, displacements in $\text{range}(C^T)$, the orthogonal complement of $\ker(C)$. On constraint-satisfying displacements (those in $\ker(C)$), $Cv = 0$ and the augmentation term contributes nothing. On constraint-violating displacements, $\gamma C^T C$ contributes an energy penalty proportional to the *square* of the violation, scaled by $\gamma$. So $K_\gamma$ is just $K$ with very stiff springs across the periodicity boundary — and the spring stiffness $\gamma$ is the augmentation parameter we get to choose. The reason the system stays well-conditioned even for large $\gamma$ is that the springs only act on the small constraint-mode subspace; the vast majority of displacement DOFs see no change.

We need to understand how $\kappa(K_\gamma)$ depends on $\gamma$, because this directly governs AMG (and AMGF) performance.

Decompose $\mathbb{R}^{n_u}$ as $\ker(C) \oplus \text{range}(C^T)$ (orthogonal under the Euclidean inner product; not under the $K$-inner product, but the Euclidean decomposition suffices for spectral bounds).

- On $\ker(C)$: $C v = 0$, so $C^T C v = C^T (Cv) = 0$. The operator $K_\gamma$ restricted to $\ker(C)$ equals $K$ restricted there. Eigenvalues unchanged.

- On $\text{range}(C^T)$: Write $v = C^T w$ for some $w \in \mathbb{R}^{n_\lambda}$. Then $C^T C v = C^T C C^T w$. The operator $C^T C$ restricted to $\text{range}(C^T)$ has eigenvalues equal to those of $C C^T$ (acting on $\mathbb{R}^{n_\lambda}$), which are bounded by $\sigma_{\min}(C)^2 \leq \mu \leq \sigma_{\max}(C)^2$.

So the eigenvalues of $K_\gamma = K + \gamma C^T C$ split into two groups:

- Group 1 (on $\ker(C)$): eigenvalues $\lambda_i(K)$ for $i$ such that the corresponding eigenvector lies in $\ker(C)$. These are unchanged by augmentation.
- Group 2 (on $\text{range}(C^T)$): eigenvalues approximately $\lambda_j(K) + \gamma \sigma_j(C)^2$ for the eigenvectors in $\text{range}(C^T)$.

The condition number $\kappa(K_\gamma) = \lambda_{\max}(K_\gamma) / \lambda_{\min}(K_\gamma)$ depends on $\gamma$ as follows:

- $\lambda_{\max}(K_\gamma) \approx \lambda_{\max}(K) + \gamma \sigma_{\max}(C)^2 \sim \gamma \sigma_{\max}(C)^2$ for large $\gamma$.
- $\lambda_{\min}(K_\gamma) = \min(\lambda_{\min}(K|_{\ker C}), \lambda_{\min}(K) + \gamma \sigma_{\min}(C)^2)$. For moderate $\gamma$, this is still $\lambda_{\min}(K)$; for large $\gamma$, it grows.

So $\kappa(K_\gamma) = O(\gamma)$ for large $\gamma$, but the prefactor is $\sigma_{\max}(C)^2 / \lambda_{\min}(K|_{\ker C})$, which can be small if the constraint operator has bounded singular values and the bulk operator is well-conditioned on its kernel.

In practice, for $\gamma$ chosen on the order of $\|K\|_\infty / \|C^T C\|_\infty$ (a natural scaling), $\kappa(K_\gamma)$ is comparable to $\kappa(K)$ — not worse. The augmentation does not significantly degrade AMG performance.

### 5.8 Spectral equivalence of the augmented Schur complement to $\gamma^{-1} I$

**Why Sherman-Morrison-Woodbury appears here.** The augmented operator $K + \gamma C^T C$ is a *low-rank update* of $K$ — specifically, a rank-$n_\lambda$ update where $n_\lambda \ll n_u$. The Sherman-Morrison-Woodbury (SMW) identity is the matrix-inverse formula for exactly this situation: given a base operator and a low-rank correction, SMW expresses the inverse of the corrected operator in terms of the inverse of the base operator and a small dense inverse on the "rank-update space." For our problem the rank-update space is the multiplier space $\mathbb{R}^{n_\lambda}$, and the small inverse will turn out to involve the unaugmented Schur complement $\hat{S}$. This is the mechanism that lets us extract a closed-form relationship between $\hat{S}_\gamma$ (the augmented Schur, which we want to precondition) and $\hat{S}$ (the unaugmented Schur, which we already understand). The closed-form result will show that for large enough $\gamma$, $\hat{S}_\gamma^{-1}$ is dominated by a multiple of the identity — which is why the trivial Schur-block preconditioner $\gamma I$ in Path D is asymptotically exact.

The Schur complement of the augmented saddle system in $\lambda$ is

$$\hat{S}_\gamma = C (K + \gamma C^T C)^{-1} C^T$$

(this is the augmented analog of the unaugmented Schur $\hat{S} = C K^{-1} C^T$ of §0.7). Computing $\hat{S}_\gamma$ in closed form via the Sherman–Morrison–Woodbury identity gives a structurally illuminating result: the *inverse* of the augmented Schur complement is the sum of the inverse of the unaugmented Schur plus a multiple of the identity.

**Derivation.** The Sherman–Morrison–Woodbury identity for a rank-update of $K$ states

$$(K + \gamma C^T C)^{-1} = K^{-1} - K^{-1} C^T (\gamma^{-1} I + C K^{-1} C^T)^{-1} C K^{-1}$$

Recognizing $\hat{S} = C K^{-1} C^T$ in the middle factor, and pre- and post-multiplying the identity by $C$ and $C^T$:

$$\hat{S}_\gamma = \hat{S} - \hat{S} (\gamma^{-1} I + \hat{S})^{-1} \hat{S}$$

To simplify, use the algebraic identity $\hat{S} = (\gamma^{-1} I + \hat{S}) - \gamma^{-1} I$ in the middle factor:

$$\hat{S} (\gamma^{-1} I + \hat{S})^{-1} \hat{S} = \big[(\gamma^{-1} I + \hat{S}) - \gamma^{-1} I\big] (\gamma^{-1} I + \hat{S})^{-1} \hat{S} = \hat{S} - \gamma^{-1} (\gamma^{-1} I + \hat{S})^{-1} \hat{S}$$

Substituting back:

$$\hat{S}_\gamma = \hat{S} - \hat{S} + \gamma^{-1} (\gamma^{-1} I + \hat{S})^{-1} \hat{S} = \gamma^{-1} (\gamma^{-1} I + \hat{S})^{-1} \hat{S}$$

Inverting:

$$\hat{S}_\gamma^{-1} = \gamma \hat{S}^{-1} (\gamma^{-1} I + \hat{S}) = \hat{S}^{-1} + \gamma I$$

**This is the central identity for Path D preconditioning.** The inverse of the augmented Schur complement decomposes exactly as the sum of two terms: a $\gamma$-independent piece $\hat{S}^{-1}$ that encodes the geometry of the unaugmented constraint, and a $\gamma$-scaled identity that grows linearly in the augmentation parameter.

**Consequence for the preconditioner.** For large $\gamma$, the $\gamma I$ term dominates and

$$\|\hat{S}_\gamma^{-1} - \gamma I\|_2 = \|\hat{S}^{-1}\|_2 = \mathcal{O}(1)$$

independent of $\gamma$. The relative error $\|\hat{S}_\gamma^{-1} - \gamma I\|_2 / \|\gamma I\|_2 = \mathcal{O}(1/\gamma)$ vanishes as $\gamma$ grows.

The choice of multiplier-block preconditioner $P_\lambda^{-1} = \gamma I$ therefore satisfies

$$\kappa(P_\lambda^{-1} \hat{S}_\gamma) = \kappa\big((\gamma I)(\gamma^{-1}(\gamma^{-1} I + \hat{S})^{-1} \hat{S})\big) = \kappa\big((\gamma^{-1} I + \hat{S})^{-1} \hat{S}\big)$$

The eigenvalues of $(\gamma^{-1} I + \hat{S})^{-1} \hat{S}$ are $\mu_i / (\gamma^{-1} + \mu_i)$ where $\{\mu_i\}$ are the eigenvalues of $\hat{S}$. As $\gamma \to \infty$, these approach $1$, so $\kappa \to 1$. For moderate $\gamma$ on the order of $1/\mu_{\min}(\hat{S})$, the condition number is bounded by approximately $\mu_{\max}(\hat{S})/\mu_{\min}(\hat{S}) \cdot \gamma \mu_{\min}(\hat{S}) = \gamma \cdot \mu_{\max}(\hat{S})$, which is moderate when $\gamma$ is chosen at the natural scaling discussed in §5.10.

**Practical implication.** The trivial Schur preconditioner $\gamma I$ is *exact* in the limit $\gamma \to \infty$ and *uniformly good* for moderate $\gamma$. This stands in stark contrast to the unaugmented case, where the diagonal lumped approximation $\text{diag}(\hat{S})^{-1}$ is only crudely accurate and degrades with mesh refinement. The augmentation has converted the hard problem of preconditioning $\hat{S}$ into the trivial problem of scaling by $\gamma$.

### 5.9 The outer Powell–Hestenes multiplier update

**Plain-language framing.** Classical augmented-Lagrangian methods for *nonlinear* equality-constrained optimization don't trust the multiplier $\lambda$ to be exactly right from the start — they iterate on it, using the current residual of the constraint to update $\lambda$ between inner solves. This is the Powell–Hestenes outer iteration. For our use, however, we're working at the *linearized* level: each Newton step is already a linear system, and the augmented Lagrangian saddle preconditioner sits inside that linear system. The point of this subsection is to explain why no outer multiplier update is needed at our level: a single full-tolerance linear solve already returns the correct $\lambda$, and the outer iteration would only matter if we deliberately under-solved the inner linear problem.

In the classical augmented Lagrangian framework, one solves the inner problem (the augmented saddle system) approximately for a fixed $\lambda$, then updates $\lambda$ via the rule

$$\lambda^{(k+1)} = \lambda^{(k)} + \gamma (C u^{(k+1)} - g)$$

and repeats. The Powell–Hestenes theorem [Hestenes 1969, Powell 1969] guarantees that this outer iteration converges to the true Lagrange multiplier provided $\gamma$ is large enough relative to the second-order data of the problem.

**For a single Newton step of the linearized ExaConstit problem, this outer iteration is unnecessary.** The inner linear solve already returns the exact $(u, \lambda)$ pair satisfying the augmented KKT system, which (by §5.4) is also the exact solution of the un-augmented system. No outer iteration is needed at the linear level.

The outer iteration becomes relevant if one chooses to solve the augmented inner problem only approximately (to relax tolerances and save Krylov iterations). In that case, $\lambda^{(k+1)} = \lambda^{(k)} + \gamma (C u^{(k+1)} - g)$ corrects for the inner inexactness, and the outer iteration converges in $O(\log(\epsilon_{\text{outer}}))$ steps. This is the "BCL" (Bound-Constrained Lagrangian) or "MINOS" framework of [Saunders Stanford CME 338 notes].

For ExaConstit's near-term implementation, we recommend running the inner linear solve to full tolerance and skipping the outer iteration. The outer iteration is a potential optimization for later, if the inner Krylov cost becomes prohibitive.

### 5.10 The choice of $\gamma$

The augmentation parameter $\gamma$ controls a trade-off:

- **Too small** ($\gamma \to 0$): the augmented system reduces to the unaugmented saddle, and AMG/AMGF on $K_\gamma$ behaves like AMG/AMGF on $K$. The trivial Schur preconditioner $\gamma I$ becomes a poor approximation to $\hat{S}_\gamma$, and the multiplier block converges slowly.
- **Too large** ($\gamma \to \infty$): $K_\gamma$ becomes dominated by $\gamma C^T C$, which is rank-$n_\lambda$. The conditioning of $K_\gamma$ blows up linearly in $\gamma$, and AMG performance degrades.

The "sweet spot" is empirically $\gamma$ on the order of $\|K\|_F / \|C^T C\|_F$ (Frobenius-norm scaling, comparable to the average diagonal entries). This makes the diagonal contributions of $K$ and $\gamma C^T C$ comparable, so neither dominates.

Concretely, for ExaConstit polycrystals, one can compute $\gamma$ once per Newton step (or once per load step) as:

$$\gamma = \alpha \cdot \frac{\text{tr}(K)/n_u}{\text{tr}(C^T C)/n_\lambda}$$

with $\alpha$ a tunable factor in $[0.1, 10]$. The default $\alpha = 1$ is a reasonable starting point.

The augmented Lagrangian literature has more sophisticated $\gamma$ selection strategies (e.g. updating $\gamma$ based on observed constraint violation across outer iterations), but for a single Newton step with no outer iteration, a fixed $\gamma$ suffices.

### 5.11 Summary table: pure penalty vs. augmented Lagrangian vs. unaugmented

| Property | Pure penalty | Augmented Lagrangian | Unaugmented saddle |
|---|---|---|---|
| System size | 1×1 in $u$ | 2×2 in $(u, \lambda)$ | 2×2 in $(u, \lambda)$ |
| Multiplier preserved | No | Yes | Yes |
| Constraint accuracy | $O(1/\gamma)$ | Exact | Exact |
| (1,1) block | $K + \gamma C^T C$ (SPD) | $K + \gamma C^T C$ (SPD) | $K$ (SPD) |
| (1,1) conditioning | $O(\gamma)$ | $O(\gamma)$ | $O(1)$ |
| Required $\gamma$ for accuracy | $\to \infty$ | $O(1)$ | N/A |
| AMG on (1,1) at large $\gamma$ | Catastrophic | Fine (with AMGF) | Fine |
| Schur block | N/A | $\sim \gamma^{-1} I$ | $\hat{S}$ requires approximation |
| Schur preconditioner | N/A | $\gamma I$ (trivial) | $\text{diag}(\hat{S})^{-1}$ or AMGF |
| Suitable for ExaConstit | No | Yes (Path D) | Yes (current path) |

---

## Section 6 — The Residual Imbalance Problem and Its Consequences

Empirical observation in ExaConstit shows $\|r_K\| \gg \|r_\lambda\|$ at the start of a Newton step, by 1 to 6 orders of magnitude depending on the problem. This asymmetry is structural and has direct implications for stopping criteria and preconditioner design.

### 6.1 Why the imbalance happens

The displacement residual $r_K = K u - b - C^T \lambda$ has units of force per node, and accumulates contributions from every interior integration point. Its norm scales roughly as $\sqrt{n_u}$ times a material-scale residual.

The multiplier residual $r_\lambda = C u - g$ is a *kinematic* residual — it measures how badly the periodic constraint is violated. Its norm scales as the displacement jump across the periodicity boundary $\Gamma$, accumulated over the boundary DOFs.

At the start of a Newton step:
- $r_K$ is large because the previous Newton iterate is not yet in equilibrium; the imbalance reflects unbalanced internal forces.
- $r_\lambda$ is small because the previous Newton iterate already satisfies (or nearly satisfies) the constraint — periodicity was already enforced.

After several Newton iterations, both decay, but the ratio $\|r_K\|/\|r_\lambda\|$ typically remains in the range $10^1$ to $10^6$ throughout the iteration.

### 6.2 What it breaks

The standard joint norm

$$\|r\| := \sqrt{\|r_K\|^2 + \|r_\lambda\|^2}$$

is dominated by $\|r_K\|$, so a stopping criterion based on $\|r\| < \epsilon$ is effectively a criterion on $\|r_K\|$ alone. The multiplier residual is driven much further below threshold than necessary, wasting Krylov iterations on a block that is already converged.

Worse, in the *opposite* situation (where $\|r_\lambda\|$ becomes the bottleneck due to constraint-induced ill-conditioning), the joint norm could be deceptively small while the multiplier component is still poorly converged. This is the failure mode that AMGF specifically targets.

### 6.3 Scaled inner products

The cleanest fix is to use a *scaled* residual norm

$$\|r\|_{\Sigma} := \sqrt{\|r_K\|^2_{\Sigma_K^{-1}} + \|r_\lambda\|^2_{\Sigma_\lambda^{-1}}}$$

where $\Sigma_K \in \mathbb{R}^{n_u \times n_u}$ and $\Sigma_\lambda \in \mathbb{R}^{n_\lambda \times n_\lambda}$ are scaling matrices chosen so that the two contributions are of comparable magnitude at the start of the iteration. Natural choices:

- $\Sigma_K = \text{diag}(K)$ — gives a Jacobi-scaled $r_K$.
- $\Sigma_\lambda = \text{diag}(\hat{S})$ — gives a Jacobi-scaled $r_\lambda$ (uses the existing `ComputeInvDiagSchur` output).

Or, more theoretically motivated:

- $\Sigma_K = M_u$, the displacement mass matrix.
- $\Sigma_\lambda = M_\lambda$, the mortar mass matrix on $\Gamma$.

The mass-matrix scaling makes $\|r\|_\Sigma$ a discrete approximation to the natural $L^2$ residual norm, which is mesh-independent. The diagonal scaling is cheaper and usually adequate.

### 6.4 Separate per-block tolerances

Alternatively (or additionally), one can use separate stopping criteria

$$\|r_K\| / \|r_K^{(0)}\| < \epsilon_K, \quad \|r_\lambda\| / \|r_\lambda^{(0)}\| < \epsilon_\lambda$$

with $\epsilon_K, \epsilon_\lambda$ independently chosen. PETSc's `KSPSetConvergenceTest` supports custom convergence tests; in MFEM one writes a custom `IterativeSolver::CheckFinalNorm`.

This is the approach used in the [Petrides 2025] implementation, where each KKT block has its own tolerance based on natural scaling.

### 6.5 The augmented Lagrangian gives partial automatic correction

In the augmented formulation, the displacement residual becomes

$$r_K^{\text{aug}} = r_K + \gamma C^T r_\lambda$$

For $\gamma$ chosen on the order of $\|K\|/\|C^T C\|$, the additional $\gamma C^T r_\lambda$ term is of magnitude comparable to $r_K$ when $r_\lambda$ is of order $1/\gamma \cdot \|r_K\|$ — i.e. *exactly the regime where the original imbalance was 1–6 orders of magnitude*. So the augmented residual is naturally balanced.

This is a side benefit of Path D that is often underappreciated: it not only improves preconditioning, it also normalizes the residual structure.

### 6.6 Practical recommendation

For the current code:
1. Add a scaled-norm option to the `MortarSaddlePointSystem` residual computation.
2. Use diagonal scaling $\Sigma_K = \text{diag}(K)$, $\Sigma_\lambda = \text{diag}(\hat{S})$ initially (zero extra setup cost).
3. Compute the imbalance ratio $\|r_K\|_\Sigma / \|r_\lambda\|_\Sigma$ in the diagnostic logger and verify it is near unity at iteration 0.

For Path D adoption, the imbalance issue largely takes care of itself.

---

## Section 7 — Implementation Roadmap in MFEM/ExaConstit

This section translates the mathematical construction of §3 and the path analysis of §2 into concrete C++ code targeting the existing ExaConstit infrastructure. It covers the MFEM 4.9 `AMGFSolver` API surface, the construction of the boundary-DOF restriction matrix $P$, the integration into the existing `MortarSaddlePreconditioner` class, the handling of non-symmetric tangent stiffness in non-associated flow, and the GPU-residency considerations that arise when the subspace solve must run on the host while the rest of the assembly runs on the device.

### 7.1 The MFEM 4.9 `AMGFSolver` API

MFEM 4.9 (released December 11, 2025) ships the `AMGFSolver` class in `linalg/filteredsolver.hpp` (included via `linalg/linalg.hpp`). The class derives from `FilteredSolver`, the abstract base for solvers with filtering. The CHANGELOG entry reads verbatim:

> Added `FilteredSolver`: a base class for solvers with filtering. ... Added `AMGFSolver`: a derived class of `FilteredSolver`, specialized for AMG with Filtering (AMGF), providing robust preconditioning for linear systems arising in constrained optimization problems such as frictionless contact.

The reference implementation is in `miniapps/contact/`, with usage patterns demonstrated in the contact-mechanics miniapp. The miniapp `README.md` specifies the build requirement verbatim:

> AMGF requires a user-specified solver for the filtered subspace. In this miniapp a parallel sparse direct solver is used, so MFEM must be built with either MUMPS (`MFEM_USE_MUMPS=YES`) or CPardiso (`MFEM_USE_CPARDISO=YES`).

For ExaConstit we substitute Ginkgo for MUMPS/CPardiso (see §7.4 below); the rest of the AMGF wiring is unchanged.

**The verified MFEM 4.9 API surface** (confirmed from the Doxygen reference at `docs.mfem.org/html/classmfem_1_1AMGFSolver.html`):

```cpp
namespace mfem {

// In linalg/filteredsolver.hpp, included via linalg/linalg.hpp.

class FilteredSolver : public Solver {
public:
    FilteredSolver();
    void SetFilteredSubspaceTransferOperator(const Operator& P);
    void SetFilteredSubspaceSolver(Solver& S);
    void Mult(const Vector& x, Vector& y) const override;

protected:
    const Operator* A = nullptr;            // system operator, not owned
    const Operator* P = nullptr;            // transfer operator, not owned
    Solver* B = nullptr;                    // base (full-space) solver, not owned
    Solver* S = nullptr;                    // subspace solver, not owned
    std::unique_ptr<const Operator> PtAP;   // projected operator (owned)
    bool solver_set = false;
};

class AMGFSolver : public FilteredSolver {
public:
    AMGFSolver();                            // constructs with default HypreBoomerAMG
    ~AMGFSolver() override = default;

    void SetOperator(const Operator& A) override;
    void SetSolver(Solver& B) override;
    void SetFilteredSubspaceTransferOperator(const HypreParMatrix& Pop);

    HypreBoomerAMG& GetAMG();
    const HypreBoomerAMG& GetAMG() const;
};

}  // namespace mfem
```

**Key API notes:**
- The constructor takes **no `MPI_Comm` argument**; the internal `HypreBoomerAMG` handles MPI internally. (This differs from what the contact-miniapp workshop slides suggested.)
- The transfer operator is set via `SetFilteredSubspaceTransferOperator(const HypreParMatrix&)` — a **`HypreParMatrix`-typed override** on `AMGFSolver` (the base class version takes a generic `Operator&`).
- Internal AMG access is `GetAMG()`, not `.AMG()`.
- The full-space solver is set via `SetSolver(Solver& B)` (defaults to the internal `HypreBoomerAMG`).
- The subspace direct solver is set via `SetFilteredSubspaceSolver(Solver& S)` (inherited from `FilteredSolver`).

Concrete usage:

```cpp
mfem::AMGFSolver prec;
prec.GetAMG().SetSystemsOptions(3);            // 3-component displacement field
prec.GetAMG().SetRelaxType(8);                 // l1-symmetric Gauss-Seidel (match ExaConstit)
prec.GetAMG().SetStrongThreshold(0.5);         // existing tuning, not 0.90
prec.SetFilteredSubspaceSolver(subspace_solver); // Ginkgo direct solver
prec.SetFilteredSubspaceTransferOperator(P);   // HypreParMatrix
prec.SetOperator(K);                           // or K_gamma for Path D
```

The `subspace_solver` is a `Solver*` (a Ginkgo Cholesky/Lu adapter — see §7.4). The transfer operator `P` is a `HypreParMatrix*` — wide-and-tall sparse Boolean matrix as described in §3.3.

### 7.2 Building $P$ for mortar PBC

The boundary-DOF index set $\mathcal{I}_C$ is exactly the union of nonmortar-side DOF indices and mortar-side DOF indices appearing as columns of $C$. In the `MortarConstraintOperator` class, this set is constructed during constraint assembly. Concretely:

```cpp
// In src/mortar_pbc/mortar_constraint_operator.cpp,
// during BuildConstraintMatrix(...):
std::set<HYPRE_BigInt> boundary_dofs;
for (each constraint row k):
    for (each nonzero column j in row k of C):
        boundary_dofs.insert(global_col_index_for_j);

// After construction, build P as a HypreParMatrix:
std::vector<HYPRE_BigInt> idx_vec(boundary_dofs.begin(), boundary_dofs.end());
HypreParMatrix* P = BuildBooleanRestrictionPrologation(
    /*n_global_rows=*/n_u,
    /*n_global_cols=*/idx_vec.size(),
    /*nonzero_pattern=*/idx_vec  // one nonzero per column, at row idx_vec[j]
);
```

The construction is straightforward MFEM/Hypre boilerplate but requires care with parallel partitioning: $P$'s rows are distributed identically to $K$'s rows (so $P^T K P$ assembles correctly), and the columns are partitioned compactly.

### 7.3 Symmetric vs. non-symmetric K

For non-associated flow (where $K$ is non-symmetric), AMGF-PCG must be replaced by AMGF-GMRES or AMGF-BiCGStab. The existing `LinearSolverType` enum already supports both. The AMGF preconditioner itself is constructed identically; only the outer Krylov method changes.

The Petrides theorem (§3.10) bound $\kappa(M^{-1}A) \leq 2(\beta + 3)$ assumes SPD $A$. For non-symmetric $A$, the bound becomes heuristic — empirically the AMGF still works well for mildly non-symmetric operators (skew part small relative to symmetric part), but no theoretical guarantee is available.

For ExaConstit, the recommendation is: develop and benchmark with the symmetric case first; for non-associated-flow runs, switch the outer Krylov to GMRES without changing the preconditioner; verify empirically that iteration counts remain reasonable.

### 7.4 GPU/RAJA considerations and the EA/FA assembly constraint

**The EA preference and the FA requirement.** ExaConstit prefers **Element Assembly (EA)** on GPU because it is faster than Partial Assembly (PA) for the polynomial orders relevant to crystal plasticity and gives access to per-element block values useful for material-state computations. The existing solver-option validation in `src/options/option_solvers.cpp` enforces three rules consistent with this:

1. GPU runtime model is incompatible with FULL assembly (rejected with an error).
2. GPU runtime model forces the JACOBI preconditioner (auto-correct with warning).
3. EA or PA assembly forces the JACOBI preconditioner regardless of runtime model (auto-correct with warning), because BoomerAMG/ILU/L1GS/Chebyshev require a fully assembled `HypreParMatrix`.

AMGF *requires* a `HypreParMatrix` for both (a) the outer `HypreBoomerAMG` and (b) the boundary-submatrix factorization in the Ginkgo subspace solver. **AMGF therefore requires FULL assembly for the preconditioner branch of the code**, even if the residual evaluation path uses EA.

Three options to reconcile this with ExaConstit's preferred GPU residency:

**Option 1 (recommended initial path): Require FULL assembly with CPU/OpenMP runtime when AMGF is selected.** The new AMGF option in the option parser carries the constraint: `if use_amgf_path_a || use_amgf_path_d, then assembly must be FULL and rtmodel must be CPU or OPENMP`. On violation, abort with a clear error message rather than silently auto-correcting to JACOBI. This loses the EA-on-GPU benefit but guarantees correctness. For Q2 PBC problems where the saddle-stagnation is the dominant cost (10k+ Krylov iters per Newton step), even running on CPU with AMGF should be substantially faster than running on GPU with Jacobi.

**Option 2 (future work): Hybrid EA-for-residual + FA-for-preconditioner.** Keep the existing `mech_operator` using EA on GPU for residual evaluation, but additionally assemble a FULL `HypreParMatrix` on CPU (or GPU if MFEM's HypreParMatrix assembly from EA data is available) just for the preconditioner. The MFEM `ParBilinearForm::FormSystemMatrix` path is the closest existing machinery; adapting it for `ParNonlinearForm::GetGradient` requires a separate FA-flagged form alongside the EA-flagged one. This is a real implementation effort but preserves GPU residency for the dominant per-iter cost (residual evaluation). Defer until Option 1 is validated.

**Option 3 (not recommended): Assemble the boundary submatrix $P^T K P$ directly from EA element matrices without going through a full HypreParMatrix.** Possible in principle (the EA element matrices contain all the entries needed to populate the boundary submatrix), but requires custom assembly code that doesn't exist in MFEM. The boundary-submatrix is small ($n_c \ll n_u$) so the savings are not large.

The implementation guide takes Option 1. The validation step in `option_solvers.cpp::validate()` gets a new clause: when AMGF is requested, FULL assembly is required, and a clear error message instructs the user to either disable AMGF or switch to CPU/OpenMP runtime.

**Ginkgo subspace solver on host.** With FULL assembly forced by Option 1, the run is CPU-resident anyway, so the Ginkgo subspace solver naturally uses an OpenMP executor. No PCIe transfer per iteration; no GPU memory pressure. When Option 2 is implemented later, the Ginkgo executor can be switched to CUDA/HIP via the existing `gko::CudaExecutor::create(...)` API.

**Cost characterization for Option 1.** The dominant per-Newton-step costs become:
- FULL assembly of K on CPU: comparable to one GPU EA assembly in absolute cost (CPU is slower per FLOP but FA does fewer FLOPs than EA element matrices). Profile this; if it dominates, Option 2 becomes the priority.
- BoomerAMG hierarchy setup on K (or $K_\gamma$): same as the existing FA-on-CPU AMG path. No change.
- Ginkgo Cholesky factorization on $P^T K P$ or $P^T K_\gamma P$: a few hundred milliseconds at $n_c \sim 10^6$, reusable across all Krylov iterations within the Newton step.
- Per-Krylov-iteration: one BoomerAMG V-cycle (CPU) + one Ginkgo forward/back-substitute (CPU) + the BlockOperator action. Comparable to the existing Jacobi-prec-on-CPU path per iter.

The win comes from the Krylov iteration count dropping from 10k+ to 30–80, not from per-iter speedups.

### 7.5 Sketch of full Path A implementation

The existing `mortar_pbc::MortarSaddlePreconditioner` constructor signature (verified in `src/mortar_pbc/mortar_saddle_preconditioner.hpp`) is

```cpp
MortarSaddlePreconditioner(
    std::shared_ptr<mfem::Solver> K_block_prec,
    std::shared_ptr<mfem::Solver> K_jacobi_prec,
    const MortarConstraintOperator& C_op);
```

The new AMGF preconditioner mirrors this signature, adding the prolongation $P$, the Ginkgo subspace solver, and Path D toggles:

```cpp
namespace mortar_pbc {

class MortarSaddlePreconditionerAMGF : public mfem::Solver {
public:
    MortarSaddlePreconditionerAMGF(
        std::shared_ptr<mfem::Solver> K_jacobi_prec,
        const MortarConstraintOperator& C_op,
        std::unique_ptr<mfem::HypreParMatrix> P,
        std::shared_ptr<mfem::Solver> subspace_solver,
        bool use_path_d,
        double gamma_override = -1.0);

    void SetOperator(const mfem::Operator& op) override {
        // 1. Extract K from saddle BlockOperator
        const auto& block_op = dynamic_cast<const mfem::BlockOperator&>(op);
        K_ = dynamic_cast<mfem::HypreParMatrix*>(&block_op.GetBlock(0, 0));
        MFEM_VERIFY(K_, "BlockOperator (0,0) must be HypreParMatrix for AMGF");

        // 2. Refresh K_jacobi_prec for the Schur-diagonal probe path (Path A)
        K_jacobi_prec_->SetOperator(*K_);

        // 3. Path A or Path D branch
        if (use_path_d_) {
            const mfem::HypreParMatrix& CTC = C_op_.GetCTransposeC();
            gamma_ = (gamma_override_ > 0.0) ?
                gamma_override_ :
                ComputeDefaultGamma(*K_, CTC, n_lambda_, n_u_, MPI_COMM_WORLD);
            K_gamma_.reset(mfem::Add(1.0, *K_, gamma_, CTC));
            amgf_->SetOperator(*K_gamma_);
        } else {
            amgf_->SetOperator(*K_);
            schur_diag_inv_ = C_op_.ComputeInvDiagSchur(*K_jacobi_prec_);
        }

        // 4. Update inherited Solver dimensions to match the BlockOperator
        height = K_->Height() + C_op_.Height();
        width = height;
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        // x and y must be BlockVectors with the saddle (u, lambda) layout
        const auto& xb = static_cast<const mfem::BlockVector&>(x);
        auto& yb = static_cast<mfem::BlockVector&>(y);

        // (1,1): AMGF on K (Path A) or K_gamma (Path D)
        amgf_->Mult(xb.GetBlock(0), yb.GetBlock(0));

        // (2,2): Path A — diag(Schur) scaling; Path D — gamma * I
        if (use_path_d_) {
            yb.GetBlock(1) = xb.GetBlock(1);
            yb.GetBlock(1) *= gamma_;
        } else {
            const int n_lam = schur_diag_inv_.Size();
            const double* xd  = xb.GetBlock(1).HostRead();
            const double* idd = schur_diag_inv_.HostRead();
            double*       yd  = yb.GetBlock(1).HostWrite();
            for (int i = 0; i < n_lam; ++i) { yd[i] = idd[i] * xd[i]; }
        }
    }

private:
    // Shared infrastructure
    std::shared_ptr<mfem::Solver> K_jacobi_prec_;
    const MortarConstraintOperator& C_op_;
    std::unique_ptr<mfem::HypreParMatrix> P_;
    std::shared_ptr<mfem::Solver> subspace_solver_;
    std::shared_ptr<mfem::AMGFSolver> amgf_;
    mfem::HypreParMatrix* K_ = nullptr;  // not owned; from BlockOperator

    // Path A state
    mfem::Vector schur_diag_inv_;

    // Path D state
    bool use_path_d_;
    double gamma_override_;
    mutable double gamma_ = 0.0;
    std::unique_ptr<mfem::HypreParMatrix> K_gamma_;

    // Dimensions
    HYPRE_BigInt n_u_ = 0;
    HYPRE_BigInt n_lambda_ = 0;
};

}  // namespace mortar_pbc
```

The constructor body sets up the AMGF wrapper once (the AMG hierarchy is rebuilt per `SetOperator` call):

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
    amgf_ = std::make_shared<mfem::AMGFSolver>();
    // Copy the existing ExaConstit BoomerAMG configuration verbatim — do NOT retune.
    amgf_->GetAMG().SetSystemsOptions(3);
    amgf_->GetAMG().SetRelaxType(8);
    amgf_->GetAMG().SetStrongThreshold(0.5);
    amgf_->GetAMG().SetPrintLevel(0);
    amgf_->SetFilteredSubspaceTransferOperator(*P_);
    amgf_->SetFilteredSubspaceSolver(*subspace_solver_);

    n_u_      = P_->GetGlobalNumRows();
    n_lambda_ = C_op_.Height();  // local; ComputeDefaultGamma does the Allreduce
}
```

The `MortarConstraintOperator` provides the existing public methods `Mult`, `MultTranspose`, `ComputeInvDiagSchur`, `Reset`, `Height`, `Width`; the AMGF path adds `GetConstraintCoupledDofIndices()` and `GetCTransposeC()`. See the implementation guide §1.2 and §2.2 for those additions.

For Path D, the structural changes from Path A are:
1. An additional `HypreParMatrix K_gamma_` assembled as $K + \gamma C^T C$ on each `SetOperator`.
2. The AMGF is built on $K_\gamma$ rather than $K$.
3. The Schur block preconditioner is the trivial $\gamma I$ rather than the diagonal scaling.
4. The right-hand side is augmented to $r_K + \gamma C^T r_\lambda$ via a `SaddleResidualScaler` wired into `ExaTrustRegionSolver::SetScaler` (see implementation guide §2.6) — **not via direct modification in `system_driver.cpp`**.

### 7.6 Testing strategy

A phased testing plan:

**Phase T1: Verify AMGF infrastructure on a clean problem.** Take a simple periodic-elasticity problem (one grain, homogeneous material, no plasticity). Verify that Path A gives mesh-independent iteration counts as the mesh is refined uniformly. This is the unit test that AMGF itself is working correctly.

**Phase T2: Heterogeneous polycrystal benchmark.** Take a representative polycrystal RVE with 100–1000 grains, mild contrast (factor 5–10 in moduli). Verify that AMGF gives substantial iteration count reduction over the baseline. Target: at least $5\times$ reduction.

**Phase T3: Severe heterogeneity benchmark.** Take a polycrystal with porosity (factor 100+ contrast). Verify that AMGF remains robust. Compare with Path D.

**Phase T4: Plasticity benchmark.** Run a full plastic deformation history (10–100 load steps) and verify Newton + Krylov convergence is consistent across the load history.

**Phase T5: GPU benchmark.** Verify performance on a GPU-resident run with host-side subspace solve. Measure the host-device communication overhead.

---

## Section 8 — Comparison with Alternative Saddle-AMG Approaches

AMGF is one of several algebraic-multigrid-based approaches to the saddle-preconditioning problem. This section situates it among the alternatives that have appeared in the recent literature: the Wiesner–Mayr–Popp–Gee–Wall framework for saddle-preserving AMG on mortar contact problems, the static condensation approach of Duan, and the broader landscape of domain-decomposition methods. For each, the assessment focuses on the question of whether it would be a better fit than AMGF for the specific ExaConstit setting — 3D polycrystals, RVE-surface mortar constraints, existing BoomerAMG infrastructure, GPU residency — and on what conditions would shift the recommendation.

### 8.1 Wiesner, Mayr, Popp, Gee, Wall (2021)

The paper [Wiesner 2021], arXiv:1912.09056, develops algebraic multigrid methods specifically for saddle point systems arising from mortar contact formulations. The approach builds aggregation-based AMG hierarchies that *preserve the saddle-point structure* on all coarse levels — meaning the coarse-level operators are themselves saddle-point matrices, and the interpolation operators map between primary and Lagrange-multiplier spaces consistently.

The implementation is in Trilinos/MueLu. The key conceptual difference from AMGF:

- **AMGF** treats the saddle problem by first reducing to an SPD operator (the primal Schur in contact-IP, or the K block / augmented $K_\gamma$ / dual Schur in our setting) and then applying AMG with a small subspace correction. Multigrid acts on the SPD operator.
- **Wiesner et al.** preserves the saddle structure throughout the multigrid hierarchy. Multigrid acts directly on the saddle operator.

For mortar PBC, Wiesner's approach would mean: build aggregation-based AMG hierarchies for $K$ and $C$ jointly, with special aggregation rules to handle the multiplier-displacement coupling at the constraint surface.

**Trade-offs:**

| | AMGF | Wiesner et al. 2021 |
|---|---|---|
| Multigrid acts on | SPD reduced operator | Indefinite saddle operator directly |
| Coarse-level structure | Standard AMG | Saddle-preserving |
| Existing implementation | MFEM 4.9 `AMGFSolver` | Trilinos/MueLu (BACI code) |
| Maturity | New (2025) | Production (5+ years) |
| Implementation complexity | Modest (wrap existing AMG) | Heavy (full AMG rewrite for saddle structure) |
| Best for | Small constraint subspace ($n_\lambda \ll n_u$) | Large or geometrically diffuse constraint sets |

For ExaConstit's mortar PBC where $n_\lambda \sim n_u^{2/3}$ is small relative to $n_u$, AMGF is the simpler choice. For contact problems with large active sets or for problems where the constraint surface is geometrically diffuse, Wiesner's approach would have advantages.

### 8.2 Duan, An, Mo (2024) — DOFs condensation

The paper [Duan 2024], arXiv:2409.14979, develops a 2D-specific approach: exploit the tridiagonal structure of the mortar matrix in 2D to eliminate the multipliers exactly, then solve a reduced SPD problem.

The abstract states verbatim: *"In contact computation, the saddle point system makes the design of iterative methods difficult and the use of inappropriate algorithms can result in tens of thousands of iterations. We propose an algorithm based on degrees-of-freedom condensation for solving saddle point systems with general convergence."*

The approach is essentially: in 2D, the mortar matrix is tridiagonal, so $D u^+ = A^m u^-$ can be inverted explicitly (block-wise) to give $u^+$ in terms of $u^-$. Substituting into $K$ gives a reduced system on the unknowns $(u^-, u_{\text{int}})$ — the mortar side and interior DOFs.

The 2D-specific structure does not generalize cleanly to 3D, where the mortar matrix is no longer tridiagonal and the condensation is more involved. For 3D ExaConstit, this approach is not directly applicable, but the conceptual point is worth noting: when the constraint structure permits exact condensation, do it.

The Lopes 2021 paper [Lopes 2021] discusses condensation as the "Condensation Method" and reports verbatim: *"With the direct solvers employed, the SPS [Saddle Point Solution] is more efficient despite increasing the number of unknowns in the linear system of equations."* This is consistent with our recommendation to retain the saddle formulation for ExaConstit and improve its preconditioning via AMGF, rather than to condense.

### 8.3 Duan, An, Mo (2024b) — Two-level preconditioning

The companion paper [Duan 2024b], arXiv:2409.15165, develops a two-level preconditioner for the same saddle systems. The coarse level uses a physical-quantity-based coarsening (rather than purely algebraic). Less mature than AMGF; relevant primarily as a benchmark.

### 8.4 Decision matrix for choosing among approaches

For ExaConstit's mortar PBC, the choice among AMGF, Wiesner et al., and condensation depends on:

| Problem feature | Preferred approach |
|---|---|
| Small $n_\lambda/n_u$ (RVE surface, $\sim 1\text{--}5\%$) | AMGF |
| Large $n_\lambda/n_u$ (dense constraint, $\geq 10\%$) | Wiesner et al. |
| Severe polycrystal heterogeneity | AMGF (Path A) combined with Path D augmentation |
| 2D problems | Condensation (Duan 2024) |
| Strong existing AMG investment | AMGF (reuses BoomerAMG) |
| Existing MueLu infrastructure | Wiesner et al. |

For the ExaConstit setting (3D polycrystals, RVE surface constraint, existing BoomerAMG): **AMGF is the recommended approach**, in the phased manner of §9.

---

## Section 9 — Recommended Sequenced Experiments

This section converts the analysis of the preceding sections into an executable experimental plan. The five phases are ordered by implementation cost and risk: each phase builds on the data from the previous one, and each has an explicit decision criterion that determines whether to proceed, stop, or branch to an alternative. The phases are designed to attribute any iteration-count improvement (or regression) to a specific change, rather than testing combinations whose individual effects cannot be untangled. Phase 1 establishes the baseline benefit of Path A; Phase 2 layers in the augmented-Lagrangian structure of Path D only if needed; Phase 3 stress-tests the Path A (+ Path D) configuration on production problems; Phase 4 adds Path B only if production benchmarks reveal multiplier-block conditioning as a remaining bottleneck; Phase 5 explores non-AMGF alternatives only if everything else proves inadequate.

### Phase 1 — Path A: AMGF on the K block (Week 1)

**Goal:** Wrap the existing BoomerAMG in an `AMGFSolver`, with $P$ targeting RVE-boundary displacement DOFs. Measure the iteration count reduction.

**Tasks:**
1. Build MFEM 4.9 with `MFEM_USE_MUMPS=YES`.
2. Implement `BuildBooleanRestrictionPrologation` utility for constructing $P$ as a `HypreParMatrix`.
3. Extract boundary-adjacent DOF indices from `MortarConstraintOperator` and build $P$.
4. Modify `MortarSaddlePreconditioner::SetOperator` to construct `AMGFSolver` around the existing BoomerAMG.
5. Replace `K_block_prec_->Mult(...)` with `amgf_K_->Mult(...)` in the application phase.

**Diagnostic measurements:**
- Outer Krylov iteration count vs. mesh refinement (baseline vs. Path A).
- Per-iteration cost breakdown (AMG V-cycle + subspace solve + matvecs).
- Total wall-clock time per Newton step.
- Memory overhead from MUMPS factor.

**Decision criterion:** If Path A reduces iteration count by $\geq 2\times$ on a representative polycrystal RVE without significantly increasing per-iteration cost, adopt it. If not, the bottleneck is in the Schur block rather than the K block, and proceed to Phase 2 (Path D).

### Phase 2 — Path D: Augmented Lagrangian (Weeks 2–3)

**Goal:** Implement the augmented Lagrangian formulation, with AMGF on $K_\gamma = K + \gamma C^T C$.

**Tasks:**
1. Implement matrix-free $C^T C$ operator (or assemble explicitly via `HypreParMatrix::Mult`).
2. Add assembly of $K_\gamma$ in the saddle preconditioner setup.
3. Modify the right-hand side to $r_K^{\text{aug}} = r_K + \gamma C^T r_\lambda$.
4. Replace the Schur block preconditioner with the trivial $\gamma I$.
5. Implement $\gamma$ selection: default to $\gamma = \text{tr}(K)/\text{tr}(C^T C) \cdot n_\lambda / n_u$.

**Diagnostic measurements:**
- Iteration count: Path A vs. Path D.
- Sensitivity to $\gamma$: sweep $\gamma \in \{0.1, 1, 10, 100\} \cdot \gamma_{\text{default}}$.
- Newton convergence: does the augmented form change the outer Newton iteration count?
- Residual imbalance: is $\|r_K^{\text{aug}}\| \sim \|r_\lambda^{\text{aug}}\|$ throughout the iteration?

**Decision criterion:** Path D should be *robust* to factor-of-10 changes in $\gamma$ (iteration count should change by at most $2\times$). If it isn't, the AMG quality on $K_\gamma$ is the bottleneck — investigate alternative AMG configurations (Phase 5) before proceeding. If Path D is robust and the iteration counts are acceptable, move to Phase 3 (production benchmarks).

### Phase 3 — Production benchmarks and edge cases (Month 2)

**Goal:** Comprehensive validation across the full range of ExaConstit production problems.

**Tasks:**
1. Run a representative production-scale polycrystal study (1000+ grains, $10^7$+ DOFs, full plastic deformation history).
2. Run a non-associated-flow case (non-symmetric K) with GMRES outer Krylov.
3. Run GPU-resident benchmarks; profile host-device communication overhead for the subspace solve.
4. Test sub-XYZ periodicity cases (constraint applied on only some axis pairs).

**Diagnostic measurements:**
- Total simulation wall-clock time: baseline vs. final (Path A alone or Path A + Path D combined).
- Memory usage at peak (for both production scale and stress-test cases).
- Scalability: weak and strong scaling on representative HPC systems.

**Decision criterion:** With Phases 1 and 2 implemented and validated, the next decision is whether the combined approach handles the full production envelope. Failure modes that warrant escalation: residual stagnation specifically attributable to corner-modified multiplier conditioning (which Paths A and D do not directly address) → Phase 4 (Path B) is the targeted response; or large iteration counts on problems where the bulk AMG quality cannot be improved further → Phase 5 (Alternative AMG) is the broader response. If Phase 3 reveals acceptable wall-clock performance across the production envelope, neither escalation is needed.

### Phase 4 — Path B: AMGF on the dual Schur (conditional, Month 2+)

**Goal:** If Phases 1–3 reveal that the multiplier-block conditioning remains a bottleneck — specifically when corner-modified Wohlmuth multipliers create localized ill-conditioning that Path D's $\gamma I$ Schur preconditioner does not adequately mask — implement Path B as a targeted augmentation.

**Tasks:**
1. Extend `MortarConstraintOperator` with an API (e.g. `GetCornerModifiedRows()`) that exposes which multiplier rows are corner-modified.
2. Build the multiplier-subspace prolongation matrix $P \in \mathbb{R}^{n_\lambda \times n_c^\lambda}$ restricting to those rows (plus any rows adjacent to high-contrast material interfaces, as a problem-dependent heuristic).
3. Implement the $K^{-1}$-surrogate for the subspace operator (§3.6): the recommended starting point is option (ii) — per-column BoomerAMG V-cycles to construct $Z^T V_{\text{AMG}}(K) Z$ — with option (iii) (loose-tolerance PCG) as a fallback if accuracy is insufficient.
4. Replace the diagonal lumped Schur preconditioner with the AMGF-on-$\hat{S}$ preconditioner inside the saddle preconditioner structure.

**Diagnostic measurements:**
- Iteration count on problems known to stress corner-modified multipliers (e.g. cubic RVEs with sharp Wohlmuth modifications at vertices).
- Sensitivity of iteration count to the choice of multiplier-subspace cardinality $n_c^\lambda$.
- The $K^{-1}$-surrogate error: compare $Z^T V_{\text{AMG}}(K) Z$ against a small problem where $Z^T K^{-1} Z$ can be computed exactly.

**Decision criterion:** Path B should reduce iteration counts on its targeted problem class by $\geq 2\times$ compared to Paths A+D combined. Otherwise, the multiplier-conditioning issue is not the actual bottleneck and Phase 5 alternatives should be considered instead.

### Phase 5 — Alternative AMG (only if needed, Month 2+)

**Goal:** If Phases 1–4 leave large iteration counts on the most extreme problems (e.g. RVEs with porosity and 100×+ contrast), evaluate alternatives.

**Candidate alternatives:**
1. `PCBDDC` from PETSc, with custom subdomain decomposition.
2. MueLu (Trilinos) smoothed-aggregation with explicit RBMs and adaptive coarsening.
3. GenEO via FreeFem++ interface (research-grade, not production).

These are major dependencies and code restructurings; only undertake if the AMGF-based approach is provably inadequate.

---

## Section 10 — Caveats and Open Questions

The preceding analysis establishes a clean structural case for AMGF and a phased implementation plan. This section enumerates the assumptions, limitations, and unresolved questions that the analysis depends on or leaves open. Several of these are routine engineering caveats (mesh-refinement behavior, parameter sensitivities); others are more substantive and would warrant separate investigation if they prove material in practice (the interaction between Wohlmuth corner modifications and the AMGF subspace; the absence of a Petrides bound for the inexact-$K^{-1}$ Path B surrogate; the behavior of the augmented operator on non-associated-flow problems where $K$ itself is non-symmetric). Surfacing these explicitly is part of presenting an honest analysis rather than an oversold one.

### 10.1 The Wohlmuth corner modifications and the AMGF subspace

The dual basis requires corner modifications to retain partition of unity [Wohlmuth 2001]. These modifications break the strict biorthogonality at corner DOFs, with the consequence that the affected multiplier rows in $C$ have different structure (and different conditioning) than interior face multipliers.

The AMGF subspace $P$ should include these corner-modified DOFs explicitly. For Path A (P on displacements), corner displacement DOFs are already included in the boundary-adjacent set. For Path B (P on multipliers), corner multiplier rows must be identified and included. The `MortarConstraintOperator` class knows which rows are corner-modified — exposing this information via a new API (e.g. `GetCornerModifiedRows()`) would support a more targeted Path B implementation.

**Linearly dependent rows of $C$ and rank-deficient $C^T C$.** A subtler issue at corners is that the corner modifications can produce *linearly dependent* rows of $C$, not just badly-scaled ones. Two consequences for the analysis:

1. The "every constraint-violating mode is penalized by $\gamma$" interpretation of $K_\gamma = K + \gamma C^T C$ (§5.7) needs qualification: if $C$ has rank $< n_\lambda$, then $C^T C$ has a kernel of dimension $n_\lambda - \text{rank}(C)$, and the augmentation only penalizes modes in $\text{range}(C^T)$. Constraint modes in the cokernel of $C^T C$ contribute nothing. In practice this is rarely a problem because the existing `MortarConstraintOperator` and `MortarPbcManager` already drop the redundant corner rows via the `m_corner_ess_tdofs` mechanism, so the $C$ passed to the saddle preconditioner is full-rank by construction.

2. The natural-scaling formula $\gamma_{\text{default}} = \text{tr}(K)/\text{tr}(C^T C) \cdot (n_\lambda/n_u)$ degenerates if $\text{tr}(C^T C) = 0$. The implementation guide's `ComputeDefaultGamma` includes a fallback (`MFEM_WARNING` and $\gamma = 1$) for this case, but in production it indicates an upstream constraint-builder bug — the operator should not be passed to AMGF with an all-zero $C$.

A defensive validation step in the new preconditioner's constructor should check `C_op.GetCTransposeC().Norm() > 0` and abort with a clear error message if not.

### 10.2 Multiplier scaling for sub-XYZ periodicity

ExaConstit's code supports sub-XYZ periodic BCs (introduced in the ExaConstit 5.9 development cycle), where periodicity is enforced on a subset of axis pairs. In this case, the constraint matrix $C$ has rows only for the active axis pairs. The Path A subspace $P$ should include only DOFs in the active periodicity faces; the Path D augmentation $\gamma C^T C$ is naturally restricted to those faces.

No special handling is required at the AMGF level, but the boundary-DOF identification step (§7.2) must use the current axis-pair specification. This is straightforward but requires care during the implementation.

### 10.3 GPU residency and the EA-vs-FA tradeoff

§7.4 documented the resolution: the AMGF preconditioner branch requires FULL assembly, which the existing `option_solvers.cpp` validation forbids on GPU. The initial implementation forces CPU/OpenMP runtime when AMGF is enabled (Option 1 of §7.4). This is acceptable because:

- For Q2 PBC problems where Krylov stagnates at 10k+ iterations per Newton step, even a CPU AMGF run completing in 30–80 iterations is dramatically faster than GPU+Jacobi.
- The Ginkgo subspace solver naturally runs on the same OpenMP executor in this regime, with no PCIe traffic per iteration.

The future Option 2 (hybrid EA-for-residual + FA-for-preconditioner on GPU) requires MFEM machinery that does not currently exist in MFEM 4.9: assembling a `HypreParMatrix` from `ParNonlinearForm` element matrices when the form is configured with EA assembly. The `ParBilinearForm::FormSystemMatrix` path is the closest existing analog; adapting it for `ParNonlinearForm::GetGradient(x)` is feasible but is its own engineering project. The implementation guide flags Option 2 as future work and proceeds with Option 1.

If Option 1's CPU FA assembly cost dominates Newton-step wall-clock in profiling (unlikely for the Q2 problems where Krylov is the bottleneck, but worth verifying), Option 2 moves up the priority list.

### 10.4 Choice of $P$: displacements vs. multipliers

All of §2 treats $P$ as restricting to displacement DOFs adjacent to $\Gamma$. An alternative is to put a subspace correction on the multiplier block (Path B). The two are not equivalent; the displacement-side $P$ matches [Petrides 2025] most closely.

A third option, related to constraint preconditioning [Keller 2000], is to use an oblique projection onto $\ker(C)$ rather than a Boolean subspace. This is conceptually more elegant but operationally harder; we don't pursue it here but note it as a potential research direction.

### 10.5 Non-associated flow and the SPD assumption

The Petrides theorem (§3.10) assumes SPD $A$. For non-associated plastic flow, $K$ has a non-symmetric component, and the theorem bound is no longer rigorous. Empirically, AMGF works well for mildly non-symmetric matrices (skew part bounded by, say, $10\%$ of the symmetric part), but breaks down for strongly non-symmetric problems.

For ExaConstit's non-associated flow cases, the recommendation is to use the symmetrized AMGF as a right preconditioner for GMRES on the full operator. The skew-symmetric corrections enter only through the outer GMRES; the AMGF preconditioner sees only the symmetric part.

### 10.6 Comparison with hard-coded condensation

For 2D problems, [Duan 2024] shows that exact condensation outperforms saddle-form solution. ExaConstit is 3D, so this doesn't directly apply, but the conceptual point is worth retaining: if the constraint structure permits cheap exact condensation, it's worth considering. For dual-basis mortar in 3D, the corner modifications and the higher-dimensional surface geometry make condensation impractical, so the saddle approach (with AMGF) is the right path.

### 10.7 Outer Newton convergence

All of this work focuses on the inner linear solver. The outer Newton iteration is governed by the residual norm, which (per §6) should be normalized to give balanced contributions from $r_K$ and $r_\lambda$. The augmented Lagrangian Path D does this automatically; Paths A and B require explicit residual scaling.

For non-trivial plasticity problems, the outer Newton may converge slowly or fail to converge regardless of the inner solver. This is a separate problem from preconditioning and is addressed by line-search modifications, trust-region methods (already implemented in `ExaTrustRegionSolver`), or adaptive load stepping. AMGF improves the inner linear solver; it does not address outer Newton convergence directly.

---

## Section 11 — Bibliography

### Academic references

[Baker 2010] Baker, A. H., Kolev, T. V., and Yang, U. M. *Improving algebraic multigrid interpolation operators for linear elasticity problems.* Numerical Linear Algebra with Applications, 17(2–3):495–517, 2010. LLNL-JRNL-412928. DOI: 10.1002/nla.688.

[Benzi 2005] Benzi, M., Golub, G. H., and Liesen, J. *Numerical solution of saddle point problems.* Acta Numerica, 14:1–137, 2005. DOI: 10.1017/S0962492904000212.

[Brezina 2005] Brezina, M., Falgout, R., MacLachlan, S., Manteuffel, T., McCormick, S., and Ruge, J. *Adaptive Smoothed Aggregation (αSA) Multigrid.* SIAM Review, 47(2):317–346, 2005. DOI: 10.1137/050626272.

[Duan 2024] Duan, X., An, H., and Mo, Z. *A DOFs condensation based algorithm for solving saddle point systems in contact computation.* arXiv:2409.14979, 2024.

[Duan 2024b] Duan, X., An, H., and Mo, Z. *Two-Level preconditioning method for solving saddle point systems in contact computation.* arXiv:2409.15165, 2024.

[Farhat 2001] Farhat, C., Lesoinne, M., LeTallec, P., Pierson, K., and Rixen, D. *FETI-DP: a dual-primal unified FETI method—part I.* International Journal for Numerical Methods in Engineering, 50(7):1523–1544, 2001.

[Fortin 1983] Fortin, M. and Glowinski, R. *Augmented Lagrangian Methods: Applications to the Numerical Solution of Boundary-Value Problems.* Studies in Mathematics and its Applications, vol. 15. North-Holland, Amsterdam, 1983. ISBN 0444866809.

[Henson 2002] Henson, V. E. and Yang, U. M. *BoomerAMG: A parallel algebraic multigrid solver and preconditioner.* Applied Numerical Mathematics, 41(1):155–177, 2002. DOI: 10.1016/S0168-9274(01)00115-5.

[Hestenes 1969] Hestenes, M. R. *Multiplier and gradient methods.* Journal of Optimization Theory and Applications, 4(5):303–320, 1969.

[Keller 2000] Keller, C., Gould, N. I. M., and Wathen, A. J. *Constraint preconditioning for indefinite linear systems.* SIAM Journal on Matrix Analysis and Applications, 21(4):1300–1317, 2000.

[Lopes 2021] Rodrigues Lopes, I. A., Ferreira, B. P., and Andrade Pires, F. M. *On the efficient enforcement of uniform traction and mortar periodic boundary conditions in computational homogenisation.* Computer Methods in Applied Mechanics and Engineering, 384:113930, 2021. DOI: 10.1016/j.cma.2021.113930.

[Mandel 1993] Mandel, J. *Balancing domain decomposition.* Communications in Numerical Methods in Engineering, 9(3):233–241, 1993.

[Murphy 2000] Murphy, M. F., Golub, G. H., and Wathen, A. J. *A note on preconditioning for indefinite linear systems.* SIAM Journal on Scientific Computing, 21(6):1969–1972, 2000.

[Petrides 2025] Petrides, S., Hartland, T., Kolev, T., Lee, C. S., Puso, M., Solberg, J., Chin, E. B., Wang, J., and Petra, C. *AMG with Filtering: An Efficient Preconditioner for Interior Point Methods in Large-Scale Contact Mechanics Optimization.* arXiv:2505.18576 v2, April 2026. URL: https://arxiv.org/abs/2505.18576

[Petrides workshop 2025] Petrides, S. et al. *AMG with Filtering – An Efficient Preconditioner for Large-Scale Contact Mechanics Interior-Point Optimization.* MFEM Community Workshop, Portland State University, September 10–11, 2025. LLNL-CFPRES-2010794. Slides at mfem.org.

[Powell 1969] Powell, M. J. D. *A method for nonlinear constraints in minimization problems.* In R. Fletcher, ed., *Optimization*, pp. 283–298. Academic Press, 1969.

[Reis 2014] Reis, F. J. P. and Andrade Pires, F. M. *A mortar based approach for the enforcement of periodic boundary conditions on arbitrarily generated meshes.* Computer Methods in Applied Mechanics and Engineering, 274:168–191, 2014.

[Saad 2003] Saad, Y. *Iterative Methods for Sparse Linear Systems.* 2nd edition. SIAM, Philadelphia, 2003.

[Spillane 2014] Spillane, N., Dolean, V., Hauret, P., Nataf, F., Pechstein, C., and Scheichl, R. *Abstract robust coarse spaces for systems of PDEs via generalized eigenproblems in the overlaps.* Numerische Mathematik, 126(4):741–770, 2014.

[Trefethen 1997] Trefethen, L. N. and Bau, D. *Numerical Linear Algebra.* SIAM, Philadelphia, 1997.

[Wiesner 2021] Wiesner, T. A., Mayr, M., Popp, A., Gee, M. W., and Wall, W. A. *Algebraic multigrid methods for saddle point systems arising from mortar contact formulations.* International Journal for Numerical Methods in Engineering, 122(15):3749–3779, 2021. arXiv:1912.09056. DOI: 10.1002/nme.6680.

[Wohlmuth 2000] Wohlmuth, B. I. *A mortar finite element method using dual spaces for the Lagrange multiplier.* SIAM Journal on Numerical Analysis, 38(3):989–1012, 2000. DOI: 10.1137/S0036142999350929.

[Wohlmuth 2001] Wohlmuth, B. I. *Discretization Methods and Iterative Solvers Based on Domain Decomposition.* Lecture Notes in Computational Science and Engineering, vol. 17. Springer, Berlin, 2001.

[Xu 2002] Xu, J. and Zikatanov, L. *The method of alternating projections and the method of subspace corrections in Hilbert space.* Journal of the American Mathematical Society, 15(3):573–597, 2002.

### Practical references

- MFEM 4.9 library, released December 11, 2025. Source: github.com/mfem/mfem. The `AMGFSolver` class is in `mfem/linalg/`; the reference implementation is in `miniapps/contact/`. Build requirement: `MFEM_USE_MUMPS=YES` or `MFEM_USE_CPARDISO=YES` for the filtered subspace direct solver.

- Hypre / BoomerAMG documentation: hypre.readthedocs.io. The BoomerAMG section covers all coarsening types, smoother options, and elasticity-specific configurations.

- PETSc manual: petsc.org. Particularly relevant are `PCFieldSplit` (block-structured preconditioning), `PCHYPRE` (Hypre wrapper), and `PCBDDC` (BDDC). For saddle-point preconditioning, see the PETSc manual sections on `KSPMINRES` and `KSPGMRES`.

- Trilinos / MueLu user guide: trilinos.github.io/muelu.html. Implements the Wiesner et al. (2021) saddle-AMG approach.

- ExaConstit source: github.com/LLNL/ExaConstit. The AMG configuration discussed in §4 is in `src/system_driver.cpp`; the mortar PBC implementation discussed throughout is in `src/mortar_pbc/`.

- AMGCL near-null-space tutorial: amgcl.readthedocs.io/en/latest/tutorial/Nullspace.html. Provides a worked example of rigid-body-mode injection and its memory/performance trade-offs.

- AMG with Filtering, MFEM 4.9 release notes: github.com/mfem/mfem/blob/master/CHANGELOG, entry under "v4.9" describing `FilteredSolver` and `AMGFSolver`.

- Petrides workshop slides: mfem.org/pdf/workshop25/29_Petrides_AMG_Filtering.pdf. The most accessible introduction to the AMGF implementation, with concrete code examples.

- Stanford CME 338 lecture notes (Saunders): stanford.edu/class/cme338/notes/notes11-BCL.pdf. The cleanest practical treatment of augmented Lagrangian methods (BCL — Bound-Constrained Lagrangian), including the outer multiplier update and adaptive $\gamma$ strategies.

---

*End of document.*
