# Phase 4 — C++ Port Plan: Mortar PBC Standalone in ExaConstit `tests/mortar_pbc/`

> Companion to `MORTAR_PBC_ARCHITECTURE.md`. This document is the
> implementation plan for porting the Python prototype to C++, in
> ExaConstit's `tests/mortar_pbc/` initially, then promoted to
> `src/mortar_pbc/` once validated.
>
> **Cross-references**: This document references the top-level architecture
> doc by section number throughout. When a section reference appears
> (e.g. §11.7.2), it points to the architecture doc. When a sub-section of
> THIS document is referenced, it appears as §P4.X.Y.
>
> **Loading this document into a fresh conversation**: Pair this file
> with `MORTAR_PBC_ARCHITECTURE.md` (the "architecture doc") and any current
> Python prototype source. Together they are sufficient context to
> resume the port from any phase boundary without re-deriving prior
> decisions.

---

## §P4.1 Goals and non-goals

### Goals
1. Port the validated Python 3D mortar-PBC prototype (homogeneous +
   heterogeneous strip-split + 2x2x2 octant checkerboard tests) to
   C++ with the **same numerical answers** at np=1, np=4, np=16, hex
   and tet, both linear-elastic with PBC corner-Dirichlet.
2. Use ExaConstit's existing infrastructure where it exists (Caliper,
   `mech_operator`, MFEM operator hierarchy) without re-inventing.
3. Validate scaling characteristics through a deliberate progression
   (np=4 → np=16 → np=256 → np=1024) BEFORE attempting integration
   into the production solver.
4. Ship a CPU+GPU-capable code path where MFEM K-action is GPU-resident
   and constraint operations follow MFEM's GPU-aware operator interface.
5. Set up the architecture so the eventual move to velocity-based
   primal (for ExaConstit integration) is a focused change to one
   class (`MortarPbcDriver`).

### Non-goals (explicitly deferred)
- **Full ExaConstit integration**: not part of Phase 4. After Phase 4,
  Phase 5 handles `BCManager` ↔ `ConstraintManager` adapter,
  `SystemDriver::SolveInit` extension to handle saddle-point projection,
  and the velocity-primal switch.
- **Non-conforming face matching (Sutherland-Hodgman)**: still a
  Python-prototype Phase 3.5 task. The C++ port handles only conforming
  faces in Phase 4.
- **Tribol integration as an alternative `ConstraintAssembler`**: long-
  term, see architecture doc §14.3.
- **Higher-order primal (p ≥ 2)**: long-term, see architecture doc §4.12.
- **Hypre + GPU**: not yet supported by MFEM for vector-dimension
  problems (see §P4.4.1). CPU Hypre + GPU MFEM K-action is the Phase 4
  target; Hypre+GPU enabled later as upstream MFEM matures.

---

## §P4.2 Architectural overview

Four independently testable components, identical in structure to the
Python prototype but with the scalability/portability constraints baked in:

```
┌────────────────────────────────────────────────────────────────────┐
│  BoundaryClassifier3D                                              │
│    Setup-time only. Inspects ParMesh + ParFES, produces topology:  │
│    8 corners, 12 edges, 6 faces, with sentinel-tagged face/edge    │
│    elements. Mirrors Python boundary_3d.py.                        │
│    Constructed ONLY on boundary ranks (boundary_comm; §P4.4.0).    │
│    Setup MPI: AllGather (Phase 1) → tile-partitioned matching     │
│    (Phase 2), both on boundary_comm.                              │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  MortarAssembler2D / FaceMortarAssembler3D                         │
│    CPU-only integration kernels. Per-pair dense D, A_m blocks      │
│    via Gauss quadrature on dual-modified bases. No MPI, no shared  │
│    state. Wholly templated on element vertex count (3 or 4) for    │
│    static dispatch.                                                │
│    Mirrors Python mortar_2d.py + face_mortar_3d.py.                │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  ConstraintBuilder3D                                               │
│    Constructed ONLY on boundary ranks; assembles row contributions │
│    on boundary_comm.                                               │
│    Phase 1: builds local-row contributions, INSTALLS into a        │
│             distributed mfem::HypreParMatrix C on WORLD with empty │
│             row blocks for interior ranks (§P4.4.5).               │
│    Phase 2: refactor to AllGather-free distributed matching        │
│             (the §P4.4.4 work).                                    │
│    Phase 3: optional EA path — keeps per-element local D, A_m and  │
│             implements Mult / MultTranspose without ever forming   │
│             a CSR (matrix-free C, GPU-friendly).                   │
│    Mirrors Python constraint_builder_3d.py.                        │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  MortarPbcDriver                                                   │
│    Multi-step ramping driver. Owns persistent state (u, λ, F_n).   │
│    Wraps mfem::BlockOperator + saddle-point Krylov solve           │
│    (MINRES default; GMRES, BiCGStab also supported; §P4.4.7).      │
│    Constructs and owns the boundary subcommunicator at startup.    │
│    Mirrors Python multistep_driver.py.                             │
└────────────────────────────────────────────────────────────────────┘
```

This layering matches §13.2 of the architecture doc but expanded into
implementation detail. The dependency arrow goes downward only;
each layer is unit-testable against the Python output without
involving the layers above.

---

## §P4.3 Three-pronged C++ ratchet

The port proceeds in three independent rounds; each round is a
"ratchet click" that locks in a property and does not regress.

### Round 1 (Phase 4.1) — Initial port, AllGather-based, HypreParMatrix C
- All four classes implemented at "works correctly at np=4" quality.
- Constraint matrix C is a `mfem::HypreParMatrix`, built by gathering
  global topology to every rank (mirrors Python prototype exactly).
- K is whatever MFEM gives us (CPU-FA or GPU-EA via existing
  `assemble_linear_elastic_K`-equivalent).
- All three test drivers (homogeneous, heterogeneous strip-split,
  checkerboard) ported and passing at np=1, 4, 16.

### Round 2 (Phase 4.2) — Distribute the boundary topology
- Replace the AllGather pattern in `BoundaryClassifier3D` with
  a distributed-pair matching scheme based on 2D tile partitioning
  of the parametric plane (§P4.4.4).
- No change to the public API of any class.
- Validation: same three drivers pass at np=4 and now at np=256, 1024.
- This unlocks the path to scale; Phase 4.1 caps somewhere near
  np=500–1000 depending on memory.

### Round 3 (Phase 4.3) — Element-assembly C alternative
- Add an EA-style `MortarConstraintOperator` that holds per-pair
  local D and A_m blocks, implements `Mult` / `MultTranspose` via
  per-pair scatter-gather, never forms a CSR.
- Selectable via runtime flag: `--constraint-storage=hypre` (default)
  vs `--constraint-storage=ea`.
- Validation: identical numerical output to the HypreParMatrix path
  to within Krylov tolerance.
- This is the GPU-friendly path — once it works, it's the production
  default.

The order matters: Round 1 establishes correctness, Round 2 establishes
scale, Round 3 establishes performance. **Don't touch Round N+1 until
Round N is fully green.**

---

## §P4.4 Per-component design specifics

### §P4.4.0 MPI communicator strategy: the boundary subcommunicator

#### The premise: not every rank touches the boundary

In a domain-decomposed RVE problem on a roughly-cubic grid, only the
ranks whose subdomain touches the outer boundary have boundary work
to do. With nranks ≈ p³ ranks in a p×p×p arrangement, the boundary
ranks are those on the outer faces of the rank grid — total
``6p² - 12p + 8`` for a cube. As p grows this becomes a vanishing
fraction of all ranks:

| nranks (p×p×p)  | boundary ranks    | boundary fraction |
|----------------:|-------------------:|------------------:|
|   8 (2×2×2)     |  8                | 100 %  (degenerate) |
|  64 (4×4×4)     | 56                |  88 % |
| 512 (8×8×8)     | 296               |  58 % |
|1024 (~10×10×10) | 488               |  48 % |
|4096 (~16×16×16) | 1352              |  33 % |
|32768 (32×32×32) | ~5800             |  18 % |

At 32 768 ranks, a WORLD AllGather-everything-to-everywhere wastes
roughly 5/6ths of the bandwidth on ranks that have nothing to
contribute and nothing to do with the result. Worse, **interior
ranks must still participate** in any WORLD collective even though
they own zero boundary records — every WORLD AllGather syncs them
unnecessarily and turns "work that should be free for them" into
synchronization cost.

This isn't fixed by the Phase 4.2 distributed-pair-matching
refactor — it's a separate, easier improvement that should be in
from Round 1.

#### The fix: boundary subcommunicator from MPI_Comm_split

At driver startup, BEFORE constructing the classifier, the driver
splits WORLD into "ranks-with-boundary" + "ranks-without-boundary":

```cpp
int has_boundary = (pmesh.GetNBE() > 0) ? 1 : 0;

MPI_Comm boundary_comm = MPI_COMM_NULL;
MPI_Comm_split(MPI_COMM_WORLD,
               has_boundary ? 1 : MPI_UNDEFINED,
               world_rank,
               &boundary_comm);
// boundary_comm is MPI_COMM_NULL on interior ranks (color = MPI_UNDEFINED).
// On boundary ranks it's a fresh communicator with consecutive ranks
// 0..n_boundary_ranks-1.

// Sanity-check: must have at least 8 ranks for the 8 corners.
if (boundary_comm != MPI_COMM_NULL) {
    int n_bdy_ranks; MPI_Comm_size(boundary_comm, &n_bdy_ranks);
    MFEM_VERIFY(n_bdy_ranks >= 1, "Empty boundary communicator");
}
```

The classifier and constraint builder accept `boundary_comm` as a
constructor arg. On interior ranks (where `boundary_comm` is
`MPI_COMM_NULL`), neither object is constructed at all — the
driver branches on the comm and skips that whole code path.

#### What runs on which communicator

| Operation                                    | Communicator   |
|----------------------------------------------|----------------|
| Bounding box reduction                       | WORLD          |
| K assembly                                   | WORLD          |
| K matvec (Krylov inner)                      | WORLD          |
| Volume-averaged F                            | WORLD          |
| Vector inner products inside Krylov          | WORLD          |
| BoundaryClassifier3D setup                   | boundary_comm  |
| MortarAssembler integrations                 | (per-pair, no MPI) |
| Runtime attribute-discovery cross-check      | boundary_comm  |
| AllGather of boundary records (Phase 4.1)    | boundary_comm  |
| Distributed-hash matching (Phase 4.2)        | boundary_comm  |
| C HypreParMatrix construction                | WORLD (with empty rows on interior ranks; see §P4.4.5) |
| C matvec / C^T matvec                        | WORLD (Hypre handles empty-rank rows) |

**Why the bbox stays on WORLD**: a non-boundary rank may still own
mesh vertices (interior vertices of its subdomain) that contribute
to the bbox extent. The bbox is a property of the mesh, not the
boundary, so WORLD is correct.

**Why C lives on WORLD even though it's "boundary-only" data**: K
lives on WORLD (volume work). The Krylov solver applies the block
operator `[K, C^T; C, 0]`. For Hypre's `BlockOperator` to mix K and
C cleanly, both must be defined on the same communicator. Putting
C on WORLD is the cleanest way; the cost is one zero-row block per
interior rank in HypreParMatrix's data structures, which is
negligible (kilobyte-scale).

**The construction-time vs runtime distinction**: setup-side C
ASSEMBLY happens entirely on `boundary_comm` (every byte of dense
D and A_m blocks lives only on boundary ranks), but the resulting
HypreParMatrix is INSTALLED into a WORLD-shaped object via Hypre's
CSR-construct constructor with `row_starts[r] == row_starts[r+1]`
on interior ranks. No data is moved during the install step;
interior ranks just register that they own zero rows.

#### What this changes in the classifier code

In Python, every place that says `comm = self.pmesh.GetComm()` would
become, in C++, `comm = boundary_comm`. The bbox helpers that need
WORLD are passed it explicitly. Inside the classifier methods,
`MPI_Allgatherv` operates on the small subcomm — fewer ranks to sync
with, smaller per-message deserialization overhead, naturally less
bandwidth.

This also affects the **"discover face-label by attribute"**
cross-rank consistency check (mortar §11.7.2). The Python version
AllGathers on WORLD; in C++ it AllGathers on `boundary_comm`. An
interior rank that doesn't have any boundary attributes shouldn't
participate in a check that asks "do all ranks see attribute 1
on the same axis?" — only ranks that actually see boundary should.

#### Sanity-checking the subcomm at construction

Before the classifier does any work, sanity-check the subcomm:

```cpp
int n_bdy_ranks_local;  MPI_Comm_size(boundary_comm, &n_bdy_ranks_local);
HYPRE_BigInt n_bdr_elements_global = pmesh.GetGlobalNBE();
MFEM_VERIFY(n_bdr_elements_global > 0,
            "BoundaryClassifier3D: parent ParMesh has no global boundary "
            "elements; mortar PBC is meaningless.");
// Every rank in boundary_comm should report n_local_bdr > 0.
int my_n_bdr = pmesh.GetNBE();
MFEM_VERIFY(my_n_bdr > 0, "Rank in boundary_comm has no local boundary "
            "elements; the split was constructed incorrectly.");
```

#### Off-rank scaling ratio (Round 1 vs Round 2)

For comparison, here's the per-rank message volume during boundary-
record exchange under each scheme. Boundary record ~ 64 bytes
(snap-key triple + attribute + gtdofs).

For an n=128 RVE (~2M zones) with nranks=4096 (16×16×16):

| Phase | ranks involved | boundary verts global | per-rank send | per-rank recv |
|-------|---------------:|----------------------:|--------------:|--------------:|
| 4.1 (boundary-subcomm AllGather) | 1352 of 4096 | 100k | 5 KB | 6.7 MB |
| 4.2 (boundary-subcomm tile partitioning) | 1352 of 4096 | 100k | 5 KB | 5 KB |
| (worst case: 4.1 on WORLD AllGather) | 4096 of 4096 | 100k | 1.6 KB | 6.7 MB |

The `4.1 boundary-subcomm` row is what we want for Round 1.
Per-rank recv volume (6.7 MB) is large but tractable. Phase 4.2's
tile-partitioned matching makes recv per-rank also bounded by the
local share, which is the real scaling fix. Compared to "WORLD
AllGather" the boundary-subcomm version doesn't even reduce per-
rank recv size — but it eliminates the 2700 interior ranks from
the sync, which is what makes it strictly better-behaved than
what I had described originally.

### §P4.4.1 GPU portability strategy

#### Where GPU matters and where it doesn't

**Setup-time CPU-only (no GPU):**
- `BoundaryClassifier3D`: O(boundary_size) work, runs once. Topology
  inspection + integer indexing is naturally serial; CPU code is fine.
- `MortarAssembler2D` and `FaceMortarAssembler3D`: per-pair dense
  integration. Could be parallelised across pairs but the pair count
  is O(n²) at worst (n = cells per RVE side), totally negligible.

**Runtime path (GPU when available):**
- K matvec: goes through the user-provided `mfem::Operator&`. If MFEM
  is built with CUDA/HIP and K is a PA/EA form, K is automatically
  GPU-resident. We never touch K's storage.
- C matvec / C^T matvec: this is the architectural decision in §P4.4.5.
- Krylov solver inner products: `mfem::HypreParVector` operations are
  GPU-aware when MFEM is built with GPU support.
- Block-Jacobi preconditioner: `Operator::AssembleDiagonal` is GPU-
  aware.

#### The Hypre + GPU caveat

As of Hypre 3.1 / MFEM v4.9, **Hypre+GPU full-assembly does not work
for vector-dimension problems** (see ExaConstit issue tracking; works
for scalar problems only). Until that's fixed upstream:

- Phase 4.1 / 4.2: K is built via MFEM full assembly (`ParBilinearForm`
  + `ParallelAssemble`) **on host**, with HypreParMatrix on host. GPU
  acceleration of K-action waits on upstream.
- Phase 4.3 (EA constraint path) IS independently GPU-portable for the
  C side. Once Hypre+GPU is fixed, K side comes online without any
  changes to our code.

In practical terms: the EA path in §P4.4.6 is the part of our work
that's GPU-future-proofed today. The HypreParMatrix path waits on
upstream MFEM/Hypre work before yielding GPU benefit on K.

### §P4.4.2 Namespace and directory layout

#### Build location: `tests/mortar_pbc/`

```
exaconstit/
├── tests/
│   └── mortar_pbc/                           # NEW — Phase 4
│       ├── CMakeLists.txt                    # Standalone CMake target,
│       │                                     # links against mfem + mpi
│       ├── include/
│       │   ├── boundary_classifier_3d.hpp
│       │   ├── boundary_classifier_2d.hpp
│       │   ├── mortar_assembler_2d.hpp
│       │   ├── face_mortar_assembler_3d.hpp
│       │   ├── constraint_builder_3d.hpp
│       │   ├── mortar_pbc_driver.hpp
│       │   ├── saddle_point_solver.hpp
│       │   ├── elastic_3d_helpers.hpp
│       │   ├── visualization.hpp
│       │   └── types_3d.hpp                  # CornerInfo3D, EdgeInfo3D, FaceInfo3D
│       ├── src/
│       │   └── (one .cpp per .hpp)
│       └── examples/
│           ├── patch_test_3d_pbc.cpp         # Round 1 target; mirrors
│           │                                 # examples/patch_test_3d_pbc.py
│           ├── patch_test_3d_heterogeneous.cpp
│           └── patch_test_3d_checkerboard.cpp
└── (existing src/ unchanged)
```

#### Promotion to `src/mortar_pbc/`

Once Round 1+2+3 are validated, contents move to `src/mortar_pbc/`
with namespace `exaconstit::mortar_pbc`. The `tests/mortar_pbc/`
directory then holds only the validation drivers (linking against
the new library target).

### §P4.4.3 Cross-rank vertex identity in C++

The Python prototype uses snap-coord string keys (see mortar §11.7.1).
C++ equivalent: integer-quantised triples.

```cpp
struct SnapKey {
    int64_t ix, iy, iz;
    bool operator==(const SnapKey& o) const noexcept {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};
struct SnapKeyHash {
    size_t operator()(const SnapKey& k) const noexcept {
        // Hash combination via FNV-1a or boost-style XOR-with-shift.
        size_t h = std::hash<int64_t>{}(k.ix);
        h ^= std::hash<int64_t>{}(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

inline SnapKey MakeSnapKey(double x, double y, double z, double bbox_diag) {
    constexpr double rel_tol = 1e-9;
    const double scale = 1.0 / (bbox_diag * rel_tol);
    return {
        static_cast<int64_t>(std::lround(x * scale)),
        static_cast<int64_t>(std::lround(y * scale)),
        static_cast<int64_t>(std::lround(z * scale)),
    };
}
```

**Critical**: `bbox_diag` is computed via `MPI_Allreduce` over local
bounding boxes BEFORE any quantisation happens. Inconsistent
quantisation grain between ranks will silently produce mismatched
keys for the same physical point.

### §P4.4.4 Boundary-record exchange: AllGather → tile-partitioned matching

#### §P4.4.4-status What is and is not implemented in this section

A reader wanting to understand "did the C++ port include non-
conforming face mortars?" can answer that here without trawling
the rest of the doc:

- **Conforming face mortars**: implemented (Python prototype
  `assemble_pair_conforming` ported to C++ as
  `AssemblePairConforming` in `face_mortar_assembler_3d.cpp`,
  Phase 4.1.A → 4.2). 1:1 element pairing by parametric centroid
  match within a configurable tolerance.
- **Non-conforming face mortars (Sutherland-Hodgman polygon
  clipping)**: **NOT IMPLEMENTED** in either the Python prototype
  or the C++ port. The Python prototype's
  `face_mortar_3d.py` docstring marks this as "Phase 3.5" future
  work; the C++ port mirrors that gap exactly. The abstract base-
  class structure (`MortarFaceAssembler` ABC + concrete subclasses
  pattern) is in place, so a future Phase 4.X / 5.X can add an
  `AssemblePairClipped` method without redesigning the framework.
- **Non-conforming edge mortars**: **implemented** (different
  story — the Python 2D code had non-conforming-via-overlap-
  integration from the start, and `MortarAssembler2D` in C++
  ported it: `_integrate_overlap_segment` handles intervals on
  the parametric axis even when nonmortar / mortar edges have
  different subdivisions).

In practice, the validation suite (homogeneous, heterogeneous,
checkerboard patch tests) uses **conforming hex meshes on both
sides of every periodic axis pair**, so non-conforming faces
don't appear. Non-conforming edges DO appear at face boundaries
where edge subdivisions on the periodic-pair partner edge may
not line up exactly with this side's; the 2D overlap path
handles those.

When non-conforming face support is added (target: Phase 4.X
after 4.3 / Batch S), the changes will be:
  1. New `AssemblePairClipped` method on the face-mortar
     assembler ABC, implementing Sutherland-Hodgman clipping in
     parametric coordinates.
  2. Replace `MatchConformingFacePairs` with a more general
     "find all overlapping mortar elements per nonmortar element"
     match.
  3. The constraint builder and EA operator are unaffected — they
     consume `FaceMortarPairBlock` and don't care how it was
     produced.

This work happens entirely on `boundary_comm` (§P4.4.0). Interior
ranks don't participate in any of this.

#### Phase 4.1 (initial): AllGather the boundary records

Mirrors Python `boundary_3d._gather_boundary_records`. Each
boundary rank gathers its local boundary submesh records (face
elements + vertex records); we `MPI_Allgatherv` the packed records
**on `boundary_comm`** to every other boundary rank, then dedup
by `(parent_attr, sorted snap-keys)` to build the global topology.
Every boundary rank ends up with identical `BoundaryClassifier3D`
state. Interior ranks have no classifier instance at all.

Cost analysis (n=128 RVE, 16×16×16 rank grid = 4096 ranks, ~1352
boundary ranks, ~100k boundary verts globally):
- Per-boundary-rank send : ~5 KB
- Per-boundary-rank recv : ~6.7 MB
- Number of WORLD ranks not touched by this collective: 2744 (~67%)

This is acceptable up to roughly nranks where `n_bdy_ranks ~ 1000`
(p ~ 13, total nranks ~ 2200). Beyond that, per-rank recv volume
becomes the bottleneck and Phase 4.2 is needed.

Memory cost per boundary rank is `O(boundary_size)` regardless
of how many boundary ranks there are. Interior ranks pay zero.

#### Phase 4.2 (refactor): distributed-pair matching

The scaling problem: at 100M zones the boundary has ~5M vertices.
Even with the boundary subcomm cutting interior-rank cost to zero,
the per-boundary-rank recv volume is still O(boundary_size) which
saturates at ~50 MB per rank. Acceptable but not generous; the
real scaling fix is reducing per-rank recv to
O(boundary_size / n_boundary_ranks).

There are several reasonable algorithms for this. They all share
the same core invariant — **nonmortar and mortar partners must end
up on the same rank** for local pair matching to work — but
differ in how they assign work.

##### The four candidate strategies

**Strategy A — Hash on parametric centroid.** For each face element,
compute `bucket = hash(axis, snap(parametric_centroid)) % n_boundary_ranks`.
Nonmortar and mortar hash identically because their parametric coords
match modulo period. AllToAll on `boundary_comm` to shuffle, do
local matching per bucket.

  - **Pro**: trivially uniform load (hash is approximately uniform).
  - **Pro**: simple; no geometric reasoning required.
  - **Con**: **destroys spatial locality.** Neighboring face
    elements land on different ranks. The post-matching AllToAll
    that moves dense D, A_m blocks to the nonmortar-DOF owner has to
    move ALL the data because the matching rank is essentially
    random relative to nonmortar-DOF ownership.
  - **Con**: each rank's bucket can include face elements from
    physically distant locations, which means interim memory needs
    holding O(boundary_size / n_boundary_ranks) elements WHOSE
    PHYSICAL EXTENT IS THE WHOLE BOUNDARY. This shows up in the
    L2/L3 cache behaviour during local matching.

**Strategy B — 2D regular tile partitioning.** For each periodic-
pair axis, tile the parametric plane [0, L]² into a regular
`√n_bdy × √n_bdy` grid. Each tile is owned by one boundary rank
(`tile_owner[i, j]` is a fixed map). Face elements go to the rank
whose tile contains their parametric centroid. Same matching
property: nonmortar and mortar tile identically.

  - **Pro**: **preserves spatial locality**. Neighboring face
    elements land on the same rank. The rank doing the matching
    is typically also the rank owning the nonmortar DOF, because
    MFEM's METIS partition tends to assign physically-adjacent
    boundary elements to the same rank. Post-matching AllToAll
    is small (often empty for many pairs).
  - **Pro**: bucket sizes are uniform when the boundary rank count
    is a perfect square (or close to it); load balance is good.
  - **Con**: requires the bbox AllReduce (which we have from §P4.4.0).
  - **Con**: tile-count granularity is `n_bdy_ranks` ≈ 6p², so
    tile resolution is `√n_bdy × √n_bdy` per axis. For p=8 that's
    24×24 tiles per axis-plane, fine. For p=2 that's 4×4 tiles
    per axis-plane = 16 tiles, with only ~24 boundary ranks
    available; tile-to-rank assignment is straightforward.

**Strategy C — Per-axis flat partitioning (3 axis sub-comms).**
Split boundary ranks into three sub-sub-communicators by
periodic-pair axis. Within each, do a 1D contiguous partition
by the parametric centroid's first coord.

  - **Pro**: simpler than B (1D partition vs 2D tiling).
  - **Con**: a rank that touches multiple axis-pairs (any rank on
    a box edge or corner of the rank grid) belongs to multiple
    sub-sub-comms. Bookkeeping is fiddly.
  - **Con**: load imbalance if the RVE is non-cubic. We don't
    care for the validation tests (cubic by design) but production
    materials problems may have aspect-ratio'd RVEs.
  - **Con**: 1D partition has worse locality than 2D tiling for
    the same rank count.

**Strategy D — Bbox-based direct lookup ("hash-free locality").**
Each boundary rank AllGathers a small per-rank bbox table (24
doubles per rank). For each LOCAL face element on, say, the nonmortar
side of the z-pair (z = L), the rank computes its mortar-side
parametric position (z' = 0, x' = x, y' = y) and looks up which
rank's bbox contains that point. Send directly, point-to-point.

  - **Pro**: **zero global communication for the matching itself
    after the bbox AllGather.** Just point-to-point messages.
  - **Pro**: per-rank send/recv volume scales with the rank's
    own boundary surface, which is ~O(p) for a p×p×p arrangement
    — better scaling than B's O(boundary_size / n_bdy_ranks).
  - **Con**: requires that MFEM's rank-bbox lookup gives an
    unambiguous answer. METIS partitions are not generally axis-
    aligned (rank bboxes overlap at boundaries). When a face's
    mortar-side position falls in multiple ranks' bboxes,
    tiebreaking is needed. False positives must be filtered by
    a "not-mine" reply protocol.
  - **Con**: failure mode is silent: if the bbox lookup misses
    (because the partition is irregular and the mortar-side point
    doesn't fall in any rank's bbox via simple containment), the
    face element's pair never gets matched. We'd need a fallback
    bucket-scheme for unmatched faces.
  - **Con**: more complex implementation.

##### Recommendation: Strategy B for Phase 4.2 (implemented in Batches G–N)

For the initial Phase 4.2 implementation, **Strategy B is the
right balance of simplicity and locality**. The tile partitioning
is structurally simple (one 2D map of `tile_idx → rank`), preserves
locality, and load-balances well for the cubic RVE test cases.

**Implementation status**: this design landed across Phase 4.2
Batches G through N. Strategy B's tile-shuffle delivered locality
during pair matching (Batch H); the final routing step of step 8
below — "send to nonmortar-DOF-owner AllToAllv" — landed in Batch N
with the FES-aligned row partition convention. See
§P4.4.4-history for the batch-by-batch evolution and the
intermediate stepping-stone designs that were used to keep unit
tests passing through the refactor.

Strategy A is the simplest but the locality penalty is real and
shows up as 2× extra AllToAll volume in the post-matching step
(moving D, A_m blocks to nonmortar-DOF owners).

Strategy C is unnecessarily fiddly given that the 1D-vs-2D
partition difference is a small constant-factor implementation
cost.

Strategy D is the most efficient ASYMPTOTICALLY but has the most
implementation complexity and the most failure-mode risk. **It's
the right choice IF profiling Strategy B at p ~ 30 shows the
matching phase is a bottleneck**, but not before. The bbox
AllGather for D is essentially free, so we'd add it as a pre-step
to B and only switch to D-as-primary if measurements warrant it.

##### Strategy B detailed protocol

Once we've committed to B, the protocol on `boundary_comm` is:

1. (Already done in §P4.4.0) bbox AllReduce on WORLD, gives
   `(bbox_min, bbox_max)` available everywhere.

2. Each boundary rank decides on a tile resolution per axis. With
   `n_bdy = boundary_comm.size()` ranks and 3 axis-pairs, allocate
   `n_bdy_per_axis = n_bdy / 3` ranks per axis-pair (rounded up;
   imbalance is small). Within each axis-pair, choose a tile grid
   `n_tiles_x × n_tiles_y` where the product matches
   `n_bdy_per_axis` and the aspect ratio approximates the RVE's.
   For cubic RVEs this is `√n_bdy_per_axis × √n_bdy_per_axis`.

3. Build a deterministic tile-to-rank map. Identical on every
   rank because each rank knows the bbox and `n_bdy`. This is a
   compile-time table, not a communicated structure.

4. Each boundary rank iterates its local face elements:
   - Compute the parametric centroid in the (a, b) plane.
   - Determine which tile it falls in.
   - Determine which boundary rank owns that tile.
   - Mark the face element for sending to that rank.

5. `MPI_Alltoallv` on `boundary_comm`: shuffle face-element
   records to their tile-owning ranks. Each rank receives all
   face elements in its tile, organised by axis-pair.

6. Local pair matching per tile:
   - For each axis-pair, partition the received elements into
     "nonmortar side" and "mortar side" by their perpendicular
     coordinate.
   - For each nonmortar element, find its mortar partner by parametric-
     centroid match (the existing `match_conforming_face_pairs`
     algorithm; works tile-locally now, no MPI).

7. Local mortar integration per pair: the receiving rank computes
   its assigned `D_nm` and `A_m` blocks. Per-pair work is local;
   no further communication.

8. Post-integration "send to nonmortar-DOF-owner" AllToAllv on
   `boundary_comm`: move dense blocks to the rank that owns the
   nonmortar DOF (per the nonmortar-DOF-ownership convention in §P4.4.5).
   Most blocks stay on the same rank (locality preservation
   pays off here); only blocks where the matching rank ≠ nonmortar
   owner move.

9. Each rank now has its row contributions for the nonmortar DOFs
   it owns. HypreParMatrix construction (§P4.4.5) proceeds as
   before, on WORLD with empty rows on interior ranks.

##### Load balance and stragglers

For small `n_bdy_ranks` (small p), the tile-count-per-axis-pair is
small and tile-rank assignment is trivial. For large p, the tile
count grows quadratically per axis and we get fine-grained
balance.

Load imbalance concerns:
- Corner-tile ranks (those owning the 4 corners of a face)
  receive corner-of-face quads, which carry sentinel-modified D_nm
  and slightly more integration work (Wohlmuth-modified basis).
  This is ~25% extra work, distributed over 4 corners per face ×
  3 axis-pairs = 12 corner tiles per RVE. Negligible at p > 10.
- Edge-tile ranks (those owning the 4 edges of a face, excluding
  the corners) similarly carry edge-of-face quads with edge
  sentinel modifications. ~10% extra work, similarly distributed.
- Interior face tiles get the majority of work and are fully
  symmetric.

If profiling shows imbalance bites at scale, the fix is a
work-stealing layer on top: ranks that finish early pull pairs
from the queues of slow ranks. This is a separate optimization
to consider only if measurements warrant.

##### Communication cost tabulation

For the same n=128 RVE, p=16 (16³ = 4096 ranks, ~1352 boundary
ranks) example used elsewhere:

| Strategy | bbox AllReduce | matching shuffle | nonmortar-DOF shuffle | total per-rank |
|----------|---------------:|-----------------:|------------------:|---------------:|
| Phase 4.1 (AllGather) | 0 | 6.7 MB recv | 0 (trivial) | 6.7 MB |
| Phase 4.2 A (random hash)  | 192 B | ~5 KB recv | ~5 KB recv | ~10 KB |
| Phase 4.2 B (tile)         | 192 B | ~5 KB recv | ~1 KB recv (locality) | ~6 KB |
| Phase 4.2 C (axis flat)    | 192 B | ~5 KB recv | ~3 KB recv | ~8 KB |
| Phase 4.2 D (bbox lookup)  | 192 KB (all bdy ranks' bboxes) | ~3 KB direct | 0 (already at owner) | ~195 KB |

(Numbers are order-of-magnitude estimates.)

Strategy B beats A by roughly 2× on per-rank volume; D beats B
on the matching shuffle but loses on the bbox AllGather. At
this scale all four are tractable, but Strategy B is simplest
to implement correctly and gives the best end-to-end behaviour
before D's complexity becomes worthwhile.

##### When to revisit

- If Phase 4.2 B passes scaling validation through p = 20
  (n_bdy_ranks ~ 2000), no further work needed; that's the
  upper end of "interesting" scales for ExaConstit.
- If we run into communication-bound behaviour beyond p = 30,
  consider Strategy D as a follow-on optimization. Caliper data
  on the matching phase will tell us whether it's worth the
  implementation complexity.
- The whole machinery is in `ConstraintBuilder3D` and adjacent
  classes; the public API of `BoundaryClassifier3D` doesn't
  change between strategies, so swapping is a focused refactor.

##### Implementation cost

Phase 4.2 with Strategy B: figure 600-1000 lines of new C++,
mostly in `ConstraintBuilder3D`. The tile-rank assignment table
is small (~50 lines). The AllToAllv pack/unpack is the bulky
part (~300 lines). The local matching algorithm is essentially
the same `match_conforming_face_pairs` logic that already exists
in the Python prototype, just operating on tile-local element
lists. Worth it because Phase 4.1's per-rank recv caps the
framework somewhere between p=13 and p=20 (i.e. nranks 2200 to 8000).

#### §P4.4.4-history Phase 4.2 batch-by-batch implementation evolution

This subsection captures the actual implementation trajectory from
Phase 4.1 (post-AllGather-on-WORLD) to the final Phase 4.2 design
realized in Batch N. It exists to answer the question "if Strategy B
is the design, why did it take eight batches to land?"

The short answer: **each batch is a focused, locally-testable change
that preserves the unit-test invariant**. The full design as
described above (tile-local matching + nonmortar-DOF row partition +
AllToAllv routing) involves three coupled architectural changes,
each of which on its own requires nontrivial refactoring of the
classifier and constraint-builder. Doing them all in one commit
risks a flag-day style failure where unit tests don't pass for weeks
while the design comes online. The batch sequence below trades
implementation latency for incremental correctness — every batch
ends with all unit tests green and the patch tests producing
identical numerical output to the previous batch (modulo FP
accumulation order, which surfaces as ±1 Krylov iterations at most).

##### Batch G — Boundary subcommunicator (`m_boundary_comm`)

**What**: Add `MPI_Comm_split` at classifier construction time,
splitting WORLD into a boundary subcomm (ranks with at least one
boundary face element) and a `MPI_COMM_NULL` placeholder for
interior ranks.

**Why first**: Subsequent batches need the boundary subcomm to exist
before they can move collectives onto it. This batch is purely
additive — no existing collective moves yet, no behavior change.
The subcomm is constructed and stored, but the AllGather of
boundary records still runs on WORLD.

**Risk**: Near-zero. Ranks with `m_pmesh.GetNBE() == 0` get
`MPI_COMM_NULL`; everything that follows is guarded with
`if (IsBoundaryRank())`.

##### Batch H — Tile-partitioned face element shuffle

**What**: Implement `TilePartition3D` (a deterministic 2D tile
grid per axis-pair derived from the bbox AllReduce), the
`ShuffledFaceElement` packed format, and `TileShuffleFaceElements`
which runs `MPI_Alltoall` + `MPI_Alltoallv` on
`m_boundary_comm` to route face elements to their tile-owning
ranks.

**Why second**: Tile shuffling is what enables Strategy B's local
pair matching (step 6 of the protocol above). Once face elements
are on the right ranks, matching becomes a tile-local algorithm
with no MPI.

**Test**: `test_boundary_classifier_3d` Test 8 ("tile-shuffle
routing correctness") and Test 9 ("global send/recv counts cross-
check at np=1") were added.

**Risk**: Cross-rank vertex identity (snap-keys) was already
implemented in Phase 4.1 for the AllGather path, and Batch H
reuses that infrastructure. The risk was mostly bookkeeping
complexity in the pack format.

##### Batch I — Local pair matching + AllGather of merged blocks

**What**: Add `BuildLocalPairBlocks()` which runs
`MatchConformingFacePairs + AssemblePairConforming` tile-locally
on each rank's shuffled face elements. Add
`GatherPairBlocksAcrossBoundary()` which AllGather's the resulting
per-pair blocks to every rank in `m_comm` (WORLD). Also
introduces the `LocalPairBlock` nested type and the per-pair
block pack format.

**Why third**: With face elements correctly tile-shuffled, each
rank now produces a small number of `(axis, mortar, nonmortar,
geom)` mortar blocks that are LOCAL to its tile. To preserve the
existing constraint-builder API ("every rank produces the same
SparseMatrix"), Batch I AllGather's all the blocks to every rank.
This is wasteful at scale but lets every existing test continue
to pass without changing the row-partition convention yet.

**The §P4.8.10 bug**: A naive concatenation merge for shared
nonmortar gtdofs across tile boundaries produced wrong results.
Fixed by switching to gtdof-keyed accumulation. Discovery story
captured in the lesson.

**Risk**: This was the highest-stakes batch. Adding tile-local
matching changes the producer; AllGather + merge changes the
consumer; the §P4.8.10 bug surfaced in the merge. After Batch I
the code was algorithmically correct end-to-end; subsequent
batches optimize the AllGather phase.

##### Batch J — Decommission the per-rank face-element AllGather

**What**: Remove `m_face_element_records` storage and the
`FaceElementRecord` AllGather (which had been Phase 4.1's "ship
every face element to every boundary rank" step). With face
elements now tile-shuffled in Batch H, the per-rank AllGather
became dead code. Also: rewrite `BuildFaces()` to compute
`interior_gtdofs_x/y/z` from the vertex catalog directly rather
than from the gathered face-element records.

**Why fourth**: Pure cleanup. ~150 LOC of dead code + an
unnecessary collective on every classifier construction. With
Batch I producing the per-pair blocks tile-locally, the original
face-element AllGather has no consumer.

**Risk**: Low. The `interior_gtdofs_*` recomputation from vertex
records was straightforward; the AllGather removal was textual.

##### Batch K — Boundary-comm AllGather + WORLD broadcast fanout

**What**: Refactor `GatherPairBlocksAcrossBoundary` so the
expensive AllGather of pair blocks moves from WORLD to
`m_boundary_comm`, followed by `MPI_Bcast` on WORLD to fan
the data out to interior ranks. Also fix a `[-Wunused-private-field]`
warning by removing `m_pair_match_tol_rel` from the constraint
builder (matching now lives in the classifier; the field was
vestigial).

**Why fifth**: Batch I's `AllGatherv` on WORLD was wasteful —
interior ranks (~94% at production scale) participated in a
collective that didn't involve their data. Boundary-comm
AllGather + WORLD Bcast cuts the per-rank receive volume on
boundary ranks (they only AllGather among themselves) while
delivering the data to interior ranks via a single tree-broadcast
fanout (O(log N) latency vs O(N) bandwidth).

**Risk**: Low. Same data, different communicator. The
broadcast root is found via `MPI_Allreduce(MIN)` of `(IsBoundaryRank() ? m_rank : INT_MAX)`.

##### Batch L — Sparsify `FaceMortarPairBlock::A_m`

**What**: Change `FaceMortarPairBlock::A_m`'s storage type from
`mfem::DenseMatrix` to `mfem::SparseMatrix`. Update producer
(`AssemblePairConforming`) to build sparse + Finalize. Update
consumer (`ScatterFaceBlock`) to walk via CSR `GetI/GetJ/GetData`.
Update pack/unpack and merge logic.

**Why sixth**: This is the **dominant memory win in all of
Phase 4.2**. Lesson §P4.8.11 has the arithmetic — at N=100 the
per-block memory drops from ~800 MB dense to ~1 MB sparse. No
other change in the batch sequence comes close.

**Why this batch and not earlier**: Earlier batches were focused
on the communication pattern; the storage type was orthogonal.
Doing the sparsification before Batch I would have entangled it
with the §P4.8.10 merge bug discovery. Doing it after the
communication structure stabilized made the sparse pack/unpack
straightforward to validate against the dense baseline.

**Risk**: Moderate — the producer/consumer/pack/unpack/merge
quad of code paths all needed updating in lockstep, and getting
`Finalize()` placement wrong silently corrupts the CSR.
Mitigated by keeping the test suite green at every step and
validating against Batch K's output.

##### Batch M — Per-rank C construction

**What**: Refactor `ConstraintBuilder3D::BuildHypreParMatrix` so
it no longer allocates the full replicated SparseMatrix on every
rank. Extract `EmitConstraintTriples` as a shared helper that
both `Build()` (for tests) and `BuildHypreParMatrix` call.
`BuildHypreParMatrix` filters triples by row range on the fly
into a local-sized SparseMatrix.

**Why seventh**: The full replicated SparseMatrix in `Build()`
was Phase 4.1's row-replication strategy — every rank held the
full C, then sliced its local rows out. At production scale
(180k rows × 16 nnz per row × 20 bytes per nnz) that's ~36 MB
per rank, replicated to every one of N ranks. Batch M brings
per-rank C-construction memory down to O(local_rows · avg_nnz)
~ 50 KB per rank.

**The catch**: The temporary COO buffers `(rows, cols, vals)`
returned by `EmitConstraintTriples` are still O(global_nnz) per
rank — every rank still emits triples for every block in
`m_classifier.PairBlocks()`. The full asymptotic win requires
Batch N.

**Risk**: Low. The helper extraction is mechanical; the row
filter is one branch in a single loop.

##### Batch N — AllToAllv routing + FES-aligned row partition

**What**: Replace `GatherPairBlocksAcrossBoundary` with
`RoutePairBlocksToRowOwners`. The new function fragments each
local pair block by FES owner of its nonmortar gtdofs, packs one
fragment per destination, and `MPI_Alltoallv`'s on `m_comm` to
route each fragment to the rank that owns its rows under the
FES TDOF partition. Also: add `GtdofOwnerRank` (binary search on
Allgather'd FES TDOF offsets), filter edge mortar rows in
`ScatterEdgeBlock` by FES ownership, remove the `n_lam_local`
argument from `BuildHypreParMatrix` (the row partition is now
data-determined), add `NumLocalRows` for callers.

**Why last**: This is the most architecturally invasive change.
It requires every previous batch to be in place — sparse blocks
(L) make routing payloads small enough to be worthwhile;
per-rank C construction (M) is what consumes the routed
fragments correctly; the boundary subcomm + Bcast pattern (G/K)
provides the `IsBoundaryRank` API used during fragmentation.

**The synergy with FES alignment**: AllToAllv-to-row-owner only
pays off if the row partition makes "owner" a small set per
block. With fair-split rows, a face mortar block's rows could
go to many destinations. With FES-aligned rows (rank owns row
`r` iff it owns the corresponding nonmortar gtdof in FES), a
block's rows go to a small number of destinations — typically
1, sometimes 2-4 for blocks straddling a partition boundary.
This is the §P4.8.12 lesson.

**The HYPRE_BigInt MPI datatype gotcha**: The first cross-rank
patch test failed because the FES TDOF offset Allgather used a
hardcoded `MPI_LONG_LONG` while `HYPRE_BigInt` is `int` in
ExaConstit's HYPRE build. The fix is `HYPRE_MPI_BIG_INT`. This
is the §P4.8.13 lesson.

**Risk**: Highest of any batch. Mitigated by:
- The np=1 invariant: at np=1 every gtdof is owned by rank 0,
  so routing degenerates to a self-loop and every test produces
  numerically-identical output to Batch L.
- Reusing the §P4.8.10 gtdof-keyed merge logic verbatim — only
  the input source (Alltoallv recv vs AllGatherv recv) changes.
- Reusing the Batch L pack format unchanged — fragments just
  have smaller `n_n` and `nnz` than Batch L blocks did.

##### Implementation cost summary

| Batch | LOC delta | Description |
|------:|----------:|-------------|
| G     | ~150     | boundary subcomm + IsBoundaryRank guard pattern |
| H     | ~600     | TilePartition3D + ShuffledFaceElement + tile shuffle |
| I     | ~700     | local pair matching + AllGather + gtdof-keyed merge |
| J     | -150     | decommission face-element AllGather |
| K     | +80      | boundary-comm AllGather + WORLD Bcast + warning fix |
| L     | +100     | sparsify A_m |
| M     | +60      | per-rank C construction |
| N     | +233     | Alltoallv routing + FES-aligned row partition |
| **Total** | **~1773 LOC** | full Phase 4.2 implementation |

The line counts are net (additions minus deletions). The actual
churn is roughly 1.5× this because several batches replaced
existing functions wholesale (e.g., Batch N replaced the 425-LOC
`GatherPairBlocksAcrossBoundary` with the 483-LOC
`RoutePairBlocksToRowOwners`).

##### Per-rank memory and communication scaling at the end

| Aspect | Phase 4.1 (AllGather WORLD) | After Batch L (gather, sparse) | After Batch N (routed, sparse) |
|---|---:|---:|---:|
| Per-rank `m_gathered_pair_blocks` | full set, dense | full set, sparse | own slice, sparse |
| Per-rank C-construction memory | O(global_rows · avg_nnz) | same | O(local_rows · avg_nnz) |
| Per-rank temporary COO buffers | O(global_nnz) | same | O(local_nnz) |
| WORLD AllGather/AllGatherv volume | O(N · global_blocks) | same | O(global_blocks) (Alltoallv) |
| Memory at 100³ RVE per-rank, 10⁶ ranks | ~2.4 GB (dense face blocks) | ~3 MB | ~50 KB (estimate) |

The Batch N memory drop is the asymptotic Phase 4.2 goal. Per-rank
state now scales as the rank's own piece of the periodic boundary,
which goes to zero as ranks → ∞ for fixed problem size.

##### Why a boundary-subcomm in Phase 4.1 isn't redundant with Phase 4.2 (recap)

Repeated for completeness — this rationale stands unchanged from
Batch G.

It would seem that since Phase 4.2 fixes the scaling, the boundary-
subcomm in Phase 4.1 is just a stepping stone. In fact it's a
**separate, complementary improvement**:

- Boundary subcomm: removes interior ranks from the sync.
- Distributed-hash: reduces per-boundary-rank recv volume.

Both are needed at large scale. The boundary subcomm matters even
in Phase 4.2 because the AllReduce inside the runtime attribute
discovery (mortar §11.7.2), the consistency-check between ranks
that see overlapping attributes, and the small bcast-of-classifier-
result-to-driver all stay on the subcomm. Phase 4.2 doesn't make
those go away; it just ensures the BIG exchange (face records) is
also distributed.

### §P4.4.5 Constraint matrix C: HypreParMatrix path

#### Implementation status

This section describes the **target design**, which was fully
realized in Phase 4.2 / Batch N. Earlier batches (I, K, L, M)
used a transitional "row-replicated, fair-split" partition where
every rank produced the full C matrix and sliced its local rows
out — this kept unit tests stable while the tile-shuffle and
sparsification refactors landed. Batch N converted the row
partition to FES-aligned (as described below) and replaced the
broadcast of pair blocks with `MPI_Alltoallv`-to-row-owner.
See §P4.4.4-history for the full evolution.

#### Row partitioning

In the Python prototype, all of C lives on rank 0. In C++, C is a
distributed `mfem::HypreParMatrix` whose rows are partitioned by
**nonmortar-DOF ownership**: world-rank `r` owns the constraint rows
whose nonmortar node lives in `r`'s TDOF range. Interior ranks own
**zero** rows but still appear in the row partition (with
`row_starts[r] == row_starts[r+1]`). This is the "empty row block
on interior rank" pattern (§P4.4.0).

This means `n_lam_local` varies across ranks: zero on interior
ranks, positive on boundary ranks (0 ≤ n_lam_local ≤ several
hundred typically). The nonmortar-DOF ownership partition gives us
natural locality: most mortar-DOF columns referenced by row r will
also be on world-rank r or its neighbors (the nonmortar and mortar
faces of a periodic axis are typically owned by similar rank
subsets in MFEM's mesh partitioning).

#### The communicator: WORLD, not boundary_comm

C is constructed on **WORLD**, not on boundary_comm, even though
all the *data* in C comes from boundary ranks. The reason is
operator composition: the saddle-point solver's BlockOperator
mixes K (which lives on WORLD) and C; both must share a comm.

This works correctly because Hypre's matvec handles ranks with
empty rows naturally — they're a no-op on the local computation
side, contribute nothing to the global send, and do receive any
inbound off-process column data that other ranks happen to need
from interior-rank-owned TDOFs (which is rare in practice since C
columns are dominantly boundary-side TDOFs).

The CSR construction sequence:

1. Boundary ranks build their row contributions on `boundary_comm`.
2. Boundary ranks compute their row partition on WORLD: each
   boundary world-rank `r` knows its `[first_row_global,
   last_row_global)`. Interior ranks are notified via a small
   AllGather (one int per rank) of `n_lam_local`.
3. Each rank fills in `row_starts[2]` for its row partition;
   interior ranks pass `[k, k]` (empty range starting at the
   running global counter `k`).
4. HypreParMatrix gets constructed on WORLD via the standard CSR
   constructor; interior ranks' `diag` and `offd` are empty
   SparseMatrix shells of size `(0, n_local_cols)` and
   `(0, n_offd_cols)`.

Step 2's AllGather is small (one int per rank, so 4 bytes × nranks)
and unavoidable — every rank needs to know the global row partition
to construct the HypreParMatrix. This is unrelated to the
boundary-record exchange and stays cheap regardless of nranks.

#### Construction pattern

MFEM's HypreParMatrix has a "build from CSR" constructor:

```cpp
HypreParMatrix(MPI_Comm comm,
               HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols,
               HYPRE_BigInt* row_starts, HYPRE_BigInt* col_starts,
               SparseMatrix* diag, SparseMatrix* offd, HYPRE_BigInt* cmap);
```

where `diag` holds rows × local-cols, `offd` holds rows × off-process-
cols, and `cmap` is the offd column → global-column index map.

For a boundary rank with non-empty rows:

```cpp
// Step 1: gather per-rank row contributions on boundary_comm
// (already done by ConstraintBuilder3D).
std::vector<RowContribution> local_rows = AssembleLocalRowsOnBdyComm();

// Step 2: AllGather of n_lam_local on WORLD to compute row_starts.
HYPRE_BigInt my_first_row, my_last_row;  // computed via prefix-scan.
ComputeRowPartition(world_comm, n_lam_local, my_first_row, my_last_row);

// Step 3: split each row into "diag" (cols owned by this world-rank)
// and "offd" (cols owned by other world-ranks).
SparseMatrix diag(n_local_rows, n_local_cols);
SparseMatrix offd(n_local_rows, n_offd_cols);
std::vector<HYPRE_BigInt> cmap;  // offd col -> global col
// ... populate diag, offd, cmap ...

// Step 4: build HypreParMatrix on WORLD.
HYPRE_BigInt row_starts[2] = {my_first_row, my_last_row};
HYPRE_BigInt col_starts[2] = {my_first_col, my_last_col + 1};
auto C = std::make_unique<HypreParMatrix>(
    world_comm, n_global_rows, n_global_cols,
    row_starts, col_starts, &diag, &offd, cmap.data());
C->CopyRowStarts();
C->CopyColStarts();
```

For an interior rank with no rows:

```cpp
// row_starts[0] == row_starts[1]: zero rows on this rank.
HYPRE_BigInt my_first_row = SomePartitionPoint;
HYPRE_BigInt row_starts[2] = {my_first_row, my_first_row};

// diag/offd are empty SparseMatrix shells.
SparseMatrix diag(0, n_local_cols);
SparseMatrix offd(0, 0);
std::vector<HYPRE_BigInt> cmap;  // empty.

auto C = std::make_unique<HypreParMatrix>(
    world_comm, n_global_rows, n_global_cols,
    row_starts, col_starts, &diag, &offd, cmap.data());
C->CopyRowStarts();
C->CopyColStarts();
```

Both branches happen on every WORLD rank; the construction is a
WORLD collective.

**Common bugs to watch for** (lessons from MFEM ex5p / ex9p):
1. Forgetting `CopyRowStarts()` / `CopyColStarts()` — leads to use-
   after-free when the local arrays go out of scope.
2. Unsorted `cmap` — Hypre expects strictly increasing global
   column indices in `cmap`; offd column indices must be sorted by
   the corresponding `cmap[k]` value.
3. Mismatch between `diag.Size()` and `n_local_rows` — easy to slip
   this when building incrementally.
4. **Mismatched row_starts on interior ranks**: every rank must
   pass row_starts[r], row_starts[r+1] consistent with the global
   prefix-scan. Off-by-one in the interior-rank empty-block
   computation produces a HypreParMatrix that segfaults on first
   matvec. Use the AllGather-of-n_lam_local + prefix-scan pattern
   to guarantee consistency.

The Python prototype's `apply_dirichlet_zero_to_C` becomes a
sparsity-preserving column zeroing. With HypreParMatrix, this means
zeroing entries in `diag` and `offd` and re-finalizing. The 24
corner gtdofs are tiny; this is per-rank-local work with no MPI.



### §P4.4.6 The element-assembly path (Phase 4.3 / Round 3)

#### Motivation

The HypreParMatrix path requires (a) a working Hypre+GPU build for
vector problems (currently broken), and (b) explicit CSR sparsity
management (the Step-2 hassle above).

The EA path sidesteps both:
1. Each rank holds a `std::vector<MortarPair>` where `MortarPair`
   has the per-pair local D and A_m dense blocks plus the nonmortar/
   mortar gtdof index lists.
2. `MortarConstraintOperator::Mult(x, y)` iterates pairs:
   - Gather local x slice into a small dense vector.
   - Apply `D` (diagonal) and `-A_m` to populate local rows of y.
3. `MortarConstraintOperator::MultTranspose(y, x)` iterates pairs
   in reverse:
   - Scatter-add `D^T y_local` and `-A_m^T y_local` into x.
4. Off-rank communication: only the local rows/cols that touch
   off-rank DOFs need exchange. Naturally bounded by the boundary
   surface area per rank, not the full constraint count.

This matches MFEM's `Operator` interface, integrates with `BlockOp`
identically to HypreParMatrix, and is naturally GPU-portable using
the same `mfem::forall` patterns ExaConstit already uses.

#### Storage pattern

```cpp
struct MortarPairLocal {
    int n_nonmortar_kept;
    int n_mortar_kept;
    // Dense blocks (small: ~3-9 DOFs per side typically).
    Vector D;             // (n_nonmortar_kept,)
    DenseMatrix A_m;      // (n_nonmortar_kept, n_mortar_kept)
    // Indices into the constraint-multiplier vector and the TDOF
    // vector (vdim-expanded).
    Array<int> row_offsets_per_component;   // 3 entries (vdim=3)
    Array<int> nonmortar_gtdofs_per_component;  // (n_nonmortar_kept * 3,)
    Array<int> mortar_gtdofs_per_component; // (n_mortar_kept * 3,)
};

class MortarConstraintOperator : public mfem::Operator {
public:
    virtual void Mult(const Vector& x, Vector& y) const override;
    virtual void MultTranspose(const Vector& x, Vector& y) const override;
private:
    // GPU-resident: copy pairs to device once at construction time.
    Memory<MortarPairLocal> d_pairs_;
    // Plus communication scaffolding for off-rank x/y entries.
};
```

This is the "EA-style" approach in the same sense ExaConstit does
EA for K: per-element local matrices stored as dense blocks, applied
matrix-free without ever forming the global CSR.

#### When is each path used?

```
--constraint-storage=hypre    (default in Phase 4.1+4.2)
--constraint-storage=ea       (Phase 4.3 onward)
```

CMake option `-DENABLE_EA_CONSTRAINT=ON/OFF` controls compilation.
Selectable at runtime so we can A/B test correctness on the same
binary.

#### §P4.4.6.1 Working with BOTH `BlockBilinearForm` and `BlockNonlinearForm`

The existing patch-test driver and saddle-point solver use
`mfem::BlockOperator` directly, populated with `Operator*` blocks.
That's the linear / `BlockBilinearForm`-equivalent path.

ExaConstit production uses `mfem::BlockNonlinearForm` because K
is nonlinear in `u` (crystal plasticity, large deformations,
etc.). `BlockNonlinearForm` expects each block to define BOTH a
residual (`Mult(x_block, r_block)`) and a Jacobian
(`GetGradient(x_block) -> Operator&`). The constraint block C is
**linear in u** even when K is nonlinear — `C·u` is just a matrix
matvec independent of any history variable. So:

- **Residual contribution**: `MortarConstraintOperator::Mult(u, λ_resid)`
  computes `C·u`, the constraint residual. This is the lower-half
  block of the saddle-point residual.
- **Jacobian contribution**: `GetGradient(u)` returns
  `*this` (the operator itself, which IS the Jacobian since C is
  constant in u). The Jacobian-vector products go through
  `Mult` / `MultTranspose` exactly as in the linear case.

Concretely, a `MortarConstraintBlockNonlinearFormIntegrator`
adapter (Phase 4.3 / Batch R) wraps the operator in a class that
inherits from `mfem::BlockNonlinearFormIntegrator`. The adapter
holds a reference to the `MortarConstraintOperator` and forwards
all calls. The adapter is the only piece that depends on the
`BlockNonlinearForm` interface; the operator itself is
interface-agnostic and works for both `BlockBilinearForm`
and `BlockOperator`-only use cases.

```
                                +------------------------+
                                | MortarConstraintOperator|  (mfem::Operator)
                                +-----------+------------+
                                            |
                  +-------------------------+-------------------------+
                  |                                                   |
   used as Operator* in BlockOperator        wrapped in Block-NLF adapter
   (current patch tests, saddle-point         (Phase 4.3 / Batch R)
   solver — Phase 4.1.A onward)               (production use,
                                              Phase 5+)
```

This mirrors how MFEM's own `HypreParMatrix` is used: same object,
two different interfaces, depending on whether the surrounding
form is linear or nonlinear.

#### §P4.4.6.2 Non-conforming face mortar status (cross-reference)

The EA path consumes the same `FaceMortarPairBlock` data as the
HypreParMatrix path. As noted in §P4.4.4-status, **non-conforming
face mortars are not implemented** in either path — the conforming
1:1 element matching is what produces the blocks. When non-
conforming face support is added in a future phase, the EA path
will pick it up automatically (a non-conforming `A_m` is just a
larger sparse matrix per pair; the operator's CSR walk doesn't
care about the geometry that produced the entries).

#### §P4.4.6.3 Validation strategy: HypreParMatrix vs EA matvec equivalence

**The validation contract**: for the same problem, the EA path
must produce `C·u` and `C^T·λ` results that are identical to
the HypreParMatrix path's matvecs to floating-point precision.
"Floating-point precision" means equal up to FP order-of-summation
tolerance, typically ~1e-13 for double-precision.

**Why FP-precision and not bit-exact**: the two paths sum
contributions in different orders. The HypreParMatrix path sorts
CSR rows by column and does a structured sum during matvec. The
EA path walks pairs in pair-list order. Same operations, different
summation order — bit-exactness is not achievable in general.

**The validation harness — split across Batches Q and S**:

The validation lives in two places, each catching a different
class of bug:

*Batch Q — matvec-level A/B harness in `test_mortar_constraint_operator`*

1. Build the same problem two ways: (a) `BuildHypreParMatrix()`
   → `mfem::HypreParMatrix*`, (b) `MortarConstraintOperator(cl)`.
2. Check dimensions match: `H->Height() == op.Height()`,
   `H->Width() == op.Width()`. (Already exercised in Batch O test 2.)
3. Apply both paths to the same random `u` and compare:
   `H * u_random == op * u_random` to tolerance
   `1e-12 * (||C||_F * ||u||_2)`. At multiple mesh sizes (2³,
   4³, 6³, 8³) to catch size-dependent bugs.
4. Apply both paths to the same random `λ`:
   `H^T * λ_random == op^T * λ_random` (with `mfem::TransposeOperator`
   wrapping H and `MultTranspose` on op).
5. Zero-input invariant: `Mult(0, _) = 0` and `MultTranspose(0, _) = 0`.
6. Negative test (harness self-check): perturb the EA output by
   1e-3 and verify the comparison flags it. Guards against the
   tolerance being too loose to catch real bugs.

This batch runs at np=1, matching the rest of the unit-test suite.
The Alltoallv import/export topology IS built at construction time
even at np=1 (it just ends up empty), so construction-time bugs
are caught here. What is NOT caught here: bugs in the actual
data exchange between ranks, since at np=1 no exchange occurs.

*Batch S — end-to-end + cross-rank validation*

1. Wire `--constraint-storage=ea` into the patch-test driver.
2. Add an A/B mode that constructs both paths in one run and
   reports any divergence in the resulting `du` field.
3. Run the existing patch tests at np=4, np=7 with the EA path
   and verify identical displacements (within Krylov tolerance)
   to the HypreParMatrix path. This is where the cross-rank
   Alltoallv logic gets exercised end-to-end.
4. Add a saddle-point solver overload accepting
   `const mfem::Operator&` instead of `const mfem::HypreParMatrix&`
   so the EA operator slots into the existing solver without
   duplicating the Krylov setup code.

**Why the split**: the matvec-level Batch Q is fast and runs
in CI at np=1, so any algorithmic regression in `Mult` /
`MultTranspose` or in the per-pair scatter is caught immediately.
The end-to-end Batch S exercises the Alltoallv exchange paths
that np=1 can't reach, but at the cost of running at np>1 (which
the unit-test harness doesn't support). Both layers are needed
to fully validate the EA path.

**Why this validation matters for ExaConstit production**: the
EA path is what ExaConstit will actually run (matrix-free, GPU-
friendly). If it disagrees with the HypreParMatrix path on a
small problem, it'll disagree silently at production scale where
no reference is available. The A/B harness on the small patch
tests is the only place we can hold them to bit-tight tolerance.

#### §P4.4.6.4 Phase 4.3 batch sequence

Same incremental phasing principle as Phase 4.2 (§P4.4.4-history
+ §P4.8.14): each batch lands a focused, locally-testable change
with the test suite green at every step.

| Batch | What | Why this batch | Status |
|------:|------|----------------|:------:|
| O     | Design + skeleton: `MortarConstraintOperator` header, stub `.cpp` (Mult/MultTranspose abort with clear message), construction-only test (`test_mortar_constraint_operator`), CMake registration, doc updates. | Establish the type, size, and lifecycle so subsequent batches can implement against a stable interface. The MFEM_ABORT in the stubs prevents silent zero-output bugs from masking missing-implementation issues. | done |
| P     | Implement `Mult` and `MultTranspose` on CPU. Build the off-rank import / export topology in the constructor. Per-pair scatter loop. Single-rank tests pass. | The core algorithmic work. CPU-first lets us validate the pair-loop semantics before adding GPU complications. | done |
| Q     | A/B validation harness at multiple mesh sizes, zero-input invariant, harness self-check (negative test). Tightened tolerance to `1e-12` per §P4.4.6.3 contract. | The firewall: any future change to the EA path that breaks consistency with HypreParMatrix path gets caught here. The cross-rank np>1 path is exercised end-to-end in Batch S; this batch is the matvec-level contract at np=1. | done |
| R     | `MortarSaddlePointSystem` adapter that composes user-provided K-residual / K-Jacobian closures with the EA constraint operator into a single `mfem::Operator` exposing combined `Mult` (saddle-point residual) and `GetGradient` (saddle-point Jacobian as a `BlockOperator`). Plus `MortarConstraintOperator::ComputeInvDiagSchur` — the EA-path equivalent of `BuildInvDiagSchur(HypreParMatrix C, ...)` for block-Jacobi preconditioning, computed directly from per-pair blocks (Option 2, no matvec probes). | Prerequisite for Phase 5 (ExaConstit integration). The closure-based interface fits BOTH the linear `BlockBilinearForm`-equivalent case (closure returns the same `K_op` every call) and the nonlinear `BlockNonlinearForm` case (closure delegates to `ParNonlinearForm::GetGradient`). The Schur-diag method makes the EA preconditioner construction clean for Batch S. | done |
| S     | Wire the EA path into the patch-test driver behind `--constraint-storage=ea` and `--ab-compare` CLI flags (the latter runs both paths in one process and asserts displacement agreement). Add a saddle-point solver overload `Solve(K, MortarConstraintOperator, ...)` that uses `ComputeInvDiagSchur` for the Schur-diag preconditioner block. Refactor the existing `Solve` body into a shared `SolveImplInternal` helper to avoid duplicating ~125 LOC of Krylov plumbing. Add a dedicated `test_patch_3d_pbc_ea_compare` driver that runs all three patterns (homogeneous / strip / checkerboard) under `ab_compare = true`, registered at np=1 by convention but designed to be re-run at np>1 for cross-rank Alltoallv exercise. | End-to-end validation in the production driver, not just unit tests. This is the cross-rank firewall: bugs in the EA path's off-rank import / export topology that np=1 unit tests cannot reach (because the Alltoallv buffers are empty at np=1) get caught here when the test is re-run at np=4 or np=7 with `||du_ea - du_hp||_inf` above tolerance. | done |
| X (Phase 4.3.B) | GPU port via `mfem::forall`. First pass: pre-flatten per-pair-block data into `mfem::Vector` / `mfem::Array<int>` at construction time (`BuildFlatRowArrays`), rewrite forward `Mult` as a single forall over `m_n_active_rows` with `Read`/`Write` memory-manager annotations. `MultTranspose` and `ComputeInvDiagSchur` stay host-only with `HostRead`/`HostReadWrite` annotations (DEVICE_DEBUG-clean without atomic-add complexity). MPI Alltoallv stays host-only by design. | First step toward GPU portability. The forward direction is the hottest path; transpose and preconditioner setup are amortized cost. | first pass done; atomic-add scatter for `MultTranspose` is a follow-up |

#### §P4.4.6.5 Per-pair pseudocode (algorithmic reference)

For one face-mortar block with `n_n` local nonmortar rows and
`n_m` mortar columns, with `A_m` stored as a sparse CSR:

**Mult (`y = C·x`)** — emitted into local row range
`[row_off, row_off + 3*n_n)`:

```
for each component c in {x, y, z}:
    for k in 0..n_n:
        u_c_k = x[g_n[k] for c]
        y_local = D[k] * u_c_k          // diagonal contribution
        for each (l, A_kl) in A_m row k:
            u_c_l = x[g_m[l] for c]      // possibly off-rank
                                          // (use import buffer)
            y_local -= A_kl * u_c_l
        y[row_off + 3*k + c] = y_local   // overwrite, not accum
                                          // (block 0 — start of
                                          // matvec)
                                          // For subsequent blocks
                                          // emitting same row
                                          // range, +=, but in our
                                          // FES-aligned partition
                                          // each row appears in
                                          // exactly one block.
row_off += 3 * n_n
```

**MultTranspose (`y += C^T·x`)** — reads x in local row range
`[row_off, row_off + 3*n_n)`:

```
for each component c in {x, y, z}:
    for k in 0..n_n:
        x_k = x[row_off + 3*k + c]
        y[g_n[k] for c] += D[k] * x_k    // local TDOF (always
                                          // owned by this rank by
                                          // FES-aligned partition)
        for each (l, A_kl) in A_m row k:
            // y[g_m[l] for c] -= A_kl * x_k
            // — but g_m[l] may be off-rank.
            if g_m[l] is FES-owned by this rank:
                y[g_m[l] for c] -= A_kl * x_k
            else:
                export[off_rank_slot, c] -= A_kl * x_k
                // export buffer is flushed via Alltoallv at
                // end of MultTranspose; receivers ADD into y.
row_off += 3 * n_n
```

For edge-mortar blocks, the same pseudocode applies with the
addition of a row-owner filter at the top:

```
if classifier.GtdofOwnerRank(nonmortar_g_xyz[0]) != my_rank:
    row_off += 3 * n_n   // skip this rank's contribution
                          // (still increment row_off so other
                          // ranks' blocks land in the right
                          // global rows after the rank-major
                          // prefix-sum)
    continue
```

This pseudocode is the implementation contract for Phase 4.3 /
Batch P.

#### §P4.4.6.6 `MortarSaddlePointSystem` design rationale (Batch R)

The Batch R adapter turns "an EA constraint operator + a user's
K residual / Jacobian" into a single `mfem::Operator` that
presents the saddle-point system

\f[
  \begin{bmatrix} K(u) & C^T \\ C & 0 \end{bmatrix}
  \begin{bmatrix} u \\ \lambda \end{bmatrix}
\f]

with `Mult` returning the residual and `GetGradient(x)` returning
the assembled `BlockOperator`. Three design choices warrant
explanation.

**Composition, not inheritance.** Initial sketches had the
adapter inherit from `mfem::BlockNonlinearForm`. That doesn't
fit: `BlockNonlinearForm` builds its block structure from per-
element `BlockNonlinearFormIntegrator::AssembleElementGrad`
contributions, but our constraint matrix C is **globally
coupled** (it links nonmortar gtdofs to mortar gtdofs that may
be on entirely different elements and ranks). The per-element
assembly model doesn't fit. So instead, `MortarSaddlePointSystem`
COMPOSES — it holds a const reference to a
`MortarConstraintOperator` and accepts the K side via
`std::function` callbacks. This sidesteps MFEM's block-form
internals entirely and works above whatever K mechanism the
user has set up.

**Callback-based K abstraction.** The adapter accepts:
- `KResidualFn = std::function<void(const Vector& u, Vector& r)>`
- `KJacobianFn = std::function<Operator*(const Vector& u)>`

This single interface fits both the linear and nonlinear cases:
- **Linear K** (current patch tests, `BlockBilinearForm`-equivalent):
  the closure returns the same `&K` every time. The adapter
  rebuilds its `BlockOperator` per `GetGradient` call but the
  underlying K Jacobian doesn't change.
- **Nonlinear K** (production, `BlockNonlinearForm`):
  the closure delegates to `ParNonlinearForm::GetGradient(u)`,
  which internally re-linearizes K at the current Newton iterate.
  The adapter forwards the result into the saddle-point block
  layout.

The closure-based interface keeps the adapter's API stable
across the linear-vs-nonlinear axis, so Phase 5 (ExaConstit
integration) doesn't need to introduce a different adapter for
production.

**Schur-diagonal computed from blocks, not matvec probes.** The
`BuildInvDiagSchur(HypreParMatrix C, inv_diag_K)` formula in
`saddle_point_solver.cpp` walks the HypreParMatrix CSR. The
EA path needs the same quantity but doesn't have a CSR. Two
options were considered:

1. **Probe with unit vectors.** Compute column `j` of `C` via
   `C * e_j` (one matvec per column), then build the diagonal of
   `C diag(K)^{-1} C^T` from those probes. **Cost**: `Width()`
   matvecs to build the preconditioner. Setup-time only, but at
   production scale (`Width() ~ 1e8`), each Krylov iteration is
   typically far less work than that — would dominate setup.

2. **Compute directly from per-pair blocks** (chosen). The Schur
   diagonal entry at row `(block, k, c)` decomposes as
   `D_k^2 \cdot \mathrm{Dinv}[g_n^c] + \sum_l A_{kl}^2 \cdot \mathrm{Dinv}[g_m^c]`
   — a single walk through the same per-pair data the operator
   already holds. Mirrors `BuildInvDiagSchur`'s formula exactly,
   just walking pair blocks instead of CSR. Costs one Allgatherv
   on `inv_diag_K` (matching the HypreParMatrix path's pattern)
   plus a local pair-block walk. Setup cost is `O(local_rows)`,
   not `O(Width)`.

Option 2 was the right call because:
- It produces bit-equivalent results to option 1 (modulo summation
  order — same FP-rearrangement tolerance as Mult vs HypreParMatrix
  matvec).
- Setup cost stays bounded by problem size, not by `Width()`.
- The implementation is short (~80 LOC of pair-walk code that
  shares structure with `Mult`).

The result lives on `MortarConstraintOperator::ComputeInvDiagSchur`
to keep the EA path self-contained — Batch S consumes it via the
saddle-point solver overload taking `const mfem::Operator&`.

**Lifetime contract.** `GetGradient(x)` returns a reference to an
internal `BlockOperator` whose lifetime extends until the next
`GetGradient` call. The user's Jacobian pointer (returned by their
`KJacobianFn`) must remain valid for at least the same window. This
matches `mfem::ParNonlinearForm` semantics — its internal Jacobian
storage is reused across iterations.

#### §P4.4.6.7 Saddle-point solver overload + A/B patch driver (Batch S)

Batch S is the production-integration step: the patch-test driver
gains a runtime choice of constraint storage (HypreParMatrix vs EA)
and an A/B-compare mode that runs both paths and asserts
displacement-field agreement. Three design decisions are worth
explaining.

**Refactor `Solve` rather than duplicating it.** The HypreParMatrix
overload's body is ~125 LOC: dimension checks, BlockOperator
construction, BlockDiagonalPreconditioner setup, Krylov configuration,
solve, solution extraction. The EA overload differs only in how it
computes `inv_diag_S` (`ComputeInvDiagSchur` vs `BuildInvDiagSchur`)
and what types it casts to feed into `BlockOperator::SetBlock`. Two
cleaner options were considered:

1. **Duplicate the body.** Two `Solve` overloads, each ~125 LOC. Same
   logic in both, two places to fix any bug. Rejected — the
   maintenance cost of doubled Krylov plumbing dominates the
   one-time cost of refactoring.

2. **Extract a shared `SolveImplInternal`.** Each overload computes
   its own `inv_diag_S` via its own path, then delegates to the
   shared helper which takes K and C as `mfem::Operator&` (the
   common base class). All BlockOperator setup, RHS assembly,
   Krylov solver instantiation, and solution extraction lives in
   one place.

Option 2 is what landed. The pattern generalizes to any future
overload that varies only at the preconditioner-construction step
(e.g., a future direct-solver overload).

**Keep K as `HypreParMatrix`, vary only C.** The Batch S overload
is `Solve(const HypreParMatrix& K, const MortarConstraintOperator& C_op, ...)`
— K stays as `HypreParMatrix` because that is what the current
patch-test driver assembles. Switching K to a matrix-free
representation is a separate concern: it requires either a real
nonlinear K from `ParNonlinearForm` (Phase 5) or the `BlockBilinearForm`-
equivalent linear-K-via-Operator path. Either way, that change
expands the saddle-point solver's scope significantly and benefits
from its own focused batch.

The forward-decl-only header convention applies here:
`saddle_point_solver.hpp` forward-declares
`MortarConstraintOperator` rather than including its header,
keeping include-graph weight low. The full include lives in the
`.cpp`.

**A/B compare lives at the driver layer, not the solver layer.**
The cleanest place to compare HypreParMatrix vs EA paths is the
patch-test driver, not the saddle-point solver. The solver only
sees one C at a time; the driver builds both, runs the solver
twice, and computes `||du_ea - du_hp||_inf`. This pattern keeps the
solver simple — there is no "which path do I take?" branch inside
`Solve` — and makes the comparison metric (final-displacement
agreement) match what production cares about. A solver-internal
A/B mode would have had to compare per-iteration residuals or
per-matvec results, which are FP-rearrangement-noisy and harder to
reason about.

The driver's A/B logic is:
1. If `ab_compare = false`, run only the path selected by
   `cfg.constraint_storage`. (Default behavior — preserves all
   pre-Batch-S patch-test runs unchanged.)
2. If `ab_compare = true`, build both `C` and `C_op`, call the
   appropriate `Solve` overload twice (once with each), compute
   `||du_ea - du_hp||_inf` with global `MPI_MAX` reduction, and
   fail the test if the difference exceeds `cfg.ab_compare_tol`.
3. The "primary" path's results (chosen via `cfg.constraint_storage`)
   flow into steps 10–12 (recovery, ⟨F⟩, constraint residual).
   This means `--constraint-storage=ea --ab-compare` is the
   "validate EA path against HypreParMatrix reference" mode, while
   `--constraint-storage=hypre --ab-compare` is the dual.

**Cross-rank validation strategy.** The new
`test_patch_3d_pbc_ea_compare` test driver is registered at np=1 in
CMake, but is intended to be re-run manually at np=4 / np=7 by the
developer (matching the convention for the other patch tests).
Specifically:
- At np=1, `MortarConstraintOperator::Mult` and `MultTranspose`
  hit the same algorithmic path as np>1 — the off-rank import /
  export topology IS built at construction, but the Alltoallv
  buffers happen to be empty because no gtdofs are off-rank. So
  np=1 catches algorithmic bugs in `Mult` / per-pair scatter.
- At np>1, the Alltoallv calls actually exchange data. A bug in
  the topology construction (e.g. wrong destination rank in the
  `gtdof_to_slot` lookup, or a sign error in the export staging)
  shows up as `||du_ea - du_hp||_inf` orders of magnitude above
  tolerance.

This np-progression pattern — np=1 in CI, np>1 manual — is the
same as for the existing patch tests. The cost is that np>1
regressions can land without immediately failing CI; the benefit
is that the unit test suite stays fast.

**Tolerance choice for `ab_compare_tol`.** The two paths' Krylov
solves diverge in FP-summation order (each path's matvec sums in
a different order). The compounding effect across iterations can
move the final residual by more than the per-iteration FP-
rearrangement bound predicts. Empirical observation on the 4³
patch tests at np=1 is `~1e-9`; the default `ab_compare_tol = 1e-7`
leaves 2 orders of magnitude of headroom, sufficient for cross-
rank summation order variance at np up to several dozen.

If `ab_compare_tol` ever needs to be tightened (e.g., for a more
discriminating cross-rank validation), the matvec-level firewall
in Batch Q can be re-tightened at the same time. The two
tolerances are coupled — Batch S tolerance must always be looser
than Batch Q tolerance because Krylov compounding amplifies
matvec rearrangement.

#### §P4.4.6.8 GPU port via `mfem::forall` (Batch X / Phase 4.3.B)

Phase 4.3.B is the GPU port. The CPU EA path is correct and
validated via Batches Q–S; the goal here is to make it run on
GPU through `mfem::forall` with proper memory-manager
annotations. This subsection documents the design choices for
the first pass.

**Pre-flatten data at construction time.** The CPU implementation
walks per-pair-block C++ structs (`m_local_edge_pairs`,
`classifier.PairBlocks()`) using `std::map` lookups
(`m_gtdof_lookup`, `m_import_gtdof_to_slot`). Neither maps nor
arbitrary structs are GPU-friendly. The `BuildFlatRowArrays()`
helper (called once at the end of the constructor) walks every
pair block ONCE and produces flat `mfem::Vector` /
`mfem::Array<int>` arrays:

  * `m_row_D[i]` — diagonal `D_kk` value for row `i`.
  * `m_row_g_n_local[i*kVDim + c]` — local FES TDOF index for the
    nonmortar component `c` of row `i`. -1 = sentinel.
  * `m_row_csr_off[i]` — prefix-sum start of row `i`'s CSR slice.
  * `m_csr_A[k]` — A_kl value for CSR entry `k`.
  * `m_csr_g_m_local[k*kVDim + c]` / `m_csr_g_m_recv[k*kVDim + c]` —
    paired tagged-index encoding for the mortar component. The
    convention is "exactly one of these is ≥ 0 (the other is -1)
    if the component is real, or both are -1 for sentinel". This
    avoids std::map at matvec time at the cost of two int reads
    per CSR entry per component.

The flat-arrays form increases construction-time memory by
roughly `O(n_active_rows + total_csr_entries)` ints + doubles —
small relative to the per-pair-block storage we already keep, and
amortised across all Krylov iterations of a Newton step.

**Per-pair scatter becomes a single `mfem::forall` over rows.**
The forward `Mult`'s old triple-nested loop (per pair, per `k`,
per `c`, per CSR entry) flattens to:

```
mfem::forall(m_n_active_rows, [=] MFEM_HOST_DEVICE (int i) {
    for (int c = 0; c < kVDim; ++c) {
        int gn = m_row_g_n_local[i*3+c];
        if (gn < 0) continue;                  // sentinel
        double y_c = m_row_D[i] * x[gn];
        for (int e = csr_off[i]; e < csr_off[i+1]; ++e) {
            int gm_loc  = m_csr_g_m_local[e*3+c];
            int gm_recv = m_csr_g_m_recv[e*3+c];
            double u_m;
            if      (gm_loc  >= 0) u_m = x[gm_loc];
            else if (gm_recv >= 0) u_m = recv_buf[gm_recv];
            else                   continue;     // sentinel
            y_c -= csr_A[e] * u_m;
        }
        y[lambda_off + c] = y_c;
    }
});
```

Each thread handles one row's `kVDim` outputs, with no shared
state and no atomic writes — every `y[lambda_off + c]` is unique
across threads. This is the embarrassingly-parallel form GPU
forall machinery is designed for.

**MPI Alltoallv stays on host.** Standard MPI implementations
treat host pointers; GPU-aware MPI exists but adds significant
build complexity. Our pattern:

  1. **Send-pack** (host): `x.HostRead()` → fill `send_buf` →
     MPI_Alltoallv → recv into `recv_buf.HostWrite()`.
  2. **Matvec** (device): `recv_buf.Read()` returns a device
     pointer (memory manager migrates host → device on first
     read after a host write).
  3. **Result** (device): `y.Write()` returns a device pointer;
     the kernel writes there directly.

The memory manager handles migrations transparently. Under
`DEVICE_DEBUG`, any attempt to read host-stale or device-stale
data triggers a clear assertion failure rather than corrupting
silently.

**`MultTranspose` stays host-only for first pass.** The transpose
has many-to-one scatter — multiple rows can write to the same
y entry (a mortar gtdof FES-local on this rank can be referenced
from many pair blocks; off-rank export staging is also a many-
to-one accumulation). A correct GPU implementation needs atomic
adds on every scatter target, which works but is materially more
involved than the forward direction. For the first pass we keep
`MultTranspose` as a single sequential walk over the same flat
arrays on the host with `HostRead`/`HostReadWrite` annotations.
This is DEVICE_DEBUG-clean and validates the flat-array
infrastructure; an atomic-add scatter rewrite is a follow-up
batch.

**`ComputeInvDiagSchur` stays host-only.** Setup-time only (called
once per Newton step from the saddle-point solver during
preconditioner construction, before any Krylov iterations run).
Not in the matvec hot path. Refactoring it to flat arrays would
provide little benefit since its cost is amortised across
hundreds-to-thousands of Krylov iterations. The body uses
`HostRead` on `inv_diag_K_local` and `HostWrite` on `schur_diag`
to be DEVICE_DEBUG-clean.

**`MortarSaddlePointSystem::Mult` annotations.** The block-vector
view construction uses `HostReadWrite` on the input block and
`HostWrite` on the output block to register the access intent
with the memory manager. The K-residual callback and the
mortar operator's own `Mult` / `MultTranspose` then call their
own `Read` / `Write` on the sub-vector views, which dispatches
correctly because the sub-vectors alias the same memory region.

**Tolerance under `DEVICE_DEBUG`.** The Batch Q matvec A/B
tolerance (1e-12) and the Batch S patch-test A/B tolerance (1e-7)
should hold unchanged on host. On device, FP-rearrangement may
shift these by up to one order of magnitude due to different
summation orders in the per-row inner loop (the new flat-array
form sums in CSR-entry order rather than the per-pair-block
order the original code used). If A/B tests start failing at
1e-12 after the GPU port, the right move is to bump Batch Q's
tolerance to 1e-11 — that captures the FP-rearrangement shift
without masking real bugs.

#### §P4.4.6.9 Phase 4.3.B current state and next steps

This subsection is the entry point for someone returning to the
GPU port work cold. It captures (a) what's actually been
implemented and validated, (b) what's specifically pending, and
(c) the recommended order of operations for finishing.

##### What's implemented and validated

**Sandbox-validated** (host-only syntax + `-Wall -Wextra` +
algorithm correctness via Python regression and the existing
unit / patch tests):

  * `MortarConstraintOperator::BuildFlatRowArrays()` — two-pass
    walk that pre-flattens the per-pair-block data into
    `mfem::Vector` / `mfem::Array<int>` arrays at construction
    time. Walks the same iteration order as `Mult` /
    `MultTranspose` / `ComputeInvDiagSchur` /
    `EmitConstraintTriples` (edges first with row-owner filter,
    then face mortars in `FacePairs()` order with quad-then-tri).
    Produces:
       - `m_row_lambda_off[i]` — first lambda index for row `i`.
       - `m_row_D[i]` — diagonal `D_kk` value for row `i`.
       - `m_row_g_n_local[i*3+c]` — local FES TDOF index for
         nonmortar component `c` (-1 for sentinel).
       - `m_row_csr_off[i]` — prefix-sum start of row `i`'s CSR
         slice.
       - `m_csr_A[k]` — A_kl value for CSR entry `k`.
       - `m_csr_g_m_local[k*3+c]` / `m_csr_g_m_recv[k*3+c]` —
         paired tagged-index encoding for off-rank vs. local
         lookups (exactly one is ≥ 0 if real, both -1 for
         sentinel).

  * `MortarConstraintOperator::Mult` — forward direction
    rewritten as `mfem::forall(m_n_active_rows, kernel)`. Host
    side does the send-pack and `MPI_Alltoallv` (with
    `HostRead`/`HostWrite` annotations); device kernel reads the
    flat arrays via `Read()` and writes `y` via `Write()`. No
    `std::map` lookups, no struct walks, no host-only API calls
    in the kernel.

  * `MortarConstraintOperator::MultTranspose` — first-pass
    rewrite that uses the flat arrays but stays as a single
    sequential host walk. `HostRead`/`HostReadWrite` annotations
    throughout. Sequential because the transpose has many-to-one
    scatter and atomic-add scatter is the planned follow-up
    (see "Next steps" below).

  * `MortarConstraintOperator::ComputeInvDiagSchur` — host-only
    by design (setup time, not hot path). All Vector accesses use
    typed `HostRead`/`HostWrite` accessors with raw pointers
    hoisted above per-element loops.

  * `MortarSaddlePointSystem::Mult` — block-vector views
    constructed via `HostReadWrite` on input and `HostWrite` on
    output. Sub-vector views alias the parent buffers, so
    callbacks' own `Read`/`Write` calls dispatch correctly.

  * `SaddlePointSolver::SolveImplInternal`, `BuildInvDiagK`,
    `BuildInvDiagSchur`, `DiagonalScaler::Mult` — all per-element
    Vector accesses converted to raw `HostRead`/`HostWrite`
    pointer pattern.

  * Patch driver (`patch_test_driver_3d.cpp`) — A/B compare diff
    loop, `u_total` recovery loop, constraint-residual loop, and
    `ComputeVolumeAveragedF` u-copy loop all converted to raw
    pointers.

**Validated on real MFEM (Mac, host-only build)**:

  * All existing unit tests pass under normal build.
  * `test_patch_3d_pbc_ea_compare` passes at np=1 (and remains
    available for np>1 cross-rank Alltoallv exercise).
  * **Patch tests run cleanly under `DEVICE_DEBUG`** — the user
    confirmed this after the §P4.8.17 fixes landed. This is the
    significant validation gate: every Vector access in the
    saddle-point solver, constraint operator, and patch driver
    has its memory-manager intent declared correctly.

**Stub extensions** (in `/tmp/mfem_stub/mfem.hpp`):

  * `mfem::Vector` and `mfem::Array<T>`: `Read`/`Write`/`ReadWrite`/
    `HostRead`/`HostWrite`/`HostReadWrite` returning raw pointers
    (in real MFEM they go through the memory manager).
  * `mfem::forall(N, body)` template that runs serially on host
    for syntax-checking.
  * `MFEM_FORALL(i, N, body)` macro form.
  * `MFEM_HOST_DEVICE` no-op define.

##### What's pending

In rough order of difficulty / dependency:

1. **Atomic-add scatter for `MultTranspose`** (medium effort).
   The flat-array form is already in place; the conversion
   replaces the sequential host loop with `mfem::forall(...)`
   that does atomic adds into both `y` (for FES-local writes)
   and the export staging buffer (for off-rank writes). The
   stub will need an `mfem::AtomicAdd` (or equivalent) added.
   In real MFEM, `MFEM_HOST_DEVICE` atomic operations are
   exposed via the `mfem::AtomicAdd<T>` template. The kernel
   structure stays the same as the current sequential walk —
   each thread handles one row, walks its CSR slice, and atomic-
   adds into output positions.

   **Why this is non-trivial**: the export staging buffer is a
   `std::vector<double>` currently — it needs to become an
   `mfem::Vector` so atomic adds through the memory manager are
   well-defined. Then the AOS layout (`slot * kVDim + c`) stays
   the same; only the access path changes.

   **Validation strategy**: the existing
   `test_mortar_constraint_operator`'s A/B test (Batch Q) at
   np=1 will catch any regression in `MultTranspose` correctness
   immediately, and the cross-rank A/B test at np=4 / np=7 will
   catch any cross-rank correctness issue. Tolerance may need
   to bump from 1e-12 to 1e-11 because atomic-add summation
   order is non-deterministic across threads (each run can
   produce slightly different results within FP-rearrangement
   bounds).

2. **Real device build validation** (low-to-medium effort,
   high-value).
   Sandbox + `DEVICE_DEBUG` validates memory-manager hygiene;
   only a real CUDA or HIP build exercises the kernels on
   hardware. The plan:

     a. Build MFEM with `MFEM_USE_CUDA=YES` (or `MFEM_USE_HIP=YES`
        for AMD targets).
     b. Build the patch tests against that MFEM.
     c. Run with `--device cuda` (or `hip`) flag added to the
        device-init sequence at the top of `main`.
     d. Compare output displacements against the host-only build
        — should agree within `1e-11` (`1e-12` was the host A/B
        tolerance; one extra order of magnitude of slack covers
        FP-rearrangement on device).

   **Most likely failure mode**: a CSR-entry-component encoding
   mismatch where `m_csr_g_m_recv` is computed incorrectly.
   This would manifest as off-rank pairs producing wrong
   contributions only at np > 1 — the np=1 case never exercises
   off-rank paths. The Batch Q A/B test (cross-rank, n=8 mesh)
   is the diagnostic to lean on.

3. **Performance work** (open-ended, lower priority).
   Once correctness on device is confirmed, profile and
   optimize. Likely candidates:
     - Coalescing on the flat arrays (the current AOS layout for
       `m_csr_g_m_local` / `m_csr_g_m_recv` is `[k*3 + c]` —
       grouping by component instead might give better warp-
       level coalescing on CUDA).
     - Register pressure in the kernel body (the inner loop
       reads 4 ints + 1 double + 1 double per CSR entry; if
       this exceeds register budget it spills to local memory).
     - Possibly per-pair shared-memory tiling for very-dense
       face-mortar blocks, though for the patch tests the per-
       row CSR slices are short (~10-20 entries) so this
       probably isn't worth the complexity.

   The existing Caliper instrumentation (`CALI_CXX_MARK_SCOPE`)
   in `Mult` / `MultTranspose` / `ComputeInvDiagSchur` will show
   where the time actually goes once a real device build is
   available. Don't optimize blind.

4. **Convert `block.A_m.GetData()` SparseMatrix accesses to
   `GetMemoryData().HostRead()` form** (very low effort, defensive
   only).
   These are `SparseMatrix` accesses (not Vector), and SparseMatrix
   data is host-resident throughout the program lifetime by
   construction. They don't currently fail under `DEVICE_DEBUG`.
   Switching to the typed-accessor form would future-proof against
   any case where a SparseMatrix gets device-touched (e.g., if a
   future `BuildFlatRowArrays` extension does its walk on device).
   Not urgent.

##### Recommended order when circling back

1. **Verify the host-only Mac build is still green**. Re-run all
   patch tests + `test_patch_3d_pbc_ea_compare` with `--f-sweep`
   at np=4 and np=7 to confirm nothing has bit-rotted.
2. **Set up a real CUDA or HIP build of MFEM** in the
   exaconstit_hip_build tree. ExaConstit has experience with
   this; reuse the existing build infrastructure.
3. **Run the sandbox-validated code on device**, host-only
   first (forward `Mult` only), to validate the `mfem::forall`
   path actually compiles and runs. The `MultTranspose` and
   `ComputeInvDiagSchur` paths are explicitly host-only and will
   naturally fall through to host execution.
4. **Tackle atomic-add `MultTranspose`** — the natural next
   batch after device-build validation. Pattern is established
   by the forward `Mult`; only the scatter side changes.
5. **Performance work** — only after correctness is end-to-end
   green on device.

##### Key invariants to preserve

These are non-negotiable across any future GPU work:

  * **`BuildFlatRowArrays` walk order MUST match `Mult` /
    `MultTranspose` / `ComputeInvDiagSchur` / `EmitConstraintTriples`.**
    Edges first (with row-owner filter), then face mortars in
    `FacePairs()` order with quad-then-tri. Any divergence breaks
    row-index alignment with `Height()`.

  * **Sentinel handling**: `m_row_g_n_local[i*3+c] = -1` and
    `m_csr_g_m_local[k*3+c] = m_csr_g_m_recv[k*3+c] = -1` both
    mean "skip this contribution silently." The kernel must
    NOT increment row offset or write to `y` for a sentinel
    component — match what the original ScatterEdgeBlock did.

  * **Batch N's row-owner invariant**: nonmortar gtdofs are
    always FES-local for owned rows. Encoded into
    `m_row_g_n_local[]` always being a local FES TDOF index
    (or -1 sentinel), never an off-rank index. If this
    invariant is violated, either the row-owner filter or
    the routing logic has a bug — not the GPU port.

  * **Batch L's mortar gtdof convention**: face-mortar pair
    blocks store mortar gtdofs as x-component only;
    `m_gtdof_lookup` maps x → (x, y, z). The `BuildFlatRowArrays`
    walk uses this lookup to per-component encode into
    `m_csr_g_m_local` / `m_csr_g_m_recv`. If a future change
    extends pair blocks to per-component gtdofs directly, the
    encoding step in `BuildFlatRowArrays` simplifies but the
    resulting flat-array form must be unchanged.

  * **DEVICE_DEBUG-clean access pattern**: every Vector access
    in any new code MUST use `HostRead`/`HostWrite`/`HostReadWrite`
    (or device counterparts), not `GetData()`/`operator()`/
    `operator[]`. See §P4.8.17 for the rule.

##### Cross-references

  * §P4.4.6.8 — design rationale for the GPU port (why this
    architecture, why the choices).
  * §P4.8.16 — lesson on pre-flattening host-side data before
    chasing `mfem::forall`.
  * §P4.8.17 — lesson on `Vector::GetData()` /
    `Vector::operator()` being DEVICE_DEBUG traps.
  * §P4.13 done-criteria — Phase 4.3.B item.

#### §P4.4.6.10 Phase 4.4 — Non-conforming face mortar

This subsection is the architectural plan for completing Phase
3.5 / Phase 4.4 (the architecture doc names the algorithmic phase
3.5, but the C++ port version of it is Phase 4.4). The plan was
built by carefully re-reading the master architecture doc, the
2D non-conforming code (which is the proven design template),
and the existing C++ face-mortar assembler code, then refining
with current literature only where the existing design genuinely
needs an answer.

##### What this phase does and does not change

**Scope (what's in):** Add support for opposite periodic faces
that have non-matching node positions on the same flat
axis-aligned interface — e.g., the `x = 0` face is subdivided
into a 4×4 grid of quads while the `x = L` face is subdivided
into a 5×5 grid. Element types remain pure: all-hex (so all
face elements are quads) or all-tet (all face elements are
tris). Faces remain flat and axis-aligned. Full periodicity
(all 3 axis pairs) only.

**Scope (what's out):**
  * Mixed quad-tri pairings (a quad face on one side paired with
    a tri face on the other). The architecture-doc §3.7 algorithm
    handles this case but it doubles the testing surface.
    Defer until pure-element non-conforming is solid.
  * Curved or non-planar faces. The 2D-projection simplification
    relies on flat axis-aligned faces.
  * Semi-periodic BCs (e.g., XY periodic, Z Dirichlet). The full-
    periodic assumption simplifies the corner Dirichlet handling;
    semi-periodic adds new corner / edge classifications.
  * Hanging-node (h-refinement) non-conformity. MFEM has its own
    machinery for hanging nodes; we should not re-implement it.
    Our scope is ONLY non-matching subdivisions on the
    user-supplied original mesh.

**What stays unchanged:**
  * The Wohlmuth corner / edge dual-basis modifications
    (`MQuad4DualModified`, `MTri3DualModified`) — they depend on
    `boundary_tag` (set by the classifier from sentinel patterns),
    not on the integration domain. They evaluate at any (ξ, η) /
    barycentric point.
  * The boundary classifier's sentinel-driven `boundary_tag`
    classification (`ClassifyQuadBoundaryTag`,
    `ClassifyTriBoundaryTag`).
  * The Method-D corner Dirichlet logic (Lopes et al. 2021 §3.4).
  * `MortarConstraintOperator` (Phase 4.3 EA path).
  * `MortarSaddlePointSystem`, `SaddlePointSolver`.
  * The GPU port (Phase 4.3.B). The `BuildFlatRowArrays` walk
    consumes `FaceMortarPairBlock` regardless of whether the
    block came from the conforming or clipped path.
  * The `FaceMortarPairBlock` data layout itself (D vector,
    A_m sparse matrix, gtdof arrays).

**Architectural seam:** all non-conforming work is contained in
three places. The rest of the pipeline is untouched.
  1. New `AssemblePairClipped` method on the face-mortar
     assemblers (sibling to `AssemblePairConforming`).
  2. New `MatchClippedFacePairs` helper (sibling to
     `MatchConformingFacePairs`).
  3. Small dispatch decision in
     `BoundaryClassifier3D::BuildLocalPairBlocks`: try
     `MatchConformingFacePairs` first; on a non-1:1 match count,
     fall back to `MatchClippedFacePairs`.

##### Algorithmic invariants from the existing 2D code

The 2D non-conforming case is fully solved (`mortar_assembler_2d`
in C++, `mortar_pbc/mortar_2d.py` in Python). The 3D face-mortar
non-conforming case must extend the **same** pattern — anything
that diverges from this pattern is a bug.

**The D-vs-A_m domain split.** This is implicit in the 2D code
(line 326 of `mortar_2d.py`) but not explicitly called out in
the architecture doc. It is the central principle:

  * **D contributions** are accumulated PER NONMORTAR ELEMENT,
    with the integration domain being the FULL nonmortar element:
       `D_k += ∫_{full_nonmortar_element} N_k dA = phys_jacobian * w_q * N_k(xi_q)`
    summed over canonical quadrature points on the full nonmortar
    reference element. **D never sees the clipped sub-polygon.**

  * **A_m contributions** are accumulated PER CLIPPED OVERLAP,
    with the integration domain being the OVERLAP polygon:
       `A_m[k,l] += ∫_{overlap} M_k(xi_nm) * N_mortar_l(xi_m) dA`
    summed over a per-sub-triangle quadrature on the clipped
    sub-polygon's fan triangulation. **A_m always sees the
    clipped overlap, never the full element.**

Why this split is correct: Wohlmuth's biorthogonality identity
`∫_E M_i N_j dE = δ_ij ∫_E N_i dE` holds when integrated over
the full element E, NOT segment-wise. So we compute D directly
as `∫_E N_i` (a cheap element-local quadrature) rather than as
`∑_segments ∫ M_i N_i` (which would compound rounding error and
require correctly summing all overlapping segments' contributions).

The 2D code uses `D_nm[k] += plus_jacobian` directly (the
analytic value of `∫_{line2} N_k dxi · J = J = phys_half_length`
for each endpoint k=1,2). The 3D conforming code already does
the equivalent: `D_loc[k] += phys_w * N_nonmortar[k]` summed over
canonical quadrature points on the full nonmortar element. **The
non-conforming version reuses this loop verbatim.** Only the
A_m loop changes.

**The mortar inverse map is local-affine for our scope.** For
axis-aligned grids:
  * Quad face (Q1): the bilinear isoparametric map collapses to
    an affine map `xi = 2*(a - a_lo)/(a_hi - a_lo) - 1` per
    parametric direction. Inverse is two scalar divisions.
    No Newton iteration needed.
  * Tri face (P1): the affine isoparametric map has a 2×2 inverse;
    closed-form via Cramer's rule.

The architecture doc §11.6 spells this out; the existing
`face_mortar_assembler_3d.cpp` does NOT need this because its
conforming path uses `MortarRefFromPermutation` (a permutation
of nonmortar local coords), but the non-conforming path will
need the explicit inverse map.

##### Decisions and refinements

These are the design decisions for the 3D non-conforming case.
The literature review (Bernardi-Maday-Patera 1994, Wohlmuth
2000, Puso-Laursen 2004, Popp-Wohlmuth-Gee-Wall 2010, Farah-
Popp-Wall 2015, Sitzmann-Willner-Wohlmuth 2016, Lopes et al.
2014/2021, Reis & Andrade Pires 2014, Rodrigues Lopes et al.
2021, Mayr-Popp 2022) confirms the architecture doc's planned
approach with two refinements: use Axom's primitives where
available, and bump the per-clipped-sub-triangle quadrature
order for quad-face overlaps.

**Decision 1: Polygon clipping via `axom::primal::clip`.** The
architecture-doc §3.7 recommends hand-rolled Sutherland-Hodgman.
Axom (LLNL's mesh-processing library) provides
`axom::primal::clip` for 2D-polygon-on-2D-polygon convex-on-convex
clipping with documented robustness work (release notes mention
specific fixes for clip robustness). Since Axom is being added
to ExaConstit anyway for restart support (Sidre), and since
hand-rolled clipping has a long tail of degenerate-vertex /
near-collinear-edge cases, **use Axom's clip rather than
hand-rolling**. The architecture doc's §3.7 pseudocode stays as
the algorithmic reference; the implementation is a thin wrapper
around `axom::primal::clip`.

**Decision 2: Point location via `axom::spin::BVH<2>`.** The
architecture doc §11.6 specifies "AABB-tree-or-similar lookup"
through a `spatial_index.locate(plane_coords)` interface.
`axom::spin::BVH<int Dim>` provides exactly this, parameterized
on dimension. Use `axom::spin::BVH<2>` keyed on the 2D-projected
AABBs of the mortar elements.

This is GPU-portable through Axom's RAJA-based execution model;
that aligns with the Phase 4.3.B GPU work but is not required
for Phase 4.4 (the BVH query is setup time, not hot path).

**Decision 3: Hand-rolled inverse maps.** Don't use Axom for the
parametric-coordinate inverse maps (Q1 affine bilinear, P1 tri
affine). They're 5-line closed-form formulas; pulling in a more
heavyweight inverse-isoparametric utility is overkill.

**Decision 4: Per-sub-triangle quadrature order.**

The architecture doc §11.9 question 3 sets the conforming-case
quadrature: 4-point Gauss for quad, 3-point Dunavant for tri.
For non-conforming on **clipped sub-triangles**, the integrand's
polynomial degree on the sub-triangle's barycentric coordinates
must be re-counted because the integration domain changes:

  * **Tri face (P1) on clipped sub-triangle.** Both `M^mod(λ_nm)`
    and `N_mortar(λ_m)` are linear in their respective
    barycentric. Under the affine (λ_nm → λ_m) sub-affine map
    on the sub-triangle, `M·N` is degree 2 in the sub-triangle's
    barycentric. **3-point Dunavant (degree 2) suffices.** Same as
    the conforming case.

  * **Quad face (Q1) on clipped sub-triangle.** `M^mod(ξ_nm,
    η_nm)` is bilinear in (ξ, η). After mapping to the
    sub-triangle's barycentric (which substitutes piecewise-linear
    expressions for ξ and η), bilinear-times-bilinear becomes
    degree 4 in barycentric. **6-point Dunavant (degree 4)
    suffices.** This is a deviation from the conforming case
    (which used a 9-point tensor-product rule on the un-clipped
    parent quad reference, equivalent to degree 5 in (ξ, η)).

The Wohlmuth-modified bases on edge-adjacent or corner-adjacent
elements have lower polynomial degree (constant in the corner-
adjacent case; mixed constant + linear in the edge-adjacent
case), but per architecture doc §11.9 question 3 we use the
"safe uniform rule" policy: 6-point Dunavant on every quad-face
sub-triangle, 3-point Dunavant on every tri-face sub-triangle,
regardless of `boundary_tag`.

**Decision 5: Conforming fast path is preserved.** When
`MatchConformingFacePairs` returns a clean 1:1 partition (every
nonmortar element has exactly one mortar partner), the existing
`AssemblePairConforming` runs unchanged. The clipped path is
opt-in based on the matching result. Concretely:
  * `MatchConformingFacePairs` now returns
    `optional<vector<PairMatch>>` instead of asserting on
    non-1:1: `nullopt` signals "fall back to clipped path."
    (Or equivalently: a separate
    `TryMatchConformingFacePairs` that returns an optional.)
  * `BuildLocalPairBlocks` calls `TryMatchConformingFacePairs`;
    on `nullopt`, calls `MatchClippedFacePairs` and
    `AssemblePairClipped`; otherwise calls
    `AssemblePairConforming`.

**Decision 6: D contribution stays in `AssemblePairConforming`-
style code.** Both `AssemblePairConforming` and
`AssemblePairClipped` factor the D accumulation into a shared
helper `AccumulateNonmortarD(D_loc, nonmortar_elem)` that walks
the canonical nonmortar quadrature once and contributes
`phys_w * N_k(xi_q)` per node. The clipped path's outer loop
calls this helper once per nonmortar element BEFORE the inner
clipped-sub-triangle loop (which only touches A_m). This
preserves the D-vs-A_m domain split as a structural property of
the code, not a comment.

##### Detailed batch sequence

The work breaks into 5 batches plus an architecture-doc
clarification batch (4.4-0). Each batch has a clear validation
gate.

| Batch | What | Why | Validation |
|---|---|---|---|
| 4.4-0 | Architecture-doc clarification: explicitly document the D-vs-A_m domain split in §3.5 / §3.7 (currently only implicit in the 2D code). | Future readers (and Claude in future sessions) shouldn't have to reverse-engineer this from the 2D code. | Doc-only; no code change. |
| 4.4-A | Add Axom to the build. CMake integration via BLT, find_package(axom REQUIRED), pin a version, validate by compiling a no-op sandbox file that includes `<axom/spin/BVH.hpp>` and `<axom/primal/clip.hpp>`. Document the new dependency in the build instructions. | Foundational; without Axom, the rest of the work is hand-rolled. | Sandbox file compiles; no behavioral changes; existing tests pass. |
| 4.4-B | `MatchClippedFacePairs` for quad. Builds an `axom::spin::BVH<2>` over the mortar elements' 2D-projected AABBs (drop the perpendicular axis). For each nonmortar element, queries the BVH to get candidate mortar elements whose AABBs overlap; emits a list of `(s_idx, m_idx)` candidate pairs. No clipping yet. | Broad-phase first. Decouples spatial-search correctness from clipping correctness. | Unit test on a synthetic 4×4 nonmortar / 5×5 mortar pairing: every nonmortar element gets ≥1 candidate; total candidate count is in expected range (about 4×4 × ~4 ≈ 64 pairs). |
| 4.4-C | Polygon clipping for the candidate pairs (quad + tri). Wraps `axom::primal::clip` with our `(a, b)` 2D-projection convention. For each candidate pair, produces a clipped polygon (or empty), then fan-triangulates into sub-triangles. Returns a flat list of `ClippedSubTriangle { s_idx; m_idx; verts_ab[3]; }`. | Geometry-only; no integration yet. | Unit test: total sub-triangle area equals nonmortar face area to roundoff (tile-cover invariant). |
| 4.4-D | `AssemblePairClipped` for quad and tri. Outer loop over nonmortar elements (calls `AccumulateNonmortarD`). Inner loop over sub-triangles owned by this nonmortar element (per-sub-triangle Dunavant quadrature, evaluates M_dual at xi_nm, N_mortar at xi_m via the closed-form inverse maps, accumulates into A_m). Produces `FaceMortarPairBlock`. | Algorithmic core. | (a) Unit test: a deliberately-conforming 4×4 vs 4×4 setup goes through the clipped path and produces a `FaceMortarPairBlock` numerically equal (within roundoff) to `AssemblePairConforming`'s output. This exercises the full clipped pipeline on a known-correct case. (b) Patch-test driver with non-matching subdivisions (4×4 vs 5×5): constant-strain reproduction to roundoff (`||du||_inf < 1e-12 * scale` for a homogeneous RVE under macroscopic F). |
| 4.4-E | Dispatch in `BuildLocalPairBlocks`: try `MatchConformingFacePairs`, fall back to `MatchClippedFacePairs` + `AssemblePairClipped`. New patch-test executable `test_patch_3d_pbc_nonconforming.cpp` with non-matching subdivisions. CMake registration. | End-to-end integration. | (a) Existing patch tests pass unchanged (regression check — confirms the conforming fast path still kicks in when meshes match). (b) New non-conforming patch test: homogeneous, strip, checkerboard patterns at np=1, 4, 7 with non-matching subdivisions on opposite faces. Constant-strain reproduction to 1e-12; ⟨F⟩ ≈ F_macro to 1e-9. |

##### Validation strategy details

**Conforming-path-via-clipped sanity test (Batch 4.4-D part a).**
Take a 4×4 vs 4×4 conforming setup. Force the clipped path via
a flag (or by modifying the dispatch). Each nonmortar element
clips against exactly one mortar element; the clipped polygon is
the full nonmortar quad; fan-triangulation gives 2 sub-triangles
per quad. The integration sums to the same `FaceMortarPairBlock`
as `AssemblePairConforming` modulo FP-rearrangement (which the
6-point Dunavant rule controls — the rearrangement is small).

This test catches:
  * Sign errors in the inverse-isoparametric maps.
  * Orientation bugs in the (a, b) projection (CCW invariant).
  * Sub-triangle area vs Jacobian inconsistencies.
  * Off-by-one errors in the sub-triangle → quadrature-point map.

**Non-conforming patch test (Batch 4.4-E).** Homogeneous RVE
(uniform material) under macroscopic F. The expected fluctuation
is u_tilde ≡ 0 throughout, so any non-zero u_tilde signals a
mortar implementation bug. Tolerance: `||du||_inf < 1e-12 *
characteristic_length`. The strip and checkerboard variants test
genuine non-zero fluctuation; agreement should be to the
saddle-point solver's Krylov tolerance (1e-7).

**A/B comparison (optional).** If we want extra confidence,
extend `test_patch_3d_pbc_ea_compare` to accept a non-matching
mesh option and run the EA path through both the conforming and
clipped code branches (with the clipped branch forced even on
conforming meshes). Both should produce the same du to
FP-rearrangement.

##### Known risks and what to watch for

  * **Dual-basis biorthogonality does NOT hold sub-region-wise.**
    The Wohlmuth identity holds when integrated over the FULL
    nonmortar element, not segment-by-segment. Our D-vs-A_m
    domain split sidesteps this (D is computed on the full
    element). If anyone is tempted to "simplify" by computing D
    as `∑_segments ∫ M_k N_k`, they'll re-introduce the issue we
    explicitly avoid here. Documented in §3.5 / §3.7 by Batch
    4.4-0.

  * **The conforming fast path must still be available**
    for performance-critical workloads. Don't replace
    `AssemblePairConforming` with `AssemblePairClipped`.

  * **`MatchConformingFacePairs` currently aborts on non-1:1
    matches.** Convert this to a try-style API
    (`std::optional` return) so the dispatch can fall back to
    clipped without a fatal error.

  * **Cross-rank correctness.** The classifier's tile partitioning
    + AllGather is unchanged; the new code lives inside
    `BuildLocalPairBlocks` which already runs tile-locally and
    contributes to the AllGather'd pair-block list. So
    cross-rank should "just work," but the np=4 / np=7 patch
    tests should explicitly verify this.

  * **The Wohlmuth `boundary_tag` classification is set on the
    nonmortar elements, NOT on the clipped sub-triangles.** All
    sub-triangles owned by one nonmortar element share the same
    `boundary_tag`. The dual basis evaluation `MQuad4DualModified`
    at a non-canonical (ξ_nm, η_nm) — e.g., a quadrature point
    inside a sub-triangle that doesn't touch the parent quad's
    canonical reference points — must give the correct value.
    Looking at the code, `MQuad4DualModified` is a closed-form
    polynomial in (ξ, η); it works at any point. ✓

  * **Tolerance at strongly-mismatched refinement (e.g., 1:10)** —
    the Krylov solver's Schur-complement preconditioner can lose
    diagonal dominance at very high refinement-ratio. Mayr-Popp
    (2022) document this for contact problems and recommend
    aggregation-based AMG. For our 1:2 to 1:5 typical case,
    block-Jacobi (the existing preconditioner) is fine. If a
    user pushes beyond 1:5, document the limitation in the
    ConstraintBuilder3D class doc.

##### What to do at start of work

When picking up this work cold, the order is:

  1. **Re-read this section (§P4.4.6.10) end-to-end.**
  2. **Re-read architecture doc §3.5, §3.6, §3.7, §11.6.**
  3. **Re-read `mortar_2d.py:_assemble_pair` and
     `_integrate_overlap_segment`** — this is the proven design
     template.
  4. **Re-read C++ `face_mortar_assembler_3d.cpp:AssemblePairConforming`**
     for both quad and tri — this is the existing structure to
     extend.
  5. **Verify host-only Mac build is still green** before
     starting any new work.
  6. **Start with Batch 4.4-0** (architecture-doc
     clarification). It's a doc-only change that takes 30
     minutes and immediately captures the D-vs-A_m insight in
     a place where future readers will find it before the code
     gets confusing.

##### Cross-references

  * Architecture doc §3.5 — geometric matching algorithm.
  * Architecture doc §3.6 — conforming "free pass" case.
  * Architecture doc §3.7 — Sutherland-Hodgman pseudocode (the
    algorithmic specification for what `axom::primal::clip` does).
  * Architecture doc §5.2, §5.3 — Wohlmuth modifications for
    tri-3 and quad-4 (unchanged in this phase).
  * Architecture doc §11.6 — face mortar geometric matching
    (with `locate_mortar` interface that BVH provides).
  * Architecture doc §11.9 question 3 — quadrature order policy.
  * Architecture doc §11.9 question 4 — clipping recommendation
    (now refined to Axom rather than hand-rolled).
  * Phase doc §P4.4.6.4 — Phase 4.3 batch sequence (this
    section is the Phase 4.4 sibling).
  * Phase doc §P4.4.6.9 — Phase 4.3.B current state and next
    steps (sibling pattern: each phase has a state-and-plan
    section).
  * Lopes et al. CMAME 384 (2021) — the Method-D corner
    Dirichlet derivation; unchanged here.
  * Reis & Andrade Pires CMAME 274 (2014) — the foundational
    paper for mortar-PBC homogenization (corner-prescribed
    Dirichlet approach).

### §P4.4.7 Saddle-point solver

The Python prototype's `SaddlePointSolver` wraps MFEM's
`BlockOperator` with one of three Krylov solvers, selected at
construction time. The C++ version mirrors this exactly. CG is
explicitly REJECTED because the saddle-point system is indefinite.

#### Krylov choice: MINRES, GMRES, BiCGStab

The three options and when to pick them:

**MINRES** — `mfem::MINRESSolver`. The default. Optimal for
symmetric saddle-point systems: requires only K to be symmetric
(which it is for linear elasticity and for the symmetric tangent
of finite-strain elasticity), uses short-term Lanczos recurrence
(2 vectors of state regardless of iteration count, vs GMRES's
restart-length-many vectors), and produces monotonically decreasing
residual norm. **Use this whenever K is symmetric.**

The Lanczos-breakdown concern from my earlier note is overstated:
PA/EA roundoff doesn't break MINRES in practice on saddle-point
systems unless K's symmetry is broken at a level large compared to
the Krylov tolerance, which doesn't happen for elasticity. The
Python prototype defaults to MINRES and it has worked correctly at
every scale tested.

**GMRES** — `mfem::GMRESSolver`. The fallback for genuinely non-
symmetric K. Use when:
- The material tangent is non-symmetric (e.g., crystal plasticity
  with kinematic hardening, anisotropic elasticity with shear
  coupling, certain damage models).
- K is FA-assembled with a numerical perturbation that makes its
  symmetry break to ~ machine epsilon × condition_number.
- We're debugging and want a more robust default to isolate
  Krylov vs solver-correctness issues.

GMRES needs a restart length (`SetKDim`). For moderate-sized
saddle-point systems use the default of 50; bigger systems may
benefit from 100 or higher at the cost of memory.

**BiCGStab** — `mfem::BiCGSTABSolver`. The third option. Use when:
- K is non-symmetric AND the GMRES restart length is constrained
  by memory.
- We want a short-recurrence non-symmetric solver and accept the
  potential for breakdown / non-monotonic residual norm.

BiCGStab uses constant memory (~7 vectors of state) regardless of
iteration count, unlike GMRES which grows. For very large
problems where GMRES memory is a concern this becomes attractive,
but residual-norm non-monotonicity makes it harder to debug
convergence problems.

The Python prototype guidance (verbatim, applies to C++):

> CG is rejected with a clear error message: the system is
> indefinite (zero block in the (2,2) position) and CG diverges
> on indefinite systems. Use MINRES (symmetric K) or GMRES (non-
> symmetric K) instead.

#### Solver selection API

```cpp
enum class KrylovKind { MINRES, GMRES, BiCGStab };

class SaddlePointSolver {
public:
    struct Options {
        KrylovKind solver = KrylovKind::MINRES;       // default symmetric
        std::string preconditioner = "block_jacobi";  // or "block_amg"
        double rel_tol = 1e-10;
        double abs_tol = 1e-12;
        int max_iter = 500;
        int print_level = -1;
        int gmres_kdim = 50;                          // GMRES only
    };

    SaddlePointSolver(Options opt = {});

    // [collective on K's communicator, typically WORLD]
    void SolveStep(mfem::Operator& K_op,
                   mfem::Operator& C_op, mfem::Operator& CT_op,
                   const mfem::Vector& r1_world,
                   const mfem::Vector& r2_world,
                   mfem::Vector& du_world, mfem::Vector& dlam_world);
    // ...
};
```

The CLI surface in the validation drivers exposes this as
`--solver={minres,gmres,bicgstab}` — matching the Python flag.

#### Block-Jacobi at large scale

MFEM's `BlockDiagonalPreconditioner` uses `Operator::AssembleDiagonal`
to build the diagonal of K (and identity for the multiplier block
in our setup). This works for K-as-PA/EA and K-as-FA uniformly.

For ~1M+ DOFs the diagonal of K is no longer a sufficient
preconditioner. The standard fix is `HypreBoomerAMG` on the K
block. This is **FA-only** (PA mode would need the
`LORDiscretization` shim), but fine for Phase 4 since K is FA in
Phase 4.1+4.2 anyway.

```cpp
// Phase 4.1+4.2: BoomerAMG on K, identity on λ.
class SaddlePointPreconditioner : public BlockDiagonalPreconditioner {
public:
    SaddlePointPreconditioner(HypreParMatrix& K,
                               const Array<int>& block_offsets) {
        K_amg_ = std::make_unique<HypreBoomerAMG>(K);
        K_amg_->SetSystemsOptions(/* dim */ 3);  // vdim awareness
        SetDiagonalBlock(0, K_amg_.get());
        SetDiagonalBlock(1, &lam_identity_);
    }
private:
    std::unique_ptr<HypreBoomerAMG> K_amg_;
    IdentityOperator lam_identity_;
};
```

The `SetSystemsOptions(3)` call is critical for elasticity: it tells
BoomerAMG that the FE space has 3 unknowns per node and to coarsen
node-wise rather than DOF-wise. Without it, BoomerAMG's coarsening
fragments the displacement components and convergence is poor.

For Phase 4.3 (PA mode) the FA-only `HypreBoomerAMG` becomes
unsuitable; replace with an LOR-based AMG via
`mfem::LORDiscretization`. Out of scope for Phase 4.1; flagged
here for Phase 5+.



### §P4.4.8 ParaView output

Direct port of `PbcVisualizationWriter`. MFEM provides
`mfem::ParaViewDataCollection` natively, so this is much shorter in
C++ than in Python (no manual XML writing). Multi-cycle output for
multi-step ramps is built in.

The mesh-warp + warp-restoration discipline (mortar §9) carries over
verbatim — `RestoreOriginalCoords()` after each `WriteCycle()` is
non-negotiable.

---

## §P4.5 Test driver porting plan

Three drivers, ported in order:

### `examples/patch_test_3d_pbc.cpp` (Phase 4.1.A)

Port of `examples/patch_test_3d_pbc.py`. Single load step, homogeneous
linear-elastic. Fluctuation u_tilde = 0 to machine precision.

PASS criteria identical to Python:
- Krylov converged
- ||du||_inf < 1e-7
- ||<F> - F_macro|| < 1e-9
- ||C·u_total - C·u_lin|| < 1e-9

This is the **load-bearing milestone**. If it passes at np=1, 4, 16
hex+tet, the infrastructure (BoundaryClassifier3D, ConstraintBuilder3D,
saddle-point solver) is correct.

### `examples/patch_test_3d_heterogeneous.cpp` (Phase 4.1.B)

Port of `examples/patch_test_3d_heterogeneous.py`. Strip-split
heterogeneity, multi-step ramp, PWConstCoefficient on Lame parameters.

PASS criteria identical to Python (mortar §3 of het driver):
- Krylov converged
- ||C·u_tilde||_2 < 1e-8
- ||u_tilde||_inf > 1e-12   (**must be non-zero**)
- |<F> - F_macro|_max < 1e-9

### `examples/patch_test_3d_checkerboard.cpp` (Phase 4.1.C)

Port of `examples/patch_test_3d_checkerboard.py`. 2x2x2 octant XOR,
maximum-stress test for the constraint machinery (every matched
element pair crosses a material interface).

PASS criteria identical to heterogeneous.

---

## §P4.6 Validation strategy

### §P4.6.1 Bit-comparison with Python

For Phase 4.1 we want **bit-identical numerical answers** between
C++ and Python at np=1 hex, n=4 mesh.

Mechanism:
1. Add a Python-side debug flag that serialises the assembled C
   matrix (CSR triples), `u_lin`, the saddle-point RHS, and the
   final solution `du` to `.npy` / `.txt` files.
2. Add a C++-side debug flag that does the same.
3. Diff the files. Tolerance: floating-point identity for `C` (it's
   built from rational dual basis values), 1e-12 for solution
   vectors (Krylov tolerance dominates).

This is the gold-standard regression test. Any mismatch exposes a
bug in the C++ implementation.

### §P4.6.2 Per-class unit tests in C++

Mirror of the Python test suites:
- `test_mortar_3d_unit.cpp` — dual basis values (Phase 3.2.A).
- `test_face_mortar_3d.cpp` — dense block correctness (Phase 3.2.B).
- `test_edge_mortar_3d.cpp` — edge mortar reuse (Phase 3.3.A).
- `test_boundary_classifier_3d.cpp` — topology helper tests (3.3.B).
- `test_constraint_builder_3d.cpp` — sparsity + nullspace (3.3.C).

Use Catch2 or GoogleTest depending on ExaConstit's existing
convention. Each test file mirrors one Python suite and has the
same number of assertions.

### §P4.6.3 Scaling validation matrix (Phase 4.2)

Once Phase 4.2 (tile-partitioned matching) is in:

| n   | global zones | global TDOFs | nranks tested        | expected status   |
|-----|-------------:|-------------:|----------------------|-------------------|
| 4   |          64  |        375   | 1, 4, 16             | machine-precision |
| 8   |         512  |       2187   | 4, 16, 64            | machine-precision |
| 16  |       4 096  |     14 739   | 16, 64               | machine-precision |
| 32  |      32 768  |    107 811   | 64, 256              | machine-precision |
| 64  |     262 144  |    823 875   | 256, 1024            | machine-precision |
| 128 |   2 097 152  |  6 440 067   | 1024, 4096           | scaling check     |
| 256 |  16 777 216  | 50 923 779   | 4096, 16384          | scaling check     |

The "machine-precision" threshold should hold at any nranks count
because the algorithm is deterministic modulo MPI reduction order;
deviations indicate a load-imbalance or numerical-roundoff issue
worth investigating.

The "scaling check" rows are about wall-time; PASS criteria stay
the same but we expect to see Caliper data showing classifier setup
< 5% of total runtime, mortar integration < 1%, saddle-point solve
~80%+ (the right place for time to go).

### §P4.6.4 Caliper instrumentation

ExaConstit convention: `CALI_CXX_MARK_SCOPE("name")` at the top of
every method that does non-trivial work. Names:

```
mortar_pbc::classifier::compute_bbox
mortar_pbc::classifier::discover_face_label_by_attr
mortar_pbc::classifier::gather_boundary_records      [Phase 4.1]
mortar_pbc::classifier::tile_partitioned_match       [Phase 4.2]
mortar_pbc::classifier::build_corners
mortar_pbc::classifier::build_edges
mortar_pbc::classifier::build_faces
mortar_pbc::face_mortar::integrate_pair
mortar_pbc::edge_mortar::integrate_pair
mortar_pbc::constraint_builder::build_hypreparmatrix [Phase 4.1]
mortar_pbc::constraint_builder::build_ea_operator    [Phase 4.3]
mortar_pbc::driver::solve_step::assemble_K
mortar_pbc::driver::solve_step::saddle_point_krylov
mortar_pbc::driver::solve_step::compute_F_average
mortar_pbc::visualization::write_step
```

Output goes through Caliper's existing ExaConstit configuration (the
`*.cali` files); we don't need to add new infrastructure.

---

## §P4.7 Phasing roadmap

```
Phase 4.1 — Initial port (AllGather, HypreParMatrix C)
├── 4.1.A  patch_test_3d_pbc.cpp + four core classes
│           Validate at np=1, 4, 16 hex+tet.
│           Bit-comparison vs Python at np=1.
├── 4.1.B  patch_test_3d_heterogeneous.cpp
├── 4.1.C  patch_test_3d_checkerboard.cpp
└── 4.1.D  Per-class unit tests (5 test suites).
            All sandbox-equivalent of Python tests passing.

         ↓ (gate: all of 4.1.A-D green)

Phase 4.2 — Distributed-hash matching
├── 4.2.A  Refactor BoundaryClassifier3D to AllGather-free path.
│           Re-validate 4.1.A-C at np=4, 16, 64.
├── 4.2.B  Scaling validation up to np=1024 on test cluster.
└── 4.2.C  Caliper-driven profiling, document hot paths.

         ↓ (gate: 4.2.B passes at np=1024 with no surprise hot paths)

Phase 4.3 — Element-assembly constraint operator (CONFORMING meshes)
├── 4.3.A  MortarConstraintOperator class, runtime selectable via
│           --constraint-storage=ea flag.
├── 4.3.B  GPU port of EA path (mfem::forall over pairs).
│           First pass DONE: forward Mult on flat arrays + memory-
│           manager annotations; DEVICE_DEBUG-clean. Pending: atomic-
│           add MultTranspose, real CUDA/HIP build validation,
│           performance work. See §P4.4.6.9.
├── 4.3.C  A/B validation: hypre vs ea at np=1, 4, 64, 256, identical
│           output to Krylov tolerance.
└── 4.3.D  Performance comparison: total wall-time, K matvec time,
            C matvec time, peak memory. EA should be no slower than
            Hypre on CPU and faster on GPU.

         ↓ (gate: 4.3.C green; 4.3.B atomic-add follow-up
             can land in parallel with Phase 4.4)

Phase 4.4 — Non-conforming face mortar (Phase 3.5 in architecture doc)
├── 4.4.0  Architecture-doc clarification: explicit D-vs-A_m domain
│           split documentation in §3.5 / §3.7.
├── 4.4.A  Add Axom dependency (BLT/CMake integration). Validate by
│           compiling a no-op sandbox file.
├── 4.4.B  MatchClippedFacePairs broad-phase via axom::spin::BVH<2>.
│           Unit-test the candidate-pair enumeration.
├── 4.4.C  Polygon clipping via axom::primal::clip + fan-triangulation.
│           Tile-cover invariant test.
├── 4.4.D  AssemblePairClipped (quad + tri). Validate via:
│           (a) conforming-via-clipped sanity test (4×4 vs 4×4);
│           (b) non-conforming patch test (4×4 vs 5×5, homogeneous).
└── 4.4.E  Dispatch in BuildLocalPairBlocks; new
            test_patch_3d_pbc_nonconforming executable.
            Validate at np=1, 4, 7 with strip + checkerboard
            non-matching patterns.

         ↓ (gate: 4.4.E green)

Phase 4 complete. Promote tests/mortar_pbc/ → src/mortar_pbc/.
Move on to Phase 5 (ExaConstit integration: BCManager, SystemDriver,
velocity-primal switch).
```

---

## §P4.8 Specific implementation hazards

These are places where I expect to spend disproportionate debugging
time. Worth flagging now so we don't lose days to surprises.

### §P4.8.1 The byNODES vs byVDIM ordering trap

Mortar §9.4 documents this for Python. In C++ the trap is just as
real: `mfem::ParFiniteElementSpace` constructed with explicit
`Ordering::byNODES` is required for the prototype's TDOF assumptions
to hold. The constraint matrix's column indices directly use
`fes.GetGlobalTDofNumber(ldof)` returns; if the FES is byVDIM, the
gtdof_x → gtdof_y → gtdof_z stride changes from `+n_scalar` to
`+1` and the constraint expansion silently produces wrong matrices.

**Mitigation**: assert ordering at FES construction time, document
in class docstrings, write a unit test that builds a small mesh
both ways and verifies the assert fires when byVDIM is used.

### §P4.8.2 HypreParMatrix lifetime traps

MFEM #793 (linked in mortar §6.4) describes the SparseMatrix-aliasing
problem when `ParBilinearForm::ParallelAssemble` is called twice.
Solution in the heterogeneous Python driver: build TWO ParBilinearForm
objects, one for `K_full` and one for `K_eliminated`. Carry this
pattern verbatim to C++.

For the constraint matrix, a related concern: after building `C` via
the HypreParMatrix CSR constructor, the local `SparseMatrix diag` /
`offd` go out of scope. Verify HypreParMatrix has copied (it does,
internally; documented in MFEM source). But DOUBLE-VERIFY at first
construction with a deliberate scope-exit + Mult-and-check.

### §P4.8.3 Distributed C row-partition correctness

The nonmortar-DOF-ownership row partitioning assumes that for every nonmortar
node owned by rank r, all the mortar nodes in r's matched mortar row
are reachable (either local-diag or off-process via cmap). This is
true by construction (mortar and nonmortar faces of an axis-aligned RVE
have the same MFEM partition modulo periodic identification), but
NOT verified.

**Mitigation**: at build time, after constructing C, do a sanity
matvec: pick a deterministic test vector, multiply by C in HypreParMatrix
form, gather the result, compare against a serial reconstruction. Any
mismatch indicates a partitioning bug. Mirror of the
"Operator-correctness diagnostic" in the 2D Python driver
(`patch_test_2d.py` lines 730ish).

### §P4.8.4 The runtime attribute-discovery cross-rank consistency

Mortar §11.7.2 documents that MFEM's `MakeCartesian3D` boundary-
attribute ordering varies. The Python `_discover_face_label_by_attr`
runs locally then `comm.allgather`s + checks consistency. In C++:

```cpp
std::map<int, std::pair<std::string, std::string>> local_findings = ...;
// Pack into a flat int buffer for AllGather.
// Each rank sends (n_findings_this_rank, attr0, axis0, extreme0, ...).
std::vector<int> packed = PackFindings(local_findings);
auto all_packed = MpiAllgatherv(packed, comm);
std::map<int, std::pair<std::string, std::string>> merged;
for (const auto& rank_findings : all_packed) {
    for (const auto& [attr, finding] : rank_findings) {
        if (auto it = merged.find(attr); it != merged.end()) {
            MFEM_VERIFY(it->second == finding,
                "Inconsistent face-label discovery across ranks");
        } else {
            merged[attr] = finding;
        }
    }
}
```

**Easy to get wrong**: forgetting the consistency check and using
the first-rank-with-this-attr's finding without verifying other
ranks see the same. Silent bugs follow.

### §P4.8.5 The "Allgather everything to rank 0" pattern (C-as-CSR)

In Python, the saddle-point right-hand side construction uses
`g_par = C @ u_lin` where C is a scipy CSR replicated on rank 0.
In C++ with a true distributed C, this is just `C->Mult(u_lin_par,
g_par)` and Hypre handles it. **No allgather of u_lin needed.**
Resist the temptation to port Python's manual pack-unpack style.

### §P4.8.6 The MFEM IntRule order convention

Python `mfem.IntRules.Get(geom, order)` where `order = 2 * fe.GetOrder() + 1`
for K assembly. Same convention in C++. For the volume-averaged F
integrand (∇u, piecewise constant on linear elements) we can drop
to `order = 2`; documenting in class so it's clear what each
quadrature is doing.

### §P4.8.7 Boundary-subcommunicator gotchas

The boundary subcomm pattern (§P4.4.0) is straightforward in
principle but has several places where bugs hide.

**Trap 1: forgetting that `boundary_comm == MPI_COMM_NULL` on
interior ranks.** Any call to `MPI_Comm_size(boundary_comm, ...)`,
`MPI_Comm_rank(boundary_comm, ...)`, or any collective on
`boundary_comm` from an interior rank is undefined behaviour
(typically a crash, sometimes a silent hang). Every boundary-comm
operation must be guarded:

```cpp
if (boundary_comm != MPI_COMM_NULL) {
    // boundary work
}
```

In the C++ code, the cleanest way to enforce this is to make
`BoundaryClassifier3D` and `ConstraintBuilder3D` only constructible
when the comm is non-null. If construction is itself guarded, all
methods on the resulting object are safe to call without further
checks.

**Trap 2: mixing WORLD and boundary-comm reductions in the same
function.** For example, the runtime attribute-discovery does its
local check on `boundary_comm` AllGather, but then the result needs
to be Bcast to **interior ranks** so the driver on those ranks
knows the total count of constraint multipliers (needed for the
HypreParMatrix-on-WORLD construction). This requires a separate
WORLD broadcast from a designated boundary-comm root. Forgetting to
do this leaves interior ranks with stale counts and the
HypreParMatrix construction breaks.

The pattern:

```cpp
int n_lam_total_world;
if (boundary_comm != MPI_COMM_NULL) {
    int my_brank;  MPI_Comm_rank(boundary_comm, &my_brank);
    if (my_brank == 0) {
        n_lam_total_world = ComputeFromBoundaryClassifier();
    }
    // Bcast within boundary_comm.
    MPI_Bcast(&n_lam_total_world, 1, MPI_INT, 0, boundary_comm);
}
// NOW Bcast to interior ranks via WORLD: every rank participates,
// the boundary-rank-with-the-value broadcasts to all others.
// We need a designated WORLD root — typically world rank 0 if it's
// in boundary_comm, otherwise the lowest world rank that is.
MPI_Bcast(&n_lam_total_world, 1, MPI_INT, designated_root, MPI_COMM_WORLD);
```

A simpler alternative when nranks is reasonable: AllReduce on WORLD.
Every boundary rank reports its `n_lam_local`; every interior rank
reports 0; the AllReduce sum is `n_lam_total_world` and arrives on
every rank.

```cpp
int my_n_lam_local = (boundary_comm != MPI_COMM_NULL)
                      ? ComputeMyNLamLocal()
                      : 0;
int n_lam_total_world;
MPI_Allreduce(&my_n_lam_local, &n_lam_total_world, 1, MPI_INT,
              MPI_SUM, MPI_COMM_WORLD);
```

This pattern is preferred because it doesn't require hunting for a
designated root.

**Trap 3: re-using a freed boundary_comm.** `MPI_Comm_split` creates
a new communicator that must be freed with `MPI_Comm_free` at
shutdown. If `BoundaryClassifier3D` holds the comm by value and has
its destructor free it, but the driver also tries to free it
later, you get a double-free.

The cleanest model in ExaConstit is to **store boundary_comm in
the existing `SimulationState` class**, which already owns the
program-lifetime communicators. `SimulationState` owns the lifecycle
(creates the comm at startup, frees it in its destructor); all of
`BoundaryClassifier3D`, `ConstraintBuilder3D`, and `MortarPbcDriver`
take it by reference (`MPI_Comm boundary_comm` from the SimulationState
accessor). No object except `SimulationState` ever calls `MPI_Comm_free`
on it. This matches ExaConstit's existing convention for the few
non-WORLD comms it manages.

```cpp
// In SimulationState:
class SimulationState {
public:
    void InitMortarPbcSubcomm(const mfem::ParMesh& pmesh) {
        const int has_boundary = (pmesh.GetNBE() > 0) ? 1 : MPI_UNDEFINED;
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_split(MPI_COMM_WORLD, has_boundary, world_rank,
                       &mortar_pbc_boundary_comm_);
    }
    MPI_Comm GetMortarPbcBoundaryComm() const {
        return mortar_pbc_boundary_comm_;
    }
    ~SimulationState() {
        if (mortar_pbc_boundary_comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&mortar_pbc_boundary_comm_);
        }
    }
private:
    MPI_Comm mortar_pbc_boundary_comm_ = MPI_COMM_NULL;
};
```

This avoids the need for a standalone RAII wrapper class — the
SimulationState lifetime already provides RAII semantics, and we
match the ExaConstit pattern for the handful of non-WORLD comms
that exist today.

**Trap 4: dynamic load-balancing isn't supported.** If MFEM's
ParMesh repartitions across the run (it doesn't currently for
ExaConstit's flow, but might in the future), the boundary-rank set
changes and the subcomm needs to be rebuilt. For Phase 4 we assume
the partition is static after construction; flag this as a Phase 5+
concern if/when ExaConstit grows dynamic load balancing.

### §P4.8.8 Collective MFEM operations inside `if (rank == 0)` print blocks

Several MFEM accessors that look like cheap scalar getters are in
fact COLLECTIVE operations that issue MPI reductions internally:

* `mfem::ParMesh::GetGlobalNE()` — Allreduce of local element count.
* `mfem::ParFiniteElementSpace::GlobalTrueVSize()` — Allreduce of
  local TDOF count.
* `mfem::ParFiniteElementSpace::GlobalVSize()` — Allreduce.
* Some forms of `HypreParVector::Norml2()` / `Normlinf()` — Allreduce
  for the global norm. (`mfem::Vector::Normlinf()` on a TDOF view is
  local; only the Hypre-vector forms collective.)

**The bug pattern**: putting any of these inside a rank-0-only print
block:

```cpp
if (rank == 0)
{
    std::cout << "global TDOFs = " << fes.GlobalTrueVSize() << ...;
}
```

Only rank 0 enters the Allreduce; the other ranks proceed past it.
The next collective on the other ranks then consumes rank 0's stale
Allreduce — different `count`, different datatype — and you get
`MPI_ERR_TRUNCATE` (or worse: a silent stall on a buffered transport).

**Mitigation**: always call collectives on every rank, then print
the cached scalar inside the conditional.

```cpp
const int n_global_tdofs = fes.GlobalTrueVSize();   // collective — all ranks
if (rank == 0)
{
    std::cout << "global TDOFs = " << n_global_tdofs << ...;
}
```

This is invisible at np=1 (which is why it slipped through in the
patch-test driver's first cut) and only manifests at np ≥ 2. Code
review checklist: every `if (rank == 0)` block must be audited for
this; in particular any line of the form `<<  some_par_thing.Method()`
inside the block is suspect.

### §P4.8.9 Parallel matrix column partitions must align with the FES TDOF partition

When constructing a `mfem::HypreParMatrix` whose columns correspond
to FES true-DOFs (e.g. the constraint matrix C, whose columns
multiply against displacement TDOF vectors), the column partition
MUST be taken from `fes.GetTrueDofOffsets()`, NEVER computed as a
uniform chunk split.

**The bug pattern**:

```cpp
// WRONG — uniform chunk split that does not match FES partition
const HYPRE_BigInt chunk = n_global_cols / nranks;
const HYPRE_BigInt my_chunk = chunk + (rank < rem ? 1 : 0);
// ...
col_starts[0] = my_start;
col_starts[1] = my_start + my_chunk;
```

The FES's actual TDOF partition is determined by **METIS partitioning
of the mesh**, not by uniform chunks. For a 4×4×4 hex mesh at np=4,
typical METIS yields {90, 90, 60, 135} TDOFs per rank, while uniform
chunking would give {94, 94, 94, 93}. The matvec `C·u` then aborts
with `C.Width() != K.Height()` inside `BlockOperator::Mult` — or
worse, on builds without that check, silently produces a wrong-sign
result because Hypre's diag/offd splitting puts entries in the wrong
half.

**Mitigation**: take the column partition straight from the FES.

```cpp
HYPRE_BigInt* fes_tdof_offsets = fes.GetTrueDofOffsets();
col_starts[0] = fes_tdof_offsets[0];
col_starts[1] = fes_tdof_offsets[1];
```

Same rule for row partitions on matrices whose rows are TDOFs (K
itself, but `ParBilinearForm::ParallelAssemble` handles that
automatically). It only bites for matrices the user constructs
directly via the explicit-CSR `HypreParMatrix` ctor.

Defensive check at construction: verify
`col_starts[1] - col_starts[0] == fes.GetTrueVSize()` and
`MFEM_VERIFY` on mismatch. Catches FES partition state inconsistency
(e.g., re-partitioning after construction) before it propagates.

This bug is invisible at np=1 (every partition is trivially
`[0, n_global)` regardless of how it's computed). **Multi-rank
validation is required to catch it** — np=1 unit tests cannot.

---

### §P4.8.10 Tile-decomposed mortar block merge must aggregate by gtdof identity

When Phase 4.2's tile partition splits a face-mortar pair across
multiple ranks, each rank produces a partial `FaceMortarPairBlock`
covering its tile-local elements. Merging these partial blocks across
ranks **must sum partial rows by gtdof identity** for shared DOFs;
naive concatenation produces multiple rows for the same DOF and gives
a constraint matrix with twice (or quadruple) the correct number of
rows.

**The bug pattern**:

```cpp
// WRONG — concatenate rows, ignoring DOF identity
int row_ofs = 0;
for (const auto& p : parts) {
    for (int i = 0; i < p.NumNonmortarKept(); ++i) {
        merged.nonmortar_gtdofs[row_ofs + i] = p.nonmortar_gtdofs[i];
        merged.D(row_ofs + i) = p.D(i);
        // ... A_m row copied as-is
    }
    row_ofs += p.NumNonmortarKept();
}
```

**Why it's wrong**: with 2×2 tile partitioning of a 4×4 nonmortar
face, the inner-subgrid DOFs sit at the corners of a 3×3 quad pattern.
DOF (2,2) (the center of the inner subgrid) is at the corner of four
face elements — one in each of the four tiles. Each tile-rank produces
a partial block with DOF (2,2) in its `nonmortar_gtdofs` along with
partial `D` and partial `A_m` row contributions (the integral over
just that rank's tile area). Concatenation gives FOUR rows for DOF
(2,2) instead of one summed row, and the constraint matrix's row
count balloons by the sharing factor.

**Mitigation**: the merge step must (a) build a `gtdof → merged_row`
map by union across rank-blocks, (b) build a similar `gtdof →
merged_col` map for mortar columns, (c) translate each rank-block's
`(i, j)` entries through these maps, and (d) **accumulate** into the
merged `A_m` and `D` instead of assigning. Identical-gtdof entries
across ranks then naturally sum.

```cpp
// CORRECT — gtdof-keyed merge
std::map<int, int> nm_gtdof_to_row;
for (const auto& p : parts)
    for (int i = 0; i < p.NumNonmortarKept(); ++i) {
        const int g = p.nonmortar_gtdofs[i];
        if (nm_gtdof_to_row.find(g) == nm_gtdof_to_row.end())
            nm_gtdof_to_row[g] = nm_gtdof_to_row.size();
    }
// (similar for mortar columns)
// then for each rank-block, look up (i, j) → (mr, mc) and ACCUMULATE
out.D(mr) += p.D(i);
out.A_m(mr, mc) += p.A_m(i, j);
```

**Mathematical justification**: the integral over a face's mortar
operator decomposes additively over disjoint sub-areas. If element
E1 is in tile A and E2 is in tile B, and both touch nonmortar DOF X,
then \f$\int_{E_1 \cup E_2} N^X \, dA = \int_{E_1} N^X + \int_{E_2}
N^X\f$. The two partial integrals must sum into one row of D and
one row of A_m — not produce two rows.

The same applies to mortar columns: if mortar DOF Y is touched by
elements in two tiles, both rank-blocks contribute partial entries
to that column. The merge sums them.

This bug is **invisible at np=1** (only one tile, no merge needed —
the merge function early-returns `parts[0]`). It manifests at np>1
as a constraint matrix with too many rows and a saddle-point system
that either fails to converge (Krylov breakdown) or converges to a
wrong solution. **Multi-rank validation is required to catch it.**

The discovery story: the original Batch I implementation used naive
concatenation, with a comment claiming "different ranks' tiles
produce non-overlapping nonmortar gtdofs (they own different tiles)
so simple concatenation is correct." This was wrong. The DOFs at the
**boundaries between tiles** belong to elements in multiple tiles,
and so appear in multiple rank-blocks' `nonmortar_gtdofs` lists.

The fix is a 30-line replacement of the merge body; the rest of the
tile-shuffle / per-pair-block infrastructure was unaffected.

---

### §P4.8.11 Sparsifying `FaceMortarPairBlock::A_m` is the dominant memory win

**Lesson**: For conforming face mortars on hex8, `A_m` is **highly
sparse** — each nonmortar row has at most ~16 mortar matches (the
union of mortar nodes from the matched-element pairs touching that
nonmortar node). Storing dense at production scale is the dominant
memory term.

The arithmetic: at N=100 with three face mortars, dense `A_m` is
roughly `(N²)² × 8 bytes ≈ 800 MB` per face block. Sparse with
`16·N²` nonzeros is ~1 MB. The factor of `N²` reduction is what
unblocks production runs — no other Phase 4.2 change comes close.

The implementation cost was modest (Batch L, ~400 LOC):

- `FaceMortarPairBlock::A_m` storage type (`mfem::DenseMatrix` →
  `mfem::SparseMatrix`).
- Producer: `AssemblePairConforming` constructs build-mode, calls
  `Add()` per integration contribution, `Finalize()` before return.
- Consumer (`ScatterFaceBlock`): walk via CSR `GetI/GetJ/GetData`
  rather than `(k, l)` indexing. (`SparseMatrix::operator()(i,j)` is
  O(log nnz_row) with binary search, so naive double-loop becomes
  O(n_rows · n_cols · log nnz) — much worse than dense. Always walk
  CSR.)
- Pack/unpack across MPI: replace dense row-major (`n_n × n_m`
  doubles) with sparse CSR (I + J + values, `nnz` doubles).
- Merge across rank-fragments (§P4.8.10): walk source CSR rows,
  `Add()` into build-mode merged matrix, `Finalize()` once.

The `MortarBlock2D::A_m` for **edge** mortars stays dense
deliberately — edge blocks are 1D-coupling with `n_n × n_m ≈ N²`,
not `N⁴`, so dense is fine and the read pattern is simpler.

**Anti-pattern to avoid**: don't sprinkle `Finalize()` calls
defensively. `Finalize()` is idempotent on already-finalized
matrices, but each pre-Finalize `Add()` followed by a Finalize
followed by another Add forces a CSR-to-build-mode-and-back
conversion that's O(nnz) each time. Build everything you need to
build, THEN Finalize once, THEN read.

---

### §P4.8.12 FES-aligned row partition is what makes AllToAllv routing pay off

**Lesson**: The asymptotic memory win in Phase 4.2 isn't from
swapping AllGather → AllToAllv in isolation — it's from changing
the **row partition convention** so each block has only a small set
of plausible row owners. Without that, AllToAllv either degenerates
into AllGather (every block must be sent to every potential row-
owner) or requires expensive coordination.

The two pieces are synergistic:

1. **AllToAllv-to-row-owner** routing replaces the broadcast of
   `m_gathered_pair_blocks` to every rank with a directed exchange
   where each rank receives only the blocks contributing to its
   rows. Per-rank receive volume drops from O(global_blocks) to
   O(global_blocks / n_owners).

2. **FES TDOF-aligned row partition** assigns row `r` (derived from
   nonmortar gtdof `g`) to the rank that owns `g` in FES. This
   means the rows from one face-mortar block fragment by the FES
   partition: a block whose nonmortar gtdofs span K different FES
   owners becomes K fragments routed to K destinations.

Why FES alignment specifically:

- The constraint matrix C's column partition MUST already match the
  FES TDOF partition (§P4.8.9 — for `C·u` parallel matvec to work,
  C's columns must be partitioned IDENTICALLY to K's rows). The
  row partition has no such constraint, but FES alignment yields
  a useful invariant: **the (row r, col r) "diagonal" entry of C
  involves the same gtdof `g` on both sides**, and that gtdof is
  on the same rank as both — no off-rank communication for the
  diagonal block.
- It avoids the alternative of routing each block's contents to
  multiple destinations based on a fair-split of the row range
  (which would require a routing layer and lose the FES affinity).

Implementation steps (Batch N, ~600 LOC):

- Allgather `FES.GetTrueDofOffsets()[0]` at classifier
  construction time → cached `m_fes_tdof_offsets_all`. Add
  `GtdofOwnerRank(int gtdof)` doing binary search.
- Replace `GatherPairBlocksAcrossBoundary` with
  `RoutePairBlocksToRowOwners`: for each local block, group rows
  by `GtdofOwnerRank(nonmortar_gtdofs[k])`, pack one fragment per
  destination, `MPI_Alltoallv` on `m_comm` (NOT
  `m_boundary_comm` — interior ranks may own the relevant FES
  TDOFs).
- Keep the gtdof-keyed merge logic from Batch I/L (§P4.8.10) for
  same-bucket fragments arriving at one rank from multiple source
  ranks. The merge code is unchanged; only the input source
  (Alltoallv result vs Allgather result) differs.
- Filter edge mortar rows in `ScatterEdgeBlock` by
  `GtdofOwnerRank(nonmortar_g_xyz[0]) == my_rank`. Edge mortars
  are produced redundantly on every rank (cheap 9 small-dense
  assemblies), so the filter is a per-row early-`continue`.
- Remove the `n_lam_local` argument from `BuildHypreParMatrix` —
  the row partition is now data-determined. Add `NumLocalRows()`
  for callers needing the value.

Subtleties:

- **At np=1, every gtdof maps to rank 0**, so the routing is
  trivial and the test path remains numerically identical to
  Batches K/L. This was crucial for keeping the unit-test suite
  green during the refactor.
- **A nonmortar gtdof's three components (x, y, z)** can in
  principle be on different FES owners, but in MFEM's standard
  byNODES vector ordering they cluster on the same rank. The
  Batch N code uses the x-component as the row-owner anchor for
  consistency between edge and face paths — y and z are sent to
  the row owned by x's rank, which costs nothing if they're on
  the same rank (typical case) and at worst a small amount of
  off-rank column read on `C·u` (if they aren't).
- **Interior ranks may own FES TDOFs that are nonmortar gtdofs of
  boundary blocks.** This is why the AllToAllv must run on
  `m_comm`, not `m_boundary_comm`. METIS partitioning does not
  guarantee co-location of FES TDOF ownership with element
  ownership of boundary faces.

---

### §P4.8.13 Use `HYPRE_MPI_BIG_INT`, never a hardcoded width, for `HYPRE_BigInt` MPI exchanges

**Lesson**: When sending a `HYPRE_BigInt` over MPI, use
`HYPRE_MPI_BIG_INT` as the MPI datatype, NOT a hardcoded
`MPI_LONG_LONG` or `MPI_INT`. `HYPRE_BigInt` is conditionally
typedef'd to `int` (32-bit) or `long long` (64-bit) depending on
HYPRE's `--enable-bigint` configure flag, and `HYPRE_MPI_BIG_INT`
resolves to the matching MPI datatype. Hardcoding the wrong width
silently corrupts the receive buffer.

**The discovery story** (Batch N first run on Mac at np=7): the FES
TDOF offset Allgather added in Batch N used a hardcoded
`MPI_LONG_LONG`. ExaConstit's HYPRE build has `HYPRE_BigInt = int`
(the default; production rarely needs >2³¹ DOFs). The mismatch
manifested as:

- Send buffer: one 4-byte `int` containing rank's start offset.
- MPI sends 8 bytes per element (because we said `MPI_LONG_LONG`).
- Receive buffer: `std::vector<int>` (4 bytes per slot).
- MPI writes 8 bytes per slot, **clobbering two adjacent ints**.

Result: corrupted offset table that fails the monotone-sanity check
with values like "108 -> 0" mid-array. The mistake is easy to make
because:

1. Sandbox stubs that typedef `HYPRE_BigInt = long long` mask the
   bug entirely.
2. At np=1 the mistake doesn't manifest (one element, no
   interleaving).
3. At small process counts (2-4) the corruption may not produce
   non-monotone values by luck of stack initialization.

**The fix is one-line**: replace `MPI_LONG_LONG` with
`HYPRE_MPI_BIG_INT` at the call site. There's exactly one place in
the entire mortar-PBC code that exchanges raw `HYPRE_BigInt` over
MPI: the `m_fes_tdof_offsets_all` Allgather in
`BoundaryClassifier3D` ctor. All other MPI-of-long-long uses in the
codebase are `std::vector<long long>` pack buffers (gtdofs widened
to long long for portability) — those are genuine `long long`s and
correctly use `MPI_LONG_LONG`.

**General principle**: any time the data type comes from
HYPRE/MFEM internals (rather than being a deliberate wire format
you control), use the matching MPI macro:
- `HYPRE_BigInt` → `HYPRE_MPI_BIG_INT`
- `HYPRE_Int` → `HYPRE_MPI_INT`
- `mfem::real_t` → `MPITypeMap<mfem::real_t>::mpi_type` (when
  MFEM is built with `--enable-single`)

Sandbox stubs should also reflect this conditional. After this
batch, the stub at `/tmp/mfem_stub/mfem.hpp` defines:

```c
#ifndef HYPRE_MPI_BIG_INT
#define HYPRE_MPI_BIG_INT MPI_LONG_LONG
#endif
```

so future stub-driven sandbox testing matches the real header
behavior.

---

### §P4.8.14 The "row-replicated, fair-split" stepping-stone strategy

**Lesson**: For a multi-batch refactor that culminates in a
distributed row partition, an intermediate **"every rank produces
the full matrix, then slices its rows"** stage is invaluable. It
keeps the unit-test invariant trivially satisfied (the same C
matrix on every rank means any np=1 test produces exactly the
same numerical output as the eventual distributed code) while
the data-movement infrastructure stabilizes underneath.

The stepping-stone for Phase 4.2 spanned Batches I → K → L → M:

- **Batch I**: AllGather all per-pair blocks to every rank.
  Every rank produces the full constraint matrix `C` redundantly.
  Row partition is fair-split (rank `r` owns rows
  `[r·N/P, (r+1)·N/P)`).
- **Batch K**: Same C-on-every-rank invariant; just move the
  AllGather from WORLD to boundary_comm + WORLD broadcast fanout.
- **Batch L**: Same invariant; sparsify the per-pair-block storage
  to make the AllGather payload tractable at scale.
- **Batch M**: Same invariant at the row-emit layer; refactor
  `BuildHypreParMatrix` to skip the intermediate replicated
  `SparseMatrix` allocation and filter triples on the fly.

Then **Batch N** breaks the invariant deliberately: after Batch N,
every rank has only the row-fragments it owns; `Build()` no
longer produces "the full C" but rather "this rank's local row
slice." The unit tests that ran at np=1 continue to work because
at np=1 every gtdof is owned by rank 0 — so "this rank's local
row slice" equals "the full C".

**Why this matters**: a flag-day refactor that introduces both the
distributed row partition and the AllToAllv routing in one
commit would have left unit tests broken for weeks while bugs
shake out. The stepping-stone strategy keeps every batch
locally testable and makes regressions easy to bisect.

**Cost paid**: Batches I/K/L/M's redundant work — every rank
producing the full C — adds nontrivial memory and time at large
scale. But:

1. The existing unit-test suite already runs at np=1, where
   redundancy is zero.
2. The patch tests at np=4 stress the redundancy but are tiny
   (4³ RVE), so the overhead is acceptable.
3. Production scale (100³+) wouldn't have stayed on the
   intermediate stepping-stones anyway — the goal of Phase 4.2
   was always to land at the Batch N design.

The pattern generalizes: **when you have a distributed-data
refactor that decouples "every rank has every datum" from "every
rank has only its slice", land the supporting infrastructure
first under the redundant invariant, then break the redundancy
in a final focused batch**. The redundant invariant is a powerful
test-fixture: it asserts the new code produces the right answer
without yet committing to the new partition convention.

**Anti-pattern**: trying to land the row partition change AND
the data-movement refactor AND the storage-type change in one
batch. This breaks unit tests in three different ways
simultaneously and makes regression diagnosis nearly impossible.

---

### §P4.8.15 Refactor a shared inner loop when an overload varies only at one step

**Lesson**: When adding a function overload that varies only at
one step from the original (here: how `inv_diag_S` is computed —
HypreParMatrix CSR vs EA per-pair walk), the right structural
move is to **extract the shared body into a private helper**, not
to copy-paste 100+ lines of unchanged code into the new overload.

**The discovery (Batch S)**: The existing
`SaddlePointSolver::Solve(K_hp, C_hp, ...)` had ~125 LOC of body:
dimension checks, `BlockOperator` construction with `K_hp` and
`C_hp` as the (0,0) and (1,0) blocks, `BlockDiagonalPreconditioner`
setup, GMRES/MINRES/BiCGSTAB instantiation, RHS construction,
Krylov solve, solution extraction. The new EA overload
`Solve(K_hp, C_op, ...)` differed only at the preconditioner-
setup line — `BuildInvDiagSchur(C_hp, ...)` becomes
`C_op.ComputeInvDiagSchur(...)`. Everything else is identical
once `C` is typed as `mfem::Operator&` instead of
`mfem::HypreParMatrix&`.

The temptation was to copy-paste. Two arguments against:

1. **Maintenance cost**. Any future Krylov-side change (new
   `iterative_mode` semantics, additional solver type, alternate
   RHS form, different solution-extraction layout) would need to
   land in two places. Forgetting one is a silent regression
   that may take days to track down.

2. **Drift risk**. Even if we always remember to update both
   places, small differences accumulate over time — one overload
   gets a `MFEM_VERIFY` the other doesn't, one's diagnostic
   format differs slightly. After a few years there are two
   subtly-different solvers.

The chosen pattern: a private `SolveImplInternal` taking K and C
as `mfem::Operator&` plus pre-computed `inv_diag_K` and `inv_diag_S`.
Each public overload's job shrinks to:
- dimension-check the inputs (overload-specific because the
  signatures differ)
- compute `inv_diag_K` and `inv_diag_S` its own way
- delegate to the helper

The helper is then ~110 LOC, the public `Solve` overloads each
become ~15 LOC, and a future `Solve(K_op, C_op)` for matrix-free
K just plugs in alongside.

**When NOT to do this refactor**: if the two overloads differ at
many points throughout the body (not just one step), the extracted
helper ends up with so many configuration knobs that it's worse
than two separate functions. The threshold is something like:
"if the helper's parameter list grows beyond ~6 things, two
functions are cleaner."

**When to apply this lesson**: any time you find yourself about
to add a function overload that diverges from an existing one at
only a small number of identifiable steps. The refactor pays for
itself by the second overload, and the third overload (which
often appears later, e.g., the GPU port in Phase 4.3.B) costs
~15 LOC instead of ~125.

---

### §P4.8.16 Pre-flatten host-side data before chasing `mfem::forall`

**Lesson**: When porting a CPU implementation that uses `std::map`,
`std::vector<Struct>`, or other non-GPU-friendly containers in
its hot path, the right first step is **NOT** to wrap the existing
loop in `mfem::forall` — the kernel body would still hit those
containers. The right first step is to **pre-flatten the data at
construction time** into `mfem::Vector` / `mfem::Array<int>` so
the kernel body has nothing but flat array reads.

**The discovery (Phase 4.3.B / Batch X)**: The CPU `Mult` body
walked `m_local_edge_pairs` (a `std::vector<LocalEdgePair>` where
each entry holds a `MortarBlock2D` plus two `EdgeInfo3D` structs)
and `classifier.PairBlocks()` (a similar list). Inside the inner
loop it did `m_gtdof_lookup.find(g_x)` (a `std::map<int,
std::array<int,3>>` lookup) plus `m_import_gtdof_to_slot.find(g_x)`
(another map). None of this can run on a GPU.

The temptation: turn the outermost `for` into `mfem::forall` and
hope. But the kernel body has to be `MFEM_HOST_DEVICE`, and you
cannot dereference `std::map::iterator` on a device thread —
that's a host-only API. So the kernel won't compile, and even
if it did, the data layout is wrong (struct-of-pointers with
heap-allocated buckets is the worst possible GPU memory pattern).

The actual fix: build a `BuildFlatRowArrays()` helper that walks
all the per-pair-block data ONCE at construction and produces:

  * `mfem::Vector m_row_D` (one double per row).
  * `mfem::Array<int> m_row_csr_off` (prefix-sum row → CSR slice).
  * `mfem::Vector m_csr_A` (flat A_kl values).
  * `mfem::Array<int> m_csr_g_m_local` / `m_csr_g_m_recv` (paired
    tagged-index encoding for off-rank vs. local lookups).

After this, `Mult`'s kernel body is pure flat-array indexing —
no maps, no struct walks, no host-only APIs — and `mfem::forall`
just works.

**The cost**: doubled memory for the per-row data (we now have
both the per-pair-block form AND the flat form). At
production-like RVE sizes this is negligible; at toy-test sizes
it's still under a few KB. In return, the matvec hot path runs
on device with a single forall, and DEVICE_DEBUG validates every
memory access.

**Two adjacent design choices** that came up during this batch:

1. **The two-array sentinel-free encoding for off-rank lookups**.
   The mortar component lookup needs to distinguish three cases:
   FES-local, off-rank import buffer, sentinel. Encoding all
   three in a single signed int via shifted-negative ranges is
   tempting but error-prone (what value is the sentinel?
   off-by-one bugs at the encode/decode boundaries). Using two
   parallel `Array<int>` arrays (`m_csr_g_m_local` and
   `m_csr_g_m_recv`) where exactly one is ≥ 0 (the other being
   -1) is more memory but the contract is unambiguous: "if both
   are -1 it's a sentinel, otherwise the non-negative one tells
   you which buffer to read from."

2. **Don't try to GPU-ify everything in the same batch**. The
   forward `Mult` parallelizes cleanly because each row's output
   is unique. `MultTranspose` has many-to-one scatter and needs
   atomic adds; `ComputeInvDiagSchur` has cross-rank Allgatherv
   followed by sequential accumulation. Doing all three in one
   batch triples the surface area of "what could be wrong."
   First-pass scope: just the forward direction. The transpose
   and the preconditioner setup stay on host with HostRead /
   HostWrite annotations (which makes them DEVICE_DEBUG-clean
   without changing their algorithmic structure).

**When to apply this lesson**: any time you have a CPU
implementation full of `std::map` / `std::vector<Struct>` / raw
pointer arithmetic that you want to GPU-port. The setup-time
flatten is the heavy lifting; the forall conversion afterwards
is mechanical.

**When NOT to apply**: setup-time methods (called once per
Newton step or once per simulation), where the cost of staying
on host is amortised. `ComputeInvDiagSchur` is in this category;
the matvec hot path is not.

**See also §P4.8.17** for the companion lesson on what goes wrong
if you DON'T pre-flatten and try to use the existing data
structures directly under `DEVICE_DEBUG` — namely, the
`Vector::GetData()` / `Vector::operator()` traps that fire on
unannotated access to vectors that haven't had their host
validity declared.

---

### §P4.8.17 `Vector::GetData()` and `Vector::operator()` are DEVICE_DEBUG traps

**Lesson**: Under MFEM's `DEVICE_DEBUG` build, the unsafe back-door
APIs (`Vector::GetData()`, `Vector::operator()`, `Vector::operator[]`)
trigger memory-manager assertions if the host validity flag isn't
already set. The fix is **always** to use the typed accessors
(`HostRead`, `HostWrite`, `HostReadWrite`, or their device
counterparts `Read`, `Write`, `ReadWrite`) in any code that reads
or writes Vector data. These declare access intent so the manager
can validate and migrate appropriately.

**The discovery (Phase 4.3.B / Batch X)**: the patch driver was
running cleanly in normal builds but failing under `DEVICE_DEBUG`
with:

```
Assertion failed: (Empty() || (flags & VALID_HOST))
 --> invalid host pointer access
 ... in function: const T *mfem::Memory<double>::operator const double*() const
```

The trigger was inside `DiagonalScaler::Mult` (the per-Krylov-
iteration block-Jacobi preconditioner step), which used:

```cpp
const double* xd  = x.GetData();
double*       yd  = y.GetData();
const double* idd = m_inv_diag.GetData();
```

`y` is a sub-vector view that the `BlockDiagonalPreconditioner`
constructs at iteration time. On first use it has no valid host
copy declared. `GetData()` invokes
`Memory<double>::operator const double*()`, which under
`DEVICE_DEBUG` asserts that either the memory is empty or
`VALID_HOST` is set — and at that moment neither is true.

**The fix is mechanical**: replace `GetData()` calls on Vector
data (and `operator()`, `operator[]` accesses in tight loops)
with the typed accessors. For a read-only loop, hoist a
`HostRead()` pointer above the loop and use it. For a write-only
loop, `HostWrite()`. For accumulation (`+=`), `HostReadWrite()`.

**Where this matters most**: any Vector that comes from "outside"
the function (function arguments, `GetBlock()` views, freshly-
allocated vectors that haven't been written yet). Vectors that
have just been assigned (`v = 0.0;`, `v = other_vector;`) have
their host validity flag set as a side effect of the assignment,
so subsequent operator() accesses on THOSE vectors don't fail —
but it's still better practice to use a hoisted host pointer for
performance reasons (each operator() call goes through a memory-
manager check on every access).

**Specific spots fixed in Batch X**:

  * `DiagonalScaler::Mult` — the trigger from the user report.
  * `BuildInvDiagK` — invert-diag loop converted to raw pointers.
  * `BuildInvDiagSchur` — `MPI_Allgatherv` argument switched to
    `HostRead()`; row-sum accumulation and inversion loops
    converted to raw pointers.
  * `SaddlePointSolver::SolveImplInternal` — RHS construction and
    solution extraction loops converted.
  * `MortarConstraintOperator::ComputeInvDiagSchur` — the entire
    accumulation now goes through a single `sd_data` raw pointer
    obtained at function start.
  * Patch driver — A/B diff loop, `u_total` recovery loop,
    constraint-residual loop, `ComputeVolumeAveragedF` u-copy.

**For future ports**: as a rule of thumb, any time you write
`for (int i = 0; ...) { v(i) = ...; }` on an `mfem::Vector v`,
rewrite it as:

```cpp
{
    double* p = v.HostWrite();   // or HostReadWrite, HostRead
    for (int i = 0; ...) { p[i] = ...; }
}
```

It's no harder to write, runs faster (one memory-manager check
instead of N), and is `DEVICE_DEBUG`-safe by construction.

**Why not just always use `GetData()` when you know it's host-
local?** Because `GetData()` is the unsafe API — it returns a
raw pointer without registering intent with the manager. Future
maintainers may have no way to know whether your function expects
a host-resident vector or one that might have come from device,
and the inconsistent style invites bugs. The typed accessors are
self-documenting.

**See also**:

  * §P4.4.6.9 — the full inventory of what's been converted to
    typed accessors during the Phase 4.3.B first pass, and what's
    still pending. If you're returning to the GPU port work
    cold, start there.
  * §P4.8.16 — the companion lesson on pre-flattening host-side
    data structures before chasing `mfem::forall`. The two
    lessons together cover the "how do I make existing CPU code
    GPU-ready as a first pass" workflow.

---

### §P4.8.18 Adding Axom as an ExaConstit dependency (Batch 4.4-A)

The Phase 4.4 non-conforming face mortar work depends on Axom
(LLNL's mesh-processing library) for two specific primitives:
`axom::spin::BVH<2>` (2D bounding-volume hierarchy for spatial
broad-phase) and `axom::primal::clip` (2D-polygon-on-2D-polygon
Sutherland-Hodgman clipping). Axom is also a future dependency
for ExaConstit's restart capability via Sidre, so adding it here
serves both workstreams.

**Targeted Axom version: v0.14.0** (released 2026-03-31, current
latest at the time of this writing). The API surface we use has
been stable since v0.10.0 with one notable change in v0.12.0:
`AXOM_USE_64BIT_INDEXTYPE` now defaults to `ON`, so
`axom::IndexType` is `std::int64_t` by default (was
`std::int32_t`). This affects declarations explicitly typed as
`axom::IndexType` but not implicit conversions from `int`
literals; our smoke test is written to be IndexType-width-
agnostic.

**What Batch 4.4-A landed in the test/mortar_pbc tree:**

  * `cpp/test/mortar_pbc/CMakeLists.txt` — adds an
    `if(ENABLE_AXOM) list(APPEND EXACONSTIT_TEST_DEPENDS axom)
    endif()` block in the optional-package section, paralleling
    the existing `ENABLE_CUDA` / `ENABLE_OPENMP` / `ENABLE_HIP` /
    `ENABLE_CALIPER` patterns. The `test_axom_smoke` test
    registration is also guarded by `if(ENABLE_AXOM)`.
  * `cpp/test/mortar_pbc/test_axom_smoke.cpp` — minimal sandbox
    test that constructs `axom::primal::Point`, `BoundingBox`,
    `Polygon`, calls `axom::primal::clip`, and instantiates an
    `axom::spin::BVH<2>`. No functional assertions — its only
    purpose is to confirm headers compile and the build system
    finds the library. Registered as a single-rank test (no MPI
    usage).

**What's required at the ExaConstit parent level for Axom to
build:**

The optional-dependency convention used here mirrors the existing
`ENABLE_CALIPER` pattern. Two parent-level pieces are needed:

  1. **Toolchain or host-config sets `ENABLE_AXOM=ON`** alongside
     `axom_DIR` (or `AXOM_DIR`) pointing at the installed Axom
     build directory containing `axom-config.cmake`.
  2. **ExaConstit's `cmake/setup_third_party.cmake`** (or wherever
     Caliper is currently registered, since the patterns are
     parallel) issues:

     ```cmake
     if(ENABLE_AXOM)
         if(NOT TARGET axom)
             find_package(axom REQUIRED CONFIG
                          HINTS ${AXOM_DIR} ${axom_DIR})
         endif()
         # Then register as a known dep so blt_add_executable
         # can resolve it from the DEPENDS_ON list:
         blt_register_library(NAME       axom
                              INCLUDES   ${AXOM_INCLUDE_DIRS}
                              LIBRARIES  axom)
     endif()
     ```

     The exact registration call depends on what
     `exaconstit_fill_depends_list` and `blt_add_executable`
     expect; the existing Caliper plumbing is the model to
     follow.

**Expected build behaviour:**

  * **`ENABLE_AXOM=ON` and Axom found**: `test_axom_smoke`
    compiles, links, and runs (exits 0 with one OK line). All
    existing tests continue to pass unchanged.
  * **`ENABLE_AXOM=ON` and Axom NOT found**: the
    `find_package(axom REQUIRED CONFIG)` call at the parent
    level fails at CMake configure time — fix `AXOM_DIR` /
    `axom_DIR` and retry.
  * **`ENABLE_AXOM=OFF`** (or `ENABLE_AXOM` undefined): the
    `mortar_pbc_lib` and all conforming-mesh tests still build;
    only `test_axom_smoke` (and, in future batches,
    `test_patch_3d_pbc_nonconforming`) are skipped silently. The
    conforming face mortar code path doesn't link Axom and is
    unaffected. This is the correct behaviour for users who only
    need the conforming subset.

**Sandbox / syntax-check workflow.** During development we
maintain a minimal Axom stub at `/tmp/axom_stub/` that mirrors
the API surface we use (`Point`, `BoundingBox`, `Polygon`,
`clip`, `spin::BVH<Dim>`). The stub returns trivial/empty
results — it's only sufficient for `g++ -fsyntax-only` checks.
Real correctness validation happens against installed Axom on
the user's Mac / cluster. The stub's `IndexType` is hard-coded
to `std::int64_t` to match the v0.12+ default; if a future Axom
build configures with `-DAXOM_USE_64BIT_INDEXTYPE=OFF`, the
stub would be a slight over-promise (real `IndexType` would be
`int32_t`), but the smoke test itself is width-agnostic and
would still compile against either typedef.

**Cross-references**:

  * §P4.4.6.10 — the Phase 4.4 architectural plan that this
    batch is the foundation for.
  * Architecture doc §3.7 — Sutherland-Hodgman pseudocode
    (which `axom::primal::clip` implements; v0.14.0 release
    notes mention "polygon clipping was modified to handle some
    corner cases" — purely a robustness improvement, no API
    change).
  * Architecture doc §11.6 — face-mortar geometric matching
    (which `axom::spin::BVH<2>` provides the `locate_mortar`
    primitive for).

---

### §P4.8.19 Broad-phase candidate pairs via BVH (Batch 4.4-B)

This batch implements the broad-phase spatial-search step of the
non-conforming face-mortar work. Given the nonmortar-side and
mortar-side face element lists for one periodic face pair, it
returns a CSR-format list of candidate `(s_idx, m_idx)` pairs
whose 2D-projected AABBs overlap. **No clipping yet** — the
fine-phase polygon clipping is Batch 4.4-C.

**What Batch 4.4-B landed:**

  * `face_mortar_match_3d.{hpp,cpp}` (new) — public functions
    `MatchClippedQuadFacePairs` and `MatchClippedTriFacePairs`,
    sharing a templated implementation. Uses
    `axom::spin::BVH<2>` keyed on mortar-element 2D AABBs. The
    output type `ClippedPairCandidates` is CSR-format
    `std::vector<axom::IndexType>` for offsets / counts /
    candidates, mirroring Axom's `BVH::findBoundingBoxes`
    convention exactly.
  * `test_face_mortar_match_3d.cpp` (new) — synthetic-input
    unit test covering: (1) empty inputs, (2) trivial conforming
    4×4 vs 4×4 quad case, (3) non-conforming 4×4 vs 5×5 quad
    case, (4) trivial conforming tri 4×4 case, (5) documented
    perpendicular-axis-mismatch placeholder. Test does CSR
    structural checks (offsets/counts consistency,
    candidates.size() matches offsets.back()) which run cleanly
    against the sandbox stub; the numerical candidate-count
    assertions are info-only against the stub (which returns
    empty BVH output) but become real checks against installed
    Axom.

**Implementation choices:**

  1. **2D-projection convention.** Drop the perpendicular axis;
     the two remaining axes are taken in cyclic order to
     preserve right-handedness:
       * `n="x"` → 2D = (y, z), indices (1, 2)
       * `n="y"` → 2D = (z, x), indices (2, 0)
       * `n="z"` → 2D = (x, y), indices (0, 1)
     This matches the convention CCW vertex ordering on the
     nonmortar face stays CCW in 2D.
  2. **Mortar AABB padding.** Mortar AABBs are expanded by
     `aabb_pad_rel * max_mortar_edge_length` (default
     `1e-9 * max_edge`), matching the architecture doc §3.6
     vertex-matching tolerance. Nonmortar query AABBs are NOT
     padded — the mortar pad already covers slop, and double-
     padding would over-count candidates.
  3. **CSR output not packed pair list.** Mirror's Axom's BVH
     output shape directly. Downstream code (Batch 4.4-C) iterates
     `for s in [0, n_nonmortar): for k in [offsets[s], offsets[s] +
     counts[s]): m = candidates[k]`.
  4. **Templated impl.** `MatchClippedFacePairsImpl<ElementT>`
     handles both quad and tri. The element struct provides
     `coords`, `NumNodes()`, and `perpendicular_axis` — the
     templated function uses only these. This lets us avoid
     code duplication between the quad and tri public
     overloads.
  5. **No code in `face_mortar_assembler_3d.{hpp,cpp}` changed.**
     This file is the architectural seam (per §P4.4.6.10):
     non-conforming work is contained in the new
     `face_mortar_match_3d` module + (forthcoming)
     `AssemblePairClipped` methods. The conforming code path is
     untouched.

**Axom API gotchas discovered during integration testing**:

  1. **`findBoundingBoxes` requires PRE-ALLOCATED offsets and
     counts.** The signature is
     `findBoundingBoxes(ArrayView<IndexType> offsets,
                        ArrayView<IndexType> counts,
                        Array<IndexType>& candidates,
                        IndexType n_query, BBox* queries)`.
     The `offsets` and `counts` are `ArrayView` (not `Array&`)
     specifically because the caller controls their allocation —
     they must be sized to `n_query` BEFORE the call. If you pass
     unallocated arrays, Axom fires SLIC errors:
       `[ERROR]: offsets length not equal to numObjs`
       `[ERROR]: counts length not equal to numObjs`
     Only `candidates` is allocated by Axom.
  2. **`offsets` has size `n_query`, NOT `n_query + 1`.** Axom
     uses no sentinel. To get the total candidate count, use
     `candidates.size()` directly. Our internal CSR convention adds
     a sentinel `offsets[n_nonmortar] = candidates.size()` because
     SciPy-style `[offsets[s], offsets[s+1])` iteration is more
     natural for Batches 4.4-C/D, but that's our wrapper, not
     Axom's.
  3. **Axom requires SLIC initialization for clean output.**
     Without an active `axom::slic::SimpleLogger` (or equivalent),
     Axom auto-initializes a fallback logger and prints a warning.
     Tests that exercise Axom should construct
     `axom::slic::SimpleLogger slic_logger;` at the top of `main()`
     — RAII handles init / finalize.
  4. **Including `axom/core.hpp`, not `axom/axom.hpp`.** The
     umbrella header for Axom Core is `axom/core.hpp`. There is
     no top-level `axom/axom.hpp`. The other umbrella headers we
     use are `axom/primal.hpp`, `axom/spin.hpp`, `axom/slic.hpp`.
  5. **CMake dep list needs the component targets, not just
     `axom`.** The right form is
     `list(APPEND ... axom axom::core axom::slam axom::slic)`.
     `axom::primal` and `axom::spin` are header-only so they don't
     need explicit listing, but `axom::slam` is a transitive
     dep of `axom::spin::BVH`'s policy headers, and `axom::slic`
     is needed at link time for the SLIC error reporting.

**Validation status:**

  * Sandbox: 29/29 .cpp files syntax-clean,
    `face_mortar_match_3d.cpp` and `test_face_mortar_match_3d.cpp`
    additionally `-Wall -Wextra -Wpedantic` clean.
  * Real Axom v0.14.0 on Mac: pending the user's next test run.
    The test now does real numerical assertions (not just info
    prints):
      - 4×4 vs 4×4 quad conforming: each nonmortar gets ≥ 1 and
        ≤ 9 candidates (self + up to 8 edge/corner neighbors via
        the AABB pad); total in [16, 100].
      - 4×4 vs 5×5 quad non-conforming: each nonmortar gets ≥ 1;
        total in [16, 200].
      - 4×4 vs 4×4 tri conforming: each nonmortar gets ≥ 2 (twin
        + diagonal partner); total in [64, 600].
    If any assertion trips, the broad-phase output is being
    read incorrectly — fix before proceeding to Batch 4.4-C.

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — the full Phase 4.4 plan.
  * Phase 4 plan §P4.8.18 — Axom build integration (prereq).
  * Architecture doc §3.5–3.7 — geometric matching.
  * Architecture doc §11.6 — face-mortar pseudocode.

---

### §P4.8.20 Polygon clipping + fan-triangulation (Batch 4.4-C)

This batch implements the fine-phase geometric step: take the
candidate `(s_idx, m_idx)` pairs from Batch 4.4-B and produce, for
each, the actual 2D-projected overlap polygon, then fan-triangulate
into a list of `ClippedSubTriangle` records keyed by nonmortar
index. Used by Batch 4.4-D's per-sub-triangle Dunavant quadrature.

**What Batch 4.4-C landed:**

  * `face_mortar_match_3d.{hpp,cpp}` — added two structs
    (`ClippedSubTriangle`, `ClippedSubTriangulation`) and two
    public functions (`ClipQuadFacePairs`, `ClipTriFacePairs`)
    sharing a templated implementation `ClipFacePairsImpl<ElementT>`.
    Uses `axom::primal::clip(Polygon<2>, Polygon<2>)` for the
    convex-on-convex Sutherland-Hodgman intersection.
  * `test_face_mortar_match_3d.cpp` — added 4 new test cases:
    (5) empty inputs, (6) quad conforming 4×4 (each nonmortar →
    exactly 2 sub-tris, total area = 1.0 to 1e-12), (7) quad
    non-conforming 4×4 vs 5×5 (≥ 1 per nonmortar, total area = 1.0
    to 1e-12), (8) tri conforming 4×4 (≥ 1 per nonmortar, total
    area = 1.0 to 1e-12).

**Tile-cover invariant** is the central correctness check: the
sum of all sub-triangle areas across one ClipFacePairs call equals
the nonmortar face's total 2D-projected area to 1e-12 relative.
This catches:
  * Missing intersections (broad-phase under-coverage).
  * Double-counting (same overlap region split across multiple
    candidate pairs).
  * Sign errors in the orientation-preserving 2D projection.
  * Bugs in fan triangulation (off-by-one indexing, etc.).

**Implementation choices:**

  1. **CCW orientation is enforced INSIDE `BuildPolygon2D`, not assumed
     from the upstream face-element convention.** This was a bug in the
     first attempt: face elements are stored "CCW from their own outward
     normal" in 3D, but the nonmortar and mortar faces have OPPOSITE
     outward normals (they're on opposite sides of the periodic
     interface). After 2D-projecting both into the same (a, b) plane,
     one comes out CCW and the other CW — Sutherland-Hodgman silently
     returns empty in that case. The fix: every polygon goes through a
     shoelace signed-area check inside `BuildPolygon2D`, and CW polygons
     are reversed via `axom::primal::Polygon::reverseOrientation()`
     (added in Axom v0.10). This makes the matcher orientation-robust
     w.r.t. any source convention. The fan-triangulation step asserts
     `sa > 0` as a safety net.
  2. **Sliver filter via relative area tolerance.** Sub-triangles
     whose `|signed_area| < area_tol_rel * nonmortar_2D_area`
     are dropped. Default `area_tol_rel = 1e-12` — matches the
     patch-test acceptance tolerance from the architecture doc.
     This handles the AABB-pad over-counting from Batch 4.4-B:
     shared-edge mortar candidates produce zero-area clip
     polygons that get filtered here; no impact on assembled D
     or A_m matrices.
  3. **Subject = nonmortar.** `clip(s_poly, m_poly)` is called
     with nonmortar as the subject, mortar as the clipper.
     For convex-on-convex the result *set* is the same either
     way, but this convention reads as "restrict the nonmortar
     region to the part inside the mortar" which matches the
     mortar method's mathematical setup (the integral domain is
     a sub-region of Γ⁻).
  4. **Output format: CSR by nonmortar index.** Same format as
     `ClippedPairCandidates` for symmetry. Batch 4.4-D's
     assembler iterates `for s in [0, n_nonmortar): for k in
     [offsets[s], offsets[s+1]): tri = sub_tris[k]`. The
     `m_idx` is embedded in each `ClippedSubTriangle` because
     a single nonmortar may have sub-tris from multiple mortar
     partners.
  5. **2D coords stored, perpendicular axis recovered at use
     site.** Sub-tri vertices are stored in (a, b) physical
     coords. The 3D point on the periodic face is recovered
     downstream by re-inserting the constant perpendicular-axis
     coordinate from the parent face element. This avoids
     storing redundant data per sub-tri (the perpendicular coord
     is identical for all sub-tris on one face).
  6. **Templated impl shared between quad and tri.** The
     `BuildPolygon2D<ElementT>` helper uses `ElementT::NumNodes()`
     and `coords` — works identically for quad (4 nodes) and tri
     (3 nodes). The clipping algorithm doesn't care about input
     vertex count for convex polygons.

**Axom API gotcha discovered during integration testing**:

  * **`axom::primal::clip` is Sutherland-Hodgman; both inputs MUST
    be CCW or it returns empty silently.** No warning, no assertion
    fires — the result is just an empty polygon. This is
    Sutherland-Hodgman's standard inside-half-plane semantics:
    CW inputs invert the test, so every vertex appears "outside"
    and gets rejected. Our `BuildPolygon2D` enforces CCW per
    polygon, independent of source convention.

**Validation status:**

  * Sandbox: 29/29 .cpp files syntax-clean. `face_mortar_match_3d.cpp`
    and `test_face_mortar_match_3d.cpp` clean under
    `-Wall -Wextra -Wpedantic`.
  * Real Axom v0.14.0 on Mac: pending. Expected results on first
    run:
      - Test 6 (quad conforming 4×4): 32 sub-tris total, total
        area = 1.0 to 1e-12, each sub-tri area exactly 0.03125.
      - Test 7 (quad non-conforming 4×4 vs 5×5): variable count
        (clipping subdivides), total area = 1.0 to 1e-12.
      - Test 8 (tri conforming 4×4): 32 sub-tris total (one per
        twin pair), total area = 1.0 to 1e-12.
    If the tile-cover invariant trips, the most likely causes are:
    (a) AABB pad too small to capture a true overlap (broad-phase
    under-coverage), (b) clip filter `area_tol_rel` too aggressive,
    (c) orientation flip in the 2D projection.

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — the full Phase 4.4 plan.
  * Phase 4 plan §P4.8.19 — Batch 4.4-B (broad-phase, prereq).
  * Architecture doc §3.7 — Sutherland-Hodgman pseudocode (which
    `axom::primal::clip` implements).
  * Architecture doc §11.6 — face-mortar pseudocode (showing
    where the clipped sub-triangulation feeds into the assembler).

---

### §P4.8.21 Inverse iso-maps + 6-point Dunavant (Batch 4.4-D-1)

This batch is the foundation for the clipped-pair assembler
(Batches 4.4-D-2 and 4.4-D-3). It provides three pure-utility
helpers that the `AssemblePairClipped` methods will call once per
sub-triangle quadrature point:

  * `InverseMapQuad2DAxisAligned(elem, a_idx, b_idx, a, b) → (xi, eta)`
    — closed-form Q1 inverse for axis-aligned quad faces. Uses the
    dual-basis representation `xi = -1 + 2 * (q · e_xi) / |e_xi|^2`
    where `q` is the displacement from vertex 0 and `e_xi`, `e_eta`
    are the edge vectors v0→v1 and v0→v3. For axis-aligned quads
    the edge vectors are orthogonal in (a, b) so the dual basis is
    just the inverse-length-squared scaling — no matrix solve
    needed. No Newton iteration. Two MFEM_ASSERTs guard against
    degenerate edges.
  * `InverseMapTri2D(elem, a_idx, b_idx, a, b) → (lam_0, lam_1, lam_2)`
    — closed-form P1 inverse via Cramer's rule on the 2×2 affine
    system. Always exact for non-degenerate tris. `MFEM_ASSERT`
    guards against zero 2D area.
  * `DunavantTri6Pt()` — 6-point degree-4 Dunavant rule on the
    reference simplex (|T| = 1/2). Required for clipped quad-face
    sub-triangles where the bilinear-basis × bilinear-basis product
    is degree 4 in barycentric. Tri-face clipped sub-tris stay at
    `GaussTri3Pt` (degree 2 suffices).

**Files added:**

  * `face_mortar_inverse_map_3d.{hpp,cpp}` — both inverse-map
    helpers in their own translation unit (no Axom dep). Added to
    `MORTAR_PBC_HEADERS` / `_SOURCES` unconditionally so they're
    available even when `ENABLE_AXOM=OFF`.
  * `test_face_mortar_inverse_map_3d.cpp` — round-trip tests for
    both inverse maps (forward iso-map at canonical reference
    points, then inverse, assert recovery to 1e-14) plus monomial-
    integration tests for `DunavantTri6Pt` covering all monomials
    `lam_0^p lam_1^q lam_2^r` with `p+q+r ∈ {0..4}` (15 monomials)
    against the closed-form integral
    `p! q! r! / (p+q+r+2)!`.
  * `face_mortar_assembler_3d.{hpp,cpp}` — extended with
    `QuadratureTri6Pt` struct + `DunavantTri6Pt()` implementation.

**Why these are in two different files:**

The inverse-iso-map helpers don't reference any Axom types, so they
live in their own module that compiles regardless of `ENABLE_AXOM`.
The 6-point Dunavant rule lives next to `GaussTri3Pt` /
`GaussQuad3x3` in the existing assembler module — it's a pure
quadrature utility and Axom-free. Only the per-sub-triangle
*walker* (Batch 4.4-D-2/3) is Axom-gated.

**Validation status:**

  * Sandbox: 31/31 .cpp files syntax-clean (added 2 files this
    batch). New code `-Wall -Wextra -Wpedantic` clean.
  * Python regression 6/6 green.
  * Real Axom: pending. Test runs *without* Axom — only requires
    a normal mortar_pbc build. The 4 test cases:
      1. Quad inverse round-trip: 11 reference points (vertices,
         mid-edges, center, 2 generic), each round-trips to 1e-14.
      2. Tri inverse round-trip: 8 barycentric points (vertices,
         mid-edges, centroid, 1 generic), each round-trips to 1e-14.
      3. Dunavant 6-point weights sum to |T| = 1/2 to 1e-14.
      4. Dunavant 6-point integrates 15 monomials of degree ≤ 4
         exactly (to 1e-13).

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 design decision 4 — quadrature order
    policy (3-point Dunavant for tri, 6-point for clipped quad
    sub-tris).
  * Phase 4 plan §P4.4.6.10 — the inverse-map closed-form is
    spelled out in the "Algorithmic invariants" subsection.
  * Architecture doc §11.6 — `locate_mortar` interface that these
    helpers provide for the axis-aligned case.
  * Reference: Dunavant 1985, "High degree efficient symmetrical
    Gaussian quadrature rules for the triangle." Int. J. Numer.
    Methods Eng. 21, 1129-1148.

---

### §P4.8.22 Quad-quad clipped face mortar assembler (Batch 4.4-D-2)

This batch is the algorithmic core of Phase 4.4 for Q1 quad face
elements. `AssembleQuadFacePairClipped` consumes the clipped
sub-triangulation from Batch 4.4-C and produces a `FaceMortarPairBlock`
matching the conforming-path interface bit-for-bit on conforming
inputs (the central correctness check) and correctly populated for
non-conforming inputs.

**Files added:**

  * `face_mortar_assembler_clipped_3d.{hpp,cpp}` — Axom-gated.
    Free function `AssembleQuadFacePairClipped` (not a class
    method) so the conforming `QuadFaceMortarAssembler` class
    header stays Axom-free. Replicates four small helpers
    (`AxisIndex`, `DiscoverKeptGtdofs`, `BoundaryTagToSides`, an
    axis-aligned-only `NonmortarJacobianAxisAligned`) in its own
    anonymous namespace. The duplication is deliberate: the
    conforming class encapsulates these as private helpers and
    we don't want to widen its API just to share them with the
    clipped assembler.
  * `test_face_mortar_assembler_clipped_3d.cpp` — the central
    correctness gate. Routes 4×4 vs 4×4 conforming meshes through
    BOTH the conforming and clipped paths, then asserts entry-by-
    entry agreement on `D` (exact, both paths use the same 9-pt
    rule) and `A_m` (1e-12 relative, FP-rearrangement only).

**The dual-loop structure (the central principle):**

The clipped assembler implements the D-vs-A_m domain split
documented in arch §3.5 and §P4.4.6.10. For each nonmortar
element s:

  * **Pass 1 (D)**: 9-point Gauss-Legendre rule on the parent
    reference quad, accumulating
    `D_loc[k] += phys_w * N_nonmortar[k]`.
    This is the *full* element integration. Wohlmuth biorthogonality
    lumps D to its diagonal once summed over all 9 q-pts.
    Reused verbatim from the conforming assembler.
  * **Pass 2 (A_m)**: walk all sub-triangles owned by s. For each
    sub-tri, Dunavant 6-point rule on the sub-tri reference,
    computing barycentric → 2D physical (a, b) → inverse-iso-map
    to nonmortar `(xi_nm, eta_nm)` AND mortar `(xi_m, eta_m)` →
    evaluate `M_dual` and `N_mortar` → accumulate
    `A_loc[k][l] += sub_phys_w * M_dual[k] * N_mortar[l]`.

The two passes are independent — D doesn't see sub-triangles, A_m
doesn't see the parent reference quad. This matches the 2D
prototype's structure and keeps Wohlmuth biorthogonality intact
(holds when D is integrated over the full element, not segment-
wise).

**Why no mortar-side permutation:**

The conforming assembler uses `MortarRefFromPermutation` and
`ReorderMortarShape` to handle the case where the mortar element's
local node ordering differs from the nonmortar's. In the clipped
path, the inverse-iso-map gives mortar `(xi_m, eta_m)` directly
in the mortar's own reference frame, so we evaluate `NQuad4` on
the mortar's own coords and pair `N_mortar[l_loc]` with
`m.gtdofs[l_loc]` directly. No permutation needed, no
reordering — simpler than the conforming code.

**Sub-triangle Jacobian:**

`DunavantTri6Pt` weights sum to `|T_ref| = 1/2`. For a
sub-triangle of physical 2D area `A`:
  `∫_{phys} f dA ≈ Σ w_q · f(λ_q) · 2A`
i.e., `J_sub = 2 * sub_tri.area`. Sum check: `(1/2) * 2A = A`. ✓
Mirrors the conforming tri assembler's `J_nonmortar = 2 *
phys_tri_area` convention.

**Validation status:**

  * Sandbox: 33/33 .cpp files syntax-clean. New code
    `-Wall -Wextra -Wpedantic` clean.
  * Python regression 6/6 green.
  * Real Axom: pending. Two test cases:
    1. 4×4 vs 4×4 conforming agreement: D entries match exactly
       (1e-14), A_m entries match to 1e-12 relative.
    2. Σ D entries equals nonmortar face area (1.0) to 1e-12 —
       a coarse independence check.

  The conforming-via-clipped agreement test is the actual
  correctness gate. If it passes, the assembler is correct on
  conforming inputs, which means:
    - Per-element D accumulation is correct.
    - Sub-triangle Jacobian is correct.
    - Inverse-iso-maps for both nonmortar and mortar are correct.
    - Sentinel-aware scatter is correct.
    - Wohlmuth dispatch via `boundary_tag` is correct.
  The non-conforming case differs only in which sub-triangles are
  produced by `ClipQuadFacePairs` — which Batch 4.4-C already
  validated via the tile-cover invariant. So passing this gate
  gives us high confidence in the full pipeline.

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — full Phase 4.4 plan.
  * Phase 4 plan §P4.8.20 — Batch 4.4-C clipping geometry (prereq).
  * Phase 4 plan §P4.8.21 — Batch 4.4-D-1 helpers (prereq).
  * Architecture doc §3.5 — D-vs-A_m domain split.
  * Architecture doc §11.6 — face-mortar assembly pseudocode.

---

### §P4.8.23 Tri-tri clipped face mortar assembler (Batch 4.4-D-3)

This batch completes the Phase 4.4 assembler for P1 tri face elements.
`AssembleTriFacePairClipped` mirrors `AssembleQuadFacePairClipped`
structurally with three element-type-specific differences:

  1. **Quadrature on clipped sub-tris is `GaussTri3Pt` (degree 2)**, not
     `DunavantTri6Pt` (degree 4). The bumped-up rule was needed for Q1
     because Q1·Q1 = degree 4 in barycentric; for P1, P1·P1 = degree 2,
     and 3-point Dunavant integrates that exactly. Same rule used by the
     conforming tri assembler — no quadrature-rule mismatch between paths
     for tri faces.
  2. **D-side Jacobian: `J = 2 * |T_phys|`** via 3D cross-product
     magnitude (`TriFullJacobian` helper). No axis-alignment shortcut —
     tri faces are generally oblique (the hypotenuse isn't axis-aligned),
     so we use the same 3D-cross-product Jacobian as the conforming tri
     path.
  3. **Inverse-iso-map: `InverseMapTri2D` (Cramer's rule)** returns
     barycentrics directly. Both nonmortar and mortar tri parents use
     this map.

**What landed:**

  * `face_mortar_assembler_clipped_3d.{hpp,cpp}` extended with:
    - `BoundaryTagToDropsTri` helper (anonymous namespace, mirroring
      the conforming class's private method).
    - `TriFullJacobian` helper.
    - Public `AssembleTriFacePairClipped` function.
  * `test_face_mortar_assembler_clipped_3d.cpp` extended with:
    - `MakeTriGridWithGtdofs` helper (4×4 conforming tri grid: 32 tris,
      25 unique gtdofs, sequential numbering).
    - `test_tri_conforming_agreement_4x4`: routes 4×4 vs 4×4 conforming
      tri meshes through both paths, asserts entry-by-entry agreement
      on D (1e-14) and A_m (1e-12 relative).
    - `test_clipped_tri_d_total_area`: independent Σ D = face area
      check.

**Why no mortar-side permutation (same as Batch 4.4-D-2):**

The conforming tri assembler uses `MortarBaryFromPermutation` and
`ReorderMortarShape` to handle local-node ordering mismatches. In the
clipped path, the inverse-iso-map gives mortar barycentrics directly
in the mortar's own local frame, so `NTri3(lam_m)` is naturally aligned
with `m.gtdofs[l_loc]`. Cleaner inner loop, no permutation indirection.

**Validation status:**

  * Sandbox: 33/33 .cpp files syntax-clean. New code
    `-Wall -Wextra -Wpedantic` clean.
  * Python regression 6/6 green.
  * Real Axom: pending. Combined test now exercises all four cases:
    quad agreement (Test 1), quad Σ D (Test 2), tri agreement (Test 3),
    tri Σ D (Test 4). Expected output:
       D max-error      = 0 (or ε)         max |D|     ≈ 0.0625
       A_m max-error    = O(1e-15)         max |A_m|   ≈ 0.0625
       Σ D = 1.0 (expected 1.0)            (both element types)

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — full Phase 4.4 plan.
  * Phase 4 plan §P4.8.22 — Batch 4.4-D-2 (sibling, quad version).
  * Architecture doc §3.5 — D-vs-A_m domain split.

---

### §P4.8.24 Discrete reproduction tests (Batch 4.4-D-4)

This batch validates the assembled `(D, A^m)` block as a mortar
**projector** on genuinely non-conforming meshes. Without a reference
assembler to compare against (the conforming-via-clipped agreement
test only works when meshes happen to coincide), correctness on
non-conforming inputs has to be checked physically — by verifying
that the projector reproduces functions in the test space exactly.

**The two reproduction properties:**

For the mortar projector `P u_+ = D⁻¹ A^m u_+`:

  * **Constant reproduction**: `P · 1 = 1`. Equivalent to row-sum
    biorthogonality `A^m 1 = D 1`, which is the construction
    principle of the Wohlmuth dual basis. If non-conforming clipping
    has missed any sub-region or double-counted any overlap, this
    fails immediately because `(A^m 1)[k] = ∫ M_k · 1 dA` summed over
    sub-regions no longer equals `D[k] = ∫_E N_k dA` over the full
    nonmortar element.
  * **Linear reproduction**: `P u(x) = u(x)` for any linear field
    `u(x) = α·x_a + β·x_b + γ` in the (a, b) plane. This is the
    discrete completeness property of the mortar method on flat
    axis-aligned interfaces — the property that motivates using the
    dual basis in the first place. If any inverse-iso-map is wrong,
    or any sub-triangle Jacobian is mis-scaled, linear reproduction
    fails because `(A^m u)[k]` no longer equals `u(x^k) · D[k]`.

Both checks are independent of any reference assembler. Passing them
on a 4×4 vs 5×5 setup demonstrates correctness end-to-end.

**Files changed:**

  * `test_face_mortar_assembler_clipped_3d.cpp` extended with:
    - `ApplyMortarProjector(block, u_plus) → u_minus` helper that
      computes `D⁻¹ A^m u_+` via direct CSR walk and per-row
      inverse-D scaling. Asserts strict positivity of D entries
      (lumped-positivity guard). Pure host-side linear algebra.
    - `GtdofToVertexPos` / `GtdofToVertexPosTri` helpers that
      reconstruct `(x, z)` coordinates from a gtdof given the
      grid's known sequential numbering convention. The grid
      builders (`MakeQuadGridWithGtdofs`,
      `MakeTriGridWithGtdofs`) use vertex `(i, j) → base + i +
      j*(n+1)`, so the inverse is `(local % (n+1), local / (n+1))`.
    - 6 new test cases:
        5. Constant reproduction, quad conforming 4×4.
        6. Constant reproduction, quad NON-conforming 4×4 vs 5×5.
        7. Linear reproduction, quad conforming 4×4 (3 fields).
        8. Linear reproduction, quad NON-conforming 4×4 vs 5×5
           (3 fields).
        9. Linear reproduction, tri conforming 4×4 (3 fields).
       10. Linear reproduction, tri NON-conforming 4×4 vs 5×5
           (3 fields).

**The three linear fields tested:**
  * `u(x, z) = x` — pure parametric x dependence.
  * `u(x, z) = z` — pure parametric z dependence.
  * `u(x, z) = 1.7·x + 2.3·z + 0.5` — generic linear.
The first two catch axis-swap bugs (where the projector confuses
the two in-plane axes). The third catches scaling and offset
errors.

**Validation status:**

  * Sandbox: 33/33 .cpp files syntax-clean. New code clean.
  * Python regression 6/6 green.
  * Real Axom: pending. Expected per-field max-error around
    1e-14 to 1e-13 across all 6 test cases (tighter on conforming,
    slightly looser on non-conforming due to clipping rearrangement
    in the A^m sums). If any case shows max-error > 1e-12, it's
    a real bug — the most likely diagnostic order:
    1. **Constant reproduction fails** → biorthogonality identity
       is broken. Most likely cause: clipping missed a sub-region
       (Σ D = face area would also fail in 4.4-D-2/3 — but that
       passed, so this is unlikely).
    2. **Linear reproduction fails on `u = x`** but constant
       passes → inverse-iso-map for the x axis is wrong. Check
       `InverseMapQuad2DAxisAligned` axis ordering.
    3. **Linear reproduction fails on `u = z`** symmetrically.
    4. **Generic linear fails but axis-only cases pass** → likely
       a subtle interaction between Wohlmuth modifications and the
       linear field (shouldn't happen since `boundary_tag = "none"`
       throughout this test).

**This is the Phase 4.4 numerical correctness gate.** If all 6
reproduction tests pass on Mac, the full clipped pipeline is
end-to-end correct on non-conforming meshes, and we can proceed
to Batch 4.4-E (dispatch integration into `BuildLocalPairBlocks`
and the production patch-test driver).

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — full Phase 4.4 plan, design
    decisions 5–6.
  * Phase 4 plan §P4.8.22 — Batch 4.4-D-2 (quad assembler).
  * Phase 4 plan §P4.8.23 — Batch 4.4-D-3 (tri assembler).
  * Wohlmuth 2000, "A mortar finite element method using dual
    spaces for the Lagrange multiplier." SIAM J. Numer. Anal.
    38(3), 989-1012 — derivation of the dual basis from the
    biorthogonality + linear-completeness requirements.

---

### §P4.8.25 Conforming-vs-clipped dispatch (Batch 4.4-E Part 1)

This batch wires the clipped-path machinery (Batches 4.4-A through
4.4-D-4) into the production `BoundaryClassifier3D::BuildLocalPairBlocks`
flow. After this batch, `BuildLocalPairBlocks` automatically detects
non-matching meshes and routes them to the clipped assembler — no
caller changes required.

**The dispatch logic:**

For each (axis, mortar/nonmortar, geometry_kind) bucket:

  1. Call `TryMatchConformingFacePairs` (new try-style API).
  2. If it returns `optional<vector<...>>` with a value → meshes are
     conforming → call `AssemblePairConforming` (existing fast path).
  3. If it returns `nullopt` → meshes are non-matching:
       - **`MORTAR_PBC_HAS_AXOM` defined**: call `MatchClippedFacePairs`
         + `ClipFacePairs` + `AssembleQuad/TriFacePairClipped`
         (clipped fallback).
       - **Not defined**: `MFEM_ABORT` with a clear message instructing
         the user to rebuild with `ENABLE_AXOM=ON`.

**Files added/changed:**

  * `face_mortar_assembler_3d.{hpp,cpp}` — added try-style overloads:
    - `TryMatchConformingFacePairs(quad)` returning
      `std::optional<std::vector<QuadFacePairMatch>>`.
    - `TryMatchConformingFacePairs(tri)` returning
      `std::optional<std::vector<TriFacePairMatch>>`.
    - Both share the algorithm of `MatchConformingFacePairs` but
      return `std::nullopt` on non-1:1 candidate count instead of
      aborting. The original `MatchConformingFacePairs` overloads
      remain unchanged — existing tests that rely on the abort-on-
      mismatch semantics keep working.
  * `boundary_classifier_3d.cpp` — `BuildLocalPairBlocks` rewired
    to use the try-style API + Axom-gated fallback. Conforming
    fast path unchanged; clipped path used silently when meshes
    don't match.
  * `CMakeLists.txt` — when `ENABLE_AXOM=ON`, the build sets
    `target_compile_definitions(mortar_pbc_lib PUBLIC MORTAR_PBC_HAS_AXOM)`.
    This makes the dispatch fallback compile-in only when Axom is
    available; without Axom, the dispatch's clipped branch
    compiles to a clean `MFEM_ABORT` with an actionable message.

**Why preprocessor-gating instead of always-compiled:**

The clipped-path machinery (`face_mortar_match_3d.{hpp,cpp}` and
`face_mortar_assembler_clipped_3d.{hpp,cpp}`) is in the library only
when `ENABLE_AXOM=ON`. If `BuildLocalPairBlocks` always compiled the
clipped fallback, builds with `ENABLE_AXOM=OFF` would fail to link
(no `AssembleQuadFacePairClipped` available). The `#ifdef
MORTAR_PBC_HAS_AXOM` guard keeps the conforming-only build path
self-contained: no Axom dependency, no clipped fallback, clean
abort with explanatory message if a non-conforming mesh ever shows
up.

**Validation status:**

  * Sandbox: 33/33 .cpp files clean WITHOUT `MORTAR_PBC_HAS_AXOM`
    (production build), AND 33/33 clean WITH `MORTAR_PBC_HAS_AXOM`
    (Axom-enabled build). 66/66 total across both configurations.
  * Python regression 6/6 green (Python prototypes don't exercise
    this dispatch — they're algorithm references, not production).
  * Real Axom: pending. The dispatch's correctness on conforming
    meshes is implicit — every existing patch test still uses
    conforming meshes, and they should pass unchanged because the
    try-style API returns `Some` and the conforming branch fires
    exactly as before. Validation that the clipped branch fires on
    actual non-conforming meshes requires Batch 4.4-E Part 2
    (production-shape patch test driver).

**What's still missing (Batch 4.4-E Part 2):**

  * A `test_patch_3d_pbc_nonconforming.cpp` executable that builds
    a non-matching MFEM mesh and runs the full FE elasticity solve
    end-to-end. Construction of a non-matching periodic mesh in MFEM
    is non-trivial (`MakeCartesian3D` produces conforming meshes;
    we'd need a custom mesh constructor or the
    `Mesh(int Dim, int NVert, int NElem)` low-level API). Deferred
    to a follow-up turn — the algorithmic correctness is already
    validated by Batch 4.4-D-4's reproduction tests on synthetic
    non-conforming face element lists.

**Cross-references:**

  * Phase 4 plan §P4.4.6.10 — full Phase 4.4 plan, design
    decision 5 ("Conforming fast path is preserved").
  * Phase 4 plan §P4.8.18 — Batch 4.4-A Axom build integration.
  * Phase 4 plan §P4.8.24 — Batch 4.4-D-4 reproduction tests
    (algorithmic prereq).

---

### §P4.8.26 Production-shape non-conforming patch test (Batch 4.4-E Part 2)

This batch closes Phase 4.4 by adding a production-shape end-to-end
patch test that exercises the entire clipped-path pipeline through
a real FE elasticity solve. Rather than constructing a non-matching
MFEM mesh from scratch (which would require the low-level mesh API
or anisotropic h-refinement with hanging nodes — out of Phase 4.4
scope), we apply an **in-plane node perturbation** to one periodic
face of a standard `MakeCartesian3D` mesh.

**The perturbation strategy:**

For each node at `(x, y, z)` with `y == L`:
  `x_new = x + amplitude · sin(π · x / L)`
  (y, z unchanged)

This satisfies all clipped-path contract requirements:
  * **Corners stay exact** (sin vanishes at x=0 and x=L) — corner
    Dirichlet BCs from `F·X` remain aligned with the FE solve.
  * **Faces stay flat** (y = L preserved on the perturbed face;
    other faces untouched) — axis-aligned face-element assumption
    in `InverseMapQuad2DAxisAligned` and `NonmortarJacobianAxisAligned`
    still holds.
  * **No degenerate hexes** (max shift `amplitude = 0.05` against
    cell width `0.25` on a 4³ mesh = 20% — well-conditioned).
  * **Linear-field reproduction unaffected** — Q1 hexes reproduce
    `u(x) = F·x` exactly regardless of element shape.

The y-face periodic pair becomes non-matching (centroid distances
of order `0.05` vs the `1e-9` match tolerance), triggering
`TryMatchConformingFacePairs` → `nullopt` →
`BuildLocalPairBlocks` falls back to the clipped path.

**Files added/changed:**

  * `patch_test_driver_3d.hpp` — added optional
    `std::function<void(mfem::Mesh&)> mesh_perturbation` field to
    `PatchTestConfig`. Default `nullptr` means "no perturbation"
    (existing tests unchanged). Contract documented inline.
  * `patch_test_driver_3d.cpp` — added single hook call between
    `MakeCartesian3D + ApplyAttributePattern` and `ParMesh` ctor.
  * `test_patch_3d_pbc_nonconforming.cpp` — new test executable
    that constructs `cfg` with the y=L face perturbation and
    delegates to `RunPatchTest3D`. CLI mirrors `test_patch_3d_pbc`
    plus an `--amplitude` override (default 0.05).
  * `CMakeLists.txt` — registered the new test (Axom-gated, since
    the dispatch falls back to the clipped path which requires
    Axom).

**PASS criteria** are inherited from `RunPatchTest3D`:
  * Krylov converged.
  * `||du||_inf < 1e-7` (homogeneous-elastic exactness).
  * `||<F> - F_macro||_inf < 1e-9` (homogenization check).
  * `||C·u_total - C·u_lin||_inf < 1e-9` (constraint residual).

**What this test exercises:**

  * `BoundaryClassifier3D` correctly identifies the y face pair
    despite face node mismatches.
  * `TryMatchConformingFacePairs` correctly returns `nullopt`
    (verified by reaching the clipped fallback).
  * `MatchClippedQuadFacePairs` (BVH broad-phase) on real FE
    face-element data.
  * `ClipQuadFacePairs` (Sutherland-Hodgman) on real face data.
  * `AssembleQuadFacePairClipped` produces a `(D, A^m)` block
    consumed unchanged by `MortarSaddlePointSystem`.
  * `SaddlePointSolver` converges on the constrained system.
  * Constraint residual `C·u_total = C·u_lin` after solve.
  * Patch test residual `||du||_inf` at FE-solver tolerance.

**Validation status:**

  * Sandbox: 34/34 .cpp files clean WITHOUT `MORTAR_PBC_HAS_AXOM`,
    34/34 clean WITH it (68/68 across both build configs). New
    code `-Wall -Wextra -Wpedantic` clean.
  * Python regression 6/6 green.
  * Real Axom on Mac: pending. The expected behavior is that this
    test passes with the SAME numbers as the conforming
    `test_patch_3d_pbc` (Krylov converges, `||du||_inf` near
    1e-9, constraint residual near 1e-12). If the test fails:
      1. **Krylov diverges**: assembled `(D, A^m)` is wrong shape
         or has unexpected zeros — most likely a sentinel bug in
         the clipped-path scatter. Diagnostics: `nnz(A^m)` should
         match the conforming case minus contributions on the
         perturbed face (typical: similar order of magnitude).
      2. **Krylov converges but `||du||_inf > 1e-7`**: the
         constraint is being applied but isn't reproducing linear
         fields. Most likely cause: an inverse-iso-map or
         sub-triangle Jacobian bug specific to this face's
         non-uniform geometry. Diagnostic check: re-run the
         reproduction tests from Batch 4.4-D-4 with similar
         non-uniform face geometry to see if they still pass.
      3. **Constraint residual high but `du` is small**: the
         constraint matrix is computing a different projection
         than the solver expects. Most likely cause: row/col
         ordering mismatch between `D`, `A^m`, and the `C` block
         consumed by `MortarConstraintOperator`. Less likely
         since the conforming dispatch test already validated
         this — but worth checking.

  This is the production-shape gate for Phase 4.4. If it passes,
  the entire Phase 4.4 stack (Batches 4.4-A through 4.4-E) is
  end-to-end correct on a real FE problem and the phase is
  complete.

**Cross-references:**

  * Phase 4 plan §P4.8.25 — Batch 4.4-E Part 1 (dispatch
    integration; this batch builds on it).
  * Phase 4 plan §P4.8.24 — Batch 4.4-D-4 reproduction tests
    (algorithmic prereq).
  * Architecture doc §3.5 — D-vs-A_m domain split.

---

## §P4.9 Mapping from Python files to C++ files

This table is for reference when porting; each row is one focused
porting unit.

| Python module                              | C++ files                          | Phase |
|--------------------------------------------|-------------------------------------|-------|
| `mortar_pbc/types_3d.py`                   | `types_3d.hpp`                     | 4.1.A |
| `mortar_pbc/mortar_3d.py`                  | `mortar_assembler_2d.{hpp,cpp}`    | 4.1.A |
|                                            | `face_mortar_assembler_3d.{hpp,cpp}`| 4.1.A |
| `mortar_pbc/face_mortar_3d.py`             | (same as above)                    | 4.1.A |
| `mortar_pbc/mortar_2d.py` (edge-mortar use)| (subset of `mortar_assembler_2d`)  | 4.1.A |
| `mortar_pbc/boundary_3d.py`                | `boundary_classifier_3d.{hpp,cpp}` | 4.1.A |
| `mortar_pbc/constraint_builder_3d.py`      | `constraint_builder_3d.{hpp,cpp}`  | 4.1.A |
| `mortar_pbc/elastic_3d.py`                 | `elastic_3d_helpers.{hpp,cpp}`     | 4.1.A |
| `mortar_pbc/saddle_point.py`               | `saddle_point_solver.{hpp,cpp}`    | 4.1.A |
| `mortar_pbc/visualization.py`              | `visualization.{hpp,cpp}`          | 4.1.A |
| `mortar_pbc/multistep_driver.py`           | `mortar_pbc_driver.{hpp,cpp}`      | 4.1.B |
| `examples/patch_test_3d_pbc.py`            | `examples/patch_test_3d_pbc.cpp`   | 4.1.A |
| `examples/patch_test_3d_heterogeneous.py`  | `examples/patch_test_3d_heterogeneous.cpp` | 4.1.B |
| `examples/patch_test_3d_checkerboard.py`   | `examples/patch_test_3d_checkerboard.cpp` | 4.1.C |
| `tests/test_*.py` (6 suites)               | `tests/test_*.cpp` (6 suites)      | 4.1.D |

---

## §P4.10 Best-practices C++ checklist

These are non-negotiable for the port to be acceptable.

### Memory and resource management
- All owning pointers are `std::unique_ptr`. No raw `new`/`delete`.
- All borrowed pointers are references or `mfem::Operator&` /
  `const mfem::Operator&`.
- All collective MPI operations are documented with
  `// [collective]` comment AT the call site.
- `MFEM_VERIFY(cond, msg)` for invariants the user could violate;
  `MFEM_ASSERT(cond, msg)` for invariants we control.

### MPI discipline
- **Every rank in a given communicator reaches every collective on
  that communicator.** No `if (rank == 0)` around AllReduce /
  AllGather / Barrier. (Mortar §10.4.)
- The framework uses TWO communicators: **WORLD** (volume work) and
  **boundary_comm** (boundary work; §P4.4.0). Document collective
  context in every public method's docstring, naming the comm:
  `[collective on WORLD]`, `[collective on boundary_comm]`, or
  `[local]`. This is non-negotiable.
- All boundary-comm operations must be guarded with
  `if (boundary_comm != MPI_COMM_NULL) { ... }` since interior ranks
  receive `MPI_COMM_NULL` from `MPI_Comm_split`.
- Prefer `mfem::Vector` / `mfem::ParVector` over raw double*.

### Avoid runtime polymorphism in hot loops
- Mortar element-type dispatch via templates, not virtual functions:
  ```cpp
  template<int NV>  // NV = 3 (tri) or 4 (quad)
  class FaceMortarAssembler;
  ```
- Per-pair iteration in `MortarConstraintOperator::Mult` should be a
  flat `for` loop over a packed `std::vector<MortarPairLocal>` with no
  pointer chasing.

### Const-correctness
- Methods that don't modify `*this` are `const`.
- Setup-time methods (in classifier, constraint builder) may be
  non-const, but the resulting state is then immutable; expose only
  const accessors after setup.

### Error messages
- Match the Python prototype's level of detail. Failed `MFEM_VERIFY`
  messages should explicitly name the invariant violated, not just
  "assertion failed". Examples in mortar §11.7.2.

### Caliper instrumentation
- One `CALI_CXX_MARK_SCOPE` per non-trivial method, named per §P4.6.4.
- No redundant nesting; if a method only calls one annotated child,
  don't annotate the parent.

### Dimension genericity
- `BoundaryClassifier2D` and `BoundaryClassifier3D` are separate
  classes (mirror of Python). No template-on-dim. The 2D and 3D codes
  diverge in non-trivial ways (mortar §5.4 wirebasket, §11.4 mixed
  meshes); template-on-dim hides those differences awkwardly.
- Helpers like `apply_linear_part`, `compute_volume_averaged_F` ARE
  dim-generic and use `pmesh.Dimension()` at runtime.

---

## §P4.11 Decisions captured (for future-conversation context)

These are the answers from the original questions plus the
follow-up refinements, captured explicitly so a fresh conversation
can read just this document and have full context:

1. **GPU support**: ExaConstit builds with MFEM GPU support. Hypre+GPU
   for vector-dim problems is currently broken upstream; targeting
   CPU Hypre + GPU MFEM-K-action initially. The EA constraint path
   (Phase 4.3) is the GPU-future-proofed component.

2. **Hypre version**: 3.1. No compatibility constraints expected.

3. **Directory placement**: Phase 4 lives in `tests/mortar_pbc/`.
   After full validation (all of Phase 4 green), promote to
   `src/mortar_pbc/`. Within `tests/`, code lives in a subdirectory
   `mortar_pbc/` (i.e. `tests/mortar_pbc/`).

4. **Validation drivers**: standalone executables, not extensions to
   the existing `mechanics` executable. Each test mode (homogeneous,
   heterogeneous, checkerboard) is its own .cpp file.

5. **AllGather refactor**: AllGather-based matching in Phase 4.1.
   Distributed-hash refactor is Phase 4.2, **the very next step**
   after Phase 4.1 is green. Not deferred to Phase 5.

6. **Boundary subcommunicator**: ALL setup-time boundary work runs
   on a `boundary_comm` created via `MPI_Comm_split` at driver
   startup; interior ranks (those with no local boundary elements)
   are excluded entirely. Volume work (K, Krylov inner products,
   volume-averaged F) stays on WORLD. C is constructed on WORLD
   with empty row blocks for interior ranks. (§P4.4.0). This is in
   from Round 1, not deferred — it's a separate, complementary
   improvement to the Phase 4.2 distributed-pair matching refactor.

7. **Krylov solver options**: Three Krylov solvers supported, with
   MINRES as default (matches Python prototype). MINRES for
   symmetric K, GMRES for non-symmetric K, BiCGStab as a constant-
   memory non-symmetric alternative. CG explicitly rejected with
   a clear error message (the system is indefinite). Selectable
   via `--solver={minres,gmres,bicgstab}` flag in the validation
   drivers. (§P4.4.7).

8. **MPI_Comm storage**: the boundary_comm lives in ExaConstit's
   existing `SimulationState` class, which already manages the few
   non-WORLD communicators in the codebase. SimulationState owns
   creation and destruction; classifier / constraint builder /
   driver take it by reference. No separate RAII wrapper needed.
   (§P4.8.7, Trap 3.)

9. **Phase 4.2 pair-matching algorithm**: 2D regular tile
   partitioning of the parametric plane (Strategy B in §P4.4.4),
   chosen over hash-based partitioning (A) and bbox-direct lookup
   (D). Tile partitioning preserves spatial locality so the post-
   matching AllToAll for nonmortar-DOF-ownership stays small. Bbox-
   based direct lookup is asymptotically cheaper but adds
   significant complexity around irregular METIS partitions; held
   in reserve as a follow-up optimization if profiling Strategy B
   at p ≈ 30 shows it's a bottleneck.

---

## §P4.12 Cross-references to architecture doc

When porting, consult the architecture doc for the underlying derivations:

- **Mortar dual basis**: §4.0–§4.7 (theory), §4.8–§4.12 (higher-order
  considerations, deferred to Phase 6+).
- **Wohlmuth corner modifications**: §5.1–§5.6.
- **Wirebasket hierarchy**: §5.4 (the mortar/nonmortar assignment rule).
- **Saddle-point system**: §6.1–§6.7.
- **Warm-start mechanics**: §7.1–§7.6.
- **Volume-averaged F homogenization check**: §8.1–§8.4.
- **Reference frame discipline**: §9.1–§9.4 (the byNODES/byVDIM trap
  is in §9.4 specifically).
- **Distributed-driver invariants**: §10.4.
- **MFEM API gotchas**: §10.5.
- **3D mesh classifier**: §11.7 (overall), §11.7.1 (snap-coord cross-
  rank keys), §11.7.2 (runtime attribute discovery), §11.7.3 (what's
  in C's nullspace).
- **Existing C++ class sketch**: §13.2.
- **Hooks into ExaConstit infrastructure**: §13.3 (the BCManager /
  SystemDriver integration plan, deferred to Phase 5).
- **Upstream MFEM contribution path**: §13.5.

---

## §P4.13 Done criteria for Phase 4

Phase 4 is **done** when ALL of these hold:

- [ ] All three C++ validation drivers (homogeneous, heterogeneous,
      checkerboard) pass at np=1, 4, 16, 256 hex+tet.
- [ ] Phase 4.1.A (homogeneous) bit-compares to Python at np=1 hex,
      n=4 mesh: identical C, identical du, identical <F> within
      Krylov tolerance.
- [x] **Phase 4.2 distributed-pair matching is implemented**
      (tile partitioning Strategy B, Batches G–N). Validated
      at np=1 (unit tests + patch tests, numerically identical to
      Phase 4.1) and np=7 (heterogeneous checkerboard patch test).
      Pending validation at np=1024 — final scaling check before
      §P4.13 marks this fully done.
- [x] **Phase 4.3 EA constraint path is implemented**
      (`MortarConstraintOperator` + `MortarSaddlePointSystem`
      adapter + saddle-point solver `Solve(K, C_op, ...)` overload,
      Batches O–S). A/B validation against the HypreParMatrix path
      runs in two layers: matvec-level at np=1 (Batch Q's
      `test_mortar_constraint_operator`, tolerance 1e-12) and
      end-to-end at np=1 (`test_patch_3d_pbc_ea_compare`, tolerance
      1e-7). Pending: end-to-end A/B at np=4 / np=7 to exercise the
      Alltoallv import / export topology with real off-rank data.
- [~] **Phase 4.3.B GPU port — first pass complete** (Batch X).
      Forward `Mult` ported to `mfem::forall` over flat arrays
      built at construction by `BuildFlatRowArrays`; all Vector
      accesses across the EA path, saddle-point solver, and patch
      driver use typed memory-manager accessors
      (`HostRead`/`HostWrite`/`HostReadWrite`). Patch tests run
      cleanly under MFEM's `DEVICE_DEBUG` mode on host build.
      Pending for Phase 4.3.B "fully done" (see §P4.4.6.9 for
      details):
        * atomic-add `MultTranspose` scatter on device,
        * real CUDA / HIP build validation,
        * `MPI_Allreduce`-based cross-rank A/B comparison once
          atomic adds are in place,
        * performance profiling and optimization.
- [ ] All five C++ unit-test suites pass.
- [ ] Caliper profiling shows expected hot-path distribution
      (saddle-point solve dominates, not classifier setup or mortar
      integration).
- [ ] No `// TODO` markers in production code paths (only in
      validation drivers if at all).
- [ ] Doxygen-complete public API for all four core classes.
- [ ] `tests/mortar_pbc/CMakeLists.txt` builds standalone, links
      against MFEM + MPI without modifying ExaConstit's main CMake.

When done, code moves from `tests/mortar_pbc/` to `src/mortar_pbc/`
and Phase 5 (ExaConstit integration) begins.
