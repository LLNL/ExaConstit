# Mortar PBC prototype for ExaConstit

> **Looking for the full theory + practice + 3D-extension reference?** See
> [`docs/MORTAR_PBC_ARCHITECTURE.md`](docs/MORTAR_PBC_ARCHITECTURE.md). This
> README is the quickstart; the architecture doc is the comprehensive
> all-guiding reference (vocabulary, math, the trap list, the 3D Phase-3 plan,
> the C++ port pathway, references).

Python / pyMFEM prototype of dual-basis mortar periodic boundary
conditions for non-conforming RVE meshes, following Lopes, Ferreira &
Andrade Pires, *CMAME* **384** (2021) 113930.  Precursor to an eventual
MFEM C++ implementation that will land in ExaConstit.

Phase 1 scope: 2D rectangular RVEs, H1 vector-linear elements, MPI-aware
saddle-point Newton step solved via gather-to-root + `scipy.sparse.linalg.spsolve`.

---

## 1. Recommended environment

The Python-only unit tests need just NumPy + SciPy.  The driver
(`examples/patch_test_2d.py`) needs pyMFEM with parallel build
(MPI + HYPRE) plus mpi4py.  Targeted versions:

| Component | Version / commit                                                |
|-----------|-----------------------------------------------------------------|
| Python    | 3.10 – 3.12 (pyMFEM supports 3.8+; 3.10+ for the modern type-hint syntax used here) |
| MFEM      | 4.9 (the version pyMFEM commit `7e99b925` targets)              |
| pyMFEM    | commit `7e99b925cfcbec002c9e21230b3c561cb19436a6` (develop, MFEM 4.9 build fixes; PR #300) |
| MPI       | OpenMPI ≥ 4.0 or MPICH ≥ 3.3 (must match what mpi4py was built against) |
| SWIG      | ≥ 4.2.1 (pyMFEM build requirement)                              |
| NumPy     | ≥ 1.22                                                          |
| SciPy     | ≥ 1.10                                                          |
| mpi4py    | ≥ 3.1                                                           |

A clean conda env is the fastest path; if you prefer venv, do that.

```bash
# --- Conda variant ---
conda create -n mortar-pbc python=3.11 numpy scipy mpi4py openmpi cmake swig -c conda-forge
conda activate mortar-pbc
# --- venv variant (system MPI + SWIG must already be present) ---
python -m venv ~/.venvs/mortar-pbc
source ~/.venvs/mortar-pbc/bin/activate
pip install numpy scipy mpi4py
```

Sanity-check `mpi4py` and the matching MPI launcher are in agreement
before you do anything else:

```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
mpirun --version
```

---

## 2. Install pyMFEM (parallel build, pinned to the MFEM-4.9 commit)

```bash
# Pick a workspace
cd ~/src   # or wherever you keep checkouts

# Clone PyMFEM
git clone https://github.com/mfem/PyMFEM.git
cd PyMFEM
git checkout 7e99b925cfcbec002c9e21230b3c561cb19436a6

# Build with MPI.  This downloads + builds MFEM, METIS, and HYPRE
# locally; takes 10-20 min on a recent laptop.
pip install ./ -C"with-parallel=Yes" --verbose
```

Notes on the pyMFEM build:

- The `--verbose` flag is recommended on a first build so you can see
  where things go if something fails.
- If you want to point at an existing MFEM/HYPRE/METIS installation
  rather than letting pyMFEM download and build them, see
  [PyMFEM/INSTALL.md](https://github.com/mfem/PyMFEM/blob/mortar/INSTALL.md)
  for the `--mfem-prefix` / `--mfem-source` / `--hypre-prefix` flags.
  This is the path you'll likely want on a cluster where MFEM is
  already module-loaded.
- On macOS with Apple Silicon you may need to set
  `CFLAGS="-Wno-incompatible-function-pointer-types"` in the env before
  the pip install if SWIG-generated code triggers the strict default.

Verify pyMFEM came out parallel:

```bash
python -c "import mfem.par; print('pyMFEM parallel OK,', mfem.par.__file__)"
python -c "from mfem.common.parcsr_extra import ToScipyCSR; print('ToScipyCSR OK')"
```

If the second command works, the gather-to-root path in
`hypre_to_scipy_csr` will work.

---

## 3. Install the prototype

The prototype is plain Python — no compilation step.  Two install paths:

### 3a. Editable install (recommended for development)

From the prototype's root directory:

```bash
cd /path/to/mortar_pbc_proto
pip install -e .
```

(There's no `setup.py` shipped — see step 3b for the no-install path
that's actually being used right now.  Drop in a minimal `pyproject.toml`
later if you want.)

### 3b. PYTHONPATH (no install at all)

Easiest path right now.  From the prototype's root:

```bash
cd /path/to/mortar_pbc_proto
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then `import mortar_pbc` works.  The unit tests and the driver script
already do `sys.path.insert(...)` so they don't actually need this; only
ad-hoc `python -c "import mortar_pbc"` benefits.

---

## 4. Test the prototype

### 4a. Unit tests (no pyMFEM needed)

Five tests covering: dual-basis bi-orthogonality, partition of unity,
conforming-pair lumping, non-conforming-pair linear-field reproduction,
and the `ConstraintAssembler` ABC + `stack_constraints` machinery.
Pure NumPy — runs in any Python env.

```bash
cd /path/to/mortar_pbc_proto
python tests/test_mortar_2d_unit.py
```

Expected output:

```
Running mortar 2D unit tests
------------------------------------------------------------
Test 1: dual basis bi-orthogonality
  PASS  dual basis bi-orthogonality (max err 1.39e-17)
Test 2: shape function partition of unity
  PASS  N partition of unity (max err 0.00e+00)
Test 3: conforming pair recovers lumped mass
  ...
  PASS  conforming pair recovers lumped mass
Test 4: non-conforming pair row-sum consistency
  ...
  PASS  non-conforming pair reproduces constant + linear fields
Test 5: ConstraintAssembler ABC + stack_constraints
  ...
  PASS  ConstraintAssembler ABC + stack_constraints
------------------------------------------------------------
All unit tests passed.
```

If anything in that block fails, **stop** and don't move on to step 4b
— the unit tests cover the math; if they don't pass on your box,
nothing downstream will.

### 4b. Patch test, np = 1 (homogeneous RVE recovers `u_tilde = 0`)

```bash
cd /path/to/mortar_pbc_proto
mpirun -n 1 python examples/patch_test_2d.py
```

Or equivalently, since np=1 means no actual MPI launch is needed:

```bash
python examples/patch_test_2d.py
```

Look for these lines at the bottom:

```
  ||C u_tilde||_2     = <something < 1e-8>
  ||u_tilde||_inf     = <something < 1e-8>
  ||du||_inf          = <something < 1e-8>
  PASS
```

The patch test imposes the macroscopic deformation gradient
`F = [[1.5, 0.5], [0.5, 1.0]]` on a homogeneous square RVE.  Theory
says the fluctuation `u_tilde` should be zero everywhere — this is
exactly the discrete patch-test criterion (Lopes §5.1.1).  If it
**fails** on np = 1, the issue is one of:

- The boundary attribute layout (1=bottom, 2=left, 3=top, 4=right) was
  set wrong by the mesh builder — uncomment the diagnostic in
  `BoundaryClassifier2D.summary()` to inspect.
- The corner-Dirichlet elimination didn't reach all four corners — check
  `corner_dirichlet_gtdofs` output.
- The mortar coupling has a bug that the unit tests didn't catch —
  unlikely given the unit tests pass, but possible.

### 4c. Patch test, np = 2 (exercises the gather-to-root path)

```bash
mpirun -n 2 python examples/patch_test_2d.py
```

Or `mpirun -n 4`, `mpirun -n 8` for a stronger MPI test.  Same PASS
criteria.  If np=1 passes but np>1 fails, suspects in order:

1. **`HypreParMatrix.GetRowPartArray()` returning unexpected shape.**
   Print `np.asarray(K_hyp.GetRowPartArray())` from inside
   `hypre_to_scipy_csr` to see what your HYPRE build produces.  My code
   handles both `[first, last_excl]` (assumed-partition) and the full
   `nranks+1` form.
2. **`ToScipyCSR` not finding `MergeDiagAndOffd`.**  Check
   `python -c "from mfem.par import HypreParMatrix; m = HypreParMatrix; print(hasattr(m, 'MergeDiagAndOffd'))"`.
3. **MPI launcher / mpi4py mismatch.**  If `mpirun -n 2` runs two
   independent serial copies (each printing rank=0), the launcher and
   mpi4py are linked against different MPI implementations.  Easy
   diagnostic: run `mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"` — both ranks should
   print, with sizes = 2.
4. **`apply_linear_part` returning a different size on each rank than
   `fes.GetTrueVSize()`.**  Add `assert u_lin_local.size == fes.GetTrueVSize()`
   right after the call.

---

## 5. What's there

```
mortar_pbc_proto/
├── README.md                           ← this file
├── mortar_pbc/
│   ├── __init__.py                     ← package surface, lazy MFEM imports
│   ├── types_2d.py                     ← EdgeNodes2D, CornerInfo dataclasses
│   ├── mortar_2d.py                    ← dual basis + A^m, D^nm assembly
│   ├── constraint_builder.py           ← global C from mortar blocks
│   ├── constraint_assembler.py         ← ABC + stack helper (UT extension hook)
│   ├── saddle_point.py                 ← [[K, C^T], [C, 0]] direct solve
│   └── boundary_2d.py                  ← MFEM-dependent boundary classifier
├── examples/
│   └── patch_test_2d.py                ← driver + gather/scatter helpers
└── tests/
    └── test_mortar_2d_unit.py          ← 5 unit tests (pyMFEM-free)
```

Every module has a What/Why/References docstring tying back to the
specific equations and figures of Lopes et al. (2021).  Inline comments
flag the parts that are non-obvious to a reader familiar with
ExaConstit but new to mortar methods (corner-mod intentionally breaking
bi-orthogonality, dual-basis asymmetry, etc.).

The `K`-block of the saddle-point system is consumed *as an interface*
in the design — the prototype materializes it to scipy CSR only because
`spsolve` needs that.  ExaConstit's actual K (PA / EA / FA, whatever
the run is configured for) plugs in at this seam in the C++ port; see
the docstring of `mortar_pbc.saddle_point.SaddlePointSolver` for the
extension point.

---

## 6. Where the next round of work is going

In rough priority order:

1. Phase 2: heterogeneous RVE + neo-Hookean + Newton iteration coupled
   to `mfem.ParNonlinearForm.GetGradient()` (the C++ ExaConstit-shaped
   way of doing it).  This is the first real test that the K-as-
   interface design holds up.
2. Serial 3D: wirebaskets (4 edges per direction collapsing to one
   mortar edge with 3 non-mortar) + quadratic non-mortar treatment per
   §C of Lopes et al.
3. MPI 3D.
4. Investigate Tribol's API for D^nm / A^m exposure as standalone
   artifacts (deferred until 1–3 are solid).
5. C++ port into ExaConstit.

Uniform traction (UT) is intentionally deferred until ExaConstit grows
a traction BC.  The `ConstraintAssembler` ABC is the extension point —
adding UT later means writing one new `UniformTractionConstraintAssembler`
subclass and stacking it via `stack_constraints`.  No other code
changes.
