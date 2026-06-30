# test/mortar_pbc

Mortar-method periodic boundary condition (PBC) machinery — Phase 4 of
the C++ port from the Python prototype to ExaConstit's main codebase.

This is a **drop-in subdirectory** for `test/`. To enable it, add a
single line to the parent `test/CMakeLists.txt`:

```cmake
add_subdirectory(mortar_pbc)
```

After that the standard ExaConstit build picks it up:

```bash
cd <ExaConstit-root>/build
cmake .. -DENABLE_TESTS=ON ...   # (your existing config flags)
cmake --build . -j 8
ctest -V -R mortar
```

## Status

Phase 4.1.A (foundational classes) is in progress. Not yet ported:
boundary classifier, constraint builder, elastic helpers, saddle-point
solver, visualization wrapper, validation drivers. See
`docs/PHASE4_CPP_PORT_PLAN.md` for the full plan.

| Component                         | Status   | Files                                  |
|-----------------------------------|----------|----------------------------------------|
| Data carriers (3D types)          | ✅ Done  | `types_3d.hpp`                         |
| 1D / edge mortar (line-2)         | ✅ Done  | `mortar_assembler_2d.{hpp,cpp}`        |
| 2D / face mortar (quad-4, tri-3)  | ✅ Done  | `face_mortar_assembler_3d.{hpp,cpp}`   |
| Boundary helpers (pure logic)     | ✅ Done  | `boundary_helpers_3d.{hpp,cpp}`        |
| Boundary classifier (MFEM/MPI)    | ✅ Done (4.1); 🚧 4.2 in progress | `boundary_classifier_3d.{hpp,cpp}`     |
| Constraint builder                | ✅ Done  | `constraint_builder_3d.{hpp,cpp}`      |
| Linear-elastic helpers            | ✅ Done  | `elastic_3d_helpers.{hpp,cpp}`         |
| Saddle-point solver               | ✅ Done  | `saddle_point_solver.{hpp,cpp}`        |
| Visualization (ParaView)          | ✅ Done  | `visualization_3d.{hpp,cpp}`           |
| Shared patch-test driver          | ✅ Done  | `patch_test_driver_3d.{hpp,cpp}`       |
| Tile partition (Phase 4.2)        | ✅ Done (Batch G) | `tile_partition_3d.{hpp,cpp}` |
| Patch test (homogeneous)          | ✅ Done  | `test_patch_3d_pbc.cpp`                |
| Patch test (strip-split)          | ✅ Done  | `test_patch_3d_pbc_heterogeneous.cpp`  |
| Patch test (checkerboard)         | ✅ Done  | `test_patch_3d_pbc_checkerboard.cpp`   |

**Phase 4.1 is complete.** All components of the mortar-PBC pipeline are
ported from the Python prototype and validated end-to-end via the three
patch test variants:

* **Homogeneous** — single material; analytical solution `u = u_lin`
  exactly. Validates the orchestration; permissive on `||du||_∞`.
* **Strip-split** — two materials with 5x stiffness contrast across the
  x = L/2 plane. Genuinely non-trivial fluctuation `u_tilde`; tests
  both within-material (y, z) and across-material (x) periodicity.
* **Checkerboard** — 2x2x2 octant-XOR alternating attributes. EVERY
  matched pair of periodic boundary elements crosses a material
  interface. Maximum stress test on the constraint machinery for a
  given mesh size and contrast.

**Phase 4.2 in progress** — replace the boundary-records `MPI_Allgatherv`
in `BoundaryClassifier3D` with a tile-partitioned distributed shuffle
on a boundary-only subcomm, unlocking scalability beyond ~1000 ranks.
This batch (Batch G) lays the groundwork:

* `tile_partition_3d.{hpp,cpp}` — deterministic tile-to-rank map
  (Strategy B per §P4.4.4 of the plan). Pure arithmetic; unit-tested
  in isolation via `test_tile_partition_3d.cpp` (6 sub-tests covering
  axis-rank allocation, tile-grid factorisation, owner dispatch,
  partition coverage, round-trip consistency, and determinism).
* `BoundaryClassifier3D` now creates an `m_boundary_comm` via
  `MPI_Comm_split` (color = boundary-element-count > 0). Interior
  ranks get `MPI_COMM_NULL`. The classifier exposes `BoundaryComm()`,
  `IsBoundaryRank()`, `BdyRank()`, `NBdyRanks()` accessors.
  **No behaviour change yet** — the existing AllGatherv path still
  runs on `m_comm` (WORLD). Batch H switches the gather to the new
  subcomm + tile-shuffle pattern.

## Layout

Headers and sources are co-located, matching ExaConstit's `src/`
convention. No `include/` vs `src/` split:

```
test/mortar_pbc/
├── CMakeLists.txt
├── README.md
├── types_3d.hpp                        # Data carriers (CornerInfo3D, EdgeInfo3D, FaceInfo3D, ...)
├── mortar_assembler_2d.{hpp,cpp}       # Line-2 mortar (edge mortar in 3D)
├── face_mortar_assembler_3d.{hpp,cpp}  # Quad-4 + tri-3 face mortar
├── boundary_helpers_3d.{hpp,cpp}       # Pure topology helpers (no MFEM mesh, no MPI)
├── boundary_classifier_3d.{hpp,cpp}    # Boundary classifier (uses ParMesh + MPI)
├── constraint_builder_3d.{hpp,cpp}     # Global C matrix assembly + HypreParMatrix
├── elastic_3d_helpers.{hpp,cpp}        # Linear-elastic K assembly, u_lin projection, Dirichlet
├── saddle_point_solver.{hpp,cpp}       # Distributed Krylov saddle-point Newton-step solver
├── visualization_3d.{hpp,cpp}          # ParaView output wrapper for cross-validation
├── patch_test_driver_3d.{hpp,cpp}      # Shared driver for the three patch test variants
├── test_mortar_assembler_2d.cpp        # Unit test for edge mortar
├── test_face_mortar_assembler_3d.cpp   # Unit test for face mortar
├── test_boundary_helpers_3d.cpp        # Unit test for boundary helpers
├── test_boundary_classifier_3d.cpp     # Integration test for the classifier
├── test_constraint_builder_3d.cpp      # Integration test for the C matrix
├── test_elastic_3d_helpers.cpp         # Integration test for the elastic helpers
├── test_saddle_point_solver.cpp        # Integration test for the saddle-point solver
├── test_patch_3d_pbc.cpp               # End-to-end: homogeneous (analytic du = 0)
├── test_patch_3d_pbc_heterogeneous.cpp # End-to-end: strip-split (non-trivial u_tilde)
└── test_patch_3d_pbc_checkerboard.cpp  # End-to-end: octant-XOR (max constraint stress)
```

## Conventions

The code follows ExaConstit's existing conventions (see
`developers_guide.md`, *Name Formatting* section):

- **Functions / methods**: `PascalCase` (matches MFEM)
- **Variables / parameters / locals**: `snake_case`
- **Member variables (private)**: `m_snake_case` (e.g. `m_num_elements`,
  `m_oper_mech`). None currently — the assembler classes are
  stateless — but Phase 4.1's classifier and constraint builder will
  introduce member state.
- **Classes / structs**: `PascalCase`
- **Namespaces**: `snake_case` — code lives in `mortar_pbc::*`
- **Indentation**: 4 spaces (matches newer ExaConstit code; see
  `option_parser_v2.cpp`, `mechanics_operator.cpp`)
- **Header guards**: `#pragma once`
- **Includes**: `#include "mfem.hpp"` (quotes); siblings via bare
  filenames; `src/` headers via subdirectory path
  (e.g. `#include "utilities/mechanics_log.hpp"`)
- **Include order**: ExaConstit headers → TPLs → standard library
- **Errors**: `MFEM_VERIFY` for user-facing invariants;
  `MFEM_ASSERT` for internal consistency; `MFEM_ABORT` for
  unrecoverable errors
- **Caliper**: `CALI_CXX_MARK_SCOPE("scope_name")` from
  `utilities/mechanics_log.hpp`; compiled-out when `HAVE_CALIPER`
  is undefined
- **Doxygen**: JavaDoc-style `/** @brief ... */` blocks with
  `@param`, `@return`, `@details`, `@pre`, `@post`; LaTeX math via
  `\f$ ... \f$`

## Mapping to Python prototype

| Python module                                | C++ files                              |
|----------------------------------------------|----------------------------------------|
| `mortar_pbc/types_3d.py`                     | `types_3d.hpp`                         |
| `mortar_pbc/mortar_2d.py`                    | `mortar_assembler_2d.{hpp,cpp}`        |
| `mortar_pbc/mortar_3d.py` (basis fns)        | `face_mortar_assembler_3d.{hpp,cpp}`   |
| `mortar_pbc/face_mortar_3d.py`               | `face_mortar_assembler_3d.{hpp,cpp}`   |
| `mortar_pbc/boundary_3d.py` (helpers only)   | `boundary_helpers_3d.{hpp,cpp}`        |
| `mortar_pbc/boundary_3d.py` (classifier)     | `boundary_classifier_3d.{hpp,cpp}`     |
| `mortar_pbc/constraint_builder_3d.py`        | `constraint_builder_3d.{hpp,cpp}`      |
| `mortar_pbc/elastic_3d.py` (helpers subset)  | `elastic_3d_helpers.{hpp,cpp}`         |
| `mortar_pbc/saddle_point.py` (SaddlePointSolver class) | `saddle_point_solver.{hpp,cpp}` |
| `mortar_pbc/visualization.py` (single-step)  | `visualization_3d.{hpp,cpp}`           |
| `examples/patch_test_3d_pbc.py`              | `test_patch_3d_pbc.cpp` + `patch_test_driver_3d.{hpp,cpp}` |
| `examples/patch_test_3d_heterogeneous.py`    | `test_patch_3d_pbc_heterogeneous.cpp` (uses shared driver) |
| `examples/patch_test_3d_checkerboard.py`     | `test_patch_3d_pbc_checkerboard.cpp` (uses shared driver) |
| `tests/test_mortar_2d_unit.py`               | `test_mortar_assembler_2d.cpp`         |
| `tests/test_mortar_3d_unit.py` (subset)      | `test_face_mortar_assembler_3d.cpp`    |
| `tests/test_boundary_3d_helpers.py`          | `test_boundary_helpers_3d.cpp`         |
| `tests/test_constraint_builder_3d.py` (subset for classifier) | `test_boundary_classifier_3d.cpp` |
| `tests/test_constraint_builder_3d.py` (row count + structure) | `test_constraint_builder_3d.cpp`  |
| (new — exercises the helper API)             | `test_elastic_3d_helpers.cpp`          |
| (new — exercises the saddle-point API)       | `test_saddle_point_solver.cpp`         |

## Cross-validation against the Python prototype

The C++ `test_patch_3d_pbc` and the Python `examples/patch_test_3d_pbc.py`
implement the same 11-step pipeline with byte-meaningful equivalence:
- Same algorithmic sequence (mesh → classifier → constraint → K → Dirichlet → saddle-point → recovery → ⟨F⟩ check).
- Same PASS criteria thresholds (`||du||_∞ < 1e-7`, `||⟨F⟩ - F_macro||_∞ < 1e-9`, etc.).
- Same `--paraview` output format (cycle 0 = undeformed; cycle 1 = deformed
  warped by `u_total`; same field names `u_total / u_lin / u_tilde / material`).

Run both with the same `--F` choice and compare their outputs side-by-side
in ParaView, or numerically by examining the rank-0 stdout summary for the
`<F>` matrix and the residual-norm values.

The Python tests for higher-order element types (line-3, tri-6,
quad-8, quad-9, tet-10) are negative-result tests that verify the
lumped-positivity *failure* — we don't port them since the C++ code
doesn't ship those duals at all (out of scope for Phase 4).

## See also

- `docs/MORTAR_PBC_ARCHITECTURE.md` — top-level architecture doc
  with theoretical derivations.
- `docs/PHASE4_CPP_PORT_PLAN.md` — Phase 4 implementation plan with
  all design decisions captured.
