#------------------------------------------------------------------------------
# test/mortar_pbc/CMakeLists.txt
#------------------------------------------------------------------------------
# Mortar-method periodic boundary condition (PBC) test infrastructure.
#
# Phase 5.1 promotion: the production mortar PBC code now lives in
# src/mortar_pbc/ and is part of `exaconstit_static`. This directory
# retains ONLY:
#   - Test helpers (elastic_3d_helpers — synthetic K assembly,
#     visualization_3d — VTK debug dumps, patch_test_driver_3d —
#     patch test orchestration).
#   - The unit-test executables themselves (test_*.cpp).
#
# The tiny `mortar_pbc_lib` static library now bundles only the test
# helpers above. Tests link against `mortar_pbc_lib` AND
# `exaconstit_static` (through EXACONSTIT_TEST_DEPENDS); they get the
# production mortar machinery via the latter, the test helpers via
# the former.
#
# Test source files retain their bare-name `#include "..."` style for
# production headers — the include path below adds
# `${CMAKE_SOURCE_DIR}/src/mortar_pbc` so the existing
# `#include "boundary_classifier_3d.hpp"`, etc. lines continue to
# resolve without change. A future cleanup pass may migrate these to
# the `mortar_pbc/foo.hpp` style consistent with other src/
# subdirectory headers; not blocking Phase 5.1.
#
# This CMakeLists is included from the parent test/CMakeLists.txt via:
#
#     add_subdirectory(mortar_pbc)
#
# It picks up MFEM, MPI, RAJA, etc. from the project-level
# EXACONSTIT_DEPENDS list (populated by exaconstit_fill_depends_list()
# in the top-level CMakeLists.txt). No find_package() calls here.
#------------------------------------------------------------------------------

set(EXACONSTIT_TEST_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_TEST_DEPENDS
                             DEPENDS_ON  mfem ecmech RAJA mpi snls)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_TEST_DEPENDS blt::cuda_runtime blt::cuda CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_TEST_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_TEST_DEPENDS cuda cuda_runtime CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_TEST_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_TEST_DEPENDS blt::hip blt::hip_runtime hipblas rocsparse rocrand)
endif()

if (SNLS_USE_RAJA_PORT_SUITE)
    list(APPEND EXACONSTIT_TEST_DEPENDS chai umpire camp fmt::fmt)
endif()

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_TEST_DEPENDS caliper)
endif()

# Axom (LLNL) provides the BVH spatial index (`axom::spin::BVH<2>`)
# and 2D polygon clipping (`axom::primal::clip`) used by the Phase 4.4
# non-conforming face mortar machinery. ExaConstit will also use
# Axom's Sidre component for restart capability, so this dependency
# serves both workstreams.
#
# When ENABLE_AXOM is OFF, `mortar_pbc_lib` and all conforming-mesh
# tests still build; only `test_axom_smoke` and (future) the
# non-conforming patch test are skipped. The conforming code path
# does not link Axom.
#
# We list the umbrella `axom` target plus the component targets we
# use directly (axom::core for IndexType/Array/ArrayView, axom::slam
# for slam-mediated containers used internally by spin::BVH, and
# axom::slic for the SLIC logging that Axom calls into when
# findBoundingBoxes hits an error). spin and primal are header-only
# in the components we use, so they don't need explicit listing.
if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_TEST_DEPENDS axom axom::core axom::slam axom::slic)
endif()

list(APPEND EXACONSTIT_TEST_DEPENDS exaconstit_static)

message("-- EXACONSTIT_TEST_DEPENDS: ${EXACONSTIT_TEST_DEPENDS}")

set(MORTAR_PBC_HEADERS
    elastic_3d_helpers.hpp
    visualization_3d.hpp
    patch_test_driver_3d.hpp
    )

set(MORTAR_PBC_SOURCES
    elastic_3d_helpers.cpp
    visualization_3d.cpp
    patch_test_driver_3d.cpp
    )
# Phase 5.1 — production mortar code (incl. Axom-conditional non-
# conforming files) moved to src/mortar_pbc/ and now lives inside
# `exaconstit_static`. The MORTAR_PBC_HAS_AXOM compile definition
# is set on `exaconstit_static` in src/CMakeLists.txt under the
# corresponding `if(ENABLE_AXOM)` guard; nothing to do here.

# Static library holding the test helpers. Tests link against this
# AND `exaconstit_static` (via EXACONSTIT_TEST_DEPENDS); production
# mortar code resolves through the latter.
blt_add_library(NAME       mortar_pbc_lib
                HEADERS    ${MORTAR_PBC_HEADERS}
                SOURCES    ${MORTAR_PBC_SOURCES}
                INCLUDES   ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/src/mortar_pbc
                DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS})

#------------------------------------------------------------------------------
# Unit tests
#
# Each unit test is a small executable verifying one component of the
# mortar machinery. Single-rank tests run directly; multi-rank tests
# (BoundaryClassifier3D and downstream integration tests) launch
# under MPI via blt_add_test's NUM_MPI_TASKS parameter.
#------------------------------------------------------------------------------
function(mortar_pbc_add_unit_test test_name)
    cmake_parse_arguments(MPBCAUT "" "NUM_MPI_TASKS" "" ${ARGN})
    if(NOT MPBCAUT_NUM_MPI_TASKS)
        set(MPBCAUT_NUM_MPI_TASKS 1)
    endif()

    blt_add_executable(NAME       ${test_name}
                       SOURCES    ${test_name}.cpp
                       INCLUDES   ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${CMAKE_SOURCE_DIR}/src
                                  ${CMAKE_SOURCE_DIR}/src/mortar_pbc
                       DEPENDS_ON mortar_pbc_lib ${EXACONSTIT_TEST_DEPENDS}
                       OUTPUT_DIR ${TEST_OUTPUT_DIR})

    blt_add_test(NAME           ${test_name}
                 COMMAND        ${test_name}
                 NUM_MPI_TASKS  ${MPBCAUT_NUM_MPI_TASKS})
endfunction()

# Single-rank tests (pure helpers, no MPI dependency).
mortar_pbc_add_unit_test(test_mortar_assembler_2d)
mortar_pbc_add_unit_test(test_face_mortar_assembler_3d)
# Phase 4.4 / Batch 4.4-D-1 — closed-form inverse-isoparametric maps
# for axis-aligned face elements + 6-point Dunavant rule. No Axom
# dependency; runs regardless of ENABLE_AXOM.
mortar_pbc_add_unit_test(test_face_mortar_inverse_map_3d)
mortar_pbc_add_unit_test(test_boundary_helpers_3d)
mortar_pbc_add_unit_test(test_tile_partition_3d)

# MPI-aware tests. The boundary classifier is collective on the parent
# ParMesh's communicator; np=1 is enough to validate basic correctness
# (the mesh-construction path is the same; the classifier still goes
# through MPI_Allreduce / MPI_Allgatherv with one rank). Add np=4
# variant later if needed for cross-rank validation.
mortar_pbc_add_unit_test(test_boundary_classifier_3d         NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_constraint_builder_3d          NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_elastic_3d_helpers             NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_saddle_point_solver            NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_patch_3d_pbc                   NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_patch_3d_pbc_heterogeneous     NUM_MPI_TASKS 1)
mortar_pbc_add_unit_test(test_patch_3d_pbc_checkerboard      NUM_MPI_TASKS 1)
# Phase 4.3 / Batch S — EA A/B compare. Runs all three patch-test
# patterns once via the HypreParMatrix path and once via the EA path,
# asserting ||du_ea - du_hp||_inf < ab_compare_tol on each. Registered
# at np=1 by convention; cross-rank Alltoallv exercise comes from
# re-running this test with NUM_MPI_TASKS > 1 (np=4 / np=7) — the
# np=1 run validates dimensional and algorithmic correctness, the
# np>1 runs catch cross-rank topology bugs.
mortar_pbc_add_unit_test(test_patch_3d_pbc_ea_compare        NUM_MPI_TASKS 1)
# Phase 4.3 / Batch O — element-assembly constraint operator skeleton.
# Tests construction + dimension match with HypreParMatrix path. Batch P
# will extend with Mult/MultTranspose correctness; Batch Q adds full
# A/B harness (HypreParMatrix vs EA matvec equivalence).
mortar_pbc_add_unit_test(test_mortar_constraint_operator     NUM_MPI_TASKS 1)
# Phase 4.3 / Batch R — saddle-point system adapter (composes
# user-provided K residual/Jacobian closures with the EA constraint
# operator into a single mfem::Operator usable with NewtonSolver +
# block-Krylov methods).
mortar_pbc_add_unit_test(test_mortar_saddle_point_system     NUM_MPI_TASKS 1)
# Phase 5.3.B — corner essential-TDOF builder for MortarPbcManager.
# Exercises ComputeCornerEssTDofs (the free function the manager's
# BuildCornerEssTDofs delegates to) on 2x2x2 and 4x4x4 hex meshes.
# Registered at np=1; running by hand with NUM_MPI_TASKS > 1
# exercises the rank-split path.
mortar_pbc_add_unit_test(test_mortar_pbc_manager             NUM_MPI_TASKS 1)
# Phase 5.4 — smoke test for ParNonlinearForm::SetEssentialTrueDofs
# with a 24-element TDOF list (the path
# NonlinearMechOperator::UpdateEssTDofsCornerSubset uses for mortar
# PBC corner pinning). Self-contained; doesn't construct
# NonlinearMechOperator (that requires a full SimulationState — end-
# to-end coverage lands with the Phase 5.5/5.6 patch tests).
mortar_pbc_add_unit_test(test_mech_operator_corner_subset    NUM_MPI_TASKS 1)
# Phase 4.4 / Batch 4.4-A — Axom smoke test. Verifies that the Axom
# headers we depend on for the non-conforming face mortar
# (axom::primal::Point/BoundingBox/Polygon/clip, axom::spin::BVH<2>)
# compile and link. If this test fails to BUILD, fix the host-config
# / find_package(axom) plumbing before proceeding to Batch 4.4-B.
# Only registered when ENABLE_AXOM is ON; the conforming mortar code
# path doesn't need Axom and continues to build either way.
if(ENABLE_AXOM)
    mortar_pbc_add_unit_test(test_axom_smoke)
    # Phase 4.4 / Batch 4.4-B — broad-phase candidate-pair enumeration
    # via axom::spin::BVH<2>. Validates MatchClippedQuadFacePairs and
    # MatchClippedTriFacePairs on synthetic conforming and
    # non-conforming inputs. Single-rank — pure setup-time logic, no
    # MPI involvement.
    mortar_pbc_add_unit_test(test_face_mortar_match_3d)
    # Phase 4.4 / Batch 4.4-D-2 — non-conforming Q1 quad-quad face
    # mortar assembler. Routes a 4×4 conforming setup through both
    # AssemblePairConforming and AssembleQuadFacePairClipped, asserts
    # the resulting D and A_m blocks agree to FP roundoff. This is the
    # central correctness gate for the Phase 4.4 assembler — if it
    # passes, the assembler is correct on conforming inputs and
    # high-confidence-correct on non-conforming inputs (the only thing
    # that changes is the clipping geometry).
    mortar_pbc_add_unit_test(test_face_mortar_assembler_clipped_3d)
    # Phase 4.4 / Batch 4.4-E Part 2 — production-shape patch test on
    # a non-conforming periodic interface. Builds a conforming
    # MakeCartesian3D mesh, applies an in-plane sine perturbation to
    # the y=L face only, then runs the standard homogeneous patch
    # test. The y face pair becomes non-matching (centroid distances
    # of order amplitude=0.05, far above the 1e-9 match tolerance),
    # triggering the clipped-path fallback in BuildLocalPairBlocks.
    # End-to-end gate for Phase 4.4 — exercises BVH + clip +
    # AssembleClipped + dispatch in a real FE solve.
    mortar_pbc_add_unit_test(test_patch_3d_pbc_nonconforming
                             NUM_MPI_TASKS 1)
    # Phase 4.5 — A/B compare on the non-conforming patch test.
    #
    # Re-uses the same executable as test_patch_3d_pbc_nonconforming
    # (no new blt_add_executable needed) but invokes it with
    # --ab-compare so that BOTH the HypreParMatrix and EA constraint
    # storage paths run on the same problem and the per-rank du
    # vectors are asserted to agree within ab_compare_tol = 1e-7.
    #
    # This validates that the Phase 4.4 clipped-path FaceMortarPairBlock
    # output is consumed identically by the EA matvec (Phase 4.3 /
    # Batch X) and the HypreParMatrix path. The EA path is what
    # ExaConstit production will use; without this assertion, the EA
    # path could disagree silently at production scale where no
    # reference is available.
    blt_add_test(NAME           test_patch_3d_pbc_nonconforming_ab_compare
                 COMMAND        test_patch_3d_pbc_nonconforming --ab-compare
                 NUM_MPI_TASKS  1)
 
    # Phase 4.5 — heterogeneous strip-split on a non-conforming
    # periodic interface. Strip-split material assignment (5x stiffness
    # contrast across x = L/2) combined with the y=L face perturbation
    # of test_patch_3d_pbc_nonconforming. The y face pair is both
    # NON-CONFORMING and traverses a heterogeneous response field
    # induced by the strip-split coupling on the across-material
    # x face pair.
    #
    # This test exposes a bug class that the homogeneous non-conforming
    # test cannot: errors in A_m's column ordering or sign that don't
    # show up on u_lin = (F-I)X (linear field) but do show up on
    # the heterogeneous fluctuation u_tilde. Architecture doc §12
    # traps 18 + 19 — heterogeneous AND non-conforming together is
    # the strongest single-mesh check for the constraint pipeline.
    mortar_pbc_add_unit_test(test_patch_3d_pbc_nonconforming_heterogeneous
                             NUM_MPI_TASKS 1)
 
    # Phase 4.5 — 2x2x2 octant-checkerboard heterogeneity on a
    # non-conforming periodic interface. Maximum-stress combination
    # in the Phase 4.5 suite: every periodic element pair crosses a
    # material seam (checkerboard contribution) AND the y face pair
    # is non-conforming (sine perturbation contribution). Exercises
    # the full clipped-path constraint apparatus on a wirebasket-
    # equivalent heterogeneous configuration.
    mortar_pbc_add_unit_test(test_patch_3d_pbc_nonconforming_checkerboard
                             NUM_MPI_TASKS 1)

endif()
# Phase 4.1.A acceptance suite: the homogeneous, strip, and checkerboard
# patch tests are the three non-trivial end-to-end validations of the
# entire mortar-PBC pipeline. The homogeneous test confirms the
# zero-fluctuation analytical case; the strip and checkerboard tests
# exercise the constraint machinery on genuinely-heterogeneous RVEs
# where the periodic fluctuation must be captured exactly. Multi-rank
# correctness is validated by re-running these tests with NUM_MPI_TASKS
# > 1 in addition to the np=1 default.
