#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(EXACONSTIT_HEADERS
    ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
    system_driver.hpp
    boundary_conditions/BCData.hpp
    boundary_conditions/BCManager.hpp
    fem_operators/mechanics_integrators.hpp
    fem_operators/mechanics_operator_ext.hpp
    fem_operators/mechanics_operator.hpp
    mfem_expt/partial_qspace.hpp
    mfem_expt/partial_qfunc.hpp
    models/mechanics_model.hpp
    models/mechanics_ecmech.hpp
    models/mechanics_multi_model.hpp
    models/mechanics_umat.hpp
    mortar_pbc/types_3d.hpp
    mortar_pbc/mortar_assembler_2d.hpp
    mortar_pbc/face_mortar_assembler_3d.hpp
    mortar_pbc/face_mortar_inverse_map_3d.hpp
    mortar_pbc/boundary_helpers_3d.hpp
    mortar_pbc/boundary_classifier_3d.hpp
    mortar_pbc/constraint_builder_3d.hpp
    mortar_pbc/amgf_utils.hpp
    mortar_pbc/ginkgo_direct_subspace_solver.hpp
    mortar_pbc/surface_projector.hpp
    mortar_pbc/saddle_point_solver.hpp
    mortar_pbc/saddle_residual_scaler.hpp
    mortar_pbc/saddle_scaling_wrappers.hpp
    mortar_pbc/augmented_lagrangian_saddle.hpp
    mortar_pbc/saddle_newton_diagnostic_logger.hpp
    mortar_pbc/mortar_saddle_preconditioner.hpp
    mortar_pbc/mortar_saddle_preconditioner_amgf.hpp
    mortar_pbc/diagonal_scaler.hpp
    mortar_pbc/tile_partition_3d.hpp
    mortar_pbc/mortar_constraint_operator.hpp
    mortar_pbc/mortar_saddle_point_system.hpp
    mortar_pbc/mortar_pbc_manager.hpp
    options/option_parser_v2.hpp
    postprocessing/projection_class.hpp
    postprocessing/postprocessing_driver.hpp
    postprocessing/mechanics_lightup.hpp
    sim_state/simulation_state.hpp
    solvers/mechanics_solver.hpp
    solvers/trust_region_solver.hpp
    utilities/dynamic_function_loader.hpp
    utilities/mechanics_kernels.hpp
    utilities/mechanics_log.hpp
    utilities/assembly_ops.hpp
    utilities/rotations.hpp
    utilities/strain_measures.hpp
    utilities/unified_logger.hpp
    umats/userumat.h
    umats/unified_umat_loader.hpp
    TOML_Reader/toml.hpp
    )

set(EXACONSTIT_SOURCES
    system_driver.cpp
    boundary_conditions/BCData.cpp
    boundary_conditions/BCManager.cpp
    fem_operators/mechanics_integrators.cpp
    fem_operators/mechanics_operator_ext.cpp
    fem_operators/mechanics_operator.cpp
    mfem_expt/partial_qspace.cpp
    mfem_expt/partial_qfunc.cpp
    models/mechanics_model.cpp
    models/mechanics_ecmech.cpp
    models/mechanics_umat.cpp
    models/mechanics_multi_model.cpp
    mortar_pbc/mortar_assembler_2d.cpp
    mortar_pbc/face_mortar_assembler_3d.cpp
    mortar_pbc/face_mortar_inverse_map_3d.cpp
    mortar_pbc/boundary_helpers_3d.cpp
    mortar_pbc/boundary_classifier_3d.cpp
    mortar_pbc/constraint_builder_3d.cpp
    mortar_pbc/amgf_utils.cpp
    mortar_pbc/ginkgo_direct_subspace_solver.cpp
    mortar_pbc/surface_projector.cpp
    mortar_pbc/saddle_point_solver.cpp
    mortar_pbc/saddle_residual_scaler.cpp
    mortar_pbc/saddle_scaling_wrappers.cpp
    mortar_pbc/augmented_lagrangian_saddle.cpp
    mortar_pbc/saddle_newton_diagnostic_logger.cpp
    mortar_pbc/mortar_saddle_preconditioner.cpp
    mortar_pbc/mortar_saddle_preconditioner_amgf.cpp
    mortar_pbc/tile_partition_3d.cpp
    mortar_pbc/mortar_constraint_operator.cpp
    mortar_pbc/mortar_saddle_point_system.cpp
    mortar_pbc/mortar_pbc_manager.cpp
    options/option_parser_v2.cpp
    options/option_boundary_conditions.cpp
    options/option_enum.cpp
    options/option_material.cpp
    options/option_mesh.cpp
    options/option_post_processing.cpp
    options/option_solvers.cpp
    options/option_time.cpp
    postprocessing/postprocessing_driver.cpp
    postprocessing/projection_class.cpp
    postprocessing/mechanics_lightup.cpp
    sim_state/simulation_state.cpp
    solvers/mechanics_solver.cpp
    solvers/trust_region_solver.cpp
    utilities/mechanics_kernels.cpp
    utilities/unified_logger.cpp
    )

if (ENABLE_FORTRAN)
    list(APPEND EXACONSTIT_SOURCES ./umats/umat.f)
else()
    list(APPEND EXACONSTIT_SOURCES ./umats/umat.cxx)
endif()

# Phase 5.1 — non-conforming mortar PBC files (Axom-dependent).
# Promoted from test/mortar_pbc/ along with the conforming code; gated
# by ENABLE_AXOM the same way as the existing Axom dep above.
if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_HEADERS
        mortar_pbc/face_mortar_match_3d.hpp
        mortar_pbc/face_mortar_assembler_clipped_3d.hpp)
    list(APPEND EXACONSTIT_SOURCES
        mortar_pbc/face_mortar_match_3d.cpp
        mortar_pbc/face_mortar_assembler_clipped_3d.cpp)
endif()

set(DYNAMIC_LOADING_LIBS)

# Windows uses kernel32 automatically
if(UNIX)
    list(APPEND DYNAMIC_LOADING_LIBS ${CMAKE_DL_LIBS}) 
endif()
#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(EXACONSTIT_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_DEPENDS
                             DEPENDS_ON  mfem ecmech snls RAJA camp mpi)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DEPENDS blt::cuda_runtime blt::cuda)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DEPENDS cuda cuda_runtime)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_DEPENDS blt::hip blt::hip_runtime)
endif()

if (SNLS_USE_RAJA_PORT_SUITE)
    list(APPEND EXACONSTIT_DEPENDS chai umpire fmt::fmt)
endif()

if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_DEPENDS axom axom::core axom::slam axom::slic)
endif()

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEPENDS caliper)
endif()

if(UNIX AND NOT APPLE AND TARGET Threads::Threads)
    # Check if we determined explicit linking is needed
    if(EXACONSTIT_THREADS_EXPLICIT_LINK)
        list(APPEND EXACONSTIT_DEPENDS Threads::Threads)
    endif()
endif()

list(APPEND EXACONSTIT_DEPENDS ${DYNAMIC_LOADING_LIBS})

message("-- EXACONSTIT_DEPENDS: ${EXACONSTIT_DEPENDS}")

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------
set(EXACONSTIT_DEFINES HAVE_EXACONSTIT)
if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEFINES HAVE_CALIPER)
endif()

# MFEM can be built with Ginkgo while the Ginkgo installation itself omits the
# OpenMP backend. The AMGF subspace-solver adapter uses this definition to make
# `amgf_subspace_executor = "auto"` select the best executor actually present
# in the configured Ginkgo package.
if(DEFINED GINKGO_BUILD_OMP AND GINKGO_BUILD_OMP)
    list(APPEND EXACONSTIT_DEFINES EXACONSTIT_GINKGO_HAS_OMP)
endif()

# Phase 5.1 — make the Axom dependency visible at the C++ preprocessor
# level so non-Axom translation units (e.g. boundary_classifier_3d.cpp)
# can conditionally include and call the clipped-path machinery.
# Without this, the dispatch fallback would only work when
# ENABLE_AXOM=ON; with this, the same source compiles either way and
# gracefully aborts on non-conforming meshes when Axom is absent.
if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_DEFINES MORTAR_PBC_HAS_AXOM)
endif()
#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(EXACONSTIT_INCLUDES ${HEADER_INCLUDE_DIR}
                        ${CMAKE_SOURCE_DIR}/src/)
#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------

blt_add_library(NAME    exaconstit_static
                        OUTPUT_NAME exaconstit
                        DEFINES     ${EXACONSTIT_DEFINES}
                        INCLUDES    ${EXACONSTIT_INCLUDES}
                        DEPENDS_ON  ${EXACONSTIT_DEPENDS}
                        HEADERS     ${EXACONSTIT_HEADERS}
                        SOURCES     ${EXACONSTIT_SOURCES}
                        SHARED      FALSE)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

set(EXACONSTIT_DRIVER)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DRIVER blt::cuda_runtime blt::cuda CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DRIVER blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DRIVER cuda cuda_runtime CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DRIVER openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_DRIVER blt::hip blt::hip_runtime hipblas rocsparse rocrand)
endif()

blt_add_executable(NAME       mechanics
                   SOURCES    mechanics_driver.cpp
                   OUTPUT_DIR ${BINARY_DIR}
                   DEPENDS_ON exaconstit_static ${EXACONSTIT_DRIVER})

set(EXACONSTIT_INSTALLED_HEADERS ${EXACONSTIT_HEADERS})
install(FILES ${EXACONSTIT_INSTALLED_HEADERS} DESTINATION include)
install(TARGETS exaconstit_static DESTINATION lib)
install(TARGETS mechanics DESTINATION bin)
