#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(EXACONSTIT_HEADERS
    ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
    BCData.hpp
    BCManager.hpp
    mechanics_model.hpp
    mechanics_integrators.hpp
    mechanics_ecmech.hpp
    mechanics_kernels.hpp
    mechanics_lightup.hpp
    mechanics_log.hpp
    mechanics_umat.hpp
    mechanics_operator_ext.hpp
    mechanics_operator.hpp
    mechanics_solver.hpp
    system_driver.hpp
    option_types.hpp
    option_parser.hpp
    userumat.h
    ./TOML_Reader/toml.hpp
    )

set(EXACONSTIT_SOURCES
    BCData.cpp
    BCManager.cpp
    mechanics_model.cpp
    mechanics_integrators.cpp
    mechanics_ecmech.cpp
    mechanics_kernels.cpp
    mechanics_umat.cpp
    mechanics_operator_ext.cpp
    mechanics_operator.cpp
    mechanics_solver.cpp
    system_driver.cpp
    option_parser.cpp
    ./umat_tests/userumat.cxx
    )

if (ENABLE_FORTRAN)
    list(APPEND EXACONSTIT_SOURCES ./umat_tests/umat.f)
else()
    list(APPEND EXACONSTIT_SOURCES ./umat_tests/umat.cxx)
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

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEPENDS caliper)
endif()

message("-- EXACONSTIT_DEPENDS: ${EXACONSTIT_DEPENDS}")

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------
set(EXACONSTIT_DEFINES HAVE_EXACONSTIT)
if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEFINES HAVE_CALIPER)
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
                        OBJECT      TRUE
                        SHARED      FALSE)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

#set(EXACONSTIT_INSTALLED_HEADERS
#    ${EXACONSTIT_HEADERS})

#install(FILES ${EXACONSTIT_INSTALLED_HEADERS} DESTINATION include)

#install(TARGETS exaconstit_static DESTINATION lib)

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
