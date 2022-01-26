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
    ./umat_tests/umat.f
    )

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(EXACONSTIT_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_DEPENDS
                             DEPENDS_ON  mfem ecmech raja mpi snls)

if(ENABLE_OPENMP)
    list(APPEND EXACONSTIT_DEPENDS openmp)
endif()

if(ENABLE_CUDA)
    list(APPEND EXACONSTIT_DEPENDS cuda)
endif()

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEPENDS caliper)
endif()

#include_directories(BEFORE ${PROJECT_BINARY_DIR})

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

if(ENABLE_OPENMP)
   list(APPEND EXACONSTIT_DRIVER openmp)
endif()

if(ENABLE_CUDA)
   list(APPEND EXACONSTIT_DRIVER cuda)
endif()

blt_add_executable(NAME       mechanics
                   SOURCES    mechanics_driver.cpp
                   OUTPUT_DIR ${BINARY_DIR}
                   DEPENDS_ON exaconstit_static ${EXACONSTIT_DRIVER})
