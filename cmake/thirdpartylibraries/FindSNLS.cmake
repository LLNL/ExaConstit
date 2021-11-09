###############################################################################
#
# Setup SNLS
# This file defines:
#  SNLS_FOUND - If SNLS was found
#  SNLS_INCLUDE_DIRS - The SNLS include directories

# first Check for SNLS_DIR

if(NOT SNLS_DIR)
    MESSAGE(FATAL_ERROR "Could not find SNLS. SNLS support needs explicit SNLS_DIR")
endif()

# SNLS's installed cmake config target is lower case
if (ENABLE_SNLS_V03)
    set(snls_DIR "${SNLS_DIR}/share/snls/cmake/" )
    list(APPEND CMAKE_PREFIX_PATH ${snls_DIR})

    find_package(snls REQUIRED)

    set (SNLS_FOUND ${snls_FOUND} CACHE STRING "")

    set(SNLS_LIBRARIES snls)

    set(SNLS_DEPENDS)
    MESSAGE("SNLS RAJA_PERF_SUITE" ${SNLS_USE_RAJA_PERF_SUITE})
    blt_list_append(TO SNLS_DEPENDS ELEMENTS chai raja umpire camp IF ${SNLS_USE_RAJA_PERF_SUITE})

else()
    #find includes
    find_path( SNLS_INCLUDE_DIRS SNLS_lup_solve.h
               PATHS  ${SNLS_DIR}/include/ ${SNLS_DIR}
               NO_DEFAULT_PATH
               NO_CMAKE_ENVIRONMENT_PATH
               NO_CMAKE_PATH
               NO_SYSTEM_ENVIRONMENT_PATH
               NO_CMAKE_SYSTEM_PATH)

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set SNLS_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(SNLS  DEFAULT_MSG
                                      SNLS_INCLUDE_DIRS
                                      )
endif()

