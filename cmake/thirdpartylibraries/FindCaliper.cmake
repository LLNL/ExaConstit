###############################################################################
#
# Setup Caliper
# This file defines:
#  CALIPER_FOUND - If Caliper was found
#  CALIPER_INCLUDE_DIRS - The Caliper include directories
#  CALIPER_LIBRARY - The Caliper library

if(NOT CALIPER_DIR)
    MESSAGE(FATAL_ERROR "Could not find Caliper. Caliper support needs explicit CALIPER_DIR")
endif()

if (NOT CALIPER_CONFIG_CMAKE)
   set(CALIPER_CONFIG_CMAKE "${CALIPER_DIR}/caliper-config.cmake")
endif()
if (EXISTS "${CALIPER_CONFIG_CMAKE}")
   include("${CALIPER_CONFIG_CMAKE}")
endif()
if (NOT CALIPER_RELEASE_CMAKE)
   set(CALIPER_RELEASE_CMAKE "${CALIPER_DIR}/caliper-release.cmake")
endif()
if (EXISTS "${CALIPER_RELEASE_CMAKE}")
   include("${CALIPER_RELEASE_CMAKE}")
endif()

find_package(CALIPER)

include_directories(${caliper_INCLUDE_DIR})

find_library( CALIPER_LIBRARY NAMES caliper libcaliper
              PATHS ${caliper_LIB_DIR} ${CALIPER_DIR}/../../../lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set CALIPER_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CALIPER  DEFAULT_MSG
                                  caliper_INCLUDE_DIR
                                  CALIPER_LIBRARY)
