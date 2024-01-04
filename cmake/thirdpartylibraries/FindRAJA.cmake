###############################################################################
#
# Setup RAJA
# This file defines:
#  RAJA_FOUND - If RAJA was found
#  RAJA_INCLUDE_DIRS - The RAJA include directories
#  RAJA_LIBRARY - The RAJA library

# first Check for RAJA_DIR

if(NOT RAJA_DIR)
    MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA support needs explicit RAJA_DIR")
endif()

if (NOT RAJA_CONFIG_CMAKE)
   set(RAJA_CONFIG_CMAKE "${RAJA_DIR}/share/raja/cmake/raja-config.cmake")
endif()
if (EXISTS "${RAJA_CONFIG_CMAKE}")
   include("${RAJA_CONFIG_CMAKE}")
endif()
if (NOT RAJA_RELEASE_CMAKE)
   set(RAJA_RELEASE_CMAKE "${RAJA_DIR}/share/raja/cmake/raja-release.cmake")
endif()
if (EXISTS "${RAJA_RELEASE_CMAKE}")
   include("${RAJA_RELEASE_CMAKE}")
endif()

find_package(RAJA REQUIRED)

if(camp_DIR AND (RAJA_VERSION_MINOR GREATER 10 OR RAJA_VERSION_MAJOR GREATER 0))
   find_package(camp REQUIRED
      NO_DEFAULT_PATH
      PATHS ${camp_DIR}
      ${camp_DIR}/lib/cmake/camp
   )
   set(ENABLE_CAMP ON CACHE BOOL "")
endif()

if(RAJA_CONFIG_LOADED)
   if(ENABLE_OPENMP)
      set(BLT_CXX_FLAGS "${BLT_CXX_FLAGS} -fopenmp" CACHE PATH "")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" CACHE STRING "" FORCE)
   endif()
endif()

get_property(RAJA_INCLUDE_DIRS TARGET RAJA PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})

find_library( RAJA_LIBRARY NAMES RAJA libRAJA
              PATHS ${RAJA_LIB_DIR} ${RAJA_DIR}/../../../lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RAJA  DEFAULT_MSG
                                  RAJA_INCLUDE_DIRS
                                  RAJA_LIBRARY )
