# Defines the following variables:
#   - CONDUIT_FOUND
#   - CONDUIT_LIBRARIES
#   - CONDUIT_INCLUDE_DIRS

#If find_package doesn't find CONDUIT it will overwrite CONDUIT_DIR which we will need for alternative ways of finding the package
set(CONDUIT_DIR_TEMP ${CONDUIT_DIR})
#If this was installed using SPACK this will fail and so we have to manually find things.
find_package(CONDUIT
             NO_DEFAULT_PATH
             PATHS ${CONDUIT_DIR}/lib/cmake)

if(CONDUIT_FOUND)

  set(CONDUIT_LIBRARIES ${conduit::conduit})


  if(CONDUIT_RELAY_HDF5_ENABLED)
     set(HDF5_ROOT ${CONDUIT_HDF5_DIR})
     find_package(HDF5 REQUIRED)
     message("HDF5 Libraries found: ${HDF5_LIBRARIES}")
  else()
     set(HDF5_LIBRARIES "")
  endif()

else()
  #Find package appears to be erasing this variable if it isn't found...
  set(CONDUIT_DIR ${CONDUIT_DIR_TEMP} CACHE PATH "" FORCE)

  find_path( CONDUIT_INCLUDE_DIRS NAMES conduit.h
             PATHS  ${CONDUIT_DIR}/include/ ${CONDUIT_DIR}/include/conduit/
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)
  #We have three different libraries that we need to find

  find_library( CONDUIT_LIBRARY NAMES conduit
                PATHS ${CONDUIT_DIR}/lib/
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                                NO_CMAKE_SYSTEM_PATH)

  find_library( CONDUIT_RELAY_LIBRARY NAMES relay conduit_relay
                PATHS ${CONDUIT_DIR}/lib/
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH)

  find_library( CONDUIT_BLUEPRINT_LIBRARY NAMES blueprint conduit_blueprint
                PATHS ${CONDUIT_DIR}/lib/
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH)

  set(CONDUIT_LIBRARIES ${CONDUIT_LIBRARY} ${CONDUIT_RELAY_LIBRARY} ${CONDUIT_BLUEPRINT_LIBRARY})
  #We need to double check and see if Conduit was built with HDF5 support and if so we need to find the passed in
  #HDF5 directory or else we'll most likely just use the built in system version which might cause issues
  if(EXISTS ${CONDUIT_DIR}/include/conduit/conduit_relay_hdf5.hpp)
    message(STATUS "Conduit Relay HDF5 Support is ENABLED")
    find_package(HDF5 REQUIRED)
  else()
    set(HDF5_LIBRARIES "")
  endif()

endif()

find_package_handle_standard_args(CONDUIT  DEFAULT_MSG
                                  CONDUIT_INCLUDE_DIRS
                                  CONDUIT_LIBRARIES )