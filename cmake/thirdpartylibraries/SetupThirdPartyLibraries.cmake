# Provide backwards compatibility for *_PREFIX options
set(_tpls 
    mfem
    raja
    snls
    exacmech
    caliper)

foreach(_tpl ${_tpls})
    string(TOUPPER ${_tpl} _uctpl)
    if (${_uctpl}_PREFIX)
        set(${_uctpl}_DIR ${${_uctpl}_PREFIX} CACHE PATH "")
        mark_as_advanced(${_uctpl}_PREFIX)
    endif()
endforeach()

################################
# MFEM
################################

if (DEFINED MFEM_DIR)
    include(cmake/thirdpartylibraries/FindMFEM.cmake)
    if (MFEM_FOUND)
        blt_register_library( NAME       mfem
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${MFEM_INCLUDE_DIRS}
                              LIBRARIES  ${MFEM_LIBRARIES})
    if (ENABLE_HIP)
        find_package(HIPSPARSE REQUIRED)
    endif()
    else()
        message(FATAL_ERROR "Unable to find MFEM with given path ${MFEM_DIR}")
    endif()
else()
    message(FATAL_ERROR "MFEM_DIR was not provided. It is needed to find MFEM.")
endif()


################################
# ExaCMech
################################

if (DEFINED ECMECH_DIR)
    include(cmake/thirdpartylibraries/FindECMech.cmake)
    if (ECMECH_FOUND)
        blt_register_library( NAME       ecmech
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${ECMECH_INCLUDE_DIRS}
                              LIBRARIES  ${ECMECH_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find ExaCMech with given path ${ECMECH_DIR}")
    endif()
else()
    message(FATAL_ERROR "ECMECH_DIR was not provided. It is needed to find ExaCMech.")
endif()

################################
# RAJA
################################

if (DEFINED RAJA_DIR)
    include(cmake/thirdpartylibraries/FindRAJA.cmake)
    if (RAJA_FOUND)
        blt_register_library( NAME       raja
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${RAJA_INCLUDE_DIRS}
                              LIBRARIES  ${RAJA_LIBRARY}
                              DEPENDS_ON camp)
    else()
        message(FATAL_ERROR "Unable to find RAJA with given path ${RAJA_DIR}")
    endif()
else()
    message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()

################################
# SNLS
################################

if (SNLS_DIR)
    include(cmake/thirdpartylibraries/FindSNLS.cmake)
    if (SNLS_FOUND)
        blt_register_library( NAME       snls
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${SNLS_INCLUDE_DIRS}
                              LIBRARIES  ${SNLS_LIBRARIES}
                              DEPENDS_ON ${SNLS_DEPENDS})
    else()
        message(FATAL_ERROR "Unable to find SNLS with given path ${SNLS_DIR}")
    endif()
endif()

################################
# Caliper
################################

if (DEFINED CALIPER_DIR)
    include(cmake/thirdpartylibraries/FindCaliper.cmake)
    if (CALIPER_FOUND)
        blt_register_library( NAME       caliper
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${caliper_INCLUDE_DIR}
                              LIBRARIES  ${CALIPER_LIBRARY})
        option(ENABLE_CALIPER "Enable CALIPER" ON)
    else()
        message(FATAL_ERROR "Unable to find Caliper with given path ${CALIPER_DIR}")
    endif()
else()
    message("Caliper support disabled")
endif()