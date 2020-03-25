# Provide backwards compatibility for *_PREFIX options
set(_tpls 
    mfem
    raja
    snls
    exacmech)

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
                              LIBRARIES  ${RAJA_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find RAJA with given path ${RAJA_DIR}")
    endif()
else()
    message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()

################################
# SNLS
################################

if (DEFINED SNLS_DIR)
    include(cmake/thirdpartylibraries/FindSNLS.cmake)
    if (SNLS_FOUND)
        blt_register_library( NAME       snls
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${SNLS_INCLUDE_DIRS}
                              LIBRARIES  ${SNLS_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find SNLS with given path ${SNLS_DIR}")
    endif()
else()
    message(FATAL_ERROR "SNLS_DIR was not provided. It is needed to find SNLS.")
endif()