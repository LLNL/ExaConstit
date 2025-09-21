# Provide backwards compatibility for *_PREFIX options
set(_tpls 
    camp
    raja
    umpire
    chai
    fmt
    snls
    exacmech
    mfem
    caliper
    threads)

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
# RAJA
################################

if (RAJA_DIR)
   find_package(RAJA REQUIRED CONFIG PATHS ${RAJA_DIR})
else()
   message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
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
# SNLS
################################

if (SNLS_DIR)
    find_package(SNLS REQUIRED CONFIG PATHS ${SNLS_DIR})
    set_target_properties(snls PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SNLS_INCLUDE_DIRS}")
endif()

if(SNLS_USE_RAJA_PORT_SUITE)
################################
# camp
################################

    if (CAMP_DIR)
        find_package(camp REQUIRED CONFIG PATHS ${CAMP_DIR})
    else()
        message(FATAL_ERROR "CAMP_DIR was not provided. It is needed to find CAMP.")
    endif()

################################
# chai
################################

    if (CHAI_DIR)
        set(umpire_DIR ${UMPIRE_DIR})
        set(raja_DIR ${RAJA_DIR})
        set(fmt_DIR ${FMT_DIR})
        find_package(chai REQUIRED CONFIG PATHS ${CHAI_DIR})
    else()
        message(FATAL_ERROR "CHAI_DIR was not provided. It is needed to find CHAI.")
    endif()

################################
# fmt
################################

    if (FMT_DIR)
        find_package(fmt CONFIG PATHS ${FMT_DIR})
    else()
        message(WARNING "FMT_DIR was not provided. This is a requirement for camp as of v2024.02.0. Ignore this warning if using older versions of the RAJA Portability Suite")
    endif()

################################
# UMPIRE
################################

    if (DEFINED UMPIRE_DIR)
        find_package(umpire REQUIRED CONFIG PATHS ${UMPIRE_DIR})
    else()
        message(FATAL_ERROR "UMPIRE_DIR was not provided. It is needed to find UMPIRE.")
    endif()
endif() # End SNLS_USE_RAJA_PORT_SUITE check

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

################################
# Threads (platform-specific)
################################

set(EXACONSTIT_THREADS_EXPLICIT_LINK FALSE CACHE INTERNAL "Whether explicit thread linking is required")

if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    include(CheckCXXSourceCompiles)
    
    # Test 1: Basic thread support without any flags
    set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_LIBRARIES "")
    
    check_cxx_source_compiles("
        #include <thread>
        #include <atomic>
        #include <mutex>
        #include <condition_variable>
        int main() { 
            std::atomic<int> counter{0};
            std::mutex m;
            std::condition_variable cv;
            
            std::thread t([&]{ 
                std::unique_lock<std::mutex> lock(m);
                counter++;
                cv.notify_one();
            }); 
            
            t.join(); 
            return counter.load(); 
        }" THREADS_IMPLICIT_LINK)
    
    # Test 2: If implicit didn't work, verify explicit works
    if(NOT THREADS_IMPLICIT_LINK)
        set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
        check_cxx_source_compiles("
            #include <thread>
            int main() { 
                std::thread t([]{}); 
                t.join(); 
                return 0; 
            }" THREADS_EXPLICIT_WORKS)
        
        if(NOT THREADS_EXPLICIT_WORKS)
            message(FATAL_ERROR "Threading support not functional even with explicit linking!")
        endif()
    endif()
    
    # Restore
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
    
    # Register if needed
    if(NOT THREADS_IMPLICIT_LINK)
        message(STATUS "  Result: Explicit pthread linking REQUIRED")
        set(EXACONSTIT_THREADS_EXPLICIT_LINK TRUE CACHE INTERNAL "Whether explicit thread linking is required")
    else()
        message(STATUS "  Result: pthread implicitly linked (no action needed)")
    endif()
    
elseif(APPLE)
    message(STATUS "Threads support built-in on macOS")
endif()