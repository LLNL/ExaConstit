set(MESHING_DEPENDS )

#SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DDEBUG")
#SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(MESHING_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  MESHING_DEPENDS
                             DEPENDS_ON  mfem mpi)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND MESHING_DEPENDS blt::cuda_runtime blt::cuda)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND MESHING_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND MESHING_DEPENDS cuda cuda_runtime)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND MESHING_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND MESHING_DEPENDS blt::hip blt::hip_runtime)
endif()

message("-- MESHING_DEPENDS: ${MESHING_DEPENDS}")

blt_add_executable(NAME       mesh_generator
                   SOURCES    mesh_generator.cpp
                   OUTPUT_DIR ${SCRIPTS_OUTPUT_DIRECTORY}
                   DEPENDS_ON ${MESHING_DEPENDS})
