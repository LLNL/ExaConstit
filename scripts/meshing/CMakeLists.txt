set(MESHING_DEPENDS )

#SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DDEBUG")
#SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")


if(ENABLE_OPENMP)
   list(APPEND MESHING_DEPENDS openmp)
endif()

if(ENABLE_CUDA)
   list(APPEND MESHING_DEPENDS cuda)
endif()

blt_add_executable(NAME       mesh_generator
                   SOURCES    mesh_generator.cpp
                   OUTPUT_DIR ${SCRIPTS_OUTPUT_DIRECTORY}
                   DEPENDS_ON ${MESHING_DEPENDS} mfem mpi)
