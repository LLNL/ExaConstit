#!/usr/bin/env bash
# MFEM-stack build functions: Hypre, METIS, MFEM.
#
# Depends on the helpers in build_helpers.sh and the common stack
# defined in build_functions_common.sh (specifically RAJA / CAMP,
# which MFEM consumes).

###########################################
# Hypre
###########################################
build_hypre() {
  echo "=========================================="
  echo "Building Hypre"
  echo "=========================================="

  if [ ! -d "${BASE_DIR}/hypre" ]; then
    git clone https://github.com/hypre-space/hypre.git --branch "${HYPRE_VER}" --single-branch "${BASE_DIR}/hypre"
  fi

  prepare_build_dir "${BASE_DIR}/hypre/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/hypre/build_${BUILD_SUFFIX}"

  run_with_log my_hypre_config cmake ../src \
    -DCMAKE_INSTALL_PREFIX=../src/hypre_${BUILD_SUFFIX}/ \
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}" \
    -DMPI_C_COMPILER="${MPI_C_COMPILER}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

  run_with_log my_hypre_build make -j "${MAKE_JOBS}"
  run_with_log my_hypre_install make install

  HYPRE_ROOT="${BASE_DIR}/hypre/src/hypre_${BUILD_SUFFIX}"
  export HYPRE_ROOT
  echo "Hypre installed to: ${HYPRE_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# METIS
###########################################
build_metis() {
  echo "=========================================="
  echo "Building METIS"
  echo "=========================================="

  if [ ! -d "${BASE_DIR}/metis-${METIS_VER}" ]; then
    curl -o metis-${METIS_VER}.tar.gz "${METIS_URL}"
    tar -xzf metis-${METIS_VER}.tar.gz
    rm metis-${METIS_VER}.tar.gz
  fi

  cd "${BASE_DIR}/metis-${METIS_VER}"

  # METIS doesn't have a proper incremental build, so always clean
  make distclean 2>/dev/null || true

  prepare_build_dir "${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}"

  run_with_log my_metis_config make config \
    prefix="${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}" \
    CC="${CMAKE_C_COMPILER}" \
    CXX="${CMAKE_CXX_COMPILER}"

  run_with_log my_metis_build make -j "${MAKE_JOBS}"
  run_with_log my_metis_install make install

  METIS_ROOT="${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}"
  export METIS_ROOT
  echo "METIS installed to: ${METIS_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# SuperLU_DIST
###########################################
# Exact distributed sparse direct solver. MFEM links it (MFEM_USE_SUPERLU) and
# the AMGF mortar-PBC preconditioner uses it for the filtered-subspace solve.
#
# Built WITHOUT ParMETIS: METIS is the only graph-partitioning dependency in
# this stack, and SuperLU_DIST's METIS ordering is reachable only through
# ParMETIS, so it is disabled here. SuperLU falls back to its built-in
# MMD_AT_PLUS_A ordering, which is what the subspace solver requests and is
# fine for the small boundary-coupled block.
#
# Requires SUPERLU_REPO and SUPERLU_VER to be set alongside the other version
# variables (MFEM_REPO / MFEM_BRANCH / HYPRE_VER / METIS_VER).
build_superlu() {
  if [ "${ENABLE_SUPERLU:-OFF}" != "ON" ]; then
    echo "ENABLE_SUPERLU != ON; skipping SuperLU_DIST build."
    return 0
  fi
 
  echo "=========================================="
  echo "Building SuperLU_DIST"
  echo "=========================================="
 
  clone_if_missing "${SUPERLU_REPO}" "${SUPERLU_VER}" "${BASE_DIR}/superlu_dist"
 
  prepare_build_dir "${BASE_DIR}/superlu_dist/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/superlu_dist/build_${BUILD_SUFFIX}"
 
  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DMPI_C_COMPILER="${MPI_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBUILD_SHARED_LIBS=OFF
    # METIS-only stack: no ParMETIS (see header note). SuperLU uses its
    # built-in MMD_AT_PLUS_A ordering.
    -DTPL_ENABLE_PARMETISLIB=OFF
    # Integer width must match the Hypre build (32-bit here).
    -DXSDK_INDEX_SIZE=32
    -Denable_double=ON
    -Denable_single=OFF
    -Denable_complex16=OFF
    -Denable_tests=OFF
    -Denable_examples=OFF
    # Self-contained internal CBLAS; the subspace blocks are small.
    -DTPL_ENABLE_INTERNAL_BLASLIB=ON
  )
 
  if [ "${BUILD_TYPE}" = "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}"
      -Denable_openmp="${OPENMP_ON}"
    )
  else
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${CMAKE_GPU_COMPILER}"
      -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
      -Denable_openmp="${OPENMP_ON}"
    )
 
    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DTPL_ENABLE_CUDALIB=TRUE
        -DCMAKE_CUDA_COMPILER="${CMAKE_GPU_COMPILER}"
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
        -DCMAKE_CUDA_FLAGS="${CMAKE_GPU_FLAGS}"
      )
    elif [ "${GPU_BACKEND}" = "HIP" ]; then
      CMAKE_ARGS+=(
        -DTPL_ENABLE_HIPLIB=TRUE
        -DCMAKE_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES}"
      )
    fi
  fi
 
  run_with_log my_superlu_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_superlu_build make -j "${MAKE_JOBS}"
  run_with_log my_superlu_install make install
 
  SUPERLU_ROOT="${BASE_DIR}/superlu_dist/install_${BUILD_SUFFIX}"
  export SUPERLU_ROOT
  echo "SuperLU_DIST installed to: ${SUPERLU_ROOT}"
  cd "${BASE_DIR}"
}


###########################################
# MFEM
###########################################
build_mfem() {
  echo "=========================================="
  echo "Building MFEM"
  echo "=========================================="

  clone_if_missing "${MFEM_REPO}" "${MFEM_BRANCH}" "${BASE_DIR}/mfem"
  # Don't sync submodules for MFEM to preserve local changes

  prepare_build_dir "${BASE_DIR}/mfem/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/mfem/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DMFEM_USE_MPI=YES
    -DMFEM_USE_SIMD=NO
    -DMETIS_DIR="${METIS_ROOT}"
    -DHYPRE_DIR="${HYPRE_ROOT}"
    -DMFEM_USE_RAJA=YES
    -DRAJA_DIR="${RAJA_ROOT}"
    -DRAJA_REQUIRED_PACKAGES="camp"
    -DMFEM_USE_CAMP=ON
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
    -DMFEM_USE_OPENMP="${OPENMP_ON}"
    -DMFEM_USE_ZLIB=YES
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )

  # Optional: link the SuperLU_DIST built by build_superlu (AMGF mortar-PBC
  # subspace solve). Only added when ENABLE_SUPERLU=ON so the default build is
  # unaffected.
  if [ "${ENABLE_SUPERLU:-OFF}" = "ON" ]; then
    CMAKE_ARGS+=(
      -DMFEM_USE_SUPERLU=YES
      -DSuperLUDist_DIR="${SUPERLU_ROOT}"
    )
  fi


  if [ "${BUILD_TYPE}" = "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}"
    )
  else
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${CMAKE_GPU_COMPILER}"
      -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
      -DMFEM_USE_${GPU_BACKEND}=ON
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )

    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DCMAKE_CUDA_COMPILER="${CMAKE_GPU_COMPILER}"
        -DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CXX_COMPILER}"
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
        -DCMAKE_CUDA_FLAGS="${CMAKE_GPU_FLAGS}"
        -DENABLE_CUDA=ON
      )
    elif [ "${GPU_BACKEND}" = "HIP" ]; then
      CMAKE_ARGS+=(
        -DHIP_ARCH="${MFEM_HIP_ARCHITECTURES}"
        -DCMAKE_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES}"
      )
    fi
  fi

  run_with_log my_mfem_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_mfem_build make -j "${MAKE_JOBS}"
  run_with_log my_mfem_install make install

  MFEM_ROOT="${BASE_DIR}/mfem/install_${BUILD_SUFFIX}"
  export MFEM_ROOT
  echo "MFEM installed to: ${MFEM_ROOT}"
  cd "${BASE_DIR}"
}
