#!/usr/bin/bash
# Clean, robust ExaConstit GPU build script, with correct REBUILD behavior

set -Eeuo pipefail
trap 'echo "Build failed at line $LINENO while running: $BASH_COMMAND" >&2' ERR

# Resolve BASE_DIR portably, then operate from there
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
  SCRIPT=$(readlink -f "$0")
  BASE_DIR=$(dirname "$SCRIPT")
else
  SCRIPT="$0"
  BASE_DIR=$(cd "$(dirname "$SCRIPT")"; pwd -P)
fi
cd "$BASE_DIR"

# Toggles
REBUILD="${REBUILD:-OFF}"              # ON cleans build dir, OFF reuses if present
SYNC_SUBMODULES="${SYNC_SUBMODULES:-OFF}"  # Optional, set ON to resync .gitmodules for all repos

########################################
# Modules
########################################
module load cmake/3.29.2
module load rocmcc/6.4.2-magic
module load rocm/6.4.2
module load cray-mpich/9.0.1
module list

########################################
# Versions and branches
########################################
CAMP_VER="v2025.09.2"
RAJA_VER="v2025.09.1"
UMPIRE_VER="v2025.09.0"
CHAI_VER="v2025.09.1"

EXACMECH_REPO="https://github.com/LLNL/ExaCMech.git"
EXACMECH_BRANCH="develop"

HYPRE_VER="v2.32.0"
MFEM_REPO="https://github.com/rcarson3/mfem.git"
MFEM_BRANCH="exaconstit-smart-ptrs"

EXACONSTIT_REPO="https://github.com/llnl/ExaConstit.git"
EXACONSTIT_BRANCH="the_great_refactoring"

########################################
# Build options
########################################
OPENMP_ON="${OPENMP_ON:-OFF}"
ENABLE_HIP="ON"
ENABLE_TESTS_EXACONSTIT="${ENABLE_TESTS_EXACONSTIT:-ON}"
BUILD_SHARED_LIBS_DEFAULT="OFF"

PYTHON_VER="3.9.12"
CMAKE_PYTHON_EXE="/usr/tce/packages/python/python-${PYTHON_VER}/bin/python3"

########################################
# Toolchain and MPI
########################################
ROCM_VER_NUMBER="6.4.2"
ROCM_BASE="/usr/tce/packages/rocmcc/rocmcc-${ROCM_VER_NUMBER}-magic"

CMAKE_C_COMPILER="${ROCM_BASE}/bin/amdclang"
CMAKE_CXX_COMPILER="${ROCM_BASE}/bin/amdclang++"
CMAKE_HIP_COMPILER="${ROCM_BASE}/bin/amdclang++"

MPI_VER="9.0.1"
MPI_BASE="/usr/tce/packages/cray-mpich/cray-mpich-${MPI_VER}-rocmcc-${ROCM_VER_NUMBER}-magic"
MPI_C_COMPILER="${MPI_BASE}/bin/mpicc"
MPI_CXX_COMPILER="${MPI_BASE}/bin/mpicxx"
MPI_Fortran_COMPILER="${MPI_BASE}/bin/mpifort"

MPILIBHOME="/opt/cray/pe/mpich/${MPI_VER}/gtl/lib"
MPIAMDHOME="/opt/cray/pe/mpich/${MPI_VER}/ofi/amd/6.0/lib"
MPICRAYFLAGS="-Wl,-rpath,/opt/cray/libfabric/2.1/lib64:/opt/cray/pe/pmi/6.1.16/lib:/opt/cray/pe/pals/1.2.12/lib:/opt/rocm-${ROCM_VER_NUMBER}/llvm/lib -lxpmem"

########################################
# GPU arch and flags
########################################
CMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES:-gfx942:xnack+}"  # override with gfx942:xnack+ as needed
MFEM_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES:-gfx942}"  # override with gfx942 as MFEM doesn't play nice with xnack+ :(

GPU_TARGETS="${CMAKE_HIP_ARCHITECTURES}"
AMDGPU_TARGETS="${CMAKE_HIP_ARCHITECTURES}"

CMAKE_CXX_STANDARD="17"
CMAKE_CXX_FLAGS="-fPIC -std=c++17 -munsafe-fp-atomics"
CMAKE_C_FLAGS="-fPIC"
CMAKE_HIP_FLAGS="-munsafe-fp-atomics -fgpu-rdc"
CMAKE_EXE_LINKER_FLAGS="-lroctx64 -Wl,-rpath,${MPIAMDHOME} ${MPICRAYFLAGS} -L${MPILIBHOME} -lmpi_gtl_hsa -Wl,-rpath,${MPILIBHOME}"

########################################
# CHAI options
########################################
CHAI_DISABLE_RM="ON"
CHAI_THIN_GPU_ALLOCATE="ON"
CHAI_ENABLE_PINNED="ON"
CHAI_ENABLE_PICK="ON"
CHAI_DEBUG="OFF"
CHAI_ENABLE_GPU_SIMULATION_MODE="OFF"
CHAI_ENABLE_UM="ON"
CHAI_ENABLE_MANAGED_PTR="ON"
CHAI_ENABLE_MANAGED_PTR_ON_GPU="ON"

########################################
# Helpers
########################################
run_with_log() {
  local log="$1"; shift
  "$@" |& tee "$log"
}

# Clone only if missing, initialize submodules only on first clone
clone_if_missing() {
  local repo="$1" branch="$2" dest="$3"
  if [ ! -d "$dest/.git" ]; then
    git clone --branch "$branch" "$repo" "$dest"
    cd "$dest"
    if [ -f .gitmodules ]; then
      git submodule update --init --recursive
    fi
    cd "$BASE_DIR"
  fi
}

# Optional, force submodule sync and update when explicitly requested
sync_submodules() {
  local dest="$1"
  if [ "${SYNC_SUBMODULES}" = "ON" ] && [ -f "$dest/.gitmodules" ]; then
    cd "$dest"
    git submodule sync --recursive
    git submodule update --init --recursive
    cd "$BASE_DIR"
  fi
}

# Respect REBUILD flag when preparing build directories
prepare_build_dir() {
  local dir="$1"
  if [ "${REBUILD}" = "ON" ]; then
    mkdir -p "$dir"
    rm -rf "$dir"/*
  else
    if [ ! -d "$dir" ]; then
      mkdir -p "$dir"
    fi
  fi
}

check_required_paths() {
  local missing=0
  for p in "$@"; do
    if [[ "$p" == */bin/* ]]; then
      if [ ! -x "$p" ]; then echo "Missing executable: $p" >&2; missing=1; fi
    else
      if [ ! -e "$p" ]; then echo "Missing path: $p" >&2; missing=1; fi
    fi
  done
  if [ "$missing" -ne 0 ]; then exit 1; fi
}

preflight_summary() {
  echo "========== Preflight summary =========="
  echo "BASE_DIR: ${BASE_DIR}"
  echo "REBUILD: ${REBUILD}"
  echo "SYNC_SUBMODULES: ${SYNC_SUBMODULES}"
  echo "Compilers:"
  echo "  C:      ${CMAKE_C_COMPILER}"
  echo "  CXX:    ${CMAKE_CXX_COMPILER}"
  echo "  HIP:    ${CMAKE_HIP_COMPILER}"
  echo "MPI wrappers:"
  echo "  mpicc:  ${MPI_C_COMPILER}"
  echo "  mpicxx: ${MPI_CXX_COMPILER}"
  echo "  mpifort:${MPI_Fortran_COMPILER}"
  echo "GPU:"
  echo "  HIP arch: ${CMAKE_HIP_ARCHITECTURES}"
  echo "Flags:"
  echo "  CXX: ${CMAKE_CXX_FLAGS}"
  echo "  HIP: ${CMAKE_HIP_FLAGS}"
  echo "  EXE link: ${CMAKE_EXE_LINKER_FLAGS}"
  echo "Versions:"
  echo "  CAMP:   ${CAMP_VER}"
  echo "  RAJA:   ${RAJA_VER}"
  echo "  UMPIRE: ${UMPIRE_VER}"
  echo "  CHAI:   ${CHAI_VER}"
  echo "MFEM:"
  echo "  repo: ${MFEM_REPO}"
  echo "  branch: ${MFEM_BRANCH}"
  echo "======================================="
}

########################################
# Sanity checks
########################################
check_required_paths "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}" "${CMAKE_HIP_COMPILER}" "${MPI_C_COMPILER}" "${MPI_CXX_COMPILER}" "${MPI_Fortran_COMPILER}"
preflight_summary

########################################
# CAMP BUILD
########################################
clone_if_missing "https://github.com/LLNL/camp.git" "${CAMP_VER}" "${BASE_DIR}/camp"
sync_submodules "${BASE_DIR}/camp"

prepare_build_dir "${BASE_DIR}/camp/build"
cd "${BASE_DIR}/camp/build"
run_with_log my_camp_config cmake ../ \
  -DCMAKE_INSTALL_PREFIX=../install_dir/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_HIP_FLAGS="${CMAKE_HIP_FLAGS}" \
  -DENABLE_HIP="ON"
run_with_log my_camp_build make -j 2
run_with_log my_camp_install make install
CAMP_ROOT="${BASE_DIR}/camp/install_dir"
cd "${BASE_DIR}"

########################################
# RAJA BUILD
########################################
clone_if_missing "https://github.com/LLNL/RAJA.git" "${RAJA_VER}" "${BASE_DIR}/RAJA"
sync_submodules "${BASE_DIR}/RAJA"

prepare_build_dir "${BASE_DIR}/RAJA/build"
cd "${BASE_DIR}/RAJA/build"
run_with_log my_raja_config cmake ../ \
  -DCMAKE_INSTALL_PREFIX=../install_dir/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DRAJA_ENABLE_TESTS=OFF \
  -DRAJA_ENABLE_EXAMPLES=OFF \
  -DRAJA_ENABLE_BENCHMARKS=OFF \
  -DRAJA_ENABLE_REPRODUCERS=OFF \
  -DRAJA_ENABLE_EXERCISES=OFF \
  -DRAJA_ENABLE_VECTORIZATION=OFF \
  -DRAJA_ENABLE_DOCUMENTATION=OFF \
  -DRAJA_USE_DOUBLE=ON \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_HIP_FLAGS="${CMAKE_HIP_FLAGS}" \
  -DENABLE_HIP="ON" \
  -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
run_with_log my_raja_build make -j 4
run_with_log my_raja_install make install
RAJA_ROOT="${BASE_DIR}/RAJA/install_dir"
cd "${BASE_DIR}"

########################################
# UMPIRE BUILD
########################################
clone_if_missing "https://github.com/LLNL/Umpire.git" "${UMPIRE_VER}" "${BASE_DIR}/Umpire"
sync_submodules "${BASE_DIR}/Umpire"

prepare_build_dir "${BASE_DIR}/Umpire/build"
cd "${BASE_DIR}/Umpire/build"
run_with_log my_umpire_config cmake ../ \
  -DCMAKE_INSTALL_PREFIX=../install_dir/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DENABLE_MPI=OFF \
  -DUMPIRE_ENABLE_C=OFF \
  -DENABLE_FORTRAN=OFF \
  -DENABLE_GMOCK=OFF \
  -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=OFF \
  -DUMPIRE_ENABLE_TOOLS=ON \
  -DUMPIRE_ENABLE_BACKTRACE=ON \
  -DUMPIRE_ENABLE_BACKTRACE_SYMBOLS=ON \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_HIP_FLAGS="${CMAKE_HIP_FLAGS}" \
  -DENABLE_HIP="ON" \
  -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
run_with_log my_umpire_build make -j 8
run_with_log my_umpire_install make install
UMPIRE_ROOT="${BASE_DIR}/Umpire/install_dir"
cd "${BASE_DIR}"

# fmt detection inside Umpire
FMT_DIR_CMAKE=$(find "${UMPIRE_ROOT}" -name 'fmtConfig.cmake' -print -quit || true)
if [ -n "${FMT_DIR_CMAKE}" ]; then
  FMT_DIR=$(dirname "${FMT_DIR_CMAKE}")
else
  FMT_DIR="${UMPIRE_ROOT}"
fi

########################################
# CHAI BUILD
########################################
clone_if_missing "https://github.com/LLNL/CHAI.git" "${CHAI_VER}" "${BASE_DIR}/CHAI"
sync_submodules "${BASE_DIR}/CHAI"

prepare_build_dir "${BASE_DIR}/CHAI/build"
cd "${BASE_DIR}/CHAI/build"
run_with_log my_chai_config cmake ../ \
  -DCMAKE_INSTALL_PREFIX=../install_dir/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DENABLE_EXAMPLES=OFF \
  -DENABLE_DOCS=OFF \
  -DENABLE_GMOCK=OFF \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DENABLE_MPI=OFF \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_HIP_FLAGS="${CMAKE_HIP_FLAGS}" \
  -DENABLE_HIP="ON" \
  -DCHAI_ENABLE_RAJA_PLUGIN=ON \
  -DCHAI_ENABLE_RAJA_NESTED_TEST=OFF \
  -DCHAI_THIN_GPU_ALLOCATE="${CHAI_THIN_GPU_ALLOCATE}" \
  -DCHAI_ENABLE_PINNED="${CHAI_ENABLE_PINNED}" \
  -DCHAI_DISABLE_RM="${CHAI_DISABLE_RM}" \
  -DCHAI_ENABLE_PICK="${CHAI_ENABLE_PICK}" \
  -DCHAI_DEBUG="${CHAI_DEBUG}" \
  -DCHAI_ENABLE_GPU_SIMULATION_MODE="${CHAI_ENABLE_GPU_SIMULATION_MODE}" \
  -DCHAI_ENABLE_UM="${CHAI_ENABLE_UM}" \
  -DCHAI_ENABLE_MANAGED_PTR="${CHAI_ENABLE_MANAGED_PTR}" \
  -DCHAI_ENABLE_MANAGED_PTR_ON_GPU="${CHAI_ENABLE_MANAGED_PTR_ON_GPU}" \
  -Dfmt_DIR="${FMT_DIR}" \
  -Dumpire_DIR="${UMPIRE_ROOT}" \
  -DRAJA_DIR="${RAJA_ROOT}" \
  -Dcamp_DIR="${CAMP_ROOT}"
run_with_log my_chai_build make -j 4
run_with_log my_chai_install make install
CHAI_ROOT="${BASE_DIR}/CHAI/install_dir"
cd "${BASE_DIR}"

########################################
# ExaCMech BUILD
########################################
clone_if_missing "${EXACMECH_REPO}" "${EXACMECH_BRANCH}" "${BASE_DIR}/ExaCMech"
sync_submodules "${BASE_DIR}/ExaCMech"

prepare_build_dir "${BASE_DIR}/ExaCMech/build"
cd "${BASE_DIR}/ExaCMech/build"
run_with_log my_ecmech_config cmake ../ \
  -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF \
  -DENABLE_MINIAPPS=OFF \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DENABLE_HIP="ON" \
  -DFMT_DIR="${FMT_DIR}" \
  -DUMPIRE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire" \
  -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja" \
  -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai" \
  -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
run_with_log my_ecmech_build make -j 4
run_with_log my_ecmech_install make install
ECMECH_ROOT="${BASE_DIR}/ExaCMech/install_dir_hip"
cd "${BASE_DIR}"

########################################
# HYPRE BUILD
########################################
if [ ! -d "${BASE_DIR}/hypre" ]; then
  git clone https://github.com/hypre-space/hypre.git --branch "${HYPRE_VER}" --single-branch "${BASE_DIR}/hypre"
fi

prepare_build_dir "${BASE_DIR}/hypre/build"
cd "${BASE_DIR}/hypre/build"
run_with_log my_hypre_config cmake ../src \
  -DCMAKE_INSTALL_PREFIX=../src/hypre_hip/ \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}" \
  -DMPI_C_COMPILER="${MPI_C_COMPILER}" \
  -DCMAKE_BUILD_TYPE=Release
run_with_log my_hypre_build make -j 4
run_with_log my_hypre_install make install
HYPRE_ROOT="${BASE_DIR}/hypre/src/hypre_hip"
cd "${BASE_DIR}"

########################################
# METIS BUILD
########################################
if [ ! -d "${BASE_DIR}/metis-5.1.0" ]; then
  curl -o metis-5.1.0.tar.gz https://mfem.github.io/tpls/metis-5.1.0.tar.gz
  tar -xzf metis-5.1.0.tar.gz
  rm metis-5.1.0.tar.gz
fi
prepare_build_dir "${BASE_DIR}/metis-5.1.0/install_dir_hip"
cd "${BASE_DIR}/metis-5.1.0"
make distclean || true
run_with_log my_metis_config make config prefix="${BASE_DIR}/metis-5.1.0/install_dir_hip" CC="${CMAKE_C_COMPILER}" CXX="${CMAKE_CXX_COMPILER}"
run_with_log my_metis_build make -j 4
run_with_log my_metis_install make install
METIS_ROOT="${BASE_DIR}/metis-5.1.0/install_dir_hip"
cd "${BASE_DIR}"

########################################
# MFEM BUILD
########################################
clone_if_missing "${MFEM_REPO}" "${MFEM_BRANCH}" "${BASE_DIR}/mfem"
# No submodule sync here, keep local changes intact

prepare_build_dir "${BASE_DIR}/mfem/build_hip"
cd "${BASE_DIR}/mfem/build_hip"
run_with_log my_mfem_config cmake ../ \
  -DMFEM_USE_MPI=YES \
  -DMFEM_USE_SIMD=NO \
  -DMETIS_DIR="${METIS_ROOT}" \
  -DHYPRE_DIR="${HYPRE_ROOT}" \
  -DMFEM_USE_RAJA=YES \
  -DRAJA_DIR="${RAJA_ROOT}" \
  -DRAJA_REQUIRED_PACKAGES="camp" \
  -DMFEM_USE_CAMP=ON \
  -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp" \
  -DMFEM_USE_OPENMP="${OPENMP_ON}" \
  -DMFEM_USE_ZLIB=YES \
  -DCMAKE_CXX_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
  -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}" \
  -DMFEM_USE_HIP="ON" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_ARCH="${MFEM_HIP_ARCHITECTURES}" \
  -DCMAKE_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES}"
run_with_log my_mfem_build make -j 4
run_with_log my_mfem_install make install
MFEM_ROOT="${BASE_DIR}/mfem/install_dir_hip"
cd "${BASE_DIR}"

########################################
# ExaConstit BUILD
########################################
clone_if_missing "${EXACONSTIT_REPO}" "${EXACONSTIT_BRANCH}" "${BASE_DIR}/ExaConstit"
sync_submodules "${BASE_DIR}/ExaConstit"

prepare_build_dir "${BASE_DIR}/ExaConstit/build_hip"
cd "${BASE_DIR}/ExaConstit/build_hip"
run_with_log my_exconstit_config cmake ../ \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}" \
  -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER}" \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
  -DPYTHON_EXECUTABLE="${CMAKE_PYTHON_EXE}" \
  -DENABLE_TESTS="${ENABLE_TESTS_EXACONSTIT}" \
  -DENABLE_OPENMP="${OPENMP_ON}" \
  -DENABLE_FORTRAN=OFF \
  -DENABLE_HIP="ON" \
  -DCMAKE_INSTALL_PREFIX=../install_dir/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
  -DMFEM_DIR="${MFEM_ROOT}/lib/cmake/mfem" \
  -DECMECH_DIR="${ECMECH_ROOT}" \
  -DSNLS_DIR="${ECMECH_ROOT}" \
  -DFMT_DIR="${FMT_DIR}" \
  -DUMPIRE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire" \
  -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja" \
  -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai" \
  -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
run_with_log my_exconstit_build make -j 4
EXACONSTIT_ROOT="${BASE_DIR}/ExaConstit/install_dir"
echo "ExaConstit install prefix: ${EXACONSTIT_ROOT}"