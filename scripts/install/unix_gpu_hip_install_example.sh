#!/usr/bin/bash 
# For ease all of this should be run in its own directory

SCRIPT=$(readlink -f "$0")
BASE_DIR=$(dirname "$SCRIPT")

echo $BASH_VERSION

# This is a bit system dependent but for El Capitan-like systems the below should work
# You should be able to modify it to work for your own system easily enough.
# Most of the options are defined by the first set of bash variables defined
# below. You'll likely need to modify the ROCM_BASE, MPIHOME, and then the various
# MPI/linker flags
# While this is largely targeted towards AMD GPU builds, you can probably update
# it easily enough for a NVidia GPU build of things...
module load cmake/3.29.2 rocmcc/6.3.1-magic rocm/6.3.1 cray-mpich/8.1.31

ROCM_BASE="/usr/tce/packages/rocmcc/rocmcc-6.3.1-magic/"
CC="${ROCM_BASE}/bin/amdclang"
CXX="${ROCM_BASE}/bin/amdclang++"
HIPCC="${ROCM_BASE}/bin/hipcc"
MPIHOME="/usr/tce/packages/cray-mpich/cray-mpich-8.1.31-rocmcc-6.3.1-magic/"
MPILIBHOME="/opt/cray/pe/mpich/8.1.31/gtl/lib"
MPIAMDHOME="/opt/cray/pe/mpich/8.1.31/ofi/amd/6.0/lib"
MPICRAYFLAGS="-Wl,-rpath,/opt/cray/libfabric/2.1/lib64:/opt/cray/pe/pmi/6.1.15/lib:/opt/cray/pe/pals/1.2.12/lib:/opt/rocm-6.3.1/llvm/lib -lxpmem"
MPICXX="$MPIHOME/bin/mpicxx"
MPICC="$MPIHOME/bin/mpicc"
MPIFORT="$MPIHOME/bin/mpifort"
ROCMON="ON"
OPENMP_ON="OFF"
LOC_ROCM_ARCH="gfx942"
GPU_TARGETS="gfx942"
AMDGPU_TARGETS="gfx942"
CXX_FLAGS="-fPIC -std=c++17 -munsafe-fp-atomics"

EXE_LINK_FLAGS="--hip-link -lroctx64 -Wl,-rpath,${MPIAMDHOME} ${MPICRAYFLAGS} -L${MPILIBHOME} -lmpi_gtl_hsa -Wl,-rpath,${MPILIBHOME}"
PYTHON_EXE="/usr/tce/packages/python/python-3.9.12/bin/python3"
# Various build options for our various libaries
UMPIRE_ENABLE_TOOLS="ON"
UMPIRE_ENABLE_BACKTRACE="ON"
UMPIRE_ENABLE_BACKTRACE_SYMBOLS="ON"
# On V100s turn this off
CHAI_DISABLE_RM="ON"
# Only for MI300a s other systems we need to turn this off
CHAI_THIN_GPU_ALLOCATE="ON"
CHAI_ENABLE_PINNED="ON"
CHAI_ENABLE_PICK="ON"
CHAI_DEBUG="OFF"
CHAI_ENABLE_GPU_SIMULATION_MODE="OFF"
CHAI_ENABLE_UM="ON"
CHAI_ENABLE_MANAGED_PTR="ON"
CHAI_ENABLE_MANAGED_PTR_ON_GPU="ON"

#Build camp
if [ ! -d "camp" ]; then
    git clone https://github.com/LLNL/camp.git -b v2024.07.0
    cd ${BASE_DIR}/camp
    git submodule init
    git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/camp/build_hip" ]; then
      cd ${BASE_DIR}/camp
      mkdir build_hip
      cd ${BASE_DIR}/camp/build_hip
      rm -rf *
      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF \
                -DENABLE_OPENMP=OFF \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${HIPCC} \
                -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
                -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
                -DENABLE_HIP=$ROCMON
      make -j 2
      make install
fi

CAMP_ROOT=${BASE_DIR}/camp/install_dir_hip/
echo ${CAMP_ROOT}
cd ${BASE_DIR}

#exit
if [ ! -d "RAJA" ]; then 
   git clone https://github.com/LLNL/RAJA.git -b v2024.07.0
   cd ${BASE_DIR}/RAJA
   git submodule init
   git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/RAJA/build_hip" ]; then
      cd ${BASE_DIR}/RAJA
      mkdir build_hip
      cd ${BASE_DIR}/RAJA/build_hip
      rm -rf *
      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
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
                -DRAJA_USE_BARE_PTR=ON \
                -DRAJA_TIMER=chrono \
                -DENABLE_OPENMP=${OPENMP_ON} \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${HIPCC} \
                -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
                -DENABLE_HIP=${ROCMON} \
                -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
                -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DHIP_CXX_COMPILER=${HIPCC} \
                -Dcamp_DIR=${CAMP_ROOT}
      make -j 4
      make install
fi

RAJA_ROOT=${BASE_DIR}/RAJA/install_dir_hip/
echo ${RAJA_ROOT}
cd ${BASE_DIR}

if [ ! -d "Umpire" ]; then 
   git clone https://github.com/LLNL/Umpire.git -b v2024.07.0
   cd ${BASE_DIR}/Umpire
   git submodule init
   git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/Umpire/build_hip" ]; then
      cd ${BASE_DIR}/Umpire
      mkdir build_hip
      cd ${BASE_DIR}/Umpire/build_hip
      rm -rf *

      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF \
                -DENABLE_OPENMP=${OPENMP_ON} \
                -DENABLE_MPI=OFF \
                -DUMPIRE_ENABLE_C=OFF \
                -DENABLE_FORTRAN=OFF \
                -DENABLE_GMOCK=OFF \
                -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=OFF \
                -DUMPIRE_ENABLE_TOOLS=${UMPIRE_ENABLE_TOOLS} \
                -DUMPIRE_ENABLE_BACKTRACE=${UMPIRE_ENABLE_BACKTRACE} \
                -DUMPIRE_ENABLE_BACKTRACE_SYMBOLS=${UMPIRE_ENABLE_BACKTRACE_SYMBOLS} \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${HIPCC} \
                -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
                -DENABLE_HIP=${ROCMON} \
                -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
                -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DHIP_CXX_COMPILER=${HIPCC} \
                -Dcamp_DIR=${CAMP_ROOT}

      make -j 4
      make install
fi

UMPIRE_ROOT=${BASE_DIR}/Umpire/install_dir_hip/
echo ${UMPIRE_ROOT}
cd ${BASE_DIR}

if [ ! -d "CHAI" ]; then 
   git clone https://github.com/LLNL/CHAI.git -b v2024.07.0
   cd ${BASE_DIR}/CHAI
   git submodule init
   git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/CHAI/build_hip" ]; then
      cd ${BASE_DIR}/CHAI
      mkdir build_hip
      cd ${BASE_DIR}/CHAI/build_hip
      rm -rf *

      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF \
                -DENABLE_EXAMPLES=OFF \
                -DENABLE_DOCS=OFF \
                -DENABLE_GMOCK=OFF \
                -DENABLE_OPENMP=${OPENMP_ON} \
                -DENABLE_MPI=OFF \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${HIPCC} \
                -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
                -DENABLE_HIP=${ROCMON} \
                -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
                -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
                -DHIP_CXX_COMPILER=${HIPCC} \
                -DCHAI_ENABLE_RAJA_PLUGIN=ON \
                -DCHAI_ENABLE_RAJA_NESTED_TEST=OFF \
                -DCHAI_ENABLE_PINNED=${CHAI_ENABLE_PINNED} \
                -DCHAI_DISABLE_RM=${CHAI_DISABLE_RM} \
                -DCHAI_THIN_GPU_ALLOCATE=${CHAI_THIN_GPU_ALLOCATE} \
                -DCHAI_ENABLE_PICK=${CHAI_ENABLE_PICK} \
                -DCHAI_DEBUG=${CHAI_DEBUG} \
                -DCHAI_ENABLE_GPU_SIMULATION_MODE=${CHAI_ENABLE_GPU_SIMULATION_MODE} \
                -DCHAI_ENABLE_UM=${CHAI_ENABLE_UM} \
                -DCHAI_ENABLE_MANAGED_PTR=${CHAI_ENABLE_MANAGED_PTR} \
                -DCHAI_ENABLE_MANAGED_PTR_ON_GPU=${CHAI_ENABLE_MANAGED_PTR_ON_GPU} \
                -Dfmt_DIR=${UMPIRE_ROOT} \
                -Dumpire_DIR=${UMPIRE_ROOT} \
                -DRAJA_DIR=${RAJA_ROOT} \
                -Dcamp_DIR=${CAMP_ROOT}
      make -j 4
      make install
fi

CHAI_ROOT=${BASE_DIR}/CHAI/install_dir_hip/
echo ${CHAI_ROOT}
cd ${BASE_DIR}

if [ ! -d "ExaCMech" ]; then
      # Clone the repo
    git clone https://github.com/LLNL/ExaCMech.git
    cd ${BASE_DIR}/ExaCMech
   # Checkout the branch that has the HIP features on it
    git checkout develop
   # Update all the various submodules                  
    git submodule init && git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/ExaCMech/build_hip" ]; then
       cd ${BASE_DIR}/ExaCMech
       mkdir build_hip
       cd ${BASE_DIR}/ExaCMech/build_hip
       rm -rf *       

       cmake ../  -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DENABLE_TESTS=OFF \
                  -DENABLE_MINIAPPS=OFF \
                  -DENABLE_OPENMP=${OPENMP_ON} \
                  -DBUILD_SHARED_LIBS=OFF \
                  -DCMAKE_CXX_COMPILER=${HIPCC} \
                  -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
                  -DENABLE_HIP=$ROCMON \
                  -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
                  -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
                  -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
                  -DHIP_CXX_COMPILER=${HIPCC} \
                  -DFMT_DIR=${UMPIRE_ROOT}/lib64/cmake/fmt \
                  -DUMPIRE_DIR=${UMPIRE_ROOT}/lib64/cmake/umpire \
                  -DRAJA_DIR=${RAJA_ROOT}/lib/cmake/raja \
                  -DCHAI_DIR=${CHAI_ROOT}/lib/cmake/chai \
                  -DCAMP_DIR=${CAMP_ROOT}/lib/cmake/camp
       
       make -j 4
       make install
fi

ECMECH_ROOT=${BASE_DIR}/ExaCMech/install_dir_hip/
echo ${ECMECH_ROOT}
cd ${BASE_DIR}

# Now to build our MFEM dependencies
# First let's install Hypre v2.23.0
cd ${BASE_DIR}
if [ ! -d "hypre" ]; then
  git clone https://github.com/hypre-space/hypre.git --branch v2.32.0 --single-branch
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/hypre/build_hip" ]; then
  cd ${BASE_DIR}/hypre/
  mkdir build_hip
  cd ${BASE_DIR}/hypre/build_hip
  rm -rf *
  # Based on their install instructions
  # This should work on most systems
  # Hypre's default suggestions of just using configure don't always work
  cmake ../src  -DCMAKE_INSTALL_PREFIX=../src/hypre_hip/ \
                -DCMAKE_C_COMPILER=${CC} \
                -DMPI_CXX_COMPILER=${MPICXX} \
                -DMPI_C_COMPILER=${MPICC} \
                -DCMAKE_BUILD_TYPE=Release \
                |& tee my_hypre_config
  
  make -j 4 |& tee my_hypre_build
  make install |& tee my_hypre_install

  cd ${BASE_DIR}/hypre/src/hypre_hip
  HYPRE_ROOT="$(pwd)"

else

  echo " hypre already built "
  HYPRE_ROOT=${BASE_DIR}/hypre/src/hypre_hip

fi

cd ${BASE_DIR}

if [ ! -d "metis-5.1.0" ]; then
  
  curl -o metis-5.1.0.tar.gz https://mfem.github.io/tpls/metis-5.1.0.tar.gz
  tar -xzf metis-5.1.0.tar.gz
  rm metis-5.1.0.tar.gz
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/metis-5.1.0/install_dir_hip" ]; then
  cd ${BASE_DIR}/metis-5.1.0
  mkdir install_dir_hip
  make distclean
  make config prefix=${BASE_DIR}/metis-5.1.0/install_dir_hip/ CC=${CC} CXX=${CXX} |& tee my_metis_config
  make -j 4 |& tee my_metis_build
  make install |& tee my_metis_install
  cd ${BASE_DIR}/metis-5.1.0/install_dir_hip/
  METIS_ROOT="$(pwd)"
else
  echo " metis-5.1.0 already built "
  METIS_ROOT=${BASE_DIR}/metis-5.1.0/install_dir_hip/
fi

# cd ${BASE_DIR}
# if [ ! -d "ADIOS2" ]; then
#       # Clone the repo
#     git clone https://github.com/ornladios/ADIOS2.git
#     cd ${BASE_DIR}/ADIOS2
#    # Checkout the branch that has the HIP features on it
#     git checkout v2.10.0
#    # Update all the various submodules                  
#     git submodule init && git submodule update
# fi
# cd ${BASE_DIR}
# if [ ! -d "${BASE_DIR}/ADIOS2/build_hip" ]; then
#        cd ${BASE_DIR}/ADIOS2
#        mkdir build_hip
#        cd ${BASE_DIR}/ADIOS2/build_hip
#        rm -rf *       

#        cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
#                  -DCMAKE_BUILD_TYPE=Release \
#                  -DCMAKE_C_COMPILER=${CC} \
#                  -DCMAKE_CXX_COMPILER=${CXX} \
#                  -DADIOS2_USE_MPI=ON \
#                  -DADIOS2_USE_Blosc2=OFF \
#                  -DADIOS2_USE_BZip2=OFF \
#                  -DADIOS2_USE_ZeroMQ=OFF \
#                  -DADIOS2_USE_Endian_Reverse=OFF \
#                  -DADIOS2_USE_Fortran=OFF \
#                  -DADIOS2_USE_Python=ON \
#                  -DPYTHON_EXECUTABLE=${PYTHON_EXE} \
#                  -DADIOS2_USE_HDF5=OFF \
#                  -DADIOS2_USE_MPI=ON \
#                  -DADIOS2_USE_PNG=OFF \
#                  -DBUILD_SHARED_LIBS=ON \
#                  -DADIOS2_USE_SZ=OFF \
#                  -DADIOS2_USE_ZFP=OFF
                 
       
#        make -j 16 |& tee my_adios2_build
#        make install |& tee my_adios2_install
# fi


cd ${BASE_DIR}

if [ ! -d "mfem" ]; then
    git clone https://github.com/rcarson3/mfem.git
    cd ${BASE_DIR}/mfem/
    git checkout exaconstit-dev
fi

cd ${BASE_DIR}

if [ ! -d "${BASE_DIR}/mfem/build_hip" ]; then
  mkdir ${BASE_DIR}/mfem/build_hip
  cd ${BASE_DIR}/mfem/build_hip
  LOCAL_CMAKE_MFEM="$(which cmake)"
  echo "NOTE: MFEM: cmake = $LOCAL_CMAKE_MFEM"
    #All the options
  cmake ../ -DMFEM_USE_MPI=YES -DMFEM_USE_SIMD=NO\
            -DMETIS_DIR=${METIS_ROOT} \
            -DHYPRE_DIR=${HYPRE_ROOT} \
            -DMFEM_USE_RAJA=YES \
            -DRAJA_DIR:PATH=${RAJA_ROOT} \
            -DRAJA_REQUIRED_PACKAGES="camp" \
            -DMFEM_USE_CAMP=ON \
            -Dcamp_DIR:PATH=${CAMP_ROOT}/lib/cmake/camp/ \
            -DMFEM_USE_OPENMP=${OPENMP_ON} \
            -DMFEM_USE_ZLIB=YES \
            -DCMAKE_CXX_COMPILER=${HIPCC} \
            -DMPI_CXX_COMPILER=${MPICXX} \
            -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
            -DCMAKE_INSTALL_PREFIX=../install_dir_hip/ \
            -DCMAKE_CXX_STANDARD=17 \
            -DMFEM_USE_HIP=${ROCMON} \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
            -DHIP_ARCH=${LOC_ROCM_ARCH} \
            -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
            -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
            -DHIP_CXX_COMPILER=${HIPCC} \
            |& tee my_mfem_config
          #   -DMFEM_USE_MAGMA=ON \
          #   -DMAGMA_DIR=${BASE_DIR}/magma/install_dir/ \
          #   -DMFEM_USE_ADIOS2=ON \
          #   -DADIOS2_DIR=${BASE_DIR}/ADIOS2/install_dir_hip/ \

  make -j 16 |& tee my_mfem_build
  make install |& tee my_mfem_install
fi

cd ${BASE_DIR}

# : << 'END_COMMENT'
if [ ! -d "ExaConstit" ]; then
    git clone https://github.com/llnl/ExaConstit.git
    cd ${BASE_DIR}/ExaConstit/
    git checkout insitu_lightup
    git submodule init && git submodule update
fi
cd ${BASE_DIR}
if [ ! -d "${BASE_DIR}/ExaConstit/build_hip" ]; then
    cd ${BASE_DIR}/ExaConstit/
    mkdir build_hip

    cd ${BASE_DIR}/ExaConstit/build_hip #&& rm -rf *
    LOCAL_CMAKE_MFEM="$(which cmake)"
    echo "NOTE: ExaConstit: cmake = $LOCAL_CMAKE_MFEM"

    cmake ../ -DCMAKE_C_COMPILER=${CC} \
              -DCMAKE_CXX_COMPILER=${HIPCC} \
              -DMPI_CXX_COMPILER=${MPICXX} \
              -DHIP_CXX_COMPILER=${HIPCC} \
              -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
              -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINK_FLAGS}" \
              -DENABLE_TESTS=ON \
              -DENABLE_OPENMP=OFF \
              -DENABLE_FORTRAN=OFF \
              -DENABLE_HIP=${ROCMON} \
              -DENABLE_SNLS_V03=ON \
              -DCMAKE_INSTALL_PREFIX=../install_dir/ \
              -DRAJA_DIR:PATH=${RAJA_ROOT}/lib/cmake/raja/ \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_HIP_ARCHITECTURES=${LOC_ROCM_ARCH} \
              -DGPU_TARGETS=${LOCM_ROCM_ARCH} \
              -DAMDGPU_TARGETS=${LOCM_ROCM_ARCH} \
              -DMFEM_DIR=${BASE_DIR}/mfem/install_dir_hip/lib/cmake/mfem/ \
              -DECMECH_DIR=${BASE_DIR}/ExaCMech/install_dir_hip/ \
              -DSNLS_DIR=${BASE_DIR}/ExaCMech/install_dir_hip/ \
              -DFMT_DIR=${UMPIRE_ROOT}/lib64/cmake/fmt \
              -DUMPIRE_DIR=${UMPIRE_ROOT}/lib64/cmake/umpire \
              -DRAJA_DIR=${RAJA_ROOT}/lib/cmake/raja \
              -DCHAI_DIR=${CHAI_ROOT}/lib/cmake/chai \
              -DCAMP_DIR=${CAMP_ROOT}/lib/cmake/camp |& tee my_exconstit_config

    make -j 4|& tee my_exconstit_build
fi
###END_COMMENT
