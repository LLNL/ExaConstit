#!/bin/bash                                                                                                                                                                         
# This script should download exacmech and try to build itself on a spock-like machine                                                                                              
# For ease all of this should be run in its own directory                                                                                                                           
SCRIPT=$(readlink -f "$0")
BASE_DIR=$(dirname "$SCRIPT")
# Load up the needed modules                                                                                                                                                        
# We could build RAJA on our own but the one on the system works just fine for our needs 
#module load PrgEnv-cray/8.3.3 craype-accel-amd-gfx90a rocm/5.3.0 cmake/3.23.2 cray-mpich/8.1.19 cray-python/3.9.4.2

### as of 02/27/2023, the Frontier defaults include:
# cce/15.0.0
# PrgEnv-cray/8.3.3
# cray-mpich/8.1.23
### keeping the defaults, let's load what we need ... using what's available on Frontier ...
module load rocm/5.4.3 craype-accel-amd-gfx90a cmake/3.23.2 cray-python/3.9.13.1 openblas/0.3.17

module list |& tee my_loaded_modules
#exit

ROCM_VER=rocm-5.4.3

# If RAJA hasn't been installed set this to false if not set to true
RAJA_SYS_INSTALLED=false

if [ ! -d "camp" ]; then
    git clone https://github.com/LLNL/camp.git -b v2022.10.1
    cd ${BASE_DIR}/camp
    git submodule init
    git submodule update

    if [ ! -d "build" ]; then
      mkdir build
      cd ${BASE_DIR}/camp/build
      rm -rf *
      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF \
                -DENABLE_HIP=ON \
                -DRAJA_TIMER=chrono \
                -DENABLE_OPENMP=OFF \
                -DROCM_PATH=/opt/${ROCM_VER}/ \
                -DHIP_ROOT_DIR=/opt/${ROCM_VER}/hip \
                -DGPU_TARGETS=gfx90a \
                -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
                -DHIP_ARCH=gfx90a \
                -DBLT_ROCM_ARCH="gfx90a" \
                -DCMAKE_C_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                -DCMAKE_CXX_FLAGS='-fPIC --amdgpu-target=gfx90a -std=c++14 -munsafe-fp-atomics' \
                -DENABLE_CUDA=OFF |& tee my_camp_config
      make -j 4 |& tee my_camp_build
      make install |& tee my_camp_install
    fi
    OLCF_CAMP_ROOT=${BASE_DIR}/camp/install_dir/
else
    OLCF_CAMP_ROOT=${BASE_DIR}/camp/install_dir/
fi

cd ${BASE_DIR}

#exit

if ${RAJA_SYS_INSTALLED}
then
   module load raja/0.14.0
else
   cd ${BASE_DIR}
   if [ ! -d "RAJA" ]; then 
      git clone https://github.com/LLNL/RAJA.git -b v2022.10.3
      cd ${BASE_DIR}/RAJA
      git submodule init
      git submodule update
   fi
   cd ${BASE_DIR}/RAJA
   if [ ! -d "build" ]; then
      mkdir build
      cd ${BASE_DIR}/RAJA/build
      rm -rf *
      cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF \
                -DRAJA_ENABLE_TESTS=OFF \
                -DRAJA_ENABLE_EXAMPLES=OFF \
                -DRAJA_ENABLE_BENCHMARKS=OFF \
                -DENABLE_HIP=ON \
                -DRAJA_TIMER=chrono \
                -DENABLE_OPENMP=OFF \
                -DROCM_PATH=/opt/${ROCM_VER}/ \
                -DHIP_ROOT_DIR=/opt/${ROCM_VER}/hip \
                -DGPU_TARGETS=gfx90a \
                -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
                -DHIP_ARCH=gfx90a \
                -DBLT_ROCM_ARCH="gfx90a" \
                -DCMAKE_C_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                -DCMAKE_CXX_FLAGS='-fPIC --amdgpu-target=gfx90a -std=c++14 -munsafe-fp-atomics' \
                -DENABLE_CUDA=OFF \
                -Dcamp_DIR=${OLCF_CAMP_ROOT} |& tee my_raja_config
      make -j 4 |& tee my_raja_build
      make install |& tee my_raja_install
   fi
   OLCF_RAJA_ROOT=${BASE_DIR}/RAJA/install_dir/
   #share/raja/cmake/
   cd ${BASE_DIR}
fi

echo ${OLCF_RAJA_ROOT}

cd ${BASE_DIR}

#exit

if [ ! -d "ExaCMech" ]; then
# Clone the repo
    git clone https://github.com/LLNL/ExaCMech.git
    cd ${BASE_DIR}/ExaCMech
# Checkout the branch that has the HIP features on it
    git checkout feature/carson16/hip
# Update all the various submodules                  
    git submodule init && git submodule update
    if [ ! -d "${BASE_DIR}/ExaCMech/build" ]; then
       mkdir build
       cd ${BASE_DIR}/ExaCMech/build
       rm -rf *       

       cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
                 -DCMAKE_BUILD_TYPE=Release \
                 -DENABLE_TESTS=OFF \
                 -DENABLE_MINIAPPS=OFF \
                 -DENABLE_HIP=ON \
                 -DENABLE_OPENMP=OFF \
                 -DHIP_ROOT_DIR=/opt/${ROCM_VER}/hip \
                 -DRAJA_DIR=${OLCF_RAJA_ROOT}/lib/cmake/raja/ \
                 -DBUILD_SHARED_LIBS=OFF \
                 -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                 -DENABLE_CUDA=OFF \
                 -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
                 -DCMAKE_CXX_FLAGS='-fPIC --amdgpu-target=gfx90a -std=c++14 -munsafe-fp-atomics' \
                 -Dcamp_DIR=${OLCF_CAMP_ROOT}/lib/cmake/camp |& tee my_exacmech_config
       
       make -j 4 |& tee my_exacmech_build
       make install |& tee my_exacmech_install
    fi
fi
cd ${BASE_DIR}

#exit

# Now to build our MFEM dependencies
# First let's install Hypre v2.23.0
cd ${BASE_DIR}
if [ ! -d "hypre" ]; then

  git clone https://github.com/hypre-space/hypre.git --branch v2.26.0 --single-branch
  cd ${BASE_DIR}/hypre/
  mkdir build
  cd ${BASE_DIR}/hypre/build
  # Based on their install instructions
  # This should work on most systems
  # Hypre's default suggestions of just using configure don't always work
  cmake ../src  -DCMAKE_INSTALL_PREFIX=../src/hypre/ \
                -DWITH_MPI=TRUE \
                -DWITH_ROCTX=TRUE \
                -DWITH_ROCTRACER=TRUE \
                -DROCM_PATH=/opt/${ROCM_VER}/ \
                -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
                -DCMAKE_C_COMPILER=/opt/${ROCM_VER}/bin/amdclang \
                -DCMAKE_Fortran_COMPILER=/opt/${ROCM_VER}/bin/amdflang \
                -DMPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicc \
                -DMPI_Fortran_COMPILER=${MPICH_DIR}/bin/mpifort \
                -DMPI_INCLUDE_DIR=${MPICH_DIR}/include \
                -DMPI_libmpi_LIBRARY="${MPICH_DIR}/lib/libmpi" \
                -DMPI_libmpi_gtl_hsa_LIBRARY="${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa" \
                -DMPI_CXX_COMPILER_FLAGS=" -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa" \
                |& tee my_hypre_config
  
  make -j 4 |& tee my_hypre_build
  make install |& tee my_hypre_install

  cd ${BASE_DIR}/hypre/src/hypre
  OLCF_HYPRE_ROOT="$(pwd)"

else

  echo " hypre already built "
  OLCF_HYPRE_ROOT=${BASE_DIR}/hypre/src/hypre

fi

# Now to install metis-5.1.0
# It appears that there are some minor differences in performance between metis-4 and metis-5
# If you'd like to install metis-4 instead here's the commands needed
# uncomment the below and then comment the metis-5 commands
# cd ${BASE_DIR}
# curl -o metis-4.0.3.tar.gz http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz
# tar -xzf metis-4.0.3.tar.gz
# rm metis-4.0.3.tar.gz
# cd metis-4.0.3
# make
# METIS_DIR="$(pwd)"
# metis-5 install down below
cd ${BASE_DIR}

#exit

if [ ! -d "metis-5.1.0" ]; then
  curl -o metis-5.1.0.tar.gz https://mfem.github.io/tpls/metis-5.1.0.tar.gz
  tar -xzf metis-5.1.0.tar.gz
  rm metis-5.1.0.tar.gz
  cd metis-5.1.0
  mkdir install_dir
  make config prefix=${BASE_DIR}/metis-5.1.0/install_dir/ CC=/opt/${ROCM_VER}/bin/amdclang CXX=/opt/${ROCM_VER}/bin/amdclang++ |& tee my_metis_config
  make -j 4 |& tee my_metis_build
  make install |& tee my_metis_install
  cd ${BASE_DIR}/metis-5.1.0/install_dir/
  OLCF_METIS_ROOT="$(pwd)"
else

  echo " metis-5.1.0 already built "
  OLCF_METIS_ROOT=${BASE_DIR}/metis-5.1.0/install_dir/

fi

cd ${BASE_DIR}

#exit

if [ ! -d "Caliper" ]; then
    git clone https://github.com/LLNL/Caliper.git
    if [ ! -d "${BASE_DIR}/Caliper/build/" ]; then
       cd ${BASE_DIR}/Caliper/
       mkdir build/
       cd ${BASE_DIR}/Caliper/build/
       rm -rf *
       cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
                 -DWITH_MPI=TRUE \
                 -DWITH_ROCTX=TRUE \
                 -DWITH_ROCTRACER=TRUE \
                 -DROCM_PATH=/opt/${ROCM_VER}/ \
                 -DMPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicc \
                 -DMPI_Fortran_COMPILER=${MPICH_DIR}/bin/mpifort \
                 -DMPI_INCLUDE_DIR=${MPICH_DIR}/include \
                 -DMPI_libmpi_LIBRARY="${MPICH_DIR}/lib/libmpi" \
                 -DMPI_libmpi_gtl_hsa_LIBRARY="${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa" \
                 -DMPI_CXX_COMPILER_FLAGS=" -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -munsafe-fp-atomics" |& tee my_caliper_config

       make -j 4 |& tee my_caliper_build
       make install |& tee my_caliper_install
    fi
fi

cd ${BASE_DIR}

#exit

if [ ! -d "magma" ]; then
    git clone https://bitbucket.org/icl/magma.git
    cd ${BASE_DIR}/magma/
    ln -s ${BASE_DIR}/make.inc ${BASE_DIR}/magma/make.inc
    make -j 8 |& tee my_magma_build
    mkdir install_dir
    make install prefix=${BASE_DIR}/magma/install_dir/ |& tee my_magma_install
fi

cd ${BASE_DIR}

#exit


if [ ! -d "mfem" ]; then
    git clone https://github.com/rcarson3/mfem.git
    cd ${BASE_DIR}/mfem/
    git checkout exaconstit-hip-batches
    if [ ! -d "build" ]; then
       mkdir build
    fi
    cd ${BASE_DIR}/mfem/build
    rm -rf *
    LOCAL_CMAKE_MFEM="$(which cmake)"
    echo "NOTE: MFEM: cmake = $LOCAL_CMAKE_MFEM"
      #All the options
    cmake ../ -DMFEM_USE_MPI=YES -DMFEM_USE_SIMD=NO\
              -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
              -DCMAKE_CXX_FLAGS='-fPIC --amdgpu-target=gfx90a -std=c++14 -munsafe-fp-atomics' \
              -DMETIS_DIR=${OLCF_METIS_ROOT} \
              -DHYPRE_DIR=${OLCF_HYPRE_ROOT} \
              -DCMAKE_INSTALL_PREFIX=../install_dir/ \
              -DMFEM_USE_OPENMP=NO \
              -DMFEM_USE_RAJA=YES \
              -DRAJA_DIR:PATH=${OLCF_RAJA_ROOT} \
              -DMFEM_USE_ZLIB=YES \
              -DMFEM_USE_HIP=YES \
              -DENABLE_HIP=ON \
              -DROCM_PATH=/opt/${ROCM_VER}/ \
              -DHIP_ROOT_DIR=/opt/${ROCM_VER}/hip \
              -DGPU_TARGETS=gfx90a \
              -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
              -DHIP_ARCH=gfx90a \
              -DCMAKE_BUILD_TYPE=Release \
              -DMPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicc \
              -DMPI_INCLUDE_DIR=${MPICH_DIR}/include \
              -DMPI_libmpi_LIBRARY="${MPICH_DIR}/lib/libmpi" \
              -DMPI_libmpi_gtl_hsa_LIBRARY="${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa" \
              -DMPI_CXX_COMPILER_FLAGS=" -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa" \
              -DRAJA_REQUIRED_PACKAGES="camp" \
              -DMFEM_USE_CAMP=ON \
              -Dcamp_DIR:PATH=${OLCF_CAMP_ROOT}/lib/cmake/camp/ \
              -DCMAKE_CXX_STANDARD=14 \
              -DMFEM_USE_MAGMA=ON \
              -DMAGMA_DIR=${BASE_DIR}/magma/install_dir/ |& tee my_mfem_config
#              -DMFEM_USE_CALIPER=YES \
#              -DCALIPER_DIR=${BASE_DIR}/Caliper/install_dir/ \

    
    make -j 8 |& tee my_mfem_build
    make install |& tee my_mfem_install
fi

cd ${BASE_DIR}

#exit

# : <<'END'

if [ ! -d "ExaConstit" ]; then
    git clone https://github.com/llnl/ExaConstit.git
    cd ${BASE_DIR}/ExaConstit/
    git checkout exaconstit-hip
    git submodule init && git submodule update

cd ${BASE_DIR}/ExaConstit/
if [ ! -d "build" ]; then
    mkdir build
fi
fi
cd ${BASE_DIR}/ExaConstit/build && rm -rf *
LOCAL_CMAKE_MFEM="$(which cmake)"
echo "NOTE: ExaConstit: cmake = $LOCAL_CMAKE_MFEM"

cmake ../ -DCMAKE_CXX_COMPILER=/opt/${ROCM_VER}/bin/hipcc \
          -DCMAKE_C_COMPILER=/opt/${ROCM_VER}/bin/amdclang \
          -DCMAKE_Fortran_COMPILER=/opt/${ROCM_VER}/bin/amdflang \
          -DENABLE_HIP=ON \
          -DENABLE_TESTS=ON \
          -DROCM_PATH=/opt/${ROCM_VER}/ \
          -DHIP_ROOT_DIR=/opt/${ROCM_VER}/hip \
          -DGPU_TARGETS=gfx90a \
          -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
          -DHIP_ARCH=gfx90a \
          -DCMAKE_CXX_FLAGS='-fPIC --amdgpu-target=gfx90a -std=c++14 -munsafe-fp-atomics' \
          -DMFEM_DIR=${BASE_DIR}/mfem/install_dir/lib/cmake/mfem/ \
          -DECMECH_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
          -DSNLS_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
          -DENABLE_SNLS_V03=ON \
          -DCMAKE_INSTALL_PREFIX=../install_dir/ \
          -DRAJA_DIR:PATH=${OLCF_RAJA_ROOT}/lib/cmake/raja/ \
          -DCMAKE_BUILD_TYPE=Release \
          -DMPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicc \
          -DMPI_Fortran_COMPILER=${MPICH_DIR}/bin/mpifort \
          -DMPI_INCLUDE_DIR=${MPICH_DIR}/include \
          -DMPI_libmpi_LIBRARY="${MPICH_DIR}/lib/libmpi" \
          -DMPI_libmpi_gtl_hsa_LIBRARY="${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa" \
          -DMPI_CXX_COMPILER_FLAGS=" -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa" \
          -Dcamp_DIR=${OLCF_CAMP_ROOT}/lib/cmake/camp |& tee my_exconstit_config
#          -DCALIPER_DIR=${BASE_DIR}/Caliper/install_dir/share/cmake/caliper/
make -j 4|& tee my_exconstit_build

#fi

#ln -s ${BASE_DIR}/ExaConstit/build/bin/mechanics ${BASE_DIR}/test_dir/mechanics
