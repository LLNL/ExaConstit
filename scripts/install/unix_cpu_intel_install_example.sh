#!/usr/bin/bash 
# For ease all of this should be run in its own directory
# Build and run this in $SCRATCH/csm3_builds/

SCRIPT=$(readlink -f "$0")
BASE_DIR=$(dirname "$SCRIPT")

# On macs the above two lines won't work but can be replaced with this line
# BASE_DIR=$(cd "$(dirname "$0")"; pwd -P)

module load intel/2023.2.1-magic
module load CMake/3.26.3
module list 

CC="/usr/tce/packages/intel/intel-2023.2.1-magic/bin/icx"
CXX="/usr/tce/packages/intel/intel-2023.2.1-magic/bin/icpx"
MPICXX="/usr/tce/packages/mvapich2/mvapich2-2.3.7-intel-2023.2.1-magic/bin/mpicxx"
MPICC="/usr/tce/packages/mvapich2/mvapich2-2.3.7-intel-2023.2.1-magic/bin/mpicc"

#Build raja
if [ ! -d "camp" ]; then
    git clone https://github.com/LLNL/camp.git -b v2024.07.0
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
                -DRAJA_TIMER=chrono \
                -DENABLE_OPENMP=OFF \
                -DCMAKE_C_COMPILER=${CC} \
                -DCMAKE_CXX_COMPILER=${CXX} \
                -DENABLE_CUDA=OFF |& tee my_camp_config
      make -j 2 |& tee my_camp_build
      make install |& tee my_camp_install
    fi
fi

OLCF_CAMP_ROOT=${BASE_DIR}/camp/install_dir/

cd ${BASE_DIR}

#exit
if [ ! -d "RAJA" ]; then 
   git clone https://github.com/LLNL/RAJA.git -b v2024.07.0
   cd ${BASE_DIR}/RAJA
   git submodule init
   git submodule update
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
                  -DRAJA_TIMER=chrono \
                  -DENABLE_OPENMP=OFF \
                  -DCMAKE_C_COMPILER=${CC} \
                  -DCMAKE_CXX_COMPILER=${CXX} \
                  -DENABLE_CUDA=OFF \
                  -Dcamp_DIR=${OLCF_CAMP_ROOT} |& tee my_raja_config
      make -j 4 |& tee my_raja_build
      make install |& tee my_raja_install
   fi
fi

OLCF_RAJA_ROOT=${BASE_DIR}/RAJA/install_dir/

echo ${OLCF_RAJA_ROOT}

cd ${BASE_DIR}
if [ ! -d "ExaCMech" ]; then
      # Clone the repo
    git clone https://github.com/LLNL/ExaCMech.git
    cd ${BASE_DIR}/ExaCMech
   # Checkout the branch that has the HIP features on it
    git checkout develop
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
                 -DENABLE_OPENMP=OFF \
                 -DRAJA_DIR=${OLCF_RAJA_ROOT}/lib/cmake/raja/ \
                 -DBUILD_SHARED_LIBS=OFF \
                 -DCMAKE_C_COMPILER=${CC} \
                 -DCMAKE_CXX_COMPILER=${CXX} \
                 -DENABLE_CUDA=OFF \
                 -Dcamp_DIR=${OLCF_CAMP_ROOT}/lib/cmake/camp |& tee my_exacmech_config
       
       make -j 4 |& tee my_exacmech_build
       make install |& tee my_exacmech_install
    fi
fi
cd ${BASE_DIR}

# Now to build our MFEM dependencies
# First let's install Hypre v2.23.0
cd ${BASE_DIR}
if [ ! -d "hypre" ]; then

  git clone https://github.com/hypre-space/hypre.git --branch v2.30.0 --single-branch
  cd ${BASE_DIR}/hypre/
  mkdir build
  cd ${BASE_DIR}/hypre/build
  rm -rf *
  # Based on their install instructions
  # This should work on most systems
  # Hypre's default suggestions of just using configure don't always work
  cmake ../src  -DCMAKE_INSTALL_PREFIX=../src/hypre/ \
                -DWITH_MPI=TRUE \
                -DCMAKE_C_COMPILER=${MPICC} \
                -DCMAKE_CXX_COMPILER=${MPICXX} \
                -DCMAKE_Fortran_COMPILER=${MPIFORT} \
		-DCMAKE_BUILD_TYPE=Release \
                |& tee my_hypre_config
  
  make -j 4 |& tee my_hypre_build
  make install |& tee my_hypre_install

  cd ${BASE_DIR}/hypre/src/hypre
  OLCF_HYPRE_ROOT="$(pwd)"

else

  echo " hypre already built "
  OLCF_HYPRE_ROOT=${BASE_DIR}/hypre/src/hypre

fi

cd ${BASE_DIR}

if [ ! -d "metis-5.1.0" ]; then
  
  curl -o metis-5.1.0.tar.gz https://mfem.github.io/tpls/metis-5.1.0.tar.gz
  tar -xzf metis-5.1.0.tar.gz
  rm metis-5.1.0.tar.gz
  cd metis-5.1.0
  mkdir install_dir
  make config prefix=${BASE_DIR}/metis-5.1.0/install_dir/ CC=${CC} CXX=${CXX} |& tee my_metis_config
  make -j 4 |& tee my_metis_build
  make install |& tee my_metis_install
  cd ${BASE_DIR}/metis-5.1.0/install_dir/
  OLCF_METIS_ROOT="$(pwd)"
else

  echo " metis-5.1.0 already built "
  OLCF_METIS_ROOT=${BASE_DIR}/metis-5.1.0/install_dir/

fi

cd ${BASE_DIR}
if [ ! -d "ADIOS2" ]; then
  # Clone the repo
  git clone https://github.com/ornladios/ADIOS2.git
  cd ${BASE_DIR}/ADIOS2
  # Checkout the branch that has the HIP features on it
  git checkout v2.10.1
  # Update all the various submodules                  
  git submodule init && git submodule update

  cd ${BASE_DIR}
  if [ ! -d "${BASE_DIR}/ADIOS2/build" ]; then
        cd ${BASE_DIR}/ADIOS2
        mkdir build
        cd ${BASE_DIR}/ADIOS2/build
        rm -rf *       

        cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
                  -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_C_COMPILER=${CC} \
                  -DCMAKE_CXX_COMPILER=${CXX} \
                  -DADIOS2_USE_MPI=ON \
                  -DADIOS2_USE_Blosc2=OFF \
                  -DADIOS2_USE_BZip2=OFF \
                  -DADIOS2_USE_ZeroMQ=OFF \
                  -DADIOS2_USE_Endian_Reverse=OFF \
                  -DADIOS2_USE_Fortran=OFF \
                  -DADIOS2_USE_Python=OFF \
                  -DADIOS2_USE_HDF5=OFF \
                  -DADIOS2_USE_MPI=ON \
                  -DADIOS2_USE_PNG=OFF \
                  -DBUILD_SHARED_LIBS=ON \
                  -DADIOS2_USE_SZ=OFF \
                  -DADIOS2_USE_ZFP=OFF
                  
        
        make -j 4 |& tee my_adios2_build
        make install |& tee my_adios2_install
  fi
fi

cd ${BASE_DIR}

if [ ! -d "mfem" ]; then
    git clone https://github.com/rcarson3/mfem.git
    cd ${BASE_DIR}/mfem/
    git checkout exaconstit-dev
    if [ ! -d "build" ]; then
       mkdir build
    fi
    cd ${BASE_DIR}/mfem/build
    LOCAL_CMAKE_MFEM="$(which cmake)"
    echo "NOTE: MFEM: cmake = $LOCAL_CMAKE_MFEM"
      #All the options
    cmake ../ -DMFEM_USE_MPI=YES -DMFEM_USE_SIMD=NO\
              -DCMAKE_CXX_COMPILER=${MPICXX} \
              -DMETIS_DIR=${OLCF_METIS_ROOT} \
              -DHYPRE_DIR=${OLCF_HYPRE_ROOT} \
              -DCMAKE_INSTALL_PREFIX=../install_dir/ \
              -DMFEM_USE_OPENMP=OFF \
              -DMFEM_USE_RAJA=YES \
              -DRAJA_DIR:PATH=${OLCF_RAJA_ROOT} \
              -DMFEM_USE_ZLIB=YES \
              -DMFEM_USE_ADIOS2=ON \
              -DADIOS2_DIR=${BASE_DIR}/ADIOS2/install_dir/ \
              -DCMAKE_BUILD_TYPE=Release \
              -DRAJA_REQUIRED_PACKAGES="camp" \
              -DMFEM_USE_CAMP=ON \
              -Dcamp_DIR:PATH=${OLCF_CAMP_ROOT}/lib/cmake/camp/ \
              -DCMAKE_CXX_STANDARD=14 \
              -DCMAKE_BUILD_TYPE=Release \
              |& tee my_mfem_config
  
    make -j 4 |& tee my_mfem_build
    make install |& tee my_mfem_install
fi

cd ${BASE_DIR}

if [ ! -d "ExaConstit" ]; then
  git clone https://github.com/llnl/ExaConstit.git
  cd ${BASE_DIR}/ExaConstit/
  git checkout insitu_lightup
  git submodule init && git submodule update

  cd ${BASE_DIR}/ExaConstit/
  if [ ! -d "build" ]; then
      mkdir build
  fi

  cd ${BASE_DIR}/ExaConstit/build && rm -rf *
  LOCAL_CMAKE_MFEM="$(which cmake)"
  echo "NOTE: ExaConstit: cmake = $LOCAL_CMAKE_MFEM"

  cmake ../ -DCMAKE_C_COMPILER=${MPICC} \
            -DCMAKE_CXX_COMPILER=${MPICXX} \
            -DENABLE_TESTS=ON \
            -DENABLE_OPENMP=OFF \
            -DENABLE_FORTRAN=OFF \
            -DMFEM_DIR=${BASE_DIR}/mfem/install_dir/lib/cmake/mfem/ \
            -DECMECH_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
            -DSNLS_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
            -DENABLE_SNLS_V03=ON \
            -DCMAKE_INSTALL_PREFIX=../install_dir/ \
            -DRAJA_DIR:PATH=${OLCF_RAJA_ROOT}/lib/cmake/raja/ \
            -DCMAKE_BUILD_TYPE=Release \
            -Dcamp_DIR=${OLCF_CAMP_ROOT}/lib/cmake/camp |& tee my_exconstit_config

  make -j 4|& tee my_exconstit_build

fi
