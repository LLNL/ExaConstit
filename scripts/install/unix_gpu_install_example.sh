#!/bin/bash 
# The below is a bash script that should work on most UNIX systems to download all of ExaConstit and its dependencies
# and then install them.
# 
# For ease all of this should be run in its own directory
SCRIPT=$(readlink -f "$0")
BASE_DIR=$(dirname "$SCRIPT")
#change this to the cuda compute capability for your gpu
# LOC_CUDA_ARCH='sm_70'
#CMAKE_CUDA_ARCHITECTURES drops the sm_ aspect of the cuda compute capability
LOC_CUDA_ARCH='70'

# If you are using SPACK or have another module like system to set-up your developer environment
# you'll want to load up the necessary compilers and devs environments
# In other words make sure what ever MPI you want is loaded, C++, C, and Fortran compilers are loaded, and
# a cmake version b/t 3.12 and 3.18. 

# Build raja
if [ ! -d "raja" ]; then
  git clone --recursive https://github.com/llnl/raja.git --branch v2022.10.5 --single-branch
  cd ${BASE_DIR}/raja
  # Instantiate all the submodules
  git submodule init
  git submodule update
  # Build everything
  mkdir build
  cd ${BASE_DIR}/raja/build/
  # GPU build
  cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
            -DENABLE_OPENMP=OFF \
            -DENABLE_CUDA=ON \
            -DRAJA_TIMER=chrono \
            -DCMAKE_CUDA_ARCHITECTURESmbly=${LOC_CUDA_ARCH} \
            -DENABLE_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release
  make -j 4
  make install
else

  echo " RAJA already built "

fi

# Now to build ExaCMech
cd ${BASE_DIR}

if [ ! -d "ExaCMech" ]; then

  git clone https://github.com/LLNL/ExaCMech.git --single-branch
  cd ${BASE_DIR}/ExaCMech
  # Instantiate all the submodules
  git submodule init
  git submodule update
  # Build everything
  mkdir build
  cd ${BASE_DIR}/ExaCMech/build
  # GPU build
  cmake ../ -DCMAKE_INSTALL_PREFIX=../install_dir/ \
            -DRAJA_DIR=${BASE_DIR}/raja/install_dir/lib/cmake/raja/ \
            -DENABLE_OPENMP=OFF \
            -DENABLE_CUDA=ON \
            -DENABLE_TESTS=OFF \
            -DENABLE_MINIAPPS=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURESmbly=${LOC_CUDA_ARCH} \
            -DBUILD_SHARED_LIBS=OFF
  make -j 4
  make install
else

  echo " ExaCMech already built "

fi

# Now to build our MFEM dependencies
# First let's install Hypre v2.20.0
cd ${BASE_DIR}
if [ ! -d "hypre" ]; then

  git clone https://github.com/hypre-space/hypre.git --branch v2.26.0 --single-branch
  cd ${BASE_DIR}/hypre/src
  # Based on their install instructions
  # This should work on most systems
  # Hypre's default suggestions of just using configure don't always work
  ./configure CC=mpicc CXX=mpicxx FC=mpif90
  make -j 4
  make install
  cd hypre
  HYPRE_DIR="$(pwd)"

else

  echo " hypre already built "
  HYPRE_DIR=${BASE_DIR}/hypre/src/hypre

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

if [ ! -d "metis-5.1.0" ]; then

  curl -o metis-5.1.0.tar.gz https://mfem.github.io/tpls/metis-5.1.0.tar.gz  tar -xzf metis-5.1.0.tar.gz
  rm metis-5.1.0.tar.gz
  cd metis-5.1.0
  mkdir install_dir
  make config prefix=${BASE_DIR}/metis-5.1.0/install_dir/
  make -j 4
  make install
  cd ${BASE_DIR}/metis-5.1.0/install_dir/
  METIS_DIR="$(pwd)"
else

  echo " metis-5.1.0 already built "
  METIS_DIR=${BASE_DIR}/metis-5.1.0/install_dir/

fi

# If you want anyother MFEM options installed like Conduit, ADIOS2, or etc. install them now
# We can now install MFEM with relevant data for ExaConstit

cd ${BASE_DIR}

if [ ! -d "mfem" ]; then

  git clone https://github.com/rcarson3/mfem.git --branch exaconstit-dev --single-branch
  cd ${BASE_DIR}/mfem/
  mkdir build
  cd ${BASE_DIR}/mfem/build/
  # All the options
  cmake ../ -DMFEM_USE_MPI=ON -DMFEM_USE_SIMD=OFF\
            -DMETIS_DIR=${METIS_DIR} \
            -DHYPRE_DIR=${HYPRE_DIR} \
            -DCMAKE_INSTALL_PREFIX=../install_dir/ \
            -DMFEM_USE_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURESmbly=${LOC_CUDA_ARCH} \
            -DMFEM_USE_OPENMP=OFF \
            -DMFEM_USE_RAJA=ON -DRAJA_DIR=${BASE_DIR}/raja/install_dir/ \
            -DCMAKE_BUILD_TYPE=Release
  # The below are the relevant lines needed for ADIOS2 and conduit. You'll want to put them
  # before the -DCMAKE_BUILD_TYPE call 
  #  -DMFEM_USE_ADIOS2=ON -DADIOS2_DIR=${ADIOS2_DIR} \
  #  -DMFEM_USE_CONDUIT=ON -DConduit_REQUIRED_PACKAGES=HDF5 -DCONDUIT_DIR=${CONDUIT_DIR} \
  #  -DHDF5_ROOT:PATH=${HDF5_DIR} \
  make -j 4
  make install

else

  echo " MFEM already built "

fi

#We can finally install ExaConstit
cd ${BASE_DIR}

if [ ! -d "ExaConstit" ]; then

  git clone https://github.com/LLNL/ExaConstit.git
  cd ${BASE_DIR}/ExaConstit/
  # Instantiate all the submodules
  git submodule init
  git submodule update
  # Build everything
  mkdir build
  cd ${BASE_DIR}/ExaConstit/build/

  cmake ../ -DENABLE_MPI=ON -DENABLE_FORTRAN=ON \
            -DMFEM_DIR=${BASE_DIR}/mfem/install_dir/lib/cmake/mfem/ \
            -DECMECH_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
            -DRAJA_DIR=${BASE_DIR}/raja/install_dir/lib/cmake/raja/ \
            -DSNLS_DIR=${BASE_DIR}/ExaCMech/install_dir/ \
            -DENABLE_SNLS_V03=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURESmbly=${LOC_CUDA_ARCH} \
            -DENABLE_TESTS=ON
  # Sometimes the cmake systems can be a bit difficult and not properly find the MFEM installed location
  # using the above. If that's the case the below should work:
  #  -DMFEM_DIR=${BASE_DIR}/mfem/install_dir/ \

  make -j 4
  # Check and make sure everything installed correctly by running the test suite
  make test

else

  echo " ExaConstit already built "

fi

# ExaConstit is now installed
