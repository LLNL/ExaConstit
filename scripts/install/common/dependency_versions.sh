#!/usr/bin/env bash
# Central version control for all dependencies

###########################################
# Build infrastructure
###########################################
# BLT lifted out so all RADIUSS-stack packages share a single BLT and stay in sync.
# Each package below is pointed at this via -DBLT_SOURCE_DIR=${BLT_ROOT}.
export BLT_REPO="https://github.com/LLNL/blt.git"
export BLT_VER="v0.7.2"

###########################################
# Portability libraries (RAJA Portability Suite)
###########################################
# Note: the next coordinated RADIUSS release will be v2025.12.x; bump
# all four together when that lands.
export CAMP_VER="v2025.12.0"
export RAJA_VER="v2025.12.2"
export UMPIRE_VER="v2025.12.0"
export CHAI_VER="v2025.12.0"

###########################################
# SNLS (lifted out of ExaCMech so it can be built standalone with the
# RAJA Portability Suite and the batch-solver option always enabled)
###########################################
export SNLS_REPO="https://github.com/LLNL/SNLS.git"
export SNLS_VER="v0.4.4"

###########################################
# Axom (HPC utility library suite)
###########################################
# For now we build with core + spin only. When we add Sidre we'll also need
# Conduit and HDF5 in the dependency graph (and AXOM_ENABLE_SIDRE=ON,
# CONDUIT_DIR=..., HDF5_DIR=... in build_axom). Axom will eventually consume
# MFEM as well, which is why build_axom lives in the application-stack
# build file (build_functions_exaconstit.sh) rather than the common stack.
export AXOM_REPO="https://github.com/LLNL/axom.git"
export AXOM_VER="v0.14.0"

###########################################
# Material models
###########################################
export EXACMECH_REPO="https://github.com/LLNL/ExaCMech.git"
export EXACMECH_BRANCH="develop"

###########################################
# FEM infrastructure
###########################################
export HYPRE_VER="v3.1.0"
export METIS_VER="5.1.0"
export METIS_URL="https://mfem.github.io/tpls/metis-${METIS_VER}.tar.gz"

export MFEM_REPO="https://github.com/rcarson3/mfem.git"
export MFEM_BRANCH="exaconstit-dev"

###########################################
# Main application
###########################################
export EXACONSTIT_REPO="https://github.com/llnl/ExaConstit.git"
export EXACONSTIT_BRANCH="exaconstit-dev"

###########################################
# Build standards
###########################################
export CMAKE_CXX_STANDARD="17"
export CMAKE_BUILD_TYPE="Debug"
