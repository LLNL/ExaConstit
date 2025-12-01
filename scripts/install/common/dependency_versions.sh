#!/usr/bin/env bash
# Central version control for all dependencies

# Portability libraries
export CAMP_VER="v2025.09.2"
export RAJA_VER="v2025.09.1"
#export UMPIRE_VER="v2025.09.0"
# For now we need something a little pass the v2025.09.0 release
# for Umpire as we need a small bug fix for any build with Umpire
export UMPIRE_VER="54a1909e91ce9604328977974e9b1002bf9f8781"
export CHAI_VER="v2025.09.1"

# Material models
export EXACMECH_REPO="https://github.com/LLNL/ExaCMech.git"
export EXACMECH_BRANCH="develop"

# FEM infrastructure
export HYPRE_VER="v2.32.0"
export METIS_VER="5.1.0"
export METIS_URL="https://mfem.github.io/tpls/metis-${METIS_VER}.tar.gz"

export MFEM_REPO="https://github.com/rcarson3/mfem.git"
export MFEM_BRANCH="exaconstit-dev"

# Main application
export EXACONSTIT_REPO="https://github.com/llnl/ExaConstit.git"
export EXACONSTIT_BRANCH="exaconstit-dev"

# Build standards
export CMAKE_CXX_STANDARD="17"
export CMAKE_BUILD_TYPE="Release"