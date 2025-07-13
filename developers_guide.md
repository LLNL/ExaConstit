# ExaConstit Developer's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dependency Version Compatibility](#dependency-version-compatibility)
5. [Codebase Overview](#codebase-overview)
6. [Source Directory Structure](#source-directory-structure)
7. [Key Components](#key-components)
8. [Configuration System](#configuration-system)
9. [Advanced Solver Configuration](#advanced-solver-configuration)
10. [Building and Testing](#building-and-testing)
11. [Development Workflow](#development-workflow)
12. [UMAT Development Resources](#umat-development-resources)
13. [Contributing Guidelines](#contributing-guidelines)

## Introduction

ExaConstit is a high-performance, velocity-based, updated Lagrangian finite element code for nonlinear solid mechanics problems with a focus on micromechanics modeling. Built on the MFEM library, it specializes in crystal plasticity simulations and bulk constitutive property determination for polycrystalline materials.

**Key Features:**
- Velocity-based updated Lagrangian formulation
- Crystal plasticity and micromechanics modeling
- GPU acceleration with CUDA/HIP support
- MPI parallelization for HPC systems
- Integration with ExaCMech material library
- UMAT interface support
- Advanced post-processing capabilities

## Prerequisites

### Required Knowledge
- **C++17**: Modern C++ standards and best practices
- **Finite Element Method (FEM)**: Theory and implementation
- **Solid Mechanics**: Nonlinear mechanics, crystal plasticity
- **Numerical Methods**: Newton-Raphson, Krylov iterative solvers
- **Parallel Computing**: MPI, OpenMP, GPU programming concepts

### System Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, Intel 19+)
- MPI implementation (OpenMPI, MPICH, Intel MPI)
- CMake 3.12 or higher
- Git for version control

## Installation

### Quick Start
For detailed installation instructions, refer to the build scripts in `scripts/install/`:

- **Linux/Unix**: `scripts/install/unix_install_example.sh`
- **GPU (CUDA)**: `scripts/install/unix_gpu_cuda_install_example.sh`
- **GPU (HIP/AMD)**: `scripts/install/unix_gpu_hip_install_example.sh`

### Dependencies

**Core Dependencies:**
- **MFEM** (v4.7+): Finite element library with parallel/GPU support
- **ExaCMech**: Crystal plasticity constitutive model library
- **RAJA** (≥2024.04.x): Performance portability framework
- **UMPIRE** (≥2024.04.x): (GPU-only) Performance portability framework
- **CHAI** (≥2024.04.x): (GPU-only) Performance portability framework
- **BLT**: LLNL build system
- **SNLS**: Nonlinear solver library

**Optional Dependencies:**
- **ADIOS2**: (MFEM-based) High-performance I/O for visualization
- **Caliper**: Performance profiling

### Basic Build Process
```bash
# Clone dependencies
git clone https://github.com/LLNL/blt.git cmake/blt

# Create build directory
mkdir build && cd build

# Configure
cmake .. \
  -DENABLE_MPI=ON \
  -DENABLE_FORTRAN=OFF \
  -DMFEM_DIR=${MFEM_INSTALL_DIR} \
  -DECMECH_DIR=${EXACMECH_INSTALL_DIR} \
  -DRAJA_DIR=${RAJA_INSTALL_DIR} \
  -DSNLS_DIR=${SNLS_INSTALL_DIR}

# Build
make -j 4
```

## Dependency Version Compatibility

### **MFEM Requirements**
ExaConstit requires a specific MFEM development branch with ExaConstit-specific features:

#### **Current Requirements**
- **Repository**: https://github.com/rcarson3/mfem.git
- **Branch**: `exaconstit-dev`
- **Version Dependencies**:
  - **v0.8.0**: Compatible with MFEM hashes `31b42daa3cdddeff04ce3f59befa769b262facd7` or `29a8e15382682babe0f5c993211caa3008e1ec96`
  - **v0.7.0**: Compatible with MFEM hash `78a95570971c5278d6838461da6b66950baea641`
  - **v0.6.0**: Compatible with MFEM hash `1b31e07cbdc564442a18cfca2c8d5a4b037613f0`
  - **v0.5.0**: Required MFEM hash `5ebca1fc463484117c0070a530855f8cbc4d619e`

#### **MFEM Build Requirements**
```bash
# Required dependencies for MFEM
cmake .. \
  -DMFEM_USE_MPI=ON \
  -DMFEM_USE_METIS_5=ON \
  -DMFEM_USE_HYPRE=ON \    # v2.26.0-v2.30.0
  -DMFEM_USE_RAJA=ON \     # v2022.x+
  -DMFEM_USE_ADIOS2=ON \   # Optional: high-performance I/O
  -DMFEM_USE_ZLIB=ON       # Optional: compressed mesh support
```

**Note**: Future releases will integrate these changes into MFEM master branch, eliminating the need for the development fork.

### **ExaCMech Version Requirements**
- **Repository**: https://github.com/LLNL/ExaCMech.git
- **Branch**: `develop` (required)
- **Version**: v0.4.1+ required
- **SNLS Dependency**: https://github.com/LLNL/SNLS.git

### **RAJA Portability Suite**
For GPU builds of ExaCMech >= v0.4.1:

#### **Required Components**
- **RAJA**: Performance portability framework
- **Umpire**: Memory management
- **CHAI**: Array abstraction

#### **Version Requirements**
- **Tag**: `v2024.07.0` for all RAJA Portability Suite repositories
- **Minimum RAJA**: v2022.10.x due to MFEMv4.5 dependency updates

### **Additional Dependencies**
- **HYPRE**: v2.26.0 - v2.30.0 (algebraic multigrid)
- **METIS**: Version 5 (mesh partitioning)
- **ADIOS2**: Optional (high-performance parallel I/O)
- **ZLIB**: Optional (compressed mesh and data support)

## Codebase Overview

ExaConstit follows a modular architecture designed for extensibility and performance:

```
ExaConstit/
├── src/                   # Main source code
├── test/                  # Test cases and examples
├── scripts/               # Build scripts and utilities
├── workflows/             # Optimization and UQ workflows
└── cmake/                 # Build system configuration
```

### Design Philosophy
- **Modularity**: Clear separation of concerns between FEM operators, material models, and solvers
- **Performance**: GPU acceleration and memory-efficient algorithms
- **Extensibility**: Plugin architecture for material models and boundary conditions
- **Standards**: Modern C++17 practices and comprehensive documentation

## Source Directory Structure

The `src/` directory contains the core ExaConstit implementation:

### Primary Files
- **`mechanics_driver.cpp`**: Main application entry point and simulation orchestration
- **`system_driver.hpp/cpp`**: Core driver class managing the Newton-Raphson solution process
- **`userumat.h`**: Interface definitions for UMAT integration

### Key Directories

#### `boundary_conditions/`
**Purpose**: Boundary condition management and enforcement
- **`BCData.hpp/cpp`**: Data structures for boundary condition storage
- **`BCManager.hpp/cpp`**: Boundary condition application and management

**Key Features**:
- Dirichlet velocity and velocity gradient boundary conditions
- Time-dependent boundary condition support
- Mixed boundary condition types

#### `fem_operators/`
**Purpose**: Finite element operators and assembly
- **`mechanics_operator.hpp/cpp`**: Core nonlinear mechanics operator
- **`mechanics_operator_ext.hpp/cpp`**: Extended operators for GPU assembly
- **`mechanics_integrators.hpp/cpp`**: Element integration kernels

**Key Features**:
- Partial assembly (PA) and element assembly (EA) modes
- GPU-accelerated integration
- B-bar formulation support

#### `models/`
**Purpose**: Material constitutive model interface and implementations
- **`mechanics_model.hpp/cpp`**: Base material model interface
- **`mechanics_ecmech.hpp/cpp`**: ExaCMech crystal plasticity integration
- **`mechanics_umat.hpp/cpp`**: UMAT interface implementation
- **`mechanics_multi_model.hpp/cpp`**: Multi-region material management

**Material Model Types**:
- Crystal plasticity models via ExaCMech
- User-defined material models via UMAT interface
- Multi-phase and composite materials

#### `solvers/`
**Purpose**: Nonlinear and linear solver implementations
- **`mechanics_solver.hpp/cpp`**: Newton-Raphson solver with line search

**Solver Features**:
- Newton-Raphson and Newton with line search
- Krylov iterative solvers (GMRES, CG, MINRES)
- Matrix-free and matrix-based approaches

#### `options/`
**Purpose**: Configuration parsing and validation
- **`option_parser_v2.hpp/cpp`**: Main configuration parser
- **`option_*.cpp`**: Specialized option parsers for different components

**Configuration System**:
- TOML-based configuration files
- Modular configuration with external file support
- Comprehensive validation and error reporting

#### `postprocessing/`
**Purpose**: Analysis and visualization output
- **`postprocessing_driver.hpp/cpp`**: Main post-processing orchestration
- **`projection_class.hpp`**: Field projection and averaging
- **`mechanics_lightup.hpp`**: In-situ lattice strain calculations

**Post-processing Capabilities**:
- Volume averaging and stress-strain curves
- Visualization output (VisIt, ParaView, ADIOS2)
- Lattice strain calculations for diffraction analysis

#### `utilities/`
**Purpose**: Common utilities and helper functions
- **`mechanics_kernels.hpp`**: RAJA kernels for material evaluation
- **`mechanics_log.hpp`**: Logging and performance monitoring
- **`assembly_ops.hpp`**: Assembly operation utilities
- **`rotations.hpp`**: Rotation and orientation utilities
- **`strain_measures.hpp`**: Strain computation utilities

#### `sim_state/`
**Purpose**: Simulation state management
- **`simulation_state.hpp/cpp`**: Central simulation state container

#### `mfem_expt/`
**Purpose**: MFEM extensions and experimental features
- **`partial_qspace.hpp/cpp`**: Partial quadrature space implementations
- **`partial_qfunc.hpp/cpp`**: Partial quadrature function utilities

## Key Components

### SystemDriver Class
The `SystemDriver` class orchestrates the entire simulation workflow:

**Responsibilities**:
- Newton-Raphson nonlinear solution
- Linear solver and preconditioner management
- Boundary condition enforcement
- Time stepping control
- Material model coordination

**Key Methods**:
```cpp
void Initialize();           // Setup and initialization
void Solve();               // Main solution loop
void UpdateMesh();          // Mesh updates for large deformation
void ApplyBoundaryConditions(); // BC enforcement
```

### NonlinearMechOperator Class
The finite element operator that provides:
- Residual evaluation for Newton-Raphson
- Jacobian computation and assembly
- Essential DOF management
- GPU-accelerated assembly options

### Material Model Interface
Base class `ExaModel` defines the constitutive model interface:
```cpp
virtual void GetStress() = 0;           // Stress computation
virtual void UpdateState() = 0;        // State variable updates
virtual void GetTangent() = 0;         // Tangent stiffness
```

## Configuration System

ExaConstit uses TOML-based configuration files for all simulation parameters:

### Main Configuration File (`options.toml`)
```toml
basename = "simulation_name"
version = "0.9.0"

[Mesh]
filename = "mesh.mesh"
refinement_levels = 0

[Time.Fixed]
dt = 1.0e-3
t_final = 1.0

[Solvers.Krylov]
newton_rel_tol = 1.0e-6
newton_abs_tol = 1.0e-10
linear_solver = "gmres"
assembly = "pa"

[Materials]
# Material definitions...

[BCs]
# Boundary condition specifications...
```

### Modular Configuration
- **External material files**: `materials = ["material1.toml", "material2.toml"]`
- **External post-processing**: `post_processing = "postproc.toml"`
- **Grain data files**: `grain_file = "grain.txt"`, `orientation_file = "orientations.txt"`

## Advanced Solver Configuration

### **Assembly Methods**
ExaConstit supports multiple finite element assembly strategies optimized for different hardware:

#### **Partial Assembly (PA)**
```toml
[Solvers]
assembly = "PA"
```
- **Memory efficient**: No global matrix formation
- **GPU optimized**: Ideal for GPU acceleration
- **Matrix-free**: Jacobian actions computed on-the-fly
- **Preconditioning**: Currently limited to Jacobi preconditioning

#### **Element Assembly (EA)**
```toml
[Solvers]
assembly = "EA"
```
- **Element-level**: Only element matrices formed
- **Memory balanced**: Moderate memory requirements
- **GPU compatible**: Supports GPU execution
- **Flexibility**: Suitable for complex material models

#### **Full Assembly**
```toml
[Solvers]
assembly = "FULL"
```
- **Traditional**: Complete global matrix assembly
- **Preconditioning**: Full preconditioner options available
- **Memory intensive**: Requires significant memory for large problems
- **CPU optimized**: Best for CPU-only calculations

### **Integration Schemes**

#### **Default Integration**
```toml
[Solvers]
integ_model = "DEFAULT"
```
- **Full integration**: Complete quadrature point evaluation
- **Standard**: Traditional finite element approach
- **Most materials**: Suitable for general material models

#### **B-Bar Integration**
```toml
[Solvers]
integ_model = "BBAR"
```
- **Mixed formulation**: Deviatoric and volumetric split
- **Near-incompressible**: Prevents volumetric locking
- **Advanced**: Based on Hughes-Brezzi formulation (Equation 23)
- **Limitation**: Not compatible with partial assembly

### **Linear Solver Options**

#### **Krylov Methods**
```toml
[Solvers.Krylov]
linear_solver = "GMRES"    # or "cg", "minres"
linear_rel_tol = 1.0e-6
linear_abs_tol = 1.0e-10
linear_max_iter = 1000
```

**GMRES**: General minimal residual
- **Nonsymmetric systems**: Handles general Jacobian matrices
- **Memory**: Requires restart for memory management
- **Robust**: Suitable for challenging material models

**Conjugate Gradient (CG)**: 
- **Symmetric positive definite**: Requires symmetric Jacobian
- **Memory efficient**: Minimal memory requirements
- **Fast convergence**: Optimal for appropriate problems

**MINRES**: Minimal residual for symmetric indefinite
- **Symmetric indefinite**: Handles saddle point problems
- **Specialized**: Useful for constrained problems

#### **Preconditioning**
```toml
[Solvers.Krylov]
preconditioner = "AMG"     # or "jacobi", "none"
```

**Algebraic Multigrid (AMG)**:
- **BoomerAMG**: HYPRE implementation
- **Scalable**: Excellent for large-scale problems
- **Setup cost**: Requires matrix assembly

**Jacobi Preconditioning**:
- **Matrix-free**: Compatible with PA and EA assembly
- **Simple**: Diagonal scaling preconditioning
- **GPU friendly**: Efficient device implementation

### **Nonlinear Solver Configuration**

#### **Newton-Raphson Variants**
```toml
[Solvers.NR]
nonlinear_solver = "NEWTON"           # or "newton_ls"
newton_rel_tol = 1.0e-6
newton_abs_tol = 1.0e-10
newton_max_iter = 20
```

**Standard Newton-Raphson**:
- **Full steps**: Always takes complete Newton step
- **Fast convergence**: Quadratic convergence near solution
- **Robustness**: May fail for poor initial guesses

**Newton with Line Search**:
- **Globalization**: Backtracking line search for robustness
- **Convergence**: Improved convergence from poor starting points
- **Cost**: Additional function evaluations per iteration

## Building and Testing

### Build Configuration Options
```bash
# Enable GPU support
-DENABLE_CUDA=ON
-DENABLE_HIP=ON

# Enable specific features
-DENABLE_CALIPER=ON
```

### Running Tests
```bash
# Run example simulations
cd test/data
mpirun -np 4 ../../build/mechanics -opt example.toml
```

### Example Workflows
The `test/data/` directory contains various example cases:
- **Crystal plasticity simulations**
- **Multi-material problems**
- **Complex boundary condition examples**
- **GPU acceleration tests**

## Development Workflow

### Code Organization Best Practices
1. **Header-only utilities**: Place in `utilities/` directory
2. **New material models**: Extend `ExaModel` base class in `models/`
3. **Post-processing features**: Add to `postprocessing/` directory
4. **Configuration options**: Update corresponding `option_*.cpp` files

### Adding New Features

#### New Material Model
1. Create header/source in `models/mechanics_newmodel.hpp/cpp`
2. Inherit from `ExaModel` base class
3. Implement required virtual methods
4. Add configuration parsing support
5. Update `CMakeLists.txt`

#### New Boundary Condition Type
1. Extend `BCManager` class
2. Add parsing support in `option_boundary_conditions.cpp`
3. Update documentation and examples

### Performance Considerations
- **GPU kernels**: Use RAJA for performance portability
- **Memory management**: Follow MFEM memory patterns
- **MPI communication**: Minimize collective operations
- **Assembly strategy**: Choose PA vs EA based on problem size

### Debugging and Profiling
- **Caliper integration**: Built-in performance profiling
- **MFEM debugging**: Use MFEM's debugging capabilities
- **GPU debugging**: CUDA/HIP debugging tools
- **MPI debugging**: TotalView, DDT support

## UMAT Development Resources

### **Interface Requirements**
While UMAT interfaces are traditionally described using Fortran signatures, ExaConstit supports implementation in **Fortran, C++, or C**:

#### **Standard UMAT Signature** (Fortran style)
```fortran
SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
 1 RPL,DDSDDT,DRPLDE,DRPLDT,
 2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
 3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
 4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
```

#### **C++ Implementation Example**
```cpp
extern "C" void umat_(double* stress, double* statev, double* ddsdde,
                     double* sse, double* spd, double* scd,
                     // ... additional parameters
                     int* ndi, int* nshr, int* ntens, int* nstatv,
                     double* props, int* nprops,
                     // ... remaining parameters
                     );
```

### **UMAT Development Best Practices**

#### **Memory Management**
- **ExaConstit handles**: State variable allocation and persistence
- **UMAT responsible**: Local variable management within subroutine
- **No dynamic allocation**: Avoid malloc/new within UMAT calls

#### **Thread Safety**
- **No global variables**: UMATs must be thread-safe
- **Local computations**: All calculations using passed parameters
- **State persistence**: Only through provided state variable arrays

#### **Error Handling**
- **Convergence issues**: Set appropriate flags for Newton-Raphson
- **Material failure**: Handle through state variables or stress reduction
- **Numerical stability**: Check for divide-by-zero and overflow conditions

#### **Performance Considerations**
- **CPU execution only**: No current GPU acceleration for UMATs
- **Vectorization**: Ensure compiler optimization is possible
- **Minimal function calls**: Reduce computational overhead within UMAT

### **Development Resources**

#### **Reference Implementations**
- **`src/umat_tests/`**: Example UMAT implementations and conversion guides
- **Template UMATs**: Starting points for custom development

#### **External Resources**
- **NJIT UMAT Collection**: https://web.njit.edu/~sac3/Software.html
- **Academic examples**: Various constitutive models available
- **License considerations**: Verify licensing before use

#### **Build System Integration**
```bash
# Compile UMAT to shared library (Fortran)
gfortran -shared -fPIC -o my_umat.so my_umat.f90

# Compile UMAT (C++)
g++ -shared -fPIC -o my_umat.so my_umat.cpp

# Compile UMAT (C)
gcc -shared -fPIC -o my_umat.so my_umat.c
```

#### **Configuration Integration**
```toml
[Materials.regions.model.UMAT]
library_path = "/path/to/my_umat.so"
num_props = 8
num_state_vars = 12
props = [
    210000.0,  # Young's modulus
    0.3,       # Poisson's ratio
    # ... additional parameters
]
```

## Contributing Guidelines

### Code Standards
- **C++17 compliance**: Use modern C++ features
- **Documentation**: Doxygen-style comments for all public interfaces
- **Testing**: Include test cases for new features
- **Performance**: Maintain GPU and MPI scalability

### Pull Request Process
1. Fork the repository
2. Create feature branch from `exaconstit-dev`
3. Implement changes with tests
4. Ensure all existing tests pass
5. Submit pull request with detailed description

### Licensing
- **BSD-3-Clause license**: All contributions must use this license
- **Third-party code**: Ensure compatible licensing for external dependencies

### Getting Help
- **Primary Developer**: Robert A. Carson (carson16@llnl.gov)
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Refer to MFEM and ExaCMech documentation for underlying libraries

## Additional Resources

### Related Projects
- **ExaCMech**: Crystal plasticity library (https://github.com/LLNL/ExaCMech)
- **MFEM**: Finite element library (https://mfem.org)
- **ExaCA**: Cellular automata for microstructure generation

### Workflows and Applications
- **Optimization workflows**: Multi-objective genetic algorithm parameter optimization
- **UQ workflows**: Uncertainty quantification for additive manufacturing
- **Post-processing tools**: Python scripts for data analysis

### Citation
If using ExaConstit in your research, please cite:
```bibtex
@misc{ exaconstit,
title = {{ExaConstit}},
author = {Carson, Robert A. and Wopschall, Steven R. and Bramwell, Jamie A.},
abstractNote = {The principal purpose of this code is to determine bulk constitutive properties and response of polycrystalline materials. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based). Within this context, there is flexibility in the type of constitutive model employed, with the code allowing for various UMATs to be interfaced within the code framework or for the use of the ExaCMech library. Using crystal-mechanics-based constitutive models, the code can be used, for example, to compute homogenized response behavior over a polycrystal. },
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20191024.2}},
url = {https://github.com/LLNL/ExaConstit},
doi = {10.11578/dc.20191024.2},
year = {2019},
month = {Aug},
annote = {
   https://www.osti.gov//servlets/purl/1571640
   https://www.osti.gov/biblio/1571640-exaconstit
}
}
```

---

This guide provides a foundation for new developers to understand and contribute to ExaConstit. For specific implementation details, refer to the extensive inline documentation throughout the codebase and the example configurations in `test/data/`.