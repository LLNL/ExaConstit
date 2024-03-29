# ExaConstit App

Updated: June. 10, 2022

Version 0.7.0

# Description: 
A principal purpose of this code app is to probe the deformation response of polycrystalline materials; for example, in homogenization to obtain bulk constitutive properties of metals. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based).
               
Currently, only Dirichlet boundary conditions (homogeneous and inhomogeneous by degree-of-freedom component) have been implemented. Neumann (traction) boundary conditions and a body force are not implemented. These Dirichlet boundary conditions can applied per surface/boundary using either the use of applied velocity or applied velocity gradients (constant strain) boundary conditions. One can also mix and match the use of those two boundary condition types across the various boundaries of the problem in order to more complicated material deformations such as pure torsion. Additionally, we support changing boundary conditions. So, it's possible to run cyclic, strain-rate jump tests, or a number of other type simulations.

On the material modelling front of things, ExaConstit can easily handle various material models. We provide a base class, `ExaModel`, to build each material model or class of material models. We currently support  two very different material model libraries/interfaces through UMATs or ExaCMech. Crystal plasticity model capabilities are primarily provided through the ExaCMech library.

Through the ExaCMech library, we are able to offer a range of crystal plasticity models that can run on the GPU. The current models that are available are a power law slip kinetic model with both nonlinear and linear variations of a voce hardening law for BCC and FCC materials, and a single Kocks-Mecking dislocation density hardening model with balanced thermally activated slip kinetics with phonon drag effects for BCC, FCC, and HCP materials. Any future model types to the current list are a simple addition within ExaConstit, but they will need to be implemented within ExaCMech. Given the templated structure of ExaCMech, some additions would be comparatively straightforward. 

The code is capable of running on the GPU by making use of either a partial assembly formulation (no global matrix formed) or element assembly (only element assembly formed) of our typical FEM code. These methods currently only implement a simple matrix-free jacobi preconditioner. The MFEM team is currently working on other matrix-free preconditioners. Additionally, ExaConstit can be built to run with either CUDA or HIP-support in-order to run on most GPU-capable machines out there.

The code supports constant time steps, user-supplied variable time steps, or automatically calculated time steps. Boundary conditions are supplied for the velocity field on a surface. The code supports a number of different preconditioned Krylov iterative solvers (PCG, GMRES, MINRES) for either symmetric or nonsymmetric positive-definite systems. We also support either a newton raphson or newton raphson with a line search for the nonlinear solve. We might eventually look into supporting a nonlinear solver such as L-BFGS as well.

Finally, we support being able to make use of full integration or BBar type integration schemes to be used with various models. The default feature is to perform full integration of the element at the quadrature point. The BBar integration performs full integration of the deviatoric response with an element average integration for the volume response. The BBar method is based on the work given in [this paper](https://doi.org/10.1002/nme.1620150914) and more specifically we make use of Eq 23. It should be noted that currently we don't support a partial assembly formulation for the BBar integrations.


## Remark:
This code is still very much actively being developed. It should be expected that breaking changes can and will occur. So, we make no guarantees about stability at this point in time. Any available release should be considered stable but may be lacking several features of interest that are found in the ```exaconstit-dev``` branch.

Currently, the code has been tested using monotonic and cyclic loading with either an auto-generated mesh that has been instantiated with grain data from some voxel data set or meshes formed from ```MFEM v1.0```. Meshes produced from Neper can also be used but do require some additional post-processing. See the ```Script``` section for ways to accomplishing this.

ExaCMech models are capable of running on the GPU. However, we currently have no plans for doing the same for UMAT-based kernels. The ExaCMech material class can be used as a guide for how to do the necessary set-up, material kernel, and post-processing step if a user would like to expand the UMAT features and submit a pull request to add the capabilities into ExaConstit.

See the included ```options.toml``` to see all of the various different options that are allowable in this code and their default values.

A TOML parser has been included within this directory, since it has an MIT license. The repository for it can be found at: https://github.com/ToruNiina/toml11/.

Example UMATs maybe obtained from https://web.njit.edu/~sac3/Software.html . We have not included them due to a question of licensing. The ones that have been run and are known to work are the linear elasticity model and the neo-Hookean material. The ```umat_tests``` subdirectory in the ```src``` directory can be used as a guide for how to convert your own UMATs over to one with which ExaConstit can interface.

Note: the grain.txt, props.txt and state.txt files are expected inputs for crystal-plasticity problems. If a mesh is provided it should be in the MFEM or cubit format which has the grains IDs already assigned to the element attributes.

# Scripts
Useful scripts are provided within the ```scripts``` directory. The ```mesh_generator``` executable when generated can create an ```MFEM v1.0``` mesh for auto-generated mesh when provided a grain ID file. It is also capable of taking in a ```vtk``` mesh file that MFEM is capable of reading, and then it will generate the appropriate ```MFEM v1.0``` file format with the boundary element attributes being generated in the same way ExaConstit expects them. The ```vtk``` mesh currently needs to be a rectilinear mesh in order to work. All of the options for ```mesh_generator``` can be viewed by running ```./mesh_generator --help```

If you have version 4 of ```Neper``` then you can make use of the `-faset 'faces'` option while meshing and output things as a `gmsh` v2.2 file. Afterwards, you can make use of the `neper_v4_mesh.py` cli script in `scripts/meshing` to automatically use the faset information and autogenerate the boundary attributes that `MFEM\ExaConstit` can understand and use. Although, you will need to check and see which face corresponds to what boundary attribute, so you can correctly apply boundary conditions to the body. Further information is provided in the top level comment of the script for how to do this.

For older versions of neper v2-v3, an additional python script is provided called ```fepx2mfem_mesh.py``` that provides a method to convert from a mesh generated using Neper v3.5.2 in the FEpX format into the ```vtk``` format that can now be converted over to the ```MFEM v1.0``` format using the ```mesh_generator``` script.

# Examples

Several small examples that you can run are found in the ```test/data``` directory. These examples cover a wide range of different use cases of the code, but the `toml` file for each test case may not be representative of all the options as found in the `src/options.toml` file.

# Postprocessing

The ```scripts/postprocessing``` directory contains several useful post-processing tools. The ```macro_stress_strain_plot.py``` file can be used to generate macroscopic stress strain plots. An example script ```adios2_example.py``` is provided as example for how to make use of the ```ADIOS2``` post-processing files if ```MFEM``` was compiled with ```ADIOS2``` support. It's highly recommended to install ```MFEM``` with this library if you plan to be doing a lot of post-processing of data in python.

A set of scripts to perform lattice strain calculations similar to those found in powder diffraction type experiments can be found in the ```scripts/postprocessing``` directory. The appropriate python scripts are: `adios2_extraction.py`, `strain_Xtal_to_Sample.py`, and `calc_lattice_strain.py`. In order to use these scripts, one needs to run with the `light_up=true` option set in the `Visualization` table of your simulation option file.

# Workflow Examples

We've provided several different useful workflows in the `workflows` directory. One is an optimization set of scripts that makes use of a genetic algorithm to optimize material parameters based on experimental results. Internally, it makes use of either a simple workflow manager for something like a workstation or it can leverage the python bindings to the Flux job queue manager created initially by LLNL to run on large HPC systems.

The other workflow is based on a UQ workflow for metal additive manufacturing that was developed as part of the ExaAM project. You can view the open short workshop paper for an overview of the ExaAM project's workflow and the results https://doi.org/10.1145/3624062.3624103 . This workflow connects microstructures provided by an outside code such as LLNL's ExaCA code (https://github.com/LLNL/ExaCA) or other sources such as nf-HEDM methods to local properties to be used by a part scale application code. The goal here is to utilize ExaConstit to run a ton of simulations rather than experiments in order to obtain data that can be used to parameterize macroscopic material models such as an anisotropic yield surface.

# Installing Notes:

* git clone the LLNL BLT library into cmake directory. It can be obtained at https://github.com/LLNL/blt.git
* MFEM will need to be built with hypre v2.26.0-v2.30.0; metis5; RAJA v2022.x+; and optionally Conduit, ADIOS2, or ZLIB.
  * Conduit and ADIOS2 supply output support. ZLIB allows MFEM to read in gzip mesh files or save data as being compressed.
  * You'll need to use the exaconstit-dev branch of MFEM found on this fork of MFEM: https://github.com/rcarson3/mfem.git
  * We do plan on upstreaming the necessary changes needed for ExaConstit into the master branch of MFEM, so you'll no longer be required to do this
  * Version 0.7.0 of Exaconstit is compatible with the following mfem hash 78a95570971c5278d6838461da6b66950baea641
  * Version 0.6.0 of ExaConstit is compatible with the following mfem hash 1b31e07cbdc564442a18cfca2c8d5a4b037613f0
  * Version 0.5.0 of ExaConstit required 5ebca1fc463484117c0070a530855f8cbc4d619e
* ExaCMech is required for ExaConstit to be built and can be obtained at https://github.com/LLNL/ExaCMech.git and now requires the develop branch. ExaCMech depends internally on SNLS, from https://github.com/LLNL/SNLS.git. We depend on v0.3.4 of ExaCMech as of this point in time.
  * For versions of ExaCMech >= 0.3.3, you'll need to add `-DENABLE_SNLS_V03=ON` to the cmake commands as a number of cmake changes were made to that library and SNLS.
* RAJA is required for ExaConstit to be built and should be the same one that ExaCMech and MFEM are built with. It can be obtained at https://github.com/LLNL/RAJA. Currently, RAJA >= 2022.10.x is required for ExaConstit due to a dependency update in MFEMv4.5.
* An example install bash script for unix systems can be found in ```scripts/install/unix_install_example.sh```. This is provided as an example of how to install ExaConstit and its dependencies, but it is not guaranteed to work on every system. A CUDA version of that script is also included in that folder, and only minor modifications are required if using a version of Cmake  >= 3.18.*. In those cases ```CUDA_ARCH``` has been changed to ```CMAKE_CUDA_ARCHITECTURES```. You'll also need to look up what you're CUDA architecture compute capability is set to and modify that within the script. Currently, it is set to ```sm_70``` which is associated with the Volta architecture. 


* Create a build directory and cd into there
* Run ```cmake .. -DENABLE_MPI=ON -DENABLE_FORTRAN=ON -DMFEM_DIR{mfem's installed cmake location} -DBLT_SOURCE_DIR=${BLT cloned location if not located in cmake directory} -DECMECH_DIR=${ExaCMech installed cmake location} -DRAJA_DIR={RAJA installed location} -DSNLS_DIR={SNLS installed cmake location}```
* Run ```make -j 4```


#  Future Implemenations Notes:
               
* Multiple phase materials
* Commonly used post-processing tools either through Python or C++ code

# Contributors:
* Robert A. Carson (Principal Developer)
  * carson16@llnl.gov

* Nathan Barton

* Steven R. Wopschall (initial contributions)

* Jamie Bramwell (initial contributions)

# CONTRIBUTING

ExaConstit is distributed under the terms of the BSD-3-Clause license. All new contributions must be made under this license.

# Citation
If you're using ExaConstit and would like to cite us please use the below `bibtex` entry. Additionally, we would love to be able to point to ExaConstit's use in the literature and elsewhere so feel free to message us with a link to your work as Google Scholar does not always pick up the below citation. We can then list your work among the others that have used our code.

```
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

# LICENSE

License is under the BSD-3-Clause license. See [LICENSE](LICENSE) file for details. And see also the [NOTICE](NOTICE) file. 

`SPDX-License-Identifier: BSD-3-Clause`

``LLNL-CODE-793434``
