# ExaConstit App

Updated: April. 4, 2021

Version 0.5.0

# Description: 
A principal purpose of this code app is to probe the deformation response of polycrystalline materials; for example, in homogenization to obtain bulk constitutive properties of metals. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based).
               
Currently, only Dirichlet boundary conditions (homogeneous and inhomogeneous by degree-of-freedom component) have been implemented. Neumann (traction) boundary conditions and a body force are not implemented. We support changing boundary conditions as well. So, it's possible to run cyclic, strain-rate jump tests, or a number of other type simulations. A new ExaModel class allows one to implement arbitrary constitutive models. Crystal plasticity model capabilities are primarily provided through the ExaCMech library. The code also currently allows for various UMATs to be interfaced within the code framework.

Through the ExaCMech library, we are able to offer a range of crystal plasticity models that can run on the GPU. The current models that are available are a power law slip kinetic model with both nonlinear and linear variations of a voce hardening law for BCC and FCC materials, and a single Kocks-Mecking dislocation density hardening model with balanced thermally activated slip kinetics with phonon drag effects for BCC, FCC, and HCP materials. Any future model types to the current list are a simple addition within ExaConstit, but they will need to be implemented within ExaCMech. Given the templated structure of ExaCMech, some additions would be comparatively straightforward. 

The code is capable of running on the GPU by making use of either a partial assembly formulation (no global matrix formed) or element assembly (only element assembly formed) of our typical FEM code. These methods currently only implement a simple matrix-free jacobi preconditioner. The MFEM team is currently working on other matrix-free preconditioners.

The code supports either constant time steps or user-supplied variable time steps. Boundary conditions are supplied for the velocity field on a surface. The code supports a number of different preconditioned Krylov iterative solvers (PCG, GMRES, MINRES) for either symmetric or nonsymmetric positive-definite systems. We also support either a newton raphson or newton raphson with a line search for the nonlinear solve. We might eventually look into supporting a nonlinear solver such as L-BFGS as well.

Finally, we support being able to make use of full integration or BBar type integration schemes to be used with various models. The default feature is to perform full integration of the element at the quadrature point. The BBar integration performs full integration of the deviatoric response with an element average integration for the volume response. The BBar method is based on the work given in [this paper](https://doi.org/10.1002/nme.1620150914) and more specifically we make use of Eq 23. It should be noted that currently we don't support a partial assembly formulation for the BBar integrations.


## Remark:
This code is still very much actively being developed. It should be expected that breaking changes can and will occur. So, we make no guarantees about stability at this point in time. Any available release should be considered stable but may be lacking several features of interest that are found in the ```exaconstit-dev``` branch.

Currently, the code has been tested using monotonic and cyclic loading with either an auto-generated mesh that has been instantiated with grain data from some voxel data set or meshes formed from ```MFEM v1.0```. Meshes produced from Neper can also be used but do require some additional post-processing. See the ```Script``` section for ways to accomplishing this.

ExaCMech models are capable of running on the GPU. However, we currently have no plans for doing the same for UMAT-based kernels. The ExaCMech material class can be used as a guide for how to do the necessary set-up, material kernel, and post-processing step if a user would like to expand the UMAT features and submit a pull request to add the capabilities into ExaConstit.

See the included ```options.toml``` to see all of the various different options that are allowable in this code and their default values.

A TOML parser has been included within this directory, since it has an MIT license. The repository for it can be found at: https://github.com/skystrife/cpptoml .

Example UMATs maybe obtained from https://web.njit.edu/~sac3/Software.html . We have not included them due to a question of licensing. The ones that have been run and are known to work are the linear elasticity model and the neo-Hookean material. The ```umat_tests``` subdirectory in the ```src``` directory can be used as a guide for how to convert your own UMATs over to one with which ExaConstit can interface.

Note: the grain.txt, props.txt and state.txt files are expected inputs for crystal-plasticity problems. If a mesh is provided it should be in the MFEM or cubit format which has the grains IDs already assigned to the element attributes.

# Scripts
Useful scripts are provided within the ```scripts``` directory. The ```mesh_generator``` executable when generated can create an ```MFEM v1.0``` mesh for auto-generated mesh when provided a grain ID file. It is also capable of taking in a ```vtk``` mesh file that MFEM is capable of reading, and then it will generate the appropriate ```MFEM v1.0``` file format with the boundary element attributes being generated in the same way ExaConstit expects them. The ```vtk``` mesh currently needs to be a rectilinear mesh in order to work. All of the options for ```mesh_generator``` can be viewed by running ```./mesh_generator --help```

If you have version 4 of ```Neper``` then you can make use of the `-faset 'faces'` option while meshing and output things as a `gmsh` v2.2 file. Afterwards, you can make use of the `neper_v4_mesh.py` cli script in `scripts/meshing` to automatically use the faset information and autogenerate the boundary attributes that `MFEM\ExaConstit` can understand and use. Although, you will need to check and see which face corresponds to what boundary attribute, so you can correctly apply boundary conditions to the body. Further information is provided in the top level comment of the script for how to do this.

For older versions of neper v2-v3, an additional python script is provided called ```fepx2mfem_mesh.py``` that provides a method to convert from a mesh generated using Neper v3.5.2 in the FEpX format into the ```vtk``` format that can now be converted over to the ```MFEM v1.0``` format using the ```mesh_generator``` script.

# Examples

Several small examples that you can run are found in the ```test\data``` directory.

# Postprocessing

The ```scripts/postprocessing``` directory contains several useful post-processing tools. The ```macro_stress_strain_plot.py``` file can be used to generate macroscopic stress strain plots. An example script ```adios2_example.py``` is provided as example for how to make use of the ```ADIOS2``` post-processing files if ```MFEM``` was compiled with ```ADIOS2``` support. It's highly recommended to install ```MFEM``` with this library if you plan to be doing a lot of post-processing of data in python.

# Installing Notes:

* git clone the LLNL BLT library into cmake directory. It can be obtained at https://github.com/LLNL/blt.git
* MFEM will need to be built with hypre v2.18.2 - v2.20.*; metis5; RAJA; and optionally Conduit, ADIOS2, or ZLIB.
  * Conduit and ADIOS2 supply output support. ZLIB allows MFEM to read in gzip mesh files or save data as being compressed.
  * You'll need to use the exaconstit-dev branch of MFEM found on this fork of MFEM: https://github.com/rcarson3/mfem.git
  * We do plan on upstreaming the necessary changes needed for ExaConstit into the master branch of MFEM, so you'll no longer be required to do this
* ExaCMech is required for ExaConstit to be built and can be obtained at https://github.com/LLNL/ExaCMech.git and now requires the develop branch. ExaCMech depends internally on SNLS, from https://github.com/LLNL/SNLS.git.
* RAJA is required for ExaConstit to be built and should be the same one that ExaCMech and MFEM are built with. It can be obtained at https://github.com/LLNL/RAJA. Currently, RAJA >= v0.13.0 is required for ExaConstit due to a dependency update to MFEMv4.3.
* An example install bash script for unix systems can be found in ```scripts/install/unix_install_example.sh```. This is provided as an example of how to install ExaConstit and its dependencies, but it is not guaranteed to work on every system. A CUDA version of that script is also included in that folder, and only minor modifications are required if using a version of Cmake  >= 3.18.*. In those cases ```CUDA_ARCH``` has been changed to ```CMAKE_CUDA_ARCHITECTURES```. You'll also need to look up what you're CUDA architecture compute capability is set to and modify that within the script. Currently, it is set to ```sm_70``` which is associated with the Volta architecture.


* Create a build directory and cd into there
* Run ```cmake .. -DENABLE_MPI=ON -DENABLE_FORTRAN=ON -DMFEM_DIR{mfem's installed cmake location} -DBLT_SOURCE_DIR=${BLT cloned location if not located in cmake directory} -DECMECH_DIR=${ExaCMech installed cmake location} -DRAJA_DIR={RAJA installed location} -DSNLS_DIR={SNLS location in ExaCMech}```
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
ExaConstit can be cited using the following ```bibtex``` entry:

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
