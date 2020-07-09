# ExaConstit App
## Author:
* Robert A. Carson (Principal Developer)
  * carson16@llnl.gov

* Steven R. Wopschall
  * wopschall1@llnl.gov

* Jamie Bramwell
  * bramwell1@llnl.gov

Date: Aug. 6, 2017

Updated: July 9, 2020

# Description: 
The purpose of this code app is to determine bulk constitutive properties of metals. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based).
               
Currently, only Dirichlet boundary conditions (homogeneous and inhomogeneous by dof component) have been implemented. Neumann (traction) boundary conditions and a body force are not implemented. Changing essential boundary conditions is on our roadmap but has not been implemented at this time. A new ExaModel class allows one to implement arbitrary constitutive models. The code currently successfully allows for various UMATs to be interfaced within the code framework.

The code is capable of running on the GPU by making use of either a partial assembly formulation (no global matrix formed) or element assembly (only element assembly formed) of our typical FEM code. These methods currently only implement a simple matrix-free jacobi preconditioner. MFEM team is currently working on other matrix-free preconditioners.

The code supports either constant time steps or user supplied delta time steps. Boundary conditions are supplied for the velocity field applied on a surface. It supports a number of different preconditioned Krylov iterative solvers (PCG, GMRES, MINRES) for either symmetric or nonsymmetric positive-definite systems. 


## Remark:
This code is still very much actively being developed. It should be expected that breaking changes can and will occur. So, we make no guarantees about stability at this point in time. Any available release should be considered stable but may be lacking several features of interest that are found in the ```exaconstit-dev``` branch.

Currently, the code has been tested using monotonic loading with either auto-generated mesh that's been instantiated with grain data from some voxel data set or meshes formed from ```MFEM v1.0```. Meshes produced from Neper can also be used but do require some additional post-processing into the ```MFEM v1.0``` mesh format. See the ```Script``` section for one way of accomplishing this.

ExaCMech models are capable of running on the GPU. However, we currently have no plans for doing the same for UMATs based kernels. The ExaCMech material class can be used as a guide for how to do the necessary set-up, material kernel, and post-processing step if a user would like to expand the UMAT features and submit a pull request to add the capabilities into ExaConstit.

See the included ```options.toml``` to see all of the various different options that are allowable in this code and their default values.

A TOML parser has been included within this directory, since it has an MIT license. The repository for it can be found at: https://github.com/skystrife/cpptoml .

Example UMATs maybe obtained from https://web.njit.edu/~sac3/Software.html . We have not included them due to a question of licensing. The ones that have been run and are known to work are the linear elasticity model and the neo-Hookean material. The ```umat_tests``` subdirectory in the ```src``` directory can be used as a guide for how to convert your own UMATs over to one that ExaConstit can interface with.

Note: the grain.txt, props.txt and state.txt files are expected inputs for CP problems. If a mesh is provided it should be in the MFEM format which has the grains IDs already assigned to the element attributes.

# Scripts
Useful scripts are provided within the ```scripts``` directory. The ```mesh_generator``` executable when generated can create an ```MFEM v1.0``` mesh for auto-generated mesh when provided a grain ID file. It is also capable of taking in a ```vtk``` mesh file that MFEM is capable of reading, and then it will generate the appropriate ```MFEM v1.0``` file format with the boundary element attributes being generated in the same way ExaConstit expects them. The ```vtk``` mesh currently needs to be a rectilinear mesh in order to work. All of the options for ```mesh_generator``` can be viewed by running ```./mesh_generator --help```

An additional python script is provided called ```fepx2mfem_mesh.py``` that provides a method to convert from a mesh generated using Neper in the FEpX format into the ```vtk``` format that can now be converted over to the ```MFEM v1.0``` format using the ```mesh_generator``` script.

# Examples

Several small examples that you can run are found in the ```test\data``` directory.

# Installing Notes:

* git clone the LLNL BLT library into cmake directory. It can be obtained at https://github.com/LLNL/blt.git
* MFEM will need to be built with Conduit (built with HDF5). The easiest way to install Conduit is to use spack install instruction provided by Conduit
  * You'll need to use the exaconstit-dev branch of MFEM found on this fork of MFEM: https://github.com/rcarson3/mfem.git
  * We do plan on upstreaming the necessary changes needed for ExaConstit into the master branch of MFEM, so you'll no longer be required to do this
* ExaCMech is required for ExaConstit to be built and can be obtained at https://github.com/LLNL/ExaCMech.git.   

* Create a build directory and cd into there
* Run ```cmake .. -DENABLE_MPI=ON -DENABLE_FORTRAN=ON -DMFEM_DIR{mfem's installed cmake location} -DBLT_SOURCE_DIR=${BLT cloned location if not located in cmake directory} -DECMECH_DIR=${ExaCMech installed cmake location} -DRAJA_DIR={RAJA installed location} -DSNLS_DIR={SNLS location in ExaCMech}```
* Run ```make -j 4```


#  Future Implemenations Notes:
               
* Multiple phase materials
* Evolving loading conditions
* Commonly used post-processing tools either through Python or C++ code

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
