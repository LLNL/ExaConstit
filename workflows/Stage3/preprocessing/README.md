# Preprocessing Steps

Prerequisites:
- Build ExaConstit as we need the mesh_generator binary that is generated as part of the build process
- Python 3 needs to be installed on the system as all our scripts are for python3.
- ExaCA data that currently has a header with that looks something like:
```
Coordinates are in CA units, 1 cell = 200.0 microns. Data is cell-centered. Origin at 1,1,1
X coord, Y coord, Z coord, Grain ID
```
We need the  cell size in order to ensure the generated mesh has the correct dimensions.

- The unique unit quaternion file provided to this script must line up with the equivalent unique rotation matrices file that ExaCA used to run its simulations. In the past, this file has been called `uni_cubic_10k_quats.txt`. However, if ExaCA ends up using the 1e6 list of orientations I created then we will need to point to a new equivalent unit quaternion file.

So, the driving script for everything is `exaconstit_cli_preprocessing.py`. It is a command line interface tool as the name suggests, but you can just run it as a regular python script by modifying the default options for the cli interface. This script is considered the preprocessing portion of Stage 3 as it takes in the microstructure data, coarsens the data as need be, generates unique orientations for every grain, creates an MFEM mesh file with microstructure captured in the element attribute feature of the mesh file, and saves all the necessary data out to a designated folder with a given name. Finally, it takes in the unique RVE name; property and state files associated with the RVE; temperatures; delta time step file; and number of state, properties, and time steps to generate the test matrix file used in the main simulation script. This test matrix is currently hard coded to an effective 1e-3 engineering strain rate for each simulation, and the loading directions are also hard coded. 

You can run the script doing something like:

```
python3 ./exaconstit_cli_preprocessing.py --help

python3 ./exaconstit_cli_preprocessing.py -ifdir ./ -ifile exaca.csv -ofdir ./output_dir/ -runame super_cool_microstructure -c 1 -mg -mgdir ./ -t 298.0 -fprops ./props_cp_voce_in625.txt -nprops 17 -fstate ./state_cp_voce.txt -nstates 24 -orifile ./uni_cubic_10k_quats.txt -dtfile custom_dt_fine.txt -ntsteps 61 -cfd ../common_simulation_files/
```

# Versions

## v0.2
* Swapped over to use standard Python methods for handling paths. We were already doing this in the `job_cli.py` script in `../main_simulations`, and it just made sense to bring that feature over here.
* Mesh, property file, state file, and delta time file locations are now all absolute locations in the test directory. This change allows us to avoid some extra copying of files.
* New additions to the script have been the addition of a `common_file_directory` option which allows us to point towards a directory that contains files shared between all the RVEs (property, state, and delta time files). This addition was added to reduce the number of duplicated files all over the place. In the future, we could also add the mechanics and mesh binary files and maybe the unique orientations here as well.
* Allow users to point to a specific delta time file and specify number of time steps we want to take.
* Allow users to specify the absolute path of the unique unit quaternion file that is based on what ExaCA uses for their runs. This allows us to not have this file in the same locations as the ExaCA results. 

## v0.1
* This is the version that existed as of commit: ce1e746d5174a5c8bc7d41e3a2c17c5ad2735b16 of the https://code.ornl.gov/ecpcitest/exaam/workflow/workflow_files repo. 
