#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:21:16 2020

@author: carson16
"""

import argparse
#import subprocess
import os
import glob
import re
import pandas as pd

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def fixEssVals(repl_val):
    repl_val = re.sub("\n+", "", repl_val)
    repl_val = ' '.join(repl_val.split())
    repl_val = repl_val.replace("0. ", "0.0")
    repl_val = repl_val.replace("0.000000e+00", "0.0")
    return repl_val

parser = argparse.ArgumentParser(description='Workflow flux and batch cli')

parser.add_argument(
       '-sdir',
       '--simulation_file_dir',
       type=str,
       default='./',
       help='Directory of simulation data from which everything will be symlinked to (default: ./)'
)

parser.add_argument(
       '-odir',
       '--output_directory',
       type=str,
       default='./../wf_runs',
       help='Directory from which all the jobs will be run from (default: ./../wf_runs)'
)

parser.add_argument(
       '-ijfile',
       '--input_job_filename',
       type=str,
       default='hip_mechanics.flux',
       help='Input job script filename for ExaConstit for which we\'ll use to run all of our jobs (default: hip_mechanics.flux)'
)

parser.add_argument(
       '-imtfile',
       '--input_master_toml',
       type=str,
       default='options_master.toml',
       help='Master option toml file for ExaConstit for which we\'ll use to run all of our jobs (default: option_master.toml)'
)

parser.add_argument(
       '-iotfile',
       '--input_output_toml',
       type=str,
       default='options.toml',
       help='Option toml file used for ExaConstit for which we\'ll use to run a job (default: options.toml)'
)

parser.add_argument(
       '-iofile',
       '--input_option_filename',
       type=str,
       default='simulation_test_matrix.csv',
       help='CSV file that contains all the options params which we\'ll use to run all of our jobs (default: simulation_test_matrix.csv)'
)

parser.add_argument(
       '-ijfd',
       '--input_job_filedir',
       type=str,
       default='./',
       help='Path directory of the input job script (default: ./)'
)

parser.add_argument(
       '-bsub',
       '--bsub_jobs',
       action="store_true",
       help='Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool (default: False)'
)

parser.add_argument(
    '-rt',
    '--rtmodel',
    type=str,
    default='CUDA',
    help='Value to use as Solvers.rtmodel in configured options file'
)

args = parser.parse_args()

#%%
# Simulation directory
fdirs = args.simulation_file_dir
if (fdirs == "./"):
    fdirs = os.getcwd()
    os.path.join(fdirs, '')
#%%
# Input job script filename and directory
fin   = args.input_job_filename
ftoml = args.input_master_toml
fotoml = args.input_output_toml
#fsv in the preprocessing has a name like
#rve_unique_name+'_test_matrix.csv'
fcsv  = args.input_option_filename 
fdiri = args.input_job_filedir

if (fdiri == "./"):
    fdiri = os.getcwd()
    os.path.join(fdiri, '')

#%%
# Output directory
fdiro = os.path.abspath(args.output_directory)

#%%
# Job submission info
bsub_use = args.bsub_jobs

job_str = ""
script_name = ""

if bsub_use:
    job_str = "bsub "
    script_name = "batch_script.sh"
else:
    job_str = "flux mini batch --nslots=6 --cores-per-slot=40 --nodes=6 ./" + os.path.basename(fin) + "\n"
    script_name = "themis_batch.sh"

rtmodel = args.rtmodel

#%%
# Create all of the necessary symlink files and job script
fh = os.path.join(fdiri, os.path.basename(fin))

job_script = []

with open(fh, "rt") as f:
    job_script = f.readlines()
#%%
# If we want to be closer to how Themis (LLNL software that operates on top of flux and can automate a lot of this job submission/creation)
# does things then we'll need to have a csv file with our various test matrix parameters in it.
# We'll then have the headers act as a key to python dictionary of a list that contains the parameter value 
# for each test. Next, we'd want a master .toml/option file for our simulations where we have %%key_name%% 
# through out the file for the various params that need to be modified/filled in from the provided csv file.
# These parameters will be the output name for the body average or volume values text files. The name
# should compose of the unique microstructure name, temperature, and loading info. The unique microstructure name
# will have provided by the ExaCA people. The other parts we'll come up which won't be hard.
# Other params that need to be changed are the orientation, mesh, property, state, and time step file names. These will largely
# be constant for a given microstructure. The only one which will vary will be the property file, since the properties vary with temperature.
# Finally, we'll need to modify the  loading conditions. Currently, the plan is for a total of 21 tests per temperature.
# The file replacement shouldn't be too bad. Our toml file is fairly small, so we can quickly iterate through it line by line
# and just do the search for the keys and replace them with the appropriate value for each test case.
# 
# Long story short, if we're doing the real workflow and not just a test of the system we'll read in the csv
# file first before this loop. We'll then know how many tests to run.
# Then, we'll add a function call in here to generate the correct toml file for each
# simulation in here. In the end, we can keep the numeric numbering of the directories, since it's not super
# important or we can rename it as microstructure_name_temperature_incr_#. 
# Some logic will need to be added to the optimization script to walk through our work flow directory once all
# simulations are finished and group the necessary simulations together (same microstructure and same temperature). 

# Create all of the necessary symlink files and job script
fh = os.path.join(fdiri, os.path.basename(ftoml))

mtoml = []

# Read all the data in as a single string
# should make doing the regex easier
with open(fh, "rt") as f:
    mtoml = f.read()

# The csv file should be generated in the pre-processing step
# as we'll have much of the information already available there
# The csv should have the following lines:
# RVE unique name, orientation file name, # of grains
# temperature (K), property file name, # of properties
# state file name, # of state variables, essential id array,
# essential components array, essential_vals array,
# loading direction name, mesh file location
#
# Header for the should be:
# rve_unique_name, ori_file_name, ngrains, tempk,
# prop_file_loc, nprops, state_file_loc, nstates,
# ess_id_array, ess_comp_array, ess_vals_array,
# loading_name, mesh_file_loc

fh = os.path.join(fdiri, os.path.basename(fcsv))
df = pd.read_csv(fh)
nruns = df.shape[0]
headers = list(df.columns)
headers.pop(0)

avg_headers = ["avg_stress_ext", "avg_pl_work_ext", "avg_dp_tensor_ext", "avg_def_grad_ext"]
if not os.path.exists(fdiro):
  os.makedirs(fdiro)
for iDir in range(nruns):
    rve_name = df["rve_unique_name"][iDir]
    load_dir_name = df["loading_name"][iDir]
    temp_k = str(int(df["tempk"][iDir]))
    fdiron = os.path.join(fdiro, rve_name, "") 
    fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")
    if not os.path.exists(fdironl):
        os.makedirs(fdironl)
    # Create symlink
    for src in glob.glob(os.path.join(fdirs,"*")):
        fh = os.path.join(fdironl, os.path.basename(src))
        if not os.path.exists(fh):
            os.symlink(src, fh)
    toml = mtoml
    for iheader in headers:
        search = "%%" + iheader + "%%"
        repl_val = str(df[iheader][iDir])
        # This line is needed as toml parsers might get mad with just the
        # 0. and not 0.0
        repl_val = fixEssVals(repl_val)
        toml = re.sub(search, repl_val, toml)
    # Now do the avg_stress, avg_pl_work, avg_dp_tensor replacements
    # We always want these to be unique names so if they're moved somewhere
    # else we can always associate them with the correct run
    # therefore, the name contains the rve name, temperature, and loading dir name
    # frve_name+"_"+str(temp)+"_"+loading_dir_names[0]
    ext_name = rve_name +"_" + temp_k + "_" + load_dir_name
    for iheader in avg_headers:
        search = "%%" + iheader + "%%"
        replace = ext_name
        toml = re.sub(search, replace, toml)

    search = "%%rtmodel%%"
    toml = re.sub(search, rtmodel, toml)

    # Output toml file
    fh = os.path.join(fdironl, os.path.basename(fotoml))
    # Check to see if it is a symlink and if so remove the link
    if os.path.islink(fh):
        os.unlink(fh)
    # We can now safely write out the file
    with open(fh, "w") as f:
        f.write(toml)
    # Output job script file
    fh = os.path.join(fdironl, os.path.basename(fin))
    # Check to see if it is a symlink and if so remove the link
    if os.path.islink(fh):
        os.unlink(fh)
    # We can now safely write out the file
    with open(fh, "w") as f:
        f.writelines(job_script)
    os.chmod(fh, 0o775)
#%%
# Our job script portion

script = ["#! /bin/bash\n\n",
          "SCRIPT=$(readlink -f \"$0\")\n", 
          "BASE_DIR=$(dirname \"$SCRIPT\")\n\n"]

for iDir in range(nruns):
    rve_name = df["rve_unique_name"][iDir]
    load_dir_name = df["loading_name"][iDir]
    temp_k = str(int(df["tempk"][iDir]))
    fdiron = os.path.join(fdiro, rve_name, "") 
    fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")
    line = "cd " + fdironl + "\n"
    script.append(line)
    if(not bsub_use):
        script.append(job_str)
    else:
        line = job_str + " -J \"ExaConstit_" + str(iDir+1) + "\" ./" + os.path.basename(fin) + "\n"
        script.append(line)

if(not bsub_use):
    script.append("\ncd ${BASE_DIR}\n")
    script.append("flux jobs -a\n")
    script.append("flux queue drain\n")
    script.append("flux jobs -a\n")
else:
    for iDir in range(nruns):
        script.append("\nbwait -w \"done(ExaConstit_" + str(iDir+1) + ")\"")
    

fh = os.path.join(os.getcwd(), script_name)

with open(fh, "w") as f:
    f.writelines(script)         

os.chmod(fh, 0o775)
