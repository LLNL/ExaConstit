import argparse
import subprocess
import os
import glob
import re
import sys

from typing import Union
from pathlib import Path

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
# based on solutions provided on this stackoverflow question: https://stackoverflow.com/a/68817065
# for how to zip a file from python
def zip_dir(dir: Union[Path, str], filename: Union[Path, str]):
    """Zip the provided directory without navigating to that directory using `pathlib` module"""
    # Convert to Path object
    import zipfile
    from os import PathLike

    dir = Path(dir)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in dir.rglob("*"):
            rel_dir = Path(dir.name)
            rel_file = rel_dir.joinpath(entry.relative_to(dir))
            zip_file.write(entry, rel_file)

def check_for_files(dir: Union[Path, str], pattern: Union[Path, str]):
    """Check to see if a file/pattern exists in a directory and if so return True"""
    from os import PathLike
    dir = Path(dir)

    for entry in dir.rglob(pattern):
        if os.path.isfile(entry)
            return True
    return False

def zip_rm_avgs(dir: Union[Path, str], filename: Union[Path, str]):
    """Zip the provided directory without navigating to that directory using `pathlib` module"""
    # Convert to Path object
    import zipfile
    from os import PathLike

    dir = Path(dir)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in dir.rglob("avg*"):
            zip_file.write(entry, entry.relative_to(dir))

    for entry in dir.rglob("avg*"):
        entry.unlink()

def job_directories(args, output_file_dir, df, nrves):
    #%%
    # Simulation directory, binary, and modules

    rve_dirs = []
    rve_binary = []
    rve_source_module_file = []
    rve_job_nranks = []
    rve_common_dirs = []

    for irve in range(nrves):
        local_args = args.loc[irve]

        fdirs = os.path.abspath(output_file_dir)
        frve = local_args["rve_unique_name"]
        fdir_rve = os.path.join(fdirs, frve, "")
        fdirc = os.path.abspath(local_args["common_file_directory"])
        fdiro = fdir_rve

        rve_common_dirs.append(fdirc)

        nruns = df[irve].shape[0]
        headers = list(df[irve].columns)
        headers.pop(0)
        sub_rve_dirs = []

        rb = []
        smf = []
        jnr = []

        for iDir in range(nruns):
            rve_name = df[irve]["rve_unique_name"][iDir]
            load_dir_name = df[irve]["loading_name"][iDir]
            temp_k = str(int(df[irve]["tempk"][iDir]))
            fdiron = fdiro
            fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")
            sub_rve_dirs.append(fdironl)

            rb.append(local_args["exaconstit_binary"])
            smf.append(local_args["exaconstit_module_source_file"])
            jnr.append(local_args["rve_job_num_ranks"])

        rve_binary.append(rb.copy())
        rve_source_module_file.append(smf.copy())
        rve_job_nranks.append(jnr.copy())
        rve_dirs.append(sub_rve_dirs.copy())
    
    return (rve_dirs, rve_common_dirs, rve_binary, rve_source_module_file, rve_job_nranks)

def job_scripts_entk_rerun(tasks, tasks_map, output_file_dir, job_info):
    print("in entk rerun")
    from entk_wf import UQWF, re, rp
    from shutil import rmtree

    TASK_ATTRS_BASE = ['executable', 'arguments', 'pre_exec',
                       'cpu_reqs', 'gpu_reqs', 'sandbox']

    task_failed = []
    
    for task in tasks:
        if task.state == rp.FAILED:
            # task_failed.append(task)
            t_new = re.Task()
            for attr in TASK_ATTRS_BASE:
                t_new[attr] = task[attr]
            task_failed.append(t_new)
            # num_ranks += task.cpu_reqs["cpu_processes"]
            sandbox_fdir = os.path.join(tasks_map[task.uid], 'tmp', '')
            sandbox_zip = os.path.join(tasks_map[task.uid], 'tmp.zip')
            zip_dir(sandbox_fdir, sandbox_zip)
            tasks_map[t_new.uid] = sandbox_fdir
            rmtree(sandbox_fdir)
            sim_avgs_zip = os.path.join(tasks_map[task.uid], 'sim_avg_vals.zip')
            zip_rm_avgs(tasks_map[task.uid], sim_avgs_zip)

    if (len(task_failed) == 0):
        return
    
    job_node_cpus, job_node_gpus, job_max_nodes, job_walltime, task_max_nodes = job_info 
    # Could probably try and figure out a smart way of doing things here but...
    job_nodes = task_max_nodes * len(task_failed)
    if (job_nodes > job_max_nodes):
        job_nodes = job_max_nodes

    wf = UQWF(walltime=int(job_walltime), cpus=job_nodes * job_node_cpus, gpus=job_nodes * job_node_gpus, num_nodes=job_nodes)
    wf.stage.add_tasks(task_failed)
    wf.run()  # run the execution of tasks (within one stage)

    task_fail = {"Failed" : []}
    for task in task_failed:
        if task.state == rp.FAILED:
            task_fail["Failed"].append(tasks_map[task.uid])

    if (len(task_fail["Failed"]) > 0):

        import pandas as pd
        df = pd.DataFrame(task_fail)
        fhdf = os.path.join(output_file_dir, os.path.basename('task_fail_location.csv'))
        df.to_csv(fhdf)
        
        raise ValueError("One or more jobs failed during the retry run. See task failure csv for which ones")

def job_scripts_entk(args, output_file_dir, df):

    from entk_wf import UQWF, re, rp
    from shutil import rmtree
    
    THREADS_PER_RANK = 7

    tasks = []

    nrves = len(args.index)
    job_num_nodes  = int(args["job_num_nodes"][0])
    job_walltime   = int(args["job_walltime"][0])
    job_node_cpus  = int(args["job_node_cpus"][0])
    job_node_gpus  = int(args["job_node_gpus"][0])

    job_rve_max_nodes = int(args["rve_job_num_nodes"].max())
    job_max_nodes_fail      = int(args["job_max_nodes_fail"][0])
    job_max_walltime_fail   = int(args["job_max_walltime_fail"][0])
    
    rve_dirs, _, rve_binary, rve_source_module_file, rve_job_nranks = \
        job_directories(args, output_file_dir, df, nrves)

    tasks_map = {}
    
    for irve in range(nrves):
        for sidx, fdironl in enumerate(rve_dirs[irve]):
            # we could also hash fdironl and use that as the uid and from there
            # we would be able to query what things failed.
            # Alternatively, we grab the uid after the creation of things query which items
            # failed if any after things were run and work with a subset of things and rerun
            # those failed examples

            # Check to see if we already have an existing tmp sandbox file if so
            # zip up the old one and then remove the folder
            sandbox_fdir = os.path.join(fdironl, 'tmp', '')
            if os.path.exists(sandbox_fdir):
                sandbox_zip = os.path.join(fdironl, 'tmp_old_run.zip')
                zip_dir(sandbox_fdir, sandbox_zip)
                rmtree(sandbox_fdir)

            tasks.append(re.Task({
                'executable': rve_binary[irve][sidx],
                'arguments' : ['-opt', 'options.toml'],
                'pre_exec'  : ['. /sw/frontier/init/profile',
                               'source %s' % rve_source_module_file[irve][sidx],
                               'cd %s'     % fdironl],
                'cpu_reqs'  : {'cpu_processes'   : rve_job_nranks[irve][sidx],
                               'cpu_threads'     : THREADS_PER_RANK,
                               'cpu_thread_type' : rp.OpenMP},
                'gpu_reqs'  : {'gpu_processes'   : 1,
                               'gpu_process_type': rp.POSIX},
                'sandbox'   : sandbox_fdir
            }))

            # Check to see if we had a previous simulation that generated the avg* files
            # if so we want to zip those old ones up and then remove them for the new
            # runs
            if check_for_files(fdironl, "avg*"):
                sim_avgs_zip = os.path.join(fdironl, 'sim_avg_vals_old_run.zip')
                zip_rm_avgs(fdironl, sim_avgs_zip)

            tasks_map[tasks[-1].uid] = fdironl

    # to configure the size of a batch job, set the following parameters
    # `walltime`, `cpus`, `gpus`; otherwise default values from the config
    # will be used. For example:
    #
    #    wf = UQWF(walltime=240, cpus=56*1000, gpus=8*1000)
    #
    # NOTE: Parameters SMT (threads per core) and CoreSpec (blocked cores)
    #       cannot be changed during the run, they are configured separately.

    # example on estimating required parameters:
    #    task_cores = num_ranks x threads_per_rank
    #    cpus       = num_tasks x task_cores
    #    walltime   = num_tasks x task_runtime
    
    wf = UQWF(walltime=int(job_walltime), cpus=job_num_nodes * job_node_cpus, gpus=job_num_nodes * job_node_gpus, num_nodes=job_num_nodes)
    wf.stage.add_tasks(tasks)
    wf.run()  # run the execution of tasks (within one stage)

    job_info = (job_node_cpus, job_node_gpus, job_max_nodes_fail, job_max_walltime_fail, job_rve_max_nodes)
    
    job_scripts_entk_rerun(tasks, tasks_map, output_file_dir, job_info)


def job_scripts_lsf(args, output_file_dir, df):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
    '''
    nrves = len(args.index)
    #%%
    # Job submission info
    job_str = "bsub "
    script_name = "batch_script.sh"

    #%%
    # Our job script portion

    # else:
    # job_str = "flux mini batch --nslots=6 --cores-per-slot=40 --nodes=6 ./" + os.path.basename(fin) + "\n"
    # script_name = "themis_batch.sh"

    rve_dirs, rve_common_dirs, rve_binary, rve_source_module_file, rve_job_nranks = job_directories(args, output_file_dir, df, nrves)

    for irve in range(nrves):
        job_script = []
        local_args = args.loc[irve]
        fin   = local_args["input_job_filename"]
        fh = os.path.join(rve_common_dirs[irve], os.path.basename(fin))
        with open(fh, "rt") as f:
            job_script = f.readlines()

        iDir = 0
        script = ["#! /bin/bash\n\n",
                  "SCRIPT=$(readlink -f \"$0\")\n", 
                  "BASE_DIR=$(dirname \"$SCRIPT\")\n\n"]
        for fdironl in rve_dirs[irve]:
            # Output job script file
            fh = os.path.join(fdironl, os.path.basename(fin))
            # Check to see if it is a symlink and if so remove the link
            if os.path.islink(fh):
                os.unlink(fh)
            # We can now safely write out the file
            with open(fh, "w") as f:
                f.writelines(job_script)
            os.chmod(fh, 0o775)

            line = "cd " + fdironl + "\n"
            script.append(line)
            line = job_str + " -J \"ExaConstit_" + str(iDir+1) + "\" ./" + os.path.basename(fin) + "\n"
            script.append(line)
            script.append("\nbwait -w \"done(ExaConstit_" + str(iDir+1) + ")\"\n")
            iDir += 1

        fdirs = os.path.abspath(output_file_dir)
        frve = local_args["rve_unique_name"]
        fdir_rve = os.path.join(fdirs, frve, "")
        fscript = os.path.join(fdir_rve, script_name)
        with open(fscript, "w") as f:
            f.writelines(script)
        os.chmod(fscript, 0o775)
        iDir = 0

        cmd = "sh " + fscript
        print("Job script ran: " + cmd)
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)

def job_scripts_slurm(args, output_file_dir, df):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
    '''
    nrves = len(args.index)
    #%%
    # Job submission info
    job_str = "bsub "
    script_name = "batch_script.sh"

    #%%
    # Our job script portion

    # else:
    # job_str = "flux mini batch --nslots=6 --cores-per-slot=40 --nodes=6 ./" + os.path.basename(fin) + "\n"
    # script_name = "themis_batch.sh"

    rve_dirs, rve_common_dirs, rve_binary, rve_source_module_file, rve_job_nranks = job_directories(args, output_file_dir, df, nrves)

    for irve in range(nrves):
        job_script = []
        local_args = args.loc[irve]
        fin   = local_args["input_job_filename"]
        fh = os.path.join(rve_common_dirs[irve], os.path.basename(fin))
        with open(fh, "rt") as f:
            job_script = f.readlines()

        iDir = 0
        script = ["#! /bin/bash\n\n",
                  "SCRIPT=$(readlink -f \"$0\")\n", 
                  "BASE_DIR=$(dirname \"$SCRIPT\")\n\n"]
        for fdironl in rve_dirs[irve]:
            # Output job script file
            fh = os.path.join(fdironl, os.path.basename(fin))
            # Check to see if it is a symlink and if so remove the link
            if os.path.islink(fh):
                os.unlink(fh)
            # We can now safely write out the file
            with open(fh, "w") as f:
                f.writelines(job_script)
            os.chmod(fh, 0o775)

            line = "cd " + fdironl + "\n"
            script.append(line)
            line = job_str + " -J \"ExaConstit_" + str(iDir+1) + "\" ./" + os.path.basename(fin) + "\n"
            script.append(line)
            iDir += 1

        fdirs = os.path.abspath(output_file_dir)
        frve = local_args["rve_unique_name"]
        fdir_rve = os.path.join(fdirs, frve, "")
        fscript = os.path.join(fdir_rve, script_name)
        with open(fscript, "w") as f:
            f.writelines(script)
        os.chmod(fscript, 0o775)
        iDir = 0

        cmd = "sh " + fscript
        print("Job script ran: " + cmd)
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        
def job_scripts_flux_cli(args, output_file_dir, df):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
    '''
    nrves = len(args.index)
    #%%
    rve_dirs, rve_common_dirs = job_directories(args, output_file_dir, df, nrves)

    for irve in range(nrves):
        job_script = []
        local_args = args.loc[irve]
        fin   = local_args["input_job_filename"]

        # Job submission info
        job_str = "flux mini batch --nslots=6 --cores-per-slot=40 --nodes=6 ./" + os.path.basename(fin) + "\n"
        script_name = "themis_batch.sh"

        fh = os.path.join(rve_common_dirs[irve], os.path.basename(fin))
        with open(fh, "rt") as f:
            job_script = f.readlines()

        iDir = 0
        script = ["#! /bin/bash\n\n",
                  "SCRIPT=$(readlink -f \"$0\")\n", 
                  "BASE_DIR=$(dirname \"$SCRIPT\")\n\n"]

        for fdironl in rve_dirs[irve]:
            # Output job script file
            fh = os.path.join(fdironl, os.path.basename(fin))
            # Check to see if it is a symlink and if so remove the link
            if os.path.islink(fh):
                os.unlink(fh)
            # We can now safely write out the file
            with open(fh, "w") as f:
                f.writelines(job_script)
            os.chmod(fh, 0o775)

            line = "cd " + fdironl + "\n"
            script.append(line)
            script.append(job_str)
    
        script.append("\ncd ${BASE_DIR}\n")
        script.append("flux jobs -a\n")
        script.append("flux queue drain\n")
        script.append("flux jobs -a\n")

        fdirs = os.path.abspath(output_file_dir)
        frve = local_args["rve_unique_name"]
        fdir_rve = os.path.join(fdirs, frve, "")
        fscript = os.path.join(fdir_rve, script_name)
        with open(fscript, "w") as f:
            f.writelines(script)
        os.chmod(fscript, 0o775)

        cmd = "sh " + fscript
        print("Job script ran: " + cmd)
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
