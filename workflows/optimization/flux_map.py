import os
import os.path
import glob
import sys
import re
import numpy as np

import flux
import flux.job
import flux.job.stats
import flux.job.info
import flux.resource

from ExaConstit_Logger import write_ExaProb_log
from ExaConstit_Problems import cd


def map_custom(problem, igeneration, genes):    
    status = []
    f_objective = []
    flux_handle = flux.Flux()

    machine_name = os.uname().nodename
    spectrum_machine = False
    if ("lassen" in machine_name or "sierra" in machine_name or "ansel" in machine_name or "summit" in machine_name or "andes" in machine_name):
        spectrum_machine = True

    # Preprocess all of the genes first
    for igene, gene in enumerate(genes):
        problem.preprocess(gene, igeneration, igene)

    # Submit all of the flux jobs
    jobids = []
    jobnames = []
    for iobj in range(problem.n_obj - 1, -1, -1):
        for igene, gene in enumerate(genes):
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(problem.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")

            ngpus = None
            if problem.ngpus[iobj] > 0:
                ngpus = 1

            jobspec = flux.job.JobspecV1.from_command(
                [problem.bin_mechanics, "-opt", "options.toml"],
                num_nodes=problem.nnodes[iobj],
                num_tasks=problem.ncpus[iobj],
                cores_per_task=1,
                gpus_per_task=ngpus
            )

            jobspec.cwd = fdironl
            jobspec.stdout = "flux_output.txt"
            jobspec.stderr = "flux_error.txt"
            jobspec.environment = dict(os.environ)
            if spectrum_machine:
                jobspec.setattr_shell_option("mpi", "spectrum")
            jobspec.setattr_shell_option("gpu-affinity", "per-task")
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
            jobspec.duration = int(problem.timeout[iobj])
            jobids.append(flux.job.submit(flux_handle, jobspec, waitable=True))
            jobnames.append(rve_name)

    # Wait on all of our flux jobs to finish
    stats = flux.job.stats.JobStats(flux_handle)
    for j in jobids:
        write_ExaProb_log(f"[running: {stats.run}, pending: {stats.pending}, run/complete: {stats.running}]", "info")
        jobid, istatus, errnum = flux.job.wait(flux_handle, j)
        stats.update_sync()
        if not istatus:
            status.append(errnum)
        else:
            status.append(0)

    # Summit jobs might produce spurious core dumps at end of simulation
    # due to some weird issues regarding flux's MPI library finalize function
    # We don't have this issue when just using jsruns instead...
    # Nonetheless, we want to delete these to save disk space
    for igene, gene in enumerate(genes):
        for iobj in range(problem.n_obj):
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(problem.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")
            with cd(fdironl):
                for f in glob.glob("core*"):
                    os.remove(f)
    # Get info about the job like runtime, status, and errnum
    jobs = flux.job.JobList(flux_handle, ids=jobids).jobs()
    for job, name in zip(jobs, jobnames):
        write_ExaProb_log(f"RVE name: {name}, flux id: {job.id}, runtime: {job.runtime:12.3f}s, status: {job.status}, errnum: {job.returncode}", "info")
    # Convert status back into a 2d array
    status = [
        status[i : i + len(genes)] for i in range(0, len(status), len(genes))
    ]
    # Post-process all of the data last
    for igene, gene in enumerate(genes):
        stats = [status[i][igene] for i in range(problem.n_obj)]
        f = problem.postprocess(igeneration, igene, stats)
        f_objective.append(np.copy(f))

    return f_objective


# Will want a custom way to handle one off launches for failed tests
def map_custom_fail(problem, igeneration, gene, igene):
    status = []
    flux_handle = flux.Flux()

    # Preprocess all of the genes first
    problem.preprocess(gene, igeneration, igene, True)

    machine_name = os.uname().nodename
    spectrum_machine = False
    if ("lassen" in machine_name or "sierra" in machine_name or "ansel" in machine_name or "summit" in machine_name or "andes" in machine_name):
        spectrum_machine = True
    # Submit all of the flux jobs
    jobids = []
    jobnames = []
    istatus = []
    for iobj in range(problem.n_obj):
        gene_dir = "gen_" + str(igeneration)
        fdir = os.path.join(problem.workflow_dir, gene_dir, "")
        rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
        fdironl = os.path.join(fdir, rve_name, "")

        ngpus = None
        if problem.ngpus[iobj] > 0:
            ngpus = 1

        jobspec = flux.job.JobspecV1.from_command(
            [problem.bin_mechanics, "-opt", "options.toml"],
            num_nodes=problem.nnodes[iobj],
            num_tasks=problem.ncpus[iobj],
            cores_per_task=1,
            gpus_per_task=ngpus
        )

        jobspec.cwd = fdironl
        jobspec.stdout = "flux_output.txt"
        jobspec.stderr = "flux_error.txt"
        jobspec.environment = dict(os.environ)
        if spectrum_machine:
            jobspec.setattr_shell_option("mpi", "spectrum")
        jobspec.setattr_shell_option("gpu-affinity", "per-task")
        jobspec.setattr_shell_option("cpu-affinity", "per-task")
        jobspec.duration = int(problem.timeout[iobj])
        jobids.append(flux.job.submit(flux_handle, jobspec, waitable=True))
        jobnames.append(rve_name)

    # Wait on all of our flux jobs to finish
    stats = flux.job.stats.JobStats(flux_handle)
    for j in jobids:
        write_ExaProb_log(f"[running: {stats.run}, pending: {stats.pending}, run/complete: {stats.running}]", "info")
        jobid, istatus, errnum = flux.job.wait(flux_handle, j)
        stats.update_sync()
        if not istatus:
            status.append(errnum)
        else:
            status.append(0)

    # Summit jobs might produce spurious core dumps at end of simulation
    # due to some weird issues regarding flux's MPI library finalize function
    # We don't have this issue when just using jsruns instead...
    # Nonetheless, we want to delete these to save disk space
    for iobj in range(problem.n_obj):
        gene_dir = "gen_" + str(igeneration)
        fdir = os.path.join(problem.workflow_dir, gene_dir, "")
        rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
        fdironl = os.path.join(fdir, rve_name, "")
        with cd(fdironl):
            for f in glob.glob("core*"):
                os.remove(f)

    # Convert status back into a 2d array
    status = [
        status[i : i + problem.n_obj] for i in range(0, len(status), problem.n_obj)
    ]

    # Post-process all of the data last
    f = problem.postprocess(igeneration, igene, status[0], True)
    f_objective = np.copy(f)

    return f_objective
