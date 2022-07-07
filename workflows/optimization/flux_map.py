import os
import os.path
import subprocess
import sys
import numpy as np

import flux
import flux.job

from ExaConstit_Logger import write_ExaProb_log


def map_custom(problem, igeneration, genes):

    status = []
    f_objective = []
    fh = flux.Flux()

    # Preprocess all of the genes first
    for igene, gene in enumerate(genes):
        problem.preprocess(gene, igeneration, igene)

    # Submit all of the flux jobs
    jobids = []
    for igene, gene in enumerate(genes):
        istatus = []
        for iobj in range(problem.n_obj):
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(problem.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")

            flux_obj = self.job_script
            fh_output = os.path.join(fdironl, "optimization_out.txt")
            fh_error  = os.path.join(fdironl, "optimization_err.txt")
            modifiers = {"binary":problem.bin_mechanics, "nnodes":problem.nnodes, "nntasks":problem.npus, "ngpus":problem.ngpus,
            "output_name":fh_output, "error_name":fh_error}
            for iheader, repl_val in modifiers:
                search = "%%" + iheader + "%%"
                flux_obj = re.sub(search, repl_val, flux_obj)
            # Output toml file
            fh = os.path.join(fdironl, os.path.basename("mechanics.flux"))
            # Check to see if it is a symlink and if so remove the link
            if os.path.islink(fh):
                os.unlink(fh)
            # We can now safely write out the file
            with open(fh, "w") as f:
                f.write(flux_obj)

            jobspec = flux.job.JobSpecV1.from_nest_command(
                [fh],
                num_slots=problem.nnodes,
                cores_per_slot=problem.ncpus,
                num_nodes=problem.nnodes,
            )

            jobspec.cwd = fdironl
            jobspec.stdout = fh_output
            jobspec.stderr = fh_error
            jobids.append(flux.job.submit(fh, jobspec, waitable=True))

    # Wait on all of our flux jobs to finish
    print(flux.job.job_list(fh))
    for j in jobids:
        jobid, istatus, errnum = flux.job.wait(fh, j)
        if not istatus:
            status.append(errnum)
        else:
            status.append(0)
    print(flux.job.job_list(fh))

    # Convert status back into a 2d array
    status = [
        status[i : i + problem.n_obj] for i in range(0, len(status), problem.n_obj)
    ]
    # Post-process all of the data last
    for igene, gene in enumerate(genes):
        f = problem.postprocess(igeneration, igene, status[igene])
        f_objective.append(np.copy(f))

    return f_objective


# Will want a custom way to handle one off launches for failed tests
def map_custom_fail(problem, igeneration, gene, igene):
    status = []
    f_objective = []
    fh = flux.Flux()

    # Preprocess all of the genes first
    problem.preprocess(gene, igeneration, igene)

    # Submit all of the flux jobs
    jobids = []

    istatus = []
    for iobj in range(problem.n_obj):
        gene_dir = "gen_" + str(igeneration)
        fdir = os.path.join(problem.workflow_dir, gene_dir, "")
        rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
        fdironl = os.path.join(fdir, rve_name, "")

        flux_obj = self.job_script
        fh_output = os.path.join(fdironl, "optimization_out.txt")
        fh_error  = os.path.join(fdironl, "optimization_err.txt")
        modifiers = {"binary":problem.bin_mechanics, "nnodes":problem.nnodes, "nntasks":problem.npus, "ngpus":problem.ngpus,
        "output_name":fh_output, "error_name":fh_error}
        for iheader, repl_val in modifiers:
            search = "%%" + iheader + "%%"
            flux_obj = re.sub(search, repl_val, flux_obj)
        # Output toml file
        fh = os.path.join(fdironl, os.path.basename("mechanics.flux"))
        # Check to see if it is a symlink and if so remove the link
        if os.path.islink(fh):
            os.unlink(fh)
        # We can now safely write out the file
        with open(fh, "w") as f:
            f.write(flux_obj)

        jobspec = flux.job.JobSpecV1.from_nest_command(
            [fh],
            num_slots=problem.nnodes,
            cores_per_slot=problem.ncpus,
            num_nodes=problem.nnodes,
        )

        jobspec.cwd = fdironl
        jobspec.stdout = fh_output
        jobspec.stderr = fh_error
        jobids.append(flux.job.submit(fh, jobspec, waitable=True))

    # Wait on all of our flux jobs to finish
    print(flux.job.job_list(fh))
    for j in jobids:
        jobid, istatus, errnum = flux.job.wait(fh, j)
        if not istatus:
            status.append(errnum)
        else:
            status.append(0)
    print(flux.job.job_list(fh))

    # Convert status back into a 2d array
    status = [
        status[i : i + problem.n_obj] for i in range(0, len(status), problem.n_obj)
    ]

    # Post-process all of the data last
    f = problem.postprocess(igeneration, igene, status[0])
    f_objective.append(np.copy(f))

    return f_objective
