import os
import os.path
import subprocess, shlex
import sys
import numpy as np
from ExaConstit_Logger import write_ExaProb_log


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def map_custom(problem, igeneration, genes):
    """
    Probably won't be as efficient as just doing it the current map way
    but this should allow us to repeat this process more or less in other areas and
    have something that is like what the other mapping functions might do.
    """

    status = []

    run_exaconstit = "mpiexec -np {ncpus} {mechanics} -opt {toml_name}".format(
        ncpus=problem.ncpus, mechanics=problem.bin_mechanics, toml_name="options.toml"
    )

    f_objective = []

    # Preprocess all of the genes first
    for igene, gene in enumerate(genes):
        problem.preprocess(gene, igeneration, igene)

    # Run all of the gene data next
    for igene, gene in enumerate(genes):
        istatus = []
        for iobj in range(problem.n_obj):
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(problem.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")
            # cd into directory and run command and then when this code block exits it returns us
            # to the working directory
            with cd(fdironl):
                print("Running: " + rve_name)
                output = os.path.join(fdironl + "run_output.txt")
                with open(output, "w") as f:
                    try:
                        run_exaconstit_split = shlex.split(run_exaconstit)
                        p = subprocess.Popen(
                            run_exaconstit_split, start_new_session=True, stdout=f
                        )
                        returncode = p.wait(timeout=problem.timeout[iobj])
                    except subprocess.TimeoutExpired:
                        print(
                            f"Timeout for {run_exaconstit} ({problem.timeout[iobj]}s) expired",
                            file=sys.stderr,
                        )
                        p.terminate()
                        returncode = 143
                    except KeyboardInterrupt:
                        try:
                            p.terminate()
                            sys.exit("ctrl-c interrupt")
                        except:
                            p.kill()
                            sys.exit("sent SIGKILL to mpi call as terminate failed...")

                    istatus.append(returncode)
        status.append(istatus)

    # Post-process all of the data last
    for igene, gene in enumerate(genes):
        f = problem.postprocess(igeneration, igene, status[igene])
        f_objective.append(np.copy(f))

    return f_objective


# Will want a custom way to handle one off launches for failed tests
def map_custom_fail(problem, igeneration, gene, igene):
    """
    Probably won't be as efficient as just doing it the current map way
    but this should allow us to repeat this process more or less in other areas and
    have something that is like what the other mapping functions might do.
    """

    status = []

    run_exaconstit = "mpirun -np {ncpus} {mechanics} -opt {toml_name}".format(
        ncpus=problem.ncpus, mechanics=problem.bin_mechanics, toml_name="options.toml"
    )

    f_objective = []

    # Preprocess all of the genes first
    problem.preprocess(gene, igeneration, igene)

    # Run all of the gene data next
    istatus = []
    for iobj in range(problem.n_obj):
        gene_dir = "gen_" + str(igeneration)
        fdir = os.path.join(problem.workflow_dir, gene_dir, "")
        rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
        fdironl = os.path.join(fdir, rve_name, "")
        # cd into directory and run command and then when this code block exits it returns us
        # to the working directory
        with cd(fdironl):
            print("Running: " + rve_name)
            output = os.path.join(fdironl + "run_output.txt")
            with open(output, "w") as f:
                try:
                    run_exaconstit_split = shlex.split(run_exaconstit)
                    p = subprocess.Popen(
                        run_exaconstit_split, start_new_session=True, stdout=f
                    )
                    returncode = p.wait(timeout=problem.timeout[iobj])
                except subprocess.TimeoutExpired:
                    print(
                        f"Timeout for {run_exaconstit} ({problem.timeout[iobj]}s) expired",
                        file=sys.stderr,
                    )
                    p.terminate()
                    returncode = 143
                except KeyboardInterrupt:
                    try:
                        p.terminate()
                        sys.exit("ctrl-c interrupt")
                    except:
                        p.kill()
                        sys.exit("sent SIGKILL to mpi call as terminate failed...")
                istatus.append(returncode)
    status.append(istatus)

    # Post-process all of the data last
    f = problem.postprocess(igeneration, igene, status[0])
    f_objective.append(np.copy(f))

    return f_objective
