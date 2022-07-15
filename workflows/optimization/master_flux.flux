import os
import glob

import flux
import flux.job
import flux.resource


def main():
    fh = flux.Flux()
    ngpus = %%ngpus%%
    gpt = 0
    if ngpus > 0:
        gpt = 1
    jobspec = flux.job.JobspecV1.from_command(
        ["%%binary%%", "-opt", os.path.join(os.getcwd(), "options.toml")],
        num_nodes=%%nnodes%%,
        num_tasks=%%ntasks%%,
        gpus_per_task=gpt,
        cores_per_task=1
    )
    jobspec.setattr_shell_option("mpi", "spectrum")
    jobspec.setattr_shell_option("gpu-affinity", "per-task")
    jobspec.setattr_shell_option("cpu-affinity", "per-task")
    jobspec.stdout = "%%output_name%%"  # TODO: fill in
    jobspec.stderr = "%%error_name%%"
    jobspec.environment = dict(os.environ)
    jobspec.cwd = os.getcwd()
    jobspec.duration = %%timeout%%
    flux.job.wait(fh, flux.job.submit(fh, jobspec, waitable=True))
    # Summit jobs might produce spurious core dumps at end of simulation
    # due to some weird issues regarding flux's MPI library finalize function
    # We don't have this issue when just using jsruns instead...
    # Nonetheless, we want to delete these to save disk space
    for f in glob.glob("core*"):
        os.remove(f)

if __name__ == '__main__':
    main()