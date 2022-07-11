import os

import flux
import flux.job
import flux.resource


def main():
    fh = flux.Flux()
    jobspec = flux.job.JobspecV1.from_command(
        ["%%binary%%", "-opt", os.path.join(os.getcwd(), "options.toml")],
        num_nodes=%%nnodes%%,
        num_tasks=%%ntasks%%,
        gpus_per_task=%%ngpus%%
    )
    jobspec.setattr_shell_option("mpi", "spectrum")
    jobspec.setattr_shell_option("gpu-affinity", "per-task")
    jobspec.stdout = "%%output_name%%"  # TODO: fill in
    jobspec.stderr = "%%error_name%%"
    jobspec.environment = dict(os.environ)
    jobspec.cwd = os.getcwd()
    jobspec.duration = %%timeout%%
    flux.job.wait(fh, flux.job.submit(fh, jobspec, waitable=True))


if __name__ == '__main__':
    main()