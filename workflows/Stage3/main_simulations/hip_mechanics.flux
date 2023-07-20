#! /bin/bash

echo ${FLUX_URI}

flux resource list
#Below is something on the scale that we'd use for the full scale simulation
#we'd run for the challenge problem
#flux mini run -N 6 -n 24 -g 1 -o gpu-affinity=per-task -o mpi=spectrum ./mechanics -opt voce_fepx.toml
flux mini run -N 1 -n 6 -g 1 -o gpu-affinity=per-task -o mpi=spectrum ${exaconstit_build_dir}/bin/mechanics -opt options.toml
flux resource list
