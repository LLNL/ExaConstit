#! /bin/bash

echo ${FLUX_URI}

#SCRIPT=$(readlink -f "$0")
#BASE_DIR=$(dirname "$SCRIPT")

#cd ${BASE_DIR}/runs/1/
#Below is something on the scale that we'd use for the full scale simulation
#we'd run for the challenge problem
#flux mini batch --nslots=6 --cores-per-slot=40 --nodes=6 ./hip_mechanics.flux
flux mini batch --nslots=1 --cores-per-slot=40 --nodes=1 ./hip_mechanics.flux
#cd ${BASE_DIR}/runs/2/
#flux mini batch -N 1 -c 40 -n 1 ./hip_mechanics.flux

#cd ${BASE_DIR}
flux jobs -a
flux queue drain
flux jobs -a
