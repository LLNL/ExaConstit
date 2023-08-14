### LSF syntax
#BSUB -nnodes 20                  #number of nodes
#BSUB -W 15                       #walltime in minutes
#BSUB -G MAT190                   #account
#BSUB -e err_flux.txt             #stderr
#BSUB -o out_flux.txt             #stdout
#BSUB -J flux_exaconstit          #name of job
#BSUB -q pbatch                   #queue to use
#BSUB -core_isolation 2

# Above BSUB was primarily from LLNL's bsub submission.
# It might need some slight adjustments.
# Although, I did modify the account so that should at least be correct

# LLNL's module loads for Lassen another CORAL1 machine
# module use /usr/global/tools/flux/blueos_3_ppc64le_ib/modulefiles
# Summit's module load
module use /sw/summit/modulefiles/ums/gen007flux/Core
module load pmi-shim flux
# Might need to modify this on Summit. I know this is needed on Lassen to get things to work.
# module load hwloc/1.11.10-cuda

# For the challenge problem we'd probably want to use a large allocation on the order of
# 360+, which gets us on the order of 60+ concurrent simulations at a time if
# we end up needing 24 GPUs per simulation. 
# I've tested it on Lassen with 120 nodes with the challenge problem size mesh
# I did this over Summit, since I was noting a large regression on Lassen at the time where Lassen
# was roughly 20% faster on its 6 nodes of 4 GPUs per node over a comparable 4 nodes of Summit.
# NUM_NODES=120

PMIX_MCA_gds="^ds12,ds21" jsrun -a 1 -c ALL_CPUS -g ALL_GPUS -n ${NUM_NODES} --bind=none --smpiargs="-disable_gpu_hooks" flux start -o,-Slog-filename=out ./themis_batch.sh

