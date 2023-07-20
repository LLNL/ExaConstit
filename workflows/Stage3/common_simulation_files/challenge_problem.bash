### LSF syntax
#BSUB -nnodes 4                   #number of nodes
#BSUB -W 2:00                     #walltime in minutes
#BSUB -P MAT190                   #account
#BSUB -e exa_error.txt            #stderr
#BSUB -o exa_out.txt              #stdout
#BSUB -J ExaConstit               #name of job
#BSUB -alloc_flags smt1

module load gcc/7.5.0 cmake/3.18.4 git/2.31.1 cuda/11.4.2

NRANK=24
EXACONSTIT_BINARY=/gpfs/alpine/world-shared/mat190/exaconstit/exaconstit-mechanics-v0_6_2

date

