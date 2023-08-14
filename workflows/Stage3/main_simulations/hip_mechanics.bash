### LSF syntax
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 1:00                     #walltime in minutes
#BSUB -P MAT190                   #account
#BSUB -e exa_error.txt            #stderr
#BSUB -o exa_out.txt              #stdout
#BSUB -J ExaConstit               #name of job
#BSUB -alloc_flags smt1

case $WORKFLOW_BUILDER_BACKEND in
    serial)
        NRANK=42
        NGPU=0
        ;;
    cuda)
        NRANK=6
        NGPU=1
        ;;
esac

date
jsrun -n ${NRANK} -c 1 -g ${NGPU} \
    ${exaconstit_build_dir}/bin/mechanics -opt options.toml
