# Workflow Stage
Prerequisites needed to run everything:
- Flux needs to be installed on the machine
- Python 3 for the job generation script
- ExaConstit's main binary `mechanics` needs to be built
- The preprocessing script should have been built or you need to point to existing data

In order to get everything to run on the CORAL machine, you need to run the `job_cli.py` file in order to create the necessary job scripts and prepare the run directories for all the different simulations. The preprocessing script generates the necessary test matrix file which this script reads in through the iofile option. From this test matrix file, the provided master toml file is used to generate the appropriate option file that each individual simulation will use to run.

The cli script can be run like:

```
python3 ./job_cli.py --help

python3 ./job_cli.py -sdir ./ -odir ./../workflow_runs/ -imtfile options_master.toml -iotfile options.toml -ijfile hip_mechanics.flux -ijfd ./ -iofile options.csv
```

The actual LSF job script is in `flux.sh` if you're using Flux. So, you can then just submit that script in order to run everything.

