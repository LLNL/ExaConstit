import numpy as np
import pandas as pd
import os
import os.path
import subprocess
import sys
import glob
import re
import pandas as pd
from ExaConstit_MatGen import Matgen
from ExaConstit_Logger import write_ExaProb_log
from Smooth_SS_generator.smoothening_ss_data_fcn import smooth_stress_strain_data

class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def fixEssVals(repl_val):
    repl_val = re.sub("\n+", "", repl_val)
    repl_val = " ".join(repl_val.split())
    repl_val = repl_val.replace("0. ", "0.0")
    repl_val = repl_val.replace("0.000000e+00", "0.0")
    return repl_val

class ExaProb:

    """
    This is the constructor of the objective function evaluation
    All the assigned files must have same length with the n_obj
    (for each obj function we need a different Experiment data set etc.)
    for loc_file and loc_mechanics, give absolute paths
    """

    # Eventually will also be good to have a way to set additional parameters and cut down
    # on input list here aka we could pass in a data frame that has all the input options per simulation.
    # In other words, we might have a number of columns such as temperature, number of steps,
    # time step file, mesh file, orientation file, experimental input files, and maybe others.
    def __init__(
        self,
        n_dep=None,
        dep_unopt=None,
        nnodes=[1],
        ncpus=[2],
        ngpus=[0],
        test_dataframe=None,
        bin_mechanics="~/ExaConstit/ExaConstit/build/bin/mechanics",
        sim_input_file_dir="./input_files/",
        master_toml_file="./master_options.toml",
        workflow_dir="./wf_files",
        job_script_file=None,
    ):
        """
        Initializes class
        Inputs:
        n_dep - optional variable (fix me: more info needed)
        dep_unopt - optional variable (fix me: more info needed)
        nnodes - number of nodes each simulation should make use of
        ncpus - number of CPUs each simulation should make use of
        ngpus - number of GPUs each simulation should make use of
        test_dataframe = data frame that contains all of the experimental file info, 
                         simulation steps to take, simulation parameters, and other useful info
                         to run simulation. The number of rows ignoring the headers should be equal
                         to number objective functions were evaluating for    
        bin_mechanics - binary location of ExaConstit
        sim_input_file_dir - directory of the simulation input files
        master_toml_file - location of master toml file which all option files will be derived from
        workflow_dir - directory that we would like all of our workflow to live in (generates dir for every iteration, gene, and exp observation)
        job_script_file - optional if provided is the location of a job script that all directories should have available
        """
        # Where possible make sure to use absolute path for things
        if test_dataframe is None:
            write_ExaProb_log(
                "Input test_dataframe was provided None type and this isn't allowed",
                type="error",
                changeline=True,
            )
            sys.exit()
        self.test_dataframe = test_dataframe
        self.n_obj = len(test_dataframe)
        self.n_dep = n_dep
        self.dep_unopt = dep_unopt
        # Could probably have this in the test data frame as well if we wanted per
        # objective to use different resources...
        self.nnodes = nnodes
        self.ncpus = ncpus
        self.ngpus = ngpus
        self.timeout = test_dataframe["timeout"]
        self.bin_mechanics = os.path.abspath(bin_mechanics)
        self.master_toml_file = os.path.abspath(master_toml_file)
        self.exper_input_files = test_dataframe["experiments"]
        self.desired_strain = test_dataframe["desired_strain"]
        self.strain_rate = test_dataframe["strain_rate"]
        self.min_max_strain = test_dataframe["minmax_strain"]
        self.sim_input_file_dir = os.path.abspath(sim_input_file_dir)
        self.workflow_dir = os.path.abspath(workflow_dir)
        self.eval_cycle = 0
        self.runs = 0
        self.flag = []

        # Not all workflow managers might need a job script
        # If one is provided then we need to get out the location of that file
        # the file itself, and finally the basename of that file
        if job_script_file is not None:
            self.job_script_file = os.path.abspath(job_script_file)
            self.job_script = []
            with open(self.job_script_file, "rt") as f:
                self.job_script = f.read()
            self.job_script_name = os.path.basename(job_script_file)
        else:
            self.job_script_file = None
            self.job_script = None
            self.job_script_name = None

        self.mtoml = []
        # Read all the data in as a single string
        # should make doing the regex easier
        with open(self.master_toml_file, "rt") as f:
            self.mtoml = f.read()

        self.rtmodel = []
        for i in range(self.n_obj):
            if ngpus[i] > 0:
                self.rtmodel.append("CUDA")
            else:
                self.rtmodel.append("CPU")

        for iexpt in range(len(self.exper_input_files)):
            self.exper_input_files[iexpt] = os.path.abspath(
                self.exper_input_files[iexpt]
            )

        # Check if we have as many files as the objective functions
        if len(self.dep_unopt) != len(self.exper_input_files):
            write_ExaProb_log(
                'The length of "{}" is not equal to len(exper_input_files)={}'.format(
                    name, len(exper_input_files)
                ),
                type="error",
                changeline=True,
            )
            sys.exit()

        # Read Experiment data sets and save to S_exp
        self.s_exp = []
        for file, k in zip(self.exper_input_files, range(self.n_obj)):
            try:
                s_exp_data = np.loadtxt(file, dtype="float", ndmin=2)
                ind = np.argmax(np.abs(s_exp_data[:, 1]) > np.abs(self.desired_strain[k]))
                s_exp_data = s_exp_data[0 : ind + 1, :]
            except:
                write_ExaProb_log(
                    "{file} was not found!".format(file=file),
                    type="error",
                    changeline=True,
                )
                sys.exit()

            # Assuming that each experiment data file has at the first column the stress values
            # s_exp will be a list that contains a numpy array corresponding to each file
            self.s_exp.append(np.copy(s_exp_data))

        self.s_sim = []
        self.flag = []

    def preprocess(self, x, igeneration, igene, failed_test=False):
        """
        This is used to preprocess everything and create the necessary directories
        which our files will live in
        """
        # Just reset this during preprocessing stage as it will be set during post-processing
        if not failed_test:
            self.s_sim = []
            self.flag = []
        # I mean these never really change so we could just make these
        # created during init if we really wanted to...
        headers = list(self.test_dataframe.columns)
        headers.pop(0)

        if not os.path.exists(self.workflow_dir):
            os.makedirs(self.workflow_dir)

        # Separate parameters into dependent (thermal) and independent (athermal) groups
        # x_group[0] will be the independent group. The rest will be the dependent groups for each objective
        if self.n_dep != None:
            self.n_ind = len(x) - len(self.Exper_input_files) * self.n_dep
            x_dep = x[self.n_ind :]
            x_group = [x[0 : self.n_ind]]
            x_group.extend(
                [x_dep[k : (k + self.n_dep)] for k in range(0, len(x_dep), self.n_dep)]
            )
        else:
            self.n_ind = len(x)
            x_group = [x[0 : self.n_ind]]

        # Count iterations and save solutions
        self.eval_cycle += 1
        write_ExaProb_log("\tEvaluation Cycle: {}".format(self.eval_cycle), "info")
        write_ExaProb_log("\tSolution: x = {}".format(x_group))

        # Create all of our work directories and such
        for iobj in range(self.n_obj):
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(self.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")
            if not os.path.exists(fdironl):
                os.makedirs(fdironl)
            # Create symlink
            for src in glob.glob(os.path.join(self.sim_input_file_dir, "*")):
                fh = os.path.join(fdironl, os.path.basename(src))
                if not os.path.exists(fh):
                    os.symlink(src, fh)
            # Create symlink for mechanics binary
            fh = os.path.join(fdironl, os.path.basename(self.bin_mechanics))
            if not os.path.exists(fh):
                os.symlink(self.bin_mechanics, fh)

            fh = os.path.join(fdironl, "avg_stress.txt")
            if os.path.exists(fh):
                os.remove(fh)

            fh = os.path.join(fdironl, "auto_dt_out.txt")
            if os.path.exists(fh):
                os.remove(fh)

            # Copying a string object so a deep copy is performed here
            toml = self.mtoml
            for iheader in headers:
                search = "%%" + iheader + "%%"
                repl_val = str(self.test_dataframe[iheader][iobj])
                # This line is needed as toml parsers might get mad with just the
                # 0. and not 0.0
                repl_val = fixEssVals(repl_val)
                toml = re.sub(search, repl_val, toml)

            search = "%%rtmodel%%"
            toml = re.sub(search, self.rtmodel[iobj], toml)

            # Output toml file
            fh = os.path.join(fdironl, os.path.basename("options.toml"))
            # Check to see if it is a symlink and if so remove the link
            if os.path.islink(fh):
                os.unlink(fh)
            if os.path.exists(fh):
                os.remove(fh)
            # We can now safely write out the file
            with open(fh, "w") as f:
                f.write(toml)

            # Create mat file: props_cp_mts.txt and use the file for multiobj if more files
            try:
                if self.dep_unopt:
                    if self.n_dep:
                        Matgen(
                            x_ind=x_group[0],
                            x_dep=x_group[iobj + 1],
                            x_dep_unopt=self.dep_unopt[iobj],
                            mts=False,
                            fdir=fdironl,
                        )
                    else:
                        Matgen(
                            x_ind=x_group[0],
                            x_dep_unopt=self.dep_unopt[iobj],
                            mts=False,
                            fdir=fdironl,
                        )
                else:
                    if self.n_dep:
                        Matgen(x_ind=x_group[0], x_dep=x_group[iobj + 1], mts=False, fdir=fdironl)
                    else:
                        Matgen(x_ind=x_group[0], mts=False, fdir=fdironl)
            except:
                text = "Unable to generate material properties using Matgen!"
                write_ExaProb_log(text, "error", changeline=True)
                sys.exit()

            # Output job script file if one was provided earlier
            if self.job_script is not None:
                fh = os.path.join(fdironl, self.job_script_name)
                # Check to see if it is a symlink and if so remove the link
                if os.path.islink(fh):
                    os.unlink(fh)
                # We can now safely write out the file
                with open(fh, "w") as f:
                    f.writelines(self.job_script)
                os.chmod(fh, 0o775)

    def postprocess(self, igeneration, igene, status, failed_test=False):
        """
        This is used to postprocess everything after the runs have completed.
        Output: the objective function(s)
        """
        # Initialize
        # We never actually make use of this so should just get rid of
        s_sim = []
        flag = -1
        f = np.zeros(self.n_obj * 2)
        write_ExaProb_log("Generation: " + str(igeneration) + " gene: " + str(igene))

        # Run k simulations. One for each objective function
        for iobj in range(self.n_obj):
            # Within this loop we could automatically generate the option file and job directory
            # We can then within here cd to the subdirectory that we generated
            # Count GA and Exaconstit iterations
            self.runs += 1
            gene_dir = "gen_" + str(igeneration)
            fdir = os.path.join(self.workflow_dir, gene_dir, "")
            rve_name = "gene_" + str(igene) + "_obj_" + str(iobj)
            fdironl = os.path.join(fdir, rve_name, "")
            # Read the simulation output
            # If output file exists and it is not empty, read stress
            output_file = os.path.join(fdironl, "avg_stress.txt")
            if os.path.exists(output_file) and os.stat(output_file).st_size != 0:
                s_sim_data = np.loadtxt(output_file, dtype="float", ndmin=2)
                # Macroscopic stress in the direction of load: 3rd column (z axis)
                # If we're loading in the y-direction or z then this may not always be true
                _s_sim = s_sim_data[:, 2]
                # We use unique so to exclude repeated values from cyclic loading steps. Is it relevant for ExaConstit?
                # I don't believe this is necessary even in the cyclic case for ExaConstit unless someone is doing
                # everything in just the elastic regime. Also, I think for the cyclic and dwell type loading conditions
                # you might need to be more careful about how things fit as it can be hard to get exactly
                # _s_sim = np.unique(_s_sim)
                # Check if data size is the same with experiment data-set in case there is a convergence issue
                auto_dt_file = os.path.join(fdironl, "auto_dt_out.txt")
                (
                    smooth_exp_stress,
                    smooth_exp_strain,
                    error_strain,
                ) = smooth_stress_strain_data(
                    self.s_exp[iobj],
                    auto_dt_file,
                    self.strain_rate[iobj],
                    self.desired_strain[iobj],
                    _s_sim.shape[0]
                )
                # It's not clear to me this is the best way to do things in the long term.
                # Although, it is fine for the time being.
                # A user might provide more experimental data then needed or they may provide
                # raw data where it hasn't been smoothed out or anything
                # Additionally, they might only care about up to a given point for their data
                # We should work on a better way of doing this for the future at least for uniaxial loading
                # One method might be take expt. strain and stress data and do a minimal amount of smoothing first
                # After smoothing we could locally do a linear interpolation between experimental stress-strain
                # data to guess what the value should be for simulation values. It won't be perfect but it might work
                # good enough. Alternatively, instead of a smoothing function we could look at doing something akin
                # to a smooth spline of the data potentially from which we would then be able to obtain the values of interest.
                if error_strain <= 0.01:
                    flag = 0  # successful
                    # Could produce a ton of logging noise
                    write_ExaProb_log("\t\tSUCCESSFULL SIMULATION!!!")
                # Check to see if error in strain values between expected and smoothed strain value
                # is more then 1%
                # Note we might want to eventually crank this percent down even more...
                # If we detect this then we say all the final stress values are 0
                # by doing this we still allow the gene to exist but we penalize it for the
                # portions that it did badly on
                elif error_strain > 0.01:
                    flag = 0  # partially successful
                    text = "Simulation has unconverged results for eval_cycle = {}: sim_data_strain = {} < exp_strain_data = {}".format(
                        self.eval_cycle, smooth_exp_strain[-1], self.s_exp[iobj][-1, 1]
                    )
                    write_ExaProb_log(text, "warning", changeline=True)
                    _s_sim = np.append(_s_sim, np.zeros(24))
                # s_sim will be a list that contains a numpy array of stress corresponding to each file
                s_sim.append([np.copy(_s_sim), np.copy(smooth_exp_strain)])
            else:
                flag = 2
                text = "Simulation did not run for eval_cycle = {}. The output file was empty or not existent!".format(
                    self.eval_cycle
                )
                write_ExaProb_log(text, "warning", changeline=True)
                self.eval_cycle = self.eval_cycle - 1
                if not failed_test:
                    self.flag.append(flag)
                else:
                    self.flag[igene]  = flag
                return

            # Evaluate the individual objective function. Will have k functions. (Normalized Root-mean-square deviation (RMSD)- 1st Moment (it is the error percentage))
            # https://en.wikipedia.org/wiki/Root-mean-square_deviation
            if self.min_max_strain is not None:
                ind_min = np.argmax(np.abs(smooth_exp_strain[:]) > np.abs(self.min_max_strain[iobj][0])) - 1
                if ind_min < 0:
                    ind_min = 0
                ind_max = np.argmax(np.abs(smooth_exp_strain[:]) > np.abs(self.min_max_strain[iobj][1])) + 1
                if ind_max == 1:
                    ind_max = -1
            else:
                ind_min = 0
                ind_max = smooth_exp_strain.size
            s_exp_abs = smooth_exp_stress
            s_sim_abs = s_sim[iobj][0]
            # Lots of ways to normalize
            # 1: NRMSD = \frac{RMSD}{max(obsevable) - min(observable)}
            # 2: NRMSD = \frac{RMSD}{Q3(obsevable) - Q1(observable)}
            # 3: NRMSD = \frac{RMSD}{STD(obsevable)}
            # 4: NRMSD = \frac{RMSD}{MEAN(observable)}
            # Based on my testing, it appears that #3/#2 might give the most reasonable
            # set of answers as they penalize results that nail the elastic regime,
            # hit portions of the elastic plastic and then are just way off after that point.
            # If our plastic regime is way off we want to penalize things.
            # NRMSD = \frac{RMSD}{max(obsevable) - min(observable)}
            # RMSD = \sqrt(\frac{\sum_{i=1}^{n} ( simulated_i - observable_i )^2 }{n})
            # f[iobj] = np.sqrt(np.sum(np.power((s_sim_abs - s_exp_abs), 2)) / s_sim_abs.size) / (np.max(s_exp_abs) - np.min(s_exp_abs))
            # f[iobj] = np.sqrt(np.sum(np.power((s_sim_abs - s_exp_abs), 2)) / s_sim_abs.size) / (np.quantile(s_exp_abs, 0.75) - np.quantile(s_exp_abs, 0.25))
            f[iobj * 2] = np.sqrt(
                np.sum(np.power((s_sim_abs[ind_min:ind_max] - s_exp_abs[ind_min:ind_max]), 2)) / (ind_max - ind_min)
            ) / (np.std(s_exp_abs[ind_min:ind_max]))
            diff_strain = np.diff(smooth_exp_strain[ind_min:ind_max]) 
            exp_slope = np.diff(s_exp_abs[ind_min:ind_max]) / diff_strain
            sim_slope = np.diff(s_sim_abs[ind_min:ind_max]) / diff_strain
            f[iobj * 2 + 1] = np.sqrt(
                np.sum(np.power((sim_slope[:] - exp_slope[:]), 2)) / (sim_slope.size)
            ) / (np.std(exp_slope[:]))
            # f[iobj] = np.sqrt(np.sum(np.power((s_sim_abs - s_exp_abs), 2)) / s_sim_abs.size) / (np.mean(s_exp_abs))
            write_ExaProb_log("\t\tIndividual obj function: fit = " + str(f[2 * iobj]) + " slope obj fit: " + str(f[2 * iobj + 1]))

        self.s_sim.append(s_sim)
        if not failed_test:
            self.flag.append(flag)  # will be equal to 0
        else:
            self.flag[igene] = flag
        # If use a simple GA scheme then return the summation of all the objective functions
        # If use a multiple_objective GA scheme then return individual objective functions
        if self.n_obj == 1:
            F = [np.sum(f)]
            write_ExaProb_log("\tGlobal obj function: fit = " + str(F))
        else:
            F = f

        write_ExaProb_log("")

        return F

    def return_stress(self, igene):
        # save stresses/strains in a list for the particular iteration that returnStress() function is called
        stress = []
        stress.append(self.s_exp)
        stress.append(self.s_sim[igene])
        return stress

    def is_simulation_done(self, igene):
        print([igene, self.flag[igene]])
        return self.flag[igene]
