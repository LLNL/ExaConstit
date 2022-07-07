from deap import creator, base, tools, algorithms
from deap.benchmarks.tools import hypervolume, convergence, diversity
import numpy as np
import os
import random
from math import factorial
import pickle
import sys

# Normal map used for basic jobs that aren't on a cluster
import normal_map as map_custom
import pandas as pd

from ExaConstit_Problems import ExaProb
from ExaConstit_Logger import initialize_ExaProb_log, write_ExaProb_log


"""
ExaConstit Optimization Routine (LZ)

This script is using the NSGAIII algorithm from the DEAP module and intends
to optimize Crystal Plasticity Parameters calling the ExaConstit simulation
program via ExaProb class.

The method that the objective functions are calculated can be found in 
the evaluation function in the ExaProb class.

For details about how DEAP module works please look at their tutorial:
https://deap.readthedocs.io/en/master/index.html 
Also please look at the associated paper for the NSGAIII
"""


"""============================== Basic Parameter (INPUT) ================================"""
# Choose if we want to run the framework with U-NSGA-III
## If NOBJ = 1 (one objective), this must be True
UNSGA3 = True

# Number of obj functions
## We can have 1 objective even if we have multiple different data with the following way:
## The algorithm will use the summation of the individual objectives and the summation as one objective
NOBJ = 1

"""==========================  CP Parameter Constraints (INPUT) ==========================="""
# Specify independent per experiment data file parameters (e.g. athermal parameters)
## How to use: Specify their upper and their lower bounds
IND_LOW = [150, 100, 50, 1500, 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]
IND_UP = [200, 150, 100, 2500, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]

# Specify parameters are different (dependent) per experiment data file (e.g. thermal parameters).
## If no such parameters, set DEP_LOW = None, DEP_UP = None
## How to use: Specify their upper and their lower bounds
DEP_LOW = None
DEP_UP = None

# Specify parameters that will not be optimized and are different (dependent) per experiment data file (e.g. the temperatures, the strain rates etc).
## If no such parameters, set DEP_UNOPT = None.
## How to use: DEP_UNOPT = [[file1], [file2], ...], where [fileN] = [param1, param2, ...]
DEP_UNOPT = [[300]]

# Calcualte Final Bounds considering above inputs (make list with repeated dependent parameters)
BOUND_LOW = IND_LOW
BOUND_UP = IND_UP
if (DEP_LOW != None) and (DEP_UP != None):
    for i in range(NOBJ):
        BOUND_LOW.extend(DEP_LOW)
        BOUND_UP.extend(DEP_UP)
    n_dep = len(DEP_LOW)
else:
    n_dep = None

# Number of total parameters or dimensions or genes: NDIM = len(IND) + len(DEP)*NOBJ
NDIM = len(BOUND_LOW)

"""============================== Checkpoint Parameters (INPUT) ==============================="""
# Specify seed. Each time we run the framework we will get same results if inputs and seed is the same (good for debugging)
seed = None

# Specify checkpoint frequency (generations per checkpoint)
checkpoint_freq = 1

# Specify checkpoint file or set None if you want to start from the beginning
# checkpoint= "checkpoint_files/checkpoint_gen_2.pkl"
checkpoint = None

if checkpoint is not None:
    restart = True
else:
    restart = False

"""================ Initialize ExaProb object (this calls ExaConstit) (INPUT) ================"""
# Initialize NSGA3 and ExaProb logger and specify log level (threshold):
initialize_ExaProb_log(
    glob_loglvl="info", filename="logbook3_ExaProb.log", restart=restart
)

# Absolute path that we can find the ExaConstit binary (mechanics)
loc_mechanics = "ExaConstit/build/bin/mechanics"

# Specify the files that contain the Experiment data
# When NOBJ > 1 then it must be len(exper_input_files) == NOBJ. If NOBJ == 1 then we solve a multi-objective problem with
# the tradditional way (1 objective function), and thus len(Exper_input_files) > NOBJ is allowable.
exper_input_files = ["expt_stress_strain_data.txt"]

# Specify number of cpus that will use running each simualtion
ncpus = 2
ngpus = 0
nnodes = 1

temperature_k = [300]
# The strain rate for each one of our simulations
strain_rate = [1.0e-3]
# desired strain we're trying to reach
desired_strain = [0.001]
# how long we want each simulation to run before killing it
timeout = [1 * 60]

test_dataframe = {
    "experiments": exper_input_files,
    "temp_k": temperature_k,
    "strain_rate": strain_rate,
    "desired_strain": desired_strain,
    "timeout": timeout,
}
test_dataframe = pd.DataFrame(data=test_dataframe)

# Specify ExaProb class arguments to run ExaConstit simulations and evaluate the objective functions
problem = ExaProb(
    n_dep=n_dep,
    dep_unopt=DEP_UNOPT,
    nnodes=nnodes,
    ncpus=ncpus,
    ngpus=ngpus,
    test_dataframe=test_dataframe,
    bin_mechanics=loc_mechanics,
    sim_input_file_dir="./input_files",
    master_toml_file="./master_options.toml",
    workflow_dir="./wf_files",
    job_script_file=None,
)

"""================ Specify reference points thus, population number (INPUT) ================="""
## For NOBJ == 1 specify the number of reference points P only
P = 20

## For NOBJ != 1 specify p = [P1 , P2] and scaling = [scale1, scale2]. If want only one hyperplane, set P2 and scale2 equal to zero
# We can specify more reference point planes
p = [10, 0]
scaling = [1, 0]

# Generate reference points considering above inputs
if NOBJ == 1:
    # Final number of population
    NPOP = int(P + (4 - P % 4))
    # Since NOBJ = 1 this will generate only one reference point
    ref_points = tools.uniform_reference_points(NOBJ, P)

else:
    # Generate reference points
    ref1 = tools.uniform_reference_points(NOBJ, p[0], scaling[0])
    if p[1] != 0 and scaling[1] != 0:
        ref2 = tools.uniform_reference_points(NOBJ, p[1], scaling[1])
        ref_points = np.concatenate((ref1, ref2), axis=0)
    else:
        ref_points = ref1

    # Total number of Reference Points (NSGAIII paper)
    P = sum(p)
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))

    # Final Population number (NSGAIII paper)
    NPOP = int(H + (4 - H % 4))


"""========================== NSGA-III Operators Parameters (INPUT) =========================="""
# Those are the default parameters used in NSGA-III paper for NOBJ>2. For NOBJ<=2 please use and look at U-NSGA-III paper
# Mating Parameters
mat_eta = 30.0

# Mutation Parameters
mut_eta = 20.0
indpb = 1.0 / NDIM

"""========================== Stopping criteria parameters (INPUT) ============================"""
# Number of generations limit (e.g. if NGEN=2 it will perform the population initiation gen=0, and then gen=1 and gen=2. Thus, NGEN+1 generations)
NGEN = 10

# Specify how many simulation failures in total to have so to terminate the optimization framework
# Note that if ExaConstit does not run because of error in the ExaProb calling code line, it will still do a number of fails equal to fail_limit.
# Framework does not know what causes failures in the ExaConstit part. Please check toml files and names of the output files if have failures in the beggining)
fail_limit = 2

# Specify number of concecutive generation limit that the population size (NPOP) and the number of non-dominated solutions (ND) are equal (stop_limit).
# When reach this number, framework stops
# Specify the number of generations (Imin) after which the criteria becomes active
# Stopping criteria according to https://doi.org/10.1007/s10596-019-09870-3
Imin = int(round(NGEN / 2))
stop_limit = 5

# Another possible multi-objective stopping criteria could be to set a specific HV value (look line 469). The framework should stop when this value has been achieved
# Note that for NOBJ == 1 case we haven't implement any stopping criteria yet
# That is easier comparing to multi-objective case


# Print some useful information before run the framework
print("\nNumber of objective functions = {}".format(NOBJ))
print("Number of parameters = {}".format(NDIM))
print("Number of generations = {}".format(NGEN))
if NOBJ != 1:
    print("Number of reference points = {}".format(H))
print("Population size = {}".format(NPOP))
print("Expected Total Iterations = {}".format(NPOP * NGEN))
print("Expected Total Simulation Runs = {}\n".format(NPOP * NOBJ * NGEN))


# ====================== Initialize Optimization Strategy ========================
# Create minimization problem (multiply -1 weights)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
# Create the Individual class that it has also the fitness (obj function results) as a list
creator.create(
    "Individual",
    list,
    fitness=creator.FitnessMin,
    rank=None,
    nich=None,
    nich_dist=None,
    stress=None,
    generation=None,
    gene=None,
)


# =========================== Initialize Population ==============================
# Generate a random individual with respect to his gene boundaries.
# Low and Up can be columns with same size as the number of genes of the individual
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


# ===================== Population Initialization Method =========================
toolbox = base.Toolbox()
# Register the above individual generator method in the toolbox class. That is attr_float with arguments low=BOUND_LOW, up=BOUND_UP, size=NDIM
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
# Function that produces a complete individual with NDIM number of genes that have Low and Up bounds
toolbox.register(
    "individual", tools.initIterate, creator.Individual, toolbox.attr_float
)
# Function that instantly produces MU individuals (population). We assign the attribute number_of_population at the main function in this problem
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# =========================== Initialize GA Operators ============================
#### Evolution Methods
# Function that returns the objective functions values as a dictionary (if n_obj=3 it will evaluate the obj function 3 times and will return 3 values (str) - It runs the problem.evaluate for n_obj times)
toolbox.register("map_custom", map_custom.map_custom)
toolbox.register("map_custom_fail", map_custom.map_custom_fail)
# Crossover function using the cxSimulatedBinaryBounded method
toolbox.register(
    "mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=mat_eta
)
# Mutation function that mutates an individual using the mutPolynomialBounded method. A high eta will produce a mutant resembling its parent, while a small eta will produce a ind_fit much more different.
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=BOUND_LOW,
    up=BOUND_UP,
    eta=mut_eta,
    indpb=indpb,
)
# Selection function that selects individuals from population + offspring using selNSGA3 method (non-domination levels, etc (look at paper for NSGAIII))
toolbox.register("select", tools.selNSGA3, ref_points=ref_points, nd="standard")


# ================================ Main NSGA-III Algorithm ===========================
# Here we construct our main algorithm NSGAIII
def main(seed=None, checkpoint=None, checkpoint_freq=1):

    # Initialize statistics object
    stats1 = tools.Statistics(lambda ind: ind.fitness.values)
    stats1.register("avg", np.mean, axis=0)
    stats1.register("std", np.std, axis=0)
    stats1.register("min", np.min, axis=0)
    stats1.register("max", np.max, axis=0)

    # If checkpoint has a file name, read and retrieve the state of last checkpoint from this file
    # If not then start from the beginning by generating the initial population
    if checkpoint:

        with open(checkpoint, "rb+") as ckp_file:
            ckp = pickle.load(ckp_file)

        try:
            # Retrieve random state
            random.setstate(ckp["rndstate"])

            # Retrieve the state of the last checkpoint
            last_gen = ckp["generation"]
            pop_library = ckp["pop_library"]
            pop = pop_library[last_gen]

            # Retrieve counters
            iter_tot = ckp["iter_tot"]
            start_gen = last_gen + 1
            if start_gen > NGEN:
                gen = start_gen
            fail_count = ckp["fail_count"]
            stop_count = ckp["stop_count"]
            if not stop_count < stop_limit:
                stop_optimization = True
            else:
                stop_optimization = False

            # Retrieve logbooks
            logbook1 = ckp["logbook1"]
            logbook2 = ckp["logbook2"]

        except:
            write_ExaProb_log(
                "Wrong Checkpoint file: Could not retrieve all the dictionaries of the checkpoint file.",
                "error",
                changeline=True,
            )
            sys.exit()

        if len(pop) != NPOP:
            write_ExaProb_log(
                "Wrong Checkpoint file: NPOP not equal between checkpoint file and problem initialization.",
                "error",
                changeline=True,
            )
            sys.exit()

        # Open excisting log files for writting
        logfile1 = open("logbook1_stats.log", "a")
        logfile1.write("loaded checkpoint: {}\n".format(checkpoint))
        logfile2 = open("logbook2_solutions.log", "a")
        logfile2.write("loaded checkpoint: {}\n".format(checkpoint))

    else:
        # Specify seed (need both numpy and random OR change niching in DEAP script)
        random.seed(seed)

        # Initialize loggers and open for writting
        logbook1 = tools.Logbook()
        logfile1 = open("logbook1_stats.log", "w+")
        logbook2 = tools.Logbook()
        logfile2 = open("logbook2_solutions.log", "w+")
        write_ExaProb_log("Generation: 0", "info")

        # Initialize counters and lists
        iter_pgen = 0  # iterations per generation
        iter_tot = 0  # total iterations
        start_gen = 1  # starting generation
        fail_count = 0  # stopping criteria counter
        stop_count = 0  # stopping criteria counter
        stop_optimization = False
        pop_library = (
            []
        )  # initiate population library that will save population for every generation

        # Produce initial population
        # We use the registered "population" method MU times and produce the population
        pop = toolbox.population(n=NPOP)
        pop_library.append(pop)

        # Returns the individuals with an invalid fitness
        # invalid_ind is a list with NDIM genes in col and invalid_ind IDs in rows)
        # Maps the fitness with the invalid_ind. Initiates the obj function calculation for each invalid_ind
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        len_invalid_ind = len(invalid_ind)

        fitness_eval = toolbox.map_custom(problem, 0, invalid_ind)
        # This whole map / thing almost seems a bit over kill to me as that whole process
        # is super expensive for us.
        genes_index = np.r_[0 : len(invalid_ind)]
        # _______________________________________________________________________________________________
        # Evaluates the fitness for each invalid_ind and assigns them the new values
        for ind, fit, igene in zip(invalid_ind, fitness_eval, genes_index):
            iter_pgen += 1
            iter_tot += 1

            # If simulation failed due to parameters, generate a new individual and run simulation with the new one
            # Stopping criteria: If there is more than fail_limit number of simulation failures, terminate framework
            if not problem.is_simulation_done(igene) == 0:

                while fail_count < fail_limit:
                    fail_count += 1
                    text = "Attempt to find another Parameter set to converge, fail_count = {}\n\n".format(
                        fail_count
                    )
                    write_ExaProb_log(text, "warning", changeline=False)
                    # Replace old individual with the new random one with the hope that now the simulation will run normally
                    ind[:] = toolbox.individual()
                    # Run simulation to find the obj functions
                    fit = toolbox.map_custom_fail(problem, 0, ind)
                    # If simulation successful break the loop
                    if problem.is_simulation_done() == 0:
                        break
                else:
                    text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(
                        fail_count
                    )
                    write_ExaProb_log(text, "error", changeline=True)
                    sys.exit()

            ind.fitness.values = fit
            ind.stress = problem.return_stress(igene)
            ind.gene = igene
            ind.generation = 0
        # _______________________________________________________________________________________________

        # Write logs and their headers
        if NOBJ == 1:
            logbook1.header = "gen", "iter", "simRuns", "std", "min", "avg", "max"
            record = stats1.compile(pop)
            logbook1.record(gen=0, iter=iter_pgen, simRuns=iter_pgen * NOBJ, **record)
        else:
            logbook1.header = (
                "gen",
                "iter",
                "simRuns",
                "ND",
                "GD",
                "HV",
                "std",
                "min",
                "avg",
                "max",
            )
            record = stats1.compile(pop)
            logbook1.record(
                gen=0,
                iter=iter_pgen,
                simRuns=iter_pgen * NOBJ,
                ND="None ",
                GD="None    ",
                HV="None    ",
                **record
            )

        logfile1.write("{}\n".format(logbook1.stream))
        logbook2.header = "gen", "fitness", "sol_generation", "sol_gene", "solutions"
        pop = toolbox.select(pop, NPOP)

        for ind in pop:
            logbook2.record(
                gen=0,
                sol_generation=ind.generation,
                sol_gene=ind.gene,
                fitness=list(ind.fitness.values),
                solutions=list(ind),
            )
            logfile2.write("{}\n".format(logbook2.stream))

        # Fill the dictionary using the dict(key=value[, ...]) constructor
        ckp = dict(
            pop_library=pop_library,
            iter_tot=iter_tot,
            generation=0,
            fail_count=fail_count,
            stop_count=stop_count,
            logbook1=logbook1,
            logbook2=logbook2,
            rndstate=random.getstate(),
        )

        # Checkpoint initial values just incase this is an expensive calculation
        fdironl = os.path.join(os.getcwd(), "checkpoint_files", "")
        if not os.path.exists(fdironl):
            os.makedirs(fdironl)
        fout = os.path.join(fdironl, "checkpoint_gen_{}.pkl".format(0))
        with open(fout, "wb+") as ckp_file:
            pickle.dump(ckp, ckp_file)

    # Begin the generational process
    gen = start_gen
    while gen < NGEN + 1 and stop_optimization == False:

        write_ExaProb_log("Generation: {}".format(gen), "info")
        logfile1 = open("logbook1_stats.log", "a+")
        logfile2 = open("logbook2_solutions.log", "a+")

        # If UNSGA3 == True then we apply the UNSGA3 niching to the population
        # Look at the corresponding paper: https://link.springer.com/chapter/10.1007/978-3-319-15892-1_3
        if UNSGA3 == True:

            Upop = tools.niching_selection_UNSGA3(pop)
            offspring = algorithms.varAnd(Upop, toolbox, 1, 1)

        else:
            # If NSGA-III
            # varAnd does the previously registered crossover and mutation methods
            # Produces the offsprings and deletes their previous fitness values
            offspring = algorithms.varAnd(pop, toolbox, 1, 1)

        # Evaluate the individuals that their fitness has not been evaluated
        # Returns the invalid_ind (in each row, returns the genes of each invalid_ind)
        # Invalid_ind are those which their fitness value has not been calculated
        # Evaluates the obj functions for each invalid_ind (here we have 3 obj function thus it does 3 computations)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_eval = toolbox.map_custom(problem, gen, invalid_ind)
        genes_index = np.r_[0 : len(invalid_ind)]

        # _______________________________________________________________________________________________
        # Evaluates the fitness for each invalid_ind and assigns them the new values
        iter_pgen = 0
        for ind, fit, igene in zip(invalid_ind, fitness_eval, genes_index):
            iter_pgen += 1
            iter_tot += 1

            # If simulation failed due to parameters, pick randomly 2 indivuduals, mate and mutate and try to run simulation with the new individual
            # Stopping criteria: If there is more than fail_limit number of simulation failures, terminate framework
            if not problem.is_simulation_done(igene) == 0:

                while fail_count < fail_limit:
                    fail_count += 1
                    text = "Attempt to find another Parameter set to converge, fail_count = {}\n\n".format(
                        fail_count
                    )
                    write_ExaProb_log(text, "warning", changeline=False)
                    # Need 2 different individuals to apply mate and mutate and derive a new individual
                    ind_idx = random.sample(range(0, len_invalid_ind), 2)
                    ind1 = invalid_ind[ind_idx[0]]
                    ind2 = invalid_ind[ind_idx[1]]
                    # Replace old individual with the new one with the hope that now the simulation will run normally
                    ind[:] = toolbox.mate(ind1, ind2)[round(random.random())]
                    ind[:] = toolbox.mutate(ind)[0]
                    # Run simulation to find the obj functions
                    fit = toolbox.map_custom_fail(problem, gen, ind)
                    # If simulation successful break the loop
                    if problem.is_simulation_done() == 0:
                        break
                else:
                    text = "The evaluation failed for a total of {} attempts! Framework will terminate!".format(
                        fail_count
                    )
                    write_ExaProb_log(text, "error", changeline=True)
                    sys.exit()

            ind.fitness.values = fit
            ind.stress = problem.return_stress(igene)
            ind.gene = igene
            ind.generation = gen
        # _______________________________________________________________________________________________
        if NOBJ == 1:
            # pop is ordered considering the objective values, thus, the pop[0] will be the best solution
            pop = toolbox.select(pop + offspring, NPOP)
            pop_library.append(pop)

            # Find best solution of current pop (first since the individuals are ordered)
            best_ind = pop[0]

            # Stopping criterial for one objective problems
            # TBD

        else:

            # Select (selNSGAIII) MU individuals as the next generation population from pop+offspring
            # Returns optimal front and population after selection
            pop = toolbox.select(pop + offspring, NPOP)
            pop_library.append(pop)

            # Find best front with rank=0
            best_front = []
            best_front_fit_gen = []
            for ind in pop:
                if ind.rank == 0:
                    best_front.append(ind)
                    best_front_fit_gen.append(ind.fitness.values)

            # Number of Non_Dominand solutions
            ND = len(best_front)

            # Average Eucledean distance of the non-dominated soultions (best_front)
            # Here, optimum point will be the the origin [0,0,...,0]
            # Average Euclidean distance according to: https://doi.org/10.1007/s10596-019-09870-3
            # Since this is a minimization problem, it is expected to decrease over generations but not always
            Di = np.sqrt((1 / ND) * np.sum(np.array(best_front_fit_gen) ** 2))
            HV = hypervolume(pop, [1] * NOBJ)

            # Stopping criteria according to https://doi.org/10.1007/s10596-019-09870-3
            if gen > Imin:
                if ND == NPOP:
                    stop_count += 1
                    write_ExaProb_log(
                        "INFO: Stopping criteria: Consecutive stop_count = {}\n".format(
                            stop_count
                        ),
                        "info",
                        changeline=True,
                    )
                else:
                    stop_count = 0
                if not stop_count < stop_limit:
                    stop_optimization = True

        # Write logs
        logfile1 = open("logbook1_stats.log", "a+")
        logfile2 = open("logbook2_solutions.log", "a+")
        record = stats1.compile(pop)

        if NOBJ == 1:
            logbook1.record(gen=gen, iter=iter_pgen, simRuns=iter_pgen * NOBJ, **record)
        else:
            logbook1.record(
                gen=gen,
                iter=iter_pgen,
                simRuns=iter_pgen * NOBJ,
                ND=ND,
                GD=Di,
                HV=HV,
                **record
            )

        logfile1.write("{}\n".format(logbook1.stream))
        for ind in pop:
            logbook2.record(
                gen=gen,
                sol_generation=ind.generation,
                sol_gene=ind.gene,
                fitness=list(ind.fitness.values),
                solutions=list(ind),
            )
            logfile2.write("{}\n".format(logbook2.stream))

        # Generate a checkpoint
        if gen % checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            ckp = dict(
                pop_library=pop_library,
                iter_tot=iter_tot,
                generation=gen,
                fail_count=fail_count,
                stop_count=stop_count,
                logbook1=logbook1,
                logbook2=logbook2,
                rndstate=random.getstate(),
            )

            fdironl = os.path.join(os.getcwd(), "checkpoint_files", "")
            if not os.path.exists(fdironl):
                os.makedirs(fdironl)
            fout = os.path.join(fdironl, "checkpoint_gen_{}.pkl".format(gen))
            with open(fout, "wb+") as ckp_file:
                pickle.dump(ckp, ckp_file)

        # Count gen
        gen += 1

    else:
        if gen >= NGEN + 1:
            text = "INFO: Stopping criteria: Reached the maximum number of generations: GEN = {}!\
                \nINFO: Framework has finished successfully!".format(
                gen - 1
            )
            write_ExaProb_log(text, "info", changeline=True)
        elif stop_optimization == True:
            text = "INFO: Stopping criteria: The number of non-dominant solutions is equal to the number of population for {} times consecutively!\
                \nINFO: Framework has finished successfully!".format(
                stop_count
            )
            write_ExaProb_log(text, "info", changeline=True)

    logfile1.close()
    logfile2.close()

    return pop_library


# Call the optimization routine
pop_library = main(seed, checkpoint, checkpoint_freq)
