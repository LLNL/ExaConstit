import pickle
import os
from deap import creator, base
import numpy
from ExaConstit_SolPicker import BestSol
from ExaPlots import StressStrain
import matplotlib.pyplot as plt


"""
This script is using pymoo module visualization folder to show useful plots of the data as a PostProcessing procedure and
it has no DEAP dependencies. 
This script can be used for any output file independantly of the optimization procedure. Therefore, it can be called 
during the optimization framework to watch its progress for a specified generation that has already been calculated.
How to run: You can call this function from any script or you can specify the inputs and run this script
"""


# ========================= Inputs ==============================
NOBJ = 8
NEXP = 4
GEN = -1  # show last gen
checkpoint = "./checkpoint_files/checkpoint_gen_125.pkl"


# ====================== Post Processing ========================
# Create classes needed for pickle to read the data
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin, rank=None, stress=None)


def ExaPostProcess(pop_library=None, checkpoint=None, NOBJ=NOBJ, GEN=GEN, nsmallest=1):

    if pop_library == None and checkpoint == None:
        raise "No inputs provided"

    elif not checkpoint == None:

        with open(checkpoint, "rb+") as ckp_file:
            ckp = pickle.load(ckp_file)
        try:
            # Retrieve the state of the last checkpoint
            pop_library = ckp["pop_library"]
        except:
            raise "Could not read checkpoint file"

    # Retrieve some more info
    NGEN = numpy.array(pop_library).shape[0]
    NPOP = numpy.array(pop_library).shape[1]
    NDIM = numpy.array(pop_library).shape[2]

    # Initialize the finale data lists
    pop_fit = []
    pop_param = []
    pop_stress = []
    best_idx = []

    # For gen=0 we don not perform selection, thus, there is no best_front
    best_front_par = [[None]]
    best_front_fit = [[None]]

    for gen in range(NGEN):

        # Initialize temporary data lists
        pop_fit_gen = []
        pop_par_gen = []
        pop_stress_gen = []
        pop_gene_gen = []
        pop_autotime_dt = []
        best_front_par_gen = []
        best_front_fit_gen = []
        best_front_stress_gen = []
        best_front_gene_gen = []

        for ind in pop_library[gen]:
            pop_fit_gen.append(ind.fitness.values)
            pop_par_gen.append(ind)
            pop_stress_gen.append(ind.stress)

            if NOBJ > 1 and gen != 0:
                if ind.rank == 0:
                    best_front_fit_gen.append(ind.fitness.values)
                    best_front_par_gen.append(ind)
                    best_front_stress_gen.append(ind.stress)

        # Store all data for each generation
        if NOBJ > 1:
            best_front_fit.append(best_front_fit_gen)
            best_front_par.append(best_front_par_gen)
        pop_fit.append(pop_fit_gen)
        pop_param.append(pop_par_gen)
        pop_stress.append(pop_stress_gen)

        # Find best solution for each generation and store it
        if NOBJ == 1:
            best_idx.append(numpy.argmin(pop_fit[gen]))
        else:
            # Weights must have the same length with objectives
            weights = [1] * NOBJ
            best_idx.append(BestSol(pop_fit[gen], weights=weights, nsmallest=nsmallest).EUDIST())

    # VISUALIZATION

    # Make data numpy type (best_front has different size per generation, thus it is not so simple)
    pop_fit = numpy.array(pop_fit)

    # Visualize the results (here we used the visualization module of pymoo extensively)

    # Note that: pop_stress[gen][ind][expSim][file]
    # first dimension is the selected generation,
    # second is the selected individual, third is if we want to use experiment [0] or simulation [1] data,
    # forth is the selected experiment file used for the simulation
    # strain_rate = 1e-3
    fig_setup = False
    initial_loop = True
    for ismall in range(nsmallest-1, -1, -1):
        print(ismall)
        print(best_idx[GEN][ismall])
        print(pop_library[GEN][best_idx[GEN][ismall]])
        if initial_loop:
            for k in range(numpy.array(pop_stress).shape[3]):
                S_exp = pop_stress[GEN][best_idx[GEN][ismall]][0][k]
                S_sim = pop_stress[GEN][best_idx[GEN][ismall]][1][k]
                if  not fig_setup:
                    fig, axis = StressStrain(S_exp, S_sim, ismall, k)
                    fig_setup = True
                else:
                    fig, axis = StressStrain(S_exp, S_sim, ismall, k, fig, axis, initial_loop)
        initial_loop = False
        for k in range(numpy.array(pop_stress).shape[3]):
            S_exp = pop_stress[GEN][best_idx[GEN][ismall]][0][k]
            S_sim = pop_stress[GEN][best_idx[GEN][ismall]][1][k]
            fig, axis = StressStrain(S_exp, S_sim, ismall, k, fig, axis, initial_loop)

        initial_loop = False

    fig.show()
    plt.show()

    # from Visualization.scatter import Scatter

    # plot = Scatter()
    # plot.add(pop_fit[GEN], s=20)
    # if NOBJ != 1 and len(best_front_fit[GEN]) != 0:
    #     plot.add(numpy.array(best_front_fit[GEN]), s=20, color="orange")
    # plot.add(pop_fit[GEN][best_idx[GEN]], s=30, color="red")
    # plot.show()

    # if NOBJ != 1:
    #     from Visualization.pcp import PCP

    #     plot = PCP(tight_layout=False)
    #     plot.set_axis_style(color="grey", alpha=0.5)
    #     plot.add(pop_fit[GEN], color="grey", alpha=0.3)
    #     plot.add(pop_fit[GEN][best_idx[GEN]], linewidth=2, color="red")
    #     plot.show()

    # from Visualization.petal import Petal

    # bounds = [0, 0.5]
    # plot = Petal(bounds=bounds, tight_layout=False)
    # plot.add(pop_fit[GEN][best_idx[GEN]])
    # plot.show()
    # # Put out of comments if we want to see all the individual fitnesses and not only the best
    # plot = Petal(
    #     bounds=bounds, title=["Sol %s" % t for t in range(0, NPOP)], tight_layout=False
    # )
    # for k in range(1, NPOP + 1):
    #     if k % 4 == 0:
    #         plot.add(pop_fit[GEN][k - 4 : k])
    # plot.show()


ExaPostProcess(pop_library=None, checkpoint=checkpoint, NOBJ=NOBJ, GEN=95, nsmallest=1)
