import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


def ObjFun3D(ref_points, pop_fit, best_idx):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Coordinate origin
    ax.scatter(0, 0, 0, c="k", marker="+", s=100)
    ax = fig.add_subplot(111, projection="3d")

    # Plot best_solution
    if best_idx:
        ax.scatter(
            pop_fit[best_idx, 0],
            pop_fit[best_idx, 1],
            pop_fit[best_idx, 2],
            marker="o",
            linewidths=1,
            facecolors="none",
            edgecolors="black",
            s=60,
        )

    # Plot ref_points
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o")

    # Plot last population fitness
    ax.scatter(pop_fit[:, 0], pop_fit[:, 1], pop_fit[:, 2], marker="x")

    # final figure details
    ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
    ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
    ax.set_zlabel("$f_3(\mathbf{x})$", fontsize=15)
    ax.view_init(elev=11, azim=-25)
    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    # ax.autoscale(tight=True)

    plt.legend()
    plt.tight_layout()
    plt.show()


def ObjFun2D(ref_points, pop_fit, best_idx=None):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # Coordinate origin
    ax.scatter(0, 0, c="k", marker="+", s=100)

    # Plot best solution
    if not best_idx == None:
        ax.scatter(
            pop_fit[best_idx, 0],
            pop_fit[best_idx, 1],
            marker="o",
            linewidths=1,
            facecolors="none",
            edgecolors="black",
            s=60,
        )

    # Plot ref_points
    ax.scatter(ref_points[:, 0], ref_points[:, 1], marker="o")

    # Plot last population fitness
    ax.scatter(pop_fit[:, 0], pop_fit[:, 1], marker="x")

    # final figure details
    ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
    ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)

    plt.show()


def StressStrain(Exper_data, Simul_data, epsdot, custom_dt_file=None):
    # How to plot the macroscopic stress strain data (Robert Carson)

    # Need both S_sim and S_exp to be 1d arrays
    S_exp = np.array(Exper_data)
    S_sim = np.array(Simul_data)

    # only here to have something that'll plot
    nsteps = len(S_exp)
    if custom_dt_file == None:
        time = np.ones(nsteps)
    else:
        time = np.loadtxt(custom_dt_file)

    font = {"size": 11}
    rc("font", **font)
    rc("mathtext", default="regular")

    # We can have differnt colors for our curves
    clrs = ["red", "blue", "green", "black"]
    mrks = ["*", ":", "--", "solid"]

    fig, ax = plt.subplots(1)

    # initiate strain
    eps = np.zeros(nsteps)

    for i in range(1, nsteps):
        dtime = time[i]
        if S_sim[i] - S_sim[i - 1] >= 0:
            eps[i] = eps[i - 1] + epsdot * dtime
        else:
            eps[i] = eps[i - 1] - epsdot * dtime

    ax.plot(eps, S_exp, color="r", label="S_exp")  # , linestyle='--')
    ax.plot(eps, S_sim, color="b", label="S_sim")
    ax.grid(linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    # change this to fit your data
    # ax.axis([0, 0.01, 0, 0.3])

    ax.set_ylabel("Macroscopic engineering stress [GPa]")
    ax.set_xlabel("Macroscopic engineering strain [-]")

    fig.show()
    plt.show()
