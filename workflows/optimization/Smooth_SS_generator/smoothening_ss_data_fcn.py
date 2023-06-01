import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate


def smooth_stress_strain_data(
    raw_experimental_data, dt_file_loc, strain_rate, strain_final, sim_stress_steps, visualize=False
):
    """
    This function takes in a raw set of experimental data and a dt file location
    and then smooths out the experimental data to match the location of the dt file
    Note: Interpolation between stress/strain values is assumed to be a linear interpolation
          and this was chosen in order to reduce potential oscillation between values
          This function also would not work best for cyclic data at this point in time.
    Input: raw_experimental_data - 2d array where column 0 = stress and column 1 = strain
           dt_file_loc - absolute file path to the delta time step file
    Output: smooth_stress - 1d array of smoothed stress values that correspond to the values located at the
                            dt_file_loc
            smooth_true_strain - 1d array of smoothed true strain values that correspond to the values located at the
                                 dt_file_loc
    """

    # ======================= Inputs =======================
    # Read y2 x2 data
    # Get points without smoothening input y0 and x0 which are the stress and strain respectively

    ss_data = np.copy(raw_experimental_data)

    x0 = raw_experimental_data[:, 1]
    y0 = raw_experimental_data[:, 0]

    # Read custom_dt data
    custom_dt = np.loadtxt(dt_file_loc, dtype="float", ndmin=1)
    # It's possible for custom_dt to have more data than stress values if the job times out before the average stress states are saved off
    # Therefore, we need to check only use as many dt values as stress values or else the code will crash on us
    custom_dt = custom_dt[0:sim_stress_steps]

    total_strain = np.log(1.0 + np.sum(custom_dt) * strain_rate)
    # If total strain is greater than desired_strain we're still fine
    error_strain = np.sqrt((strain_final - total_strain) ** 2.0 / strain_final ** 2.0)
    if error_strain > 0.01:
        time_final = np.sum(custom_dt)
        time_desired = (np.exp(strain_final) - 1.0) / strain_rate
        times = np.linspace(time_final, time_desired, num=25)
        time_deltas = np.diff(times)
        custom_dt = np.append(custom_dt, time_deltas)

    # ================ Smoothening Script =================

    # Offset x2 to start from zero
    # Might make more sense to just append 0.0 to start for strain and stress
    # values if that doesn't already exist in raw data.

    # Neglect repeated values of x2
    _x, ind = np.unique(ss_data[:, 1], return_index=True)
    ss_data = ss_data[ind]

    # Find convex hull data
    hull = ConvexHull(ss_data, qhull_options="Q14")
    ind = hull.vertices
    ss_data_hull = ss_data[ind]
    # print('INFO: Length of data after performing convex hull has been reduced to: {} rows\n'.format(len(ss_data_hull)))

    # Sort data using x2 column
    ind = np.argsort(ss_data_hull[:, 1])
    ss_data_hull = ss_data_hull[ind]

    # Stress strain points derived from convex hull, where y1 (stresses) and x1 (strains)
    x1 = ss_data_hull[:, 1]
    y1 = ss_data_hull[:, 0]

    # Perform smoothening using Spline Interpolation
    # Used 1st order polynomials since if higher order we might have oscilation problems
    # The more data we use as input, the more precise the derived hull points and thus, the spline
    # interpolation will be more accurate.
    spl = interpolate.splrep(x1, y1, k=1)

    # Calculate new strains (x2) by using specified strain rate and custom_dt
    # Find stresses corresponding to the above x2 (strains) using Cubic Spline
    # We provide the strain rate as an input
    nsteps = len(custom_dt) + 1
    x2 = np.zeros(nsteps)
    for i in range(1, nsteps):
        x2[i] = x2[i - 1] + custom_dt[i - 1] * strain_rate

    x2 = np.log(x2 + 1.0)
    x2 = x2[1::]
    y2 = interpolate.splev(x2, spl)

    # if(np.max(x1) > np.max(x2)):
    #     print("WARNING: The maximum value of final strains derived using custon_dt file \"max(x2)\", should be smaller \
    #     than maximum strain value of the original data (about 5%). That is to compensate for the last convex_hull \
    #     points that usually won't be smooth. If have done this already, disregard the warning!\
    #     \nHere: max_input_strain = {}, max_output_strain = {}\n\n".format(max(x1), max(x2)))

    # Visualize
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.style.use("seaborn-poster")
        plt.plot(x0, y0, "r", linewidth=1)
        plt.plot(x1, y1, "k", linewidth=1, marker="o", markersize=6)
        plt.plot(x2, y2, "b", linewidth=1, marker="o", markersize=6)
        plt.title("Spline Interpolation")
        plt.xlabel("strain [mm/mm]")
        plt.ylabel("stress [MPa]")
        plt.legend(["original curve", "convex_hull curve", "interpolated curve"])
        plt.show()

    return (y2, x2, error_strain)
