import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate


# ======================= Inputs =======================
# Read y2 x2 data
# Get points without smoothening input y0 and x0 which are the stress and strain respectively
SS_data = np.loadtxt("V3_Original_SS_data_file.txt", dtype="float", ndmin=2)
x0 = SS_data[:, 1]
y0 = SS_data[:, 0]
print("INFO: Original length of data is: {} rows\n".format(len(SS_data)))

# Read custom_dt data
custom_dt = np.loadtxt("custom_dtv01.txt", dtype="float", ndmin=1)


# ================ Smoothening Script =================

# Offset x2 to start from zero
SS_data[:, 1] = SS_data[:, 1] - SS_data[:, 1][0]

# Neglect repeated values of x2
_x, ind = np.unique(SS_data[:, 1], return_index=True)
SS_data = SS_data[ind]


# Find convex hull data
hull = ConvexHull(SS_data, qhull_options="Q14")
ind = hull.vertices
SS_data_hull = SS_data[ind]
print(
    "INFO: Length of data after performing convex hull has been reduced to: {} rows\n".format(
        len(SS_data_hull)
    )
)


# Sort data using x2 column
ind = np.argsort(SS_data_hull[:, 1])
SS_data_hull = SS_data_hull[ind]


# Stress strain points derived from convex hull, where y1 (stresses) and x1 (strains)
x1 = SS_data_hull[:, 1]
y1 = SS_data_hull[:, 0]


# Perform smoothening using Spline Interpolation
# Used 1st order polynomials since if higher order we might have oscilation problems
# The more data we use as input, the more precise the derived hull points and thus, the spline
# interpolation will be more accurate.
spl = interpolate.splrep(x1, y1, k=1, s=len(x1) * 1e-4)


# Calculate new strains (x2) by using specified strain rate and custom_dt
# Find stresses corresponding to the above x2 (strains) using Cubic Spline
nsteps = len(custom_dt)
strain_rate = 1e-3
x2 = np.zeros(nsteps)
for i in range(1, nsteps):
    x2[i] = x2[i - 1] + custom_dt[i] * strain_rate

y2 = interpolate.splev(x2, spl)


# Write a txt file with the derived experiment y2-x2 file corresponding to custom_dt input
np.savetxt("SS_data_file.txt", np.column_stack((y2, x2)))


print(
    'WARNING: The maximum value of final strains derived using custon_st file "max(x2)", should be smaller \
than maximum strain value of the original data (about 5%). That is to compensate for the last convex_hull \
points that usually won\'t be smooth. If have done this already, disregard the warning!\
\nHere: max_input_strain = {}, max_output_strain = {}\n\n'.format(
        max(x1), max(x2)
    )
)


# Visualize
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
