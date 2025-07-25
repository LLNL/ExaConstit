#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shows how to plot the macroscopic stress strain data from the average stress file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# How to plot the macroscopic stress strain data

rc('mathtext', default='regular')

font = {'size'   : 14}

rc('font', **font)
fileLoc = 'path of dir of avg_stress.txt and custom_dt.txt'

# We can have differnt colors for our curves
clrs = ['red', 'blue', 'green', 'black']
mrks = ['*', ':', '--', 'solid']

fig, ax = plt.subplots(1)

nsteps = 40
# uncomment the below when the fileLoc is valid
#data = np.loadtxt(fileLoc+'avg_stress.txt', comments='%')
# only here to have something that'll plot
data = np.ones((nsteps, 8))
# First two columns are time and volume
# Next 6 columns are the Cauchy stress in voigt notation
sig = data[:,4]
vol = data[:,1]
time = data[:,0]
nsteps = data.shape[0] + 1

sig = np.r_[0, sig]
vol = np.r_[1, vol]
time = np.r_[0, time]

# If setup for it you can also request either the eulerian strain or deformation gradient
# in which case most of the below can be ignored

epsdot = 1e-3
# only here to have something that'll plot
eps = np.zeros(nsteps)

for i in range(0, nsteps):
    if (i == 0):
        dtime = time[i]
    else:
        dtime = time[i] - time[i - 1]
    # Stress is not always monotonically increasing so this is not always the
    # best assumption
    if sig[i] - sig[i - 1] >= 0:
        eps[i] = eps[i - 1] + epsdot * dtime
    else:
        eps[i] = eps[i - 1] - epsdot * dtime

# For true strain the np.log(1 + eps) will provide the correct conversion
ax.plot(np.log(1.0 + eps), sig, 'r')
ax.grid()

# change this to fit your data
# ax.axis([0, 0.01, 0, 0.3])

ax.set_ylabel('Macroscopic true stress [GPa]')
ax.set_xlabel('Macroscopic true strain [-]')

fig.show()
plt.show()