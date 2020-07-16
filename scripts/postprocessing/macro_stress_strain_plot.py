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

#number of time steps
nsteps = 40

# uncomment the below when the fileLoc is valid
#data = np.loadtxt(fileLoc+'avg_stress.txt', comments='%')
# only here to have something that'll plot
data = np.ones((nsteps, 6))

epsdot = 1e-3


sig = data[:,2]
# uncomment the below when the fileLoc is valid
#time = np.loadtxt(fileLoc+'custom_dt.txt')
# only here to have something that'll plot
time = np.ones(nsteps)

eps = np.zeros(nsteps)

for i in range(0, nsteps):
    dtime = time[i]
    if sig[i] - sig[i - 1] >= 0:
        eps[i] = eps[i - 1] + epsdot * dtime
    else:
        eps[i] = eps[i - 1] - epsdot * dtime

ax.plot(eps, sig, 'r')

ax.grid()

# change this to fit your data                 
# ax.axis([0, 0.01, 0, 0.3])

ax.set_ylabel('Macroscopic engineering stress [GPa]')
ax.set_xlabel('Macroscopic engineering strain [-]')

plt.close()
fig.show()
plt.show()