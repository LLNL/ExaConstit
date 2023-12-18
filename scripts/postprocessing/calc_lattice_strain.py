#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rjk5987 , dcp5303

This script calculates 'lattice strains', elastic strains along a specified sample direction
averaged over subsets of elements (grains) belonging to different crystallographic fibers.

Designed to take in 'raw_LatticeOrientation' outputs from adios2_extraction.py
and 'SampleElasticStrain' outputs from strain_Xtal_to_Sample.py

As-is, also requires that 'ElementVolume' was output from adios2_extraction.py
for use in a volume-weighted average. Typically does not make a significant difference,
so the user may wih to remove this below.

Saves lattice strains in ASCII text format (.txt) for use in other scripts.
Output file named 'lattice_strains' and arranged as step # by lattice plane.

Recommended to configure script in an interactive shell, then run from the command line.

User inputs marked with #!!!
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from hexrd import rotations as rot
from hexrd import material , valunits

# makes console print-outs nicer, can CTRL+F and remove these to eliminate this dependency
from sty import bg , ef , fg

# conversions between radians and degrees
d2r = np.pi / 180.
r2d = 180. / np.pi

# # experienced HEXRD users with an existing material file can uncomment this block to load from it
# # a proper materials.h5 file is overkill for this script, really just need space group and lattice parameters
# def load_pdata(filename , d_min , energy , mat): 

#     kev  = valunits.valWUnit('beam_energy' , 'energy' , energy , 'keV')
#     dmin = valunits.valWUnit('dmin' , 'length' , d_min , 'angstrom')
#     mats = material.load_materials_hdf5(filename , dmin = dmin , kev = kev)
#     pd   = mats[mat].planeData

#     return pd

# # requires a materials.h5 file created in HEXRD
# mat_file = '/dir/to/materials.h5' #!!!
# # material of interest in material file
# mat = 'IN625' #!!!
# # load peak data for specified material from material file
# # other arguments are minimum d-spacing filter (in angstroms) and beam energy (in keV)
# pd = load_pdata(mat_file , 0.5 , 61.332 , mat) #!!!

# users that do not have an existing HEXRD material file should use this function instead
# this accomplishes the same thing with less hassle
def make_matl(mat_name , sgnum , lparms , hkl_ssq_max = 50 , dmin_angstroms = 0.5) :
    """
    
    Parameters
    ----------
    mat_name : str
        label for material.
    sgnum : int
        space group number for material.
    lparms : list of floats
        lattice parameters in angstroms.
    hkl_ssq_max : int, optional
        maximum hkl sum of squares (peak upper bound). The default is 50.
    dmin_angstroms : float, optional
        minimum d-spacing in angstroms (alt peak upper bound). The default is 0.5.
        
    """
    
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max
    matl.dmin = valunits.valWUnit('lp' , 'length' , dmin_angstroms , 'angstrom')
    
    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls , dtype = bool))
    
    return matl

# here is a simple example for IN625 - see function above for what the arguments are
matl = make_matl(mat_name = 'FCC' , sgnum = 225 , lparms = [3.60 ,]) #!!!
pd = matl.planeData

#%% Booleans.

# whether to plot the resulting lattice strains vs. cycle when finished
plot = True #!!!
# whether to save plot (will save to the same directory as the lattice strain file)
save_plot = False #!!!

#%% Establish filepaths and directories.

# directory where 'raw_LatticeOrientation' files were saved off from adios2_extraction.py
raw_dir = 'dir/to/adios2_extraction_files/' #!!!
# directory where 'SampleElasticStrain' files were saved off from strain_Xtal_to_Sample.py
strain_dir = 'dir/to/strain_Xtal_to_Sample_files/' #!!!
# save directory for script outputs, recursively create directory if it doesn't exist
out_dir = 'dir/to/store/outputs/' #!!!
if not os.path.exists(out_dir) :
    print(fg.magenta + 'Output directory not found, creating.' + fg.rs)
    os.makedirs(out_dir)
else :
    print(fg.green + 'Output directory found, proceeding.' + fg.rs)

#%% Specify information relevant to simulations and lattice strain calculations.

# count number of simulation steps saved off
steps = 0
for fname in os.listdir(raw_dir) :
    if fname.startswith('raw_LatticeOrientation') :
        steps += 1

# distance bound - maximum angular distance from crystallographic fiber in degrees (may prefer smaller tolerances for larger meshes)
distance_bnd = 5

# array of Miller indices for lattice planes/strains of interest (gets transposed to be HEXRD-amenable)
hkl = np.array([
    [1,1,1],
    [2,0,0],
    [2,2,0],
    [3,1,1]
    ]).T

# strain direction of interest in ExaConstit z-up coordinate system (example below for z-axis)
s_dir = np.array([0,0,1])

#%% Calculate lattice strains.

# initialize array of lattice strains
lattice_strains = np.zeros((steps , hkl.shape[1]))

print(ef.bold + bg.green + '\nCalculating lattice strains.' + bg.rs + ef.rs)

# iterate over cycles
for ii in range(steps) :
    
    print(ef.bold + bg.da_blue + '\nProcessing step ' + str(ii+1) + ' of ' + str(steps) + '.' + bg.rs + ef.rs)
    
    print(fg.li_blue + '\nLoading data.' + fg.rs)
    vols = np.loadtxt(raw_dir + 'raw_ElementVolume_%0.02d.txt' %ii)
    quats = np.loadtxt(raw_dir + 'raw_LatticeOrientation_%0.02d.txt' %ii).T
    strain = np.loadtxt(strain_dir + 'SampleElasticStrain_%0.02d.txt' %ii)
    print('Data loaded.')
    
    print(fg.li_blue + '\nCalculating lattice strains.' + fg.rs)
    # iterate over lattice planes
    for jj in range(hkl.shape[1]) :
        
        print('Processing hkl ' + str(jj+1) + ' of ' + str(hkl.shape[1]) + '.' , end = '\r')
        
        # compute crystal direction from peak data
        c_dir = np.atleast_2d(np.dot(pd.latVecOps['B'] , hkl[:,jj])).T
        
        # compute distance from quaternions to crystallographic fiber
        distance = rot.distanceToFiber(c_dir , s_dir , quats , pd.getQSym()) * r2d
        
        # filter for distances within specified distance bound
        in_fiber = np.where(distance < distance_bnd)[0]
        
        # clever math trick for projecting strain 6-vector along strain direction of interest without converting to tensor form
        project = np.array([s_dir[0]**2 , s_dir[1]**2 , s_dir[2]**2 , 2*s_dir[1]*s_dir[2] , 2*s_dir[0]*s_dir[2] , 2*s_dir[0]*s_dir[1]])        
        
        # calculate volume-weighted average of lattice strain for each fiber
        # remove weights = vols[in_fiber] to get rid of volume-averaging
        lattice_strains[ii,jj] = np.average(np.dot(project.T , strain[in_fiber , :].T) , weights = vols[in_fiber])
        
        del c_dir , distance , project , in_fiber
    print('Lattice strains calculated.')
        
    del vols , quats , strain
    
print(fg.li_blue + '\nSaving lattice strains.' + fg.rs)
np.savetxt(out_dir + 'lattice_strains.txt' , lattice_strains)
print('Lattice strains saved.')

#%% Plot lattice strain vs. cycle if enabled, and save figure if enabled.
 
if plot :
    
    fig , ax = plt.subplots(1 , 1 , figsize = (4 , 4))
    
    for kk in range(hkl.shape[1]) :
        ax.plot(lattice_strains[:,kk] , label = '%s' %hkl[:,kk])
    
    ax.set_xlabel('step')
    ax.set_ylabel(r'lattice strain, $\epsilon_{hkl}$')
    ax.legend()
    
    if save_plot :
        
        plt.savefig(out_dir + 'fig_lattice_strain_vs_step.png')

    plt.show()
