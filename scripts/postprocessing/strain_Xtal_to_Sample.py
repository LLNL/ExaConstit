#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rjk5987, dcp5303

This script converts the strain tensors for each element in an ExaConstit simulation
at each time step from the crystal frame to the sample frame.
These can be used, for example, to calculate lattice strains.

Designed to take in outputs from adios2_extraction.py
Requires that 'LatticeOrientation' and 'XtalElasticStrain' were output.

Saves converted strains in ASCII text format (.txt) for use in other scripts.
Output files named 'SampleElasticStrain' and labeled by cycle.

Recommended to configure script in an interactive shell, then run from the command line.

User inputs marked with #!!!
"""

import numpy as np
import os

from hexrd import rotations as rot

# makes console print-outs nicer, can CTRL+F and remove these to eliminate this dependency
from sty import bg , ef , fg

#%% Specify filepaths. (USER INPUTS HERE)

# directory where 'raw_' files were saved off from adios2_extraction.py
in_dir = 'dir/to/adios2_extraction_files/' #!!!
# save directory for script outputs, recursively create directory if it doesn't exist
out_dir = 'dir/to/store/outputs/' #!!!
if not os.path.exists(out_dir) :
    print(fg.magenta + 'Output directory not found, creating.' + fg.rs)
    os.makedirs(out_dir)
else :
    print(fg.green + 'Output directory found, proceeding.' + fg.rs)
    
#%% Convert from crystal frame to sample frame. (NO INPUTS HERE)

# count number of simulation steps saved off
steps = 0
for fname in os.listdir(in_dir) :
    if fname.startswith('raw_LatticeOrientation') :
        steps += 1
        
print(ef.bold + bg.green + '\nConverting strains to sample frame.' + bg.rs + ef.rs)
    
for ii in range(steps) :
    
    print(ef.bold + bg.da_blue + '\nProcessing step ' + str(ii+1) + ' of ' + str(steps) + '.' + bg.rs + ef.rs)
    
    print(fg.li_blue + '\nLoading Xtal-to-ExaConstit lattice orientations.' + fg.rs)
    quats_c_to_s = np.loadtxt(in_dir + 'raw_LatticeOrientation_%0.02d.txt' %ii).T           
    print('Xtal-to-ExaConstit lattice orientations loaded.')
    
    # crystal-to-sample rotation matrices from quaternions
    rmats_c_to_s = rot.rotMatOfQuat(np.atleast_2d(quats_c_to_s))
    del quats_c_to_s

    print(fg.li_blue + '\nLoading elastic strain in Xtal frame.' + fg.rs)
    # strain 6-vector in crystal frame
    strainV_c = np.loadtxt(in_dir + 'raw_XtalElasticStrain_%0.02d.txt' %ii)
    print('Elastic strain in Xtal frame loaded.')
    strainV_s = np.empty_like(strainV_c)
    
    for jj in range(strainV_c.shape[0]) :
        print('Processing element ' + str(jj+1) + ' of ' + str(strainV_c.shape[0]) + '.' , end = '\r')
        
        # strain tensor in crystal frame (does not need factors of 2)
        strainT_c = np.array([
            [strainV_c[jj,0] , strainV_c[jj,5] , strainV_c[jj,4]] ,
            [strainV_c[jj,5] , strainV_c[jj,1] , strainV_c[jj,3]] ,
            [strainV_c[jj,4] , strainV_c[jj,3] , strainV_c[jj,2]]
            ])   
        
        # strain tensor in HEXRD frame
        strainT_s = np.dot(rmats_c_to_s[jj] , np.dot(strainT_c , rmats_c_to_s[jj].T))
        # strain 6-vector in HEXRD frame
        strainV_s[jj] = np.array([strainT_s[0,0] , strainT_s[1,1] , strainT_s[2,2] , strainT_s[1,2] , strainT_s[0,2] , strainT_s[0,1]])
     
    print(fg.li_blue + '\n\nSaving elastic strain in HEXRD frame.' + fg.rs)
    np.savetxt(out_dir + 'SampleElasticStrain_%0.02d.txt' %ii , strainV_s)
    print('Elastic strain in HEXRD frame saved.')
    del strainV_c , strainV_s , strainT_c , strainT_s , rmats_c_to_s