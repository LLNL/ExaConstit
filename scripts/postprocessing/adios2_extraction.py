#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rjk5987, dcp5303

Developed using ExaConstit/scripts/postprocessing/adios2_example.py

This script extracts specified variables from ExaConstit ADIOS2 binary-pack (.bp) files
and saves them off in ASCII text format (.txt) for use in other scripts.

Output files prefixed by 'raw_' and labeled by variable name and cycle.

Recommended to configure script in an interactive shell to explore variables,
then run from the command line.

Heavily compartmentalized to handle large meshes (tested on up to 8M elements),
so may not be the most efficient method for processing smaller meshes.

User inputs marked with #!!!
"""

import adios2
import numpy as np
import os

# makes console print-outs nicer, can CTRL+F and remove these to eliminate this dependency
from sty import bg , ef , fg

#%% Global variables related to inputs/outputs. (USER INPUTS HERE)

# int - number of resource sets used for the simulation
# IMPORTANT: need to set this variable correctly to process all elements in the simulation
nranks = 48 #!!!
# bool - whether to overwrite existing outputs from this script
# (can also take in a list of bools corresponding to each variable)
overwrite = True #!!!
# bool - whether to resume from last step saved off if extraction was killed
# (can also take in a list of bools corresponding to each variable)
resume = False #!!!
# bool - whether to use an alternate unpacking procedure that is slower but more memory-efficient (may be necessary for handling larger variables)
# (can also take in a list of bools corresponding to each variable)
unpack_alt = [False , False , False , True , False , False] #!!!

#%% Specify filepaths. (USER INPUTS HERE)

# save directory for script outputs, recursively create directory if it doesn't exist
out_dir = 'dir/to/store/outputs' #!!!
if not os.path.exists(out_dir) :
    print(fg.magenta + 'Output directory not found, creating.' + fg.rs)
    os.makedirs(out_dir)
else :
    print(fg.green + 'Output directory found, proceeding.' + fg.rs)

# list of variables to save off (can view available variables in init_vars in next block below)
# different variables are stored in different ways - not all variables are supported by this script
# this script should work for any variables that are saved off for every element - some examples of working variables are given below
vars_out = [
    'DpEff' ,
    'ElementVolume' ,
    'LatticeOrientation' ,
    'ShearRate' ,
    'Stress' ,
    'XtalElasticStrain'
    ] #!!!

#%% Open ADIOS2 file and explore variables. (USER INPUTS HERE)

# open ADIOS2 binary-pack (.bp) file
fh = adios2.open('dir/to/exaconstit.bp' , 'r' , engine_type = 'BP4') #!!!
# list of variables stored in adios2 file
init_vars = fh.available_variables()
# total number of cycles saved off (+ initial step at time = 0)
# if stride > 1 in options.toml, this number will be (# of ExaConstit steps / stride) + 1
steps = fh.steps()

#%% Extract connectivity information - needed for other variables. (NO INPUTS HERE)

con1d = list()
index = np.zeros((nranks,2) , dtype = np.int32)
iend = 0

# Get the initial end node vertices and connectivity array for everything.
# ADIOS2 doesn't save higher order values, so only have the end nodes.
# Many repeat vertices, since these are saved off for each element.
for i in range(nranks) :
    
    if (i == 0) :
        
        # Pull out connectivity information.
        con = fh.read('connectivity' , block_id = i)
        
        # # Can uncomment to also pull out vertices.
        # vert = fh.read('vertices' , block_id = i)
        
        # # Can uncomment to also pull out grain IDs ('ElementAttribute').
        # grain = fh.read('ElementAttribute' , block_id = i)
        # grain = grain[con[:,1]]
        
        con1d.append(con[:,1])
        con = con[:,1::]
        
    else :
        
        # Pull out connectivity information.
        tmp = fh.read('connectivity' , block_id = i)
        con1d.append(tmp[:,1])

        # Connectivity is local to resource set rather than global, so increment to global.
        tmp = tmp + np.max(con)
        con = np.vstack((con , tmp[:,1::]))
        
        # # Can uncomment to also pull out vertices.
        # tmp = fh.read('vertices' , block_id = i)
        # vert = np.vstack((vert , tmp))
        
        # # Can uncomment to also pull out grain IDs ('ElementAttribute').
        # tmp = fh.read('ElementAttribute' , block_id = i)
        # grain = np.hstack((grain , tmp[con1d[i]]))
        
        del tmp

    # indexing variable that will be used later on
    index[i,0] = iend
    iend = con.shape[0]
    index[i,1] = iend

# # Can uncomment to convert grain IDs to int32.
# grain = np.int32(grain)

conshape = con.shape
del con

#%% Pull variables directly from adios2 files. (NO INPUTS HERE)

print(ef.bold + bg.green + '\nExtracting raw data.' + bg.rs + ef.rs)

for var in vars_out :
    
    print(ef.bold + bg.da_blue + '\nCurrent var: %s' %var + bg.rs + ef.rs)
    
    start = 0
    proceed = True
    
    # case handling for whether global bools are lists
    try :
        var_overwrite = overwrite[vars_out.index(var)]
    except :
        var_overwrite = overwrite
    
    try :
        var_resume = resume[vars_out.index(var)]
    except :
        var_resume = resume
    
    try :
        var_unpack_alt = unpack_alt[vars_out.index(var)]
    except :
        var_unpack_alt = unpack_alt
    
    # case handling for overwrite and resume bools
    if any(fname.startswith('raw_%s' %var) for fname in os.listdir(out_dir)) :
        if var_overwrite :
            if var_resume :
                for fname in os.listdir(out_dir) :
                    if fname.startswith('raw_%s' %var) :
                        start += 1
                start -= 1
                print(fg.red + 'File conflicts found for %s in output directory. overwrite = True and resume = True, starting from cycle %0.02d.' %(var , start) + fg.rs)
            else :
                print(fg.red + 'File conflicts found for %s in output directory. overwrite = True and resume = False, starting fresh.' %var + fg.rs)
        else :
            if var_resume :
                for fname in os.listdir(out_dir) :
                    if fname.startswith('raw_%s' %var) :
                        start += 1
                print(fg.magenta + 'File conflicts found for %s in output directory. overwrite = False and resume = True, starting from cycle %0.02d.' %(var , start) + fg.rs)
            else :
                print(fg.magenta + 'File conflicts found for %s in output directory. overwrite = False and resume = False, skipping variable.' %var + fg.rs)
                proceed = False
    else :
        print(fg.green + 'No conflicting files found for %s in output directory, proceeding.' %var + fg.rs)       
    
    # case handling for unpack_alt bool (whether to use alternate unpacking scheme)
    if proceed and not var_unpack_alt :
        
        # accounts for some variables starting at cycle 0 and others at cycle 1
        dif = steps - int(init_vars[var]['AvailableStepsCount'])
        if dif == 1 :
            offset = 1
        else :
            offset = 0
        del dif
        
        # initialize
        top = fh.read(var , block_id = 0)
        try :
            var_len = top.shape[1]
            var_raw = np.empty((var_len , conshape[0] , steps-offset) , order = 'F')
        except :
            var_len = 1
            var_raw = np.empty((conshape[0] , steps-offset) , order = 'F')
        del top
        
        # read
        print(fg.li_blue + '\nReading %s.' %var + fg.rs)
        for ii in range(nranks) :
            print('Loading resource set ' + str(ii+1) + ' of ' + str(nranks) + '.' , end = '\r')           
            isize = con1d[ii].shape[0] * conshape[1]
            if var_len > 1 :
                arr = fh.read(var , start = [0 , 0] , count = [isize , var_len] , step_start = 0 , step_count = steps-offset , block_id = ii)
                var_raw[: , index[ii , 0]:index[ii , 1] , :] = np.swapaxes(arr[:, con1d[ii] , :], 0 , 2)
            else :
                arr = fh.read(var , start = [0] , count = [isize] , step_start = 0 , step_count = steps-offset , block_id = ii).T
                var_raw[index[ii , 0]:index[ii , 1] , :] = arr[con1d[ii] , :]                
            del isize , arr
        print('\nDone reading %s.' %var)
        
        # save
        print(fg.li_blue + '\nSaving %s.' %var + fg.rs)
        if var_len > 1 :
            for jj in range(start , steps-offset) :
                print('Saving step ' + str(jj+1) + ' of ' + str(steps-offset) + '.' , end = '\r')
                np.savetxt(out_dir + 'raw_%s_%0.02d.txt' %(var , jj+offset) , var_raw[:,:,jj].T)
        else :
            for jj in range(start , steps-offset) :
                print('Saving step ' + str(jj+1) + ' of ' + str(steps-offset) + '.' , end = '\r')
                np.savetxt(out_dir + 'raw_%s_%0.02d.txt' %(var , jj+offset) , var_raw[:,jj].T)
        print('\nDone saving %s.' %var)
        del offset , var_len , var_raw
        
    elif proceed and var_unpack_alt :
        
        # accounts for some variables starting at cycle 0 and others at cycle 1
        dif = steps - int(init_vars[var]['AvailableStepsCount'])
        if dif == 1 :
            offset = 1
        else :
            offset = 0
        del dif
        
        # initialize
        top = fh.read(var , block_id = 0)
        try :
            var_len = top.shape[1]
            var_raw = np.empty((var_len , conshape[0] , 1) , order = 'F')
        except :
            var_len = 1
            var_raw = np.empty((conshape[0] , 1) , order = 'F')
        del top
        
        step = start
        
        # read and save
        while step < (steps-offset) :
            print('Processing step ' + str(step+1) + ' of ' + str(steps-offset) + '.' , end = '\r')
            for ii in range(nranks) :
                isize = con1d[ii].shape[0] * conshape[1]
                if var_len > 1 :
                    arr = fh.read(var , start = [0 , 0] , count = [isize , var_len] , step_start = step , step_count = 1 , block_id = ii)
                    var_raw[: , index[ii , 0]:index[ii , 1] , :] = np.swapaxes(arr[: , con1d[ii] , :] , 0 , 2)
                else :
                    arr = fh.read(var , start = [0] , count = [isize] , step_start = step , step_count = 1 , block_id = ii).T
                    var_raw[index[ii , 0]:index[ii , 1] , :] = arr[con1d[ii] , :]                
                del isize , arr
            if var_len > 1 :
                np.savetxt(out_dir + 'raw_%s_%0.02d.txt' %(var , step+offset) , var_raw[:,:,0].T)
            else :
                np.savetxt(out_dir + 'raw_%s_%0.02d.txt' %(var , step+offset) , var_raw[:,0].T)
            step += 1
        print('\nDone saving %s.' %var)
        del offset , var_len , var_raw , step
        
    del start , proceed
    
#%% Always close the file when you're finished loading data from it.

fh.close()