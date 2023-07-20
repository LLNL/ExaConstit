#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:21:16 2020

@author: carson16

Current version: v0.2
"""

import numpy as np
import pandas as pd
import argparse
import subprocess
import os

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

parser = argparse.ArgumentParser(description='ExaCA data to ExaConstit preprocessing data')

parser.add_argument(
       '-ifdir',
       '--input_file_dir',
       type=str,
       default='./',
       help='Directory of ExaCA data (default: ./)'
)

parser.add_argument(
       '-ifile',
       '--input_filename',
       type=str,
       default='exaca.csv',
       help='File name of ExaCA data (default: exaca.csv)'
)

parser.add_argument(
       '-orifile',
       '--unique_ori_filename',
       type=str,
       default='./uni_cubic_10k_quats.txt',
       help='File name / path of unique orientation file as quaternions used by ExaCA (default: ./uni_cubic_10k_quats.txt)'
)

parser.add_argument(
       '-ofdir',
       '--output_file_dir',
       type=str,
       default='./',
       help='Output directory of ExaConstit data (default: ./)'
)

parser.add_argument(
       '-c',
       '--coarsening',
       type=int,
       default=1,
       help='Level of coarsening to apply (default: 1)'
)

parser.add_argument(
       '-mg',
       '--mesh_generator',
       action="store_true",
       help='Run the mesh generator (default: False)'
)

parser.add_argument(
       '-mgdir',
       '--mesh_generator_dir',
       type=str,
       default='./',
       help='Mesh generator directory (default: ./)'
)

parser.add_argument(
       '-runame',
       '--rve_unique_name',
       type=str,
       default='simulation',
       help='Unique name for a given microstructure (default: simulation)'
)

parser.add_argument(
        '-t',
        '--temperature',
        nargs='+',
        default=[298.0],
        help='List of temperatures in Kelvin to read in (default: 298.0)'
)

parser.add_argument(
        '-fprops',
        '--property_file_names',
        nargs='+',
        default=['props.txt'],
        help='List of property file names to be used in simulations, and it should be same length as the temperature list (default: props.txt)'
)

parser.add_argument(
        '-nprops',
        '--num_properties',
        default=0,
        help='Number of properties in the property files (default: 0)'
)

parser.add_argument(
        '-fstates',
        '--state_file_names',
        nargs='+',
        default=['states.txt'],
        help='List of state file names to be used in simulations, and it should be same length as the temperature list (default: states.txt)'
)

parser.add_argument(
        '-nstates',
        '--num_states',
        default=0,
        help='Number of states variables in the state.txt file (default: 0)'
)

parser.add_argument(
       '-dtfile',
       '--dt_file_name',
       type=str,
       default='custom_dt_fine.txt',
       help='Custom dt time step file we want to use (default: custom_dt_fine.txt)'
)

parser.add_argument(
       '-ntsteps',
       '--number_time_steps',
       type=int,
       default=61,
       help='Number of time steps to be taken (default: 61)'
)

parser.add_argument(
        '-cfd',
        '--common_file_directory',
        type=str,
        default='../common_simulation_files/',
        help='Location of common/shared property/state/dt files for various RVE (default: ../common_simulation_files/).'
)

args = parser.parse_args()

#%%
# Input filename and directory
fdiri = os.path.abspath(args.input_file_dir)

fin = args.input_filename

fdirc = os.path.abspath(args.common_file_directory)
#%%
# Output filenames and directory
fdiro = os.path.abspath(args.output_file_dir)
fout = args.rve_unique_name
rve_unique_name = args.rve_unique_name
ori_out = fout + "_ori.txt"
gr_out = fout + "_grains.txt"

if not os.path.exists(fdiro):
    os.makedirs(fdiro)

dt_file = args.dt_file_name
dt_step = args.number_time_steps

#%%
# Temperatue ranges that simulations were run at
tempk = args.temperature
prop_files = args.property_file_names
num_props = args.num_properties
state_files = args.state_file_names
num_states = args.num_states

if(len(tempk) != len(prop_files)):
    raise ValueError('Temperature input and property file names not of the same length')

if(len(tempk) != len(state_files)):
    raise ValueError('Temperature input and state file names not of the same length')

#%%
# For our orientation data we want to be flexible for what ExaCA is using as they
# now have both a 10k and 1e6 unique orientation list and potentially this could change
# per simulation (hopefully not though). We used to require this to be uni_cubic_10k_quats.txt
# and then we assumed it was in the same file as the ExaCA data. Now we allow it to be anywhere
# and have any name. The only requirement is it must be a list of quaternions that are crystal to sample
# using passive rotations, and the quats must be equivalent of the rotation matrices that they used
# to run their simulations. This should always be the case as I've generated the rotation matrice files for them.
fh = os.path.abspath(args.unique_ori_filename)

ori_quat = np.loadtxt(fh)
ori_quat = ori_quat.T

#%%
# Time to read in our voxel data
fh = os.path.join(fdiri, os.path.basename(fin))
# ExaCA has the following headers
# Coordinates are in CA units, 1 cell = #.#### microns. Data is cell-centered. Origin at #,#,#
# X coord, Y coord, Z coord, Grain ID
# If we read in just the first line we can get out what the voxel size
# should be with the following set of code.
voxel_size = 1.0
with open(fh, "rt") as f:
    line = f.readline()
    sub_line = line.split("=")
    sub_line = sub_line[1]
    sub_line = sub_line.strip().split(" ")
    voxel_size = float(sub_line[0])
    print("Voxel size: " + str(voxel_size))
#%%
#Provided xyz gr#

gr_ori = pd.read_csv(fh, delimiter=",", dtype=np.int32, skiprows=1)
gr_ori = gr_ori.to_numpy()

gr_ori_min = np.min(gr_ori, axis=0)
gr_ori_max = np.max(gr_ori, axis=0)

# 1 is needed to get the correct number of voxels
gr_ori_diff = gr_ori_max - gr_ori_min + 1

#%%
# Need to figure out a good way to get the info related to
# the overall voxel shape of our input data. It's right now
# provided somewhat in a sentence at the top of the 
# ExaCA csv file
data = np.squeeze(gr_ori[:,3]).reshape((gr_ori_diff[0], gr_ori_diff[1], gr_ori_diff[2]))
print("Read in data")
#%%

# If we want to coarsen the data then we can do that down below.
# Checks should probably also be put into place to make sure
# that all of our voxel dimensions are divisible by the 
# coarsening level.
voxel = args.coarsening
cnx = data.shape[0]
cny = data.shape[1]
cnz = data.shape[2]

dnx = np.int32(cnx / voxel)
dny = np.int32(cny / voxel)
dnz = np.int32(cnz / voxel)

cdata = np.zeros((dnz, dny, dnx), dtype=np.int32)

for k in range(dnz):
    kz = k * voxel
    kz1 = kz + voxel
    for j in range(dny):
        jy = j * voxel
        jy1 = jy + voxel
        for i in range(dnx):
            ix = i * voxel
            ix1 = ix + voxel
            sub = data[kz:kz1, jy:jy1, ix:ix1]
            values, counts = np.unique(sub.flatten(), return_counts=True)
            indlog = counts == counts.max()
            size = np.sum(indlog)
            # Here we want to only pick one value
            if size > 1:
                arr = values[indlog]
                cdata[k, j, i] = arr[np.random.choice(size, 1)]
            else:
                cdata[k, j, i] = values[indlog]

#%%
print("Finished coarsening data")
# Here we find all of the unique grain numbers which correspond to 1-10k
# We then find what quaternions are available
ugr, ret_inv_gr, ret_cnts_gr = np.unique(cdata.flatten(), return_counts=True, return_inverse=True)
gr_num = np.abs(np.mod(ugr, 10000))
uori, ret_inv, ret_cnts = np.unique(gr_num, return_counts=True, return_inverse=True)
quats = ori_quat[:, gr_num[ret_inv]]

#%%
# This section is responsible for returning to a unique set of
# unit quaternions for all of our grains. Once, we have that
# the data can be saved off.
indlog = ret_cnts > 1
ind = np.r_[0:uori.shape[0]]
index = ind[indlog]
#
for i in index:
    ad = np.argwhere(ret_inv == i)
    if(ad.shape[0] == 2):
        quats[0, ad[1]] = 1.1 * quats[0, ad[1]]
    else:
        rands = np.random.uniform(0.9,1.1, (ad.shape[0],1))
        jj = 0
        for j in ad[1:]:
            quats[0, j] = rands[jj] * quats[0, j]
            jj += 1

quats = quats * np.tile(1.0 / np.linalg.norm(quats, axis=0), (4,1))

ngrains = quats.shape[1]

fh = os.path.join(fdiro, os.path.basename(ori_out))
np.savetxt(fh, quats.T)
print("Saved off orientation file")
#%%
# The grain numbers from the ExaCA simulation are most likely
# not sequential from 1..ngrains, so we need to go ahead and
# do that down below for the ExaConstit simulation.
# We can then save the data off.
ngrains = ugr.shape[0]
fh = os.path.join(fdiro, os.path.basename(gr_out))

vec = np.squeeze(cdata.flatten())
vec2 = np.copy(vec)

for i in range(ngrains):
    indlog = vec == ugr[i]
    vec2[indlog] = i + 1

np.savetxt(fh, vec2, fmt = "%d")
print("Saved off grain file")
#%%
# Since we have all of the relevant info about the mesh dimensions,
# we could just run the mesh generator down below as well...

print("Starting mesh generation")

# Our dimensions are usually in mm and our supplied voxel sizes from ExaCA are usually in microns.
# So, we need to divide by 1000 here.
lx = round(voxel_size * cnx, 1) / 1000.
ly = round(voxel_size * cny, 1) / 1000.
lz = round(voxel_size * cnz, 1) / 1000.

mesh_file_loc = os.path.join(os.path.abspath('./'), os.path.basename("simulation.mesh"))

if (args.mesh_generator):

    fhg = os.path.join(fdiro, os.path.basename(gr_out))
    fhm = os.path.join(fdiro, os.path.basename(fout + '.mesh'))

    with cd(args.mesh_generator_dir):
        cmd = './mesh_generator'
        args = '-nx ' + str(dnx) + ' -ny ' + str(dny) + ' -nz ' + str(dnx)
        args = args + ' -lx ' + str(lx) + ' -ly ' + str(ly) + ' -lz ' + str(lz)
        args = args + ' -grain ' + fhg
        args = args + ' -o ' + fhm
        args = args + ' -ord 1 -auto_mesh'
        cmd = cmd + ' ' + args
        print(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    mesh_file_loc = fhm

print("Starting calculation of work matrix")

# At this point this is quite manual. We could probably end up automating this construction later on if 
# we really wanted to

# The lists that we'll use to make our dataframe
loading_name = []
essential_vals = []
essential_ids = []
essential_comps = []
# These will be repeated in a loop
rve_name = []
ori_file_name = []
grain_num = []
temperature = []
mesh_file = []
lprop_file = []
nprop = []
lstate_file = []
nstate = []
ldt_file = []
ntstep = []

loading_dir_names_xy = ["x_0_y_90", "x_15_y_75", "x_30_y_60", "x_45_y_45", "x_60_y_30", "x_75_y_15", "x_90_y_0"]
loading_dir_names_xz = ["x_15_z_75", "x_30_z_60", "x_45_z_45", "x_60_z_30", "x_75_z_15", "x_90_z_0"]
loading_dir_names_yz = ["y_15_z_75", "y_30_z_60", "y_45_z_45", "y_60_z_30", "y_75_z_15"]
loading_dir_names_shear = ["shear_xy", "shear_xz", "shear_yz"]

loading_cosines_xy = [(0.0, 90.0, 90.0), (15.0, 75.0, 90.0), (30.0, 60.0, 90.0), (45.0, 45.0, 90.0), (60.0, 30.0, 90.0),
                      (75.0, 15.0, 90.0), (90.0, 0.0, 90.0)]
loading_cosines_xz = [(15.0, 90.0, 75.0), (30.0, 90.0, 60.0), (45.0, 90.0, 45.0), (60.0, 90.0, 30.0), (75.0, 90.0, 15.0),
                      (90.0, 90.0, 0.0)]
loading_cosines_yz = [(90.0, 15.0, 75.0), (90.0, 30.0, 60.0), (90.0, 45.0, 45.0), (90.0, 60.0, 30.0), (90.0, 75.0, 15.0)]

strain_rate = 0.001

vel_x = strain_rate * lx
vel_y = strain_rate * ly
vel_z = strain_rate * lz

essential_ids_xz   = str(np.array2string(np.asarray([1, 2, 3, 4, 5]), separator=', '))
essential_comps_xz = str(np.array2string(np.asarray([3, 1, 2, 3, 1]), separator=', '))
essential_comps_xz_fx = str(np.array2string(np.asarray([3, 1, 2, 3, 0]), separator=', '))
essential_vals_xz  = np.asarray([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, vel_z], 
                                [vel_x, 0.0, 0.0]])

# rve_unique_name, ori_file_name, ngrains, tempk,
# prop_file_loc, nprops, state_file_loc, nstates,
# ess_id_array, ess_comp_array, ess_vals_array,
# mesh_file_loc

for iname in range(len(loading_dir_names_xz)):
    arr = np.copy(essential_vals_xz)
    xd, yd, zd = loading_cosines_xz[iname]
    arr[:, 0] = arr[:, 0] * np.cos(np.deg2rad(xd))
    arr[:, 2] = arr[:, 2] * np.cos(np.deg2rad(zd))
    arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
    loading_name.append(loading_dir_names_xz[iname])
    essential_vals.append(arr1d_str)
    if(iname < len(loading_dir_names_xz) - 1):
        essential_comps.append(essential_comps_xz)
    else:
        essential_comps.append(essential_comps_xz_fx)
    essential_ids.append(essential_ids_xz)


essential_ids_yz   = str(np.array2string(np.asarray([1, 2, 3, 4, 6]), separator=', '))
essential_comps_yz = str(np.array2string(np.asarray([3, 1, 2, 3, 2]), separator=', '))
essential_vals_yz  = np.asarray([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, vel_z], 
                                 [0.0, vel_y, 0.0]])

for iname in range(len(loading_dir_names_yz)):
    arr = np.copy(essential_vals_yz)
    xd, yd, zd = loading_cosines_yz[iname]
    arr[:, 1] = arr[:, 1] * np.cos(np.deg2rad(yd))
    arr[:, 2] = arr[:, 2] * np.cos(np.deg2rad(zd))
    arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
    loading_name.append(loading_dir_names_yz[iname])
    essential_vals.append(arr1d_str)
    essential_ids.append(essential_ids_yz)
    essential_comps.append(essential_comps_yz)

essential_ids_xy   = str(np.array2string(np.asarray([1, 2, 3, 5, 6]), separator=', '))
essential_comps_xy = str(np.array2string(np.asarray([3, 1, 2, 1, 2]), separator=', '))
essential_comps_xy_fy = str(np.array2string(np.asarray([3, 1, 2, 1, 0]), separator=', '))
essential_comps_xy_fx = str(np.array2string(np.asarray([3, 1, 2, 0, 2]), separator=', '))
essential_vals_xy  = np.asarray([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [vel_x, 0.0, 0.0], 
                                 [0.0, vel_y, 0.0]])

for iname in range(len(loading_dir_names_xy)):
    arr = np.copy(essential_vals_xy)
    xd, yd, zd = loading_cosines_xy[iname]
    arr[:, 0] = arr[:, 0] * np.cos(np.deg2rad(xd))
    arr[:, 1] = arr[:, 1] * np.cos(np.deg2rad(yd))
    arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
    loading_name.append(loading_dir_names_xy[iname])
    essential_vals.append(arr1d_str)
    essential_ids.append(essential_ids_xy)
    if(iname == 0):
        essential_comps.append(essential_comps_xy_fy)
    elif(iname < len(loading_dir_names_xy) - 1):
        essential_comps.append(essential_comps_xy)
    else:
        essential_comps.append(essential_comps_xy_fx)

essential_ids_shear_xy   = str(np.array2string(np.asarray([2, 5]), separator=', '))
essential_comps_shear_xy = str(np.array2string(np.asarray([-1, -1]), separator=', '))
essential_vals_shear_xy  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, vel_y, 0.0]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

loading_name.append(loading_dir_names_shear[0])
essential_vals.append(essential_vals_shear_xy)
essential_ids.append(essential_ids_shear_xy)
essential_comps.append(essential_comps_shear_xy)

essential_ids_shear_xz   = str(np.array2string(np.asarray([2, 5]), separator=', '))
essential_comps_shear_xz = str(np.array2string(np.asarray([-1, -1]), separator=', '))
essential_vals_shear_xz  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, vel_z]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

loading_name.append(loading_dir_names_shear[1])
essential_vals.append(essential_vals_shear_xz)
essential_ids.append(essential_ids_shear_xz)
essential_comps.append(essential_comps_shear_xz)

essential_ids_shear_yz   = str(np.array2string(np.asarray([3, 6]), separator=', '))
essential_comps_shear_yz = str(np.array2string(np.asarray([-1, -1]), separator=', '))
essential_vals_shear_yz  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, vel_z]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

loading_name.append(loading_dir_names_shear[2])
essential_vals.append(essential_vals_shear_yz)
essential_ids.append(essential_ids_shear_yz)
essential_comps.append(essential_comps_shear_yz)

nruns = len(loading_name)

lloading_name = loading_name.copy()
lessential_vals = essential_vals.copy()
lessential_ids = essential_ids.copy()
lessential_comps = essential_comps.copy()

for itemp in range(len(tempk)):
    for i in range(nruns):
        rve_name.append(rve_unique_name)
        fho = os.path.join(fdiro, os.path.basename(ori_out))
        ori_file_name.append(fho)
        grain_num.append(ngrains)
        temperature.append(tempk[itemp])
        mesh_file.append(mesh_file_loc)
        fhp = os.path.join(fdirc, os.path.basename(prop_files[itemp]))
        lprop_file.append(fhp)
        fhs = os.path.join(fdirc, os.path.basename(state_files[itemp]))
        lstate_file.append(fhs)
        fhc = os.path.join(fdirc, os.path.basename(dt_file))
        ldt_file.append(fhc)
        nprop.append(num_props)
        nstate.append(num_states)
        ntstep.append(dt_step)
    if(itemp > 0):
        lloading_name.extend(loading_name)
        lessential_vals.extend(essential_vals)
        lessential_ids.extend(essential_ids)
        lessential_comps.extend(essential_comps)

data = {"rve_unique_name" : rve_name, "ori_file_name" : ori_file_name, "ngrains" : grain_num, "tempk" : temperature,
        "prop_file_loc" : lprop_file, "nprops" : nprop, "state_file_loc" : lstate_file, "nstates" : nstate,
        "ess_id_array" : lessential_ids, "ess_comp_array" : lessential_comps, "ess_vals_array" : lessential_vals,
        "loading_name" : lloading_name, "mesh_file_loc" : mesh_file, "dt_file" : ldt_file, "dt_steps" : ntstep}

df = pd.DataFrame(data)

fhdf = os.path.join(fdiro, os.path.basename(fout+'_test_matrix.csv'))

df.to_csv(fhdf)
