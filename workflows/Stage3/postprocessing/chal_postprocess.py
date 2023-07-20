#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:11:28 2023

@author: carson16
"""
import numpy as np
import scipy.stats as scist
import pandas as pd

import os
import pickle
import math


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

                
if __name__ == "__main__":
    coarse_levels = [2, 4, 5, 8, 10]
    sy_zz  = np.asarray([-750.0, -741.26, -741.24, -686.13])
    sy_yy  = np.asarray([-740.0, -726.19, -720.87, -675.80]) 
    s5_zz  = np.asarray([-951.0, -967.0, -951.0, -916.0])
    s5_yy  = np.asarray([-915.0, -921.0, -908.0, -858.2])
    
    results = dict()
    chal_df_dicts = dict()
    super_err_df = None
    fdir = "/lustre/orion/world-shared/mat190/exaam-challenge-problem-dummy/workflow_files_frontier/ExaConstit/postprocessing/"
    
    for ref_layer in coarse_levels:
        print("Starting coarse layer: c" +str(ref_layer))
        rve_data = fdir+"cs_pickle/rve_chal_ss_vals_xyz_c"+str(ref_layer)+".pickle"
        with open(rve_data, "rb") as f_handle:
                chal_df = pickle.load(f_handle)
                
        yy_df = chal_df[chal_df["loading_name"].str.contains("x_90_y_0")]
        yy_df = yy_df.copy()
        zz_df = chal_df[chal_df["loading_name"].str.contains("x_90_z_0")]
        zz_df = zz_df.copy()
        
        npts = sy_zz.shape[0]
        nrve = zz_df["yield_stress"].shape[0]
        
        sy_err = np.tile(zz_df["yield_stress"], (npts,1))  - np.tile(sy_zz, (nrve, 1)).T
        sy_err_zz = (sy_err.T / np.tile(sy_zz, (nrve, 1)))
        
        sy_err = np.tile(yy_df["yield_stress"], (npts,1))  - np.tile(sy_yy, (nrve, 1)).T
        sy_err_yy = (sy_err.T / np.tile(sy_yy, (nrve, 1)))
        
        sy_err = np.tile(zz_df["eng_stress_5"], (npts,1))  - np.tile(s5_zz, (nrve, 1)).T
        s5_err_zz = (sy_err.T / np.tile(s5_zz, (nrve, 1)))
        
        sy_err = np.tile(yy_df["eng_stress_5"], (npts,1))  - np.tile(s5_yy, (nrve, 1)).T
        s5_err_yy = (sy_err.T / np.tile(s5_yy, (nrve, 1)))
        
        min_sy_err_zz = np.min(np.abs(sy_err_zz),axis=1)
        ind = np.argmin(np.abs(sy_err_zz),axis=1)
        for i in range(125):
            min_sy_err_zz[i] *= np.sign(sy_err_zz[i, ind[i]])
        
        min_sy_err_yy = np.min(np.abs(sy_err_yy),axis=1)
        ind = np.argmin(np.abs(sy_err_yy),axis=1)
        for i in range(125):
            min_sy_err_yy[i] *= np.sign(sy_err_yy[i, ind[i]])
        
        min_s5_err_zz = np.min(np.abs(s5_err_zz),axis=1)
        ind = np.argmin(np.abs(s5_err_zz),axis=1)
        for i in range(125):
            min_s5_err_zz[i] *= np.sign(s5_err_zz[i, ind[i]])
        
        min_s5_err_yy = np.min(np.abs(s5_err_yy),axis=1)
        ind = np.argmin(np.abs(s5_err_yy),axis=1)
        for i in range(125):
            min_s5_err_yy[i] *= np.sign(s5_err_yy[i, ind[i]])
            
            
        zz_df["min_rel_err_yld_stress"]   = min_sy_err_zz
        zz_df["min_rel_err_eng_stress_5"] = min_s5_err_zz
        
        yy_df["min_rel_err_yld_stress"]   = min_sy_err_yy
        yy_df["min_rel_err_eng_stress_5"] = min_s5_err_yy
        
        yz_dict = {"load_zz_df" : zz_df, "load_yy_df" : yy_df}
        
        chal_df_dicts["c"+str(ref_layer)] = yz_dict
        
    #%%
        major_minor_label = list()
        err = np.zeros(nrve)
        zz_err = np.zeros(nrve)
        yy_err = np.zeros(nrve)
        ind = 0
        coarsen = list()
        zz_yield = np.zeros(nrve)
        yy_yield = np.zeros(nrve)
        zz_5ss = np.zeros(nrve)
        yy_5ss = np.zeros(nrve)
        for index, row in zz_df.iterrows():
            major_minor_label.append(row["major_minor_base_name"])
            row2 = yy_df[yy_df["major_minor_base_name"] == row["major_minor_base_name"]]
            
            zz_err_val = row["min_rel_err_yld_stress"]**2 + row["min_rel_err_eng_stress_5"]**2
            yy_err_val = row2["min_rel_err_yld_stress"]**2 + row2["min_rel_err_eng_stress_5"]**2
            # Doing a simple root mean square relative error (RMSRE) calculation 
            zz_err[ind] = np.sqrt(zz_err_val * 0.25)
            yy_err[ind] = np.sqrt(yy_err_val * 0.25)
            err[ind] = np.sqrt((yy_err_val + zz_err_val) * 0.25)
            
            coarsen.append("C" + str(ref_layer))
            zz_yield[ind] = row["min_rel_err_yld_stress"]
            zz_5ss[ind] = row["min_rel_err_eng_stress_5"]
            yy_yield[ind] = row2["min_rel_err_yld_stress"]
            yy_5ss[ind] = row2["min_rel_err_eng_stress_5"]
            
            ind += 1
            
        err_data = {"major_minor_base_name" : major_minor_label,
                    "RMSRE" : err, "ZZ RMSRE" : zz_err, "YY RMSRE" : yy_err,
                    "min_zz_yield_err" : zz_yield, "min_zz_eng_stress_5_err" : zz_5ss,
                    "min_yy_yield_err" : yy_yield, "min_yy_eng_stress_5_err" : yy_5ss,
                    "Coarsen Level" : coarsen}
        
        err_df = pd.DataFrame(data=err_data)
        
        results["c"+str(ref_layer)] = err_df
#%%      
    super_err_df = pd.concat([results["c2"], results["c4"], 
                              results["c5"], results["c8"], 
                              results["c10"]])
    
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)

    ptot = sns.kdeplot(data=super_err_df, x="RMSRE", clip=(0, 0.1), hue="Coarsen Level", palette="viridis", ax=ax1)
    pzz = sns.kdeplot(data=super_err_df, x="ZZ RMSRE", clip=(0, 0.1), hue="Coarsen Level", palette="viridis", ax=ax2)
    pyy = sns.kdeplot(data=super_err_df, x="YY RMSRE", clip=(0, 0.1), hue="Coarsen Level", palette="viridis", ax=ax3)

    line_style_types = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('long dash with offset', (5, (10, 3))),
     ('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),    
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    
    lss = [style for (name, style) in line_style_types]
    
    handles = ptot.legend_.legendHandles[::-1]  
    for line, ls, handle in zip(ptot.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)
        
    handles = pzz.legend_.legendHandles[::-1]  
    for line, ls, handle in zip(pzz.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    handles = pyy.legend_.legendHandles[::-1]  
    for line, ls, handle in zip(pyy.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)  
      
    ax1.set_xlabel('Root Mean Square Relative Error (-)')
    ax1.set_ylabel('Density (-)')
    ax1.axis([0, 0.05, 0, 30.0])
    
    ax2.set_xlabel('ZZ Root Mean Square Relative Error (-)')
    ax2.set_ylabel('Density (-)')
    ax2.axis([0, 0.05, 0, 30.0])
    
    ax3.set_xlabel('YY Root Mean Square Relative Error (-)')
    ax3.set_ylabel('Density (-)')
    ax3.axis([0, 0.05, 0, 30.0])
    
    picLoc = fdir+'total_rmse_kde_plot.png'
    fig1.savefig(picLoc, dpi = 300, bbox_inches='tight')
    
    picLoc = fdir+'zz_rmse_kde_plot.png'
    fig2.savefig(picLoc, dpi = 300, bbox_inches='tight')
    
    picLoc = fdir+'yy_rmse_kde_plot.png'
    fig3.savefig(picLoc, dpi = 300, bbox_inches='tight')
    
    plt.show()
    
