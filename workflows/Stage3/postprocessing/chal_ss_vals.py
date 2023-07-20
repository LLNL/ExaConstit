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

def yield_stress_calc_5_strain(dfs, uq_dfs, fdir_base, ftime, strain_rate=0.001):
    
    stress_bname    = "avg_stress_"
    plwork_bname    = "avg_pl_work_"
    dp_tensor_bname = "avg_dp_tensor_"
    
    load_names = ["x_90_y_0", "x_90_z_0", "x_0_y_90"]
    ss_inds = [1, 2, 0]
    
    nrves = len(dfs)
    print(nrves)
    results = {}
    
    time = np.loadtxt(ftime)
    nsteps = time.shape[0]

    data = { "major_minor_base_name" : [],
             "major_base_name" : [],
             "minor_base_name" : [],
             "rve_name"  : [],
             "adfoam_radius" : [],
             "exaca_density" : [],
             "exaca_stddev" : [],
             "temperature" : [],
             "loading_name" : [],
             "yield_stress" : [],
             "eng_stress_5" : [],
             "true_stress_5eng" : []
            }

    eps = np.zeros(nsteps)
    for i in range(0, nsteps):
        dtime = time[i]
        eps[i] = eps[i - 1] + strain_rate * dtime
    
    for irve in range(nrves):
        local_df = dfs[irve]

        fdirs = os.path.abspath(fdir_base)        
        frve =  local_df["rve_unique_name"][0]
        fdiro = os.path.join(fdirs, frve, "")

        print("Working on rve name " + frve)
        
        for ind_ss, name in zip(ss_inds, load_names):
            print("Working on loading case " + name)
            pdf = local_df[local_df["loading_name"].str.contains(name)]
            nruns = pdf.shape[0]
            
            for irun in range(nruns):
                rve_name = pdf["rve_unique_name"].values[irun]
                
                load_dir_name = pdf["loading_name"].values[irun]
                temp_k = str(int(pdf["tempk"].values[irun]))
                print("Working on tempk: " + str(temp_k))
                if temp_k != str(int(298)):
                    continue
                fdiron = fdiro
                fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")

                ext_name = rve_name+"_"+temp_k+"_"+name+'.txt'            
                stress = np.loadtxt(fdironl+stress_bname+ext_name)
                nsteps = stress.shape[0]
                pl_work = np.loadtxt(fdironl+plwork_bname+ext_name)
                
                # This part is a bit manual at this point
                # We might need a finer dt set for the bi-axial load set then what we currently
                # use for the monotonic loading cases
                slope, intercept, r, p, se = scist.linregress(eps[0:9], np.abs(stress[0:9, ind_ss]))
                stress_offset = slope * (eps - 0.002)
                
                for j in range(2, nsteps):
                    if stress_offset[j] > np.abs(stress[j, ind_ss]):
                        # J would be our point of yield
                        sx1 = eps[j - 1]
                        sy1 = np.abs(stress[j - 1, ind_ss])
                        sx2 = eps[j]
                        sy2 = np.abs(stress[j, ind_ss])
                        oy1 = stress_offset[j - 1]
                        oy2 = stress_offset[j]
                        plwork_driver = pl_work[j]
                        break
                    
                YS = ((sx1 * oy2 - sx2 * oy1) * (sy1 - sy2) - (oy1 - oy2) * (sx1 * sy2 - sx2 * sy1)) / ((sx1 - sx2) * (sy1 - sy2) - (sx1 - sx2) * (oy1 - oy2))
                print([YS, plwork_driver])
                log_eps = np.log(1.0 + (eps * np.sign(stress[-1, ind_ss])))
                eng_ss = stress[:, ind_ss] / (1.0 + log_eps[0:nsteps])
                ind_step = np.argmin(np.abs(eps[0:nsteps] - 0.05))
                true_ss = stress[:, ind_ss]
                print("Engineering stress -5.0% engineering strain: {:.2f}".format(eng_ss[ind_step]))
                print("True stress {:.2f}".format(log_eps[ind_step]*100.0) + "% true strain: {:.2f}".format(true_ss[ind_step]))
                
                major_minor = rve_name.split("_")[1]
                major = major_minor.split(".")[0]
                minor = major_minor.split(".")[1]

                data["major_minor_base_name"].append(major_minor)
                data["major_base_name"].append(major)
                data["minor_base_name"].append(minor)

                uq_case = uq_dfs[uq_dfs["caseID"].str.contains(major_minor)]
                data["adfoam_radius"].append(uq_case["radius"].values[0])
                data["exaca_density"].append(uq_case["Density"].values[0])
                data["exaca_stddev"].append(uq_case["StDev"].values[0])
                
                data["rve_name"].append(rve_name)
                data["temperature"].append(temp_k)
                data["loading_name"].append(name)
                data["yield_stress"].append(YS*np.sign(stress[-1, ind_ss]))
                data["eng_stress_5"].append(eng_ss[ind_step])
                data["true_stress_5eng"].append(true_ss[ind_step])
                
    chal_df = pd.DataFrame(data=data)
        
    return chal_df
                
if __name__ == "__main__":
    fdir_wf = "/lustre/orion/world-shared/mat190/exaam-challenge-problem/CY22-DEMO/cases/exaconstit_chal_mini_xyz/"
    rve_test_file = fdir_wf + "rve_test_matrices.pickle"
    with open(rve_test_file, "rb") as f_handle:
            dfs = pickle.load(f_handle)
    fdir_rve = "/lustre/orion/world-shared/mat190/exaam-challenge-problem-dummy/workflow_files_frontier/ExaConstit/common_simulation_files/"
    ftime = os.path.join(fdir_rve, "custom_dt_fine2.txt")

    fh = "/lustre/orion/world-shared/mat190/exaam-challenge-problem/CY22-DEMO/parameters.csv"
    uq_dfs = pd.read_csv(fh, dtype=str)
    
    chal_df = yield_stress_calc_5_strain(dfs, uq_dfs, fdir_wf, ftime, 0.001)
    
    with open("./rve_chal_ss_vals_xyz.pickle", "wb") as f_handle:
            pickle.dump(chal_df, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
