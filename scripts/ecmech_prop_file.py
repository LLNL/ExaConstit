#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Whether or not you're using voce models
voce = False
# File location
floc = '/some/location/for/param/file/'
# File name
fname = 'props_cp_mts.txt'
params = []
# Params start off with:
# initial density, heat capacity at constant volume,
# and a tolerance param
density = 8.920e-6 #g/cubic cm
heat_cap = 0.003435984 #J/Kg-°C
tol = 1e-10 #is usually good here
params.extend([density, heat_cap, tol])
# Elastic Constants:
# (c11, c12, c44 for Cubic crystals) or
# (c11, c12, c13, c33, and c44) for Hexagonal crystals
c11 = 168.4 # GPa
c12 = 121.4 # GPa
c44 = 75.2 # GPa
elastic_const = [c11, c12, c44]
params.extend(elastic_const)
#Calculation of an average shear modulus value for Cubic materials
mu = (c11 - c12) / 2.0
nu = c44
voigt_shear = 0.2 * (2.0 * mu + 3.0 * nu)
reuss_shear = (mu * nu) / (nu + 3.0 * (mu - nu) * 0.2 )
#Shear modulus calculation if not available in literature
avg_shear = (voigt_shear + reuss_shear) / 2.0 # GPa

#Temperature simulation is running at
ref_temp = 300. # Kelvin
#Gruneisen parameter
gruneisen_param = 0.0
#Internal energy reference
int_eng_ref = -heat_cap * ref_temp # J / kg

hard_params = []
hard_params.append(avg_shear)
# Voce hardening and power law slip
# Note HCP is not supported with this model
# Params then include the following:
# shear modulus, m parameter seen in slip kinetics, gdot_0 term found in slip kinetic eqn,
# hardening coeff. defined for g_crss evolution eqn, initial CRSS value,
# initial CRSS saturation strength, CRSS saturation strength scaling exponent,
# CRSS saturation strength rate scaling coeff, tausi -> hdn_init (not used)
if(voce):
   m_exp = 0.02#
   hard_params.append(m_exp)
   gdot0 = 1.0#
   hard_params.append(gdot0)
   hard_coef = 400e-3#
   hard_params.append(hard_coef)
   crss0 = 17e-3#
   hard_params.append(crss0)
   crss_sat = 122.4e-3#
   hard_params.append(crss_sat)
   crss_sat_scal_exp = 0.0#
   hard_params.append(crss_sat_scal_exp)
   crss_sat_scal_coef = 5.0e9#
   hard_params.append(crss_sat_scal_exp)
   hdn_init = crss0
   hard_params.append(hdn_init)
#MTS slip and Kocks-Mecking dislocation density hardening model
# reference shear modulus, reference temperature, g_0 * b^3 / \kappa where b is the
# magnitude of the burger's vector and \kappa is Boltzmann's constant**,
# Peierls barrier, MTS curve shape parameter (p), MTS curve shape parameter (q),
# reference thermally activated slip rate, reference drag limited slip rate,
# drag reference stress, slip resistance const (g_0)**, slip resistance const (s)**,
# dislocation density production constant (k_1), dislocation density production
# constant (k_{2_0}), dislocation density exponential constant,
# reference net slip rate constant, reference relative dislocation density
# **These values are defined for all 24 HCP slip systems so make sure
#   they're all defined here if using HCP materials
else:
   rt = ref_temp
   hard_params.append(rt)
   # This param is defined for each slip system for HCP so
   # you need to either build it up as one big array and
   # append it to to the hard_params or just loop through
   # and append the same value to hard_params for the number
   # of slip systems.
   c1 = 1.944106926e3
   hard_params.append(c1)
   tau_pb = 4.0e-4
   hard_params.append(tau_pb)
   p = 1.0
   hard_params.append(p)
   q = 1.0
   hard_params.append(q)
   ref_gdot_therm = 1.0
   hard_params.append(ref_gdot_therm)
   ref_gdot_drag = 1.0
   hard_params.append(ref_gdot_drag)
   ref_drag_stress = 3.0e-2
   hard_params.append(ref_drag_stress)
   # This param is defined for each slip system for HCP so
   # you need to either build it up as one big array and
   # append it to to the hard_params or just loop through
   # and append the same value to hard_params for the number
   # of slip systems.
   slip_resist_const_g0 = 8.0e-3
   hard_params.append(slip_resist_const_g0)
   # This param is defined for each slip system for HCP so
   # you need to either build it up as one big array and
   # append it to to the hard_params or just loop through
   # and append the same value to hard_params for the number
   # of slip systems.
   slip_resist_const_s = 1.0e-1
   hard_params.append(slip_resist_const_s)
   k1 = 3.0e-4
   hard_params.append(k1)
   k2_0 = 5.0e-5
   hard_params.append(k2_0)
   ninv = 0.1
   hard_params.append(ninv)
   ref_slip_rate_const = 1.0e-2
   hard_params.append(ref_slip_rate_const)
   ref_rel_dis_dens = 9.0e-4
   hard_params.append(ref_rel_dis_dens)

params.extend(hard_params)
# Params then include the following:
# the Grüneisen parameter, reference internal energy
params.extend([gruneisen_param, int_eng_ref])

np.asarray(params)
np.savetxt(floc + fname, params)