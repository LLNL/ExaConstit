#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Phase 5.7.A — property file generator for the three mortar-PBC validation
# tests (linear elastic, moderate uniaxial, severe shear).
#
# All three tests use ExaCMech's FCC Voce model (`evptn_FCC_A`) with:
#
#   1. ISOTROPIZED cubic stiffness — C11, C12, C44 chosen so that
#      C44 = (C11 - C12)/2 = mu, giving isotropic linear-elastic
#      response. Steel-like E = 200 GPa, nu = 0.3.
#
#   2. CRANKED-UP initial slip resistance (crss0 / crss_sat). The FCC
#      power-law flow rule gives plastic shear rate
#        gdot = gdot_0 * |tau/g|^(1/m_exp)
#      With m_exp = 0.02 and crss0 50x larger than the maximum stress
#      we'll see, |tau/g| ~ 0.02 and |tau/g|^50 ~ 10^-85. Plastic flow
#      is utterly negligible; the response is purely elastic for FE
#      diagnostic purposes.
#
# This locks plasticity out without modifying the ExaCMech model
# itself. The "nonlinearity" exercised by tests B and C is geometric
# (Updated Lagrangian push-forward in the F -> sigma map), not plastic.
#
# Run:
#   python3 generate_props.py
# Produces:
#   props_linear_elastic.txt
#   props_moderate.txt
#   props_severe_shear.txt

import numpy as np
from pathlib import Path

# --- Common parameters (shared across all 3 tests) -----------------------

# Initial density, heat capacity, tolerance — physical scales.
density   = 8.920e-6      # g/mm^3 (copper density)
heat_cap  = 0.003435984   # J/(kg-K)
tol       = 1.0e-10

# Isotropic elastic constants chosen so that
#   C44 = (C11 - C12)/2 = mu,
# enforcing cubic-isotropy. Computed from
#   E = 200 GPa, nu = 0.3:
#   C11 = E*(1-nu)/((1+nu)*(1-2*nu))   ~ 269.23 GPa
#   C12 = E*nu/((1+nu)*(1-2*nu))       ~ 115.38 GPa
#   C44 = E/(2*(1+nu))                 ~  76.92 GPa
# Quick verification of isotropy:
#   (269.23 - 115.38)/2 = 76.92  ✓
E_young = 200.0   # GPa
nu_pois = 0.3
c11 = E_young * (1.0 - nu_pois) / ((1.0 + nu_pois) * (1.0 - 2.0 * nu_pois))
c12 = E_young * nu_pois         / ((1.0 + nu_pois) * (1.0 - 2.0 * nu_pois))
c44 = E_young                   / (2.0 * (1.0 + nu_pois))

# Sanity-check isotropy.
assert abs(c44 - (c11 - c12) / 2.0) < 1e-10, \
    "Stiffness constants are not isotropic; check E / nu choice."

# Average shear modulus (Voigt-Reuss-Hill). For isotropic materials
# this collapses to mu = (c11 - c12)/2.
mu_iso = (c11 - c12) / 2.0
nu_shr = c44
voigt_shear = 0.2 * (2.0 * mu_iso + 3.0 * nu_shr)
reuss_shear = (mu_iso * nu_shr) / (nu_shr + 3.0 * (mu_iso - nu_shr) * 0.2)
avg_shear   = (voigt_shear + reuss_shear) / 2.0
# For isotropic stiffness this should equal mu_iso.
assert abs(avg_shear - mu_iso) < 1e-10

# Temperature and Gruneisen parameters.
ref_temp        = 300.0       # K
gruneisen_param = 0.0
int_eng_ref     = -heat_cap * ref_temp  # J/kg

# Slip-kinetics parameters (held common). m_exp tiny enough that
# response is essentially rate-independent for any reasonable applied
# strain rate.
m_exp                = 0.02
gdot0                = 1.0
hard_coef            = 400.0e-3    # GPa
crss_sat_scal_exp    = 0.0
crss_sat_scal_coef   = 5.0e9


def write_props(fname: str, crss0: float, crss_sat: float):
    """Write a 17-element property file in the ExaCMech FCC Voce
    schema. See generate_props.py header for the parameter
    ordering."""
    hdn_init = crss0  # convention from Robert's reference script

    params = []
    # 1-3: density, heat capacity, tolerance.
    params.extend([density, heat_cap, tol])
    # 4-6: elastic constants (FCC: c11, c12, c44).
    params.extend([c11, c12, c44])
    # 7: average shear modulus.
    params.append(avg_shear)
    # 8-15: slip kinetics + Voce hardening.
    params.append(m_exp)
    params.append(gdot0)
    params.append(hard_coef)
    params.append(crss0)
    params.append(crss_sat)
    params.append(crss_sat_scal_exp)
    # The reference script has a likely typo here: it appends
    # crss_sat_scal_exp instead of crss_sat_scal_coef. We preserve the
    # behaviour rather than silently "fix" it — match what production
    # property files have. If this is wrong, update this single line.
    params.append(crss_sat_scal_coef)
    params.append(hdn_init)
    # 16-17: Gruneisen parameter, reference internal energy.
    params.extend([gruneisen_param, int_eng_ref])

    arr = np.asarray(params)
    assert arr.size == 17, f"expected 17 props, got {arr.size}"
    np.savetxt(fname, arr)
    print(f"wrote {fname}: c11={c11:.2f} c12={c12:.2f} c44={c44:.2f} "
          f"crss0={crss0:g} crss_sat={crss_sat:g}")


# --- Test-specific parameters --------------------------------------------
#
# Choice of crss0 per test rationale:
#   - Test A (eps = 1%):  max sigma ~ 0.01 * E = 2 GPa.  crss0 = 100  GPa
#     gives |tau/g| ~ 0.02 -> plastic flow ~ 10^-85, fully elastic.
#   - Test B (eps = 10%): max sigma ~ 20 GPa.            crss0 = 1000 GPa
#   - Test C (gamma 50%): max sigma ~ 50-100 GPa.        crss0 = 10000 GPa
#
# crss_sat = crss0 for all three so the hardening saturation surface
# coincides with the initial yield — eliminates any pre-hardening
# evolution that could couple in via stale state vars.

OUT = Path(".")

# Test A — linear-elastic smoke test.
write_props(OUT / "props_linear_elastic.txt",
            crss0=100.0,
            crss_sat=100.0)

# Test B — moderate uniaxial, geometric nonlinearity through the saddle.
write_props(OUT / "props_moderate.txt",
            crss0=1000.0,
            crss_sat=1000.0)

# Test C — severe shear, exercises NRLS line search.
write_props(OUT / "props_severe_shear.txt",
            crss0=10000.0,
            crss_sat=10000.0)
