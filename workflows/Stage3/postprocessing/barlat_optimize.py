#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:37:37 2021

@author: carson16
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root
import scipy.stats as scist

import argparse
#import subprocess
import os
import glob

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# Some more helper functions to go from voigt to matrix format and vice versa
def voigtNotation(mat):
    return np.asarray([mat[0, 0], mat[1, 1], mat[2, 2], mat[1, 2], mat[0, 2], mat[0, 1]])

def matNotation(voigt):
    return np.asarray([[voigt[0], voigt[5], voigt[4]],
                       [voigt[5], voigt[1], voigt[3]],
                       [voigt[4], voigt[3], voigt[2]]])

def effectiveTerm(mat):
    term1 = mat[0, 0] - mat[1, 1]
    term2 = mat[1, 1] - mat[2, 2]
    term3 = mat[2, 2] - mat[0, 0]
    term4 = mat[1, 2] * mat[1, 2] \
          + mat[0, 2] * mat[0, 2] \
          + mat[0, 1] * mat[0, 1]

    term1 *= term1
    term2 *= term2
    term3 *= term3
    term4 *= 6.0

    return np.sqrt(0.5 * (term1 + term2 + term3 + term4))

# A few helper functions to compute our various tensors down below
def outer4tensor(vec1, vec2, vec3, vec4):
    return np.einsum('i,j,k,l->ijkl', vec1, vec2, vec3, vec4)

# Takes a symmetric 4d matrix and transforms it into its
# 2d voigt form
def full_4d_to_Voigt_2d(C):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    tol = 1e-14

    Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    C = np.asarray(C)
    Voigt = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i,j] = C[k,l,m,n]

    return Voigt

# This function returns the 4th order deviatoric identity matrix
# This is used to convert a 3x3 matrix into it's deviatoric form
def ItenMat():
    Iten = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            if i == j:
                delij = 1.
            else:
                delij = 0.
            for k in range(3):
                if i == k:
                    delik = 1.
                else:
                    delik = 0.
                if j == k:
                    deljk = 1.
                else:
                    deljk = 0.
                for l in range(3):
                    if j == l:
                        deljl = 1.
                    else:
                        deljl = 0.
                    if i == l:
                        delil = 1.
                    else:
                        delil = 0.
                    if k == l:
                        delkl = 1.
                    else:
                        delkl = 0.
                        
                    Iten[i,j,k,l] = 0.5 * (delik * deljl + delil * deljk) - 1./3. * delij * delkl
    return Iten


# The Steinberg Guinan strength model
# A simple strength/hardening model
def hardSG(eps, Y0 = 1.2000e-03, beta = 36.0, Ymax = 6.4000e-03, n = 4.5000e-01, eps0 = 0.0):    

    powY = (1.0 + beta * (eps + eps0))
    Y = Y0 * powY**n
    dYdeps = Y * beta * n / powY
    if (Y > Ymax):
        Y = Ymax
        dYdeps = 0.0
    
    return (Y, dYdeps)

############################################################
# Functions related to the Barlat Yld2004-18p Yield Surface
# model. 
############################################################
# A simple function that can return the evaluated Barlat yld2004-18p function
def computeBarlatYieldFunc(sig, L1, L2, a):
    sigv = voigtNotation(sig)
    sig_prime = L1.dot(sigv)
    sig_prime2 = L2.dot(sigv)
    
    spe, spev = np.linalg.eig(matNotation(sig_prime))
    sp2e, sp2ev = np.linalg.eig(matNotation(sig_prime2))
    ia = 1.0 / a
    one_four = 0.25
    yield_fnc_inner = one_four * (np.abs(spe[0] - sp2e[0])**a + np.abs(spe[0] - sp2e[1])**a + np.abs(spe[0] - sp2e[2])**a \
                                + np.abs(spe[1] - sp2e[0])**a + np.abs(spe[1] - sp2e[1])**a + np.abs(spe[1] - sp2e[2])**a \
                                + np.abs(spe[2] - sp2e[0])**a + np.abs(spe[2] - sp2e[1])**a + np.abs(spe[2] - sp2e[2])**a)
    
    yield_fnc = yield_fnc_inner**ia
    
    return yield_fnc 
 
# These are the necessary gradient and hessian of the Barlat model
# It takes in a deviatoric stress tensor and the L',L", and a param defined
# in the literature.
def computeBarlatDerivs(sig, L1, L2, a):
    
    # sig is the deviatoric cauchy stress and is symmetric
    # so sqrt(3/2 sig:sig) = sqrt(3/2 tr(sig * sig))
#    svm = np.sqrt(1.5 * np.trace(sig.dot(sig)))
    svm = effectiveTerm(sig)
    if (svm < np.finfo(np.float64).eps):
        return (np.zeros(6), np.zeros((6,6)), 0.0)
    
    sigv = voigtNotation(sig)
    sig_prime = L1.dot(sigv)
    
    if (effectiveTerm(matNotation(sig_prime)) < np.finfo(np.float64).eps):
        return (np.zeros(6), np.zeros((6,6)), 0.0)
    sig_prime2 = L2.dot(sigv)
    if (effectiveTerm(matNotation(sig_prime2)) < np.finfo(np.float64).eps):
        return (np.zeros(6), np.zeros((6,6)), 0.0)
    spe, spev = np.linalg.eig(matNotation(sig_prime))
    sp2e, sp2ev = np.linalg.eig(matNotation(sig_prime2))
    
    sp2eb = sp2e / svm
    speb = spe / svm
    
    ia = 1.0 / a
    one_four = 0.25
    
    dspe = np.zeros((3,3))
    yield_fnc_inner = 0.0
    for i in range(3):
        for j in range(3):
            dspe[i, j] = np.abs(speb[i] - sp2eb[j])
            yield_fnc_inner += dspe[i, j]**a
    
    yield_fnc_inner *= one_four
    
    
    yield_fnc = svm * yield_fnc_inner**ia
    
    sp2eb = sp2e / yield_fnc
    speb = spe / yield_fnc
    
    for i in range(3):
        for j in range(3):
            dspe[i, j] = np.abs(speb[i] - sp2eb[j])
    
    #d phi / d sigma =
    # \Sum_{i=1}^{3} ( d_phi / d spe_i spev \otimes spev : L1'  +
    #                  d_phi / d spe2_i spev2 \otimes spev2 : L2' )
    
    a2 = a - 2
    
    # We really need these directly. We just need the tensor product
    # with the L1 or L2 tensor
    dspedspev   = np.zeros((3,3,3))
    dspedspev2  = np.zeros((3,3,3))
    # These are really the only values we actually need
    dspedspevL1 = np.zeros((3,6))
    dspedspevL2 = np.zeros((3,6))

    for i in range(3):
        dspedspev[i,:,:]  = np.outer(spev[:, i], spev[:, i])
        dspedspev2[i,:,:] = np.outer(sp2ev[:, i], sp2ev[:, i])
        dspedspevL1[i,:]  = voigtNotation(dspedspev[i,:,:]).dot(L1)
        dspedspevL2[i,:]  = voigtNotation(dspedspev2[i,:,:]).dot(L2)

    dphidspe  = np.zeros(3)
    dphidspe2 = np.zeros(3)

    for i in range(3):
        dphidspe[0]  += dspe[0, i]**a2 * (speb[0] - sp2eb[i])
        dphidspe[1]  += dspe[1, i]**a2 * (speb[1] - sp2eb[i])
        dphidspe[2]  += dspe[2, i]**a2 * (speb[2] - sp2eb[i])
        
        dphidspe2[0] += -dspe[i, 0]**a2 * (speb[i] - sp2eb[0])
        dphidspe2[1] += -dspe[i, 1]**a2 * (speb[i] - sp2eb[1])
        dphidspe2[2] += -dspe[i, 2]**a2 * (speb[i] - sp2eb[2])
        
    dphidspe  *= one_four
    dphidspe2 *= one_four
    
    # First derivative that we'll need 
    # this in a 6 vec and we can transform it back into a 3x3 if need be
    dphidsig = np.zeros(6)
    for i in range(3):
        dphidsig += dphidspe[i] * dspedspevL1[i,:] + dphidspe2[i] * dspedspevL2[i,:]
    # The Hessian is more complicated and looks like:
    # d^2 \phi / (dsig dsig) = 
    # \Sum_{i = 1}^3 
    # \left{ 
    # \Sum_{j = 1}^3 
    # \left [
    # d^2 \phi / (ds'_i ds'_j) (ds'_i/dsv'):L_1 \otimes (ds'_j/dsv'):L_1
    # + d^2 \phi / (ds''_i ds''_j) (ds''_i/dsv''):L_2 \otimes (ds''_j/dsv''):L_2
    # + d^2 \phi / (ds'_i ds''_j) (ds'_i/dsv'):L_1 \otimes (ds''_j/dsv''):L_2
    # + d^2 \phi / (ds''_i ds'_j) (ds''_i/dsv''):L_2 \otimes (ds'_j/dsv'):L_1
    # \right ]
    # + d \phi / (ds'_i) ( L_1^T : (d^2 s'_i) / (dsv' dsv') : L_1)
    # + d \phi / (ds''_i) ( L_2^T : (d^2 s''_i) / (dsv'' dsv'') : L_2)
    # \right}
    
    a1phi = (a - 1) / yield_fnc
    
    dphispe1spe1 = np.zeros((3,3))
    dphispe2spe2 = np.zeros((3,3))
    dphispe1spe2 = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            # dphi^2 / ds_i' ds_j' and dphi^2 / ds_i'' ds_j''
            # form of derivatives
            if i == j:
                dphispe1spe1[i,i] = a1phi * (0.25 * (dspe[i, 0]**a2 + dspe[i, 1]**a2 + dspe[i, 2]**a2) - \
                                    dphidspe[i] * dphidspe[i])
                #   These are very similar to the ones above but note that the indices change
                #   in regards to dspeij we iterate on i var rather than on j up above
                #   so for example dphi^2 / ds_1'' ds_1'' we would set j = 1 and iterate on i.
                dphispe2spe2[i,i] = a1phi * (0.25 * (dspe[0, i]**a2 + dspe[1, i]**a2 + dspe[2, i]**a2) - \
                                    dphidspe2[i] * dphidspe2[i])
#                dphispe1spe2[i,i] = a1phi * (-0.25 * dspe[i,i]**a2 - dphidspe[i] * dphidspe2[i])
            else:
                dphispe1spe1[i,j] = -a1phi * dphidspe[i] * dphidspe[j]
                dphispe2spe2[i,j] = -a1phi * dphidspe2[i] * dphidspe2[j]
#                dphispe1spe2[i,j] = -a1phi * dphidspe[i] * dphidspe2[j]
            # dphi^2 / ds_i' ds_i'' form of derivatives
            #   So, we need to get all of the derivatives for dphi^2/ds_i' ds_j''
            #   it can be seen that we simply need to just iterate through things to get
            #   these values.
            #   If we wanted dphi^2 / ds_i'' ds_j' we would find it to be the transpose of the
            #   above  
            dphispe1spe2[i,j] = a1phi * (-0.25 * dspe[i,j]**a2 - dphidspe[i] * dphidspe2[j])

    E1212, E2323, E3131 = getEijijMatrices(spev)

    if (np.abs(spe[0] - spe[1]) < np.finfo(np.float64).eps):
        term1 = dphispe1spe1[0, 0] - dphispe1spe1[0, 1]
    else:
        term1 = (dphidspe[0] - dphidspe[1]) / (spe[0] - spe[1])
    if (np.abs(spe[1] - spe[2]) < np.finfo(np.float64).eps):
        term2 = dphispe1spe1[1, 1] - dphispe1spe1[1, 2]
    else:
        term2 = (dphidspe[1] - dphidspe[2]) / (spe[1] - spe[2])
    if (np.abs(spe[2] - spe[0]) < np.finfo(np.float64).eps):
        term3 = dphispe1spe1[2, 2] - dphispe1spe1[2, 0]
    else:
        term3 = (dphidspe[2] - dphidspe[0]) / (spe[2] - spe[0]) 
    dphispe1spev1 = 0.5 * ( term1 * E1212 + \
                            term2 * E2323 + \
                            term3 * E3131)
    
    E1212, E2323, E3131 = getEijijMatrices(sp2ev)
    
    
    if (np.abs(sp2e[0] - sp2e[1]) < np.finfo(np.float64).eps):
        term1 = dphispe2spe2[0, 0] - dphispe2spe2[0, 1]
    else:
        term1 = (dphidspe2[0] - dphidspe2[1]) / (sp2e[0] - sp2e[1])
    if (np.abs(sp2e[1] - sp2e[2]) < np.finfo(np.float64).eps):
        term2 = dphispe2spe2[1, 1] - dphispe2spe2[1, 2]
    else:
        term2 = (dphidspe2[1] - dphidspe2[2]) / (sp2e[1] - sp2e[2])
    if (np.abs(sp2e[2] - sp2e[0]) < np.finfo(np.float64).eps):
        term3 = dphispe2spe2[2, 2] - dphispe2spe2[2, 0]
    else:
        term3 = (dphidspe2[2] - dphidspe2[0]) / (sp2e[2] - sp2e[0])
    dphispe2spev2 = 0.5 * (term1 * E1212 + \
                           term2 * E2323 + \
                           term3 * E3131)
    
    evL1evL1 = np.zeros((3,3,6,6))
    evL2evL2 = np.zeros((3,3,6,6))
    evL1evL2 = np.zeros((3,3,6,6))
    evL2evL1 = np.zeros((3,3,6,6))
    
    dphidsigdsig = np.zeros((6,6))

    for i in range(3):
        for j in range(3):
            # We actually don't need to keep these around in the final product of things
            evL1evL1[i,j,:,:] = np.outer(dspedspevL1[i,:], dspedspevL1[j,:])
            evL2evL2[i,j,:,:] = np.outer(dspedspevL2[i,:], dspedspevL2[j,:])
            evL1evL2[i,j,:,:] = np.outer(dspedspevL1[i,:], dspedspevL2[j,:])
            evL2evL1[i,j,:,:] = np.outer(dspedspevL2[i,:], dspedspevL1[j,:])
            
            # We can form dphidsigdsig down here:
            dphidsigdsig +=  dphispe1spe1[i,j] * evL1evL1[i,j,:,:] + \
                             dphispe2spe2[i,j] * evL2evL2[i,j,:,:] + \
                             dphispe1spe2[i,j] * evL1evL2[i,j,:,:] + \
                             dphispe1spe2[j,i] * evL2evL1[i,j,:,:]   
################
    
    dphidsigdsig += L2.T.dot(dphispe2spev2.dot(L2)) + \
                    L1.T.dot(dphispe1spev1.dot(L1))
    
    
    return (dphidsig, dphidsigdsig, yield_fnc)

def getEijijMatrices(spev):
    E1212 = outer4tensor(spev[:, 0], spev[:, 1], spev[:, 0], spev[:, 1]) + \
            outer4tensor(spev[:, 1], spev[:, 0], spev[:, 0], spev[:, 1]) + \
            outer4tensor(spev[:, 0], spev[:, 1], spev[:, 1], spev[:, 0]) + \
            outer4tensor(spev[:, 1], spev[:, 0], spev[:, 1], spev[:, 0])
  
    E1212 = full_4d_to_Voigt_2d(E1212)
 
    E2323 = outer4tensor(spev[:, 1], spev[:, 2], spev[:, 1], spev[:, 2]) + \
            outer4tensor(spev[:, 2], spev[:, 1], spev[:, 1], spev[:, 2]) + \
            outer4tensor(spev[:, 1], spev[:, 2], spev[:, 2], spev[:, 1]) + \
            outer4tensor(spev[:, 2], spev[:, 1], spev[:, 2], spev[:, 1])    

    E2323 = full_4d_to_Voigt_2d(E2323)

    E3131 = outer4tensor(spev[:, 2], spev[:, 0], spev[:, 2], spev[:, 0]) + \
            outer4tensor(spev[:, 0], spev[:, 2], spev[:, 2], spev[:, 0]) + \
            outer4tensor(spev[:, 2], spev[:, 0], spev[:, 0], spev[:, 2]) + \
            outer4tensor(spev[:, 0], spev[:, 2], spev[:, 0], spev[:, 2])

    E3131 = full_4d_to_Voigt_2d(E3131)

    return (E1212, E2323, E3131)    

def computeRJ_Barlat(x, sig_tr, eps0, muNew, bulkMod, dt, L1, L2, aparam):
    J = np.zeros((7,7))
    r = np.zeros(7)
    '''
    // x = [stress, d\lambda]
    // where dev_stress is the 6x1 form of the deviatoric stress
    // where d\lambda is used in deps = d\lambda * df / dsig
    // f = phi(sig) - sig_y(eps)
    // so df / dsig = dphi/dsig for associative flow rules which is what
    // we're doing here
    '''
    dim = 3
    dim6 = dim * 2
    nDim = 7    
    # The Jacobian is explicitly defined in this paper:
    # https://doi.org/10.1016/j.cma.2016.11.026 and https://doi.org/10.1002/nme.5515

    scaledX = np.zeros(nDim)

    ysFunc = 0.0
    scaledX = np.copy(x)
    sig_dev = scaledX[0:6]

    gradYSFunc, hessYSFunc, ysFunc = computeBarlatDerivs(matNotation(scaledX[0:6]), L1, L2, aparam)
    # We now need to compute (L_ijkl)^-1 = (C_ijkl^-1 + d\lambda hessYSFunc)
    # Only do these calculations if d\lambda > ~0
    # If not then L_ijkl^-1 = C_ijkl^-1
    if np.abs(x[dim6]) > np.finfo(np.float64).eps:
        hessYSFunc *= scaledX[dim6]
    else:
        hessYSFunc[:,:] = 0.0
    
    # We can't go down to a reduced deviatoric space representation of things
    # We have to use the full compliance matrix
    # aka 1/E * [1 -nu -nu 0 0 0;
    #            -nu 1 -nu 0 0 0;
    #            -nu -nu 1 0 0 0;
    #             0 0 0 1+nu 0 0;
    #             0 0 0 0 1+nu 0;
    #             0 0 0 0 0 1+nu]
    # where we can substitue 1+nu/E for 1/mu and 1/E for (3K + mu)/(9Kmu) 
    # and -nu for -(3K - 2mu)/(6K+2mu) so -nu/E = -(3K - 2mu)/(18Kmu)
    invE = (3.0 * bulkMod + muNew) / (9.0 * bulkMod * muNew)
    mnuIE = -(3.0 * bulkMod - 2.0 * muNew) / (18.0 * bulkMod * muNew)
    iMu = 1.0 / muNew
    
    hessYSFunc[0, 0] += invE
    hessYSFunc[1, 1] += invE
    hessYSFunc[2, 2] += invE
    hessYSFunc[0, 1] += mnuIE
    hessYSFunc[0, 2] += mnuIE
    hessYSFunc[1, 2] += mnuIE
    # This matrix should be symmetric so...
    hessYSFunc[1, 0] = hessYSFunc[0, 1]
    hessYSFunc[2, 0] = hessYSFunc[0, 2]
    hessYSFunc[2, 1] = hessYSFunc[1, 2]

    hessYSFunc[3, 3] += 0.5 * iMu
    hessYSFunc[4, 4] += 0.5 * iMu
    hessYSFunc[5, 5] += 0.5 * iMu
    
    # 
    #   Terms for Jacobian with plastic straining
    #
    eps = eps0 + scaledX[dim6]  
#    sigbr, dsigdeps = hardSG(eps, Y0 = 290.0, beta = 125, Ymax = 680.0, n = 0.10)
    sigbr, dsigdeps = hardSG(eps, Y0 = 1.2000e-03, beta = 36.0, Ymax = 6.4000e-03, n = 4.5000e-01, eps0 = 0.0)

    # Set-up our residual
    r[0] = invE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[1] - sig_tr[1]) + mnuIE * (sig_dev[2] - sig_tr[2]) \
         + scaledX[dim6] * gradYSFunc[0]
    r[1] = invE * (sig_dev[1] - sig_tr[1]) + mnuIE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[2] - sig_tr[2]) \
         + scaledX[dim6] * gradYSFunc[1]
    r[2] = invE * (sig_dev[2] - sig_tr[2]) + mnuIE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[1] - sig_tr[1]) \
         + scaledX[dim6] * gradYSFunc[2]
    r[3] = 0.5 * iMu * (sig_dev[3] - sig_tr[3]) + scaledX[dim6] * gradYSFunc[3]
    r[4] = 0.5 * iMu * (sig_dev[4] - sig_tr[4]) + scaledX[dim6] * gradYSFunc[4]
    r[5] = 0.5 * iMu * (sig_dev[5] - sig_tr[5]) + scaledX[dim6] * gradYSFunc[5]

    r[6] = ysFunc - sigbr

    # We could scale our residual if we wanted to by a residual_scale factor
    # We would also require an x_scale factor to go with this though
    r[:] *= 1.0
    # We can now form our Jacobian
    # If we were using this in a finite element software
    # after we've reached a solved state we could calculate
    # our alg. consistent tangent modulus here right before
    # forming the Jacobian
    J[0:dim6, 0:dim6] = hessYSFunc
    J[dim6, 0:dim6]   = gradYSFunc
    J[0:dim6, dim6]   = gradYSFunc
    J[dim6, dim6]     = -dsigdeps

    return (r, J)

################################################
# Functions related to a typical J2 formulation
# Useful as a comparison to the Barlat model
################################################
def J2HessGradYS(sig):
    vM = np.sqrt(1.5 * np.trace(sig.dot(sig)))
    ###
    # Comparing the Barlat Yld2004-18P gradient versus the known
    # J2 gradient which we know for this simple case
    
    gradYS  = 1.5/vM * voigtNotation(sig) 
    ###
    # Comparing the Barlat Yld2004-18P hessian versus the known
    # J2 hessian which we know for this simple case
    n = matNotation(gradYS) / np.linalg.norm(matNotation(gradYS))
    
    Iten = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            if i == j:
                delij = 1.
            else:
                delij = 0.
            for k in range(3):
                if i == k:
                    delik = 1.
                else:
                    delik = 0.
                if j == k:
                    deljk = 1.
                else:
                    deljk = 0.
                for l in range(3):
                    if j == l:
                        deljl = 1.
                    else:
                        deljl = 0.
                    if i == l:
                        delil = 1.
                    else:
                        delil = 0.
                    if k == l:
                        delkl = 1.
                    else:
                        delkl = 0.
                        
                    Iten[i,j,k,l] = 0.5 * (delik * deljl + delil * deljk) - 1./3. * delij * delkl

    nxn = np.einsum('ij,kl->ijkl',n,n)
    
    Ibar = Iten - nxn
    
    rsig4 = 1.5 / vM * Ibar
    #J2 Hessian term
    hessYS =  full_4d_to_Voigt_2d(rsig4)
    
    return (gradYS, hessYS, vM)

def computeRJ_J2(x, sig_tr, eps0, muNew, bulkMod, dt):
    J = np.zeros((7,7))
    r = np.zeros(7)

    dim = 3
    dim6 = dim * 2
    nDim = 7
    
    # The Jacobian is explicitly defined in this paper:
    # https://doi.org/10.1016/j.cma.2016.11.026 and https://doi.org/10.1002/nme.5515
    scaledX = np.zeros(nDim)
    ysFunc = 0.0
    scaledX = np.copy(x)
    sig_dev = scaledX[0:6]

    gradYSFunc, hessYSFunc, ysFunc = J2HessGradYS(matNotation(scaledX[0:6]))
    # We now need to compute (L_ijkl)^-1 = (C_ijkl^-1 + d\lambda hessYSFunc)
    # Only do these calculations if d\lambda > ~0
    # If not then L_ijkl^-1 = C_ijkl^-1
    if np.abs(x[dim6]) > np.finfo(np.float64).eps:
        hessYSFunc *= scaledX[dim6]
    else:
        hessYSFunc[:,:] = 0.0
    
    # We can't go down to a reduced deviatoric space representation of things
    # We have to use the full compliance matrix
    # aka 1/E * [1 -nu -nu 0 0 0;
    #            -nu 1 -nu 0 0 0;
    #            -nu -nu 1 0 0 0;
    #             0 0 0 1+nu 0 0;
    #             0 0 0 0 1+nu 0;
    #             0 0 0 0 0 1+nu]
    # where we can substitue 1+nu/E for 1/mu and 1/E for (3K + mu)/(9Kmu) 
    # and -nu for -(3K - 2mu)/(6K+2mu) so -nu/E = -(3K - 2mu)/(18Kmu)
    invE = (3.0 * bulkMod + muNew) / (9.0 * bulkMod * muNew)
    mnuIE = -(3.0 * bulkMod - 2.0 * muNew) / (18.0 * bulkMod * muNew)
    iMu = 1.0 / muNew

    hessYSFunc[0, 0] += invE
    hessYSFunc[1, 1] += invE
    hessYSFunc[2, 2] += invE
    hessYSFunc[0, 1] += mnuIE
    hessYSFunc[0, 2] += mnuIE
    hessYSFunc[1, 2] += mnuIE
    # This matrix should be symmetric so...
    hessYSFunc[1, 0] = hessYSFunc[0, 1]
    hessYSFunc[2, 0] = hessYSFunc[0, 2]
    hessYSFunc[2, 1] = hessYSFunc[1, 2]
    # Not clear to me if we need the 1/2 term here as well for 1/mu
    hessYSFunc[3, 3] += 0.5 * iMu
    hessYSFunc[4, 4] += 0.5 * iMu
    hessYSFunc[5, 5] += 0.5 * iMu

    # Our hessYSFunc is now our (L_ijkl)^-1 term 

    # 
    #   Terms for Jacobian with plastic straining
    #
    eps = eps0 + scaledX[dim6]
#    sigbr, dsigdeps = hardSG(eps, Y0 = 290.0, beta = 125, Ymax = 680.0, n = 0.10)
    sigbr, dsigdeps = hardSG(eps, Y0 = 1.2000e-03, beta = 36.0, Ymax = 6.4000e-03, n = 4.5000e-01, eps0 = 0.0)

    r[0] = invE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[1] - sig_tr[1]) + mnuIE * (sig_dev[2] - sig_tr[2]) \
         + scaledX[dim6] * gradYSFunc[0]
    r[1] = invE * (sig_dev[1] - sig_tr[1]) + mnuIE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[2] - sig_tr[2]) \
         + scaledX[dim6] * gradYSFunc[1]
    r[2] = invE * (sig_dev[2] - sig_tr[2]) + mnuIE * (sig_dev[0] - sig_tr[0]) + mnuIE * (sig_dev[1] - sig_tr[1]) \
         + scaledX[dim6] * gradYSFunc[2]
    r[3] = 0.5 * iMu * (sig_dev[3] - sig_tr[3]) + scaledX[dim6] * gradYSFunc[3]
    r[4] = 0.5 * iMu * (sig_dev[4] - sig_tr[4]) + scaledX[dim6] * gradYSFunc[4]
    r[5] = 0.5 * iMu * (sig_dev[5] - sig_tr[5]) + scaledX[dim6] * gradYSFunc[5]
    r[6] = ysFunc - sigbr

    # We could scale our residual if we wanted to by a residual_scale factor
    # We would also require an x_scale factor to go with this though
    r[:] *= 1.0
    # We can now form our Jacobian
    # If we were using this in a finite element software
    # after we've reached a solved state we could calculate
    # our alg. consistent tangent modulus here right before
    # forming the Jacobian
    J[0:dim6, 0:dim6] = hessYSFunc
    J[dim6, 0:dim6]   = gradYSFunc
    J[0:dim6, dim6]   = gradYSFunc
    J[dim6, dim6]     = -dsigdeps

    return (r, J)

######################################################################
# Functions used to drive an optimization script for our
# parameters in the Barlat model
######################################################################
def barlat_optimize(x0, stressVoigt, vMs, rzxs, aparam, r_include, wsig=10.0, wr=1.0):
    error = 0.0

    L1 = np.zeros((6,6))
    L2 = np.zeros((6,6))
    L1[0,1] = -x0[0]
    L1[0,2] = -x0[1]
    L1[1,0] = -x0[2]
    L1[1,2] = -x0[3]
    L1[2,0] = -x0[4]
    L1[2,1] = -x0[5]
    L1[3,3] = x0[6]
    L1[4,4] = x0[7]
    L1[5,5] = x0[8]
    
    L2[0,1] = -x0[9]
    L2[0,2] = -x0[10]
    L2[1,0] = -x0[11]
    L2[1,2] = -x0[12]
    L2[2,0] = -x0[13]
    L2[2,1] = -x0[14]
    L2[3,3] = x0[15]
    L2[4,4] = x0[16]
    L2[5,5] = x0[17]
    # We could try optimizing for the a parameter exponent but that
    # could make the optimization even harder
    aparam = x0[18]

    C1 = np.copy(L1)
    C2 = np.copy(L2)

    Iten = ItenMat()
    Iten_2d = full_4d_to_Voigt_2d(Iten)
    Iten_2d[3:6, 3:6] *= 2.0
    L1 = C1.dot(Iten_2d)
    L2 = C2.dot(Iten_2d)

    for i in range(stressVoigt.shape[0]):
        stress = matNotation(stressVoigt[i, :])
        yld_derv, yld_hsn, yld_fnc = computeBarlatDerivs(stress, L1, L2, aparam)
        # If our yield function is less than or near zero we really
        # want to penalize this test case
        if (yld_fnc < np.finfo(np.float64).eps):
            error += 10000.0
            continue
        eigVals, eigVecs = np.linalg.eig(stress)
        yld_derv = matNotation(yld_derv)
        # We need to rotate yld_derv into the correct frame
        # of reference
        e_rs = 0.0
        if r_include[i]:
            yld_derv = eigVecs.T.dot(yld_derv.dot(eigVecs))
            rzx_sig = yld_derv[2, 2] / yld_derv[1, 1]
            e_rs  = np.sqrt(wr * (rzx_sig/rzxs[i] - 1.0)**2)
        e_sig = np.sqrt(wsig * (yld_fnc/vMs[i] - 1.0)**2)
        error +=  e_sig + e_rs

    return error   

def postprocessing(frve_name, fdiro, ftime, tempk, shear_loads):
    #%%
    # Start parsing data
    # First we probably should have some way to share the loading direction names
    # between the job_cli.py file and this one but until then we'll just have the
    # list of names copied in the two.

    # Test names are loading_axis_dir1_cosine_ang1_loading_dir2_cosine_ang2
    # where the cosine angle is in degrees and is the amount of the monotonic
    # velocity applied in that direction.
    # Cases marked shear_$$ are pure shear cases.
    # The x_90_z_0 case will always exist and is our go-to standard monotonic
    # loading case that we'll base all the other models on.

    loading_dir_names = [
            "x_90_z_0", "x_0_y_90", "x_15_y_75", "x_30_y_60", "x_45_y_45", 
            "x_60_y_30", "x_75_y_15", "x_90_y_0", "x_15_z_75", "x_30_z_60",  
            "x_45_z_45", "x_60_z_30", "x_75_z_15", "y_15_z_75", "y_30_z_60",
            "y_45_z_45", "y_60_z_30", "y_75_z_15"]

    loading_dir_shear_names = ["shear_xy", "shear_xz", "shear_yz"]

    if(shear_loads):
        loading_dir_names.extend(loading_dir_shear_names)

    ntests = len(loading_dir_names)

    r_include = [False] * len(loading_dir_names)

    r_include[0] = True
    r_include[1] = True
    r_include[7] = True

    # The full name of each simulation is the 
    # base name + rve id name + tempk + loading_dir_name

    stress_bname    = "avg_stress_"
    plwork_bname    = "avg_pl_work_"
    dp_tensor_bname = "avg_dp_tensor_"

    # All simulations will have the same number of time steps taken and applied strain
    # rate for this initial work of 1e-3 1/s and time steps are found in custom_dt_fine.txt
    #

    strain_rate = 0.001

    properties = np.zeros((len(tempk), 19))

    itemp = 0
    for temp in tempk:
        print("Starting temperature: ", temp)
        temp_dir = loading_dir_names[0]+"_"+str(int(temp))
        fdir_time = os.path.join(fdir_rve, temp_dir, "")

        if isinstance(ftime, list):
            time = np.loadtxt(fdir_time+ftime[itemp])
        else:
            time = np.loadtxt(fdir_time+ftime)
        # Change tempk to be looped over
        ext_name = frve_name+"_"+str(int(temp))+"_"+loading_dir_names[0]+'.txt'
        stress = np.loadtxt(fdir_time+stress_bname+ext_name)
        pl_work = np.loadtxt(fdir_time+plwork_bname+ext_name)
        nsteps = time.shape[0]
        
        yld_rot = np.zeros((3,3,ntests))

        eps = np.zeros(nsteps)
        for i in range(0, nsteps):
            dtime = time[i]
            eps[i] = eps[i - 1] + strain_rate * dtime
        
        # This part is a bit manual at this point
        # We might need a finer dt set for the bi-axial load set then what we currently
        # use for the monotonic loading cases
        slope, intercept, r, p, se = scist.linregress(eps[0:25], stress[0:25, 2])
        stress_offset = slope * (eps - 0.001)
        
        plwork_driver = 0.0
        stress_exp = np.zeros((ntests, 6))
        dptens_exp = np.zeros((ntests, 9))
        vMs = np.zeros(ntests)
        rzx = np.zeros(ntests)
        
        for j in range(2, nsteps):
            if stress_offset[j] > stress[j, 2]:
                # J would be our point of yield
                sx1 = eps[j - 1]
                sy1 = stress[j - 1, 2]
                sx2 = eps[j]
                sy2 = stress[j, 2]
                oy1 = stress_offset[j - 1]
                oy2 = stress_offset[j]
                plwork_driver = pl_work[j]
                break
            
        YS = ((sx1 * oy2 - sx2 * oy1) * (sy1 - sy2) - (oy1 - oy2) * (sx1 * sy2 - sx2 * sy1)) / ((sx1 - sx2) * (sy1 - sy2) - (sx1 - sx2) * (oy1 - oy2))
        print([YS, plwork_driver])
        
        iload = 0
        for load_dir in loading_dir_names:
            temp_dir = load_dir+"_"+str(int(temp))
            fdir_time = os.path.join(fdir_rve, temp_dir, "")
        
            time = np.loadtxt(fdir_time+ftime)
            # Change tempk to be looped over
            ext_name = frve_name+"_"+str(int(temp))+"_"+load_dir+'.txt'
            stress = np.loadtxt(fdir_time+stress_bname+ext_name)
            dptens = np.loadtxt(fdir_time+dp_tensor_bname+ext_name)
            plwork = np.loadtxt(fdir_time+plwork_bname+ext_name)
            
            # We want to find the point closest to the original plwork_driver
            # We might need to revisit this assumption and find the minimum positive
            # difference 
            abs_dplwork = np.abs(plwork - plwork_driver)
            ind = np.argmin(abs_dplwork)
            stress_exp[iload, :] = stress[ind, :]
            dptens_exp[iload, :] = dptens[ind, :]
            
            sig = matNotation(stress_exp[iload, :])
            eigVals, eigVecs = np.linalg.eig(sig)
            vMs[iload] = effectiveTerm(sig)

            # Just some useful information logged about % relative differences in
            # our plastic work, the vonMises for that step, and which
            # load step this minimum difference in plastic work
            print(ind, 100.0 * abs_dplwork[ind]/plwork_driver, vMs[iload])
            yld_derv = np.reshape(dptens[ind, :], newshape=(3,3), order="F")
            # We now need to rotate Dp into the correct reference frame
            # for all of our calculations
            yld_rot[:,:,iload] = eigVecs.T.dot(yld_derv.dot(eigVecs))
            rzx[iload] = yld_rot[2,2,iload] / yld_rot[1,1,iload]
            
            iload += 1
        
        # Just a way to see what the mean von Mises value is
        print("Mean von Mises: ", np.mean(vMs))
        print("About to start optimizations")
        
        # Start off with the assumption that it's a J2 surface
        # We're also not going to have it try to optimize for the a param
        # in the Barlat model
        # It might eventually be something that we optimize for but not for
        # now 
        x0 = np.zeros(19)
        # For our various temperatures our solution is accelerated if we provide it
        # a better initial guess by providing the last converged solution
        if (itemp == 0):
            x0[0:6] = 1.0
            x0[6:9] = 1.0
            x0[9:15] = 1.0
            x0[15:18] = 1.0
            aparam = 8.0
            x0[-1] = aparam
        else:
            x0 = np.copy(res.x)
        
        bnds = []
        for i in range(18):
            bnds.append((0.7, 1.5))
        bnds.append((2.0,16.0))
        bnds = tuple(bnds)
        bounds=bnds
        
        # The L-BFGS-B does the best job out of all the python minimization/optimization methods
        # when provided only the function evaluation. I would imagine if we had someone more familiar with
        # optimization methods we could do even better. However, I don't do a lot of optimization work...
        # It also looks like we might want to be in the rough ballpark of the answer...
        # Although, the Nelder-Mead is having decent success on "real" data...
        # Print out the initial error value
        print("Initial error:")
        print(barlat_optimize(x0, stress_exp, vMs, rzx, aparam, r_include, 10, 1))
        res = minimize(barlat_optimize, x0, args=(stress_exp, vMs, rzx, aparam, r_include, 10, 1), method="Nelder-Mead", bounds=bounds, options={"maxiter":60000, "disp":True, "adaptive":True, "fatol":1e-10, "xatol":1e-10}, tol=1e-10)
        # res = minimize(barlat_optimize, x0, args=(stress_exp, vMs, rzx, aparam, r_include, 10, 1), bounds=bnds, options={"maxcor":10, "maxls":50, "disp":True, "ftol":1e-14}, tol=1e-9)
        print("Solution:")
        print(res.x)
        # x0 = np.copy(res.x)
        # res = minimize(barlat_optimize, x0, args=(stress_exp, vMs, rzx, aparam), method="Powell", bounds=bnds, options={"maxiter":19000, "disp":True})
        # prints the error
        properties[itemp, :] = res.x
        print("Final error:")
        print(barlat_optimize(properties[itemp, :], stress_exp, vMs, rzx, aparam, r_include, 10, 1))

        itemp += 1

    # Creates the directory if it already doesn't exist
    if not os.path.exists(fdiro):
        os.makedirs(fdiro)

    fdiron = os.path.join(fdiro, frve_name, "") 
    np.savetxt(fdiron+frve_name+"_properties", properties)

###############################################################################
# CLI tool that takes in a directory where RVE simulations live
# tool then traverses the necessary directories and pulls out the necessary
# files that we need to calculate the YS parameters.
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workflow flux and batch cli')

    parser.add_argument(
        '-sdir',
        '--simulation_file_dir',
        type=str,
        default='./../wf_runs/',
        help='Directory of the RVE simulation data (default: ./)'
    )

    parser.add_argument(
        '-odir',
        '--output_directory',
        type=str,
        default='./../wf_runs',
        help='Directory to output parameter data (default: ./../wf_runs)'
    )


    parser.add_argument(
        '-rve_id',
        '--rve_identifier_name',
        type=str,
        default='simulation',
        help='Unique identifier name for the RVE (default: rve_in625)'
    )

    parser.add_argument(
            '-t',
            '--temperature',
            nargs='+',
            default=[298.0],
            help='List of temperatures in Kelvin to read in (default: 298.0)'
    )

    parser.add_argument(
            '-shr',
            '--shear_loads',
            action="store_true",
            help='Use shear loading cases in optimization (default: False)'
    )

    args = parser.parse_args()

    #%%
    # Simulation directory
    fdirs = args.simulation_file_dir
    if (fdirs == "./"):
        fdirs = os.getcwd()
        os.path.join(fdirs, '')
    #%%
    # Input rve identifier name
    frve_name = args.rve_identifier_name
    fdir_rve  = os.path.join(fdirs, frve_name, "")
    #%%
    # Output directory
    fdiro = os.path.abspath(args.output_directory)

    #%%
    # Temperatue ranges that simulations were run at
    tempk = args.temperature

    ftime = "custom_dt.txt"

    postprocessing(frve_name, fdiro, ftime, tempk, args.shear_loads)

###############################################################################
# An example of how to solve for the updated stress and hardening state
# We can either use a simple NR method as given in the for loop
# or we can revert to something more complicated like scipy's root function
# If we're far away from the solution then a simple NR may not be good enough
# and you might need something with a trust-region in it.
###############################################################################
'''
L1 = np.zeros((6,6))
L2 = np.zeros((6,6))
L1[0,1] = x0[0]
L1[0,2] = x0[1]
L1[1,0] = x0[2]
L1[1,2] = x0[3]
L1[2,0] = x0[4]
L1[2,1] = x0[5]
L1[3,3] = x0[6]
L1[4,4] = x0[7]
L1[5,5] = x0[8]

L2[0,1] = x0[9]
L2[0,2] = x0[10]
L2[1,0] = x0[11]
L2[1,2] = x0[12]
L2[2,0] = x0[13]
L2[2,1] = x0[14]
L2[3,3] = x0[15]
L2[4,4] = x0[16]
L2[5,5] = x0[17]
# We could try optimizing for the a parameter exponent but that
# could make the optimization even harder
#    aparam = x0[18]

C1 = np.copy(L1)
C2 = np.copy(L2)

Iten = ItenMat()
Iten_2d = full_4d_to_Voigt_2d(Iten)
Iten_2d[3:6, 3:6] *= 2.0
L1 = C1.dot(Iten_2d)
L2 = C2.dot(Iten_2d)

aparam = 4.0

sig_trial = np.asarray([6.87152400165220e-03, -9.62013360231308e-03, 2.74860960066088e-03, 8.24582880198264e-04, -8.24582880198264e-04, 0.00000000000000e+00])
phi = computeBarlatYieldFunc(matNotation(sig_trial), L1, L2, aparam)
sigbr, dsigdeps = hardSG(0.0, Y0 = 1.2000e-03, beta = 36.0, Ymax = 6.4000e-03, n = 4.5000e-01, eps0 = 0.0)

if (sigbr < phi):
    x = np.zeros(7)
    x[0:6] = sig_trial
    for i in range(100):
        # Either the J2 or Barlat could be used here and we should receive the same solution
        # Although, I've noted that the J2 method doesn't always perform the best
        # in the simple NR method. So, you might need to try it with the root
        # solver
#        resid, jacob = computeRJ_J2(x, sig_trial, 0.0, 4.12291440099132e-01, 5.40201493383883e+00, 0.1)
        resid2, jacob2 = computeRJ_Barlat(x, sig_trial, 0.0, 4.12291440099132e-01, 5.40201493383883e+00, 0.1, L1, L2, aparam)
        err = np.linalg.norm(resid2)
        if ( err < 1e-8):
            break
        else:
            sol = np.linalg.solve(jacob2, resid2)
            x -= sol
    
    
    x_alt = np.zeros(7)
    x_alt[0:6] = sig_trial
#    args_Barlat = (sig_trial, 0.0, 4.12291440099132e-01, 5.40201493383883e+00, 0.1, L1, L2, aparam)
#    res = root(computeRJ_Barlat, x_alt, args=args_J2, jac=True, method='hybr', tol=1e-12)
    args_J2 = (sig_trial, 0.0, 4.12291440099132e-01, 5.40201493383883e+00, 0.1)
    res = root(computeRJ_J2, x_alt, args=args_J2, jac=True, method='hybr', tol=1e-12)
    sol = np.zeros(6)
    sol[0:6] = res.x[0:6]
    print(sol)
    print(x[0:6])
'''