#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:03:46 2020

@author: robert
"""

import numpy as np
from sklearn.preprocessing import normalize

def QuatProd(q2, q1):
    '''
    QuatProd - Product of two unit quaternions.

      USAGE:

       qp = QuatProd(q2, q1)

      INPUT:

       q2, q1 are 4 x n, 
              arrays whose columns are quaternion parameters

      OUTPUT:

       qp is 4 x n, 
          the array whose columns are the quaternion parameters of 
          the product; the first component of qp is nonnegative

       NOTES:

       *  If R(q) is the rotation corresponding to the
          quaternion parameters q, then 

          R(qp) = R(q2) R(q1)


    '''

    a = np.atleast_2d(q2[0, :])
    a3 = np.tile(a, (3, 1))
    b = np.atleast_2d(q1[0, :])
    b3 = np.tile(b, (3, 1))

    avec = np.atleast_2d(q2[1:4, :])
    bvec = np.atleast_2d(q1[1:4, :])

    qp1 = np.atleast_2d(a*b - np.sum(avec.conj() * bvec, axis=0))
    if q1.shape[1] == 1:
        qp2 = np.atleast_2d(np.squeeze(a3*bvec + b3*avec + np.cross(avec.T, bvec.T).T)).T
    else:
        qp2 = np.atleast_2d(np.squeeze(a3*bvec + b3*avec + np.cross(avec.T, bvec.T).T))

    qp = np.concatenate((qp1, qp2), axis=0)

    q1neg = np.nonzero(qp[0, :] < 0)

    qp[:, q1neg] = -1*qp[:, q1neg]

    return qp

def QuatMean(quats):
    '''
    QuatMean finds the average quaternion based upon the methodology defined in
    Quaternion Averaging by Markley, Cheng, Crassidis, and Oshman
    
    Input:
        quats - A list of quaternions of that we want to find the average quaternion
    Output:
        mquats - the mean quaternion of the system
    '''
    if(quats.shape[0] == 4):
        n = quats.shape[1]
        mmat = 1 / n * quats.dot(quats.T)
    else:
        n = quats.shape[0]
        mmat = 1 / n * quats.T.dot(quats)
    bmmat = mmat - np.eye(4)
    
    eig, eigvec = np.linalg.eig(bmmat)
    mquats = np.squeeze(eigvec[:, np.argmax(eig)])
    
    return mquats

def QuatOfAngleAxis(angle, raxis):
    '''
    QuatOfAngleAxis - Quaternion of angle/axis pair.

      USAGE:

      quat = QuatOfAngleAxis(angle, axis)

      INPUT:

      angle is an n-vector, 
            the list of rotation angles
      axis is 3 x n, 
            the list of rotation axes, which need not
            be normalized (e.g. [1 1 1]'), but must be nonzero

      OUTPUT:

      quat is 4 x n, 
           the quaternion representations of the given
           rotations.  The first component of quat is nonnegative.
   '''

    tol = 1.0e-8

    #Errors can occur when this is near pi or negative pi
    limit = np.abs(np.abs(angle) - np.pi) < tol
    
    angle[limit] = np.pi * np.sign(angle[limit])
    
    halfAngle = 0.5 * angle.T
    cphiby2 = np.atleast_2d(np.cos(halfAngle))
    sphiby2 = np.sin(halfAngle)
    scaledAxis = normalize(raxis, axis=0)*np.tile(sphiby2, (3,1))
#    rescale = sphiby2/np.sqrt(np.sum(raxis.conj()*raxis, axis=0))
#    scaledAxis = np.tile(rescale, (3, 1))*raxis
    quat = np.concatenate((cphiby2, scaledAxis), axis=0)
    q1neg = np.nonzero(quat[0, :] < 0)
    quat[:, q1neg] = -1*quat[:, q1neg]

    return quat

def ToFundamentalRegionQ(quat, qsym):
    '''
    ToFundamentalRegionQ - To quaternion fundamental region.

      USAGE:

      q = ToFundamentalRegionQ(quat, qsym)

      INPUT:

      quat is 4 x n, 
           an array of n quaternions
      qsym is 4 x m, 
           an array of m quaternions representing the symmetry group

      OUTPUT:

      q is 4 x n, the array of quaternions lying in the
                  fundamental region for the symmetry group 
                  in question

      NOTES:  

      *  This routine is very memory intensive since it 
         applies all symmetries to each input quaternion.


    '''
    quat = mat2d_row_order(quat)    
    qsym = mat2d_row_order(qsym)
    n = quat.shape[1]
    m = qsym.shape[1]

    qMat = np.tile(quat, (m, 1))

    qSymMat = np.tile(qsym, (1, n))

    qeqv = QuatProd(qMat.T.reshape(m*n, 4).T, qSymMat)

    q0_abs = np.abs(np.atleast_2d(qeqv[0, :]).T.reshape(n, m)).T

    imax = np.argmax(q0_abs, axis=0)

    ind = np.arange(n)*m + imax

    q = qeqv[:, ind]

    return q

def CubSymmetries():
    ''' CubSymmetries - Return quaternions for cubic symmetry group.

       USAGE:

       csym = CubSymmetries

       INPUT:  none

       OUTPUT:

       csym is 4 x 24, 
            quaternions for the cubic symmetry group
    '''

    '''
        array index 1 = identity
        array index 2-4 = fourfold about x1
        array index 5-7 = fourfold about x2
        array index 8-9 = fourfold about x9
        array index 10-11 = threefold about 111
        array index 12-13 = threefold about 111
        array index 14-15 = threefold about 111
        array index 16-17 = threefold about 111
        array index 18-24 = twofold about 110
    
    '''
    angleAxis = [
        [0.0, 1, 1, 1],
        [np.pi*0.5, 1, 0, 0],
        [np.pi, 1, 0, 0],
        [np.pi*1.5, 1, 0, 0],
        [np.pi*0.5, 0, 1, 0],
        [np.pi, 0, 1, 0],
        [np.pi*1.5, 0, 1, 0],
        [np.pi*0.5, 0, 0, 1],
        [np.pi, 0, 0, 1],
        [np.pi*1.5, 0, 0, 1],
        [np.pi*2/3, 1, 1, 1],
        [np.pi*4/3, 1, 1, 1],
        [np.pi*2/3, -1, 1, 1],
        [np.pi*4/3, -1, 1, 1],
        [np.pi*2/3, 1, -1, 1],
        [np.pi*4/3, 1, -1, 1],
        [np.pi*2/3, -1, -1, 1],
        [np.pi*4/3, -1, -1, 1],
        [np.pi, 1, 1, 0],
        [np.pi, -1, 1, 0],
        [np.pi, 1, 0, 1],
        [np.pi, 1, 0, -1],
        [np.pi, 0, 1, 1],
        [np.pi, 0, 1, -1]]
    #
    angleAxis = np.asarray(angleAxis).transpose()
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    csym = QuatOfAngleAxis(angle, axis)

    return csym


def HexSymmetries():
    '''
    HexSymmetries - Quaternions for hexagonal symmetry group.

      USAGE:

      hsym = HexSymmetries

      INPUT:  none

      OUTPUT:

      hsym is 4 x 12,
           it is the hexagonal symmetry group represented
           as quaternions


    '''
    p3 = np.pi/3
    p6 = np.pi/6
    ci = np.atleast_2d(np.cos(p6*(np.arange(6))))
    si = np.atleast_2d(np.sin(p6*(np.arange(6))))
    z6 = np.zeros((1, 6))
    w6 = np.ones((1, 6))
    pi6 = np.tile(np.pi, [1, 6])
    p3m = np.atleast_2d(p3*(np.arange(6)))

    sixfold = np.concatenate((p3m, z6, z6, w6))
    twofold = np.concatenate((pi6, ci, si, z6))

    angleAxis = np.asarray(np.concatenate((sixfold, twofold), axis=1))
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    hsym = QuatOfAngleAxis(angle, axis)

    return hsym


def OrtSymmetries():
    '''
    OrtSymmetries - Orthorhombic symmetry group.

      USAGE:

      osym = OrtSymmetries

      INPUT:  none

      OUTPUT:

      osym is 4 x 4, 
           the quaternions for the symmetry group


    '''
    angleAxis = [
        [0.0, 1, 1, 1],
        [np.pi, 1, 0, 0],
        [np.pi, 0, 1, 0],
        [np.pi, 0, 0, 1]]

    angleAxis = np.asarray(angleAxis).transpose()
    angle = angleAxis[0, :]
    axis = angleAxis[1:4, :]
    #
    #  Axis does not need to be normalized it is done
    #  in call to QuatOfAngleAxis.
    #
    osym = QuatOfAngleAxis(angle, axis)

    return osym

def Misorientation(q1, q2, sym=CubSymmetries()):
    '''
    Misorientation - Return misorientation data for quaternions.

      USAGE:

      angle = Misorientation(q1, q2, sym)
      [angle, mis] = Misorientation(q1, q2, sym)

      INPUT:

      q1 is 4 x n1, 
         is either a single quaternion or a list of n quaternions
      q2 is 4 x n,  
         a list of quaternions

      OUTPUT:

      angle is 1 x n, 
            the list of misorientation angles between q2 and q1
      mis   is 4 x n, (optional) 
            is a list of misorientations in the fundamental region 
            (there are many equivalent choices)

      NOTES:

      *  The misorientation is the linear tranformation which
         takes the crystal basis given by q1 to that given by
         q2.  The matrix of this transformation is the same
         in either crystal basis, and that is what is returned
         (as a quaternion).  The result is inverse(q1) * q2.
         In the sample reference frame, the result would be
         q2 * inverse(q1).  With symmetries, the result is put
         in the fundamental region, but not into the Mackenzie cell.


    '''
    q1 = mat2d_row_order(q1)
    q2 = mat2d_row_order(q2)

    f1 = q1.shape
    f2 = q2.shape

    if f1[1] == 1:
        q1 = np.tile(q1, (1, f2[1]))

    q1i = np.concatenate((np.atleast_2d(-1 * q1[0, :]), np.atleast_2d(q1[1:4, :])), axis=0)

    mis = ToFundamentalRegionQ(QuatProd(q1i, q2), sym)

    angle = 2 * np.arccos(np.minimum(1, mis[0, :]))

    return (angle, mis)


def UnitVector(vec, *args):
    '''
    UnitVector - Normalize an array of vectors.

      USAGE:

      uvec = UnitVector(vec)
      uvec = UnitVector(vec, ipmat)

      INPUT:

      vec   is m x n, 
            an array of n nonzero vectors of dimension m
      ipmat is m x m, (optional)
            this is a (SPD) matrix which defines the inner product
            on the vectors by the rule:  
               norm(v)^2 = v' * ipmat * v

            If `ipmat' is not specified, the usual Euclidean 
            inner product is used.

      OUTPUT:

      uvec is m x n,
           the array of unit vectors derived from `vec'


    '''

    vec = mat2d_row_order(vec)

    m = vec.shape[0]

    if len(args) > 0:
        ipmat = args[0]
        nrm2 = np.sum(vec.conj() * np.dot(ipmat, vec), axis=0)
    else:
        nrm2 = np.sum(vec.conj() * vec, axis=0)

    nrm = np.tile(np.sqrt(nrm2), (m, 1))
    uvec = vec / nrm

    return uvec
    
def mat2d_row_order(mat):
    '''
    It takes in a mat nxm or a vec of n length and returns a 2d numpy array
    that is nxm where m is a least 1 instead of mxn where m is 1 like the 
    numpy.atleast_2d() will do if a vec is entered
    
    Input: mat - a numpy vector or matrix with dimensions of n or nxm
    output: mat - a numpy matrix with dimensions nxm where m is at least 1    
    
    '''
    
    mat = np.atleast_2d(mat) 
    if mat.shape[0] == 1:
        mat = mat.T
        
    return mat

def RankOneMatrix(vec1, *args):
    '''
    RankOneMatrix - Create rank one matrices (dyadics) from vectors. It therefore simply computes the 
    outer product between two vectors, $v_j \otimes v_i$

      USAGE:

      r1mat = RankOneMatrix(vec1)
      r1mat = RankOneMatrix(vec1, vec2)

      INPUT:

      vec1 is m1 x n, 
           an array of n m1-vectors 
      vec2 is m2 x n, (optional) 
           an array of n m2-vectors

      OUTPUT:

      r1mat is m1 x m2 x n, 
            an array of rank one matrices formed as c1*c2' 
            from columns c1 and c2

      With one argument, the second vector is taken to
      the same as the first.

      NOTES:

      *  This routine can be replaced by MultMatArray.


    '''

    vec1 = mat2d_row_order(vec1)

    if len(args) == 0:
        vec2 = vec1.copy()
    else:
        vec2 = np.atleast_2d(args[0])

    m = vec1.shape
    n = vec2.shape[0]

    if m[0] != n:
        print('dimension mismatch: vec1 and vec2 (first dimension)')
        raise ValueError('dimension mismatch between vec1 and vec2 (1st dim)')

    rrom = np.zeros((m[0], n, m[1]))

    for i in range(m[1]):
        rrom[:, :, i] = np.outer(vec1[:, i], vec2[:, i])

    return rrom

def misorientationSpread(quats, el_vol, grains, sym=CubSymmetries()):
    '''
    Computes the intragrain heterogeneous misorientation metric for a given time step
    as obtained by taking the Winv tensor in the paper given down below.
    
    Parameters
    ----------
    quats : 2D numpy double array with dimensions of [4, nelems]
        The quaternions outputted from our simulation for a given time step
    el_vol : 1D numpy int array with dimension of nelems 
        The outputted element volumes from our simulation for a given time step
    grains : 1D numpy int array with dimensions of nelems
        The grain IDs for each nelems
    sym : 2D numpy double array with dimensions [4, npts], optional
        The symmetry group to bring the orientations back into the fundamental
        region. The default is CubSymmetries().

    Returns
    -------
    gspread : 1D numpy double array with dimension of num_unique_grains
        The intragrain heterogeneous misorientation metric for a given time step
        as obtained by taking the Winv tensor defined in the following paper:
          "A Methodology for Determining Average Lattice Orientation and 
          Its Application to the  Characterization of Grain Substructure",
    
          Nathan R. Barton and Paul R. Dawson,
    
          Metallurgical and Materials Transactions A,
          Volume 32A, August 2001, pp. 1967--1975.

    '''
    ugrains = np.unique(grains)
    
    gspread = np.empty((ugrains.shape[0]))
    
    for igrain in ugrains:
        indlog = grains == igrain
        wts = el_vol[indlog]
        misorient = quats[:, indlog]
        misorient = mat2d_row_order(misorient)
        d, n = misorient.shape  
        wts = np.tile(wts, (3, 1))        
        wts1 = np.tile(wts[0, :], (4, 1))
        #This is a good enough approximation of the mean quaternion
        # misOriCen = UnitVector(np.sum(misorient * wts1, axis=1))
        misOriCen = QuatMean(misorient)
        misAngs, misorient = Misorientation(misOriCen, misorient, sym)
    
        ang = mat2d_row_order(2 * np.arccos(misorient[0, :]))
        wsc = np.zeros(ang.shape)
    
        limit = (ang < np.finfo(float).eps)
        nlimit = (ang > np.finfo(float).eps)
    
        angn = ang[nlimit]
    
        wsc[nlimit] = angn / np.sin(angn / 2.0)
        wsc[limit] = 2
    
        wi = misorient[1:4, :] * np.tile(wsc.T, (3, 1))
    
        Winv = np.sum(RankOneMatrix(wi * wts, wi), axis=2)
        gspread[igrain - 1] = np.sqrt(np.trace(Winv))
        
    return gspread