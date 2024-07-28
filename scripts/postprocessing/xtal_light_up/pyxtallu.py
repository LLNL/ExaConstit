import numpy as np
import xtal_light_up.xtal_light_up as xlup

from hexrd import rotations as rot
from hexrd import material , valunits

from hexrd.matrixutil import  \
    columnNorm, unitVector, \
    skewMatrixOfVector, findDuplicateVectors, \
    multMatArray, nullSpace

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
        minimum d-spacing in angstroms (alt peak upper bound). The default is 0.6.
        
    """
    
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max
    matl.dmin = valunits.valWUnit('lp' , 'length' , dmin_angstroms , 'angstrom')
    
    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls , dtype = bool))
    
    return matl

def applySym(vec, qsym, csFlag=False, cullPM=False, tol=rot.cnst.sqrt_epsf):
    """
    Apply symmetry group to a single 3-vector (columnar) argument.

    csFlag : centrosymmetry flag
    cullPM : cull +/- flag
    """
    nsym = qsym.shape[1]
    Rsym = rot.rotMatOfQuat(qsym)
    if nsym == 1:
        Rsym = np.array([Rsym, ])
    allhkl = multMatArray(
        Rsym, np.tile(vec, (nsym, 1, 1))
    ).swapaxes(1, 2).reshape(nsym, 3).T

    if csFlag:
        allhkl = np.hstack([allhkl, -1*allhkl])
    eqv, uid = findDuplicateVectors(allhkl, tol=tol, equivPM=cullPM)

    return allhkl[np.ix_(list(range(3)), uid)]

def distanceToFiber(c, s, q, qsym, **kwargs):
    """
    Calculate symmetrically reduced distance to orientation fiber.

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    qsym : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    csymFlag = False
    B = np.eye(3)

    arglen = len(kwargs)

    if len(c) != 3 or len(s) != 3:
        raise RuntimeError('c and/or s are not 3-vectors')

    # argument handling
    if arglen > 0:
        argkeys = list(kwargs.keys())
        for i in range(arglen):
            if argkeys[i] == 'centrosymmetry':
                csymFlag = kwargs[argkeys[i]]
            elif argkeys[i] == 'bmatrix':
                B = kwargs[argkeys[i]]
            else:
                raise RuntimeError("keyword arg \'%s\' is not recognized"
                                   % (argkeys[i]))

    c = unitVector(np.dot(B, np.asarray(c)))
    s = unitVector(np.asarray(s).reshape(3, 1))

    nq = q.shape[1]  # number of quaternions
    rmats = rot.rotMatOfQuat(q)  # (nq, 3, 3)

    csym = applySym(c, qsym, csymFlag)  # (3, m)
    m = csym.shape[1]  # multiplicity

    if nq == 1:
        rc = np.dot(rmats, csym)  # apply q's to c's
        sdotrc = np.dot(s.T, rc).max()
    else:
        rc = multMatArray(
            rmats, np.tile(csym, (nq, 1, 1))
        )  # apply q's to c's

        sdotrc = np.dot(
            s.T,
            rc.swapaxes(1, 2).reshape(nq*m, 3).T
        ).reshape(nq, m).max(1)

    d = rot.arccosSafe(np.array(sdotrc))

    print(d)

    return d


# help(xlup)

lattice_orientations = np.zeros((1,1,4))
strains = np.zeros((1,1,6))

lattice_orientations[:,:,0] = 1.0 / np.sqrt(1.5)
lattice_orientations[:,:,1] = 0.5 / np.sqrt(1.5)
lattice_orientations[:,:,2] = 0.5 / np.sqrt(1.5)

strains[:,:,0] = 1.0
strains[:,:,1] = 2.0
strains[:,:,2] = 3.0
strains[:,:,3] = 1.0
strains[:,:,4] = 2.0
strains[:,:,5] = 3.0

xlup.strain_lattice2sample(lattice_orientations, strains)

gdots = np.zeros((1,1,12))
gdots[:,:,0] = 1e-4
gdots[:,:,1] = 1e-4
gdots[:,:,2] = -1e-4
tay_factor = np.zeros((1,1))

n_arr = (1 / np.sqrt(3)) * np.array([
    [ 1 ,  1 ,  1] ,
    [ 1 ,  1 ,  1] ,
    [ 1 ,  1 ,  1] ,
    [-1 ,  1 ,  1] ,
    [-1 ,  1 ,  1] ,
    [-1 ,  1 ,  1] ,
    [-1 , -1 ,  1] ,
    [-1 , -1 ,  1] ,
    [-1 , -1 ,  1] ,
    [ 1 , -1 ,  1] ,
    [ 1 , -1 ,  1] ,
    [ 1 , -1 ,  1]
    ])

s_arr = (1 / np.sqrt(2)) * np.array([
    [ 0 ,  1 , -1] ,
    [-1 ,  0 ,  1] ,
    [ 1 , -1 ,  0] ,
    [-1 ,  0 , -1] ,
    [ 0 , -1 ,  1] ,
    [ 1 ,  1 ,  0] ,
    [ 0 , -1 , -1] ,
    [ 1 ,  0 ,  1] ,
    [-1 ,  1 ,  0] ,
    [ 1 ,  0 , -1] ,
    [ 0 ,  1 ,  1] ,
    [-1 , -1 ,  0]
    ])

# calculate FCC Schmid tensor
m_arr = np.zeros((n_arr.shape[0] , s_arr.shape[1], n_arr.shape[1]))
for ii in range(m_arr.shape[0]) :
    m_arr[ii, :, :] = np.tensordot(s_arr[ii] , n_arr[ii] , axes = 0)
    m_arr[ii, :, :] =  0.5 * (m_arr[ii, :, :] + m_arr[ii, :, :].T)
m_arr = np.reshape(m_arr, (n_arr.shape[0] , s_arr.shape[1] * n_arr.shape[1]))
# calculate DpEff from shear strain rates on each slip system
def dp_eff(gdots) :
    
    Dp = np.sum(m_arr * gdots[: , None] , axis = 0).reshape((3,3))

    # Dp = Lp + Lp.T - np.diag(Lp.diagonal())
    
    return np.sqrt((2 / 3) * np.tensordot(Dp , Dp))

dp_effs = np.zeros_like(tay_factor)

xlup.calc_taylor_factors(gdots, tay_factor, dp_effs)

# get material symmetry - here is a simple example for IN625
matl = make_matl(mat_name = 'FCC' , sgnum = 225 , lparms = [3.60 ,])
pd = matl.planeData

hkl = np.zeros((4, 3))
hkl[0, :] = [1,1,1]
hkl[1, :] = [2,0,0]
hkl[2, :] = [2,2,0]
hkl[3, :] = [3,1,1]
s_dir = np.asarray([0.0,0.0,1.0])
quats = np.swapaxes(lattice_orientations, 0, 2)
for jj in range(4):
    # compute crystal direction from planeData
    c_dir = np.atleast_2d(np.dot(pd.latVecOps['B'] , hkl[jj, :].T)).T
    distance = distanceToFiber(c_dir, s_dir, quats, pd.getQSym())

in_fibers = np.zeros((hkl.shape[0], 1, 1), dtype=bool)

xlup.calc_within_fibers(lattice_orientations, s_dir, hkl, 3.60, np.deg2rad(5.0), in_fibers)

# # compute distance from quaternions to crystallographic fiber
# distance = np.degrees(rot.distanceToFiber(c_dir , s_dir , quats , pd.getQSym()))

# # filter for distances within specified distance bound
# in_fiber = np.where(distance < distance_bnd)[0]
# in_fiber_arr[jj] = in_fiber
