#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def fepxconn_2_vtkconn(conn):
    '''
        Takes in the fepx connectivity array and switches it to the vtk
        format.
        
        Input:
            conn - a numpy array of the elemental connectivity array for a
                quadratic tetrahedral element with FePX nodal ordering
        Output:
            vtk_conn - a numpy array of the elemental connectivity array
                given in the vtk nodal order for a quadratic tetrahedral 
                element
    
    '''
   
    #Rearrangement of array
    
    vtko = np.array([0, 2, 4, 9, 1, 3, 5, 6, 7, 8], dtype = np.int8)
    
    vtk_conn = conn[vtko, :]
     
    
    return vtk_conn

def wordParser(listVals):
    '''
        Read in the string list and parse it into a floating list
        Input: listVals = a list of strings
        Output: numList = a list of floats
    '''
    numList = []
    for str in listVals:
        num = float(str)
        numList.append(num)

    return numList

def readMesh(fileLoc, fileName):
    ''' 
        Takes in the file location and file name and it then generates a dictionary structure from those files for the mesh.
        Input: fileLoc = a string of the loaction of file on your computer
               fileName = a string of the name of the file assuming they are all equal for .mesh, .kocks, and .grain
        Outpute: mesh = a dictionary that contains the following fields in it:
            name = file location
            grains = what grain each element corresponds to
            con = connectivity of the mesh for each element
            crd = coordinates of each node
            surfaceNodes = surface nodes of the mesh
            kocks = kocks angles for each grain
            phases = phase number of each element
    '''
    surfaceNodes = []
    con = []
    crd = []
    name = fileLoc
    meshLoc = fileLoc + fileName + '.mesh'
    grainLoc = fileLoc + fileName + '.grain'
    kockLoc = fileLoc + fileName + '.kocks'        
    grains = []
    phases = []
    kocks = []
    mesh = {}
    mesh['name'] = name

    with open(meshLoc) as f:
        #        data = f.readlines()
        for line in f:
            words = line.split()
            #            print(words)
            lenWords = len(words)
            if not words:
                continue
            if lenWords == 4:
                nums = wordParser(words)
                crd.append(nums[1:4])
            if lenWords == 7:
                nums = wordParser(words)
                surfaceNodes.append(nums[0:7])
            if lenWords == 11:
                nums = wordParser(words)
                con.append(nums[1:11])

    grains = np.genfromtxt(grainLoc, usecols=(0), skip_header=1, skip_footer=0)
    ugrains = np.unique(grains)
    phases = np.genfromtxt(grainLoc, usecols=(1), skip_header=1, skip_footer=0)
    kocks = np.genfromtxt(kockLoc, usecols=(0, 1, 2), skip_header=2, skip_footer=1)
    if not kocks.shape[0] == ugrains.shape[0]:
            kocks = np.genfromtxt(kockLoc, usecols=(0, 1, 2), skip_header=2, skip_footer=0)
    mesh['con'] = np.require(np.asarray(con, order='F', dtype=np.int32).transpose(), requirements=['F'])
    mesh['crd'] = np.require(np.asarray(crd, order='F').transpose(), requirements=['F'])
    mesh['surfaceNodes'] = np.require(np.asarray(surfaceNodes, order='F',dtype=np.int32).transpose(), requirements=['F'])
    mesh['grains'] = np.asfortranarray(grains.transpose(), dtype=np.int32)
    kocks = np.atleast_2d(np.asfortranarray(kocks.transpose())) 
    if kocks.shape[0] == 1:
        kocks = kocks.T
    mesh['kocks'] = kocks
    mesh['phases'] = np.asfortranarray(phases.transpose(),dtype=np.int8)

    return mesh

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileLoc = '/home/robert/Documents/mfem_test/'
fileBase = 'n100-id256-gg-custom'

fileOut = 'fepx2mfem.vtk'

mesh = readMesh(fileLoc,fileBase)

conn_out = fepxconn_2_vtkconn(mesh['con'])
conn_len = conn_out.shape[0]

crds = mesh['crd']
grains = mesh['grains']

npts = crds.shape[1]
nelems = conn_out.shape[1]
ncells_item = nelems * (conn_len + 1)

ten_arr = np.atleast_2d(np.ones(nelems, dtype=np.int32)) * conn_len

conn_out = np.concatenate((ten_arr, conn_out),axis=0)

quad_tet_arr = np.atleast_2d(np.ones(nelems, dtype=np.int32)) * 24

with open(fileLoc+fileOut,'wb') as f_handle:
    f_handle.write(bytes('# vtk DataFile Version 3.0\n','UTF-8'))
    f_handle.write(bytes('FEpX_2_MFEM mesh\n','UTF-8'))
    f_handle.write(bytes('ASCII\n','UTF-8'))
    f_handle.write(bytes('DATASET UNSTRUCTURED_GRID\n','UTF-8'))
    f_handle.write(bytes('POINTS '+str(npts)+' double\n','UTF-8'))
    np.savetxt(f_handle,crds.T)
    f_handle.write(bytes('CELLS '+str(nelems)+' '+str(ncells_item)+'\n','UTF-8'))
    np.savetxt(f_handle,conn_out.T, fmt='%i')
    f_handle.write(bytes('CELL_TYPES '+str(nelems)+'\n','UTF-8'))
    np.savetxt(f_handle,quad_tet_arr, fmt='%i', delimiter='\n')
    f_handle.write(bytes('CELL_DATA ' +str(nelems)+ '\n','UTF-8'))
    f_handle.write(bytes('SCALARS material int\n','UTF-8'))
    f_handle.write(bytes('LOOKUP_TABLE default\n','UTF-8'))
    np.savetxt(f_handle,grains, fmt='%i', delimiter='\n')

