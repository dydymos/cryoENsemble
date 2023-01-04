import os
import glob
import numpy as np
from plot import *
import argparse
import mrcfile
import MDAnalysis
from MDAnalysis import *
import time

start_time = time.perf_counter()

def checker_method(method):
    if method != 'bonomi' and method != 'chimerax' and method !='gromaps':
        raise argparse.ArgumentTypeError('Invalid value. Only two methods are available to generate EM map based on the pdb structure: based on Bonomi et al. 2018 ("bonomi"), based on ChimeraX molmpa ("chimerax") or GROMAPS ("gromaps")')
    return method

def reference_map(map):
    """
    Loads cryoE EM map from the mrc file
    """
    map_param = {}
    exp_em = mrcfile.open(map, mode='r')
    # Number of bins
    nx=exp_em.header['nx']
    ny=exp_em.header['ny']
    nz=exp_em.header['nz']
    # voxel dimension
    VOX= exp_em.voxel_size['x']
    # Map center
    em_origin = np.array([exp_em.header['origin']['x'],exp_em.header['origin']['y'],exp_em.header['origin']['z']])
    map_param["nx"] = nx
    map_param["ny"] = ny
    map_param["nz"] = nz
    map_param["vox"] = VOX
    map_param["em_origin"] = em_origin
    return map_param, exp_em.data


def cryoEM_parameters(map_param):
    """
    Dictionary of sigma and weight per atom type from Bonomi et al 2018 - atomic
    scattering factors fitted with single Gaussian - Ai*exp(-Bi*s^2), where s
    is scattering factor; Sigma = Bi, Weights=Ai
    """
    VOX = map_param["vox"]
    cryoem_param = {}
    SIGMA={}; WEIGHT={}
    SIGMA["C"]=15.146;   WEIGHT["C"]=2.49982
    SIGMA["O"]=8.59722;  WEIGHT["O"]=1.97692
    SIGMA["N"]=11.1116;  WEIGHT["N"]=2.20402
    SIGMA["S"]=15.8952;  WEIGHT["S"]=5.14099
    SIGMA["P"]=19.4293;  WEIGHT["P"]=5.45658
    # transform sigma in real space and get useful stuff
    PREFACT={}; INVSIG2={}
    for key in SIGMA:
        SIGMA[key] = np.sqrt( 0.5 * SIGMA[key] ) / np.pi
        INVSIG2[key] = 1.0 / SIGMA[key] / SIGMA[key]
        PREFACT[key] = WEIGHT[key] / np.sqrt(2.0*np.pi) / SIGMA[key]
    # constant parameter
    maxSD = 3.0
    # build delta dictionary
    delta={};
    for key in SIGMA:
        delta[key]=int(np.ceil(maxSD * SIGMA[key] / VOX))
    cryoem_param["prefact"] = PREFACT
    cryoem_param["invsig2"] = INVSIG2
    cryoem_param["delta"] = delta
    return cryoem_param

def cryoEM_parameters_gromaps(map_param):
    """
    Dictionary of A and B parameters for four gaussian from Briones et al. 2018
    Gaussians are in the form of A_i*exp(-B_i*d**2), where A is unitless and B is A**-2
    """
    VOX = map_param["vox"]
    cryoem_param = {}
    param_A={}; param_B={}; sigma={}
    param_A["C"]=np.array([9.253835e-02, 6.260279e-01, 1.048004e+00, 3.033389e-01]); param_B["C"]=np.array([6.890005e-01, 1.834891e+00, 6.026779e+00, 2.226268e+01])
    param_A["O"]=np.array([2.126579e-02, 2.289285e-01, 1.006541e+00, 2.285777e+00]); param_B["O"]=np.array([8.683121e-01, 1.794531e+00, 5.041622e+00, 4.493332e+01])
    param_A["N"]=np.array([8.946127e-02, 5.702793e-01, 1.181245e+00, 8.856041e-01]); param_B["N"]=np.array([9.246527e-01, 2.362661e+00, 7.689153e+00, 1.112068e+02])
    param_A["S"]=np.array([2.272068e-01, 1.373602e+00, 1.905039e+00, 6.346896e-01]); param_B["S"]=np.array([7.166818e-01, 1.850285e+00, 6.119167e+00, 2.793350e+01])
    sigma["C"] = np.sqrt(0.5 * 1/param_B["C"])
    sigma["O"] = np.sqrt(0.5 * 1/param_B["O"])
    sigma["N"] = np.sqrt(0.5 * 1/param_B["N"])
    sigma["S"] = np.sqrt(0.5 * 1/param_B["S"])
    # constant parameter
    maxSD = 5.0
    # build delta dictionary
    delta={}
    for key in sigma:
        delta[key]=int(np.max(np.ceil(maxSD * sigma[key] / VOX)))
    cryoem_param["paramA"] = param_A
    cryoem_param["paramB"] = param_B
    cryoem_param["delta"] = delta
    return cryoem_param

def pdb2map(PDB,map_param,cryoem_param):
    """
    Generates EM density maps for a structure based on Bonomi
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    INVSIG2 = cryoem_param["invsig2"]
    PREFACT = cryoem_param["prefact"]
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    u = MDAnalysis.Universe(PDB)
    # all-heavy selectors
    allheavy=u.select_atoms("type C O N S")
    # now cycle on all the atoms
    for atom in allheavy:
        # get atom type
        atype = atom.type
        # get atom position
        apos = atom.position
        # get indexes in the map
        ii = int(round((apos[0]-em_origin[0])/VOX))
        jj = int(round((apos[1]-em_origin[1])/VOX))
        kk = int(round((apos[2]-em_origin[2])/VOX))
        # get delta grid
        d=delta[atype]
        # get constant parameters
        invsig2 = INVSIG2[atype]
        pref =  PREFACT[atype]
        # add contribution to map
        for k in range(max(0, kk-d), min(kk+d+1, nz)):
            distz = (em_origin[2] + float(k) * VOX - apos[2])**2
            for j in range(max(0, jj-d), min(jj+d+1, ny)):
                disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                for i in range(max(0, ii-d), min(ii+d+1, nx)):
                    # get distance squared
                    dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                    # add contribution
                    data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2)

    return data


def pdb2map_gromaps(PDB,map_param,cryoem_param):
    """
    Generates EM density maps for a structure based on GROMAPS
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    param_A = cryoem_param["paramA"]
    param_B = cryoem_param["paramB"]
    data = np.zeros((nz,ny,nx), dtype=np.float16)
    u = MDAnalysis.Universe(PDB)
    # all-heavy selectors
    allheavy=u.select_atoms("type C O N S")
    # now cycle on all the atoms
    for atom in allheavy:
        # get atom type
        atype = atom.type
        # get atom position
        apos = atom.position
        # get indexes in the map
        ii = int(round((apos[0]-em_origin[0])/VOX))
        jj = int(round((apos[1]-em_origin[1])/VOX))
        kk = int(round((apos[2]-em_origin[2])/VOX))
        # get delta grid
        d=delta[atype]
        # get constant parameters
        A = param_A[atype]
        B = param_B[atype]
        # add contribution to map
        for k in range(max(0, kk-d), min(kk+d+1, nz)):
            distz = (em_origin[2] + float(k) * VOX - apos[2])**2
            for j in range(max(0, jj-d), min(jj+d+1, ny)):
                disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                for i in range(max(0, ii-d), min(ii+d+1, nx)):
                    # get distance squared
                    dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                    # add contribution
                    data[k][j][i] += np.sum(A * np.exp(- dist * B))

    return data

def pdb2map_chimeraX(PDBs,resolution,map_param):
    """
    Generates an array with calculated EM density maps for each structure
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    u = MDAnalysis.Universe(PDBs)
    # all-heavy selectors
    allheavy=u.select_atoms("type C O N S")
    # atomic number
    at_number = {'C':6,'N':7,'O':8,'S':16}
    # constant parameter
    maxSD = 5.0
    # sigma
    sigma = resolution*0.225
    # now cycle on all the atoms
    for atom in allheavy:
        # get atom type
        atype = atom.type
        # get atom position
        apos = atom.position
        # get indexes in the map
        ii = int(round((apos[0]-em_origin[0])/VOX))
        jj = int(round((apos[1]-em_origin[1])/VOX))
        kk = int(round((apos[2]-em_origin[2])/VOX))
        # get delta grid
        d=int(np.ceil(maxSD * sigma / VOX))
        # prefactor
        pref = at_number[atype] / np.sqrt(2.0*np.pi) / sigma
        # inverted sigma
        invsig2 = 1.0 / sigma / sigma
        # add contribution to map
        for k in range(max(0, kk-d), min(kk+d+1, nz)):
            distz = (em_origin[2] + float(k) * VOX - apos[2])**2
            for j in range(max(0, jj-d), min(jj+d+1, ny)):
                disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                for i in range(max(0, ii-d), min(ii+d+1, nx)):
                    # get distance squared
                    dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                    # add contribution
                    data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2)
    return data


def write_map(map,map_name,map_param):
    """
    Writing cryoEM map
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    map_32 = map.astype(np.float32)
    mrc = mrcfile.new(map_name,overwrite=True)
    mrc.header.nx=nx; mrc.header.ny=ny; mrc.header.nz=nz
    mrc.header.cella.x=float(nx)*VOX; mrc.header.cella.y=float(ny)*VOX; mrc.header.cella.z=float(nz)*VOX
    mrc.header.origin.x=em_origin[0]; mrc.header.origin.y=em_origin[1]; mrc.header.origin.z=em_origin[2]
    mrc.set_data(map_32)
    mrc.close()

# This is a general version of the cryoBioEN script

"""
"" INPUT PARAMETERS
"""


"""
"" USE: cryoBioEN.py map_file masktype res
"" masktype:
"" - exp -> using only voxels from experimental EM map
"" - sim -> using both voxels from experimental EM map and generated from ensemble
"""



parser = argparse.ArgumentParser(description='Running cryoBioEN for the NC')
parser.add_argument('map_file', help ="cryoEM map file name")
parser.add_argument('structure_file', help = "Structure file name")
parser.add_argument('resolution', type = float, help = 'Reference map resolution')
parser.add_argument('method',type = checker_method, help = 'Method to generate EM map based on Bonomi et al. 2018 ("bonomi"), ChimeraX molmap ("chimerax") or GROMAPS ("gromaps")')

args = parser.parse_args()

"""
"" INPUT MAP
"""

# Getting cryoEM map properties
map_param, em_map  = reference_map(args.map_file)



"""
"" STRUCTURAL ENSEMBLE
"""

u = Universe(args.structure_file,args.structure_file)

# Generating array of EM maps based on MD trajectory
print("Generating an EM maps based on the structure")
if args.method == "chimerax":
    sim_em_data = pdb2map_chimeraX(args.structure_file,args.resolution,map_param)
if args.method == "bonomi":
    # CryoEM paramters for map generators
    cryoem_param = cryoEM_parameters(map_param)
    sim_em_data = pdb2map_gpt(args.structure_file,map_param,cryoem_param)
if args.method == "gromaps":
    # CryoEM paramters for map generators
    cryoem_param = cryoEM_parameters_gromaps(map_param)
    sim_em_data = pdb2map_gromaps(args.structure_file,map_param,cryoem_param)

# Saving normalized map with noise
if args.method == "chimerax":
    write_map(sim_em_data,"map_new_c.mrc",map_param)
if args.method == "bonomi":
    write_map(sim_em_data,"map_new_b_gpt.mrc",map_param)
if args.method == "gromaps":
    write_map(sim_em_data,"map_new_g.mrc",map_param)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
