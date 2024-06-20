import numpy as np
import sys
import mrcfile




def reference_map(map):
    """
    Reference cryo-EM map
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
    return map_param
