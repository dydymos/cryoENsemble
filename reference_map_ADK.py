import numpy as np
import sys
import mrcfile



def reference_map(map):
    """
    Loads simulated EM map from the open state structure
    We are using it to get initial number of voxels per xyz and voxel size
    This map has been generated in chimerax with 10A
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
    em_origin = np.array([-28.634, -4.097, -39.461])
    map_param["nx"] = nx
    map_param["ny"] = ny
    map_param["nz"] = nz
    map_param["vox"] = VOX
    map_param["em_origin"] = em_origin
    return map_param
