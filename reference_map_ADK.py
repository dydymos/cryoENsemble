import mrcfile
import numpy as np


def reference_map(map_path):
    """
    Loads simulated EM map from the open state structure
    We are using it to get initial number of voxels per xyz and voxel size
    """
    with mrcfile.open(map_pathm, mode='r') as ref_em:
        # Number of bins
        nx = ref_em.header['nx']
        ny = ref_em.header['ny']
        nz = ref_em.header['nz']
        # voxel dimension
        voxel_size = ref_em.voxel_size['x']
        # Map center
        em_origin = np.array([ref_em.header['origin']['x'], ref_em.header['origin']['y'], ref_em.header['origin']['z']])

    map_param = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "vox": voxel_size,
        "em_origin": em_origin
    }

    return map_param
