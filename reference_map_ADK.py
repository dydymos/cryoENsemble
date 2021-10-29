import numpy as np
import sys
import mrcfile

##################
# PDB2EM_DENSITY #
##################
# dictionary of sigma and weight per atom type
# From Bonomi et al 2018 - atomic scattering factors fitted with single Gaussian - Ai*exp(-Bi*s^2), where s is scattering factor
# Sigma = Bi, Weights=Ai
SIGMA={}; WEIGHT={}
SIGMA["C"]=15.146;   WEIGHT["C"]=2.49982
SIGMA["O"]=8.59722;  WEIGHT["O"]=1.97692
SIGMA["N"]=11.1116;  WEIGHT["N"]=2.20402
SIGMA["S"]=15.8952;  WEIGHT["S"]=5.14099

# transform sigma in real space and get useful stuff
PREFACT_={}; INVSIG2={}
for key in SIGMA:
    SIGMA[key] = np.sqrt( 0.5 * SIGMA[key] ) / np.pi
    INVSIG2[key] = 1.0 / SIGMA[key] / SIGMA[key]
    PREFACT_[key] = WEIGHT[key] / np.sqrt(2.0*np.pi) / SIGMA[key]



# constant parameter
maxSD_ = 3.0
# build delta dictionary
delta={};
for key in SIGMA:
    delta[key]=int(np.ceil(maxSD_ * SIGMA[key] / VOX_))

def reference_map(map):
    """
    Loads simulated EM map from the open state structure
    We are using it to get initial number of voxels per xyz and voxel size
    This map has been generated in chimerax with 10A
    """
    exp_em = mrcfile.open(map, mode='r')
    # Number of bins
    nx=exp_em.header['nx']
    ny=exp_em.header['ny']
    nz=exp_em.header['nz']
    # voxel dimension
    VOX_= exp_em.voxel_size['x']
    # Map center
    em_origin = np.array([-28.634, -4.097, -39.461])
    return nx,ny,nz,VOX_,em_origin
