import sys
from bioEN_functions import *
from cryoEM_methods import *
from reference_map_ADK import *

"""

USE: cryoBioEN.py ADK.mrc w1 w2 sigma noise

w1 - weights for 1AKE
w2 - weights for 4AKE

"""

# Getting cryoEM map properties
map_param = reference_map(sys.argv[1])
# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

"""
Creating average EM map
"""
# Weights for 1AKE and 4AKE:
em_weights=np.array([float(sys.argv[2]),float(sys.argv[3])])

# map resolution
sigma = float(sys.argv[4])

# average map
em_map = pdb2map_avg(em_weights,sigma,["1ake.pdb","4ake_aln.pdb"],map_param,cryoem_param)

# map with noise
noise = float(sys.argv[5])
em_map_noise = add_noise(em_map,noise)

write_map(em_map_noise,"mapa.mrc",map_param)
