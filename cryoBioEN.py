import sys
import os
import numpy as np
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

# map with noise plus map threshold which equals 3 x noise_std
noise = float(sys.argv[5])
em_map_noise,em_threshold = add_noise(em_map,noise)

# Mask of the EM map (where the density is > threshold)
# Plus EM density with zerroed < threshold density
tmp_data = em_map_noise - em_threshold
em_map_threshold = tmp_data.clip(min=0)
mask_exp = np.where(em_map_threshold > 0)

# Saving map with noise
os.system("rm map_noise.mrc")
write_map(em_map_noise,"map_noise.mrc",map_param)

"""
Fitting structures into density using Situs
"""
# Fitting 1ake structures
#for i in range(1,51):
# os.system('~/soft/Situs_3.1/bin/colores map_noise.mrc /home/didymos/Linux_05.2021/Projects/BioEN/ADK/1ake/structures/'+str(i)+'_fit.pdb -res 10 -nprocs 6')
# os.system('mv col_best_001.pdb /home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/1ake/structures/'+str(i)+'_rb_fit.pdb')
# os.system('rm col_*')

# Fitting 4ake structures
#for i in range(1,51):
# os.system('~/soft/Situs_3.1/bin/colores map_noise.mrc /home/didymos/Linux_05.2021/Projects/BioEN/ADK/4ake/structures/'+str(i)+'_fit.pdb -res 10 -nprocs 6')
# os.system('mv col_best_001.pdb /home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/4ake/structures/'+str(i)+'_rb_fit.pdb')
# os.system('rm col_*')

"""
STRUCTURAL ENSEMBLE
"""

# OK Now we read models from 1ake and 4ake
# We use 50 model from 1ake and 50 models from 4ake

# Number of structures/models
N_models = 100

"""
Reference Weights
"""

# Reference weights for models [UNIFORM]
w0 = np.ones(N_models)/N_models

# Initial weights for models to start optimization [UNIFORM]
w_init = np.ones(N_models)/N_models

PDBs_1ake = glob.glob("/home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/1ake/structures/*rb_fit.pdb")[:50]
PDBs_4ake = glob.glob("/home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/4ake/structures/*rb_fit.pdb")[:50]

# PDB files
PDBs=PDBs_1ake+PDBs_4ake

# Generating array of EM maps based on structures
sim_em_data = np.array(pdb2map_array(PDBs,sigma=np.float(sys.argv[3])),map_param,cryoem_param)
