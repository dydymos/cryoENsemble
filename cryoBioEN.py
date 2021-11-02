import sys
import os
import glob
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
"" Creating average EM map
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
"" Fitting structures into density using Situs
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
"" STRUCTURAL ENSEMBLE
"""

# OK Now we read models from 1ake and 4ake
# We use 50 model from 1ake and 50 models from 4ake

# Number of structures/models
N_models = 100

PDBs_1ake = glob.glob("/home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/1ake/structures/*rb_fit.pdb")[:50]
PDBs_4ake = glob.glob("/home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/4ake/structures/*rb_fit.pdb")[:50]

# PDB files
PDBs=PDBs_1ake+PDBs_4ake

# Generating array of EM maps based on structures
sim_em_data = np.array(pdb2map_array(PDBs,float(sys.argv[3]),map_param,cryoem_param))


"""
"" MASKING
"""

# Array with experimental mask
mask_exp_array=np.array(mask_exp)

# mask for simulated maps
mask_sim = mask_sim_gen(sim_em_data,N_models)

# Geting combined voxels from exp and from sim structures
mask_comb = combine_masks(mask_exp_array,mask_sim)

# Masked experimental data
exp_em_mask = exp_em[mask_comb]

# Number of non-zero voxels
N_voxels=np.shape(mask_comb)[1]

# New simulated map with only voxels corresponding to the experimental map
sim_em_v=np.zeros([N_models,N_voxels])
for i in range(0,N_models):
    sim_em_v[i,]=sim_em_data[i][mask_comb]


"""
"" Optimization via Log-Weights as in Kofinger et. al 2019
"""

# To optimize the log-posterior under the constraints: sum of weights = 1 and weights >= 0 and to determine the optimal values of the weights by gradient-based minimization, we introduce log-weights: log_w = np.log(w)
# They are determined up to an additive constant, which cancels during the normalization of the w.
# Without loss of generality, because all w>0 we can set one of the log_w to 0. Then we would need to optimize log-posterior function for the remaining N-1 weights.

"""
"" Initial parameters
"""

"""
"" Reference Weights
"""

# Reference weights for models [UNIFORM]
w0 = np.ones(N_models)/N_models
# Initial weights for models to start optimization
# UNIFORM
w_init = np.ones(N_models)/N_models

# Initial scalling factor
sf_init = 1.0

# std deviation
std = np.ones(N_voxels)*noise_lvl
# Getting optimal scalling factor
sf_start = leastsq(coeff_fit, sf_init, args=(w0,std,sim_em_v,exp_em_mask))[0]
