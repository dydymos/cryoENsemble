import sys
import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from reference_map import *
from plot import *

"""

"" USE: cryoBioEN.py ref_map.mrc resolution noise ID

"""

# Getting cryoEM map properties
map_param = reference_map(sys.argv[1])
# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

"""
"" Creating average EM map
"""

# map resolution
sigma = float(sys.argv[2])*0.225

# average map map generate from randomly chosen 10 maps
M = 10
path = "/home/didymos/Linux_05.2021/Projects/BioEN/NC/minim"
random_pdbs, rlist = random_pdbs(path,M)
em_weights=np.zeros(M)+0.1
em_map = pdb2map_avg(em_weights,sigma,random_pdbs,map_param,cryoem_param)
# writing weights
with open("random_list.dat", "a") as f:
    f.write("\n")
    np.savetxt(f,rlist,newline=" ",fmt="%i")

# map with noise plus map threshold which equals 3 x noise_std
noise = float(sys.argv[3])
em_map_noise,em_threshold = add_noise(em_map,noise)

# Mask of the EM map (where the density is > threshold)
# Plus EM density with zerroed < threshold density
tmp_data = em_map_noise - em_threshold
em_map_threshold = tmp_data.clip(min=0)
mask_exp = np.where(em_map_threshold > 0)

# ID
ID = sys.argv[4]
# Saving map with noise
os.system("rm map_noise_"+str(ID)+".mrc")
write_map(em_map_noise,"map_noise_"+str(ID)+".mrc",map_param)

"""
"" STRUCTURAL ENSEMBLE
"""

# OK Now we read 100 models previously minimized

# Number of structures/models
N_models = 100

PDBs = []

for i in range(1,101):
   PDBs.append(path+"/minim_"+str(i)+'.pdb')

# Generating array of EM maps based on structures
sim_em_data = np.array(pdb2map_array(PDBs,sigma,map_param,cryoem_param))


"""
"" MASKING
"""

# Array with experimental mask
#mask_exp_array=np.array(mask_exp)

# mask for simulated maps
#mask_sim = mask_sim_gen(sim_em_data,N_models)

# Geting combined voxels from exp and from sim structures
#mask_comb = combine_masks(mask_exp_array,mask_sim)

# Masked experimental data
exp_em_mask = em_map_threshold[mask_exp]

# Number of non-zero voxels
N_voxels=np.shape(mask_exp)[1]

# New simulated map with only voxels corresponding to the experimental signal
sim_em_v_data=np.zeros([N_models,N_voxels])
for i in range(0,N_models):
    sim_em_v_data[i]=sim_em_data[i][mask_exp]

"""
"" Optimization via Log-Weights as in Kofinger et. al 2019
"""

# To optimize the log-posterior under the constraints: sum of weights = 1 and weights >= 0 and to determine the optimal values of the weights by gradient-based minimization, we introduce log-weights: log_w = np.log(w)
# They are determined up to an additive constant, which cancels during the normalization of the w.
# Without loss of generality, because all w>0 we can set one of the log_w to 0. Then we would need to optimize log-posterior function for the remaining N-1 weights.

"""
"" Initial parameters, reference weights and log-weights
"""

w0,w_init,g0,g_init,sf_init = bioen_init_uniform(N_models)

# std deviation
std = np.ones(N_voxels)*em_threshold

# Getting initial optimal scalling factor
sf_start = leastsq(coeff_fit, sf_init, args=(w0,std,sim_em_v_data,exp_em_mask))[0]

"""
"" Parameters for optimization algorithm
"""

# For now we can only use BFGS algorithm as is coded in SCIPY

epsilon = 1e-08
pgtol = 1e-05
maxiter = 5000

# Number of iterations
n_iter = 10

# Theta
thetas = [1000000,100000,10000,1000,100,10,1,0]

# Running BioEN iterations through hyperparameter Theta:
# w_opt_array, S_array, chisqrt_array = bioen(sim_em_v_data,exp_em_mask,std,thetas, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)

# Running BioEN in a loop so we can apply knee dectection algorithm later
w_opt_d = dict()
sf_opt_d = dict()
s_dict = dict()
chisqrt_d = dict()
for th in thetas:
    w_temp, sf_temp = bioen_single(sim_em_v_data,exp_em_mask,std,th, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)
    w_opt_d[th] = w_temp
    sf_opt_d[th] = sf_temp
    s_dict[th] = get_entropy(w0,w_temp)
    chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_em_v_data*sf_temp,exp_em_mask)

# Knee locator used to find sensible theta value
theta_index = knee_loc(list(s_dict.values()),list(chisqrt_d.values()))
theta_knee = thetas[theta_index]
theta_index_sort = theta_index

for i in range(0,10):
    thetas_old = np.sort(list(w_opt_d.keys()))[::-1]
    theta_up = (thetas_old[theta_index_sort - 1] + thetas_old[theta_index_sort])/2.
    theta_down = (thetas_old[theta_index_sort + 1] + thetas_old[theta_index_sort])/2.
    for th in theta_up,theta_down:
        w_temp, sf_temp = bioen_single(sim_em_v_data,exp_em_mask,std,th, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)
        w_opt_d[th] = w_temp
        sf_opt_d[th] = sf_temp
        s_dict[th] = get_entropy(w0,w_temp)
        chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_em_v_data*sf_temp,exp_em_mask)
        # Knee locator
    theta_index_new = knee_loc(list(s_dict.values()),list(chisqrt_d.values()))
    theta_new = list(w_opt_d.keys())[theta_index_new]
    thetas_sorted = np.sort(list(w_opt_d.keys()))[::-1]
    theta_index_sort = np.where(theta_new == thetas_sorted)[0][0]


"""
"" PLOTS
"""
# L-CURVE
plot_lcurve(s_dict,chisqrt_d,theta_new,N_voxels,ID)

# Neff
plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,ID)

# WEIGHTS
plot_weights(w_opt_d,theta_new,ID)

"""
"" CORRELATION with exp map
"""
cc,cc_prior,cc_single = map_correlations(sim_em_v_data,exp_em_mask,w_opt_d,w0,theta_new)

"""
"" WRITING POSTERIOR AND PRIOR MAP
"""
# Saving posterior map
os.system("rm map_posterior_"+str(ID)+".mrc")
sim_em_rew = np.dot(sim_em_data.T,w_opt_d[theta_new]).T
write_map(sim_em_rew,"map_posterior_"+str(ID)+".mrc",map_param)

os.system("rm map_prior_"+str(ID)+".mrc")
sim_em_rew = np.dot(sim_em_data.T,w0).T
write_map(sim_em_rew,"map_prior_"+str(ID)+".mrc",map_param)

file = open("statistics.dat", "a")

file.write("Theta value chosen by Kneedle algorithm: "+str(theta_new)+"\n")
file.write("Reduced Chisqrt: " + str(chisqrt_d[theta_new]/N_voxels)+"\n")
file.write("Neff: " + str(np.exp(s_dict[theta_new]))+"\n")
file.write("Posteriori Correlation: " + str(cc)+"\n")
file.write("Priori Correlation: " + str(cc_prior)+"\n")
file.write("Single Best structure Correlation: " + str(cc_single)+"\n")
