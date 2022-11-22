import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from reference_map import *
from plot import *
from os.path import exists
import argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

"""
"" PATHWAYS
"""

# pathway to models from MD simulation
path = "/home/didymos/git/cryoBioEN_NC/minim/"

random_list = np.loadtxt("random_list.dat")

"""
"" Input
"""

def checker(a):
    num = float(a)
    if num != 10 and num != 6:
        raise argparse.ArgumentTypeError('Invalid value. Only resolution equal to 6 or 10 are acceptable')
    return num


def checker_mask(mask_type):
    if mask_type != 'exp' and mask_type != 'sim':
        raise argparse.ArgumentTypeError('Invalid value. Mask can only be based on experimental (exp) or both experimental and simulations data (sim+exp)')
    return mask_type


"""
"" USE: cryoBioEN.py ref_map.mrc resolution noise masktype rand_ID res
"" masktype:
"" - exp -> using only voxels from experimental EM map
"" - sim -> using both voxels from experimental EM map and generated from ensemble
"""



parser = argparse.ArgumentParser(description='Running cryBioEN for the NC')
parser.add_argument('resM', type = checker, help = 'Reference map resolution')
parser.add_argument('resG', type = float, help = 'Generated map resolution')
parser.add_argument('random', type = int, help = 'Id list of randomly selected 10 structures from the MD ensemble')
parser.add_argument('noise', type = float, help = 'Noise level, which is defined as normal distribution centered around 0 and with std equal to X of the maximum density in the Reference map')
parser.add_argument('mask_type', type = checker_mask, help = 'Type of mask: exp or sim')

args = parser.parse_args()


"""
"" MAP details based on 6A and 10A maps generated in ChimeraX
"""
map_param = dict()
if (args.resM == 6):
    map_param['nx'] = 43
    map_param['ny'] = 39
    map_param['nz'] = 72
    map_param['vox'] = 2.0
    map_param['em_origin'] = np.array([210.323, 184.668, 79.312])
elif (args.resM == 10):
    map_param['nx'] = 33
    map_param['ny'] = 31
    map_param['nz'] = 51
    map_param['vox']= 3.3333335
    map_param['em_origin'] = np.array([198.323, 172.668, 67.312])


# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

"""
"" Creating average EM map from 10 randomly selcected models, which ID is stored in random_list file
"""

# reference map resolution
sigma = args.resM*0.225

# simulated map resolution
sigma_sim = args.resG*0.225

# average map map generate from randomly chosen 10 maps
rand_ID = args.random
random_pdbs = [path+"/minim_"+str(int(x))+".pdb" for x in random_list[rand_ID-1]]

# Number of models
M = 10

em_weights=np.ones(M)/M
em_map = pdb2map_avg(em_weights,sigma,random_pdbs,map_param,cryoem_param)

# map with noise plus map threshold which equals 3 x noise_std
noise = args.noise
em_map_noise = add_noise(em_map,noise)

# noise based threshold - 3xstd of the noise level
noise_thr = np.max(em_map)*noise*3

# normalization
em_map_norm = (em_map_noise - np.mean(em_map_noise))/np.std(em_map_noise)
noise_thr_norm = (noise_thr - np.mean(em_map_noise))/np.std(em_map_noise)

# removing negative values of map (if em_map_norm < 0 -> 0)
#em_map_threshold = np.clip(em_map_norm,0,np.max(em_map_norm))

# Mask of the EM map (where the density is > threshold)
mask_exp = np.where(em_map_norm > noise_thr_norm)

# Saving normalized map with noise
if exists("map_norm_"+str(rand_ID)+".mrc"):
    os.system("rm map_norm_"+str(rand_ID)+".mrc")
    write_map(em_map_norm,"map_norm_"+str(rand_ID)+".mrc",map_param)
else: write_map(em_map_norm,"map_norm_"+str(rand_ID)+".mrc",map_param)


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
print("Generating an array of EM maps based on the structures from MD simulation")
sim_em_data = np.array(pdb2map_array(PDBs,sigma_sim,map_param,cryoem_param))

sim_map = np.sum(sim_em_data,0)

# Saving normalized map with noise and without negative density
if exists("map_sim_"+str(rand_ID)+".mrc"):
    os.system("rm map_sim_"+str(rand_ID)+".mrc")
    write_map(sim_map,"map_sim_"+str(rand_ID)+".mrc",map_param)
else: write_map(sim_map,"map_sim_"+str(rand_ID)+".mrc",map_param)


"""
"" MASKING
"""

# Masking can take place in two ways:
# EXP - when we use mask generated from experimental map
# SIM - when we also include voxels from the map generated for each fitted structure
# for that we use threshold eaulat to 3x the simulated map std

mask = args.mask_type
if (mask == "exp"):
    # Masked experimental data
    exp_em_mask = em_map_norm[mask_exp]
    # Number of non-zero voxels
    N_voxels=np.shape(mask_exp)[1]
    # New simulated map with only voxels corresponding to the experimental signal
    sim_em_v_data=np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_em_v_data[i]=sim_em_data[i][mask_exp]

# Mask for CCC calculations
# Masked experimental data
mask_exp_array=np.array(mask_exp)
# generate mask over simulated density, using threshold equal to 3*std
mask_sim = mask_sim_gen(sim_em_data,N_models)
mask_comb = combine_masks(mask_exp_array,mask_sim)

if (mask == "sim"):
    # Number of non-zero voxels
    N_voxels=np.shape(mask_comb)[1]
    # Masked experimental data
    exp_em_mask = em_map_norm[mask_comb]
    # New simulated map with only voxels corresponding to the exp+sim
    sim_em_v_data=np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_em_v_data[i]=sim_em_data[i][mask_comb]


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

# std deviation, which is modelled from the noise distribution
std = np.ones(N_voxels)*noise_thr_norm/3

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
print("Running BioEN through preselected values of theta")
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

print("Running BioEN iterations to narrow down the theta value")
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
name = "lcurve_"+str(float(rand_ID))+".svg"
plot_lcurve(s_dict,chisqrt_d,theta_new,N_voxels,name)

# Neff
name = "Neff_"+str(float(rand_ID))+".svg"
plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,name)

# WEIGHTS
name = "weights_"+str(float(rand_ID))+".svg"
plot_weights(w_opt_d,np.arange(0,N_models),theta_new,N_models,name)

"""
"" CORRELATION with exp map
"""
#cc,cc_prior,cc_single = map_correlations(sim_em_v_data,exp_em_mask,w_opt_d,w0,theta_new)
cc,cc_prior,cc_single = map_correlations_mask(sim_em_data,em_map_norm,mask_comb,w_opt_d,w0,theta_new)

# TOP 10
best = np.argsort(w_opt_d[theta_new])[::-1][:10]+1
best_single = np.argsort(w_opt_d[theta_new])[::-1][0]

best_ratio = 0
for i in best:
    if i in random_list[rand_ID-1]: best_ratio+=1

best_ratio/=10.0

"""
"" WRITING POSTERIOR AND PRIOR MAP
"""
# Saving posterior map
sim_em_rew = np.dot(sim_em_data.T,w_opt_d[theta_new]).T
if exists("map_posterior_"+str(rand_ID)+".mrc"):
    os.system("rm map_posterior_"+str(rand_ID)+".mrc")
    write_map(sim_em_rew,"map_posterior_"+str(rand_ID)+".mrc",map_param)
else: write_map(sim_em_rew,"map_posterior_"+str(rand_ID)+".mrc",map_param)


# Saving prior map
sim_em_rew = np.dot(sim_em_data.T,w0).T
if exists("map_prior_"+str(rand_ID)+".mrc"):
    os.system("rm map_prior_"+str(rand_ID)+".mrc")
    write_map(sim_em_rew,"map_prior_"+str(rand_ID)+".mrc",map_param)
else: write_map(sim_em_rew,"map_prior_"+str(rand_ID)+".mrc",map_param)



plik = open("statistics.dat", "a")

plik.write("Theta value chosen by Kneedle algorithm: "+str(theta_new)+"\n")
plik.write("Reduced Chisqrt: " + str(chisqrt_d[theta_new]/N_voxels)+"\n")
plik.write("Neff: " + str(np.exp(s_dict[theta_new]))+"\n")
plik.write("Posteriori Correlation: " + str(cc)+"\n")
plik.write("Priori Correlation: " + str(cc_prior)+"\n")
plik.write("Single Best structure Correlation: " + str(cc_single)+"\n")
plik.write("Single Best structure: " + str(PDBs[best_single])+"\n")
np.savetxt(plik,best,newline=" ",fmt="%i")
plik.write("\n")
plik.write("How many true models are in top10: "+str(best_ratio*100)+"%\n" )



"""
"" ITERATIVE search of the smallest set of structures
"" theta_new - selected theta value
"" w_opt_d[theta_new] - weights corresponding to the selected theta
"""

# testing for which structure we do see weights to be lower than initial one
# We will remove these structures and focus on optimizing weights for the remaining ones

np.where(w_opt_d[theta_new]>w_init)
selected_frames = np.where(w_opt_d[theta_new]>w_init)[0]
selection = selected_frames
new_chisqrt = chisqrt_d[theta_new]
old_chisqrt = chisqrt_d[theta_new]

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

"""
"" WHILE LOOP
"""
iteration = 1
while (new_chisqrt<=old_chisqrt):
    N_models_sel = len(selected_frames)
    w0,w_init,g0,g_init,sf_init = bioen_init_uniform(N_models_sel)
    # Selecting data
    sim_em_v_data_selected = sim_em_v_data[selected_frames]
    # Getting initial optimal scalling factor
    sf_start = leastsq(coeff_fit, sf_init, args=(w0,std,sim_em_v_data_selected,exp_em_mask))[0]
    # Running BioEN iterations through hyperparameter Theta:
    # w_opt_array, S_array, chisqrt_array = bioen(sim_em_v_data,exp_em_mask,std,thetas, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)
    # Running BioEN in a loop so we can apply knee dectection algorithm later
    w_opt_d_sel = dict()
    sf_opt_d_sel = dict()
    s_dict_sel = dict()
    chisqrt_d_sel = dict()
    for th in thetas:
        w_temp, sf_temp = bioen_single(sim_em_v_data_selected,exp_em_mask,std,th, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)
        w_opt_d_sel[th] = w_temp
        sf_opt_d_sel[th] = sf_temp
        s_dict_sel[th] = get_entropy(w0,w_temp)
        chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_em_v_data_selected*sf_temp,exp_em_mask)
    # Knee locator used to find sensible theta value
    theta_index = knee_loc(list(s_dict_sel.values()),list(chisqrt_d_sel.values()))
    theta_knee = thetas[theta_index]
    theta_index_sort = theta_index
    for i in range(0,10):
        thetas_old = np.sort(list(w_opt_d_sel.keys()))[::-1]
        theta_up = (thetas_old[theta_index_sort - 1] + thetas_old[theta_index_sort])/2.
        theta_down = (thetas_old[theta_index_sort + 1] + thetas_old[theta_index_sort])/2.
        for th in theta_up,theta_down:
            w_temp, sf_temp = bioen_single(sim_em_v_data_selected,exp_em_mask,std,th, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)
            w_opt_d_sel[th] = w_temp
            sf_opt_d_sel[th] = sf_temp
            s_dict_sel[th] = get_entropy(w0,w_temp)
            chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_em_v_data_selected*sf_temp,exp_em_mask)
            # Knee locator
        theta_index_new_sel = knee_loc(list(s_dict_sel.values()),list(chisqrt_d_sel.values()))
        theta_new_sel = list(w_opt_d_sel.keys())[theta_index_new_sel]
        thetas_sorted_sel = np.sort(list(w_opt_d_sel.keys()))[::-1]
        theta_index_sort_sel = np.where(theta_new_sel == thetas_sorted_sel)[0][0]
    # L-CURVE
    name = "lcurve_"+str(float(rand_ID))+"_iter_"+str(iteration)+".svg"
    plot_lcurve(s_dict_sel,chisqrt_d_sel,theta_new_sel,N_voxels,name)
    # WEIGHTS
    name = "weights_"+str(float(rand_ID))+"_iter_"+str(iteration)+".svg"
    plot_weights(w_opt_d_sel,selection,theta_new_sel,N_models,name)
    # Cross-correlation
    cc_sel,cc_prior_sel,cc_single_sel = map_correlations_mask(sim_em_data[selection],em_map_norm,mask_comb,w_opt_d_sel,w0,theta_new_sel)
    # Saving posterior map
    sim_em_rew_sel = np.dot(sim_em_data[selection].T,w_opt_d_sel[theta_new_sel]).T
    if exists("map_posterior_"+str(rand_ID)+"_iter_"+str(iteration)+".mrc"):
        os.system("rm map_posterior_"+str(rand_ID)+"_iter_"+str(iteration)+".mrc")
        write_map(sim_em_rew_sel,"map_posterior_"+str(rand_ID)+"_iter_"+str(iteration)+".mrc",map_param)
    else: write_map(sim_em_rew_sel,"map_posterior_"+str(rand_ID)+"_iter_"+str(iteration)+".mrc",map_param)
    # writing statistics
    plik_sel = open("statistics_"+str(rand_ID)+"_iter.dat","a")
    plik_weights = open("weights_"+str(rand_ID)+"_iter_"+str(iteration)+".dat","w")
    plik_sel.write("ITERATION "+str(iteration)+"\n")
    plik_sel.write("Theta value chosen by Kneedle algorithm: "+str(theta_new_sel)+"\n")
    plik_sel.write("Reduced Chisqrt: " + str(chisqrt_d_sel[theta_new_sel]/N_voxels)+"\n")
    plik_sel.write("Neff: " + str(np.exp(s_dict_sel[theta_new_sel]))+"\n")
    plik_sel.write("Number of structures: "+str(len(selection))+"\n")
    w_all = np.zeros(N_models)
    w_all[selection] = w_opt_d_sel[theta_new_sel]
    np.savetxt(plik_weights,w_all)
    plik_weights.close()
    plik_sel.write("Posteriori Correlation: "+str(cc_sel)+"\n")
    plik_sel.write("Priori Correlation: "+str(cc_prior)+"\n")
    selection = np.where(w_opt_d_sel[theta_new_sel]>w_init)[0]
    plik_sel.write("\n")
    temp_sel = selected_frames[selection]
    selected_frames = temp_sel
    selection = selected_frames
    old_chisqrt = new_chisqrt
    new_chisqrt = chisqrt_d_sel[theta_new_sel]
    iteration+=1

plik_sel.close()
plik.close()
