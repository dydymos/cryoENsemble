import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from plot import *
from validation import *
from os.path import exists
from PCA_rmsf import *
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, Bounds


import warnings
warnings.filterwarnings('ignore')

"""
"" PATHWAYS
"""

# pathway to models from MD simulation
path = "minim"

random_list = np.loadtxt("random_list.dat")

"""
"" Input
"""

def checker(a):
    num = float(a)
    if num != 10 and num != 6 and num != 3:
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
parser.add_argument('random', type = int, help = 'Id list of randomly selected 10 structures from the MD ensemble')
parser.add_argument('noise', type = float, help = 'Noise level, which is defined as normal distribution centered around 0 and with std equal to X of the maximum density in the Reference map')
parser.add_argument('mask_type', type = checker_mask, help = 'Type of mask: exp or sim')
parser.add_argument('--missing', action = 'store_true', help = 'Add it when you want to discard structures used to generate reference map from the reweighted ensemble')
args = parser.parse_args()


"""
"" MAP details based on 6A and 10A maps generated in ChimeraX
"""
map_param = dict()
if (args.resM == 3):
    map_param['nx'] = 73
    map_param['ny'] = 74
    map_param['nz'] = 122
    map_param['vox']= 1.0
    map_param['em_origin'] = np.array([219.323, 193.668, 88.312])
elif (args.resM == 6):
    map_param['nx'] = 46
    map_param['ny'] = 46
    map_param['nz'] = 70
    map_param['vox'] = 2.0
    map_param['em_origin'] = np.array([210.323, 184.668, 79.312])
elif (args.resM == 10):
    map_param['nx'] = 35
    map_param['ny'] = 35
    map_param['nz'] = 50
    map_param['vox']= 3.3333335
    map_param['em_origin'] = np.array([198.323, 172.668, 67.312])


# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

"""
"" Creating average EM map from 10 randomly selcected models, which ID is stored in random_list file
"""

# reference map resolution
sigma = args.resM*0.225

# average map map generate from randomly chosen 10 maps
rand_ID = args.random
random_pdbs = [path+"/minim_"+str(int(x))+".pdb" for x in random_list[rand_ID-1]]

# Number of models
M = 10

exp_weights=np.ones(M)/M
exp_map = pdb2map_avg_chimeraX(exp_weights,args.resM,random_pdbs,map_param,cryoem_param)

# map with noise plus map threshold which equals 3 x noise_std
noise = args.noise
exp_map_noise = add_noise(exp_map,noise)

# noise based threshold - 3xstd of the noise level
noise_thr = np.max(exp_map)*noise*3

# removing negative values of map (if em_map_norm < 0 -> 0)
exp_map_threshold = np.clip(exp_map_noise,0,np.max(exp_map_noise))

# Scale distribution to range between 0 and 1
exp_map_norm = (exp_map_threshold - np.min(exp_map_threshold))/(np.max(exp_map_threshold) - np.min(exp_map_threshold))
noise_thr_norm = (noise_thr - np.min(exp_map_threshold))/(np.max(exp_map_threshold) - np.min(exp_map_threshold))

# Mask of the EM map (where the density is > threshold)
mask_exp = np.where(exp_map_norm > noise_thr_norm)

# Saving normalized map with noise and range from 0 to 1
write_map(exp_map_norm,"map_norm_"+str(rand_ID)+".mrc",map_param)


"""
"" STRUCTURAL ENSEMBLE
"""

# OK Now we read 100 models previously minimized
# Number of structures/models
N_models = 100
if args.missing: N_models = 90

PDBs = []

for i in range(1,101):
   PDBs.append(path+"/minim_"+str(i)+'.pdb')

# Generating array of EM maps based on structures
print("Generating an array of EM maps based on the structures from MD simulation")

pdb_list = np.array([int(x)-1 for x in random_list[rand_ID - 1]])

if args.missing:
    PDBs_missing = [x for i, x in enumerate(PDBs) if i not in pdb_list]
    sim_map_array = np.array(pdb2map_array(PDBs_missing,map_param,cryoem_param))
    pdb_list = [x for i, x in enumerate(np.arange(1,100)) if i not in pdb_list]
else:
    sim_map_array = np.array(pdb2map_array(PDBs,map_param,cryoem_param))


# Normalization
sim_map_array /= np.max(sim_map_array, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
sim_map = np.sum(sim_map_array,0)

# Saving normalized simulated map
write_map(sim_map,"map_sim_"+str(rand_ID)+".mrc",map_param)


"""
"" MASKING
"""

# Masking can take place in two ways:
# EXP - when we use mask generated from experimental map
# SIM - when we also include voxels from the map generated for each fitted structure
# for that we use threshold eaulat to 3x the simulated map std

mask_arg = args.mask_type
if (mask_arg == "exp"):
    mask = mask_exp
    # Masked experimental data
    exp_map_mask = exp_map_norm[mask]
    # Number of non-zero voxels
    N_voxels=np.shape(mask)[1]
    # New simulated map with only voxels corresponding to the experimental signal
    sim_map_array_mask=np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_map_array_mask[i]=sim_map_array[i][mask]

# Mask for CCC calculations
# Masked experimental data
mask_exp_array=np.array(mask_exp)
# generate mask over simulated density, using threshold equal to 3*std
mask_sim = mask_sim_gen(sim_map_array,N_models)
mask_comb = combine_masks(mask_exp_array,mask_sim)

if (mask_arg == "sim"):
    mask = mask_comb
    # Number of non-zero voxels
    N_voxels=np.shape(mask)[1]
    # Masked experimental data
    exp_map_mask = exp_map_norm[mask]
    # New simulated map with only voxels corresponding to the exp+sim
    sim_map_array_mask=np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_map_array_mask[i]=sim_map_array[i][mask]


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
sf_start = leastsq(coeff_fit, sf_init, args=(w0,std,sim_map_array_mask,exp_map_mask))[0]

# Initial sigma
resolution = 6
sigma_min = 1 * 0.225
sigma_max = resolution * 0.225
bounds = Bounds([sigma_min], [sigma_max])
# Getting initial optimal sigma for simulated maps
sigma_start = minimize(sigma_fit, [sigma_min], args=(w0,std,sim_map_array*sf_start,exp_map_mask,mask),method="L-BFGS-B",bounds=bounds).x[0]

"""
"" Parameters for optimization algorithm
"""

# For now we can only use BFGS algorithm as is coded in SCIPY

epsilon = 1e-08
pgtol = 1e-05
maxiter = 5000

# Number of iterations
n_iter = 5

# Theta
print("Running BioEN through preselected values of theta")
thetas = [10000000,1000000,100000,10000,1000,100,10,1,0]


# Running BioEN iterations through hyperparameter Theta:
# w_opt_array, S_array, chisqrt_array = bioen(sim_em_v_data,exp_em_mask,std,thetas, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)

# Running BioEN in a loop so we can apply knee dectection algorithm later

w_opt_d = dict()
sf_opt_d = dict()
s_dict = dict()
chisqrt_d = dict()
for th in thetas:
    w_temp = bioen_sigma_one(sim_map_array,exp_map_mask,mask,std,th, g0, g_init, sf_start, sigma_start, n_iter, epsilon, pgtol, maxiter)
    w_opt_d[th] = w_temp
    s_dict[th] = get_entropy(w0,w_temp)
    sim_map_g = gaussian_filter(sim_map_array*sf_start,sigma_start)
    sim_map_v = np.array([map[mask] for map in sim_map_g])
    chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_map_v,exp_map_mask)

# Knee locator used to find sensible theta value
theta_index = knee_loc(list(s_dict.values()),list(chisqrt_d.values()))
theta_knee = thetas[theta_index]
theta_index_sort = theta_index

print("\nRunning BioEN iterations to narrow down the theta value")
for i in range(0,10):
    print("Round "+str(i+1)+" out of 10:")
    thetas_old = np.sort(list(w_opt_d.keys()))[::-1]
    theta_up = (thetas_old[theta_index_sort - 1] + thetas_old[theta_index_sort])/2.
    theta_down = (thetas_old[theta_index_sort + 1] + thetas_old[theta_index_sort])/2.
    for th in theta_up,theta_down:
        w_temp = bioen_sigma_one(sim_map_array,exp_map_mask,mask,std,th, g0, g_init, sf_start, sigma_start, n_iter, epsilon, pgtol, maxiter)
        w_opt_d[th] = w_temp
        s_dict[th] = get_entropy(w0,w_temp)
        sim_map_g = gaussian_filter(sim_map_array*sf_start,sigma_start)
        sim_map_v = np.array([map[mask] for map in sim_map_g])
        chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_map_v,exp_map_mask)
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

print("Chosen theta: "+str(theta_new))


# Neff
name = "Neff_"+str(float(rand_ID))+".svg"
plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,name)

print("Chosen Neff: "+str(np.exp(s_dict[theta_new])))

# WEIGHTS
name = "weights_"+str(float(rand_ID))+".svg"

max_weights = max(value for sublist in w_opt_d.values() for value in sublist)

if args.missing:
    plot_weights_missing(w_opt_d,np.arange(0,N_models),theta_new,N_models,name)
else:
    plot_weights(w_opt_d,np.arange(0,N_models),theta_new,N_models,pdb_list,max_weights,name)


plik_weights = open("weights_"+str(float(rand_ID))+".dat","w")
w_all = w_opt_d[theta_new]
np.savetxt(plik_weights,w_all)
plik_weights.close()


"""
"" CORRELATION with exp map
"""

mapa = gaussian_filter(sim_map_array*sf_start,sigma_start)
cc,cc_prior,cc_single = map_correlations_mask(mapa,exp_map_norm,mask_comb,w_opt_d,w0,theta_new)


print("Correlation is: "+str(cc))

# TOP 10
best = np.argsort(w_opt_d[theta_new])[::-1][:10]
best_single = np.argsort(w_opt_d[theta_new])[::-1][0]

if args.missing:
    best_m = np.array([pdb_list for i in best])

best_ratio = 0
for i in best:
    if i in pdb_list: best_ratio+=1

best_ratio/=10.0

"""
"" FOURIER SHELL CORRELATION
"""

# prior map
prior_map = np.dot(mapa.T,w0).T

# posterior map
posterior_map = np.dot(mapa.T,w_opt_d[theta_new]).T

# FSC between reference map and prior map
fsc_prior = FSC(exp_map_norm,prior_map,map_param)
fsc_poster= FSC(exp_map_norm,posterior_map,map_param)


if (((fsc_prior[:,1]<0.5).any()) & (fsc_poster[:,1]<0.5).any()):
    fsc_0_05 = fsc_prior[:,0][fsc_prior[:,1]<0.5][0]
    fsc_1_05 = fsc_poster[:,0][fsc_poster[:,1]<0.5][0]
    fsc_index = 0.5
elif (((fsc_prior[:,1]<0.6).any()) & (fsc_poster[:,1]<0.6).any()):
    fsc_0_05 = fsc_prior[:,0][fsc_prior[:,1]<0.6][0]
    fsc_1_05 = fsc_poster[:,0][fsc_poster[:,1]<0.6][0]
    fsc_index = 0.6
elif (((fsc_prior[:,1]<0.7).any()) & (fsc_poster[:,1]<0.7).any()):
    fsc_0_05 = fsc_prior[:,0][fsc_prior[:,1]<0.7][0]
    fsc_1_05 = fsc_poster[:,0][fsc_poster[:,1]<0.7][0]
    fsc_index = 0.7
elif (((fsc_prior[:,1]<0.8).any()) & (fsc_poster[:,1]<0.8).any()):
    fsc_0_05 = fsc_prior[:,0][fsc_prior[:,1]<0.8][0]
    fsc_1_05 = fsc_poster[:,0][fsc_poster[:,1]<0.8][0]
    fsc_index = 0.8

# Plot FSC
name = "FSC_"+str(float(rand_ID))+".svg"
plot_fsc(fsc_prior,fsc_poster,name)


"""
"" TEMPy
"""
# SMOC calculations
if args.missing:
    smoc_prior,smoc_poster = SMOC_missing("map_norm_"+str(rand_ID)+".mrc",args.resM,w0,w_opt_d[theta_new],pdb_list_remain)
else:
    smoc_prior,smoc_poster = SMOC("map_norm_"+str(rand_ID)+".mrc",args.resM,w0,w_opt_d[theta_new])


# Plotting SMOC
#name = "SMOC_"+str(float(rand_ID))+".svg"
#plot_smoc(smoc_prior,smoc_poster,name)

"""
"" Jensen-Shanon divergence
"""
# getting RMSD matrix
#rmsd_m = rmsd_matrix(PDBs)
#np.savetxt('matrix_rmsd.csv', rmsd_m, delimiter=',')
rmsd_m = np.loadtxt('matrix_rmsd.csv', delimiter=',')
# clustering based on RMSD matrix and cutoff = 8A
clusters = clustering(rmsd_m,8)

# Jensen-Shanon div
if args.missing:
    js_div_prior,js_div_poster = J_S_div_missing(clusters,pdb_list,w_opt_d[theta_new],pdb_list_remain)
else:
    js_div_prior,js_div_poster = J_S_div(clusters,pdb_list,w_opt_d[theta_new])


"""
"" WRITING POSTERIOR AND PRIOR MAP
"""
# Saving posterior map
write_map(posterior_map,"map_posterior_"+str(rand_ID)+".mrc",map_param)

# Saving prior map
write_map(prior_map,"map_prior_"+str(rand_ID)+".mrc",map_param)

"""
"" RMSF calculations
"""
# Get RMSF
rmsf_target, rmsf_prior, rmsf_post = get_rmsf(PDBs,random_pdbs,w_opt_d[theta_new])

# plot RMSF
name = "RMSF_"+str(float(rand_ID))+".svg"
plot_rmsf(rmsf_target, rmsf_prior,rmsf_post,name)

"""
"" PCA
"""
# Calculate PCA
eigenvectors_target, eigenvectors_prior, eigenvectors_post = do_PCA(PDBs,random_pdbs,w_opt_d[theta_new])

# Compute dot products
dot_products_prior = np.dot(eigenvectors_target, eigenvectors_prior.T)
dot_products_post = np.dot(eigenvectors_target, eigenvectors_post.T)

name = "PCA_"+str(float(rand_ID))+".svg"
plot_pca(dot_products_prior,dot_products_post,name)

# Save the eigenvector to a file
np.savetxt('eigenvector_target_'+str(float(rand_ID))+'.txt', eigenvectors_target[0])
np.savetxt('eigenvector_prior_'+str(float(rand_ID))+'.txt', eigenvectors_prior[0])
np.savetxt('eigenvector_post_'+str(float(rand_ID))+'.txt', eigenvectors_post[0])

plik = open("statistics.dat", "a")

plik.write("Theta value chosen by Kneedle algorithm: "+str(theta_new)+"\n")
plik.write("Reduced Chisqrt: " + str(chisqrt_d[theta_new]/N_voxels)+"\n")
plik.write("Neff: " + str(np.exp(s_dict[theta_new]))+"\n")
plik.write("Priori Correlation [CC_sim+exp]: "+str(cc_prior)+"\n")
plik.write("Posteriori Correlation [CC_sim+exp]: "+str(cc)+"\n")
plik.write("Single Best structure Correlation [CC_sim+exp]: "+str(cc_single)+"\n")
plik.write("Priori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_0_05)+"\n")
plik.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_1_05)+"\n")
plik.write("Avg Priori SMOC: "+str(np.mean(smoc_prior))+"\n")
plik.write("Avg Posteriori SMOC: "+str(np.mean(smoc_poster))+"\n")
plik.write("Jensen-Shanon div [prior,poster]: "+str(js_div_prior)+"\t"+str(js_div_poster)+"\n")
plik.write("Weights that are assigned to the correct structures: "+str(np.round(np.sum(w_opt_d[theta_new][pdb_list]),2))+"\n")

if args.missing:
    plik.write("Single Best structure: " + str(PDBs_missing[best_single])+"\n")
    plik.write("Top structures: ")
    np.savetxt(plik,best_m,newline=" ",fmt="%s")
else:
    plik.write("Single Best structure: " + str(PDBs[best_single])+"\n")
    plik.write("Top structures: ")
    np.savetxt(plik,np.array(PDBs)[best],newline=" ",fmt="%s")
    plik.write("\n")
    plik.write("Reference structures: ")
    np.savetxt(plik,np.array(PDBs)[pdb_list],newline=" ",fmt="%s")
    plik.write("\n")
    plik.write("How many true models are in top10: "+str(best_ratio*100)+"%\n" )

plik.write("RMSF differences [target vs prior: "+str(np.round(np.sqrt(np.sum((rmsf_target-rmsf_prior)**2)/len(rmsf_target)),2))+"\n")
plik.write("RMSF differences [target vs post: "+str(np.round(np.sqrt(np.sum((rmsf_target-rmsf_post)**2)/len(rmsf_target)),2))+"\n")
plik.write("PCA analysis [dot product target vs prior]: " + str(np.round(np.abs(dot_products_prior[0,0]),2)) + " " + str(np.round(np.abs(dot_products_prior[1,1]),2))+ " " + str(np.round(np.abs(dot_products_prior[2,2]),2))+"\n")
plik.write("PCA analysis [dot product target vs post]: " + str(np.round(np.abs(dot_products_post[0,0]),2)) + " " + str(np.round(np.abs(dot_products_post[1,1]),2))+ " " + str(np.round(np.abs(dot_products_post[2,2]),2))+"\n")
plik.write("\n")


"""
"" ITERATIVE search of the smallest set of structures
"" theta_new - selected theta value
"" w_opt_d[theta_new] - weights corresponding to the selected theta
"""

# testing for which structure we do see weights to be lower than initial one
# We will remove these structures and focus on optimizing weights for the remaining ones


Neff = np.exp(s_dict[theta_new])

selection_range = int(len(w_opt_d[theta_new])*Neff)

selected_frames = np.argsort(w_opt_d[theta_new])[::-1][:selection_range]
selection = selected_frames
new_chisqrt = chisqrt_d[theta_new]/N_voxels
old_chisqrt = chisqrt_d[theta_new]/N_voxels
#new_cc = cc
#old_cc = cc

"""
"" Parameters for optimization algorithm
"""
# For now we can only use BFGS algorithm as is coded in SCIPY
epsilon = 1e-08
pgtol = 1e-05
maxiter = 5000
# Number of iterations
n_iter = 5
# Theta
thetas = [10000000,1000000,100000,10000,1000,100,10,1,0]


"""
"" WHILE LOOP
"""
print("\nRunning BioEN iterations to find the smalles set of structures\n")
iteration = 1
while (new_chisqrt <= old_chisqrt):
    N_models_sel = len(selected_frames)
    w0,w_init,g0,g_init,sf_init = bioen_init_uniform(N_models_sel)
    # Selecting data
    sim_map_array_selected = sim_map_array[selected_frames]
    sim_map_array_mask_selected = sim_map_array_mask[selected_frames]
    # Getting initial optimal scalling factor
    sf_start = leastsq(coeff_fit, sf_init, args=(w0,std,sim_map_array_mask_selected,exp_map_mask))[0]
    # Getting initial optimal sigma for simulated maps
    sigma_start = minimize(sigma_fit, [sigma_min], args=(w0,std,sim_map_array_selected*sf_start,exp_map_mask,mask),method="L-BFGS-B",bounds=bounds).x[0]
    # Running BioEN iterations through hyperparameter Theta:
    # Running BioEN in a loop so we can apply knee dectection algorithm later
    w_opt_d_sel = dict()
    s_dict_sel = dict()
    chisqrt_d_sel = dict()
    for th in thetas:
        w_temp = bioen_sigma_one(sim_map_array_selected,exp_map_mask,mask,std,th, g0, g_init, sf_start, sigma_start, n_iter, epsilon, pgtol, maxiter)
        w_opt_d_sel[th] = w_temp
        s_dict_sel[th] = get_entropy(w0,w_temp)
        sim_map_g = gaussian_filter(sim_map_array_selected*sf_start,sigma_start)
        sim_map_v = np.array([map[mask] for map in sim_map_g])
        chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_map_v,exp_map_mask)
    # Knee locator used to find sensible theta value
    theta_index = knee_loc(list(s_dict_sel.values()),list(chisqrt_d_sel.values()))
    theta_knee = thetas[theta_index]
    theta_index_sort_sel = theta_index
    print("\nRunning BioEN iterations to narrow down the theta value")
    for i in range(0,10):
        print("Round "+str(i+1)+" out of 10:")
        thetas_old = np.sort(list(w_opt_d_sel.keys()))[::-1]
        theta_up = (thetas_old[theta_index_sort_sel - 1] + thetas_old[theta_index_sort_sel])/2.
        theta_down = (thetas_old[theta_index_sort_sel + 1] + thetas_old[theta_index_sort_sel])/2.
        for th in theta_up,theta_down:
            w_temp = bioen_sigma_one(sim_map_array_selected,exp_map_mask,mask,std,th, g0, g_init, sf_start, sigma_start, n_iter, epsilon, pgtol, maxiter)
            w_opt_d_sel[th] = w_temp
            s_dict_sel[th] = get_entropy(w0,w_temp)
            sim_map_g = gaussian_filter(sim_map_array_selected*sf_start,sigma_start)
            sim_map_v = np.array([map[mask] for map in sim_map_g])
            chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_map_v,exp_map_mask)
            # Knee locator
        theta_index_new_sel = knee_loc(list(s_dict_sel.values()),list(chisqrt_d_sel.values()))
        theta_new_sel = list(w_opt_d_sel.keys())[theta_index_new_sel]
        thetas_sorted_sel = np.sort(list(w_opt_d_sel.keys()))[::-1]
        theta_index_sort_sel = np.where(theta_new_sel == thetas_sorted_sel)[0][0]
    # Calculating New chiSqrTerm
    old_chisqrt = np.copy(new_chisqrt)
    new_chisqrt = chisqrt_d_sel[theta_new_sel]/N_voxels
    #old_cc = np.copy(new_cc)
    #mapa = gaussian_filter(sim_map_array_selected*sf_start,sigma_start)
    #cc_sel,cc_prior_sel,cc_single_sel = map_correlations_mask(mapa,exp_map_norm,mask_comb,w_opt_d_sel,w0,theta_new_sel)
    #new_cc = np.copy(cc_sel)
    if (new_chisqrt > old_chisqrt): break
    else:
        # L-CURVE
        name = "iter/lcurve_"+str(rand_ID)+"_iter.svg"
        plot_lcurve(s_dict_sel,chisqrt_d_sel,theta_new_sel,N_voxels,name)
        # WEIGHTS
        name = "iter/weights_"+str(rand_ID)+"_iter.svg"
        plot_weights(w_opt_d_sel,selected_frames,theta_new_sel,N_models,pdb_list,max_weights,name)
        plik_weights = open("iter/weights_"+str(rand_ID)+"_iter.dat","w")
        w_all = np.zeros(N_models)
        w_all[selected_frames] = w_opt_d_sel[theta_new_sel]
        np.savetxt(plik_weights,w_all)
        plik_weights.close()
        # Cross-correlation
        mapa = gaussian_filter(sim_map_array_selected*sf_start,sigma_start)
        cc_sel,cc_prior_sel,cc_single_sel = map_correlations_mask(mapa,exp_map_norm,mask_comb,w_opt_d_sel,w0,theta_new_sel)
        # posterior map
        posterior_map_sel = np.dot(mapa.T,w_opt_d_sel[theta_new_sel]).T
        # FSC between reference map and prior map
        fsc_poster_sel= FSC(exp_map_norm,posterior_map_sel,map_param)
        if (((fsc_prior[:,1]<0.5).any()) & (fsc_poster_sel[:,1]<0.5).any()):
            fsc_0_05_sel = fsc_prior[:,0][fsc_prior[:,1]<0.5][0]
            fsc_1_05_sel = fsc_poster_sel[:,0][fsc_poster_sel[:,1]<0.5][0]
            fsc_index_sel = 0.5
        elif (((fsc_prior[:,1]<0.6).any()) & (fsc_poster_sel[:,1]<0.6).any()):
            fsc_0_05_sel = fsc_prior[:,0][fsc_prior[:,1]<0.6][0]
            fsc_1_05_sel = fsc_poster_sel[:,0][fsc_poster_sel[:,1]<0.6][0]
            fsc_index_sel = 0.6
        elif (((fsc_prior[:,1]<0.7).any()) & (fsc_poster_sel[:,1]<0.7).any()):
            fsc_0_05_sel = fsc_prior[:,0][fsc_prior[:,1]<0.7][0]
            fsc_1_05_sel = fsc_poster_sel[:,0][fsc_poster_sel[:,1]<0.7][0]
            fsc_index_sel = 0.7
        elif (((fsc_prior[:,1]<0.8).any()) & (fsc_poster_sel[:,1]<0.8).any()):
            fsc_0_05_sel = fsc_prior[:,0][fsc_prior[:,1]<0.8][0]
            fsc_1_05_sel = fsc_poster_sel[:,0][fsc_poster_sel[:,1]<0.8][0]
            fsc_index_sel = 0.8
        # Plot FSC
        name = "iter/FSC_"+str(float(rand_ID))+"_iter.svg"
        plot_fsc_sel(fsc_prior,fsc_poster,fsc_poster_sel,name)
        # SMOC calculations
        smoc_poster_sel = SMOC_iter("map_norm_"+str(rand_ID)+".mrc",args.resM,w0,w_opt_d_sel[theta_new_sel],selected_frames)
        # Plotting SMOC
        name = "iter/SMOC_"+str(float(rand_ID))+"_iter.svg"
        plot_smoc_sel(smoc_prior,smoc_poster,smoc_poster_sel,name)
        # posterior map
        posterior_map_sel = np.dot(sim_map_array_selected.T,w_opt_d_sel[theta_new_sel]).T
        write_map(posterior_map_sel,"iter/map_posterior_"+str(rand_ID)+"_iter.mrc",map_param)
        # JS divergence
        js_div_poster_sel = J_S_div_inter(clusters,pdb_list,w_opt_d_sel[theta_new_sel],selected_frames)
        # Neff
        Neff_sel = np.exp(s_dict_sel[theta_new_sel])
        # Get RMSF
        selected_pdbs = np.array(PDBs)[selected_frames]
        rmsf_post_sel = get_rmsf_sel(PDBs,selected_pdbs,w_opt_d_sel[theta_new_sel])
        # plot RMSF
        name = "iter/RMSF_"+str(float(rand_ID))+"_iter.svg"
        plot_rmsf_sel(rmsf_target,rmsf_prior,rmsf_post,rmsf_post_sel,name)
        # Calculate PCA
        eigenvectors_post_sel = do_PCA_sel(PDBs,selected_pdbs,w_opt_d_sel[theta_new_sel])
        # Compute dot products
        dot_products_post_sel = np.dot(eigenvectors_target, eigenvectors_post_sel.T)
        name = "iter/PCA_"+str(float(rand_ID))+"_iter.svg"
        plot_pca_sel(dot_products_prior,dot_products_post,dot_products_post_sel,name)
        # Save the eigenvector to a file
        np.savetxt('iter/eigenvector_post_'+str(float(rand_ID))+'_iter.txt', eigenvectors_post_sel[0])
        # writing statistics
        plik_sel = open("iter/statistics_"+str(rand_ID)+"_iter.dat","a")
        plik_sel.write("ITERATION "+str(iteration)+"\n")
        plik_sel.write("Theta value chosen by Kneedle algorithm: "+str(theta_new_sel)+"\n")
        plik_sel.write("Reduced Chisqrt value: "+str(chisqrt_d_sel[theta_new_sel]/N_voxels)+"\n")
        plik_sel.write("Neff: " + str(Neff_sel)+"\n")
        plik_sel.write("Number of structures: "+str(len(selection))+"\n")
        plik_sel.write("Posteriori Correlation [CC_sim+exp]: "+str(cc_sel)+"\n")
        plik_sel.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index_sel)+" [1/A]: "+str(fsc_1_05_sel)+"\n")
        plik_sel.write("Avg Posteriori SMOC: "+str(np.mean(smoc_poster_sel))+"\n")
        plik_sel.write("Jensen-Shanon div [poster]: "+str(js_div_poster_sel)+"\n")
        pdb_selection = []
        for item in range(0,len(selected_frames)):
            if selected_frames[item] in pdb_list:
                pdb_selection.append(item)
        plik_sel.write("Weights that are assigned to the correct structures: "+str(np.round(np.sum(w_opt_d_sel[theta_new_sel][pdb_selection]),2))+"\n")
        plik_sel.write("RMSF differences [target vs post_iter: "+str(np.round(np.sqrt(np.sum((rmsf_target-rmsf_post_sel)**2)/len(rmsf_target)),2))+"\n")
        plik_sel.write("PCA analysis [dot product target vs post_iter]: " + str(np.round(np.abs(dot_products_post_sel[0,0]),2)) + " " + str(np.round(np.abs(dot_products_post_sel[1,1]),2))+ " " + str(np.round(np.abs(dot_products_post_sel[2,2]),2))+"\n")
        plik_sel.write("\n")
        plik_sel.close()
        # New selection
        selection_range_sel = int(len(w_opt_d_sel[theta_new_sel])*Neff_sel)
        selection = np.argsort(w_opt_d_sel[theta_new_sel])[::-1][:selection_range_sel]
        temp_sel = selected_frames[selection]
        selected_frames = temp_sel
        iteration += 1
