import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from reference_map import *
from PCA_rmsf import *
from plot import *
from validation import *
from os.path import exists
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
path = "/"
# Input data/files
mapa = path+"reference_map.mrc"
threshold = # map threshold
ref_pdb = "ref_pdb.pdb"
ref_traj = "trajectory.xtc"
resolution = # Map resolution
mask_arg = 'sim' # Mask type: 'exp' - only based on experimental cryo-EM density, 'sim' - both experimental cryo-EM density + based on simulated ensemble

"""
"" Reference cro-EM map
"""
# Reference map
exp_map = mrcfile.open(mapa, mode='r')
exp_map_data = exp_map.data.copy()
map_param = reference_map(mapa)

# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

# removing negative values of map (if em_map_norm < 0 -> 0)
exp_map_threshold = np.clip(exp_map_data,0,np.max(exp_map_data))

# Scale distribution to range between 0 and 1
exp_map_norm = (exp_map_threshold - np.min(exp_map_threshold))/(np.max(exp_map_threshold) - np.min(exp_map_threshold))
noise_thr_norm = (threshold - np.min(exp_map_threshold))/(np.max(exp_map_threshold) - np.min(exp_map_threshold))

# Mask of the EM map (where the density is > threshold)
mask_exp = np.where(exp_map_norm > noise_thr_norm)

# Saving normalized map with noise and range from 0 to 1
write_map(exp_map_norm,"map_norm.mrc",map_param)


"""
"" STRUCTURAL ENSEMBLE
"""

# MD structural ensemble
u = MDAnalysis.Universe(ref_pdb,ref_traj)

# Number of structures/models
N_models = u.trajectory.n_frames
# Generating array of EM maps based on structures
print("Generating an array of EM maps based on the structures from MD simulation")

sim_map_array = np.array(traj2map_array(u,map_param,cryoem_param))

# Normalization
sim_map_array /= np.max(sim_map_array, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)

print("generating final sim map")
sim_map = np.sum(sim_map_array,0)
print("Writing a map")
# Saving normalized map with noise and without negative density
write_map(sim_map,"map_sim.mrc",map_param)



"""
"" MASKING
"""

print("MASKING\n")
if (mask_arg == "exp"):
    mask = mask_exp
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
mask_sim = mask_sim_gen(sim_map_array,N_models)
mask_comb = combine_masks(mask_exp_array,mask_sim)



if (mask_arg == "sim"):
    mask = mask_comb
    # Number of non-zero voxels
    N_voxels=np.shape(mask_comb)[1]
    # Masked experimental data
    exp_map_mask = exp_map_norm[mask_comb]
    # New simulated map with only voxels corresponding to the exp+sim
    sim_map_array_mask = np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_map_array_mask[i]=sim_map_array[i][mask_comb]



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
sigma_min = 1 * 0.225
sigma_max = resolution * 0.225
bounds = Bounds([sigma_min], [sigma_max])
# Getting initial optimal sigma for simulated maps
sigma_start = minimize(sigma_fit, [sigma_min], args=(w0,std,sim_map_array*sf_start,exp_map_mask,mask),method="L-BFGS-B",bounds=bounds).x[0]

# Generating filtered and scalled map
sim_map_g = gaussian_filter(sim_map_array*sf_start,sigma_start)
sim_map_g_mask = np.array([map[mask] for map in sim_map_g])

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
thetas = [10000000,1000000,100000,10000,1000,100,10,1,0]


# Running BioEN iterations through hyperparameter Theta:
# w_opt_array, S_array, chisqrt_array = bioen(sim_em_v_data,exp_em_mask,std,thetas, g0, g_init, sf_start, n_iter, epsilon, pgtol, maxiter)

# Running BioEN in a loop so we can apply knee dectection algorithm later
w_opt_d = dict()
sf_opt_d = dict()
s_dict = dict()
chisqrt_d = dict()
for th in thetas:
    w_temp = bioen_sigma_one(sim_map_g_mask,exp_map_mask,std,th, g0, g_init, epsilon, pgtol, maxiter)
    w_opt_d[th] = w_temp
    s_dict[th] = get_entropy(w0,w_temp)
    chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_map_g_mask,exp_map_mask)

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
        w_temp = bioen_sigma_one(sim_map_g_mask,exp_map_mask,std,th, g0, g_init, epsilon, pgtol, maxiter)
        w_opt_d[th] = w_temp
        s_dict[th] = get_entropy(w0,w_temp)
        chisqrt_d[th] = chiSqrTerm(w_temp,std,sim_map_g_mask,exp_map_mask)
    # Knee locator
    theta_index_new = knee_loc(list(s_dict.values()),list(chisqrt_d.values()))
    theta_new = list(w_opt_d.keys())[theta_index_new]
    thetas_sorted = np.sort(list(w_opt_d.keys()))[::-1]
    theta_index_sort = np.where(theta_new == thetas_sorted)[0][0]


"""
"" PLOTS
"""
# L-CURVE
name = "lcurve.svg"
plot_lcurve(s_dict,chisqrt_d,theta_new,N_voxels,name)

# Neff
name = "Neff.svg"
plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,name)

# WEIGHTS
name = "weights.svg"

max_weights = max(value for sublist in w_opt_d.values() for value in sublist)

plot_weights(w_opt_d,theta_new,name)

plik_weights = open("weights.dat","w")
w_all = w_opt_d[theta_new]
np.savetxt(plik_weights,w_all)
plik_weights.close()


"""
"" CORRELATION with exp map
"""
mapa = gaussian_filter(sim_map_array*sf_start,sigma_start)
cc,cc_prior,cc_single = map_correlations_mask(mapa,exp_map_norm,mask_comb,w_opt_d,w0,theta_new)


print("Correlation is: "+str(cc))

# TOP
best_single = np.argsort(w_opt_d[theta_new])[::-1][0]


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
name = "FSC.svg"
plot_fsc(fsc_prior,fsc_poster,name)


"""
"" RMSF calculations
"""
# Get RMSF
rmsf_prior, rmsf_post = get_rmsf(u,w_opt_d[theta_new])

# plot RMSF
name = "RMSF.svg"
plot_rmsf(rmsf_prior,rmsf_post,name)

# Writing best structure with rmsf posterior saved as betafactor:
u.trajectory[best_single]
for i in range(0,len(rmsf_post)):
    v = u.select_atoms("resid "+str(i+1))
    v.atoms.tempfactors = rmsf_post[i]

u.atoms.write("best.pdb")

"""
"" PCA
"""
# Calculate PCA
eigenvectors_prior, eigenvectors_post = do_PCA(u,w_opt_d[theta_new])

# Compute dot products
dot_products_post = np.dot(eigenvectors_prior, eigenvectors_post.T)

name = "PCA.svg"
plot_pca(dot_products_post,name)

# Save the eigenvector to a file
np.savetxt('eigenvector_prior.txt', eigenvectors_prior[0])
np.savetxt('eigenvector_post.txt', eigenvectors_post[0])

"""
"" WRITING POSTERIOR AND PRIOR MAP
"""
# Saving posterior map
write_map(posterior_map,"map_posterior.mrc",map_param)

# Saving prior map
write_map(prior_map,"map_prior.mrc",map_param)

plik = open("statistics.dat", "a")

plik.write("Theta value chosen by Kneedle algorithm: "+str(theta_new)+"\n")
plik.write("Reduced Chisqrt: " + str(chisqrt_d[theta_new]/N_voxels)+"\n")
plik.write("Neff: " + str(np.exp(s_dict[theta_new]))+"\n")
plik.write("Priori Correlation [CC_sim+exp]: "+str(cc_prior)+"\n")
plik.write("Posteriori Correlation [CC_sim+exp]: "+str(cc)+"\n")
plik.write("Single Best structure Correlation [CC_sim+exp]: "+str(cc_single)+"\n")
plik.write("Priori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_0_05)+"\n")
plik.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_1_05)+"\n")
plik.write("Single Best structure: " + str(best_single)+"\n")
plik.close()

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
    # Generating filtered and scalled map
    sim_map_g_sel = gaussian_filter(sim_map_array_selected*sf_start,sigma_start)
    sim_map_g_mask_sel = np.array([map[mask] for map in sim_map_g_sel])
    # Running BioEN iterations through hyperparameter Theta:
    # Running BioEN in a loop so we can apply knee dectection algorithm later
    w_opt_d_sel = dict()
    s_dict_sel = dict()
    chisqrt_d_sel = dict()
    for th in thetas:
        w_temp = bioen_sigma_one(sim_map_g_mask_sel,exp_map_mask,std,th, g0, g_init, epsilon, pgtol, maxiter)
        w_opt_d_sel[th] = w_temp
        s_dict_sel[th] = get_entropy(w0,w_temp)
        chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_map_g_mask_sel,exp_map_mask)
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
            w_temp = bioen_sigma_one(sim_map_g_mask_sel,exp_map_mask,std,th, g0, g_init, epsilon, pgtol, maxiter)
            w_opt_d_sel[th] = w_temp
            s_dict_sel[th] = get_entropy(w0,w_temp)
            chisqrt_d_sel[th] = chiSqrTerm(w_temp,std,sim_map_g_mask_sel,exp_map_mask)
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
        name = "iter/lcurve_iter.svg"
        plot_lcurve(s_dict_sel,chisqrt_d_sel,theta_new_sel,N_voxels,name)
        # WEIGHTS
        name = "iter/weights_iter.svg"
        plot_weights_sel(w_opt_d_sel,N_models,selected_frames,theta_new_sel,name)
        plik_weights = open("iter/weights_iter.dat","w")
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
        name = "iter/FSC_iter.svg"
        plot_fsc_sel(fsc_prior,fsc_poster,fsc_poster_sel,name)
        # posterior map
        posterior_map_sel = np.dot(sim_map_array_selected.T,w_opt_d_sel[theta_new_sel]).T
        write_map(posterior_map_sel,"iter/map_posterior_iter.mrc",map_param)
        # Neff
        Neff_sel = np.exp(s_dict_sel[theta_new_sel])
        # Get RMSF
        rmsf_post_sel = get_rmsf_sel(u,selected_frames,w_opt_d_sel[theta_new_sel])
        # plot RMSF
        name = "iter/RMSF_iter.svg"
        plot_rmsf_sel(rmsf_prior,rmsf_post,rmsf_post_sel,name)
        # Calculate PCA
        eigenvectors_post_sel = do_PCA_sel(u,selected_frames,w_opt_d_sel[theta_new_sel])
        # Compute dot products
        dot_products_post_sel = np.dot(eigenvectors_prior, eigenvectors_post_sel.T)
        name = "iter/PCA_iter.svg"
        plot_pca(dot_products_post_sel,name)
        # Save the eigenvector to a file
        np.savetxt('iter/eigenvector_post_iter.txt', eigenvectors_post_sel[0])
        # writing statistics
        plik_sel = open("iter/statistics_iter.dat","a")
        plik_sel.write("ITERATION "+str(iteration)+"\n")
        plik_sel.write("Theta value chosen by Kneedle algorithm: "+str(theta_new_sel)+"\n")
        plik_sel.write("Reduced Chisqrt value: "+str(chisqrt_d_sel[theta_new_sel]/N_voxels)+"\n")
        plik_sel.write("Neff: " + str(Neff_sel)+"\n")
        plik_sel.write("Number of structures: "+str(len(selection))+"\n")
        plik_sel.write("Posteriori Correlation [CC_sim+exp]: "+str(cc_sel)+"\n")
        plik_sel.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index_sel)+" [1/A]: "+str(fsc_1_05_sel)+"\n")
        plik_sel.close()
        # New selection
        selection_range_sel = int(len(w_opt_d_sel[theta_new_sel])*Neff_sel)
        selection = np.argsort(w_opt_d_sel[theta_new_sel])[::-1][:selection_range_sel]
        temp_sel = selected_frames[selection]
        selected_frames = temp_sel
        iteration += 1



# Writing best structure with rmsf posterior saved as betafactor:
best_single_sel = selected_frames[0]
u.trajectory[best_single_sel]
for i in range(0,len(rmsf_post_sel)):
    v = u.select_atoms("resid "+str(i+1))
    v.atoms.tempfactors = rmsf_post_sel[i]

u.atoms.write("iter/best.pdb")


# Create a PDB writer
with Writer("iter/best_frames.pdb", multiframe=True) as w:
    for ts in u.trajectory:
        if ts.frame in selected_frames:
            print(f"Writing frame {ts.frame}")
            w.write(u)
