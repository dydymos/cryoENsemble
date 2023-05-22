import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from reference_map_ADK import *
from plot import *
from validation import *
import argparse
from tqdm import tqdm



import warnings
warnings.filterwarnings('ignore')

"""
"" PATHWAYS
"""

situs_path = "/users/towlodar/src/Situs_3.1/bin/"
structures_path = "structures/"

"""
"" Input
"""

def checker(a):
    num = float(a)
    if num != 10 and num != 6 and num != 3:
        raise argparse.ArgumentTypeError('Invalid value. Only resolution equal to 3, 6 or 10 are acceptable')
    return num

def checker_mask(mask_type):
    if mask_type != 'exp' and mask_type != 'sim':
        raise argparse.ArgumentTypeError('Invalid value. Mask can only be based on experimental (exp) or both experimental and simulations data (sim+exp)')
    return mask_type


parser = argparse.ArgumentParser(description='Running cryBioEN for ADK example')
parser.add_argument('weight', type = float, help = 'Weight for the 1AKE structure in the reference map')
parser.add_argument('resM', type = checker, help = 'Reference map resolution [3, 6 or 10A]')
parser.add_argument('noise', type = float, help = 'Noise level, which is defined as normal distribution centered around 0 and with std equal to X of the maximum density in the Reference map')
parser.add_argument('mask_type', type = checker_mask, help = 'Type of mask: exp or sim')


args = parser.parse_args()

"""
"" MAP details based on 3A, 6A and 10A maps generated in ChimeraX
"""
map_param = dict()
if (args.resM == 3):
    map_param['nx'] = 70
    map_param['ny'] = 56
    map_param['nz'] = 76
    map_param['vox'] = 1.0
    map_param['em_origin'] = np.array([-7.634, 16.903, -18.461])
elif (args.resM == 6):
    map_param['nx'] = 44
    map_param['ny'] = 37
    map_param['nz'] = 47
    map_param['vox'] = 2.0
    map_param['em_origin'] = np.array([-16.634,   7.903, -27.461])
elif (args.resM == 10):
    map_param['nx'] = 34
    map_param['ny'] = 30
    map_param['nz'] = 36
    map_param['vox']= 3.3333335
    map_param['em_origin'] = np.array([-28.634,  -4.097, -39.461])


# CryoEM paramters for map generators
cryoem_param = cryoEM_parameters(map_param)

"""
"" Creating average EM map of open and close state based on weights
"""
# Weights for 1AKE and 4AKE:
w_1ake = args.weight
w_4ake = 1 - w_1ake
em_weights = np.array([w_1ake,w_4ake])
em_weights_0 = np.array([0,1])
em_weights_1 = np.array([1,0])

# reference map resolution based on ChimeraX
sigma = args.resM*0.225

# average reference map with predefined resolution based on ChimeraX molmap
em_map = pdb2map_avg_chimeraX(em_weights,args.resM,["1ake.pdb","4ake_aln.pdb"],map_param,cryoem_param)

# open and close map for masked CC
#em_map_open = pdb2map_avg_chimeraX(em_weights_0,args.resM,["1ake.pdb","4ake_aln.pdb"],map_param,cryoem_param)
#em_map_close = pdb2map_avg_chimeraX(em_weights_1,args.resM,["1ake.pdb","4ake_aln.pdb"],map_param,cryoem_param)

# map with noise, which is normal distribution centered on 0 and with std equal to X*100% of the maximum density in the em_map
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

# Saving original map
write_map(em_map,"map_"+str(w_1ake)+".mrc",map_param)

# Saving open map
#write_map(em_map_open,"map_open.mrc",map_param)

# Saving close map
#write_map(em_map_close,"map_close.mrc",map_param)

# Saving original map with noise
write_map(em_map_noise,"map_noise_"+str(w_1ake)+".mrc",map_param)

# Saving normalized map with noise
write_map(em_map_norm,"map_norm_"+str(w_1ake)+".mrc",map_param)

# Saving normalized map with noise and without negative density
#os.system("rm map_thr_"+str(w_1ake)+".mrc")
#write_map(em_map_threshold,"map_thr_"+str(w_1ake)+".mrc",map_param)

"""
"" Fitting structures into density using Chimerax
"""
# Fitting 1ake structures
#for i in range(1,51):
#    os.system('chimerax --nogui --exit --cmd "open map_norm_'+str(w_1ake)+'.mrc; open /home/didymos/Linux_05.2021/Projects/BioEN/ADK/1ake/structures/'+str(i)+'_fit.pdb; fitmap #2 inMap #1 resolution 6;save best.pdb models #2 relModel #1"')
#    os.system('mv best.pdb /home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/1ake/structures/'+str(i)+'_rb_fit.pdb')


# Fitting 4ake structures
#for i in range(1,51):
#    os.system('chimerax --nogui --exit --cmd "open map_norm_'+str(w_1ake)+'.mrc; open /home/didymos/Linux_05.2021/Projects/BioEN/ADK/4ake/structures/'+str(i)+'_fit.pdb; fitmap #2 inMap #1 resolution 6;save best.pdb models #2 relModel #1"')
#    os.system('mv best.pdb /home/didymos/Linux_05.2021/Projects/BioEN/ADK/cryoBioEN/tmp/4ake/structures/'+str(i)+'_rb_fit.pdb')


"""
"" Fitting structures into density using Situs
"""
# Fitting 1ake structures
for i in tqdm(range(1,51)):
    os.system(situs_path+'colores map_norm_'+str(w_1ake)+'.mrc '+structures_path+'1ake/'+str(i)+'.pdb -res '+str(args.resM)+' -nprocs 6')
    os.system('mv col_best_001.pdb '+structures_path+'1ake/'+str(i)+'_fit.pdb')
    os.system('rm col_*')


for i in tqdm(range(1,51)):
    os.system(situs_path+'colores map_norm_'+str(w_1ake)+'.mrc '+structures_path+'4ake/'+str(i)+'.pdb -res '+str(args.resM)+' -nprocs 6')
    os.system('mv col_best_001.pdb '+structures_path+'4ake/'+str(i)+'_fit.pdb')
    os.system('rm col_*')

"""
"" STRUCTURAL ENSEMBLE
"""

# OK Now we read models from 1ake and 4ake
# We use 50 model from 1ake and 50 models from 4ake

# Number of structures/models
N_models = 100

PDBs_1ake = glob.glob(structures_path+'1ake/*_fit.pdb')[:50]
PDBs_4ake = glob.glob(structures_path+'4ake/*_fit.pdb')[:50]

# PDB files
PDBs=PDBs_1ake+PDBs_4ake

# Generating array of EM maps based on structures
print("Generating an array of EM maps based on the structures from MD simulation")
sim_em_data = np.array(pdb2map_array(PDBs,map_param,cryoem_param))

sim_map = np.sum(sim_em_data,0)

# Saving normalized map with noise and without negative density
write_map(sim_map,"map_sim_"+str(w_1ake)+".mrc",map_param)



"""
"" MASKING
"""

# Masking can take place in two ways:
# EXP - when we use mask generated from experimental map
# SIM - when we also include voxels from the map generated for each fitted structure
# for that we use threshold equal to 3x the simulated map std

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
thetas = [10000000,1000000,100000,10000,1000,100,10,1,0]

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

print("\nRunning BioEN iterations to narrow down the theta value")
for i in range(0,10):
    print("Round "+str(i+1)+" out of 10:")
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
name = "lcurve_"+str(float(w_1ake))+".svg"
plot_lcurve(s_dict,chisqrt_d,theta_new,N_voxels,name)

# Neff
name = "Neff_"+str(float(w_1ake))+".svg"
plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,name)

# WEIGHTS
name = "weights_"+str(float(w_1ake))+".svg"
plot_weights(w_opt_d,np.arange(0,N_models),theta_new,N_models,name)

plik_weights = open("weights_"+str(float(w_1ake))+".dat","w")
w_all = w_opt_d[theta_new]
np.savetxt(plik_weights,w_all)
plik_weights.close()


"""
"" CORRELATIONS
"""
#Global CC
cc,cc_prior,cc_single = map_correlations_mask(sim_em_data,em_map_norm,mask_comb,w_opt_d,w0,theta_new)
cc_r,cc_prior_r,cc_single_r = map_correlations_mask(sim_em_data,em_map_norm,mask_exp,w_opt_d,w0,theta_new)


# TOP 1 structure
best = np.argsort(w_opt_d[theta_new])[::-1][0]


"""
"" FOURIER SHELL CORRELATION
"""

# prior map
prior_map = np.dot(sim_em_data.T,w0).T

# posterior map
posterior_map = np.dot(sim_em_data.T,w_opt_d[theta_new]).T

# FSC between reference map and prior map
fsc_prior = FSC(em_map_norm,prior_map,map_param)
fsc_poster= FSC(em_map_norm,posterior_map,map_param)

if (((fsc_prior[:,1]<0.5).any()) & (fsc_poster[:,1]<0.5).any()):
    fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.5][0]
    fsc_1_05 = fsc_poster[::-1][:,0][fsc_poster[::-1][:,1]>0.5][0]
    fsc_index = 0.5
elif (((fsc_prior[:,1]<0.6).any()) & (fsc_poster[:,1]<0.6).any()):
    fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.6][0]
    fsc_1_05 = fsc_poster[::-1][:,0][fsc_poster[::-1][:,1]>0.6][0]
    fsc_index = 0.6
elif (((fsc_prior[:,1]<0.7).any()) & (fsc_poster[:,1]<0.7).any()):
    fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.7][0]
    fsc_1_05 = fsc_poster[::-1][:,0][fsc_poster[::-1][:,1]>0.7][0]
    fsc_index = 0.7
elif (((fsc_prior[:,1]<0.8).any()) & (fsc_poster[:,1]<0.8).any()):
    fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.8][0]
    fsc_1_05 = fsc_poster[::-1][:,0][fsc_poster[::-1][:,1]>0.8][0]
    fsc_index = 0.8


# Plot FSC
name = "FSC_"+str(float(w_1ake))+".svg"
plot_fsc(fsc_prior,fsc_poster,name)


"""
"" TEMPy
"""
# SMOC calculations
smoc_prior,smoc_poster = SMOC("map_norm_"+str(w_1ake)+".mrc",args.resM,w0,w_opt_d[theta_new])

# Plotting SMOC
name = "SMOC_"+str(float(w_1ake))+".svg"
plot_smoc(smoc_prior,smoc_poster,name)


"""
"" WRITING POSTERIOR AND PRIOR MAP
"""
# Saving posterior map
write_map(posterior_map,"map_posterior_"+str(w_1ake)+".mrc",map_param)

# Saving prior map
write_map(prior_map,"map_prior_"+str(w_1ake)+".mrc",map_param)



"""
"" WRITING STATISTICS
"""
plik = open("statistics.dat","a")
plik.write("POPULATION of 1AKE in the map: "+str(w_1ake)+"\n")
plik.write("Theta value chosen by Kneedle algorithm: "+str(theta_new)+"\n")
plik.write("Reduced Chisqrt value: "+str(chisqrt_d[theta_new]/N_voxels)+"\n")
plik.write("Neff: " + str(np.exp(s_dict[theta_new]))+"\n")
plik.write("Population of 1ake: "+str(np.round(np.sum(w_opt_d[theta_new][:50]),2))+"\n")
plik.write("Population of 4ake: "+str(np.round(np.sum(w_opt_d[theta_new][50:]),2))+"\n")
plik.write("Priori Correlation [CC_sim+exp, CC_exp]: "+str(cc_prior)+"\t"+str(cc_prior_r)+"\n")
plik.write("Posteriori Correlation [CC_sim+exp, CC_exp]: "+str(cc)+"\t"+str(cc_r)+"\n")
plik.write("Single Best structure Correlation [CC_sim+exp, CC_exp]: "+str(cc_single)+"\t"+str(cc_single_r)+"\n")
plik.write("Priori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_0_05)+"\n")
plik.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_1_05)+"\n")
plik.write("Avg Priori SMOC: "+str(np.mean(smoc_prior))+"\n")
plik.write("Avg Posteriori SMOC: "+str(np.mean(smoc_poster))+"\n")
plik.write("Single Best structure: "+str(PDBs[best])+"\n")
plik.write("\n")
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

#selected_frames = np.array([int(x) for x in np.where(w_opt_d[theta_new]>w_init)[0]])
selected_frames = np.argsort(w_opt_d[theta_new])[::-1][:selection_range]
selection = selected_frames
new_chisqrt = chisqrt_d[theta_new]/N_voxels
old_chisqrt = chisqrt_d[theta_new]/N_voxels

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
thetas = [10000000,1000000,100000,10000,1000,100,10,1,0]

"""
"" WHILE LOOP
"""
print("\nRunning BioEN iterations to find the smalles set of structures\n")
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
    theta_index_sort_sel = theta_index
    for i in range(0,10):
        thetas_old = np.sort(list(w_opt_d_sel.keys()))[::-1]
        theta_up = (thetas_old[theta_index_sort_sel - 1] + thetas_old[theta_index_sort_sel])/2.
        theta_down = (thetas_old[theta_index_sort_sel + 1] + thetas_old[theta_index_sort_sel])/2.
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
    #name = "iter/lcurve_"+str(float(w_1ake))+"_iter_"+str(iteration)+".svg"
    #plot_lcurve(s_dict_sel,chisqrt_d_sel,theta_new_sel,N_voxels,name)
    # Neff
    #name = "iter/Neff_"+str(float(w_1ake))+"_iter_"+str(iteration)+".svg"
    #plot_Neff(s_dict_sel,chisqrt_d_sel,theta_new_sel,N_voxels,name)
    # WEIGHTS
    name = "iter/weights_"+str(float(w_1ake))+"_iter_"+str(iteration)+".svg"
    plot_weights(w_opt_d_sel,selection,theta_new_sel,N_models,name)
    plik_weights = open("iter/weights_"+str(float(w_1ake))+"_iter_"+str(iteration)+".dat","w")
    w_all = np.zeros(N_models)
    w_all[selection] = w_opt_d_sel[theta_new_sel]
    np.savetxt(plik_weights,w_all)
    plik_weights.close()
    # Cross-correlation
    #Global CC
    cc_sel,cc_prior_sel,cc_single_sel = map_correlations_mask(sim_em_data[selection],em_map_norm,mask_comb,w_opt_d_sel,w0,theta_new_sel)
    cc_sel_r,cc_prior_sel_r,cc_single_sel_r = map_correlations_mask(sim_em_data[selection],em_map_norm,mask_exp,w_opt_d_sel,w0,theta_new_sel)
    # posterior map
    posterior_map_sel = np.dot(sim_em_data[selection].T,w_opt_d_sel[theta_new_sel]).T
    # FSC between reference map and prior map
    fsc_poster_sel= FSC(em_map_norm,posterior_map_sel,map_param)
    if (((fsc_prior[:,1]<0.5).any()) & (fsc_poster_sel[:,1]<0.5).any()):
        fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.5][0]
        fsc_1_05_sel = fsc_poster_sel[::-1][:,0][fsc_poster_sel[::-1][:,1]>0.5][0]
        fsc_index = 0.5
    elif (((fsc_prior[:,1]<0.6).any()) & (fsc_poster_sel[:,1]<0.6).any()):
        fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.6][0]
        fsc_1_05_sel = fsc_poster_sel[::-1][:,0][fsc_poster_sel[::-1][:,1]>0.6][0]
        fsc_index = 0.6
    elif (((fsc_prior[:,1]<0.7).any()) & (fsc_poster_sel[:,1]<0.7).any()):
        fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.7][0]
        fsc_1_05_sel = fsc_poster_sel[::-1][:,0][fsc_poster_sel[::-1][:,1]>0.7][0]
        fsc_index = 0.7
    elif (((fsc_prior[:,1]<0.8).any()) & (fsc_poster_sel[:,1]<0.8).any()):
        fsc_0_05 = fsc_prior[::-1][:,0][fsc_prior[::-1][:,1]>0.8][0]
        fsc_1_05_sel = fsc_poster_sel[::-1][:,0][fsc_poster_sel[::-1][:,1]>0.8][0]
        fsc_index = 0.8
    # Plot FSC
    name = "iter/FSC_"+str(float(w_1ake))+"_iter_"+str(iteration)+".svg"
    plot_fsc(fsc_prior,fsc_poster_sel,name)
    # SMOC calculations
    smoc_poster_sel = SMOC_iter("map_norm_"+str(float(w_1ake))+".mrc",args.resM,w_opt_d_sel[theta_new_sel],selected_frames)
    # Plotting SMOC
    name = "iter/SMOC_"+str(float(w_1ake))+"_iter_"+str(iteration)+".svg"
    plot_smoc(smoc_prior,smoc_poster_sel,name)
    # Saving posterior map
    sim_em_rew_sel = np.dot(sim_em_data[selection].T,w_opt_d_sel[theta_new_sel]).T
    write_map(sim_em_rew_sel,"iter/map_posterior_"+str(w_1ake)+"_iter_"+str(iteration)+".mrc",map_param)
    # Neff
    Neff_sel = np.exp(s_dict_sel[theta_new_sel])
    # writing statistics
    plik_sel = open("iter/statistics_"+str(w_1ake)+"_iter.dat","a")
    plik_weights = open("iter/weights_"+str(w_1ake)+"_iter_"+str(iteration)+".dat","w")
    w_all = np.zeros(N_models)
    w_all[selection] = w_opt_d_sel[theta_new_sel]
    np.savetxt(plik_weights,w_all)
    plik_weights.close()
    plik_sel.write("ITERATION "+str(iteration)+"\n")
    plik_sel.write("POPULATION of 1AKE in the map: "+str(w_1ake)+"\n")
    plik_sel.write("Theta value chosen by Kneedle algorithm: "+str(theta_new_sel)+"\n")
    plik_sel.write("Reduced Chisqrt value: "+str(chisqrt_d_sel[theta_new_sel]/N_voxels)+"\n")
    plik_sel.write("Neff: " + str(np.exp(s_dict_sel[theta_new_sel]))+"\n")
    plik_sel.write("Number of structures: "+str(len(selection))+"\n")
    plik_sel.write("Population of 1ake: "+str(np.round(np.sum(w_all[:50]),2))+"\n")
    plik_sel.write("Population of 4ake: "+str(np.round(np.sum(w_all[50:]),2))+"\n")
    plik_sel.write("Posteriori Correlation [CC_sim+exp, CC_exp]: "+str(cc_sel)+"\t"+str(cc_sel_r)+"\n")
    plik_sel.write("Posteriori Fourrier Shell Correlation at "+str(fsc_index)+" [1/A]: "+str(fsc_1_05_sel)+"\n")
    plik_sel.write("Avg Posteriori SMOC: "+str(np.mean(smoc_poster_sel))+"\n")
    plik_sel.write("\n")
    selection_range_sel = int(len(w_opt_d_sel[theta_new_sel])*Neff_sel)
    selection = np.argsort(w_opt_d_sel[theta_new_sel])[::-1][:selection_range_sel]
    temp_sel = selected_frames[selection]
    selected_frames = temp_sel
    selection = selected_frames
    old_chisqrt = new_chisqrt
    new_chisqrt = chisqrt_d_sel[theta_new_sel]/N_voxels
    iteration+=1



plik_sel.close()
plik.close()
