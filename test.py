import os
import glob
import numpy as np
from bioEN_functions import *
from cryoEM_methods import *
from reference_map_ADK import *
from plot import *
from os.path import exists
import argparse


"""
"" PATHWAYS
"""

situs_path = "/work/e280/e280-Christodoul/tomek/soft/Situs_3.1/bin/"
structures_path = "structures/"

"""
"" Input
"""

def checker(a):
    num = float(a)
    if num != 10 and num != 6:
        raise argparse.ArgumentTypeError('Invalid value. Only resolution equal to 6 or 10 are acceptable')
    return num


parser = argparse.ArgumentParser(description='Running cryBioEN for ADK example')
parser.add_argument('weight', type = float, help = 'Weight for the 1AKE structure in the reference map')
parser.add_argument('resM', type = checker, help = 'Reference map resolution')
parser.add_argument('resG', type = float, help = 'Generated map resolution')
parser.add_argument('noise', type = float, help = 'Noise level, which is defined as normal distribution centered around 0 and with std equal to X of the maximum density in the Reference map')
#parser.add_argument('mask', help = 'Type of mask')


args = parser.parse_args()

"""
"" MAP details based on 6A and 10A maps generated in ChimeraX
"""
map_param = dict()
if (args.resM == 6):
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

# reference map resolution
sigma = args.resM*0.225

# simulated map resolution
sigma_sim = args.resG*0.225

# average map
em_map = pdb2map_avg(em_weights,sigma,["1ake.pdb","4ake_aln.pdb"],map_param,cryoem_param)

# map with noise, which is normal distribution centered on 0 and with std equal to X% of the maximum density in the em_map
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

# Saving original map with noise
if exists("map_noise_"+str(w_1ake)+".mrc"):
    os.system("rm map_noise_"+str(w_1ake)+".mrc")
    write_map(em_map_noise,"map_noise_"+str(w_1ake)+".mrc",map_param)
else: write_map(em_map_noise,"map_noise_"+str(w_1ake)+".mrc",map_param)

# Saving normalized map with noise
if exists("map_norm_"+str(w_1ake)+".mrc"):
    os.system("rm map_norm_"+str(w_1ake)+".mrc")
    write_map(em_map_norm,"map_norm_"+str(w_1ake)+".mrc",map_param)
else: write_map(em_map_norm,"map_norm_"+str(w_1ake)+".mrc",map_param)

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
for i in range(1,51):
    os.system(situs_path+'colores map_norm_'+str(w_1ake)+'.mrc /home/didymos/Linux_05.2021/Projects/BioEN/ADK/1ake/structures/'+str(i)+'_fit.pdb -res '+str(args.resM)+' -nprocs 6')
    os.system('mv col_best_001.pdb structures/1ake/'+str(i)+'_rb_fit.pdb')
    os.system('rm col_*')


for i in range(1,51):
    os.system(situs_path+'colores map_norm_'+str(w_1ake)+'.mrc /home/didymos/Linux_05.2021/Projects/BioEN/ADK/4ake/structures/'+str(i)+'_fit.pdb -res '+str(args.resM)+' -nprocs 6')
    os.system('mv col_best_001.pdb structures/4ake/'+str(i)+'_rb_fit.pdb')
    os.system('rm col_*')
