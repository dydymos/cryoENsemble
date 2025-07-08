import numpy as np
import pandas as pd
import glob
import MDAnalysis
import MDAnalysis.analysis
import mrcfile
from MDAnalysis.analysis import density
from tqdm import tqdm


def map_correlations_mask(sim_em_data,em_map,mask,w_opt_d,w0,theta_new):
    sim_em_poster = np.zeros(np.shape(sim_em_data[0]))
    for i in range(0,np.shape(sim_em_data)[0]):
      sim_em_poster+=sim_em_data[i]*w_opt_d[theta_new][i]
    poster_map = sim_em_poster[mask]
    sim_em_prior = np.zeros(np.shape(sim_em_data[0]))
    for i in range(0,np.shape(sim_em_data)[0]):
      sim_em_prior+=sim_em_data[i]*w0[i]
    prior_map = sim_em_prior[mask]
    # Posterior ensemble avg cc
    cc = np.corrcoef(poster_map,em_map[mask])[0][1]
    # Prior ensemble avg cc
    cc_prior = np.corrcoef(prior_map,em_map[mask])[0][1]
    # Single best strucute CC
    cc_single = []
    for i in sim_em_data:
        cc_single.append(np.corrcoef(i[mask],em_map[mask])[0][1])
    cc_single_best = np.max(cc_single)
    return cc,cc_prior, cc_single_best

