import numpy as np
import pandas as pd
import glob
import MDAnalysis
from MDAnalysis import *
from MDAnalysis.analysis import *
import mrcfile
from saxstats import *
import saxstats.saxstats as saxs
from MDAnalysis.analysis import density
from TEMPy.protein.structure_parser import PDBParser
from TEMPy.maps.map_parser import MapParser
from TEMPy.protein.structure_blurrer import StructureBlurrer
import TEMPy.protein.scoring_functions as scf
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



def cubic_map(map,map_param):
    # Generating a cubic map
    nx_old = map_param['nx']
    ny_old = map_param['ny']
    nz_old = map_param['nz']
    n_max = np.max([nx_old,ny_old,nz_old])
    nx = n_max
    ny = n_max
    nz = n_max
    nx_diff = n_max - nx_old
    ny_diff = n_max - ny_old
    nz_diff = n_max - nz_old
    vox = map_param['vox']
    new_map = np.zeros([nz,ny,nx])
    new_map[nz_diff:nz_old+nz_diff,ny_diff:ny_old+ny_diff,nx_diff:nx_old+nx_diff] = map.data
    return new_map,n_max,vox


def FSC(map_0,map_1,map_param):
    # First we need to transfer maps into cubic one
    map_0_c,n_max,vox = cubic_map(map_0,map_param)
    # first map usually is map+noise+norm so we need to make sure that there are no negative values so comparison makes sense
    map_0_c[np.where(map_0_c<0)]=0
    # second map for coparison is either prior or posterior
    map_1_c,n_max,vox = cubic_map(map_1,map_param)
    side = n_max * vox
    fsc = saxs.calc_fsc(map_0_c,map_1_c,side)
    return fsc


def SMOC(ref_map_name,res,w0,w_opt):
    sc = scf.ScoringFunctions()
    path = "minim/"
    PDBs = []
    for i in range(1,101):
        PDBs.append(path+"/minim_"+str(i)+'.pdb')
    # Read map that is our reference
    map_ref = MapParser.readMRC(ref_map_name)
    smoc_all = []
    for i in tqdm(PDBs):
        # read the pdb file
        prot = PDBParser.read_PDB_file(i,i, hetatm=False, water=False)
        smoc = sc.SMOC(map_ref,res,prot)
        smoc_all.append(pd.DataFrame(smoc[0])['A'].values)
    smoc_prior = np.dot(np.array(smoc_all).T,w0)
    smoc_poster = np.dot(np.array(smoc_all).T,w_opt)
    return smoc_prior,smoc_poster



def rmsd_matrix(PDBs):
    rmsd_m = np.zeros([100,100])
    for i in tqdm(range(0,99)):
        for j in range(i+1,100):
            a = Universe(PDBs[i])
            at_0 = a.select_atoms('name CA')
            x0 = at_0.positions
            b = Universe(PDBs[j])
            at_1 = b.select_atoms('name CA')
            x1 = at_1.positions
            rmsd_m[i][j]=rms.rmsd(x0,x1)
            rmsd_m[j][i]=rms.rmsd(x0,x1)
    return rmsd_m



def clustering(rmsd_m,cutoff):
    # Not clustered - initally is the whole set
    not_cluster = np.arange(0,100)
    # logic matrix
    logic_M = (rmsd_m<cutoff).astype(int)
    # dictionary with clusters
    clusters = dict()
    # dictionary with clusters centers
    cluster_center_dict = dict()
    # initial shape of the logic_M
    n = np.shape(logic_M)[0]
    i=0
    while np.sum(logic_M)!=n*n:
        cluster_center = np.argmax(np.sum(logic_M,1))
        members = np.where(logic_M[:,cluster_center]==1)[0]
        clusters[i] = not_cluster[members]
        cluster_center_dict[i] = not_cluster[cluster_center]
        not_members = np.delete(not_cluster,members)
        not_cluster = not_members
        M_temp = np.delete(logic_M,members,0)
        logic_M = np.delete(M_temp,members,1)
        n = np.shape(logic_M)[0]
        i+=1
    clusters[i]=not_cluster
    return clusters


def J_S_div(clusters,pdb_list,w):
    prob_prior_all = []
    prob_poster_all = []
    prob_ref_all = []
    epsilon = 0.00001
    for cluster_id in range(0,len(clusters.keys())):
        prob_prior = epsilon
        prob_poster = epsilon
        prob_ref = epsilon
        for j in clusters[cluster_id]:
            if j in pdb_list: prob_ref+=0.1
            prob_poster+=w[j]
            prob_prior+=0.01
        prob_prior_all.append(prob_prior)
        prob_poster_all.append(prob_poster)
        prob_ref_all.append(prob_ref)
    # Jensen-Shannon
    D_prior_ref = np.sum(np.array(prob_prior_all)*np.log(np.array(prob_prior_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_prior_all)))))
    D_ref_prior = np.sum(np.array(prob_ref_all)*np.log(np.array(prob_ref_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_prior_all)))))
    D_JS_ref_prior = 0.5*(D_prior_ref+D_ref_prior)
    D_poster_ref = np.sum(np.array(prob_poster_all)*np.log(np.array(prob_poster_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_poster_all)))))
    D_ref_poster = np.sum(np.array(prob_ref_all)*np.log(np.array(prob_ref_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_poster_all)))))
    D_JS_ref_poster = 0.5*(D_poster_ref+D_ref_poster)
    return D_JS_ref_prior,D_JS_ref_poster



def J_S_div_missing(clusters,pdb_list,w,selected_frames):
  prob_prior_all = []
  prob_poster_all = []
  prob_ref_all = []
  counter = 0
  epsilon = 0.00001
  for cluster_id in range(0,len(clusters.keys())):
    prob_prior = epsilon
    prob_poster = epsilon
    prob_ref = epsilon
    for j in clusters[cluster_id]:
      if j in pdb_list: prob_ref+=0.1
      elif j in selected_frames:
        prob_poster+=w[counter]
        prob_prior+=0.011111
        counter+=1
    prob_prior_all.append(prob_prior)
    prob_poster_all.append(prob_poster)
    prob_ref_all.append(prob_ref)
  # Jensen-Shannon
  D_prior_ref = np.sum(np.array(prob_prior_all)*np.log(np.array(prob_prior_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_prior_all)))))
  D_ref_prior = np.sum(np.array(prob_ref_all)*np.log(np.array(prob_ref_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_prior_all)))))
  D_JS_ref_prior = 0.5*(D_prior_ref+D_ref_prior)
  D_poster_ref = np.sum(np.array(prob_poster_all)*np.log(np.array(prob_poster_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_poster_all)))))
  D_ref_poster = np.sum(np.array(prob_ref_all)*np.log(np.array(prob_ref_all)/(0.5*(np.array(prob_ref_all)+np.array(prob_poster_all)))))
  D_JS_ref_poster = 0.5*(D_poster_ref+D_ref_poster)
  return D_JS_ref_prior,D_JS_ref_poster
