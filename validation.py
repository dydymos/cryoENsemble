import numpy as np
import pandas as pd
import glob
import MDAnalysis
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
    structures_path = "structures/"
    PDBs_1ake = glob.glob(structures_path+'1ake/*_fit.pdb')[:50]
    PDBs_4ake = glob.glob(structures_path+'4ake/*_fit.pdb')[:50]
    PDBs=PDBs_1ake+PDBs_4ake
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


def SMOC_iter(ref_map_name,res,w_opt,pdb_selection):
    sc = scf.ScoringFunctions()
    structures_path = "structures/"
    PDBs_1ake = glob.glob(structures_path+'1ake/*_fit.pdb')[:50]
    PDBs_4ake = glob.glob(structures_path+'4ake/*_fit.pdb')[:50]
    PDBs=PDBs_1ake+PDBs_4ake
    # Read map that is our reference
    map_ref = MapParser.readMRC(ref_map_name)
    smoc_all = []
    for i in tqdm(np.array(PDBs)[pdb_selection]):
        # read the pdb file
        prot = PDBParser.read_PDB_file(i,i, hetatm=False, water=False)
        smoc = sc.SMOC(map_ref,res,prot)
        smoc_all.append(pd.DataFrame(smoc[0])['A'].values)
    smoc_poster = np.dot(np.array(smoc_all).T,w_opt)
    return smoc_poster



