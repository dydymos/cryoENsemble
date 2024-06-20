import numpy as np
import MDAnalysis
import mrcfile
from random import *
from MDAnalysis.analysis import density
from tqdm import tqdm


##################
# PDB2EM_DENSITY #
##################

def cryoEM_parameters(map_param):
    """
    Dictionary of sigma and weight per atom type from Bonomi et al 2018 - atomic
    scattering factors fitted with single Gaussian - Ai*exp(-Bi*s^2), where s
    is scattering factor; Sigma = Bi, Weights=Ai
    """
    VOX = map_param["vox"]
    cryoem_param = {}
    SIGMA={}; WEIGHT={}
    SIGMA["C"]=15.146;   WEIGHT["C"]=2.49982
    SIGMA["O"]=8.59722;  WEIGHT["O"]=1.97692
    SIGMA["N"]=11.1116;  WEIGHT["N"]=2.20402
    SIGMA["S"]=15.8952;  WEIGHT["S"]=5.14099
    SIGMA["P"]=19.4293;  WEIGHT["P"]=5.45658
    # transform sigma in real space and get useful stuff
    PREFACT={}; INVSIG2={}
    for key in SIGMA:
        SIGMA[key] = np.sqrt( 0.5 * SIGMA[key] ) / np.pi
        INVSIG2[key] = 1.0 / SIGMA[key] / SIGMA[key]
        PREFACT[key] = WEIGHT[key] / np.sqrt(2.0*np.pi) / SIGMA[key]
    # constant parameter
    maxSD = 3.0
    # build delta dictionary
    delta={};
    for key in SIGMA:
        delta[key]=int(np.ceil(maxSD * SIGMA[key] / VOX))
    cryoem_param["prefact"] = PREFACT
    cryoem_param["invsig2"] = INVSIG2
    cryoem_param["delta"] = delta
    return cryoem_param


def pdb2map_array(PDBs,map_param,cryoem_param):
    """
    Generates an array with calculated EM density maps for each structure
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    INVSIG2 = cryoem_param["invsig2"]
    PREFACT = cryoem_param["prefact"]
    # prepare zero data
    data_array = []
    for ipdb in tqdm(range(0,len(PDBs))):
        data = np.zeros((nz,ny,nx), dtype=np.float32)
        u = MDAnalysis.Universe(PDBs[ipdb])
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S P")
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX))
            jj = int(round((apos[1]-em_origin[1])/VOX))
            kk = int(round((apos[2]-em_origin[2])/VOX))
            # get delta grid
            d=delta[atype]
            # get constant parameters
            invsig2 = INVSIG2[atype]
            pref =  PREFACT[atype]
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 )
        data_array.append(data)
    return data_array


def traj2map_array(u,map_param,cryoem_param):
    """
    Generates an array with calculated EM density maps for each structure from trajectory
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    INVSIG2 = cryoem_param["invsig2"]
    PREFACT = cryoem_param["prefact"]
    # prepare zero data
    data_array = []
    N = u.trajectory.n_frames
    for i in tqdm(np.arange(0,N)):
        data = np.zeros((nz,ny,nx), dtype=np.float32)
        u.trajectory[i]
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S P")
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX))
            jj = int(round((apos[1]-em_origin[1])/VOX))
            kk = int(round((apos[2]-em_origin[2])/VOX))
            # get delta grid
            d=delta[atype]
            # get constant parameters
            invsig2 = INVSIG2[atype]
            pref =  PREFACT[atype]
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 )
        data_array.append(data)
    return data_array


def pdb2map(sigma,PDB,map_param,cryoem_param):
    """
    Generates EM map based on single structure
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    INVSIG2 = cryoem_param["invsig2"]
    PREFACT = cryoem_param["prefact"]
    # prepare zero data
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    u = MDAnalysis.Universe(PDB[ipdb])
    # all-heavy selectors
    allheavy=u.select_atoms("type C O N S P")
    # now cycle on all the atoms
    for atom in allheavy:
        # get atom type
        atype = atom.type
        # get atom position
        apos = atom.position
        # get indexes in the map
        ii = int(round((apos[0]-em_origin[0])/VOX))
        jj = int(round((apos[1]-em_origin[1])/VOX))
        kk = int(round((apos[2]-em_origin[2])/VOX))
        # get delta grid
        d=delta[atype]
        # get constant parameters
        invsig2 = INVSIG2[atype]
        pref =  PREFACT[atype]
        # add contribution to map
        for k in range(max(0, kk-d), min(kk+d+1, nz)):
            distz = (em_origin[2] + float(k) * VOX - apos[2])**2
            for j in range(max(0, jj-d), min(jj+d+1, ny)):
                disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                for i in range(max(0, ii-d), min(ii+d+1, nx)):
                    # get distance squared
                    dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                    # add contribution
                    data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 / (sigma**2))
    return data


def pdb2map_avg(weights,sigma,PDBs,map_param,cryoem_param):
    """
    Generates average EM map based on structural ensemble and weights
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    delta = cryoem_param["delta"]
    INVSIG2 = cryoem_param["invsig2"]
    PREFACT = cryoem_param["prefact"]
    # prepare zero data
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    for ipdb in range(0,len(PDBs)):
        u = MDAnalysis.Universe(PDBs[ipdb])
        # get normalized weight
        w = weights[ipdb]
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S P")
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX))
            jj = int(round((apos[1]-em_origin[1])/VOX))
            kk = int(round((apos[2]-em_origin[2])/VOX))
            # get delta grid
            d=delta[atype]
            # get constant parameters
            invsig2 = INVSIG2[atype]
            pref = w * PREFACT[atype]
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 / (sigma**2))
    return data

def pdb2map_avg_chimeraX(weights,resolution,PDBs,map_param,cryoem_param):
    """
    Generates an average EM map based on structural ensemble and weights based on ChimeraX molmap
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    # prepare zero data
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    for ipdb in range(0,len(PDBs)):
        u = MDAnalysis.Universe(PDBs[ipdb])
        # get normalized weight
        w = weights[ipdb]
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S P")
        # atomic number
        at_number = {'C':6,'N':7,'O':8,'S':16,'P':15}
        # constant parameter
        maxSD = 5.0
        # sigma
        sigma = resolution*0.225
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX))
            jj = int(round((apos[1]-em_origin[1])/VOX))
            kk = int(round((apos[2]-em_origin[2])/VOX))
            # get delta grid
            d=int(np.ceil(maxSD * sigma / VOX))
            # prefactor
            pref = at_number[atype] / np.sqrt(2.0*np.pi) / sigma
            # inverted sigma
            invsig2 = 1.0 / sigma / sigma
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += w * pref * np.exp(-0.5 * dist * invsig2)
    return data

def add_noise(map, noise):
    """
    Adding noise with mean = 0 and std equal to X% of the maximum value of the map
    """
    max = np.max(map)
    threshold = 3*noise*max
    noise_v = np.random.normal(0,noise*max,map.size)
    noise_m = np.reshape(noise_v,np.shape(map))
    return map + noise_m


def write_map(map,map_name,map_param):
    """
    Writing cryoEM map
    """
    nx = map_param["nx"]
    ny = map_param["ny"]
    nz = map_param["nz"]
    VOX = map_param["vox"]
    em_origin = map_param["em_origin"]
    map_32 = map.astype(np.float32)
    mrc = mrcfile.new(map_name,overwrite=True)
    mrc.header.nx=nx; mrc.header.ny=ny; mrc.header.nz=nz
    mrc.header.cella.x=float(nx)*VOX; mrc.header.cella.y=float(ny)*VOX; mrc.header.cella.z=float(nz)*VOX
    mrc.header.origin.x=em_origin[0]; mrc.header.origin.y=em_origin[1]; mrc.header.origin.z=em_origin[2]
    mrc.set_data(map_32)
    mrc.close()


def mask_sim_gen(sim_em_data,N_models):
    """
    Generates mask based on structural ensemble
    """
    mask_x = []
    mask_y = []
    mask_z = []
    threshold = np.max(sim_em_data)*0.1
    for i in range(0,N_models):
        mask_x+=np.where(sim_em_data[i]>threshold)[0].tolist()
        mask_y+=np.where(sim_em_data[i]>threshold)[1].tolist()
        mask_z+=np.where(sim_em_data[i]>threshold)[2].tolist()
    mask_sim=np.array([mask_x,mask_y,mask_z])
    mask_sim_uniq=np.unique(mask_sim,axis=1)
    return mask_sim_uniq

def combine_masks(mask_exp,mask_sim):
    """
    Combines masks from exp and MD ensemble
    """
    mask_x=[]
    mask_y=[]
    mask_z=[]
    mask_x=mask_exp[0].tolist()+mask_sim[0].tolist()
    mask_y=mask_exp[1].tolist()+mask_sim[1].tolist()
    mask_z=mask_exp[2].tolist()+mask_sim[2].tolist()
    mask_final=np.array([mask_x,mask_y,mask_z])
    mask_final_uniq=(np.unique(mask_final,axis=1)[0],np.unique(mask_final,axis=1)[1],np.unique(mask_final,axis=1)[2])
    return mask_final_uniq

