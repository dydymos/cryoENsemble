import MDAnalysis
import numpy as np
from MDAnalysis import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def get_rmsf(PDBs,w_1ake,weights):
    open_state = 1 - w_1ake
    close_state = w_1ake
    # Load the ensemble of PDB files
    universe_list = [Universe(pdb) for pdb in PDBs]
    # Extract coordinates
    ca_atoms = "name CA"
    # Number of structures from open and close conformation
    N = 50
    coordinates_close = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list])[:N]
    coordinates_open = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list])[N:]
    n_frames, n_atoms, dim = coordinates_open.shape
    # Compute mean structure prior and posterior
    open_prior_weights = np.ones(N)*open_state
    close_prior_weights = np.ones(N)*close_state
    if open_state == 0:
        mean_structure_open = np.zeros(N)
    else: mean_structure_open = np.average(coordinates_open, weights = open_prior_weights, axis=0)
    if close_state == 0:
        mean_structure_close = np.zeros(N)
    else: mean_structure_close = np.average(coordinates_close, weights = close_prior_weights,axis=0)
    mean_structure_open_post = np.average(coordinates_open, weights = weights[N:] ,axis=0)
    mean_structure_close_post = np.average(coordinates_close, weights = weights[:N], axis=0)
    # Calculate RMSF prior and posterior
    if open_state == 0:
        rmsf_open_prior = np.zeros(n_atoms)
    else: rmsf_open_prior = np.sqrt(np.average(np.square(coordinates_open - mean_structure_open), weights = open_prior_weights, axis=0).sum(axis=1))
    if close_state == 0:
        rmsf_close_prior = np.zeros(n_atoms)
    else: rmsf_close_prior = np.sqrt(np.average(np.square(coordinates_close - mean_structure_close), weights = close_prior_weights, axis=0).sum(axis=1))
    rmsf_open_post = np.sqrt(np.average(np.square(coordinates_open - mean_structure_open_post), weights = weights[N:], axis=0).sum(axis=1))
    rmsf_close_post = np.sqrt(np.average(np.square(coordinates_close - mean_structure_close_post), weights = weights[:N], axis=0).sum(axis=1))
    return rmsf_open_prior,rmsf_close_prior,rmsf_open_post,rmsf_close_post

def get_rmsf_sel(PDBs,weights,selected_frames):
    # Load the ensemble of PDB files
    w_open = np.where(selected_frames>=50)[0]
    w_close = np.where(selected_frames<50)[0]
    if (len(w_open) > 0) and (len(w_close) > 0):
        PDBs_open = np.array(PDBs)[selected_frames[w_open]]
        PDBs_close = np.array(PDBs)[selected_frames[w_close]]
        universe_list_open = [Universe(pdb) for pdb in PDBs_open]
        universe_list_close = [Universe(pdb) for pdb in PDBs_close]
        # Extract coordinates
        ca_atoms = "name CA"
        # Number of structures from open and close conformation
        coordinates_close = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list_close])
        coordinates_open = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list_open])
        n_atoms = coordinates_open.shape[1]
        # Compute mean structure posterior
        mean_structure_open_post = np.average(coordinates_open, weights = weights[w_open] ,axis=0)
        mean_structure_close_post = np.average(coordinates_close, weights = weights[w_close], axis=0)
        # Calculate RMSF prior and posterior
        rmsf_open_post = np.sqrt(np.average(np.square(coordinates_open - mean_structure_open_post), weights = weights[w_open], axis=0).sum(axis=1))
        rmsf_close_post = np.sqrt(np.average(np.square(coordinates_close - mean_structure_close_post), weights = weights[w_close], axis=0).sum(axis=1))
        return rmsf_open_post,rmsf_close_post
    return None,None

def do_PCA(PDBs,w_1ake,weights):
    # Number of structures from open or close conformation
    N = 50
    open_state = 1 - w_1ake
    close_state = w_1ake
    weights_open = np.ones(N)*open_state
    weights_close = np.ones(N)*close_state
    # Load the ensemble of PDB files
    universe_list = [Universe(pdb) for pdb in PDBs]
    # Extract coordinates
    backbone_atoms = "name N CA C O"
    coordinates_close = np.array([universe.select_atoms(backbone_atoms).positions for universe in universe_list])[:N]
    coordinates_open = np.array([universe.select_atoms(backbone_atoms).positions for universe in universe_list])[N:]
    # Perform weighted PCA
    # Compute centered structures prior and posterior
    if open_state == 0:
        center_open = np.zeros(np.shape(coordinates_open))
    else: center_open = coordinates_open - np.average(coordinates_open, weights = weights_open, axis=0)
    if close_state == 0:
        center_close = np.zeros(np.shape(coordinates_close))
    else: center_close = coordinates_close - np.average(coordinates_close, weights = weights_close,axis=0)
    center_open_post = coordinates_open - np.average(coordinates_open, weights = weights[N:] ,axis=0)
    center_close_post = coordinates_close - np.average(coordinates_close, weights = weights[:N], axis=0)
    # Calculate weighted centered coordinates
    coord_w_open = center_open.reshape(N, -1) * np.sqrt(weights_open)[:,np.newaxis]
    coord_w_close = center_close.reshape(N, -1) * np.sqrt(weights_close)[:,np.newaxis]
    coord_w_open_post = center_open_post.reshape(N, -1) * np.sqrt(weights[N:])[:,np.newaxis]
    coord_w_close_post = center_close_post.reshape(N, -1) * np.sqrt(weights[:N])[:,np.newaxis]
    # prior PCA
    pca_open_prior = PCA(n_components=3)
    pca_comp_open_prior = pca_open_prior.fit_transform(coord_w_open)
    eigenvectors_open_prior = pca_open_prior.components_
    pca_close_prior = PCA(n_components=3)
    pca_comp_close_prior = pca_close_prior.fit_transform(coord_w_close)
    eigenvectors_close_prior = pca_close_prior.components_
    # posterior PCA
    pca_open_post = PCA(n_components=3)
    pca_comp_open_post = pca_open_post.fit_transform(coord_w_open_post)
    eigenvectors_open_post = pca_open_post.components_
    pca_close_post = PCA(n_components=3)
    pca_comp_close_post = pca_close_post.fit_transform(coord_w_close_post)
    eigenvectors_close_post = pca_close_post.components_
    return eigenvectors_open_prior, eigenvectors_close_prior, eigenvectors_open_post, eigenvectors_close_post

def do_PCA_sel(PDBs,weights,selected_frames):
    w_open = np.where(selected_frames>=50)[0]
    w_close = np.where(selected_frames<50)[0]
    if (len(w_open) > 3) and (len(w_close) > 3):
        PDBs_open = np.array(PDBs)[selected_frames[w_open]]
        PDBs_close = np.array(PDBs)[selected_frames[w_close]]
        # Load the ensemble of PDB files
        universe_list_open = [Universe(pdb) for pdb in PDBs_open]
        universe_list_close = [Universe(pdb) for pdb in PDBs_close]
        # Extract coordinates
        ca_atoms =  "name N CA C O"
        # Number of structures from open and close conformation
        coordinates_close = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list_close])
        coordinates_open = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list_open])
        # Perform weighted PCA
        # Compute centered posterior
        center_open_post = coordinates_open - np.average(coordinates_open, weights = weights[w_open] ,axis=0)
        center_close_post = coordinates_close - np.average(coordinates_close, weights = weights[w_close], axis=0)
        # Calculate weighted centered coordinates
        coord_w_open_post = center_open_post.reshape(len(PDBs_open), -1) * np.sqrt(weights[w_open])[:,np.newaxis]
        coord_w_close_post = center_close_post.reshape(len(PDBs_close), -1) * np.sqrt(weights[w_close])[:,np.newaxis]
        # posterior PCA
        pca_open_post = PCA(n_components=3)
        pca_comp_open_post = pca_open_post.fit_transform(coord_w_open_post)
        eigenvectors_open_post = pca_open_post.components_
        pca_close_post = PCA(n_components=3)
        pca_comp_close_post = pca_close_post.fit_transform(coord_w_close_post)
        eigenvectors_close_post = pca_close_post.components_
        return eigenvectors_open_post, eigenvectors_close_post
    return None, None
