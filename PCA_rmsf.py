import MDAnalysis
import numpy as np
from MDAnalysis import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def get_rmsf(PDBs,random_pdbs,weights):
    # Load the ensemble of PDB files
    universe_list = [Universe(pdb) for pdb in PDBs]
    universe_selected_list = [Universe(pdb) for pdb in random_pdbs]
    # Extract coordinates
    ca_atoms = "name CA"
    coordinates = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_list])
    coordinates_selected = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_selected_list])
    n_frames, n_atoms, dim = coordinates.shape
    n_selected_frames, n_atoms, dim = coordinates_selected.shape
    # Compute mean structure prior and posterior
    start_weights = np.ones(n_frames)/n_frames
    target_weights = np.ones(n_selected_frames)/n_selected_frames
    mean_structure_target = np.average(coordinates_selected, weights = target_weights, axis=0)
    mean_structure_prior = np.average(coordinates, weights = start_weights, axis=0)
    mean_structure_post = np.average(coordinates, weights = weights, axis=0)
    # Calculate RMSF prior and posterior
    rmsf_target = np.sqrt(np.average(np.square(coordinates_selected - mean_structure_target), weights = target_weights, axis=0).sum(axis=1))
    rmsf_prior = np.sqrt(np.average(np.square(coordinates - mean_structure_prior), weights = start_weights, axis=0).sum(axis=1))
    rmsf_post = np.sqrt(np.average(np.square(coordinates - mean_structure_post), weights = weights, axis=0).sum(axis=1))
    return rmsf_target, rmsf_prior, rmsf_post

def get_rmsf_sel(PDBs,selected_pdbs,weights):
    # Load the ensemble of PDB files
    universe_selected_list = [Universe(pdb) for pdb in selected_pdbs]
    # Extract coordinates
    ca_atoms = "name CA"
    coordinates_selected = np.array([universe.select_atoms(ca_atoms).positions for universe in universe_selected_list])
    n_selected_frames, n_atoms, dim = coordinates_selected.shape
    # Compute mean structure prior and posterior
    mean_structure_post = np.average(coordinates_selected, weights = weights, axis=0)
    # Calculate RMSF prior and posterior
    rmsf_post_sel = np.sqrt(np.average(np.square(coordinates_selected - mean_structure_post), weights = weights, axis=0).sum(axis=1))
    return rmsf_post_sel


def do_PCA(PDBs,random_pdbs,weights):
    # Load the ensemble of PDB files
    universe_list = [Universe(pdb) for pdb in PDBs]
    universe_selected_list = [Universe(pdb) for pdb in random_pdbs]
    # Extract coordinates
    backbone_atoms = "name N CA C O"
    n_frames = len(PDBs)
    n_selected_frames = 10
    start_weights = np.ones(n_frames)/n_frames
    target_weights = np.ones(n_selected_frames)/n_selected_frames
    coordinates = np.array([universe.select_atoms(backbone_atoms).positions for universe in universe_list])
    coordinates_selected = np.array([universe.select_atoms(backbone_atoms).positions for universe in universe_selected_list])
    # Perform weighted PCA
    # Compute centered structures
    center_target = coordinates_selected - np.average(coordinates_selected, weights = target_weights ,axis=0)
    center_prior = coordinates - np.average(coordinates, weights = start_weights, axis=0)
    center_post = coordinates - np.average(coordinates, weights = weights, axis=0)
    # Calculate weighted centered coordinates
    coord_w_target = center_target.reshape(n_selected_frames, -1) * np.sqrt(target_weights)[:,np.newaxis]
    coord_w_prior = center_prior.reshape(n_frames, -1) * np.sqrt(start_weights)[:,np.newaxis]
    coord_w_post = center_post.reshape(n_frames, -1) * np.sqrt(weights)[:,np.newaxis]
    # PCA
    pca_target = PCA(n_components=3)
    pca_comp_target = pca_target.fit_transform(coord_w_target)
    eigenvectors_target = pca_target.components_
    pca_prior = PCA(n_components=3)
    pca_comp_prior = pca_prior.fit_transform(coord_w_prior)
    eigenvectors_prior = pca_prior.components_
    pca_post = PCA(n_components=3)
    pca_comp_post = pca_post.fit_transform(coord_w_post)
    eigenvectors_post = pca_post.components_
    return eigenvectors_target, eigenvectors_prior, eigenvectors_post


def do_PCA_sel(PDBs,selected_pdbs,weights):
    # Load the ensemble of PDB files
    universe_selected_list = [Universe(pdb) for pdb in selected_pdbs]
    # Extract coordinates
    backbone_atoms = "name N CA C O"
    n_selected_frames = len(selected_pdbs)
    coordinates_selected = np.array([universe.select_atoms(backbone_atoms).positions for universe in universe_selected_list])
    # Perform weighted PCA
    # Compute centered structures
    center_post_sel = coordinates_selected - np.average(coordinates_selected, weights = weights, axis=0)
    # Calculate weighted centered coordinates
    coord_w_post_sel = center_post_sel.reshape(n_selected_frames, -1) * np.sqrt(weights)[:,np.newaxis]
    # PCA
    pca_post_sel = PCA(n_components=3)
    pca_comp_post_sel = pca_post_sel.fit_transform(coord_w_post_sel)
    eigenvectors_post_sel = pca_post_sel.components_
    return eigenvectors_post_sel
