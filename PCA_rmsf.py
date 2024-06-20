import MDAnalysis
import numpy as np
from MDAnalysis import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def get_rmsf(u,weights):
    # Extract coordinates
    ca_atoms = "name CA"
    v = u.select_atoms(ca_atoms)
    coordinates = []
    for i in range(0,u.trajectory.n_frames):
        u.trajectory[i]
        coordinates.append(v.atoms.positions)
    coordinates_all = np.array(coordinates)
    n_frames, n_atoms, dim = coordinates_all.shape
    # Compute mean structure prior and posterior
    prior_weights = np.ones(n_frames)/n_frames
    mean_structure_prior = np.average(coordinates_all, weights = prior_weights, axis=0)
    mean_structure_post = np.average(coordinates_all, weights = weights, axis=0)
    # Calculate RMSF prior and posterior
    rmsf_prior = np.sqrt(np.average(np.square(coordinates_all - mean_structure_prior), weights = prior_weights, axis=0).sum(axis=1))
    rmsf_post = np.sqrt(np.average(np.square(coordinates_all - mean_structure_post), weights = weights, axis=0).sum(axis=1))
    return rmsf_prior, rmsf_post

def get_rmsf_sel(u,selected_frames,weights):
    # Extract coordinates
    ca_atoms = "name CA"
    v = u.select_atoms(ca_atoms)
    coordinates = []
    for i in range(0,u.trajectory.n_frames):
        u.trajectory[i]
        coordinates.append(v.atoms.positions)
    coordinates_all = np.array(coordinates)
    coordinates_selected = coordinates_all[selected_frames]
    n_selected_frames, n_atoms, dim = coordinates_selected.shape
    # Compute mean structure prior and posterior
    mean_structure_post = np.average(coordinates_selected, weights = weights, axis=0)
    # Calculate RMSF prior and posterior
    rmsf_post_sel = np.sqrt(np.average(np.square(coordinates_selected - mean_structure_post), weights = weights, axis=0).sum(axis=1))
    return rmsf_post_sel


def do_PCA(u,weights):
    # Extract coordinates
    backbone_atoms = "name N CA C O"
    v = u.select_atoms(backbone_atoms)
    coordinates = []
    for i in range(0,u.trajectory.n_frames):
        u.trajectory[i]
        coordinates.append(v.atoms.positions)
    coordinates_all = np.array(coordinates)
    n_frames, n_atoms, dim = coordinates_all.shape
    prior_weights = np.ones(n_frames)/n_frames
    # Perform weighted PCA
    # Compute centered structures
    center_prior = coordinates_all - np.average(coordinates_all, weights = prior_weights, axis=0)
    center_post = coordinates_all - np.average(coordinates_all, weights = weights, axis=0)
    # Calculate weighted centered coordinates
    coord_w_prior = center_prior.reshape(n_frames, -1) * np.sqrt(prior_weights)[:,np.newaxis]
    coord_w_post = center_post.reshape(n_frames, -1) * np.sqrt(weights)[:,np.newaxis]
    # PCA
    pca_prior = PCA(n_components=3)
    pca_comp_prior = pca_prior.fit_transform(coord_w_prior)
    eigenvectors_prior = pca_prior.components_
    pca_post = PCA(n_components=3)
    pca_comp_post = pca_post.fit_transform(coord_w_post)
    eigenvectors_post = pca_post.components_
    return eigenvectors_prior, eigenvectors_post


def do_PCA_sel(u,selected_frames,weights):
    # Extract coordinates
    backbone_atoms = "name N CA C O"
    v = u.select_atoms(backbone_atoms)
    coordinates = []
    for i in range(0,u.trajectory.n_frames):
        u.trajectory[i]
        coordinates.append(v.atoms.positions)
    coordinates_all = np.array(coordinates)
    coordinates_selected = coordinates_all[selected_frames]
    n_selected_frames = len(selected_frames)
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
