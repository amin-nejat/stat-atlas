# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:13:09 2020

@author: Amin
"""

from scipy.io import loadmat
from neurons import Neuron, Image

import numpy as np
import scipy as sp

# %%
def loat_atlas(file,bodypart):
    """Load worm's atlas file
        
    Args:
        file (std): File address of the file to be loaded
        bodypart (str): Worm's body part ('head','tail')
    
    Returns:
        mu (np.ndarray): Nx(3+C) where N is the number of neurons and C 
            is the number of color channles.  Each row corresponds to the 
            position and color of one neuron
        sigma (np.ndarray): Nx(3+C)x(3+C) covariances
        names (list): String array containing the names of the neurons
        bodypart (std): Same as the input bodypart
    """
    
    content = loadmat(file,simplify_cells=True)
    
    mu = content['atlas'][bodypart]['model']['mu']
    sigma = content['atlas'][bodypart]['model']['sigma']
    names = content['atlas'][bodypart]['N']
    
    mu[:,:3] = mu[:,:3] - 1 # Matlab to Python
    
    atlas = {
        'mu':mu,
        'sigma':sigma,
        'names': names,
        'bodypart':bodypart
    }
    return atlas

# %%
def simulate_gmm(atlas,n_samples=10):
    """Simulate samples from atlas by sampling from Normal distributions and
    randomly rotating
        
    Args:
        atlas (dict): Pre-trained statistical atlas
        n_samples (int): Number of worms to be generated
        
    Returns:
        samples (np.ndarray): Positions and colors of the worms (N,3+C,K)
    """
    
    # Sampling data
    C = atlas['mu'].shape[1]-3 # Number of colors
    K = atlas['mu'].shape[0] # Number of components

    mu = np.zeros((K,C+3,n_samples))
    for k in range(K):

        R = sp.spatial.transform.Rotation.random().as_matrix()
        sample = sp.stats.multivariate_normal.rvs(
            mean=atlas['mu'][k], 
            cov=atlas['sigma'][...,k],
            size=n_samples
        ).T

        sample[:3] = R@sample[:3]
        mu[k,:,:] = sample
    
    
    #  Creating Images
    ims = []
    for n in range(n_samples):
        neurons = []
        for k in range(len(atlas['names'])):
            neuron = Neuron.Neuron()
             # Neuron position & color
            neuron.position = mu[k,:3,n]
            neuron.color = mu[k,3:,n]
            neuron.color_readout = mu[k,3:,n]
            
            # User neuron ID
            neuron.annotation = atlas['names'][k] 
            neuron.annotation_confidence = .99
            
            neurons.append(neuron)
            
        im = Image.Image(atlas['bodypart'],neurons)
        ims.append(im)
        
    return ims

