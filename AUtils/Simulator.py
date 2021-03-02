# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:13:09 2020

@author: Amin
"""
import pyro.distributions as dist
import numpy as np
import torch
import pyro
from scipy.io import loadmat
from scipy.stats import ortho_group

def loat_atlas(file,bodypart):
    """Load C. elegans atlas file
        
    Args:
        file (string): File address of the file to be loaded
        bodypart (string): The bodypart that the atlas is requested for, choose
            between ('head','tail')
    
    Returns:
        dict: Atlas information with the following keys
            mu (numpy.ndarray): Nx(3+C) where N is the number of neurons and C 
                is the number of channles; each row corresponds to the center 
                and color of one neuron in the canonical space
            sigma (numpy.ndarray): Nx(3+C)x(3+C) where N is the number of neurons 
                and C is the number of channles; each row corresponds to the covariance
                of the position and color of one neuron in the canonical space
            names (array): String array containing the names of the neurons
            bodypart (string): Same as the input bodypart
    """
    
    content = loadmat(file)
    
    if bodypart == 'head':
        mu = content['atlas'][0][0][0][0][0][0][0,0][0]
        sigma = content['atlas'][0][0][0][0][0][0][0,0][1]
        names = [content['atlas'][0][0][0][0][0][1][i][0][0] for i in range(mu.shape[0])]
    elif bodypart == 'tail':
        mu = content['atlas'][0][0][1][0][0][0][0,0][0]
        sigma = content['atlas'][0][0][1][0][0][0][0,0][1]
        names = [content['atlas'][0][0][1][0][0][1][i][0][0] for i in range(mu.shape[0])]
    
    mu[:,:3] = mu[:,:3]-1 # Matlab to Python
    
    return {'mu':mu, 'sigma':sigma, 'names': names, 'bodypart':bodypart}

def simulate_gmm(atlas,n_samples=10):
    """Simulate samples from atlas by sampling from Gaussian distributions and
        transforming according to random rotations
        
    Args:
        atlas (dict): Pre-trained statistical atlas
        n_samples (integer): Number of samples (worms) to be generated
        
    Returns:
        samples (numpy.ndarray): Positions and colors of the sampled worms
            with size (N,3+C,K)
    """
    
    C       = atlas['mu'].shape[1]-3 # Number of colors
    K       = atlas['mu'].shape[0] # Number of components

    
    
    cov = np.zeros(((C+3)*(K),(C+3)*(K)))
    for n in range(K):
        cov[6*n:6*n+6,6*n:6*n+6] = atlas['sigma'][:,:,n]
        
    μ_p = torch.tensor(atlas['mu']).float()
    Σ_p = torch.tensor(cov).float()
    
    # %% Sample generative data
    samples = []
    for n in range(n_samples):
        mu = pyro.sample('µ', dist.MultivariateNormal(µ_p.reshape(-1), Σ_p)).view(µ_p.shape)
        beta = torch.zeros((C+3,C+3))
        
        beta[:3,:3] = torch.tensor(ortho_group.rvs(3)).float()
        beta[3:,3:] = torch.tensor(ortho_group.rvs(C)).float()

        samples.append({'mu':mu@beta,'beta':beta})
        
    return samples

