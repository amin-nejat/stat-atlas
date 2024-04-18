# %%
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:09:27 2020

@author: Amin
"""

from models import Atlas
import simulator
import visualizations

# %%
bodypart = 'tail'
n_samples = 10

# %% Load the pre-trained atlas
atlas = simulator.loat_atlas( 
    'data/atlas_xx_rgb.mat',
    bodypart
) 

# %% Simulate worms from the generative model
ims = simulator.simulate_gmm(
    atlas,
    n_samples=n_samples
)

# %% Train atlas on the sample worms
atlas_obj = Atlas(epsilon=[.1,.1])
trained_atlas, aligned, params, _, _ = atlas_obj.train_atlas(
    ims,
    bodypart,
    n_iter=100
) 

# %% Align the major axis for visualization
trained_atlas, aligned, params = visualizations.major_axis_align(
    trained_atlas, 
    aligned, 
    params, 
    shift=10
)

# %% Visualize atlas and aligned point clouds
visualizations.visualize_pretty(trained_atlas,aligned,'')
