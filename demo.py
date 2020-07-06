# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:09:27 2020

@author: Amin
"""

from Atlas.Atlas import Atlas
from AUtils import Simulator
import numpy as np

# %%

bodypart    = 'tail'
n_samples   = 100

atlas       = Simulator.loat_atlas('atlas.mat',bodypart)
samples     = Simulator.simulate_gmm(atlas,n_samples=n_samples)

trained_atlas, aligned = Atlas.train_atlas([atlas['names'] for i in range(n_samples)],
                                         [np.array([1,1,1]) for i in range(n_samples)],
                                         [sample['mu'][:,:3] for sample in samples],
                                         [sample['mu'][:,3:] for sample in samples],
                                         bodypart)

Atlas.visualize(trained_atlas,aligned,'')