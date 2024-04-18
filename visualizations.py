# -*- coding: utf-8 -*-
"""
Created on Mon Apr  18 10:07:27 2024

@author: Amin
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from skimage import color
from scipy.optimize import minimize

import copy
from scipy.stats import multivariate_normal

import scipy as sp
# %%
def visualize(
        atlas,
        aligned,
        title_str,
        fontsize=9,
        dotsize=30,
        save=False,
        file=None
    ):
    """Visualize trained atlas and aligned point clouds
    """
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(title_str)

    atlas_color = atlas['mu'][:,3:]
    atlas_color[atlas_color<0] = 0
    atlas_color = atlas_color/atlas_color.max()
    if atlas_color.shape[1] == 4:
        atlas_color[:,3] = 0
    
    sizes = 1*np.array([np.linalg.eig(atlas['sigma'][:3,:3,i])[0].sum() for i in range(atlas['sigma'].shape[2])])
    
    ax.scatter(atlas['mu'][:,0],atlas['mu'][:,1],atlas['mu'][:,2], s=dotsize, facecolors=atlas_color[:,:3], marker='.')
    ax.scatter(atlas['mu'][:,0],atlas['mu'][:,1],atlas['mu'][:,2], s=sizes, facecolors=atlas_color, edgecolors=atlas_color[:,:3], marker='o',linewidth=1)
    
    
    for i in range(len(atlas['names'])):
        ax.text(atlas['mu'][i,0],atlas['mu'][i,1],atlas['mu'][i,2],atlas['names'][i],c=atlas_color[i,:3],fontsize=fontsize)

    for j in range(len(aligned)):
        al = aligned[j]
        c_j = al[:,3:6]
        c_j[c_j < 0] = 0
        ax.scatter(al[:,0],al[:,1],al[:,2], s=10, c=c_j/c_j.max(),marker='.');
    
    ax.set_ylim([35+atlas['mu'][:,1].min(),-35+atlas['mu'][:,1].max()])
    Atlas.axis_equal(ax)
    ax.view_init(elev=90., azim=10)
    
    
    ax.axis('off')
    ax.set_facecolor('xkcd:light gray')
    fig.patch.set_facecolor('xkcd:light gray')
    
    if save:
        plt.savefig(file+'.png',format='png')
        try:
            plt.savefig(file+'.pdf',format='pdf')
        except:
            pass
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_pretty(
        atlas,
        aligned,
        title_str,
        projection=None,
        fontsize=8,
        dotsize=12,
        save=False,
        file=None,
        olp=False, # optimal label positioning
        tol=1e0,
        plot_cov=True,
        hsv_correction=False,
        connect_pairs=None,
        alpha=1,
        labels=None
    ):
    """Visualize trained atlas and aligned point clouds
    """
    
    fig = plt.figure(figsize=(18,4))
    ax = fig.add_subplot(111)
    ax.set_title(title_str)
    atlas_color = atlas['mu'][:,3:].copy()
    
    atlas_color[atlas_color<0] = 0
    atlas_color = atlas_color/(np.percentile(atlas_color,95,axis=0)+1e-5)
    atlas_color[atlas_color>1] = 1
    
    if hsv_correction:
        hsv = color.rgb2hsv(atlas_color[None,:,:3]).squeeze()
        hsv[:,2] = 1
        atlas_color[:,:3] = color.hsv2rgb(hsv[None,:,:]).squeeze()
    
    if atlas_color.shape[1] == 4:
        atlas_color[:,3] = alpha
    
    if projection is None:
        pca = PCA(n_components=2)
        pca.fit(atlas['mu'][:,:3])
        projection = pca.components_.T
    
    mus = atlas['mu'][:,:3].copy()@projection
    cov = np.zeros((2,2,mus.shape[0]))
    for i in range(cov.shape[2]):
        cov[:,:,i] = projection.T@atlas['sigma'][:3,:3,i].copy()@projection
        
    samples = [[]]*len(aligned)
    for i in range(len(aligned)):
        samples[i] = aligned[i][:,:3]@projection
        
    if plot_cov:
        for i in range(cov.shape[2]):
            draw_ellipse(
                mus[i,:],
                cov[:,:,i],
                atlas_color[i,:3][None,:],
                std_devs=1.5,ax=ax,line_width=2
            )

    ax.scatter(mus[:,0],mus[:,1],facecolors=atlas_color, s=300,
                edgecolors='k',marker='.',linewidth=1)
    
    if olp:
        if labels is None:
            label_coor = optimal_label_positioning(mus[:,:2],tol=tol)
        else:
            label_coor = mus[:,:2].copy()
            label_coor[labels,:] = optimal_label_positioning(mus[labels,:2],tol=tol)
    else:
        label_coor = mus[:,:2].copy()
    
    for i in range(len(atlas['names'])):
        if labels is None or labels is not None and labels[i]:
            ax.text(
                label_coor[i,0],
                label_coor[i,1],
                atlas['names'][i],
                c=atlas_color[i,:3],
                fontsize=fontsize
            )
        ax.plot(
            [label_coor[i,0], mus[i,0]],
            [label_coor[i,1], mus[i,1]],
            color=atlas_color[i,:3],
            linestyle='dotted',
            linewidth=1
        )
    ax.set_xlim(label_coor.min(0)[0]-5,label_coor.max(0)[0]+5)
    ax.set_ylim(label_coor.min(0)[1]-5,label_coor.max(0)[1]+5)
    
    if connect_pairs is not None:
        for pair in connect_pairs:
            ax.plot(
                [mus[pair[0],0], mus[pair[1],0]],
                [mus[pair[0],1], mus[pair[1],1]],
                color='k',
                linestyle='dotted',
                linewidth=1
            )
    
    for j in range(len(samples)):
        c_j = aligned[j][:,3:6]
        c_j[c_j < 0] = 0
        c_j = c_j/c_j.max()
        ax.scatter(
            samples[j][:,0],
            samples[j][:,1], 
            s=dotsize, 
            facecolors=c_j, 
            edgecolors=c_j, 
            marker='.'
        )
    
    ax.set_aspect('equal',adjustable='box')
    
    ax.set_facecolor('xkcd:light gray')
    fig.patch.set_facecolor('xkcd:light gray')
    fig.tight_layout()

    if save:
        plt.savefig(file+'.png',format='png')
        try: plt.savefig(file+'.pdf',format='pdf')
        except: pass
        plt.close('all')
    else:
        plt.show()

# %%
def optimal_label_positioning(mu,lambda_1=5,lambda_2=5,tol=1e0):
    '''spring parameter (these were selected for the head of the worm, 
        may need to play around with it for more dense plots) minimum 
        separation parameter
    '''
        
    # Kamada-kawai loss function
    n = mu.shape[0] # the 2D neuron mean positions
    D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(mu))
    ll = np.random.rand(n,1) # some random springyness to make the plot look "organic"
    ll = lambda_1*ll*ll.T + lambda_2
    L = np.vstack((
        np.hstack((D,D+ll-np.diag(np.diag(D+ll))+np.diag(ll))),
        np.hstack((D+ll-np.diag(np.diag(D+ll))+np.diag(ll),D-np.diag(np.diag(D+ll))+2*ll))
        ))
    K = 1/(L**2)
    k = 2*K
    l = L
    
    myfun = lambda x: kk_cost(mu,x.reshape(mu.shape),k,l)

    # optimization        
    res = minimize(myfun, mu.reshape(-1), method='L-BFGS-B', tol=tol)
    
    # output annotation coordinates for each mu
    coor = res.x.reshape(mu.shape)# output annotation position
    return coor

# %%
def kk_cost(mu,coor,k,l):
    '''kamada-kawai force directed graph cost function
    '''
    y = np.vstack((mu,coor))
    
    pdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(y))
    cost = np.triu(k*(pdist-l)**2,1).sum()
    
    return cost

# %%
def axis_equal(ax):
    """Equalize axes of a 3D plot
    """
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


# %%
def major_axis_align(atlas,aligned,params,shift=0):
    pca = PCA(n_components=3)
    pca.fit(atlas['mu'][:,:3])
    projection = pca.components_.T
    projection = projection*np.linalg.det(projection)
    projection = projection[:,[1,0,2]]
    
    shift = -(atlas['mu'][:,:3]@projection).min(0)+shift
    
    mus = atlas['mu'][:,:3]@projection + shift
    cov = np.zeros((3,3,mus.shape[0]))
    for i in range(cov.shape[2]):
        cov[:,:,i] = projection.T@atlas['sigma'][:3,:3,i]@projection
        
    
    proj_atlas = copy.deepcopy(atlas)
    proj_atlas['mu'][:,:3] = mus
    proj_atlas['sigma'][:3,:3,:] = cov
    
    proj_params = copy.deepcopy(params)
    for i in range(params['beta'].shape[2]):
        proj_params['beta'][:3,:3,i] = proj_params['beta'][:3,:3,i]@projection
        proj_params['beta0'][:,:3,i] = proj_params['beta0'][:,:3,i]@projection+shift
    
    proj_aligned = copy.deepcopy(aligned)
    for i in range(len(aligned)):
        proj_aligned[i][:,:3] = aligned[i][:,:3]@projection+shift
    
    return proj_atlas,proj_aligned,proj_params

# %%
def draw_ellipse(mean,covariance,color,std_devs=3,ax=None,line_width=2):
    # sample grid that covers the range of points
    min_p = mean - std_devs*np.sqrt(np.diag(covariance))
    max_p = mean + std_devs*np.sqrt(np.diag(covariance))
    
    x = np.linspace(min_p[0],max_p[0],256) 
    y = np.linspace(min_p[1],max_p[1],256)
    X,Y = np.meshgrid(x,y)
    
    Z = multivariate_normal.pdf(
        np.stack((X.reshape(-1),Y.reshape(-1))).T,
        mean=mean,
        cov=(std_devs**2)*covariance
    )
    Z = Z.reshape([len(x),len(y)])
    
    if ax is None: plt.contour(X, Y, Z, 0,colors=color,linewidth=line_width)
    else: ax.contour(X, Y, Z, 0,colors=color,linewidths=line_width)
