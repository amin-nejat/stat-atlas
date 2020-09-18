# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:31:11 2020

@author: Amin
"""
import pandas as pd
import numpy as np
from . import Helpers
import matplotlib.pyplot as plt

class Atlas:
    iter        = 10
    min_counts  = 2
    epsilon     = 1e-3
    
    @staticmethod
    def visualize(atlas, aligned, title_str):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
#        ax.set_aspect('equal')
        
        ax.set_title(title_str)
#        set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1]);

        atlas_color = atlas['mu'][:,3:]
        atlas_color[atlas_color<0] = 0
        atlas_color = atlas_color/atlas_color.max()
        
        sizes = 10*np.array([np.linalg.eig(atlas['sigma'][:3,:3,i])[0].sum() for i in range(atlas['sigma'].shape[2])])

        ax.scatter(atlas['mu'][:,0],atlas['mu'][:,1],atlas['mu'][:,2], s=sizes, facecolors=atlas_color, edgecolors=None, marker='o',linewidth=3)
        for i in range(len(atlas['names'])):
            ax.text(atlas['mu'][i,0],atlas['mu'][i,1],atlas['mu'][i,2],atlas['names'][i],c='r')
#        text(model['mu'](:,1),model['mu'](:,2),model['mu'](:,3), atlas.(lower(bodypart)).N, 'Color', 'r');

#        axis equal; axis tight; grid on; set(gca,'color',[0.7,0.7,0.7]); view(-30,-20);
#        xticks({}); yticks({}); zticks({}); 
#        xticklabels({}); yticklabels({}); zticklabels({});
#
        for j in range(len(aligned)):
            al = aligned[j]
            c_j = al[:,3:6]
            c_j[c_j < 0] = 0
            ax.scatter(al[:,0],al[:,1],al[:,2], s=20, c=c_j/c_j.max(),marker='.');
        

#    def major_axis_align(model,aligned):
#        [coeff_mu,~,~,~,~,mu_mu] = pca(model['mu'](:,[1 2 3]));
#        
#        coeff_mu = coeff_mu*det(coeff_mu);
#        model['mu'](:,[1 2 3]) = (model['mu'](:,[1 2 3]) - mu_mu)*coeff_mu;
#        
#        for i=1:size(model['sigma'],3)
#            model['sigma']([1 2 3],[1 2 3],i) = coeff_mu'*model['sigma']([1 2 3],[1 2 3],i)*coeff_mu;
#        
#        for j=1:size(aligned,3)
#            aligned(:,[1,2,3],j) = (aligned(:,[1,2,3],j)-mu_mu)*coeff_mu;
    @staticmethod
    def update_beta(X,model):
#        % allocating memory
        beta    = np.zeros((X.shape[1],X.shape[1],X.shape[2]))
        beta0   = np.zeros((1,X.shape[1],X.shape[2]))
        aligned = np.zeros(X.shape)
        
        params = {}
        
        C = X.shape[1]-3
        
#        % computing beta for each training worm based using multiple
#        % covariance regression solver
        
        cost = [0,0]
        for j in range(X.shape[2]):
            idx = ~np.isnan(X[:,:,j]).all(1)
            
#            % solving for positions
            R = Helpers.MCR_solver(model['mu'][idx,:3], \
                np.concatenate((X[idx,:3,j], np.ones((idx.sum(), 1)) ),1), \
                model['sigma'][:3,:3,:])

            beta[:3,:3,j] = np.linalg.inv(R[:3,:3])
            beta0[:,:3,j] = -R[3,np.newaxis]@beta[:3,:3,j]
            
#            % solving for colors
            R = Helpers.MCR_solver(model['mu'][idx,3:], \
                np.concatenate((X[idx,3:,j], np.ones((idx.sum(),1)) ), 1), \
                model['sigma'][3:,3:,:])

            beta[3:,3:,j] = np.linalg.inv(R[:C,:C])
            beta0[:,3:,j] = -R[C,np.newaxis]@beta[3:,3:,j]
            
            aligned[:,:,j] = (X[:,:,j]-beta0[:,:,j])@np.linalg.inv(beta[:,:,j])
            
            cost[0] = cost[0] + np.sqrt(((aligned[idx,:3,j]-model['mu'][idx,:3])**2).sum(1)).sum()
            cost[1] = cost[1] + np.sqrt(((aligned[idx,3:,j]-model['mu'][idx,3:])**2).sum(1)).sum()
        
        params['beta'] = beta
        params['beta0'] = beta0
        
#        print(cost)
        
        return params, aligned
    
    @staticmethod
    def initialize_atlas(col,pos):
		
#        % memory allocation
        params = {}
        model = {}
        cost    = np.zeros((pos.shape[2],pos.shape[2]))
        aligned = [np.zeros((pos.shape[0], pos.shape[1]+col.shape[1],pos.shape[2]))]*pos.shape[2]
        
#        % alignment of samples to best fit worm
        for i in range(pos.shape[2]):
            for j in range(pos.shape[2]):
                S0,R0,T0 = Helpers.scaled_rotation(pos[:,:,i],pos[:,:,j])
                cost[i,j] = np.sqrt(np.nanmean(np.nansum((pos[:,:,i]@(R0*S0)+T0-pos[:,:,j])**2,1),0))
                aligned[j][:,:3,i] = pos[:,:,i]@(R0*S0)+T0
                aligned[j][:,3:,i] = col[:,:,i]
        
        jidx = np.argmin(cost.sum(0))
        X = aligned[jidx]

#        % initializations
        params['beta']  = np.tile(np.eye(X.shape[1]),(1,1,X.shape[2]))
        params['beta0'] = np.tile(np.zeros((1,X.shape[1])),(1,1,X.shape[2]))
        
        model['mu']     = np.nanmean(X,2)          
        
        return model,params,X
    
    @staticmethod
    def estimate_mu(mu,aligned):
#		% Amin and Erdem
		
#        % eigen-push to preserve the volume, computing singular values
#        % before the update
        _,Sp,_ = np.linalg.svd(mu[:,:3]-mu[:,:3].mean(0))
        _,Sc,_ = np.linalg.svd(mu[:,3:]-mu[:,3:].mean(0))
        
#        % computing the means
        mu = np.nanmean(aligned,2)
        
#        % updaing using singular vectors of new means and singular
#        % values of the old means
        Up,_,Vp = np.linalg.svd(mu[:,:3]-mu[:,:3].mean(0),full_matrices=False)
        mu[:,:3] = Up@np.diag(Sp)@Vp + mu[:,:3].mean(0)
        
        Uc,_,Vc = np.linalg.svd(mu[:,3:]-mu[:,3:].mean(0),full_matrices=False)
        mu[:,3:] = Uc@np.diag(Sc)@Vc + mu[:,3:].mean(0)
        
        return mu
        
    @staticmethod
    def estimate_sigma(mu,aligned):
#        % memory allocation
        sigma   = np.zeros((mu.shape[1],mu.shape[1],mu.shape[0]))
        
#        % computing the covariances
        for i in range(aligned.shape[0]):
            sigma[:,:,i] = pd.DataFrame((aligned[i,:,:] - mu[i,:][:,np.newaxis]).T).cov().to_numpy()
        
#        % well-condition the sigmas by adding epsilon*I
        sigma = sigma + Atlas.epsilon*np.tile(np.eye(sigma.shape[0])[:,:,np.newaxis],(1,1,sigma.shape[2]))
        
        for i in range(aligned.shape[0]):
#            % decorrelate color and position 
            sigma[:3,3:,i] = 0
            sigma[3:,:3,i] = 0
            
#            % diagonalize color covariances
            sigma[3:,3:,i] = np.diag(np.diag(sigma[3:,3:,i]))
            
        return sigma
    
#    def cross_validation(ims,varargin):
#        accuracy = zeros(1,length(ims));
#        
#        for i=1:length(ims)
#            % test image
#            im_test = Neurons.Image(ims(i).to_superpixel, ims(i).bodypart, 'scale', ims(i).scale);
#            for neuron=1:length(im_test.neurons)
#                im_test.neurons(neuron).annotation = '';
#            end
#            
#            % train images
#            im_train = ims([1:i-1,i+1:length(ims)]);
#            atlas = AutoIdTrain.train_atlas(im_train);
#            
#            % visualizing the trained atlas
#            if any(strcmp(varargin,'visualize'))
#                AutoIdTrain.visualize(atlas, ims([1:i-1,i+1:length(ims)]), ...
#                    ims(1).bodypart, ['Atlas and Worms - ', num2str(i)]);
#            
#            % auto_id the test image and compute the accuracy
#            Methods.AutoId.instance.id(num2str(i), im_test, 'atlas', atlas);
#            accuracy(i) = mean(strcmp(ims(i).get_annotations(), im_test.get_deterministic_ids()'));
#            accuracy(i)
#            
#        return accuracy
    
    @staticmethod
    def train_atlas(annotations,scales,positions,colors,bodypart):
#        % reading the annotations
        N = list(set([item for sublist in annotations for item in sublist]))
        C = colors[0].shape[1]
#        % allocationg memory for color and position
        pos = np.zeros((len(N),3,len(annotations)))*np.nan
        col = np.zeros((len(N),C,len(annotations)))*np.nan
        
#        % re-ordering colors and positions to have the same neurons in
#        % similar rows
        for j in range(len(annotations)):
            perm = np.array([N.index(x) for x in annotations[j]])
            pos[perm,:,j] = positions[j]*scales[j][np.newaxis,:]
            col_tmp = colors[j]
            col[perm,:,j] = col_tmp
        
#        % computing the number of worms with missing data for each
#        % neuron
        counts = (~np.isnan(pos.sum(1))).sum(1)
#        % filtering the neurons based on min_count of the missing data
        good_indices = np.logical_and( counts>Atlas.min_counts, 
                                      ~np.array([x == '' or x == None for x in N]))
        pos = pos[good_indices ,:,:]
        col = col[good_indices ,:,:]
        
        N = [N[i] for i in range(len(good_indices)) if good_indices[i]]
        
#        % initialization
        model,params,aligned = Atlas.initialize_atlas(col,pos);
        init_aligned = aligned
        
        for iteration in range(Atlas.iter):
#            % updating means.
            model['mu'] = Atlas.estimate_mu(model['mu'],aligned)
            
#            % updating sigma
            model['sigma'] = Atlas.estimate_sigma(model['mu'],aligned)

#            % updating aligned
            params,aligned = Atlas.update_beta(init_aligned,model)
            
        
#        % standardize the atlas by axis aligning it
#        [model,aligned] = AutoIdTrain.major_axis_align(model,aligned);
        
#        % store the result for the output
        atlas = {'bodypart':bodypart, \
                      'mu': model['mu'], \
                      'sigma': model['sigma'], \
                      'names': N}
#        % store worm specific parameters inside their corresponding
#        % class
        aligned_coord = []
        for j in range(len(annotations)):
            perm = np.array([N.index(x) if x in N else -1 for x in annotations[j]])
            aligned_coord.append(np.array([aligned[perm[n],:,j] if perm[n] != -1 else np.zeros((C+3))*np.nan for n in range(len(perm))]))
            
        return atlas,aligned_coord
    
    @staticmethod
    def train_distance_atlas(annotations,scales,positions,colors,bodypart):
#        reading the annotations
        N = list(set([item for sublist in annotations for item in sublist]))
        N.sort()
        
        C = colors[0].shape[1]
#        allocationg memory for color and position
        pos = np.zeros((len(N),3,len(annotations)))*np.nan
        col = np.zeros((len(N),C,len(annotations)))*np.nan
        
#        re-ordering colors and positions to have the same neurons in
#        similar rows
        for j in range(len(annotations)):
            perm = np.array([N.index(x) for x in annotations[j]])
            pos[perm,:,j] = positions[j]*scales[j][np.newaxis,:]
            col_tmp = colors[j]
            col[perm,:,j] = col_tmp
        
        counts = (~np.isnan(pos.sum(1))).sum(1)
#        filtering the neurons based on min_count of the missing data
        good_indices = np.logical_and( counts>Atlas.min_counts, 
                                      ~np.array([x == '' or x == None for x in N]))
        pos = pos[good_indices ,:,:]
        col = col[good_indices ,:,:]
        
        N = [N[i] for i in range(len(good_indices)) if good_indices[i]]
        
        D1 = np.zeros((len(N),len(N)))
        D2 = np.zeros((len(N),len(N)))
        
        C1 = np.zeros((len(N),len(N)))
        C2 = np.zeros((len(N),len(N)))
        
        for i in range(len(N)):
            for j in range(len(N)):
                D1[i,j] = np.nanmean(np.sqrt(((pos[i,:,:] - pos[j,:,:])**2).sum(0)))
                C1[i,j] = np.nanstd(np.sqrt(((pos[i,:,:] - pos[j,:,:])**2).sum(0)))
                
                D2[i,j] = np.nanmean(np.sqrt(((col[i,:,:] - col[j,:,:])**2).sum(0)))
                C2[i,j] = np.nanstd(np.sqrt(((col[i,:,:] - col[j,:,:])**2).sum(0)))
                
        atlas = {'bodypart':bodypart,
                  'C1': C1,
                  'D1': D1,
                  'C2': C2,
                  'D2': D2,
                  'names': N}
        
        return atlas