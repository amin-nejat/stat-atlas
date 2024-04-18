# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:44:06 2020

@author: Amin
"""

import numpy as np

# %%
def MCR_solver(X,Y,sigma):
    """MCR - Multiple covariance regression solver
    Solving: \sum_i \|Y_i - X_i \beta\|_{sigma_i}**2
    
    Args:
        X (np.ndarray): Target (NxD)
        Y (np.ndarray): Source (NxP)
        Sigma (int): covariances for each row of Y (DxDxN)
        
    Returns:
        beta (np.ndarray): Transformation aligning the point sets given the
            covariances 
    """
    
    A = np.zeros((sigma.shape[0]*X.shape[1],sigma.shape[0]*X.shape[1],Y.shape[0]))
    B = np.zeros((X.shape[1],sigma.shape[0],Y.shape[0]))
    for i in range(Y.shape[0]):
        A[:,:,i]=np.kron(np.linalg.inv(sigma[:,:,i]),X[i,None].T@X[i,None])
        B[:,:,i]=X[i,None].T@Y[i,None]@np.linalg.inv(sigma[:,:,i])
    
    B_s = np.nansum(B,2)
    beta = (np.linalg.inv(np.nansum(A,2))@B_s.T.reshape((np.prod(B_s.shape),1))).reshape((Y.shape[1],X.shape[1])).T
    return beta

# %%
def scaled_rotation(X,Y,sigma=None):
    """Solves for Y=S@R@X+T
    
    Args:
        X (np.ndarray): Target (NxD)
        Y (np.ndarray): Source (NxP)
        Sigma (int): covariances for each row of Y (d x d x n)
        
    Returns:
        S (np.ndarray): diagonal scaling matrix
        R (np.ndarray): Rotation matrix i.e. orthonormal and det(R)=1
        T (np.ndarray): Translation
    """
    
    # Remove NAN rows
    idx = ~np.isnan(np.concatenate((X,Y),1)).any(1)
    X = X[idx,:]
    Y = Y[idx,:]

    # De-mean 
    Yhat = Y-Y.mean(0)[np.newaxis,:]
    Xhat = X-X.mean(0)[np.newaxis,:]
    
    # Scale
    sx = np.sqrt((Xhat**2).sum()/Xhat.shape[0])
    sy = np.sqrt((Yhat**2).sum()/Yhat.shape[0])
    
    Yhat=Yhat/sy
    Xhat=Xhat/sx

    # Solve rotation
    C = Yhat.T@Xhat
    U,_,V = np.linalg.svd(C)

    R0=V.T@U.T
    
    # Put it all together
    S = sy/sx
    R = R0
    T = (Y.mean(0)-X.mean(0)[np.newaxis,:]@R0*S)
    
    return S,R,T