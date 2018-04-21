import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import graph_creation as g_c
import laplacian_eigenmaps as l_eig




def cauchy_eig(adj_mat, plot_dim, sigma, gamma, nb_iter, L=0.05):
    """Returns the Cauchy local solution
    """
    n = adj_mat.shape[0]
    emb_dim = get_emb_dim(plot_dim)
    _, cauchy_eig = l_eig.laplacian_eig(adj_mat)
    cauchy_eig = cauchy_eig[:,1:(emb_dim+1)].T
    
    for itr in range(nb_iter):
        lr = L
        #print('iter : '+str(itr))
        grad = obj_gradient(cauchy_eig, adj_mat, sigma, n, emb_dim)
        opt_eig = get_new_opt_eig(cauchy_eig, lr, grad, n)
        t=0
        while check_cond(opt_eig, cauchy_eig, lr, grad) and t < 30:
            #print(check_cond(opt_eig, cauchy_eig, lr, grad))
            lr *= gamma
            opt_eig = get_new_opt_eig(cauchy_eig, lr, grad, n)  
            t+=1
        #print('Changes: '+str(np.all(cauchy_eig==opt_eig)))
        cauchy_eig = opt_eig.copy()
    return opt_eig
        

    
def plot_cauchy_eig(data_array, digits, args, sigma=1, gamma=2, nb_iter=50, L=0.05, xLim=None, yLim=None):
    """
    Generates the 2D Cauchy representations of the specified digits.
    """
    adj_mat = g_c.create_graph_on_data(data_array, **args)
    opt_eig = cauchy_eig(adj_mat, '2d', sigma, gamma, nb_iter, L)
    
    plt.figure(figsize = (15,10))
    for i in digits: # arg "digits"
        plt.scatter(opt_eig.T[(100*i):(100*(i+1)),0], opt_eig.T[(100*i):(100*(i+1)),1], s=10, label=str(i))
    plt.legend()
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.show()    
    
    
    
    
    
    
def get_new_opt_eig(cauchy_eig, L, grad, n):
    M = cauchy_eig + 1/L * grad
    inter_mat = np.dot(M, np.identity(n)-1/n*np.ones([n,n]))
    _, _, opt_eig = np.linalg.svd(M, full_matrices=False)
    return opt_eig


def obj_gradient(R, W, sigma, n, emb_dim):
    grad = []
    for i in range(n):
        grad.append(-2*np.sum([W[i,j] * (R[:,i]-R[:,j]) / (np.linalg.norm(R[:,i]-R[:,j],2)**2 +  sigma**2)**2 
                           for j in range(n)], axis=0).reshape([emb_dim,1]))
    
    grad = np.hstack([grad[i].reshape([emb_dim,1]) for i in range(n)])
    return grad
 
    
def check_cond(opt_eig, cauchy_eig, L, grad):
    part_1 = np.trace(np.dot((opt_eig-cauchy_eig).T, opt_eig-cauchy_eig))
    part_2 = -2/L * np.trace(np.dot((opt_eig-cauchy_eig).T, grad))
    #print('part1, part2 : '+str(part_1)+', '+str(part_2))
    return (part_1 + part_2 > 0)




def get_emb_dim(plot_dim):
    return 2 if plot_dim=='2d' else 3