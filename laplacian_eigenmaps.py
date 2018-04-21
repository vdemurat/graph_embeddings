import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import graph_creation as g_c




def laplacian_eig(adj_mat, normalized=True):
    """
    Returns the 4 lowest eigenvalues and eigenvectors of the laplacian. In theory, 
    combined together they should give good 2D representations of the data points.
    Params:
        - adj_mat: adjency matrix of the graph.
    """
    D = np.diag(adj_mat.sum(axis=1))
    laplacian = D - adj_mat
    if normalized==True:
        normalization = np.array(np.matrix(np.sqrt(D)).getI())
        laplacian = np.dot(normalization, np.dot(laplacian,normalization))
    eig_val, eig_vect = np.linalg.eig(laplacian)
    ordered = np.argsort(eig_val)
    return eig_val[ordered[[0,1,2,3]]], eig_vect[:, ordered[[0,1,2,3]]]


def plot_laplacian_eigenmaps(data_array, digits, eig_vects, args, plot_dim='2d', xLim=None, yLim=None, zLim=None):
    """
    Plots the 2D/3D-representation of the data points.
    Params:
        - data_array: data as a numpy array (nb_examples, nb_features).
        - digits: mnist digits to display in the 2D-representation.
        - eigen_vects: vectors to use for the 2D-representation.
        - args: parameters of the graph creation.
                Example : args = {'edge_method': 'k_nearest', 'criteria': 20, 'weight_method': 'simple', 
                                  't_heatmap':1e20, 'norm': 'l2'}
        - xLim, yLim: tuples (min,max) to bound the plot area.

    """
    adj_mat = g_c.create_graph_on_data(data_array, **args)
    eig_val, to_plot = laplacian_eig(adj_mat)
    eig_val, to_plot = np.real(eig_val), np.real(to_plot)
    
    if plot_dim == '2d':
        plt.figure(figsize = (15,10))
        for i in digits:
            plt.scatter(to_plot[(100*i):(100*(i+1)),eig_vects[0]], to_plot[(100*i):(100*(i+1)),eig_vects[1]], s=10, label=str(i))
        if xLim != None:
            plt.xlim(xLim)
        if yLim != None:
            plt.ylim(yLim)  
    elif plot_dim == '3d':
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111, projection='3d')
        for i in digits:
            ax.scatter(to_plot[(100*i):(100*(i+1)),eig_vects[0]], to_plot[(100*i):(100*(i+1)),eig_vects[1]],
                       to_plot[(100*i):(100*(i+1)),eig_vects[2]], s=10, label=str(i))
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
        ax.set_zlim(zLim)

    plt.legend()
    plt.show()