import numpy as np
import os
import matplotlib.pyplot as plt



def neighbor_based_edges(dataset, pt_i, method, criteria, norm='l2'):
    """
    For some point pt_i, this function creates edges between this point and other data points, 
    following some neighbor-based method.
    Params:
        - method: 'k_nearest': edges with its k closest neighbors, or 
                  'e_neighborhood': edges with neighbors whose dist < e.
        - criteria: depends on the method, corresponds to 'k' or 'e'.
        - norm: norm used to compute distances between data points.
    """
    
    data_point = dataset[pt_i]
    if norm == 'l2':
        dist = np.sum(np.power((dataset-data_point), 2), axis=1)
    elif norm == 'l1':
        dist = np.sum(np.absolute(dataset-data_point), axis=1)
    
    if method == 'k_nearest':
        ranked = np.argsort(dist)
        return ranked[1:(criteria+1)]
    elif method == 'e_neighborhood':
        return np.where(np.logical_and(dist>0, dist<=criteria))[0]
    else:
        print(str(method)+' is not an edge method')
        return []


    
def weighting_edges(dataset, pt_i, method, norm='l1', t_heatmap=1):
    """
    For some pt_i, return distance-based weights (depends on the method selected).
    Params:
        - method: 'heatmap': exp(-dist/t_heatmap), 
                  'simple': 1 for all.
        - norm: norm used to compute distances between data points.
    """
    
    data_point = dataset[pt_i]
    if norm == 'l2':
        dist = np.sum(np.power((dataset-data_point), 2), axis=1)
    elif norm == 'l1':
        dist = np.sum(np.absolute(dataset-data_point), axis=1) 
        
    if method == 'heatmap':
        weights = np.exp(-dist/t_heatmap)
    elif method == 'loginverse':
        weights = np.zeros(dataset.shape[0])
        well_def = np.where(dist!=0)[0]
        weights[well_def] = np.log(np.max(dist)/dist[well_def])
    elif method == 'simple':
        weights = np.ones(dataset.shape[0])
            
    else:
        print(str(method)+' is not a weighting method')
        weights = np.zeros(dataset.shape[0])
    return weights
    

    
def create_graph_on_data(dataset, edge_method, criteria, weight_method, norm='l2', t_heatmap=1):
    """
    Global method to call. Returns the adj_matrix of the graph created from the chosen methods and parameters.
    Params:
        - edge_method: method for edge creation, see neighbor_based_edges()
        - weight_method: method for weighting the edges, see weighting_edges()
        - norm: norm used to compute distances between data points.
    """
        
    n = dataset.shape[0]
    adj_matrix = np.zeros([n,n])
    
    for i in range(n):
        edges_to_create = neighbor_based_edges(dataset, i, edge_method, criteria, norm)
        weighted_edges = weighting_edges(dataset, i, weight_method, norm, t_heatmap)
        adj_matrix[i, edges_to_create] = weighted_edges[edges_to_create]
        if edge_method == 'k_nearest': # rectifies implied dissymmetry (indeed, edge if i in k_nearest of j OR j in k_nearest of i)
            adj_matrix[edges_to_create, i] = weighted_edges[edges_to_create]
    return adj_matrix

