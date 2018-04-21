import numpy as np
import os

"""
To check that the graph respects some properties, i.e that its adjency matrix is symmetric and that the graph is connected (using dfs).
"""


def convert_adj_mat(adj_matrix):
    adj_dict = {}
    for i in range(adj_matrix.shape[0]):
        adj_dict[i] = list(np.where(adj_matrix[:,i] != 0)[0])
    return adj_dict


def dfs(adj_dict, start):
    stack, visited = [start], []
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.append(current)
        for neighbor in adj_dict[current]:
            stack.append(neighbor)
    return visited


def is_connected(adj_matrix):
    adj_dict = convert_adj_mat(adj_matrix)
    visited = dfs(adj_dict, 1)
    return sorted(visited)==list(range(adj_matrix.shape[0]))


def check_graph(adj_matrix):
    """
    Checks if the graph is connected, and if the adjency matrix is symmetric
    """
    check_sym = (len(np.where(adj_matrix!=adj_matrix.T)[0])==0)
    check_connected = is_connected(adj_matrix)
    return check_sym and check_connected