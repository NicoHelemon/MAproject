
import numpy as np
import networkx as nx
import random as rd
from pygsp import utils
import time

def spectral_graph_sparsify(G, num_edges_to_keep, random = False):
    """@https://github.com/noamrazin/gnn_interactions/blob/master/edges_removal/spectral_sparsification.py"""
    r"""Sparsify a graph (with Spielman-Srivastava).
    Adapted from the PyGSP implementation (https://pygsp.readthedocs.io/en/v0.5.1/reference/reduction.html).

    Parameters
    ----------
    G : PyGSP graph or sparse matrix
        Graph structure or a Laplacian matrix
    num_edges_to_keep : int
        Number of edges to keep in graph.

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling` for more information.
    """

    """"""
    U, V, W = np.column_stack(list(G.edges(data='weight')))
    U, V = U.astype(int), V.astype(int)

    resistance_distances = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()
    Re = np.maximum(0, resistance_distances[U, V])

    Pe = W * Re
    Pe = Pe / np.sum(Pe)

    if np.count_nonzero(Pe) < num_edges_to_keep:
        print('Not enough edges to keep, retrying')
        H = G.copy()
        H.remove_edge(*rd.choice(list(H.edges())))
        return spectral_graph_sparsify(H, num_edges_to_keep)

    if random:
        edges_idx = np.random.choice(len(Pe), size=num_edges_to_keep, p=Pe, replace=False)
    else:
        edges_idx = np.argsort(Pe)[::-1][:num_edges_to_keep]

    return zip(U[edges_idx], V[edges_idx], W[edges_idx])