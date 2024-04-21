
import numpy as np
import networkx as nx
import random as rd
from pygsp import utils
from scipy import sparse

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

    W = sparse.coo_matrix(nx.adjacency_matrix(G))
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))
    weights = np.maximum(0, weights)

    resistance_distances = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])

    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    if np.count_nonzero(Pe) < num_edges_to_keep:
        print('Not enough edges to keep, retrying')
        H = G.copy()
        H.remove_edge(*rd.choice(list(H.edges())))
        return spectral_graph_sparsify(H, num_edges_to_keep)

    if random:
        selected_edges = np.random.choice(len(Pe), size=num_edges_to_keep, p=Pe, replace=False)
    else:
        selected_edges = np.argsort(Pe)[::-1][:num_edges_to_keep]

    return zip(start_nodes[selected_edges], end_nodes[selected_edges])