
import numpy as np
import networkx as nx
from pygsp import utils
from scipy import sparse

def spectral_graph_sparsify(G, num_edges_to_keep: int):
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

    L = nx.laplacian_matrix(G)
    N = np.shape(L)[0]

    # Not sparse
    resistance_distances = utils.resistance_distance(L).toarray()

    W = sparse.coo_matrix(nx.adjacency_matrix(G))
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    nonzero_p = Pe.nonzero()[0].shape[0]

    if nonzero_p < num_edges_to_keep:
        print(f'Warning: only {nonzero_p} edges in the graph, keeping all of them')
        num_edges_to_keep = nonzero_p

    results = np.random.choice(np.shape(Pe)[0], size=num_edges_to_keep, p=Pe, replace=False)
    new_weights = np.zeros(np.shape(weights)[0])
    new_weights[results] = 1

    sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                 shape=(N, N)).nonzero()

    return sparserW