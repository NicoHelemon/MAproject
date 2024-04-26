import networkx as nx
import rustworkx as rx
import numpy as np
import random

from pygsp import utils

def choice_without_replacement(X, weights, k):
    probabilities = weights / np.sum(weights)
    embeded_X = np.empty(len(X), dtype=object)
    embeded_X[:] = X
    return np.random.choice(embeded_X, k, False, probabilities)

def inverse_weight(w):
    return 1 / (1 + w)

def add_inverse_weight(G):
    for _, _, d in G.edges(data=True):
        d['inverse weight'] = inverse_weight(d['weight'])

def subgraph(G, edges):
    sG = nx.Graph()
    sG.add_nodes_from(G)
    sG.add_weighted_edges_from(edges)
    return sG


class Full:
    def __init__(self):
        self.name = 'Full'
        self.id = 'full'

    def __call__(self, G):
        return G
    

class APSP:
    def __init__(self):
        self.name = 'Apsp'
        self.id = 'apsp'

    def __call__(self, G):
        weighted_edges = list(G.edges(data='weight'))
        rxG = rx.PyGraph()
        rxG.extend_from_weighted_edge_list(weighted_edges)

        APSP = rx.all_pairs_dijkstra_path_lengths(rxG, lambda w: w)

        edges = [(u, v, w) for u, v, w in weighted_edges if w == APSP[u][v]]

        return subgraph(G, edges)
    
    
class LocalDegree:
    def __init__(self):
        self.name = 'Local degree'
        self.id = 'ld'

    def __call__(self, G, alpha = 0.5, weight_proportional = True, small_weight_preference = True):
        if weight_proportional:
            if small_weight_preference:
                add_inverse_weight(G)

                def degree(node):
                    return G.degree(node, weight='inverse weight')
            else:
                def degree(node):
                    return G.degree(node, weight='weight')
        else:
            def degree(node):
                return G.degree(node)

        edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            neighbors.sort(key=degree, reverse=True)
            num_edges_to_keep = max(1, int(len(neighbors) * alpha))

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in neighbors[:num_edges_to_keep]]

        return subgraph(G, edges)
    
    
class kNeighbor:
    def __init__(self):
        self.name = 'K-neighbor'
        self.id = 'kN'

    def __call__(self, G, k = None, small_weight_preference = True, random = False):
        if k is None:
            k = round(2 * G.number_of_edges() / G.number_of_nodes()) // 2

        if small_weight_preference:
            add_inverse_weight(G)
            attribut = 'inverse weight'
        else:
            attribut = 'weight'

        edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            k = min(k, len(neighbors))

            if random:
                weights = np.array([G[node][neighbor][attribut] for neighbor in neighbors])
                selected_neighbors = choice_without_replacement(neighbors, weights, k)

            else:
                neighbors.sort(key=lambda neighbor: G[node][neighbor][attribut], reverse=True)
                selected_neighbors = neighbors[:k]

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in selected_neighbors]
        
        return subgraph(G, edges)

    
class Random:
    def __init__(self):
        self.name = 'Random'
        self.id = 'rdm'

    def __call__(self, G, p = 3/5, weight_proportional = True, small_weight_preference = True):
        edges = list(G.edges(data='weight'))

        if weight_proportional:
            weights = np.array(edges)[:, 2]
            if small_weight_preference: weights = inverse_weight(weights)

            edges = choice_without_replacement(edges, weights, round(G.number_of_edges() * p))

        else:
            edges = random.sample(edges, round(G.number_of_edges() * p))

        return subgraph(G, edges)
    
    
class Threshold:
    def __init__(self):
        self.name = 'Threshold'
        self.id = 'thresh'

    def __call__(self, G, t = None, small_weight_preference = True):
        if t is None:
            t = np.median([w for _, _, w in G.edges(data='weight')])
            
        if small_weight_preference:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w < t]
        else:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w >= t]
        
        return subgraph(G, edges)
    

class EffectiveResistance:
    def __init__(self):
        self.name = 'Effective resistance'
        self.id = 'er'

    def __call__(self, G, p = 5/5):
        q = round(G.number_of_edges() * p)

        U, V, W = np.column_stack(list(G.edges(data='weight')))
        U, V = U.astype(int), V.astype(int)

        resistance_distances = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()
        Re = np.maximum(0, resistance_distances[U, V])

        Pe = W * Re
        Pe = Pe / np.sum(Pe)

        sampled_edges = np.random.choice(len(Pe), size=q, p=Pe, replace=True)

        idx, count = np.unique(sampled_edges, return_counts=True)
        idx, count = list(idx), np.array(count)
        
        S = count / (q * Pe[idx])

        edges = zip(U[idx], V[idx], S * W[idx])

        return subgraph(G, edges)