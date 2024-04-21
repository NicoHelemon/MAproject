import networkx as nx
import rustworkx as rx
import pygsp.graphs as pg
import numpy as np
import random

from utils.effective_resistance import *

def choice_without_replacement(X, weights, k):
    probabilities = weights / np.sum(weights)
    embeded_X = np.empty(len(X), dtype=object)
    embeded_X[:] = X
    return np.random.choice(embeded_X, k, False, probabilities)


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

        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)

        return sG
    
    
class LocalDegree:
    def __init__(self):
        self.name = 'Local degree'
        self.id = 'ld'

    def __call__(self, G, alpha = 0.5, weighted_degree = False, asc = True):
        if weighted_degree:
            if asc:
                for _, _, d in G.edges(data=True):
                    d['1/weight'] = 1 / d['weight']

                def degree(node):
                    return G.degree(node, weight='1/weight')
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

        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)

        return sG
    
    
class kNeighbor:
    def __init__(self):
        self.name = 'K-neighbor'
        self.id = 'kN'

    def __call__(self, G, k = 5, asc = True, random = False):
        edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            k = min(k, len(neighbors))
            if random:
                weights = np.array([G[node][neighbor]['weight'] for neighbor in neighbors])
                if asc: weights = 1 / weights
            
                selected_neighbors = choice_without_replacement(neighbors, weights, k)

            else:
                neighbors.sort(key=lambda x: G[node][x]['weight'], reverse=not asc)
                selected_neighbors = neighbors[:k]

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in selected_neighbors]

        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)
        
        return sG

    
class Random:
    def __init__(self):
        self.name = 'Random'
        self.id = 'rdm'

    def __call__(self, G, p = 3/5, weight_proportional = True, asc = True):

        edges = list(G.edges(data='weight'))

        if weight_proportional:
            weights = np.array(edges)[:, 2]
            if asc: weights = 1 / weights

            edges = choice_without_replacement(edges, weights, round(G.number_of_edges() * p))

        else:
            edges = random.sample(edges, round(G.number_of_edges() * p))
        
        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)

        return sG
    
    
class Threshold:
    def __init__(self):
        self.name = 'Threshold'
        self.id = 'thresh'

    def __call__(self, G, t = 1, asc = True):
        if asc:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w < t]
        else:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w >= t]

        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)
        
        return sG
    

class EffectiveResistance:
    def __init__(self):
        self.name = 'Effective resistance'
        self.id = 'er'

    def __call__(self, G, p = 3/5, random = False):
        edges = spectral_graph_sparsify(G, round(G.number_of_edges() * p), random = random)

        sG = nx.Graph()
        sG.add_nodes_from(G)
        sG.add_weighted_edges_from(edges)
        
        return sG