import networkx as nx
import rustworkx as rx
import pygsp.graphs as pg
import numpy as np
import random
import distanceclosure as dc

from utils.effective_resistance import *

def choice_without_replacement(edges, weights, k):
    s = sum(weights)
    probabilities = [w / s for w in weights]
    a = np.empty(len(edges), dtype=object)
    a[:] = edges
    return np.random.choice(a, k, False, probabilities)


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
        weighted_edges = [(source, target, data['weight']) for source, target, data in G.edges(data=True)]
        rxG = rx.PyGraph()
        rxG.extend_from_weighted_edge_list(weighted_edges)

        APSP = rx.all_pairs_dijkstra_path_lengths(rxG, lambda w: w)

        H = nx.Graph()
        APSP_edges = []

        for u, v, w in weighted_edges:
            if w == APSP[u][v]:
                APSP_edges.append((u, v, w))

        H.add_weighted_edges_from(APSP_edges)
        H.add_nodes_from(G)
        return H
    
    
class LocalDegree:
    def __init__(self):
        self.name = 'Local degree'
        self.id = 'ld'

    def __call__(self, G, alpha=0.5):
        edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            neighbors.sort(key=lambda x: G.degree(x), reverse=True)
            num_edges_to_keep = max(1, int(len(neighbors) * alpha))

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in neighbors[:num_edges_to_keep]]

        sG = nx.Graph()
        sG.add_weighted_edges_from(edges)
        sG.add_nodes_from(G)

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
                weights = [G[node][neighbor]['weight'] for neighbor in neighbors]
                if asc: weights = [1 / w for w in weights]
            
                selected_neighbors = choice_without_replacement(neighbors, weights, k)

            else:
                neighbors.sort(key=lambda x: G[node][x]['weight'], reverse=not asc)
                selected_neighbors = neighbors[:k]

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in selected_neighbors]

        sG = nx.Graph()
        sG.add_weighted_edges_from(edges)
        sG.add_nodes_from(G)
        
        return sG

    
class Random:
    def __init__(self):
        self.name = 'Random'
        self.id = 'rdm'

    def __call__(self, G, p = 3/5, weight_prop = True, asc = True):

        edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

        if weight_prop:
            _, _, weights = zip(*edges)
            if asc: weights = [1 / w for w in weights]

            edges = choice_without_replacement(edges, weights, round(G.number_of_edges() * p))

        else:
            edges = random.sample(edges, round(G.number_of_edges() * p))
        
        sG = nx.Graph()
        sG.add_weighted_edges_from(edges)
        sG.add_nodes_from(G)

        return sG
    
    
class Threshold:
    def __init__(self):
        self.name = 'Threshold'
        self.id = 'thresh'

    def __call__(self, G, t = 1, asc = True):
        if asc:
            edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] < t]
        else:
            edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] >= t]

        sG = nx.Graph()
        sG.add_edges_from(edges)
        sG.add_nodes_from(G)
        
        return sG
    

class EffectiveResistance:
    def __init__(self):
        self.name = 'Effective resistance'
        self.id = 'er'

    def __call__(self, G, p = 3/5):
        W = spectral_graph_sparsify(G, round(G.number_of_edges() * p))

        sG = G.edge_subgraph(zip(W[0], W[1])).copy()
        sG.add_nodes_from(G)
        
        return sG