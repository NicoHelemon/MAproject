import networkx as nx
import distanceclosure as dc
import random
import numpy as np

def choice_without_replacement(edges, weights, k):
    s = sum(weights)
    probabilities = [w / s for w in weights]
    a = np.empty(len(edges), dtype=object)
    a[:] = edges
    return np.random.choice(a, k, False, probabilities)

class Full:
    def __init__(self):
        self.name = 'full'
        self.id = 'full'

    def __call__(self, G):
        return G

class APSP:
    def __init__(self):
        self.name = 'apsp'
        self.id = 'apsp'

    def __call__(self, G):
        return dc.metric_backbone(G, weight='weight')
    
class LocalDegree:
    def __init__(self):
        self.name = 'local degree'
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
        sG.add_nodes_from(G.nodes())

        return sG
    
class kNeighbor:
    def __init__(self):
        self.name = 'k-neighbor'
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
        sG.add_nodes_from(G.nodes())
        
        return sG

    
class Random:
    def __init__(self):
        self.name = 'random'
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
        sG.add_nodes_from(G.nodes())

        return sG
    
class Threshold:
    def __init__(self):
        self.name = 'threshold'
        self.id = 'thresh'

    def __call__(self, G, t = 1, asc = True):
        if asc:
            edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] < t]
        else:
            edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] >= t]

        sG = nx.Graph()
        sG.add_edges_from(edges)
        sG.add_nodes_from(G.nodes())
        
        return sG