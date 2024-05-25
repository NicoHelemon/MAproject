import networkx as nx
import rustworkx as rx
import numpy as np
import random

from pygsp import utils

RDM_SPARSE_REP = 3

def choice_without_replacement(X, weights, k):
    probabilities = weights / np.sum(weights)
    idx = np.random.choice(len(X), k, replace=False, p=probabilities)
    return [X[i] for i in idx]

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

def resistance_distance(G, U = None, V = None):
    if U is None or V is None:
        U, V = np.column_stack(list(G.edges()))
        U, V = U.astype(int), V.astype(int)

    return utils.resistance_distance(nx.laplacian_matrix(G)).toarray()[U, V]

def quadratic_form(x, U, V, W):
    return np.sum(W * (x[U] - x[V]) ** 2)


class Full:
    def __init__(self):
        self.name = 'Full'
        self.id = 'full'
        self.rep = 1

    def __call__(self, G):
        return G
    

class APSP:
    def __init__(self):
        self.name = 'Apsp'
        self.id = 'apsp'
        self.rep = 1

    def __call__(self, G):
        weighted_edges = list(G.edges(data='weight'))
        rxG = rx.PyGraph()
        rxG.extend_from_weighted_edge_list(weighted_edges)

        APSP = rx.all_pairs_dijkstra_path_lengths(rxG, lambda w: w)

        edges = [(u, v, w) for u, v, w in weighted_edges if w == APSP[u][v]]

        return subgraph(G, edges)
    
    
class LocalDegree:
    def __init__(self, weight_proportional = True, small_weight_preference = False):
        self.name = 'Local degree'
        self.id = 'ld'
        self.rep = 1

        self.weight_proportional = weight_proportional
        self.small_weight_preference = small_weight_preference

    def __call__(self, G, alpha = 0.65):
        if self.weight_proportional:
            if self.small_weight_preference:
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
            num_edges_to_keep = int(np.floor(len(neighbors) ** alpha))

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in neighbors[:num_edges_to_keep]]

        return subgraph(G, edges)
    
    
class kNeighbor:
    def __init__(self, small_weight_preference = False):
        self.name = 'K-neighbor'
        self.id = 'kN'
        self.rep = RDM_SPARSE_REP

        self.small_weight_preference = small_weight_preference

    def __call__(self, G, k = None):
        if k is None:
            k = round(G.number_of_edges() / G.number_of_nodes())

        if self.small_weight_preference:
            add_inverse_weight(G)
            attribut = 'inverse weight'
        else:
            attribut = 'weight'

        H = nx.Graph()
        H.add_nodes_from(G)
        H.add_edges_from((u, v, {'weight': 0}) for u, v, data in G.edges(data=True))
        for node in G.nodes():
            neighbors = list(G.neighbors(node))

            if len(neighbors) <= k:
                for neighbor in neighbors:
                    H[node][neighbor]['weight'] += G[node][neighbor]['weight'] / 2

            else:
                W = sum([G[node][neighbor]['weight'] for neighbor in neighbors]) / (2 * k)
                weights = np.array([G[node][neighbor][attribut] for neighbor in neighbors])
                weights = weights / np.sum(weights)
                sampled_neighbors_idx = np.random.choice(len(neighbors), size=k, p=weights, replace=True)

                for idx in sampled_neighbors_idx:
                    H[node][neighbors[idx]]['weight'] += W

        edges_to_remove = [(u, v) for u, v, weight in H.edges(data='weight') if weight == 0]
        H.remove_edges_from(edges_to_remove)
        
        return H

    
class Random:
    def __init__(self, weight_proportional = True, small_weight_preference = True):
        self.name = 'Random'
        self.id = 'rdm'
        self.rep = RDM_SPARSE_REP

        self.weight_proportional = weight_proportional
        self.small_weight_preference = small_weight_preference

    def __call__(self, G, p = 3/5):
        edges = list(G.edges(data='weight'))

        if self.weight_proportional:
            weights = np.array(edges)[:, 2]
            if self.small_weight_preference: weights = inverse_weight(weights)

            edges = choice_without_replacement(edges, weights, round(G.number_of_edges() * p))

        else:
            edges = random.sample(edges, round(G.number_of_edges() * p))

        return subgraph(G, edges)
    
    
class Threshold:
    def __init__(self, small_weight_preference = True):
        self.name = 'Threshold'
        self.id = 'thresh'
        self.rep = 1

        self.small_weight_preference = small_weight_preference

    def __call__(self, G, t = None):
        if t is None:
            t = np.median([w for _, _, w in G.edges(data='weight')])
            
        if self.small_weight_preference:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w < t]
        else:
            edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w >= t]
        
        return subgraph(G, edges)
    

class EffectiveResistance:
    def __init__(self):
        self.name = f'Effective Resistance'
        self.id = f'er'
        self.rep = RDM_SPARSE_REP

    def __call__(self, G, p = 3/5, e = 0.4, Re = None):
        e_Hm_iter = 10
        len_X = 10
        find_least_Hm_error_iter = 10
        find_least_qf_error_iter = 10

        U, V, W = np.column_stack(list(G.edges(data='weight')))
        U, V = U.astype(int), V.astype(int)

        if Re is None: Re = resistance_distance(G, U, V)

        Pe = W * np.maximum(0, Re)
        Pe = Pe / np.sum(Pe)

        C = 4/30
        n = G.number_of_nodes()
        Hm = p * G.number_of_edges()
        # e = 2 * 1 / np.sqrt(n)
        # Could be optimized by finding clever starting e depending on p, q, n, m
        # e(p = 3/5, n = 1000, m = 4975) = 0.4 is hardcoded for now

        found_suitable_e = False # For desirable prune rate p

        while not found_suitable_e:
            q = round(9 * C**2 * n * np.log(n) / e**2)
            
            e_Hm = []
            for _ in range(e_Hm_iter):
                sampled_edges = np.random.choice(len(Pe), size=q, p=Pe, replace=True)
                e_Hm.append(len(list(np.unique(sampled_edges))))
            
            if np.mean(e_Hm) < Hm:
                found_suitable_e = True
            elif e > 1:
                raise ValueError('Could not find suitable e for given p, q, n, m')
            else:
                e += 1 / (4 * np.sqrt(n))

        q = round(9 * C**2 * n * np.log(n) / e**2)

        X = [np.random.normal(loc=0, scale=1000, size=n) for _ in range(len_X)]

        qf_XG = [quadratic_form(x, U, V, W) for x in X]

        samples = []
        qf_error = []

        for _ in range(find_least_qf_error_iter):

            s = []
            Hm_error = []

            for _ in range(find_least_Hm_error_iter):
                sampled_edges = np.random.choice(len(Pe), size=q, p=Pe, replace=True)

                idx, count = np.unique(sampled_edges, return_counts=True)
                idx, count = list(idx), np.array(count)

                s.append((idx, count))
                Hm_error.append(abs(len(idx) - Hm))

            idx, count = s[np.argmin(Hm_error)]
            S = count / (q * Pe[idx])
            
            samples.append((idx, S))
            qf_XH = [quadratic_form(x, U[idx], V[idx], S * W[idx]) for x in X]
            qf_error.append(np.mean([abs((qf_xH - qf_xG) / qf_xG) for (qf_xH, qf_xG) in zip(qf_XH, qf_XG)]))
 
        idx, S = samples[np.argmin(qf_error)]

        edges = zip(U[idx], V[idx], S * W[idx])

        return subgraph(G, edges)

    

        

