import networkx as nx
import rustworkx as rx
import numpy as np
import random

from pygsp import utils

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

def quadratic_form(x, U, V, W):
    return np.sum(W * (x[U] - x[V]) ** 2)


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

    def __call__(self, G, alpha = 0.65, weight_proportional = True, small_weight_preference = True):
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
            num_edges_to_keep = max(1, int(np.floor(len(neighbors) ** alpha)))

            edges += [(node, neighbor, G[node][neighbor]['weight']) for neighbor in neighbors[:num_edges_to_keep]]

        return subgraph(G, edges)
    
    
class kNeighbor:
    def __init__(self):
        self.name = 'K-neighbor'
        self.id = 'kN'

    def __call__(self, G, k = None, small_weight_preference = True, random = True):
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
        self.name = f'Effective Resistance'
        self.id = f'er'

    def __call__(self, G, p = 3/5, e = 0.4):
        e_Hm_iter = 10
        len_X = 10
        find_least_Hm_error_iter = 10
        find_least_qf_error_iter = 10

        U, V, W = np.column_stack(list(G.edges(data='weight')))
        U, V = U.astype(int), V.astype(int)

        Re = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()[U, V]

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
    

        

