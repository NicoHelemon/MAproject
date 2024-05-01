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

def qform(x, U, V, W):
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
    def __init__(self, random = True, qf_preserving = True):
        self.random = random
        self.qf_preserving = qf_preserving

        if random:
            rdm = 'random'
            rmd_id = 'rdm'
        else:
            rdm = 'deterministic'
            rmd_id = 'det'

        if qf_preserving:
            self.name = f'ER {rdm} QF preserving'
            self.id = f'er_{rmd_id}_qf'
        else:
            self.name = f'ER {rdm}'
            self.id = f'er_{rmd_id}'

    def __call__(self, G, p = 3/5):
        if self.random:
            return self.random_er(G, p)
        else:
            return self.deterministic_er(G, p)

    def deterministic_er(self, G, p):
        q = round(G.number_of_edges() * p)

        U, V, W = np.column_stack(list(G.edges(data='weight')))
        U, V = U.astype(int), V.astype(int)

        Re = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()[U, V]

        if self.qf_preserving:
            Re = np.maximum(0, Re)
            idx = list(np.argsort(Re)[::-1][:q])
            idx = [idx[i] for i in range(len(idx)) if Re[idx[i]] > 0]

            if len(idx) < q:
                print(f'Warning: q = {q} > |E| = {len(idx)}')

            Pe = W * Re
            Pe = Pe / np.sum(Pe)

            S = 1 / (q * Pe[idx])
        else:
            idx = list(np.argsort(Re)[::-1][:q])

            S = np.ones(len(idx))

        edges = zip(U[idx], V[idx], S * W[idx])

        return subgraph(G, edges)
        
    
    def random_er(self, G, p = 3/5, e = 0.4):
        e_H_m_iter = 10
        len_X = 10
        H_m_iter = 10
        best_error_iter = 10


        U, V, W = np.column_stack(list(G.edges(data='weight')))
        U, V = U.astype(int), V.astype(int)

        Re = utils.resistance_distance(nx.laplacian_matrix(G)).toarray()[U, V]

        Pe = W * np.maximum(0, Re)
        Pe = Pe / np.sum(Pe)

        C = 4/30
        n = G.number_of_nodes()
        m = G.number_of_edges()
        # e = 2 * 1 / np.sqrt(n)
        # Could be optimized by finding clever starting e depending on p, q, n, m
        # e(p = 3/5, n = 1000, m = 4975) = 0.4 is hardcoded for now

        found_suitable_e = False # For desirable prune rate p

        while not found_suitable_e:
            q = round(9 * C**2 * n * np.log(n) / e**2)
            
            H_m = []
            for _ in range(e_H_m_iter):
                sampled_edges = np.random.choice(len(Pe), size=q, p=Pe, replace=True)
                H_m.append(len(list(np.unique(sampled_edges))))
            
            if np.mean(H_m) < p * m:
                found_suitable_e = True
            elif e > 1:
                raise ValueError('Could not find suitable e for given p, q, n, m')
            else:
                e += 1 / (4 * np.sqrt(n))

        q = round(9 * C**2 * n * np.log(n) / e**2)

        X = [np.random.normal(loc=0, scale=1000, size=n) for _ in range(len_X)]

        qform_G = [qform(x, U, V, W) for x in X]

        samples = []
        qf_distances = []

        for _ in range(best_error_iter):

            s = []
            H_m = []

            for _ in range(H_m_iter):
                sampled_edges = np.random.choice(len(Pe), size=q, p=Pe, replace=True)

                idx, count = np.unique(sampled_edges, return_counts=True)
                idx, count = list(idx), np.array(count)

                s.append((idx, count))
                H_m.append(abs(len(idx) - p * m))

            idx, count = s[np.argmin(H_m)]

            if self.qf_preserving:
                S = count / (q * Pe[idx])
            else:
                S = np.ones(len(idx))
            
            samples.append((idx, S))
            qf_distances.append(np.mean([abs((qform(x, U[idx], V[idx], S * W[idx]) - qfxG) / qfxG) 
                                     for (x, qfxG) in zip(X, qform_G)]))


        least_error_idx = np.argmin(qf_distances)
        qf_error = qf_distances[least_error_idx]
        idx, S = samples[least_error_idx]

        if qf_error > e:
            # Never occurs in practice
            print(f'Warning: E_x[|(x^t L_H x - x^t L_G x) / (x^t L_G x)|] = {qf_error} > e = {e}')

        edges = zip(U[idx], V[idx], S * W[idx])

        return subgraph(G, edges)
    

        

