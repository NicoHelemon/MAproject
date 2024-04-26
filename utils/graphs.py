import numpy as np
import networkx as nx
import math

import pickle
import os

def is_subgraph(H, G):
    A = set(H.edges(data='weight'))
    B = set(G.edges(data='weight'))
    return A <= B

def add_gaussian_noise(G, σ, max):
    for (_, _, w) in G.edges(data=True):
        new_w = w['weight'] + np.random.normal(0, σ)
        while new_w < 0 or new_w > max:
            if new_w < 0:
                new_w = -new_w
            else:    
                new_w = 2*max - new_w
        w['weight'] = new_w

    return G

class Uniform:
    def __init__(self, b = 2):
        self.b = b
        self.max = b
        self.name = 'Uni'

    def __call__(self, G):
        H = G.copy()
        for (_, _, w) in H.edges(data=True):
            w['weight'] = np.random.uniform(0, self.b)
        return H
    
    def w(self):
        return np.random.uniform(0, self.b)
    
class Exponential:
    def __init__(self, λ = 1):
        self.λ = λ
        self.max = math.inf
        self.name = 'Exp'

    def __call__(self, G):
        H = G.copy()
        for (_, _, w) in H.edges(data=True):
            w['weight'] = np.random.exponential(self.λ)
        return H
    
    def w(self):
        return np.random.exponential(self.λ)
    
class Lognormal:
    def __init__(self, σ = 3/4):
        self.µ = - σ**2/2
        self.σ = σ
        self.max = math.inf
        self.name = 'Log'

    def __call__(self, G):
        H = G.copy()
        for (_, _, w) in H.edges(data=True):
            w['weight'] = np.random.lognormal(self.µ, self.σ)
        return H
    
    def w(self):
        return np.random.lognormal(self.µ, self.σ)
    
def BA(n = 1000, d = 0.01, m = None, s = 10):
    if m is None:
        m = round(d*n*(n-1)/2)
    G = nx.barabasi_albert_graph(n, round(m/n), seed = s)
    G.name = 'BA'
    return G

def ER(n = 1000, p = 0.01, m = None, s = 10):
    if m is not None:
        p = 2*m/(n*(n-1))
    G = nx.erdos_renyi_graph(n, p, seed = s)
    G.name = 'ER'
    return G

def RG(n = 1000, radius = 0.058, s = 30):
    G = nx.random_geometric_graph(n, radius, seed = s)
    G.name = 'RG'
    return G

def ABCD(n = 1000, deg_exp = 2.16, com_exp = 1.5, s = 10, xi = 0.2):
    """
    Generates a graph using the ABCD model.

    Parameters:
    - n (int): Number of nodes in the graph (default: 1000).
    - deg_exp (float): Degree exponent for the power-law degree distribution (default: 2.16).
    - com_exp (float): Community exponent for the power-law community size distribution (default: 1.5).
    - seed (int): Seed for the random number generator (default: 10).

    Returns:
    - G (networkx.Graph): Generated graph.

    """
    max_iter = 1000

    deg_min = 5
    deg_max = int(np.round(n**0.5))
    
    com_min = 50
    tau = 3/4
    com_max = int(np.round(n**tau))

    in_cluster = os.path.exists('ABCD_edges.pkl')
    print(f'in_cluster: {in_cluster}')

    """
    if not in_cluster:
        from juliacall import Main as jl

        jl.seval('using ABCDGraphGenerator')
        jl.seval('using Random')

        jl.seval(f'Random.seed!({s})')
        degs = jl.seval(f'ABCDGraphGenerator.sample_degrees({deg_exp}, {deg_min}, {deg_max}, {n}, {max_iter})')
        jl.seval(f'Random.seed!({s})')
        coms = jl.seval(f'ABCDGraphGenerator.sample_communities({com_exp}, {com_min}, {com_max}, {n}, {max_iter})')
        jl.seval(f'Random.seed!({s})')
        p = jl.seval(f'ABCDGraphGenerator.ABCDParams({degs}, {coms}, nothing, {xi}, false, false, false)')
        edges, _ = jl.seval(f'ABCDGraphGenerator.gen_graph({p})')
        edges = [(u - 1, v - 1) for (u, v) in edges]
        """

    #else:
    with open('ABCD_edges.pkl', 'rb') as f:
            edges = pickle.load(f)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    G.name = 'ABCD'
    return G