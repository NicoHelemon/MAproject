import numpy as np
import networkx as nx
import igraph as ig
import os
import math


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

def ER(n = 1000, d = 0.01, m = None, s = 10):
    if m is not None:
        d = 2*m/(n*(n-1))
    G = nx.erdos_renyi_graph(n, d, seed = s)
    G.name = 'ER'
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

    if not os.path.exists('net.dat') or any(arg is not None for arg in [n, deg_exp, com_exp, s, xi]):
        cmd = f'julia utils/deg_sampler.jl deg.dat {deg_exp} {deg_min} {deg_max} {n} {max_iter} {s}'
        os.system(cmd)
        cmd = f'julia utils/com_sampler.jl cs.dat {com_exp} {com_min} {com_max} {n} {max_iter} {s}'
        os.system(cmd)
        cmd = f'julia utils/graph_sampler.jl net.dat comm.dat deg.dat cs.dat xi {xi} false false {s}'
        os.system(cmd)

    G = nx.Graph(ig.Graph.Read_Ncol('net.dat', directed=False).get_edgelist())

    os.remove('comm.dat') if os.path.exists('comm.dat') else None
    os.remove('cs.dat') if os.path.exists('cs.dat') else None
    os.remove('deg.dat') if os.path.exists('deg.dat') else None

    G.name = 'ABCD'
    return G