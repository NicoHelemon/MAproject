import numpy as np
import networkx as nx
import igraph as ig
import os
import distanceclosure as dc
import math

def apsp(G):
    return dc.metric_backbone(G, weight='weight')

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

    def __call__(self, G):
        for (_, _, w) in G.edges(data=True):
            w['weight'] = np.random.uniform(0, self.b)
        return G
    
    def __str__(self):
        return f'Uni'
    
    def w(self):
        return np.random.uniform(0, self.b)
    
class Exponential:
    def __init__(self, λ = 1):
        self.λ = λ
        self.max = math.inf

    def __call__(self, G):
        for (_, _, w) in G.edges(data=True):
            w['weight'] = np.random.exponential(self.λ)
        return G
    
    def __str__(self):
        return f'Exp'
    
    def w(self):
        return np.random.exponential(self.λ)
    
class Lognormal:
    def __init__(self, µ = -0.5 * np.log(2), σ = np.sqrt(np.log(2))):
        self.µ = µ
        self.σ = σ
        self.max = math.inf

    def __call__(self, G):
        for (_, _, w) in G.edges(data=True):
            w['weight'] = np.random.lognormal(self.µ, self.σ)
        return G
    
    def __str__(self):
        return f'Log'
    
    def w(self):
        return np.random.lognormal(self.µ, self.σ)
    
def BA(n = 1000, d = 0.01, s = 10):
    m = round(d*n*(n-1)/2)
    G = nx.barabasi_albert_graph(n, round(m/n), seed = s)
    G.name = 'BA'
    return G

def ER(n = 1000, d = 0.01, s = 10):
    G = nx.erdos_renyi_graph(n, d, seed = s)
    G.name = 'ER'
    return G

def ABCD(n = 1000, deg_exp = 2.16, com_exp = 1.5, seed = 10):
    max_iter = 1000

    deg_min = 5
    deg_max = int(np.round(n**0.5))
    
    com_min = 50
    tau = 3/4
    com_max = int(np.round(n**tau))

    xi = 0.2

    if any(arg is not None for arg in [n, deg_exp, com_exp, seed]):
        cmd = f'julia utils/deg_sampler.jl deg.dat {deg_exp} {deg_min} {deg_max} {n} {max_iter} {seed}'
        os.system(cmd)
        cmd = f'julia utils/com_sampler.jl cs.dat {com_exp} {com_min} {com_max} {n} {max_iter} {seed}'
        os.system(cmd)
        cmd = f'julia utils/graph_sampler.jl net.dat comm.dat deg.dat cs.dat xi {xi} false false {seed}'
        os.system(cmd)

    G = nx.Graph(ig.Graph.Read_Ncol('net.dat', directed=False).get_edgelist())
    G.name = 'ABCD'
    return G