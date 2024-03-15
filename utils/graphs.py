import numpy as np
import networkx as nx
import igraph as ig
import os
import distanceclosure as dc

from utils.perturbations import *

def apsp(G):
    return dc.metric_backbone(G, weight='weight')

def uniform(G, a = 0, b = 1):
    for (_, _, w) in G.edges(data=True):
        w['weight'] = np.random.uniform(a, b)
    return G

def exp(G, λ = 1):
    for (_, _, w) in G.edges(data=True):
        w['weight'] = np.random.exponential(λ)
    return G

def log_normal(G, µ = 0, σ = 1):
    for (_, _, w) in G.edges(data=True):
        w['weight'] = np.random.lognormal(µ, σ)
    return G

def clamp(x, a = None, b = None):
    if a is not None:
        x = max(a, x)
    if b is not None:
        x = min(b, x)
    return x

def add_gaussian_noise(G, σ, min = 0, max = None, absolute = False):
    for (_, _, w) in G.edges(data=True):
        noise = np.random.normal(0, σ)
        if absolute:
            noise = abs(noise)
        w['weight'] += noise
        w['weight'] = clamp(w['weight'], min, max)
    return G

def BA(n = 1000, d = 0.01, s = 10):
    m = round(d*n*(n-1)/2)

    G = nx.barabasi_albert_graph(n, round(m/n), seed = s)
    return G

def ER(n = 1000, d = 0.01, s = 10):
    G = nx.erdos_renyi_graph(n, d, seed = s)
    return G

def ABCD(n = 1000, deg_exp = 2.15, com_exp = 1.5, seed = 10):

    max_iter = 1000

    deg_min = 5
    deg_max = int(np.round(n**0.5))
    
    com_min = 50
    tau = 3/4
    com_max = int(np.round(n**tau))

    xi = 0.2

    cmd = f'julia utils\\deg_sampler.jl deg.dat {deg_exp} {deg_min} {deg_max} {n} {max_iter} {seed}'
    os.system(cmd)
    cmd = f'julia utils\\com_sampler.jl cs.dat {com_exp} {com_min} {com_max} {n} {max_iter} {seed}'
    os.system(cmd)
    cmd = f'julia utils\\graph_sampler.jl net.dat comm.dat deg.dat cs.dat xi {xi} false false {seed}'
    os.system(cmd)

    G = nx.Graph(ig.Graph.Read_Ncol('net.dat', directed=False).get_edgelist())

    return G


