import numpy as np
import networkx as nx
import igraph as ig
import os
from utils.perturbations import *


def weighten(G):
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.uniform()
    return G

def weighted_BA(n = 1000, d = 0.01, s = 10):
    m = round(d*n*(n-1)/2)

    G = nx.barabasi_albert_graph(n, round(m/n), seed = s)
    return weighten(G)

def weighted_ER(n = 1000, d = 0.01, s = 10):
    G = nx.erdos_renyi_graph(n, d, seed = s)
    return weighten(G)

def weighted_ABCD(n = 1000, deg_exp = 2.15, com_exp = 1.5, seed = 10):

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

    return weighten(G)

"""
def weighted_LFR(n = 1000, d = 0.01, s = 10):
    m = round(d*n*(n-1)/2)

    tau1 = 3
    tau2 = 1.1
    mu = 0.2
    while True:
        try:
            G = nx.LFR_benchmark_graph(
                n, tau1, tau2, mu, average_degree=10, max_degree=100, min_community = 5, seed = s)
            break
        except:
            continue
    
    while G.number_of_edges() > m:
        edge_removal(G)

    return weighten(G)
"""

"""
def weighted_CM(n = 1000, d = 0.01, t = 1.7):
    m = round(d*n*(n-1)/2)

    while True:
        while True:
            seq = sorted([math.ceil(d) for d in nx.utils.powerlaw_sequence(n, t)], reverse=True)
            if sum(seq) % 2 == 0:
                break
            
        G = nx.Graph(nx.configuration_model(seq))
        G.remove_edges_from(nx.selfloop_edges(G))

        if m < G.number_of_edges():
            break
    
    while G.number_of_edges() > m:
        edge_removal(G)

    return weighten(G)
"""


