import numpy as np
import networkx as nx
import netlsd
from utils.portrait_divergence import portrait_divergence


def euc_distance(m, H, G, mG = None):
    #if mG is None:
    #    mG = m(G)
    return np.linalg.norm(m(H) - mG)

def lap_spec_d(H, G, mG = None):
    return euc_distance(nx.laplacian_spectrum, H, G, mG)

def adj_spec_d(H, G, mG = None):
    return euc_distance(nx.adjacency_spectrum, H, G, mG)

def nlap_spec_d(H, G, mG = None):
    return euc_distance(nx.normalized_laplacian_spectrum, H, G, mG)

def netlsd_heat_d(H, G, mG = None):
    return euc_distance(netlsd.heat, H, G, mG)

def portrait_div_d(H, G):
    return portrait_divergence(H, G)