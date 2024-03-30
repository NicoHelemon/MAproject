import numpy as np
import networkx as nx
import netlsd
import timeit
import time as ti
import scipy.cluster.hierarchy as hierarchy

from utils.portrait_divergence import portrait_divergence_weighted

def euclidean_distance(mH, mG):
    return np.linalg.norm(mH - mG)



"""
def hierarchical_clustering(graphs, metrics, time_printing = False):

    clusters = {}

    for m in metrics:
        clusters[m.id] = hierarchy.linkage(
            distance_matrix(graphs, m, time_printing), 'ward', optimal_ordering=True).flatten()

    return clusters
"""


class LaplacianSpectrum:
    def __init__(self):
        self.prec_mG = {None : None}
        self.id = 'lap'
        self.name = 'Spectral Laplacian Euclidean distance'

    def __call__(self, H, G, prec_mode = None):
        if prec_mode is not None:
            mG = self.prec_mG[prec_mode]
        else:
            mG = nx.laplacian_spectrum(G)
        mH = nx.laplacian_spectrum(H)

        return euclidean_distance(mH, mG)
    
    def set_prec_mG(self, G, mode):
        self.prec_mG[mode] = nx.laplacian_spectrum(G)
    
class NormalizedLaplacianSpectrum:
    def __init__(self):
        self.prec_mG = {None : None}
        self.id = 'nlap'
        self.name = 'Spectral Normalized Laplacian Euclidean distance'

    def __call__(self, H, G, prec_mode = None):
        if prec_mode is not None:
            mG = self.prec_mG[prec_mode]
        else:
            mG = nx.normalized_laplacian_spectrum(G)
        mH = nx.normalized_laplacian_spectrum(H)

        return euclidean_distance(mH, mG)
    
    def set_prec_mG(self, G, mode):
        self.prec_mG[mode] = nx.normalized_laplacian_spectrum(G)

class NetlsdHeat:
    def __init__(self):
        self.prec_mG = {None : None}
        self.id = 'netlsd'
        self.name = 'Network Laplacian Spectral descriptor distance'

    def __call__(self, H, G, prec_mode = None):
        if prec_mode is not None:
            mG = self.prec_mG[prec_mode]
        else:
            mG = netlsd.heat(G)
        mH = netlsd.heat(H)

        return euclidean_distance(mH, mG)
    
    def set_prec_mG(self, G, mode):
        self.prec_mG[mode] = netlsd.heat(G)

class PortraitDivergence:
    def __init__(self):
        self.prec_mG = False
        self.id = 'portrait'
        self.name = 'Portrait divergence'

    def __call__(self, H, G, prec_mode = None):
        return portrait_divergence_weighted(H, G)
