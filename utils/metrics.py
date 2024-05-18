import numpy as np
import networkx as nx
import netlsd

from numpy.linalg import norm

from utils.portrait_divergence import portrait_divergence_weighted

def euclidean_distance(mH, mG):
    return norm(mH - mG)

class LaplacianSpectrum:
    def __init__(self):
        self.id = 'lap'
        self.name = 'Spectral Laplacian Euclidean distance'

    def __call__(self, mH, mG):
        if type(mH) != np.ndarray:
            mH = nx.laplacian_spectrum(mH)
        if type(mG) != np.ndarray:
            mG = nx.laplacian_spectrum(mG)

        return euclidean_distance(mH, mG)
    
    def m(self, G):
        return nx.laplacian_spectrum(G)
    
class NormalizedLaplacianSpectrum:
    def __init__(self):
        self.id = 'nlap'
        self.name = 'Spectral Normalized Laplacian Euclidean distance'

    def __call__(self, mH, mG):
        if type(mH) != np.ndarray:
            mH = nx.normalized_laplacian_spectrum(mH)
        if type(mG) != np.ndarray:
            mG = nx.normalized_laplacian_spectrum(mG)

        return euclidean_distance(mH, mG)
    
    def m(self, G):
        return nx.normalized_laplacian_spectrum(G)

class NetlsdHeat:
    def __init__(self):
        self.prec_mG = {None : None}
        self.id = 'netlsd'
        self.name = 'Network Laplacian Spectral descriptor distance'

    def __call__(self, mH, mG):
        if type(mH) != np.ndarray:
            mH = netlsd.heat(mH)
        if type(mG) != np.ndarray:
            mG = netlsd.heat(mG)

        return euclidean_distance(mH, mG)
    
    def m(self, G):
        return netlsd.heat(G)

class PortraitDivergence:
    def __init__(self):
        self.id = 'portrait'
        self.name = 'Portrait divergence'

    def __call__(
            self, H = None, paths_H = None, UPL_H = None, G = None, paths_G = None, UPL_G = None):
        
        return portrait_divergence_weighted(
            G=G, H=H, paths_G=paths_G, UPL_G=UPL_G, paths_H=paths_H, UPL_H=UPL_H)
