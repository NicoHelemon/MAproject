import numpy as np
import networkx as nx
import netlsd
import timeit
import time as ti
import scipy.cluster.hierarchy as hierarchy

from utils.portrait_divergence import portrait_divergence_weighted

def euclidean_distance(mH, mG):
    return np.linalg.norm(mH - mG)

def distance_matrix(graphs, metric, time_printing = False):
        
        N = len(graphs)

        if time_printing:
            time = []
            print(f'Distance matrix computation with {metric.id}')
            t_iter = N*(N - 1) // 2

        distance_matrix = []

        c_iter = -1
        for i in range(N):
            for j in range(i+1, N):
                start = timeit.default_timer()
                c_iter += 1
                distance_matrix.append(metric(graphs[i], graphs[j]))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    if c_iter % 20 == 0:
                        print(f'Comparison nb {c_iter}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                        print()

        return distance_matrix #squareform


def hierarchical_clustering(graphs, metrics, time_printing = False):

    clusters = {}

    for m in metrics:
        clusters[m.id] = hierarchy.linkage(
            distance_matrix(graphs, m, time_printing), 'ward', optimal_ordering=True).flatten()

    return clusters


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
