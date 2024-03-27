import numpy as np
import pandas as pd
import timeit
import time as ti
from pathlib import Path

from utils.static import MODES, E_MES
from utils.graphs import *
from utils.metrics import hierarchical_clustering

class Perturbation:
    def __init__(self):
        self.name = 'perturbation'
        self.out_path_root = 'results/perturbation'

    def out_path(self, graph, weight, perturbation):
        g, w, p = [x.name if not isinstance(x, str) else x for x in [graph, weight, perturbation]]
        return f'{self.out_path_root}/{p}/{g}/{w}'

    def __call__(
        self, G, weight, perturbation, metrics, K = 20, N = 1000, step = 5, 
        time_printing = False, save = True):
        print(f'Perturbation test: {G.name} {weight.name} {perturbation.name}\n'.upper())
    
        time = []
        t_iter = K*N

        full_G = weight(G)
        apsp_G = apsp(full_G)

        distances, edges = {}, {}

        for mode in MODES:
            distances[mode] = {m.id : [ [] for _ in range(K) ] for m in metrics}
            edges[mode] = {mes : [ [] for _ in range(K) ] for mes in E_MES}
        
        print('Initial metrics\n')
        for mode, G in zip(MODES, [full_G, apsp_G]):
            for m in metrics:
                if m.prec_mG:
                    m.set_prec_mG(G, mode)

                mG = m(G, G, mode)
                for i in range(K):
                    distances[mode][m.id][i].append(mG)

            for i in range(K):
                    edges[mode]['count'][i].append(G.number_of_edges())
                    edges[mode]['size'][i].append(G.size(weight='weight'))

        print('Perturbation start\n')
        for i in range(K):
            full_H = full_G.copy()
            for j in range(N):
                perturbation(full_H, weight.w())
                if j % step == 0:
                    start = timeit.default_timer()
                    apsp_H = apsp(full_H)

                    for mode, H, G in zip(MODES, [full_H, apsp_H], [full_G, apsp_G]):
                        edges[mode]['count'][i].append(H.number_of_edges())
                        edges[mode]['size'][i].append(H.size(weight='weight'))
                    
                        for m in metrics:
                            distances[mode][m.id][i].append(m(H, G, mode))

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        c_iter = i*N + j
                        print(f'Iteration {i}, Perturbation nb {j}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter) / step))))
                        print()
        
        if save:
            def df_from_dict(d):
                return pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d.items()}, axis=1)

            out_path = self.out_path(G, weight, perturbation)
            Path(out_path).mkdir(parents = True, exist_ok = True)

            for mode in MODES:
                df_from_dict(distances[mode]).to_csv(f'{out_path}/{mode}.csv', index=False)
                df_from_dict(edges[mode]).to_csv(f'{out_path}/edges {mode}.csv', index=False)


class GaussianNoise:
    def __init__(self):
        self.name = 'gaussian noise'
        self.out_path_root = 'results/gaussian_noise'

    def out_path(self, graph, weight):
        g, w = [x.name if not isinstance(x, str) else x for x in [graph, weight]]
        return f'{self.out_path_root}/{g}/{w}'

    def __call__(
            self, G, weight, metrics, sigmas = np.linspace(0, 0.1, 20+1).tolist(), K = 20, 
            time_printing = False, save = True):
        print(f'Gaussian noise test: {G.name} {weight.name}\n'.upper())

        full_G = weight(G)
        
        time = []
        t_iter = len(sigmas)*K

        distances, edges = {}, {}

        for mode in MODES:
            distances[mode] = {σ : {m.id : [] for m in metrics} for σ in sigmas}
            edges[mode]     = {σ : {mes : [] for mes in E_MES} for σ in sigmas}

        for i, σ in enumerate(sigmas):
            for j in range(K):
                start = timeit.default_timer()
                full_H1 = add_gaussian_noise(full_G.copy(), σ, weight.max)
                full_H2 = add_gaussian_noise(full_G.copy(), σ, weight.max)
                apsp_H1 = apsp(full_H1)
                apsp_H2 = apsp(full_H2)

                for mode, H1, H2 in zip(MODES, [full_H1, apsp_H1], [full_H2, apsp_H2]):
                    edges[mode][σ]['count'].append(H1.number_of_edges())
                    edges[mode][σ]['size'].append(H1.size(weight='weight'))
                    edges[mode][σ]['count'].append(H2.number_of_edges())
                    edges[mode][σ]['size'].append(H2.size(weight='weight'))

                    for m in metrics:
                        distances[mode][σ][m.id].append(m(H1, H2))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    c_iter = i*K + j
                    print(f'Sigma nb {i}, Iteration nb {j}')
                    print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                    print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                    print()

        if save:
            def df_from_dict(d):
                return pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                            for σ, a in d.items()}, axis=0).reset_index(level=1, drop=True)

            out_path = self.out_path(G, weight)
            Path(out_path).mkdir(parents = True, exist_ok = True)

            for mode in MODES:
                df_from_dict(distances[mode]).to_csv(f'{out_path}/{mode}.csv')
                df_from_dict(edges[mode]).to_csv(f'{out_path}/edges {mode}.csv')


class ClusteringGaussianNoise:
    def __init__(self):
        self.name = 'clustering gaussian noise'
        self.out_path_root = 'results/clustering/gaussian_noise'

    def out_path(self, graph):
        if not isinstance(graph, str):
            graph = graph.name
        return f'{self.out_path_root}/{graph}'

    def __call__(
            self, G, weights, metrics, sigma, K = 3, N = 6, 
            time_printing = False, save = True):
        print(f'Clustering test: {G.name}\n'.upper())

        if time_printing:
            time = []
            t_iter = len(weights)*K*N
            print("Graphs initialization\n")

        graphs_full = []
        graphs_apsp = []
        graphs_label = []
        for i, w in enumerate(weights):
            for j in range(K):
                H = w(G.copy())
                for k in range(N):
                    start = timeit.default_timer()
                    H_full = add_gaussian_noise(H.copy(), sigma, w.max)
                    H_apsp = apsp(H_full)
                    graphs_full.append(H_full)
                    graphs_apsp.append(H_apsp)
                    graphs_label.append(f'{w.name} {j}')

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        c_iter = i*K*N + j*N + k
                        print(f'{w.name}, Iteration (weight) nb {j}, Iteration (noise) nb {k}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                        print()

        clustering_full = hierarchical_clustering(graphs_full, metrics, time_printing)
        clustering_apsp = hierarchical_clustering(graphs_apsp, metrics, time_printing)

        if save:
            out_path = self.out_path(G)
            Path(out_path).mkdir(parents = True, exist_ok = True)
            
            pd.DataFrame.from_dict(clustering_full).to_csv(f'{out_path}/full.csv', index=False)
            pd.DataFrame.from_dict(clustering_apsp).to_csv(f'{out_path}/apsp.csv', index=False)
            pd.DataFrame(graphs_label).to_csv(f'{out_path}/labels.csv', index=False)
    
    