import numpy as np
import pandas as pd
import timeit
import time as ti
from pathlib import Path

from utils.static import *
from utils.graphs import *
from utils.portrait_divergence import all_pairs_dijkstra_path_lengths, _get_unique_path_lengths


def print_time(time, t_iter, c_iter):
    print(f'Time spent               = ' 
          + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
    print(f'Estimated time remaining = ' 
          + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))) + '\n')
    
def read_graphs(path, ref_nodes = list(range(1000))):
                    df_graphs = pd.read_csv(path)
                    graphs =  [nx.from_pandas_edgelist(g_edges, edge_attr=True) 
                               for _, g_edges in df_graphs.groupby('graph_index')]
                    for g in graphs: g.add_nodes_from(ref_nodes)
                    return graphs

class Perturbation:
    def __init__(self):
        self.name = 'perturbation'
        self.out_path_root = 'results/perturbation'

    def out_path(self, graph, weight, perturbation):
        g, w, p = [x.name if not isinstance(x, str) else x for x in [graph, weight, perturbation]]
        return f'{self.out_path_root}/{p}/{g}/{w}'
    
    def write_graphs(
            self, G, weight, perturbation, K = 20, N = 1000, step = 5, 
            save = True, time_printing = False):
        print(f'Perturbation, graph generation: {G.name} {weight.name} {perturbation.name}\n'.upper())
        
        G = weight(G)
        out_path = self.out_path(G, weight, perturbation)

        time = []
        t_iter_sparse = 2 + (N-1) // step
        t_iter = K * t_iter_sparse

        edges = {}
        for sparse in SPARSIFIERS:
            edges[sparse.id] = {}
            for mes in E_MES:
                edges[sparse.id][mes] = [ [] for _ in range(K) ]

        for i in range(K):
            H = G.copy()
            graphs = {sparse.id : [] for sparse in SPARSIFIERS}

            nb_perturbations = [0, 1] + [step] * ((N-1) // step)

            for j, n in enumerate(nb_perturbations):
                start = timeit.default_timer()

                for _ in range(n): 
                    perturbation(H, weight.w())
                for sparse in SPARSIFIERS:
                    sH = sparse(H.copy())
                    graphs[sparse.id].append(sH)
                    edges[sparse.id]['count'][i].append(sH.number_of_edges())
                    edges[sparse.id]['size'][i].append(sH.size(weight='weight'))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    print(f'Iteration {i+1}/{K}, Sparsfication {j+1}/{t_iter_sparse}')
                    print_time(time, t_iter, i*t_iter_sparse + j)

            if save:
                def to_df(graphs):
                    dfs = []
                    for i, graph in enumerate(graphs):
                        df = nx.to_pandas_edgelist(graph)
                        df['graph_index'] = i
                        dfs.append(df)

                    return pd.concat(dfs, ignore_index=True)

                Path(f'{out_path}/graphs/{i}').mkdir(parents = True, exist_ok = True)
                
                for sparse in SPARSIFIERS:
                    to_df(graphs[sparse.id]).to_csv(f'{out_path}/graphs/{i}/{sparse.name}.csv', index=False)

        if save:
            def df_from_dict(d):
                return pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d.items()}, axis=1)

            Path(f'{out_path}/edges').mkdir(parents = True, exist_ok = True)

            for sparse in SPARSIFIERS:
                df_from_dict(edges[sparse.id]).to_csv(f'{out_path}/edges/{sparse.name}.csv', index=False)

    def compute_distances(self, G, weight, perturbation, K = 20, N = 1000, step = 5, 
                          save = True, time_printing = False):
        print(f'Perturbation, distance computation: {G.name} {weight.name} {perturbation.name}\n'.upper())

        time = []

        out_path = self.out_path(G, weight, perturbation)
        read_path = ""
        if not os.path.exists('graphs'):
            read_path = f'{out_path}/'

        distances = {}
        for sparse in SPARSIFIERS:
            distances[sparse.id] = {}
            for m in METRICS:
                distances[sparse.id][m.id] = [ [] for _ in range(K) ]

        for i in range(K):

            for j, sparse in enumerate(SPARSIFIERS):

                graphs = read_graphs(f'{read_path}graphs/{i}/{sparse.name}.csv')
                graphs = graphs[:1] + graphs[1:N//step+1]

                # Precalculating metrics for the original graph
                G = graphs[0].copy()
                mG = {}
                for m in METRICS[:-1]:
                    mG[m.id] = m.m(G)

                for m in METRICS[-1:]:
                    paths_G = all_pairs_dijkstra_path_lengths(G)
                    UPL_G = set(_get_unique_path_lengths(G, paths=paths_G))

                for k, H in enumerate(graphs):
                    start = timeit.default_timer()

                    for m in METRICS[:-1]:
                        distances[sparse.id][m.id][i].append(m(H, mG[m.id]))

                    for m in METRICS[-1:]:
                        distances[sparse.id][m.id][i].append(m(H, G, paths_G = paths_G, UPL_G = UPL_G))


                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        print(f'Iteration {i+1}/{K}, Sparsifier {j+1}/{len(SPARSIFIERS)}, Graph {k+1}/{len(graphs)}')
                        print_time(time, K * len(SPARSIFIERS) * len(graphs), 
                                      i*len(SPARSIFIERS)*len(graphs) + j*len(graphs) + k)

        if save:
            def df_from_dict(d):
                return pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d.items()}, axis=1)

            Path(f'{out_path}/distances').mkdir(parents = True, exist_ok = True)

            for sparse in SPARSIFIERS:
                df_from_dict(distances[sparse.id]).to_csv(f'{out_path}/distances/{sparse.name}.csv', index=False)


class GaussianNoise:
    def __init__(self):
        self.name = 'gaussian noise'
        self.out_path_root = 'results/gaussian_noise'

    def out_path(self, graph, weight):
        g, w = [x.name if not isinstance(x, str) else x for x in [graph, weight]]
        return f'{self.out_path_root}/{g}/{w}'

    def __call__(
            self, G, weight, sigmas = np.linspace(0, 0.1, 20+1).tolist(), K = 20, 
            time_printing = False, save = True):
        print(f'Gaussian noise : {G.name} {weight.name}\n'.upper())

        G = weight(G)
        time = []

        distances, edges = {}, {}
        for sparse in SPARSIFIERS:
            distances[sparse.id] = {σ : {m.id : [] for m in METRICS} for σ in sigmas}
            edges[sparse.id]     = {σ : {mes : [] for mes in E_MES}  for σ in sigmas}

        for i, σ in enumerate(sigmas):
            for j in range(K):
                start = timeit.default_timer()

                H1 = add_gaussian_noise(G.copy(), σ, weight.max)
                H2 = add_gaussian_noise(G.copy(), σ, weight.max)

                for sparse in SPARSIFIERS:
                    sH1, sH2 = sparse(H1.copy()), sparse(H2.copy())
                    for sH in [sH1, sH2]:
                        edges[sparse.id][σ]['count'].append(sH.number_of_edges())
                        edges[sparse.id][σ]['size'].append(sH.size(weight='weight'))

                    for m in METRICS    :
                        distances[sparse.id][σ][m.id].append(m(sH1, sH2))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    print(f'Sigma {i+1}/{len(sigmas)}, Iteration {j+1}/{K}')
                    print_time(time, len(sigmas) * K, i*K + j)

        if save:
            def df_from_dict(d):
                return pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                            for σ, a in d.items()}, axis=0).reset_index(level=1, drop=True)

            out_path = self.out_path(G, weight)
            Path(f'{out_path}/edges').mkdir(parents = True, exist_ok = True)
            Path(f'{out_path}/distances').mkdir(parents = True, exist_ok = True)

            for sparse in SPARSIFIERS:
                df_from_dict(distances[sparse.id]).to_csv(f'{out_path}/distances/{sparse.name}.csv')
                df_from_dict(edges[sparse.id]).to_csv(f'{out_path}/edges/{sparse.name}.csv')

class Clustering:
    def __init__(self):
        pass

    def distance_matrix(self, graphs, metric, time_printing = False):
        
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


    def DWG_graphs(
            self, Graph, weights, σ = 0.05, K = 3, N = 6, time_printing = False):

        graphs_full = []
        graphs_apsp = []
        graphs_label = []

        G, G_name = Graph
        G = G()

        print(f'DWG - {G_name}-graphs initialization\n'.upper())

        if time_printing:
            time = []
            t_iter = len(weights)*K*N

        for i, w in enumerate(weights):
            for j in range(K):
                H = w(G.copy())
                for k in range(N):
                    start = timeit.default_timer()

                    H_full = add_gaussian_noise(H.copy(), σ, w.max)
                    H_apsp = apsp(H_full)
                    graphs_full.append(H_full)
                    graphs_apsp.append(H_apsp)
                    graphs_label.append((w.name, j))

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        c_iter = i*K*N + j*N + k
                        print(f'{w.name}, Iteration (weight) nb {j}, Iteration (noise) nb {k}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                        print()

        return graphs_full, graphs_apsp, graphs_label

    def GDW_graphs(
            self, Graph, weights, K = 3, N = 6, time_printing = False):
        graphs_full = []
        graphs_apsp = []
        graphs_label = []

        G, G_name = Graph

        print(f'GDW - {G_name}-graphs initialization\n'.upper())

        if time_printing:
            time = []
            t_iter = len(weights)*K*N

        for i in range(K):
            G_i = G(s = FIXED_SEED[G_name][i])
            for j, w in enumerate(weights):
                for k in range(N):
                    start = timeit.default_timer()
                    H_full = w(G_i.copy())
                    H_apsp = apsp(H_full)
                    graphs_full.append(H_full)
                    graphs_apsp.append(H_apsp)
                    graphs_label.append((f'{G_name} {i}', w.name))

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        c_iter = i*K*N + j*N + k
                        print(f'Seed {i}, {w.name}, Iteration (weight) nb {k}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                        print()

        return graphs_full, graphs_apsp, graphs_label

    def GGD_graphs(
            self, Graphs, weights, N = 6, time_printing = False):
        graphs_full = []
        graphs_apsp = []
        graphs_label = []

        print(f'GGD - Graphs initialization\n'.upper())

        if time_printing:
            time = []
            t_iter = len(Graphs) * len(weights) * N

        for i, (G, G_name) in enumerate(Graphs):
            for j in range(N):
                G_j = G(s = FIXED_SEED[G_name][j])
                for k, w in enumerate(weights):
                    start = timeit.default_timer()
                    H_full = w(G_j.copy())
                    H_apsp = apsp(H_full)
                    graphs_full.append(H_full)
                    graphs_apsp.append(H_apsp)
                    graphs_label.append((G_name, f'{j}'))

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        c_iter = i*len(weights)*N + j*len(weights) + k
                        print(f'{G_name}, Seed {j}, {w.name}')
                        print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                        print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                        print()

        return graphs_full, graphs_apsp, graphs_label



    