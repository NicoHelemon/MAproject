import numpy as np
import pandas as pd
import time as ti
import timeit
import json
from pathlib import Path

from utils.static import *
from utils.graphs import *
from utils.portrait_divergence import all_pairs_dijkstra_path_lengths, _get_unique_path_lengths


def print_time(time, t_iter, c_iter):
    time_spent = int(np.sum(time))
    time_remaining = int(np.mean(time) * (t_iter - 1 - c_iter))
    print(f'Time spent               = 0{time_spent // (24*3600)}:'
          + ti.strftime('%H:%M:%S', ti.gmtime(time_spent)))
    print(f'Estimated time remaining = 0{time_remaining // (24*3600)}:'
          + ti.strftime('%H:%M:%S', ti.gmtime(time_remaining)) + '\n')
    
def read_graphs(in_path, ref_nodes = list(range(1000))):
    df_graphs = pd.read_csv(in_path)
    graphs =  [nx.from_pandas_edgelist(g_edges, edge_attr=True) 
                for _, g_edges in df_graphs.groupby('graph_index')]
    for g in graphs: g.add_nodes_from(ref_nodes)
    return graphs

def write_graphs(graphs, out_path):
    dfs = []
    for i, graph in enumerate(graphs):
        df = nx.to_pandas_edgelist(graph)
        df['graph_index'] = i
        dfs.append(df)

    pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False)

class Perturbation:
    def __init__(self):
        self.name = 'perturbation'
        self.out_path_root = 'results/perturbation'

    def out_path(self, graph, weight, perturbation):
        g, w, p = [x.name if not isinstance(x, str) else x for x in [graph, weight, perturbation]]
        return f'{self.out_path_root}/{p}/{g}/{w}'
    
    def __call__(
            self, G, weight, perturbation, K = 20, N = 1000, step = 5, 
            save = True, time_printing = False):
        print(f'Perturbation test: {G.name} {weight.name} {perturbation.name}\n'.upper())
        
        G = weight(G)

        time = []
        t_iter_sparse = 2 + (N-1) // step
        t_iter = K * t_iter_sparse

        edges, distances = {}, {}
        msG = {}
        for sparse in SPARSIFIERS:
            edges[sparse.id] = {}
            distances[sparse.id] = {}

            for mes in E_MES:
                edges[sparse.id][mes] = [ [] for _ in range(K) ]
            for m in METRICS:
                distances[sparse.id][m.id] = [ [] for _ in range(K) ]

            msG[sparse.id] = {}
            sG = sparse(G.copy())

            for m in METRICS[:-1]:
                msG[sparse.id][m.id] = m.m(sG)

            for m in METRICS[-1:]:
                paths = all_pairs_dijkstra_path_lengths(sG)
                UPL = set(_get_unique_path_lengths(sG, paths=paths))
                msG[sparse.id][m.id] = (paths, UPL)

        for i in range(K):
            H = G.copy()

            nb_perturbations = [0, 1] + [step] * ((N-1) // step)

            for j, n in enumerate(nb_perturbations):
                start = timeit.default_timer()

                for _ in range(n): perturbation(H, weight.w())

                for sparse in SPARSIFIERS:
                    sH = sparse(H.copy())

                    edges[sparse.id]['count'][i].append(sH.number_of_edges())
                    edges[sparse.id]['size'][i].append(sH.size(weight='weight'))

                    for m in METRICS[:-1]:
                        distances[sparse.id][m.id][i].append(m(sH, msG[sparse.id][m.id]))

                    for m in METRICS[-1:]:
                        distances[sparse.id][m.id][i].append(m(sH, None, 
                            None, None, *msG[sparse.id][m.id]))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    print(f'Iteration {i+1}/{K}, Sparsification {j+1}/{t_iter_sparse}')
                    print_time(time, t_iter, i*t_iter_sparse + j)

        if save:
            def df_from_dict(d):
                return pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d.items()}, axis=1)

            out_path = self.out_path(G, weight, perturbation)
            Path(f'{out_path}/distances').mkdir(parents = True, exist_ok = True)
            Path(f'{out_path}/edges').mkdir(parents = True, exist_ok = True)

            for sparse in SPARSIFIERS:
                df_from_dict(edges[sparse.id]).to_csv(f'{out_path}/edges/{sparse.name}.csv', index=False)
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
        print(f'Gaussian noise test: {G.name} {weight.name}\n'.upper())

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
        self.name = 'gaussian noise'
        self.out_path_root = 'results/clustering'

    def __call__(
            self, sparse, time_printing = False, save = True, N = None):
        
        print(f'Clustering test: {sparse.name}\n'.upper())

        read_path = "" if os.path.exists('graphs.csv') else f'{self.out_path_root}/'
        graphs = read_graphs(f'{read_path}graphs.csv')
        if N is not None: graphs = graphs[:N]

        m_sgraphs = {m.id : [] for m in METRICS}
        time = []

        for i, G in enumerate(graphs):
            start = timeit.default_timer()
            sG = sparse(G)

            for m in METRICS[:-1]:
                m_sgraphs[m.id].append(m.m(sG))
            for m in METRICS[-1:]:
                paths = all_pairs_dijkstra_path_lengths(sG)
                UPL   = set(_get_unique_path_lengths(sG, paths=paths))
                m_sgraphs[m.id].append((paths, UPL))

            if time_printing:
                time.append(timeit.default_timer() - start)
                print(f'Sparsification {i+1}/{len(graphs)}')
                print_time(time, len(graphs), i)

        distance_matrices = self.distance_matrices(m_sgraphs, time_printing = time_printing)

        if save:
            Path(f'{self.out_path_root}').mkdir(parents = True, exist_ok = True)
            pd.DataFrame.from_dict(distance_matrices).to_csv(f'{self.out_path_root}/{sparse.name}.csv', index=False)

    def write_graphs(
            self, weight_n_sample = 2, graph_n_sample = 2, gn_n_sample = 2, σ = 0.05, time_printing = False):
        
        print(f'Graphs generation\n'.upper())
        
        graphs = []
        labels = []
        time = []
        t_iter = np.prod((weight_n_sample, len(WEIGHTS), graph_n_sample, len(GRAPHS), gn_n_sample))

        c_iter = -1
        for i in range(weight_n_sample):
            for j, w in enumerate(WEIGHTS):
                for k in range(graph_n_sample):
                    start = timeit.default_timer()
                    for g in GRAPHS:
                        G = w(g(s = FIXED_SEED[g.__name__][k]))
                        for l in range(gn_n_sample):
                            c_iter += 1
                            graphs.append(add_gaussian_noise(G.copy(), σ, w.max))
                            labels.append({
                                'weight' : w.name,
                                'weight_sample' : i,
                                'graph' : G.name,
                                'graph_sample' : k,
                                'graph_noise_sample' : l
                            })

                    if time_printing:
                        time.append(timeit.default_timer() - start)
                        print(f'Graph {c_iter+1}/{t_iter}')
                        k = len(GRAPHS) * graph_n_sample
                        print_time(time, t_iter/k, (c_iter+1)/k-1)

        Path(f'{self.out_path_root}').mkdir(parents = True, exist_ok = True)
        write_graphs(graphs, f'{self.out_path_root}/graphs.csv')
        with open(f'{self.out_path_root}/labels.json', "w") as json_file:
            json.dump(labels, json_file)

    def distance_matrices(
            self, m_graphs, time_printing = False):
        
        print(f'Distance matrices computation\n'.upper())

        N = len(m_graphs['lap'])
        time = []
        t_iter = N*(N - 1) // 2

        distance_matrices = {m.id : [] for m in METRICS}

        c_iter = -1
        for i in range(N):
            for j in range(i+1, N):
                start = timeit.default_timer()
                c_iter += 1

                for m in METRICS[:-1]:
                    distance_matrices[m.id].append(m(m_graphs[m.id][i], m_graphs[m.id][j]))
                for m in METRICS[-1:]:
                    distance_matrices[m.id].append(m(None, None, 
                                                     *m_graphs[m.id][i], *m_graphs[m.id][j]))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    if (c_iter+1) % 5 == 0:
                        print(f'Comparison {c_iter+1}/{t_iter}')
                        print_time(time, t_iter, c_iter)

        return distance_matrices



    