import numpy as np
import pandas as pd
import time as ti
import timeit
import json
from pathlib import Path
from itertools import product

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
    g_edges =  [nx.from_pandas_edgelist(g_edges.iloc[:, :3], edge_attr=True) 
                for _, g_edges in df_graphs.groupby('graph_index')]
    graphs = []
    for ge in g_edges:
        g = nx.Graph()
        g.add_nodes_from(ref_nodes)
        g.add_edges_from(ge.edges(data=True))
        graphs.append(g)
    return graphs

def write_graphs(graphs, out_path):
    dfs = []
    for i, graph in enumerate(graphs):
        df = nx.to_pandas_edgelist(graph)
        df['graph_index'] = i
        dfs.append(df)

    pd.concat(dfs, ignore_index=True).to_csv(out_path, index=False)

def stats(G, edge_stats = True):
    s = {}
    if edge_stats:
        s['count'] = G.number_of_edges()
        s['size'] = G.size(weight='weight')
        s['components'] = nx.number_connected_components(G)
    for m in METRICS[:-1]:
        s[m.id] = (m.m(G),)
    for m in METRICS[-1:]:
        path = all_pairs_dijkstra_path_lengths(G)
        UPL = set(_get_unique_path_lengths(G, paths=path))
        s[m.id] = (None, path, UPL)
    return s

class Perturbation:
    def __init__(self):
        self.name = 'perturbation'
        self.out_path_root = 'results/perturbation'

    def out_path(self, graph, weight, perturbation):
        g, w, p = [x.name if not isinstance(x, str) else x for x in [graph, weight, perturbation]]
        return f'{self.out_path_root}/{p}/{g}/{w}'
    
    def __call__(
            self, G, weight, perturbation, K = 20, N = N_PERTURBATIONS, step = 5, 
            save = True, time_printing = False):
        print(f'Perturbation test: {G.name} {weight.name} {perturbation.name}\n'.upper())
        
        G = weight(G)

        time = []
        t_iter_sparse = 2 + (N-1) // step
        t_iter = K * t_iter_sparse

        edges, distances = {}, {}
        SG = {}
        for sparse in SPARSIFIERS:
            edges[sparse.id] = {}
            distances[sparse.id] = {}

            for mes in E_MES:
                edges[sparse.id][mes] = [ [] for _ in range(K) ]
            for m in METRICS:
                distances[sparse.id][m.id] = [ [] for _ in range(K) ]

            if sparse.name in 'Effective Resistance':
                Re = resistance_distance(G)
                SG[sparse.id] = [stats(sparse(G.copy(), Re = Re)) for _ in range(sparse.rep)]
            else:
                SG[sparse.id] = [stats(sparse(G.copy())) for _ in range(sparse.rep)]

        for i in range(K):
            H = G.copy()

            nb_perturbations = [0, 1] + [step] * ((N-1) // step)

            for j, n in enumerate(nb_perturbations):
                start = timeit.default_timer()

                for _ in range(n): perturbation(H, weight.w())

                for sparse in SPARSIFIERS:
                    dist = {m.id : [] for m in METRICS}

                    if sparse.name in 'Effective Resistance':
                        Re = resistance_distance(H)
                        SH = [stats(sparse(H.copy(), Re = Re)) for _ in range(sparse.rep)]
                    else:
                        SH = [stats(sparse(H.copy())) for _ in range(sparse.rep)]

                    for (sG, sH) in product(SG[sparse.id], SH):
                        for m in METRICS:
                            dist[m.id].append(m(*sH[m.id], *sG[m.id]))
                            
                    edges[sparse.id]['count'][i].append(np.mean([sH['count'] for sH in SH]))
                    edges[sparse.id]['size'][i].append(np.mean([sH['size'] for sH in SH]))
                    edges[sparse.id]['components'][i].append(np.mean([sH['components'] for sH in SH]))
                    for m in METRICS:
                        distances[sparse.id][m.id][i].append(np.mean(dist[m.id]))

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
                    if sparse.name in 'Effective Resistance':
                        Re1 = resistance_distance(H1)
                        Re2 = resistance_distance(H2)
                        SH1 = [stats(sparse(H1.copy(), Re = Re1)) for _ in range(sparse.rep)]
                        SH2 = [stats(sparse(H2.copy(), Re = Re2)) for _ in range(sparse.rep)]
                    else:
                        SH1 = [stats(sparse(H1.copy())) for _ in range(sparse.rep)]
                        SH2 = [stats(sparse(H2.copy())) for _ in range(sparse.rep)]

                    dist = {m.id : [] for m in METRICS}
                    for sH1, sH2 in product(SH1, SH2):
                        for m in METRICS:
                            dist[m.id].append(m(*sH1[m.id], *sH2[m.id]))

                    for SH in [SH1, SH2]:
                        edges[sparse.id][σ]['count'].append(np.mean([sH['count'] for sH in SH]))
                        edges[sparse.id][σ]['size'].append(np.mean([sH['size'] for sH in SH]))
                        edges[sparse.id][σ]['components'].append(np.mean([sH['components'] for sH in SH]))
                    
                    for m in METRICS:
                        distances[sparse.id][σ][m.id].append(np.mean(dist[m.id]))

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
    def __init__(self, iteration = 0):
        self.name = 'gaussian noise'
        self.out_path_root = f'results/clustering/{iteration}'

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
            sG = stats(sparse(G), edge_stats = False)

            for m in METRICS: m_sgraphs[m.id].append(sG[m.id])

            if time_printing:
                time.append(timeit.default_timer() - start)
                print(f'Sparsification {i+1}/{len(graphs)}')
                print_time(time, len(graphs), i)

        distance_matrices = self.distance_matrices(m_sgraphs, time_printing = time_printing)

        if save:
            Path(f'{self.out_path_root}').mkdir(parents = True, exist_ok = True)
            pd.DataFrame.from_dict(distance_matrices).to_csv(f'{self.out_path_root}/{sparse.name}.csv', index=False)

    def generate_graphs(
            self, weight_n_sample = 2, graph_n_sample = 2, gn_n_sample = 2, σ = 0.05, 
            time_printing = False, save = True):
        
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

        if save:
            Path(f'{self.out_path_root}').mkdir(parents = True, exist_ok = True)
            write_graphs(graphs, f'{self.out_path_root}/graphs.csv')
            with open(f'{self.out_path_root}/labels.json', "w") as json_file:
                json.dump(labels, json_file)

        return graphs

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

                for m in METRICS:
                    distance_matrices[m.id].append(m(*m_graphs[m.id][i], *m_graphs[m.id][j]))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    if (c_iter+1) % 5 == 0:
                        print(f'Comparison {c_iter+1}/{t_iter}')
                        print_time(time, t_iter, c_iter)

        return distance_matrices



    