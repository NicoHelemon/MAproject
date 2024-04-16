from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from itertools import product
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import euclidean

from utils.static import *

def pretty_upper_bound(n, λ = 1.2):
    assert n > 0

    n *= λ
    k = 0
    while n < 1 or n >= 10:
        if n < 1:
            n *= 10
            k -= 1
        else:
            n /= 10
            k += 1

    return round(n) * 10**k

class Plot:
    def __init__(self):
        pass


    def perturbation_distances(
            self, dfs, graph, weight, metric, perturbation, N = 1000, y_axis_range=None):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][metric.id]['mean'].to_numpy()
            s = 0.3*df[sparse][metric.id]['std'].to_numpy()
            i = np.arange(len(s))
            s[i[i % 5 != 1]] = 0
            x = np.concatenate(([0], np.arange(1, N+1, 5)))

            plt.errorbar(x, m, yerr = s, linestyle='--',
                         label = sparse, color=SPARSIFIER_COLORS[sparse])

        plt.title(f'{perturbation} on {weight} {graph}')
        plt.xlabel(f'# {perturbation}')
        plt.ylabel(metric.name)
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/{perturbation}/{metric.name}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.clf()

    def perturbation_edges(
            self, dfs, graph, weight, e_mes, perturbation, N = 1000, y_axis_range=None):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][e_mes]['mean'].to_numpy()
            s = df[sparse][e_mes]['std'].to_numpy() 
            i = np.arange(len(s))
            s[i[i % 5 != 1]] = 0
            x = np.concatenate(([0], np.arange(1, N+1, 5)))

            plt.errorbar(x, m, yerr = s, linestyle='--', 
                         label = sparse, color = SPARSIFIER_COLORS[sparse])
            
        plt.title(f'{perturbation} on {weight} {graph}')
        plt.xlabel(f'# {perturbation}')
        plt.ylabel('Size' if e_mes == 'size' else '# Edges')
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/{perturbation}/edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{graph} {weight}.png', dpi=200)
        plt.clf()

    def perturbation_deviation(self, D_g_dists, metric, perturbation):
        TS_NAME = S_NAME[1:]

        distances = {}
        for s in TS_NAME:
            s_distances = []
            for (g, w) in product(G_NAME, W_NAME):
                full_vec       = D_g_dists[(g, w)]['full'][metric.id]['mean']
                sparsifier_vec = D_g_dists[(g, w)][s][metric.id]['mean']
                s_distances.append(euclidean(full_vec, sparsifier_vec))
            distances[s] = s_distances

        TS_NAME.sort(key=lambda x: np.mean(distances[x]))

        _, ax = plt.subplots()
        
        graphs = [f'{g} {w}' for (g, w) in product(G_NAME, W_NAME)]
        k = len(graphs)
        ind = np.arange(k)
        width = 1 / k

        for i, s in enumerate(TS_NAME):
            ax.bar(ind + i*width, distances[s], width, label=s, color=SPARSIFIER_COLORS[s])

        plt.xticks(ind + width*(len(TS_NAME)//2), graphs, rotation = 45)    

        ax.set_title(f'Sparsified graphs vs Full graph deviation\n w.r.t {perturbation} distances')
        ax.set_xlabel('Graphs')
        ax.set_ylabel(metric.name)
        ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()

        out_path = f'plots/perturbation/{perturbation}/{metric.name}'
        plt.savefig(f'{out_path}/deviation.png', dpi=200)
        plt.clf()


    def gaussian_noise_distances(
            self, dfs, graph, weight, metric, y_axis_range=None):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][metric.id]['mean'].to_numpy()
            s = 0.3*df[sparse][metric.id]['std'].to_numpy()
            x = df[sparse].index

            plt.errorbar(x, m, yerr = s, linestyle='--',
                         label = sparse, color=SPARSIFIER_COLORS[sparse])   

        plt.title(f'Gaussian Noise N(0, σ) on {weight} {graph}')
        plt.xlabel('σ')
        plt.ylabel(metric.name)
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/gaussian_noise/{metric.name}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.clf()

    def gaussian_noise_edges(
            self, dfs, graph, weight, e_mes, y_axis_range=None):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][e_mes]['mean'].to_numpy()
            s = df[sparse][e_mes]['std'].to_numpy()
            x = df[sparse].index

            plt.errorbar(x, m, yerr = s, linestyle='--', 
                         label = sparse, color = SPARSIFIER_COLORS[sparse])
            
        plt.title(f'Gaussian Noise N(0, σ) on {weight} {graph}')
        plt.xlabel('σ')
        plt.ylabel('Size' if e_mes == 'size' else '# Edges')
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/gaussian_noise/edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{graph} {weight}.png', dpi=200)
        plt.clf()

    def gaussian_noise_deviation(self, D_g_dists, metric):
        TS_NAME = S_NAME[1:]

        distances = {}
        for s in TS_NAME:
            s_distances = []
            for (g, w) in product(G_NAME, W_NAME):
                full_vec       = D_g_dists[(g, w)]['full'][metric.id]['mean']
                sparsifier_vec = D_g_dists[(g, w)][s][metric.id]['mean']
                s_distances.append(euclidean(full_vec, sparsifier_vec))
            distances[s] = s_distances

        TS_NAME.sort(key=lambda x: np.mean(distances[x]))

        _, ax = plt.subplots()
        
        graphs = [f'{g} {w}' for (g, w) in product(G_NAME, W_NAME)]
        k = len(graphs)
        ind = np.arange(k)
        width = 1 / k

        for i, s in enumerate(TS_NAME):
            ax.bar(ind + i*width, distances[s], width, label=s, color=SPARSIFIER_COLORS[s])

        plt.xticks(ind + width*(len(TS_NAME)//2), graphs, rotation = 45)   

        ax.set_title(f'Sparsified graphs vs Full graph deviation\n w.r.t gaussian noise distances')
        ax.set_xlabel('Graphs')
        ax.set_ylabel(metric.name)
        ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()

        out_path = f'plots/gaussian_noise/{metric.name}'
        plt.savefig(f'{out_path}/deviation.png', dpi=200)
        plt.clf()



    def clustering(self, df, graphs_spec, metric, labels, graph = None, N = 54):
        labels = [f'{l1} {l2}' for (l1, l2) in labels]

        link_full = hierarchy.linkage(df['full'][metric.id].to_numpy(),
                                      method='ward', optimal_ordering=True)
        link_apsp = hierarchy.linkage(df['apsp'][metric.id].to_numpy(),
                                      method='ward', optimal_ordering=True)
        
        max_height = 1.05 * max(np.max(link_full[:, 2]), np.max(link_apsp[:, 2]))

        if graphs_spec == 'GGD':
            out_path = f'plots/clustering/{graphs_spec}/'
            label_colors = dict(zip(sorted(set(labels)), LABEL_COLORS['3 x 3']))
        else:
            out_path = f'plots/clustering/{graphs_spec}/{graph}'
            label_colors = dict(zip(sorted(set(labels)), LABEL_COLORS['3 x 3']))

        for mode, link in zip(MODES, [link_full, link_apsp]):
        
            plt.figure(figsize=(25, 10))
            hierarchy.dendrogram(link, color_threshold=0, labels=labels)
            plt.ylim(0, max_height)
            
            ax = plt.gca()
            for lbl in ax.get_xmajorticklabels():
                lbl.set_color(label_colors[lbl.get_text()])
            
            Path(out_path).mkdir(parents = True, exist_ok = True)    
                
            if graph is not None:
                plt.title(f'Clustering {graph} {mode}\n{metric.name}')
            else:
                plt.title(f'Clustering {mode}\n{metric.name}')
            plt.savefig(f'{out_path}/{metric.name} {mode}.png', dpi=200)
            plt.clf()