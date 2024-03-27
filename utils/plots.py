from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from itertools import product
import scipy.cluster.hierarchy as hierarchy

class Plot:
    def __init__(self):
        pass

    def perturbation(self, dfs, weights, graphs, metric, perturbation, by, N = 1000, y_axis_range=None):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[metric.id]['mean'].to_numpy()
                s = 0.3*df[metric.id]['std'].to_numpy()
                
                i = np.arange(len(s))
                s[i[i % 5 != 1]] = 0
                x = np.concatenate(([0], np.arange(1, N+1, 5)))

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

        if y_axis_range is not None:
            plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/{metric.name}/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
            
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]
        
        plt.xlabel(f'# {perturbation.name}')
        plt.ylabel(metric.name)
        plt.savefig(f'{out_path}/{by} {perturbation.name}.png', dpi=200)
        plt.clf()


    def perturbation_edges(self, dfs, weights, graphs, e_mes, perturbation, by, N = 1000, y_axis_range=None):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[e_mes]['mean'].to_numpy()
                s = df[e_mes]['std'].to_numpy() 

                i = np.arange(len(s))
                s[i[i % 5 != 1]] = 0
                x = np.concatenate(([0], np.arange(1, N+1, 5)))

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

        if y_axis_range is not None:
            plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/edges/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)

        if e_mes == 'count':
            e_mes = '# Edges'
        elif e_mes == 'size':
            e_mes = 'Size'
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]
        
        plt.xlabel(f'# {perturbation.name}')
        plt.ylabel(e_mes)
        plt.savefig(f'{out_path}/{e_mes} {by} {perturbation.name}.png', dpi=200)
        plt.clf()


    def gaussian_noise(self, dfs, weights, graphs, metric, by, y_axis_range=None):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[metric.id]['mean'].to_numpy()
                s = 0.3*df[metric.id]['std'].to_numpy()
                x = df.index

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

        if y_axis_range is not None:
            plt.ylim(y_axis_range)
        
        out_path = f'plots/gaussian_noise/{metric.name}/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)    
            
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]

        plt.title(f'Gaussian Noise N(0, σ)')
        plt.xlabel('σ')
        plt.ylabel(metric.name)
        plt.savefig(f'{out_path}/{by}.png', dpi=200)
        plt.clf()


    def gaussian_noise_edges(self, dfs, weights, graphs, e_mes, by, y_axis_range=None):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[e_mes]['mean'].to_numpy()
                s = df[e_mes]['std'].to_numpy()
                x = df.index

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='lower left')

        if y_axis_range is not None:
            plt.ylim(y_axis_range)

        out_path = f'plots/gaussian_noise/edges/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)

        if e_mes == 'count':
            e_mes = '# Edges'
        elif e_mes == 'size':
            e_mes = 'Size'
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]
            
        plt.title(f'Gaussian Noise N(0, σ)')
        plt.xlabel('σ')
        plt.ylabel(e_mes)
        plt.savefig(f'{out_path}/{e_mes} {by}.png', dpi=200)
        plt.clf()

    def clustering(self, df, graph, mode, metric, label, N = 54):
        Z = np.reshape(df.to_numpy(), (N-1, -1))
        
        plt.figure(figsize=(25, 10))
        hierarchy.dendrogram(Z, color_threshold=0, labels=label)
        
        label_colors = dict(zip(sorted(set(label)), 
            ['lightcoral', 'red', 'darkred', 
            'lightblue', 'blue', 'darkblue', 
            'lightgreen', 'green', 'darkgreen']))
        
        ax = plt.gca()
        for lbl in ax.get_xmajorticklabels():
            lbl.set_color(label_colors[lbl.get_text()])
            
        out_path = f'plots/clustering/gaussian_noise/{graph}'
        Path(out_path).mkdir(parents = True, exist_ok = True)    
            
        plt.title(f'Clustering {graph} {mode}\n{metric.name}')
        plt.savefig(f'{out_path}/{metric.name} {mode}.png', dpi=200)
        plt.clf()

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