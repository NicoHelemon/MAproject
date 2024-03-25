from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from itertools import product
import scipy.cluster.hierarchy as hierarchy

class Plot:
    def __init__(self):
        pass

    def perturbation(self, dfs, weights, graphs, metric, perturbation, by):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[metric.id]['mean'].to_numpy()
                s = 0.3*df[metric.id]['std'].to_numpy()
                
                i = np.arange(len(s))
                s[i[i % 5 != 1]] = 0
                x = np.concatenate(([0], np.arange(1, 1001, 5)))

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

        out_path = f'plots/perturbation/{metric.name}/by_{by}/'
        Path(out_path).mkdir(parents = True, exist_ok = True)
            
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]
        
        plt.xlabel(f'# {perturbation.name}')
        plt.ylabel(metric.name)
        plt.savefig(f'{out_path}{by} {perturbation.name}.png', dpi=200)
        plt.clf()


    def perturbation_edges(self, dfs, weights, graphs, e_mes, perturbation, by):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[e_mes]['mean'].to_numpy()
                s = df[e_mes]['std'].to_numpy()
                    
                i = np.arange(len(s))
                s[i[i % 5 != 1]] = 0
                x = np.concatenate(([0], np.arange(1, 1001, 5)))

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

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
        plt.savefig(f'{out_path}{by} {perturbation.name} {e_mes}.png', dpi=200)
        plt.clf()


    def gaussian_noise(self, dfs, weights, graphs, metric, by):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[metric.id]['mean'].to_numpy()
                s = 0.3*df[metric.id]['std'].to_numpy()
                x = df.index

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')
        
        out_path = f'plots/gaussian_noise/{metric.name}/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)    
            
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]

        plt.xlabel('Sigma')
        plt.ylabel(metric.name)
        plt.savefig(f'{out_path}{by}.png', dpi=200)
        plt.clf()


    def gaussian_noise_edges(self, dfs, weights, graphs, e_mes, by):
        keys = list(product(graphs, weights))
        assert len(keys) == 3

        for (graph, weight), c in zip(keys, ['r', 'b', 'g']):
            for (mode, df), ls in zip(dfs[(graph, weight)].items(), ['--', 'dotted']):
                m = df[e_mes]['mean'].to_numpy()
                s = df[e_mes]['std'].to_numpy()
                x = df.index

                plt.errorbar(x, m, yerr = s, color = c, linestyle=ls, label = f'{weight} {graph} {mode}')
                plt.legend(loc='upper left')

        out_path = f'plots/gaussian_noise/by_{by}'
        Path(out_path).mkdir(parents = True, exist_ok = True)

        if e_mes == 'count':
            e_mes = '# Edges'
        elif e_mes == 'size':
            e_mes = 'Size'
        if by == 'weight':
            by = weights[0]
        elif by == 'graph':
            by = graphs[0]
            
        plt.xlabel('Sigma')
        plt.ylabel(e_mes)
        plt.savefig(f'{out_path}{by} {e_mes}.png', dpi=200)
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
            
        out_path = f'plots/clustering/gaussian_noise/{graph}/'
        Path(out_path).mkdir(parents = True, exist_ok = True)    
            
        plt.title(f'Clustering {graph} {mode}\n{metric.name}')
        plt.savefig(f'{out_path}{metric.name} {mode}.png', dpi=200)
        plt.clf()