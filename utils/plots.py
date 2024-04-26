import numpy as np
from pathlib import Path
from itertools import product
from scipy.spatial.distance import euclidean as euclidean_distance

import radialtree as rt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
from sklearn.metrics import auc
from scipy.stats import expon, uniform, lognorm

import timeit

from utils.static import *
from utils.tests import *

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

def precisions_recalls(distances, labels, thresholds):
    precisions = []
    recalls = []
    for threshold in thresholds:
        predictions = distances < threshold
        tp = np.sum(predictions[labels == 1])
        fp = np.sum(predictions[labels == 0])
        fn = np.sum(labels) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def class_matrix(labels, class_characterisation):
    n = len(labels)
    D = []
    
    for i in range(n):
        for j in range(i+1, n):
            of_same_class = all(labels[i][c] == labels[j][c] for c in class_characterisation)
            D.append(of_same_class)
    
    return np.array(D)

class Plot:
    def __init__(self):
        pass

    def perturbation_distances(
            self, dfs, graph, weight, metric, perturbation, N = 1000, 
            y_axis_range=None, g_first = True):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][metric.id]['mean'].to_numpy()
            s = 0.3*df[sparse][metric.id]['std'].to_numpy()
            i = np.arange(len(s))
            s[i[i % 5 != 1]] = 0
            x = np.concatenate(([0], np.arange(1, N+1, 5)))

            plt.errorbar(x, m, yerr = s, linestyle='-',
                         label = sparse, color=S_COLORS[sparse])
            
        if g_first: graph, weight = weight, graph

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
            self, dfs, graph, weight, e_mes, perturbation, N = 1000,
            y_axis_range=None, g_first = True):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][e_mes]['mean'].to_numpy()
            s = df[sparse][e_mes]['std'].to_numpy() 
            i = np.arange(len(s))
            s[i[i % 5 != 1]] = 0
            x = np.concatenate(([0], np.arange(1, N+1, 5)))

            plt.errorbar(x, m, yerr = s, linestyle='-', 
                         label = sparse, color = S_COLORS[sparse])

        if g_first: graph, weight = weight, graph
            
        plt.title(f'{perturbation} on {weight} {graph}')
        plt.xlabel(f'# {perturbation}')
        plt.ylabel('Size' if e_mes == 'size' else '# Edges')
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/{perturbation}/Edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.clf()

    def perturbation_deviation(
            self, D_dists, metric, perturbation, log = True):
        
        TS_NAME = S_NAME[1:]

        distances = {}
        for s in TS_NAME:
            s_distances = []
            for (w, g) in product(W_NAME, G_NAME):
                full_vec       = D_dists[(g, w)]['Full'][metric.id]['mean']
                sparsifier_vec = D_dists[(g, w)][s][metric.id]['mean']
                s_distances.append(euclidean_distance(full_vec, sparsifier_vec))
            distances[s] = s_distances

        TS_NAME.sort(key=lambda x: np.mean(distances[x]))

        _, ax = plt.subplots()
        
        graphs = [f'{w} {g}' for (w, g) in product(W_NAME, G_NAME)]
        k = len(graphs)
        ind = np.arange(k)
        width = 1 / k

        for i, s in enumerate(TS_NAME):
            ax.bar(ind + i*width, distances[s], width, label=s, color=S_COLORS[s])

        plt.xticks(ind + width*(len(TS_NAME)//2), graphs, rotation = 45)    

        ax.set_title(f'Distances deviation of sparsified graphs from full graphs\n'
                     + f'on {perturbation.lower()} test')
        ax.set_xlabel('Graphs')
        ax.set_ylabel(metric.name)
        if log: ax.set_yscale('log')
        ax.legend(loc='lower left')
        plt.tight_layout()

        out_path = f'plots/perturbation/_Deviation'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{metric.name} {perturbation}.png', dpi=200)

        out_path = f'plots/perturbation/{perturbation}/{metric.name}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/_Deviation.png', dpi=200)
        plt.clf()


    def gaussian_noise_distances(
            self, dfs, graph, weight, metric,
            y_axis_range=None, g_first = True):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][metric.id]['mean'].to_numpy()
            s = 0.3*df[sparse][metric.id]['std'].to_numpy()
            x = df[sparse].index

            plt.errorbar(x, m, yerr = s, linestyle='-',
                         label = sparse, color=S_COLORS[sparse])

        if g_first: graph, weight = weight, graph

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
            self, dfs, graph, weight, e_mes,
            y_axis_range=None, g_first = True):
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][e_mes]['mean'].to_numpy()
            s = df[sparse][e_mes]['std'].to_numpy()
            x = df[sparse].index

            plt.errorbar(x, m, yerr = s, linestyle='-', 
                         label = sparse, color = S_COLORS[sparse])
            
        if g_first: graph, weight = weight, graph
            
        plt.title(f'Gaussian Noise N(0, σ) on {weight} {graph}')
        plt.xlabel('σ')
        plt.ylabel('Size' if e_mes == 'size' else '# Edges')
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/gaussian_noise/Edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.clf()

    def gaussian_noise_deviation(
            self, D_dists, metric, log = True):
        
        TS_NAME = S_NAME[1:]

        distances = {}
        for s in TS_NAME:
            s_distances = []
            for (w, g) in product(W_NAME, G_NAME):
                full_vec       = D_dists[(g, w)]['Full'][metric.id]['mean']
                sparsifier_vec = D_dists[(g, w)][s][metric.id]['mean']
                s_distances.append(euclidean_distance(full_vec, sparsifier_vec))
            distances[s] = s_distances

        TS_NAME.sort(key=lambda x: np.mean(distances[x]))

        _, ax = plt.subplots()
        
        graphs = [f'{w} {g}' for (w, g) in product(W_NAME, G_NAME)]
        k = len(graphs)
        ind = np.arange(k)
        width = 1 / k

        for i, s in enumerate(TS_NAME):
            ax.bar(ind + i*width, distances[s], width, label=s, color=S_COLORS[s])

        plt.xticks(ind + width*(len(TS_NAME)//2), graphs, rotation = 45)   

        ax.set_title(f'Distances deviation of sparsified graphs from full graphs\n'
                     + f'on gaussian noise test')
        ax.set_xlabel('Graphs')
        ax.set_ylabel(metric.name)
        if log: ax.set_yscale('log')
        ax.legend(loc='lower left')
        plt.tight_layout()

        out_path = f'plots/gaussian_noise/_Deviation'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{metric.name}.png', dpi=200)

        out_path = f'plots/gaussian_noise/{metric.name}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/_Deviation.png', dpi=200)
        plt.clf()


    def clustering(
            self, distances_matrix, metric, labels):

        linkage = {}
        for sparse in S_NAME:
            linkage[sparse] = hierarchy.linkage(distances_matrix[sparse][metric.id].to_numpy(),
                                     method='ward', optimal_ordering=True)

        display_labels = [" ".join(str(i) for i in list(l.values())) for l in labels]
        colors_dict = {"Weight": [W_COLORS[l['weight']] for l in labels],
                       "Graph" : [G_COLORS[l['graph']] for l in labels]}

        for sparse in S_NAME:
            plt.figure(figsize=(25, 10))
            D = hierarchy.dendrogram(linkage[sparse], 
                                     labels = display_labels, 
                                     color_threshold=0.3*max(linkage[sparse][:, 2]),
                                     leaf_font_size=3)
            rt.plot(D, colorlabels = colors_dict, colorlabels_legend = COLORS_LEGENDS)
            #plt.title(f'Clustering on {sparse} graphs\nMetric: {metric.name}')

            out_path = f'plots/clustering/{metric.name}/Dendrogram'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/{sparse}.png', dpi=400)
            plt.clf()

    def clustering_precision_recall(
            self, D_d, metric, labels, class_characterisation):
        
        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)

        pra = {}
        
        for sparse in S_NAME:
            distance_m = D_d[sparse][metric.id].to_numpy()
            distance_m = distance_m / np.max(distance_m)

            precisions, recalls = precisions_recalls(distance_m, class_m, thresholds)
            aupr = auc(recalls, precisions)

            pra[sparse] = (precisions, recalls, aupr)

        S_NAME.sort(key=lambda s: pra[s][2], reverse=True)

        for sparse in S_NAME:
            precisions, recalls, aupr = pra[sparse]

            plt.plot(recalls, precisions, marker=',', 
                     label=f'(AUPR = {aupr:.3f}) {sparse}', color=S_COLORS[sparse])

        rdm_chance = np.mean(np.array(class_m) == 1)
        plt.plot([0, 1], [rdm_chance, rdm_chance], 
                 linestyle=':', color='black',
                 label=f'(AUPR = {rdm_chance:.3f}) random classification')
        
        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.title(f'Precision-Recall curve\n'
                  +f'Class: {class_str}\n'
                  +f'Metric: {metric.name}', fontsize='small')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left', fontsize='x-small')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)

        out_path = f'plots/clustering/{metric.name}/Precision-Recall'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{class_str}.png', dpi=200)

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{metric.name}.png', dpi=200)
        plt.clf()

    def clustering_precision_recall_3D(
            self, D_d, metric, labels, class_characterisation):

        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)

        pra = {}

        for sparse in S_NAME:
                distance_m = D_d[sparse][metric.id].to_numpy()
                distance_m = distance_m / np.max(distance_m)

                precisions, recalls = precisions_recalls(distance_m, class_m, thresholds)
                aupr = auc(recalls, precisions)

                pra[sparse] = (precisions, recalls, aupr)

        S_NAME.sort(key=lambda s: pra[s][2], reverse=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for sparse in S_NAME:
                precisions, recalls, aupr = pra[sparse]

                ax.plot(recalls, thresholds, precisions, linestyle='-', 
                        label=f'(AUPR = {aupr:.3f}) {sparse}', color=S_COLORS[sparse])
                ax.plot(recalls, thresholds, np.zeros_like(precisions), 
                        linestyle='-', color=S_COLORS[sparse], alpha=0.3)
                

        rdm_chance = np.mean(np.array(class_m) == 1)
        ax.plot([0, 1], [0, 1], [rdm_chance, rdm_chance],
                linestyle=':', color='black',
                label=f'(AUPR = {rdm_chance:.3f}) random classification')
        ax.plot([0, 1], [0, 1], [0, 0],
                linestyle=':', color='black', alpha=0.3)

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        ax.set_title(f'Precision-Recall curve\n'
                +f'Class: {class_str}\n'
                +f'Metric: {metric.name}', fontsize='small')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Precision')
        ax.legend(loc='upper right', fontsize='xx-small')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.grid(True)

        out_path = f'plots/clustering/{metric.name}/Precision-Recall'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/3D {class_str}.png', dpi=200)

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/3D {metric.name}.png', dpi=200)
        plt.clf()

    def sparsifiers_speed(
            self, graphs = None):
        
        if graphs is None:
            graphs = read_graphs('results/clustering/graphs.csv')
            #TODO to change
            #graphs = Clustering().generate_graphs(save = False)
        
        TS_NAME = S_NAME[1:]

        avg_time = {}
        for sparse in TS_NAME:
            start = timeit.default_timer()
            for graph in graphs:
                S_MAP[sparse](graph)
            avg_time[sparse] = (timeit.default_timer() - start) / len(graphs)

        TS_NAME.sort(key=lambda s: avg_time[s])

        plt.bar(TS_NAME, [avg_time[s] for s in TS_NAME], color=[S_COLORS[s] for s in TS_NAME])
        plt.xlabel('Sparsifier')
        plt.ylabel('Average Time (s)')
        plt.title(f'Average sparsification time per graph ({len(graphs)} graphs)', fontsize='small')
        plt.xticks(fontsize='small')

        out_path = f'plots/_graphs'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/Sparsification speed.png', dpi=200)
        plt.clf()

    def graph(
            self, G, G_name, pos = None, sparse = None, e_width_from_weight = True, 
            node_color = 'black', size = 3, alpha_full = None, highlighting_factor = 1):
        
        plt.figure(figsize=(30, 24))

        if pos is None:
            pos = nx.spring_layout(G)

        edges = list(G.edges())

        if sparse.name == 'Full': sparse = None
        sparse_name = sparse.name if sparse is not None else '_Full'

        if alpha_full is None:
                alpha_full = 0.5 if sparse is not None else 1

        if e_width_from_weight:
            
            if sparse is not None:
                sG = sparse(G)
                sG_edges = list(sG.edges())
                edges = list(set(edges) - set(sG_edges))

                sG_width = [highlighting_factor * size * inverse_weight(sG[u][v]['weight']) for u, v in sG_edges]
                nx.draw_networkx_edges(G, pos, sG_edges, sG_width, 
                                   edge_color=S_COLORS[sparse.name])

            width = [size * inverse_weight(G[u][v]['weight']) for u, v in edges]

        else:
            width = size*0.5


        nx.draw_networkx_edges(G, pos, edges, width, 
                                edge_color='black', alpha=alpha_full)
        nx.draw_networkx_nodes(G, pos, node_size=size*8, node_color=node_color)
            
        out_path = f'plots/_graphs/{G_name[0]}/{G_name[1]}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.axis('off')
        plt.savefig(f'{out_path}/{sparse_name}.png', dpi=400, bbox_inches='tight', pad_inches=0)
        plt.clf()

    def weight_distributions(self):
        x = np.linspace(0, 2, 1000)

        uniform_pdf = uniform.pdf(x, loc=0, scale=2)
        exp_pdf = expon.pdf(x, scale=1)

        σ = 3/4
        µ = - σ**2/2
        lognorm_pdf = lognorm.pdf(x, s=σ, scale=np.exp(µ))

        plt.plot(x, uniform_pdf, label='Uni(0,2)', color = 'red')
        plt.plot(x, exp_pdf, label='Exp(1)', color = 'blue')
        plt.plot(x, lognorm_pdf, label='LogN(-1/8, 1/2)', color = 'green')
        plt.axvline(x=1, color='black', linestyle='--', label=f'𝔼', alpha = 0.5)

        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title('PDFs in the range [0,2]')
        plt.legend()

        plt.grid(True)
        plt.show()
