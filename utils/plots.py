import numpy as np
from pathlib import Path
from itertools import product
from scipy.spatial.distance import euclidean as euclidean_distance

import radialtree as rt
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
from sklearn.metrics import auc
from scipy.stats import expon, uniform, lognorm

import timeit

from utils.static import *
from utils.tests import *

REPORT_FONTSIZE = 12

def custom_symlog_formatter(value, _):
    return int(value) if int(value) == value else value

def sign(v):
    return 1 if np.sum(v) >= 0 else -1

def signed_euclidean_distance(v1, v2):
    return sign(v1 - v2) * euclidean_distance(v1, v2)

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

def compute_mean_aupr(
            D_d, labels, class_characterisation, N = 20):
        
        mean_aupr = {}
        aupr = {}

        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)

        for metric in M_ID:
            mean_aupr[metric] = {}
            aupr[metric] = {}
            for sparse in S_NAME:
                aupr[metric][sparse] = {}
                auprs = []

                for i in range(N):
                    distance_m = D_d[i][sparse][metric].to_numpy()
                    distance_m = distance_m / np.max(distance_m)

                    precisions, recalls = precisions_recalls(distance_m, class_m, thresholds)
                    a = auc(recalls, precisions)
                    aupr[metric][sparse][i] = a
                    auprs.append(a)

                mean, std = np.mean(auprs), np.std(auprs)
                mean_aupr[metric][sparse] = {'mean': mean, 'std': std}

        def aupr_vec(i):
            return np.array([aupr[m][s][i] for m, s in product(M_ID, S_NAME)])
        mean_aupr_vec = np.array([mean_aupr[m][s]['mean'] for m, s in product(M_ID, S_NAME)])
        distances = [euclidean_distance(aupr_vec(i), mean_aupr_vec) for i in range(N)]

        mean_aupr['best_approx_index'] = int(np.argmin(distances))
        mean_aupr['rdm_chance'] = np.mean(np.array(class_m) == 1)

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        out_path = f'results/clustering/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)

        with open(f'{out_path}/mean_aupr.json', 'w') as json_file:
            json.dump(mean_aupr, json_file)

        return mean_aupr

def compute_perturbation_distances_deviation(D_d, N = N_PERTURBATIONS):
        step = 5
        dd = {}
        for p in P_NAME:
            dd[p] = {}
            for m in M_ID:
                dd[p][m] = {}
                for s in TS_NAME:
                    dd[p][m][s] = {}
                    for g in G_NAME:
                        distances = []
                        dd[p][m][s][g] = {}
                        for w in W_NAME:
                            sparsifier_vec = D_d[p][(g, w)][s][m]['mean'][:(N//step)+1]
                            full_vec       = D_d[p][(g, w)]['Full'][m]['mean'][:(N//step)+1]
                            distances.append(signed_euclidean_distance(sparsifier_vec, full_vec))
                        
                        dd[p][m][s][g]['distances'] = np.array(distances)
                    dd[p][m][s]['distances'] = np.concatenate([dd[p][m][s][g]['distances'] for g in G_NAME])

                for g in G_NAME:
                    mean = np.mean(np.concatenate([np.abs(dd[p][m][s][g]['distances']) for s in TS_NAME]))
                    for s in TS_NAME:
                        dd[p][m][s][g]['normalized_distances'] = dd[p][m][s][g]['distances'] / mean  

                for s in TS_NAME:
                    normalized_distances = np.concatenate([dd[p][m][s][g]['normalized_distances'] for g in G_NAME])
                    dd[p][m][s]['abs_mean'] = sign(normalized_distances) * np.mean(np.abs(normalized_distances))
                    dd[p][m][s]['mean'] = np.mean(normalized_distances)

        dd['max'] = {}
        for m in M_ID:
            dd['max'][m] = np.max(np.concatenate([np.abs(dd[p][m][s]['distances']) for (p, s) in product(P_NAME, TS_NAME)]))

        return dd

def compute_gaussian_noise_distances_deviation(D_d):
        dd = {}
        for m in M_ID:
            dd[m] = {}
            for s in TS_NAME:
                dd[m][s] = {}
                for g in G_NAME:
                    distances = []
                    dd[m][s][g] = {}
                    for w in W_NAME:
                        sparsifier_vec = D_d[(g, w)][s][m]['mean']
                        full_vec       = D_d[(g, w)]['Full'][m]['mean']
                        distances.append(signed_euclidean_distance(sparsifier_vec, full_vec))
                    
                    dd[m][s][g]['distances'] = np.array(distances)
                dd[m][s]['distances'] = np.concatenate([dd[m][s][g]['distances'] for g in G_NAME])

            for g in G_NAME:
                mean = np.mean(np.concatenate([np.abs(dd[m][s][g]['distances']) for s in TS_NAME]))
                for s in TS_NAME:
                    dd[m][s][g]['normalized_distances'] = dd[m][s][g]['distances'] / mean

            for s in TS_NAME:
                normalized_distances = np.concatenate([dd[m][s][g]['normalized_distances'] for g in G_NAME])
                dd[m][s]['abs_mean'] = sign(normalized_distances) * np.mean(np.abs(normalized_distances))
                dd[m][s]['mean'] = np.mean(normalized_distances)
        return dd

def portrait(G):
    d = max(nx.diameter(G.subgraph(K)) for K in nx.connected_components(G))
    n = nx.number_of_nodes(G)
    shortest_paths = dict(nx.all_pairs_shortest_path(G))

    Shell = {}

    for v in G.nodes():
        Shell[v] = {l : [] for l in range(0, d + 1)}
        for u in G.nodes():
            try:
                path = shortest_paths[u][v]
            except KeyError:
                continue
            Shell[v][len(path) - 1].append((u, path))

    B = {l : {k : {} for k in range(0, n)} for l in range(0, d + 1)}
    for v in G.nodes():
        for l in range(0, d + 1):
            shell = Shell[v][l]
            k = len(shell)
            B[l][k][v] = shell

    return B



class Plot:
    def __init__(self):
        pass

    def perturbation_distances(
            self, dfs, graph, weight, metric, perturbation, N = N_PERTURBATIONS, 
            y_axis_range=None, g_first = True, 
            report = False):

        #plt.subplots_adjust(left=0.15)
        
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
        if y_axis_range is not None: plt.ylim(y_axis_range)

        if report:
            display_perturbation  = metric == METRICS[0]
            display_nb_iterations = metric == METRICS[-1]
            display_metric        = perturbation == P_NAME[0]
            plt.title(f'{perturbation}\n',  alpha = display_perturbation, fontsize = REPORT_FONTSIZE)
            plt.xlabel(f'# {perturbation}', alpha = display_nb_iterations, fontsize = REPORT_FONTSIZE)
            plt.gca().xaxis.set_label_coords(0.5, -0.125)
            plt.ylabel(metric.name,  alpha = display_metric, fontsize = REPORT_FONTSIZE)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)

            for label in plt.gca().get_yticklabels(): label.set_alpha(display_metric)
            for label in plt.gca().get_xticklabels(): label.set_alpha(display_nb_iterations)


            out_path = f'plots/_report/perturbation/distances'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            perturbation_nb = P_NAME.index(perturbation)
            plt.savefig(f'{out_path}/{metric.name} {perturbation_nb}.png', dpi=200)
            plt.savefig(out_path + f'/{metric.name} {perturbation_nb}.eps'.replace(' ', '_'), dpi=200)
            plt.clf()

        else:
            plt.title(f'{perturbation} on {weight} {graph}')
            plt.xlabel(f'# {perturbation}')
            plt.ylabel(metric.name)
            plt.legend(loc='upper left')

            out_path = f'plots/perturbation/{perturbation}/{metric.name}'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
            plt.savefig(out_path + f'/{weight} {graph}.eps'.replace(' ', '_'), dpi=200)
            plt.clf()

    def perturbation_edges(
            self, dfs, graph, weight, e_mes, perturbation, N = N_PERTURBATIONS,
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
        plt.ylabel(E_MAP[e_mes])
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/perturbation/{perturbation}/Edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.savefig(out_path + f'/{weight} {graph}.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def perturbation_deviation_by_graph(
            self, D_dd, metric, perturbation, ylim, mean_mode = 'abs_mean',
            report = False):
        
        display_perturbation = metric == METRICS[0]         if report else True
        display_graphs       = metric == METRICS[-1]        if report else True
        display_metric       = perturbation == P_NAME[0]    if report else True

        TS = TS_NAME.copy()
        TS.sort(key=lambda s: abs(D_dd[s][mean_mode]))

        _, ax = plt.subplots(figsize=(6, 4))
        #plt.subplots_adjust(left=0.15)

        for s in TS:
            for i, dev in enumerate(D_dd[s]['distances']):
                ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5, label=s)

        xitcks_pos = np.arange(len(G_NAME) * len(W_NAME))
        ax.set_xticks(xitcks_pos)
        ax.set_xticklabels(W_NAME * len(G_NAME), fontsize=8, alpha = display_graphs)

        for i, g_name in enumerate(G_NAME):
            pos = xitcks_pos[1+i*len(W_NAME)]
            ax.text(pos, -0.1, g_name, ha='center', va='top', transform=ax.get_xaxis_transform(),
                    fontsize=10, alpha = display_graphs)
            
        for i in range(len(G_NAME) - 1):
            pos = 1/2 * (xitcks_pos[1+i*len(W_NAME)] + xitcks_pos[1+(i+1)*len(W_NAME)])
            ax.axvline(x=pos, color='black', linestyle='-', linewidth=0.5, alpha = 0.5)

        plt.axhline(y=0, linestyle='-', color='black')

        ylim = 10 ** np.ceil(np.log10(ylim))
        ax.set_ylim([-ylim, ylim])
        ax.set_yscale('symlog', linthresh = ylim/100, subs = list(range(2, 10, 2)))
        ax.yaxis.set_major_formatter(FuncFormatter(custom_symlog_formatter))
        ax.grid(axis='y', linestyle=':')

        if report:
            ax.set_title(f'{perturbation}\n', alpha = display_perturbation, fontsize=REPORT_FONTSIZE)
            ax.set_ylabel(metric.name, alpha = display_metric, fontsize=REPORT_FONTSIZE)
            ax.yaxis.set_label_coords(-0.125, 0.5)
            for label in ax.get_yticklabels(): label.set_alpha(display_metric) 

            out_path = f'plots/_report/perturbation/deviation_by_graph'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            perturbation_nb = P_NAME.index(perturbation)
            plt.savefig(f'{out_path}/{metric.name} {perturbation_nb}.png', dpi=200)
            plt.savefig(out_path + f'/{metric.name} {perturbation_nb}.eps'.replace(' ', '_'), dpi=200)
            plt.clf()
        else:
            ax.set_title(f'Deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on {perturbation.lower()} test', fontsize='small')
            ax.set_ylabel(metric.name)
            ax.set_xlabel('Graphs')
            ax.legend(loc='lower left', fontsize='x-small')

            out_path = f'plots/perturbation/_Deviation'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/{metric.name} {perturbation}.png', dpi=200)
            plt.savefig(out_path + f'/{metric.name} {perturbation}.eps'.replace(' ', '_'), dpi=200)

            out_path = f'plots/perturbation/{perturbation}/{metric.name}'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/_Deviation.png', dpi=200)
            plt.clf()


    def perturbation_deviation_by_sparsifier(
            self, D_dd, perturbation, mean_mode = 'abs_mean', report = False):
        
        plt.figure(figsize=(8, 4))
        #plt.subplots_adjust(left=0.15)

        TS = TS_NAME.copy()
        MET = METRICS.copy()

        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][mean_mode]) for m in MET))
        maximum = np.max([abs(D_dd[m][s][mean_mode]) for m, s in product(M_ID, TS)])

        for i, sparse in enumerate(TS):
            MET.sort(key=lambda m: abs(D_dd[m.id][sparse][mean_mode]))
            for j, metric in enumerate(MET):
                mean = D_dd[metric.id][sparse][mean_mode] / maximum
                offset = -0.3 + j*(0.6/(len(MET)-1))
                plt.bar(i + offset, mean, hatch = M_HATCHES[metric],
                        width = 0.15, color = S_COLORS[sparse])
                
        fig, ax = plt.subplots()
        ax.axis('off')
        legend_handles = []
        for metric in MET:
            legend_handles.append(ax.bar([0], [0], hatch = M_HATCHES[metric], label=metric.name, 
                                          color = 'white', edgecolor = 'black'))
        plt.close(fig)

        plt.xticks(np.arange(len(TS)), labels=TS,  fontsize='x-small')
        plt.ylim([-1.05, 1.05])

        if report:
            plt.title(f'{perturbation}\n', fontsize=REPORT_FONTSIZE)
            display_deviation = perturbation == P_NAME[0]
            plt.ylabel('Normalized deviation', fontsize=REPORT_FONTSIZE, alpha = display_deviation)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)
            if display_deviation:
                plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small', 
                   handleheight=3, handlelength=3)

            for label in plt.gca().get_yticklabels(): label.set_alpha(display_deviation)

            out_path = f'plots/_report/perturbation/deviation_by_sparsifier'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            perturbation_nb = P_NAME.index(perturbation)
            plt.savefig(f'{out_path}/{perturbation_nb}.png', dpi=200)
            plt.savefig(out_path + f'/{perturbation_nb}.eps'.replace(' ', '_'), dpi=200)

        else:
            plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small', 
                   handleheight=3, handlelength=3)
            plt.title(f'Normalized mean deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on {perturbation.lower()} test', fontsize='small')
            plt.ylabel('Normalized deviation')

            out_path = f'plots/perturbation/_Deviation'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/_{perturbation} by sparsifier.png', dpi=200)
            plt.savefig(out_path + f'/_{perturbation} by sparsifier.eps'.replace(' ', '_'), dpi=200)

            out_path = f'plots/perturbation/{perturbation}'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/_Deviation by sparsifier.png', dpi=200)
            plt.clf()


    def perturbation_deviation_by_metric(
            self, D_dd, perturbation, mean_mode = 'abs_mean'):
        
        MET = METRICS.copy()
        TS = TS_NAME.copy()
        
        MET.sort(key=lambda m: sum(abs(D_dd[m.id][s][mean_mode]) for s in TS))
        maximum = np.max([abs(D_dd[m][s][mean_mode]) for m, s in product(M_ID, TS)])

        plt.figure(figsize=(8, 4))

        for i, metric in enumerate(MET):
            for sparse in TS:
                mean = D_dd[metric.id][sparse][mean_mode] / maximum
                plt.errorbar(mean, i, fmt='o', color=S_COLORS[sparse], markersize=5)

        plt.axvline(x=0, linestyle='-', color='black', label='Full')
                
        legend_handles = []
        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][mean_mode]) for m in MET))
        for sparse in TS:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        plt.title(f'Normalized mean deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on {perturbation.lower()} test', fontsize='small')
        plt.xlabel('Deviation')
        plt.xlim([-1.02, 1.02])
        plt.yticks(np.arange(len(MET)), labels=
                   [m.name.replace(' ', '\n') for m in MET], fontsize='x-small')
        plt.ylabel('Metrics')
        plt.ylim([-0.6, len(MET)-1 + 0.6])
        plt.grid(axis='x')

        out_path = f'plots/perturbation/_Deviation'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/_{perturbation} by metric.png', dpi=200)
        plt.savefig(out_path + f'/_{perturbation} by metric.eps'.replace(' ', '_'), dpi=200)

        out_path = f'plots/perturbation/{perturbation}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/_Deviation by metric.png', dpi=200)
        plt.clf()


    def gaussian_noise_distances(
            self, dfs, graph, weight, metric,
            y_axis_range=None, g_first = True, report = False):
        
        #plt.subplots_adjust(left=0.15)
        
        df = dfs[(graph, weight)]
        for sparse in S_NAME:
            m = df[sparse][metric.id]['mean'].to_numpy()
            s = 0.3*df[sparse][metric.id]['std'].to_numpy()
            x = df[sparse].index

            plt.errorbar(x, m, yerr = s, linestyle='-',
                         label = sparse, color=S_COLORS[sparse])

        if g_first: graph, weight = weight, graph

        plt.xlabel('σ')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        if report:
            plt.title(f'Gaussian noise N(0, σ)\n', fontsize = REPORT_FONTSIZE)
            plt.ylabel(metric.name, fontsize = REPORT_FONTSIZE)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)

            out_path = f'plots/_report/gaussian_noise/distances'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            metric_nb = METRICS.index(metric)
            plt.savefig(f'{out_path}/{metric_nb}.png', dpi=200)
            plt.savefig(out_path + f'/{metric_nb}.eps'.replace(' ', '_'), dpi=200)
            plt.clf()

        else:
            plt.title(f'Gaussian Noise N(0, σ) on {weight} {graph}')
            plt.ylabel(metric.name)
            plt.legend(loc='upper left')
            

            out_path = f'plots/gaussian_noise/{metric.name}'
            Path(out_path).mkdir(parents = True, exist_ok = True) 
            plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
            plt.savefig(out_path + f'/{weight} {graph}.eps'.replace(' ', '_'), dpi=200)
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
        plt.ylabel(E_MAP[e_mes])
        plt.legend(loc='upper left')
        if y_axis_range is not None: plt.ylim(y_axis_range)

        out_path = f'plots/gaussian_noise/Edges/{e_mes}'
        Path(out_path).mkdir(parents = True, exist_ok = True) 
        plt.savefig(f'{out_path}/{weight} {graph}.png', dpi=200)
        plt.savefig(out_path + f'/{weight} {graph}.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def gaussian_noise_deviation_by_graph(
            self, D_dd, metric, mean_mode = 'abs_mean', report = False):

        TS = TS_NAME.copy()
        TS.sort(key=lambda s: abs(D_dd[s][mean_mode]))

        _, ax = plt.subplots(figsize=(6, 4))
        #plt.subplots_adjust(left=0.15)

        for s in TS:
            for i, dev in enumerate(D_dd[s]['distances']):
                ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5, label=s)

        xitcks_pos = np.arange(len(G_NAME) * len(W_NAME))
        ax.set_xticks(xitcks_pos)
        ax.set_xticklabels(W_NAME * len(G_NAME), fontsize=8)

        for i, g_name in enumerate(G_NAME):
            pos = xitcks_pos[1+i*len(W_NAME)]
            ax.text(pos, -0.1, g_name, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)
            
        for i in range(len(G_NAME) - 1):
            pos = 1/2 * (xitcks_pos[1+i*len(W_NAME)] + xitcks_pos[1+(i+1)*len(W_NAME)])
            ax.axvline(x=pos, color='black', linestyle='-', linewidth=0.5, alpha = 0.5)

        plt.axhline(y=0, linestyle='-', color='black')

        ylim = np.max([np.max(np.abs(D_dd[s]['distances'])) for s in TS])
        ylim = 10 ** np.ceil(np.log10(ylim))
        ax.set_ylim([-ylim, ylim])
        ax.set_yscale('symlog', linthresh = ylim/100, subs = list(range(2, 10, 2)))
        ax.yaxis.set_major_formatter(FuncFormatter(custom_symlog_formatter))
        ax.grid(axis='y', linestyle=':')
        
        if report:
            ax.set_ylabel(metric.name, fontsize = REPORT_FONTSIZE)
            ax.yaxis.set_label_coords(-0.125, 0.5)

            out_path = f'plots/_report/gaussian_noise/deviation_by_graph'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            metric_nb = METRICS.index(metric)
            plt.savefig(f'{out_path}/{metric_nb}.png', dpi=200)
            plt.savefig(out_path + f'/{metric_nb}.eps'.replace(' ', '_'), dpi=200)
            plt.clf()

        else:
            ax.set_title(f'Deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on gaussian noise test', fontsize='small')
            ax.set_xlabel('Graphs')
            ax.set_ylabel(metric.name, fontsize='x-small')
            ax.legend(loc='lower left', fontsize='x-small')

            out_path = f'plots/gaussian_noise/_Deviation'
            Path(out_path).mkdir(parents = True, exist_ok = True) 
            plt.savefig(f'{out_path}/{metric.name}.png', dpi=200)
            plt.savefig(out_path + f'/{metric.name}.eps'.replace(' ', '_'), dpi=200)

            out_path = f'plots/gaussian_noise/{metric.name}'
            Path(out_path).mkdir(parents = True, exist_ok = True) 
            plt.savefig(f'{out_path}/_Deviation.png', dpi=200)
            plt.clf()

    def gaussian_noise_deviation_by_sparsifier(
            self, D_dd, mean_mode = 'abs_mean', report = False):
        
        plt.figure(figsize=(8, 4))
        #plt.subplots_adjust(left=0.15)

        MET = METRICS.copy()
        TS = TS_NAME.copy()

        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][mean_mode]) for m in MET))
        maximum = max(abs(D_dd[m][s][mean_mode]) for m, s in product(M_ID, TS))

        for i, sparse in enumerate(TS):
            MET.sort(key=lambda m: abs(D_dd[m.id][sparse][mean_mode]))
            for j, metric in enumerate(MET):
                mean = D_dd[metric.id][sparse][mean_mode] / maximum
                offset = -0.3 + j*(0.6/(len(MET)-1))
                plt.bar(i + offset, mean, hatch = M_HATCHES[metric],
                        width = 0.15, color = S_COLORS[sparse])
                
        fig, ax = plt.subplots()
        ax.axis('off')
        legend_handles = []
        for metric in MET:
            legend_handles.append(ax.bar([0], [0], hatch = M_HATCHES[metric], label=metric.name, 
                                          color = 'white', edgecolor = 'black'))
        plt.close(fig)
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small', 
                   handleheight=3, handlelength=3)
            
        plt.xticks(np.arange(len(TS)), labels=TS,  fontsize='x-small')
        plt.ylim([-1.05, 1.05])

        if report:
            plt.ylabel('Normalized deviation', fontsize=REPORT_FONTSIZE)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)

            out_path = f'plots/_report/gaussian_noise'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/deviation_by_sparsifier.png', dpi=200)
            plt.savefig(out_path + f'/deviation_by_sparsifier.eps'.replace(' ', '_'), dpi=200)

        else:
            plt.title(f'Normalized mean deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on gaussian noise test', fontsize='small')
            plt.ylabel('Normalized deviation')

            out_path = f'plots/gaussian_noise/_Deviation'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/_by sparsifier.png', dpi=200)
            plt.savefig(out_path + f'/_by sparsifier.eps'.replace(' ', '_'), dpi=200)
            plt.clf()


    def gaussian_noise_deviation_by_metric(
            self, D_dd, mean_mode = 'abs_mean'):
        
        MET = METRICS.copy()
        TS = TS_NAME.copy()

        MET.sort(key=lambda m: sum(abs(D_dd[m.id][s][mean_mode]) for s in TS))
        maximum = np.max([abs(D_dd[m][s][mean_mode]) for m, s in product(M_ID, TS)])

        plt.figure(figsize=(8, 4))

        for i, metric in enumerate(MET):
            for sparse in TS:
                mean = D_dd[metric.id][sparse][mean_mode] / maximum
                plt.errorbar(mean, i, fmt='o', color=S_COLORS[sparse], markersize=5)

        plt.axvline(x=0, linestyle='-', color='black', label='Full')
                
        legend_handles = []
        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][mean_mode]) for m in MET))
        for sparse in TS:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        plt.title(f'Normalized mean deviation distances\n'
                  + f'of sparsified graphs from full graphs\n'
                  + f'on gaussian noise test', fontsize='small')
        plt.xlabel('Deviation')
        plt.xlim([-1.02, 1.02])
        plt.yticks(np.arange(len(MET)), labels=
                   [m.name.replace(' ', '\n') for m in MET], fontsize='x-small')
        plt.ylabel('Metrics')
        plt.ylim([-0.6, len(MET)-1 + 0.6])
        plt.grid(axis='x')

        out_path = f'plots/gaussian_noise/_Deviation'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/_by metric.png', dpi=200)
        plt.savefig(out_path + f'/_by metric.eps'.replace(' ', '_'), dpi=200)
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
            plt.figure(figsize=(25, 12))
            D = hierarchy.dendrogram(linkage[sparse], 
                                     labels = display_labels, 
                                     color_threshold=0.3*max(linkage[sparse][:, 2]),
                                     leaf_font_size=2)
            rt.plot(D, colorlabels = colors_dict, colorlabels_legend = COLORS_LEGENDS)
            #plt.title(f'Clustering on {sparse} graphs\nMetric: {metric.name}')

            out_path = f'plots/clustering/{metric.name}/Dendrogram'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.savefig(f'{out_path}/{sparse}.png', dpi=400)
            plt.savefig(out_path + f'/{sparse}.eps'.replace(' ', '_'), dpi=400)
            plt.clf()

    def clustering_precision_recall_curve(
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
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend(loc='lower left', fontsize='x-small')
        plt.grid(True)

        out_path = f'plots/clustering/{metric.name}/Precision-Recall'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{class_str}.png', dpi=200)

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/{metric.name}.png', dpi=200)
        plt.savefig(out_path + f'/{metric.name}.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def clustering_precision_recall_curve_3D(
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
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.legend(loc='upper right', fontsize='xx-small')
        ax.grid(True)

        out_path = f'plots/clustering/{metric.name}/Precision-Recall'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/3D {class_str}.png', dpi=200)

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/3D {metric.name}.png', dpi=200)
        plt.savefig(out_path + f'/3D {metric.name}.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def clustering_aupr_by_sparsifier(
            self, D_a, class_characterisation, display_str = True):
        
        plt.figure(figsize=(8, 4))

        S_NAME.sort(key=lambda s: sum(D_a[m.id][s]['mean'] for m in METRICS), reverse=True)

        for i, sparse in enumerate(S_NAME):
            METRICS.sort(key=lambda m: D_a[m.id][sparse]['mean'], reverse=True)
            for j, metric in enumerate(METRICS):
                mean = D_a[metric.id][sparse]['mean']
                std = D_a[metric.id][sparse]['std'] if display_str else 0
                offset = -0.3 + j*(0.6/(len(METRICS)-1))
                plt.errorbar(i + offset, mean, yerr=std, fmt=M_MARKERS[metric], markersize = 3, capsize=1,
                             elinewidth=0.5, capthick=0.5, color = S_COLORS[sparse])
                
        rdm_chance = plt.axhline(y=D_a['rdm_chance'], linestyle=':', label = 'Random classification', color='black')
                
        legend_handles = [rdm_chance]
        for metric in METRICS:
            legend_handles.append(plt.Line2D([0], [0], marker = M_MARKERS[metric], label=metric.name, color = 'black'))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.title(f'Area Under Precision-Recall Curves\nClass: {class_str}')
        plt.xlabel('Sparsifier')
        plt.xticks(np.arange(len(S_NAME)), labels=S_NAME,  fontsize='x-small')
        plt.ylabel('AUPR')
        plt.ylim([0, 1.05])
        plt.grid(axis='y')
        plt.yticks(np.arange(0, 1.1, 0.1))

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{out_path}/_AUPRs by sparsifier.png', dpi=200)
        plt.savefig(out_path + f'/_AUPRs by sparsifier.eps'.replace(' ', '_'), dpi=200)
        plt.clf()


    def clustering_aupr_by_metric(
            self, D_a, class_characterisation, display_str = True):
        
        METRICS.sort(key=lambda m: sum(D_a[m.id][s]['mean'] for s in S_NAME), reverse=True)

        plt.figure(figsize=(10, 6))

        for i, metric in enumerate(METRICS):
            S_NAME.sort(key=lambda s: D_a[metric.id][s]['mean'], reverse=True)
            for j, sparse in enumerate(S_NAME):
                mean = D_a[metric.id][sparse]['mean']
                std = D_a[metric.id][sparse]['std'] if display_str else 0
                offset = -0.4 + j*(0.8/(len(S_NAME)-1))
                plt.errorbar(mean, i + offset, xerr=std, fmt='o', color=S_COLORS[sparse], markersize=3,
                             linewidth=0.5, capthick = 0.5, capsize=2)
                plt.annotate(f'{mean:.2f} ± {std:.2f}', xy=(mean, i + offset), xytext=(-41, 10),
                             textcoords='offset pixels', fontsize=5)
                
        rdm_chance = plt.axvline(x=D_a['rdm_chance'], linestyle=':', label = 'Random classification', color='black')
                
        legend_handles = [rdm_chance]
        S_NAME.sort(key=lambda s: sum(D_a[m.id][s]['mean'] for m in METRICS), reverse=True)
        for sparse in S_NAME:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.title(f'Area Under Precision-Recall Curves\n'
                +f'Class: {class_str}')
        plt.xlabel('AUPR')
        plt.xlim([0, 1.02])
        plt.yticks(np.arange(len(METRICS)), labels=
                   [m.name.replace(' ', '\n') for m in METRICS], fontsize='x-small')
        plt.ylabel('Metrics')
        plt.ylim([-0.6, len(METRICS)-1 + 0.6])

        out_path = f'plots/clustering/_Precision-Recall/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/_AUPRs by metric.png', dpi=200)
        plt.savefig(out_path + f'/_AUPRs by metric.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def sparsifiers_speed(
            self, graphs = None):
        
        if graphs is None:
            if os.path.exists('results/clustering/0/graphs.csv'):
                graphs = read_graphs('results/clustering/0/graphs.csv')
            else:
                graphs = Clustering().generate_graphs(save = False)

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

        out_path = f'plots/auxiliary'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(f'{out_path}/Sparsification speed.png', dpi=200)
        plt.savefig(out_path + f'/Sparsification speed.eps'.replace(' ', '_'), dpi=200)
        plt.clf()

    def graph(
            self, G, G_name, pos = None, sparse = None, e_width_from_weight = True, 
            node_color = 'black', size = 3/4, alpha_full = None, highlighting_factor = 1,
            explicit_out_path = None):
        
        plt.figure(figsize=(30, 24))

        if pos is None:
            pos = nx.spring_layout(G)

        edges = list(G.edges())

        if sparse is not None and sparse.name == 'Full': sparse = None
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
            width = size * 1 / 2


        nx.draw_networkx_edges(G, pos, edges, width, 
                                edge_color='gray', alpha=alpha_full)
        nx.draw_networkx_nodes(G, pos, node_size=size*32, node_color=node_color)
            
        if explicit_out_path is not None:
            out_path = f'plots/auxiliary/graphs/{explicit_out_path[:explicit_out_path.rfind("/")]}'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            out_path = f'plots/auxiliary/graphs/{explicit_out_path}'
            plt.axis('off')
            plt.savefig(f'{out_path}.png', dpi=400, bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{out_path}.eps'.replace(' ', '_'), dpi=400, bbox_inches='tight', pad_inches=0)
            plt.clf()
        else:
            out_path = f'plots/auxiliary/graphs/{G_name[0]}/{G_name[1]}'
            Path(out_path).mkdir(parents = True, exist_ok = True)
            plt.axis('off')
            plt.savefig(f'{out_path}/{sparse_name}.png', dpi=400, bbox_inches='tight', pad_inches=0)
            plt.savefig(out_path + f'/{sparse_name}.eps'.replace(' ', '_'), dpi=400, bbox_inches='tight', pad_inches=0)
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


    def sparse_toy(
            self, sparse = None, node_color = 'black', size = 3, alpha_full = None, seed = 1):
        
        plt.figure(figsize=(12, 10))

        G = BA(n = 15, m = 30)
        G = Exponential(seed=seed)(G)
        add_inverse_weight(G)

        for _, _, d in G.edges(data=True):
            d['exp weight'] = np.exp(d['weight'])

        pos = nx.kamada_kawai_layout(G, weight='exp weight')

        edges = list(G.edges())

        if sparse.name == 'Full': sparse = None
        sparse_name = sparse.name if sparse is not None else '_Full'

        if alpha_full is None:
                alpha_full = 0.5 if sparse is not None else 1

        if sparse is not None:
            sG = sparse(G)
            sG_edges = list(sG.edges())
            edges = list(set(edges) - set(sG_edges))

            sG_width = [size * inverse_weight(sG[u][v]['weight']) for u, v in sG_edges]
            sG_weights = {(u, v): f'{sG[u][v]["weight"]:.2f}' for u, v in sG_edges}
            nx.draw_networkx_edges(G, pos, sG_edges, sG_width, 
                                edge_color=S_COLORS[sparse.name])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=sG_weights, font_size=12)

        width = [size * inverse_weight(G[u][v]['weight']) for u, v in edges]
        weights = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
        
        nx.draw_networkx_edges(G, pos, edges, width, 
                                edge_color='black', alpha=alpha_full)
        nx.draw_networkx_edge_labels(G, pos, 
                                     edge_labels=weights, font_color='black', font_size=12)
        nx.draw_networkx_nodes(G, pos, node_size=size*25, node_color=node_color)

        x_center = np.mean([pos[node][0] for node in pos])
        y_center = np.mean([pos[node][1] for node in pos])
        zoom_factor = 0.2
        x_min = x_center - zoom_factor - 0.2
        x_max = x_center + zoom_factor + 0.15
        y_min = y_center - zoom_factor - 0.1
        y_max = y_center + zoom_factor + 0.35
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
            
        out_path = f'plots/auxiliary/sparsifier'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.savefig(out_path + f'/{sparse_name}.eps'.replace(' ', '_'), dpi=400)
        plt.axis('off')
        plt.savefig(f'{out_path}/{sparse_name}.png', dpi=400, bbox_inches='tight', pad_inches=0)
        plt.clf()

    def portrait_shell(self):
        
        G = nx.gnm_random_graph(30, 34, seed = 85)
        G.remove_nodes_from([8, 9, 23, 25])
        pos = nx.spring_layout(G, seed=4)

        B = portrait(G)
        Blk = B[2][2]

        kern_nodes = list(Blk.keys())
        shell_nodes, shell_edges = [], []
        for shell in Blk.values():
            shell_nodes += [u for u, _ in shell]
            shell_edges += [sorted((u, v)) for _, path in shell for (u, v) in zip(path[:-1], path[1:])]

        node_colors = ['red' if node in kern_nodes else 
                       'brown' if node in shell_nodes else 
                       'skyblue' for node in G.nodes()]
        edge_colors = ['brown' if [u, v] in shell_edges else 
                       'black' for u, v in G.edges()]

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, node_color=node_colors, with_labels=False,
                node_size=500, font_size=10, font_color='black', edge_color=edge_colors)
        
        out_path = f'plots/auxiliary'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        plt.axis('off')
        plt.savefig(f'{out_path}/portrait.png', dpi=400, bbox_inches='tight', pad_inches=0)
        plt.savefig(out_path + f'/portrait.eps'.replace(' ', '_'), dpi=400, bbox_inches='tight', pad_inches=0)
        plt.clf()


def load_perturbation_dfs():
    D_p_dists = {}
    D_p_edges = {}

    M_p_dists = {}
    M_p_edges = {e_mes : [] for e_mes in E_MES}

    for p in P_NAME:
        D_p_dists[p] = {}
        D_p_edges[p] = {}
        M_p_dists[p] = {metric.id : [] for metric in METRICS}
        for (g, w) in product(G_NAME, W_NAME):
            D_p_dists[p][(g, w)] = {}
            D_p_edges[p][(g, w)] = {}
            for s in S_NAME:
                D_p_dists[p][(g, w)][s] = pd.read_csv(
                    f'results/perturbation/{p}/{g}/{w}/distances/{s}.csv', header = [0,1])
                D_p_edges[p][(g, w)][s] = pd.read_csv(
                    f'results/perturbation/{p}/{g}/{w}/edges/{s}.csv', header = [0,1])
                
                for m in METRICS:
                    M_p_dists[p][m.id].append(D_p_dists[p][(g, w)][s][m.id]['mean'].max())
                for e_mes in E_MES:
                    M_p_edges[e_mes].append(D_p_edges[p][(g, w)][s][e_mes]['mean'].max())

    for m in METRICS:
        M_p_dists[m.id] = {}
        for (g, w) in product(G_NAME, W_NAME):
            max_gw = [D_p_dists[p][(g, w)][s][m.id]['mean'].max() for p in P_NAME for s in S_NAME]
            M_p_dists[m.id][(g, w)] = pretty_upper_bound(np.max(max_gw))

    for p in P_NAME:
        for m in METRICS:
            M_p_dists[p][m.id] = pretty_upper_bound(np.max(M_p_dists[p][m.id]))
    for e_mes in E_MES:
        M_p_edges[e_mes] = pretty_upper_bound(np.max(M_p_edges[e_mes]))

    return D_p_dists, D_p_edges, M_p_dists, M_p_edges

def load_gaussian_noise_dfs():
    D_g_dists = {}
    D_g_edges = {}

    M_g_dists = {m.id : [] for m in METRICS}
    M_g_edges = {e_mes : [] for e_mes in E_MES}

    for (g, w) in product(G_NAME, W_NAME):
        D_g_dists[(g, w)] = {}
        D_g_edges[(g, w)] = {}
        for s in S_NAME:
            D_g_dists[(g, w)][s] = pd.read_csv(
                f'results/gaussian_noise/{g}/{w}/distances/{s}.csv', header = [0,1], index_col=0)
            D_g_edges[(g, w)][s] = pd.read_csv(
                f'results/gaussian_noise/{g}/{w}/edges/{s}.csv', header = [0,1], index_col=0)
            
            for m in METRICS:
                M_g_dists[m.id].append(D_g_dists[(g, w)][s][m.id]['mean'].max())
            for e_mes in E_MES:
                M_g_edges[e_mes].append(D_g_edges[(g, w)][s][e_mes]['mean'].max())
                
    for m in METRICS:
        M_g_dists[m.id] = pretty_upper_bound(np.max(M_g_dists[m.id]))
    for e_mes in E_MES:
        M_g_edges[e_mes] = pretty_upper_bound(np.max(M_g_edges[e_mes]))

    return D_g_dists, D_g_edges, M_g_dists, M_g_edges
