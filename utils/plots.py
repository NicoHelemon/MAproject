import numpy as np
from pathlib import Path
from itertools import product, combinations
from scipy.spatial.distance import euclidean as euclidean_distance

import radialtree as rt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
from sklearn.metrics import auc
from scipy.stats import expon, uniform, lognorm
from collections import Counter

import timeit

from utils.static import *
from utils.tests import *

REPORT_FONTSIZE = 12

def save_file(out_path, file_name, dpi = 200, eps = False, clf = True, no_bbox = False):
        Path(out_path).mkdir(parents = True, exist_ok = True)

        if not no_bbox:
            plt.savefig(f'{out_path}/{file_name}.png', dpi=dpi)
            if eps: plt.savefig(out_path + f'/{file_name}.eps'.replace(' ', '_'), dpi=dpi)
            #if eps: plt.savefig(out_path + f'/{file_name}.pdf'.replace(' ', '_'), dpi=dpi)
            if clf: plt.clf()

        else:
            plt.axis('off')
            plt.savefig(f'{out_path}/{file_name}.png', 
                        dpi=400, bbox_inches='tight', pad_inches=0)
            if eps: plt.savefig(out_path + f'/{file_name}.eps'.replace(' ', '_'), 
                        dpi=400, bbox_inches='tight', pad_inches=0)
            #plt.savefig(out_path + f'/{file_name}.pdf'.replace(' ', '_'),
            #            dpi=400, bbox_inches='tight', pad_inches=0)
            if clf: plt.clf()

def custom_symlog_formatter(value, _):
    return int(value) if int(value) == value else value

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

def precisions_recalls(distances, class_m, thresholds, k):
    precisions = []
    recalls = []
    for threshold in thresholds:
        predictions = distances < threshold

        tp = sum(i * np.sum(predictions[class_m == i]) for i in range(1, k+1)) / k
        fp = sum((k - i) * np.sum(predictions[class_m == i]) for i in range(0, k)) / k
        fn = np.sum(class_m) / k - tp

        #tp = np.sum(predictions[class_m == 1])
        #fp = np.sum(predictions[class_m == 0])
        #fn = np.sum(class_m) - tp

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
            class_correspondance_score = sum(labels[i][c] == labels[j][c] for c in class_characterisation)
            D.append(class_correspondance_score)
    
    return np.array(D)

def prevalence(class_m, k):
    return np.mean(class_m) / k

def compute_mean_aupr(
            D_d, labels, class_characterisation, N = 20):
        
        mean_aupr = {}
        aupr = {}

        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)
        k = len(class_characterisation)

        for metric in M_ID:
            mean_aupr[metric] = {}
            aupr[metric] = {}
            for sparse in S_NAME:
                aupr[metric][sparse] = {}
                auprs = []

                for i in range(N):
                    distance_m = D_d[i][sparse][metric].to_numpy()
                    distance_m = distance_m / np.max(distance_m)

                    precisions, recalls = precisions_recalls(distance_m, class_m, thresholds, k)
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
        mean_aupr['prevalence'] = prevalence(class_m, k)

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        out_path = f'results/clustering/{class_str}'
        Path(out_path).mkdir(parents = True, exist_ok = True)

        with open(f'{out_path}/mean_aupr.json', 'w') as json_file:
            json.dump(mean_aupr, json_file)

        return mean_aupr

def compute_perturbation_distances_deviation(D_d, N = N_PERTURBATIONS):
        step = STEP_PERTURBATIONS
        dd = { p : { m : {s : {g : {} 
                               for g in G_NAME} 
                               for s in TS_NAME} 
                               for m in M_ID} 
                               for p in P_NAME }

        for error in ERRORS:
            for p in P_NAME:
                for m in M_ID:
                    for s in TS_NAME:
                        for g in G_NAME:
                            w_ERRs = []
                            for w in W_NAME:
                                sparsifier_vec = D_d[p][(g, w)][s][m]['mean'][:(N//step)+1]
                                full_vec       = D_d[p][(g, w)]['Full'][m]['mean'][:(N//step)+1]
                                w_ERRs.append(error(full_vec, sparsifier_vec))
                            
                            dd[p][m][s][g][error.id] = np.array(w_ERRs)
                        dd[p][m][s][error.id] = np.concatenate([dd[p][m][s][g][error.id] for g in G_NAME])

                    for g in G_NAME:
                        mean = np.sum(np.concatenate([np.abs(dd[p][m][s][g][error.id]) for s in TS_NAME]))
                        for s in TS_NAME:
                            dd[p][m][s][g][('normalized', error.id)] = dd[p][m][s][g][error.id] / mean

                    for s in TS_NAME:
                        dd[p][m][s][('normalized', error.id)] = np.concatenate([dd[p][m][s][g][('normalized', error.id)] for g in G_NAME])
                        
                        normalized_errors = np.concatenate([dd[p][m][s][g][('normalized', error.id)] for g in G_NAME])
                        dd[p][m][s][('MAN', error.id)] = np.mean(np.abs(normalized_errors))

            dd[('max', error.id)] = {}
            for m in M_ID:
                dd[('max', error.id)][m] = np.max(
                    np.concatenate([np.abs(dd[p][m][s][error.id]) for (p, s) in product(P_NAME, TS_NAME)]))

        return dd

def compute_gaussian_noise_distances_deviation(D_d):
        dd = { m : {s : {g : {} 
                               for g in G_NAME} 
                               for s in TS_NAME} 
                               for m in M_ID}
        for error in ERRORS:
            for m in M_ID:
                for s in TS_NAME:
                    for g in G_NAME:
                        w_ERRs = []
                        for w in W_NAME:
                            sparsifier_vec = D_d[(g, w)][s][m]['mean']
                            full_vec       = D_d[(g, w)]['Full'][m]['mean']
                            w_ERRs.append(error(full_vec, sparsifier_vec))
                        
                        dd[m][s][g][error.id] = np.array(w_ERRs)
                    dd[m][s][error.id] = np.concatenate([dd[m][s][g][error.id] for g in G_NAME])

                for g in G_NAME:
                    mean = np.sum(np.concatenate([np.abs(dd[m][s][g][error.id]) for s in TS_NAME]))
                    for s in TS_NAME:
                        dd[m][s][g][('normalized', error.id)] = dd[m][s][g][error.id] / mean

                for s in TS_NAME:
                    normalized_errors = np.concatenate([dd[m][s][g][('normalized', error.id)] for g in G_NAME])
                    dd[m][s][('MAN', error.id)] = np.mean(np.abs(normalized_errors))

        return dd

def portrait(G, shortest_paths = None):
    d = max(nx.diameter(G.subgraph(K)) for K in nx.connected_components(G))
    n = nx.number_of_nodes(G)
    if shortest_paths is None:
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

def graph_with_nice_portrait():
    def is_nice_portrait(G, Blk, k):
        spl = dict(nx.all_pairs_shortest_path_length(G))

        def are_distant_components(A, B, len = 1):
            return all(spl[a][b] > len for a in A for b in B)

        if len(Blk) != 3: return False
        #nodes = set(list(Blk.keys()) + [v for shell in Blk.values() for v, _ in shell])
        #return len(nodes) == len(Blk) * (k+1)

        components = [[u] + [v for v, _ in shell] for u, shell in Blk.items()]
        for A, B in combinations(components, 2):
            if not are_distant_components(A, B): return False
        return True


    found = False
    while not found:
        n = random.randint(20, 50)
        m = random.randint(30, 70)
        s = random.randint(1, 100)
        G = nx.gnm_random_graph(n, m, seed=s)
        G = G.subgraph(max(nx.connected_components(G), key=len))
        
        B = portrait(G)

        l = 2
        k = 3
        found = is_nice_portrait(G, B[l][k], k)

    print(f'Found a graph with a nice portrait: {n} nodes, {m} edges, seed = {s}')
    return G



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
            
            if display_nb_iterations and display_metric:
                legend = plt.legend(loc='upper left', framealpha = 0.5, fontsize = REPORT_FONTSIZE)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            plt.xlabel(f'# {perturbation}', alpha = display_nb_iterations, fontsize = REPORT_FONTSIZE)
            plt.ylabel(metric.name,  alpha = display_metric, fontsize = REPORT_FONTSIZE)

            plt.gca().xaxis.set_label_coords(0.5, -0.11)
            plt.gca().yaxis.set_label_coords(-0.12, 0.5)

            for label in plt.gca().get_yticklabels(): label.set_alpha(display_metric)
            for label in plt.gca().get_xticklabels(): label.set_alpha(display_nb_iterations)

            out_path = f'plots/_report/perturbation/distances'
            perturbation_nb = P_NAME.index(perturbation)
            save_file(out_path, f'{metric.name} {perturbation_nb}', eps = True)

        else:
            plt.title(f'{perturbation} on {weight} {graph}')
            plt.xlabel(f'# {perturbation}')
            plt.ylabel(metric.name)
            plt.legend(loc='upper left')

            out_path = f'plots/perturbation/{perturbation}/{metric.name}'
            save_file(out_path, f'{weight} {graph}')

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
        save_file(out_path, f'{weight} {graph}')

    def perturbation_deviation_by_graph(
            self, D_dd, metric, perturbation, error, ylim, report = False):
        
        display_perturbation = metric == METRICS[0]         if report else True
        display_graphs       = metric == METRICS[-1]        if report else True
        display_metric       = perturbation == P_NAME[0]    if report else True

        TS = TS_NAME.copy()
        if not report: TS.sort(key=lambda s: abs(D_dd[s][('MAN', error.id)]))

        _, ax = plt.subplots(figsize=(6, 4))
        #plt.subplots_adjust(left=0.15)

        for s in TS:
            for i, dev in enumerate(D_dd[s][error.id]):
                if i == 0:
                    ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5, label = s)
                else:
                    ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5)

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
        ax.set_ylim([-ylim, ylim]) if error.negative else ax.set_ylim([0, ylim])
        ax.set_yscale('symlog', linthresh = ylim/100, subs = list(range(2, 10, 2)))
        ax.yaxis.set_major_formatter(FuncFormatter(custom_symlog_formatter))
        ax.grid(axis='y', linestyle=':')

        if report:
            ax.set_title(f'{perturbation}\n', alpha = display_perturbation, fontsize=REPORT_FONTSIZE)
            if display_graphs and display_metric:
                legend = ax.legend(loc='upper left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)
            fontsize = 10 if metric.id == 'nlap' else REPORT_FONTSIZE
            ax.set_ylabel(f'{metric.name} {error.id}', alpha = display_metric, fontsize=fontsize)
            ax.yaxis.set_label_coords(-0.125, 0.5)
            for label in ax.get_yticklabels(): label.set_alpha(display_metric) 

            out_path = f'plots/_report/perturbation/deviation/{error.id}_by_graph'
            perturbation_nb = P_NAME.index(perturbation)
            save_file(out_path, f'{metric.name} {perturbation_nb}', eps = True)

        else:
            ax.set_title(f'{perturbation} test\n' + error.name)
            ax.set_ylabel(metric.name)
            ax.set_xlabel('Graphs')
            ax.legend(loc='lower left', fontsize='x-small')

            out_path = f'plots/perturbation/_Deviation/{error.id}'
            save_file(out_path, f'{metric.name} {perturbation}', clf = False)

            out_path = f'plots/perturbation/{perturbation}/{metric.name}'
            save_file(out_path, f'_{error.id}')


    def perturbation_deviation_by_sparsifier(
            self, D_dd, perturbation, error, report = False):
        
        plt.figure(figsize=(8, 4))
        #plt.subplots_adjust(left=0.15)

        TS = TS_NAME.copy()
        MET = METRICS.copy()

        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for m in MET))
        maximum = np.max([abs(D_dd[m][s][('MAN', error.id)]) for m, s in product(M_ID, TS)])

        for i, sparse in enumerate(TS):
            MET.sort(key=lambda m: abs(D_dd[m.id][sparse][('MAN', error.id)]))
            for j, metric in enumerate(MET):
                mean = D_dd[metric.id][sparse][('MAN', error.id)] / maximum
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
        plt.ylim([0, 1.1])

        if report:
            plt.title(f'{perturbation}\n', fontsize=REPORT_FONTSIZE)
            display_deviation = perturbation == P_NAME[0]
            plt.ylabel(f'Average normalized {error.id}', fontsize=REPORT_FONTSIZE, alpha = display_deviation)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)
            if display_deviation:
                legend = plt.legend(handles=legend_handles, loc='upper left', 
                   handleheight=3, handlelength=3, framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            for label in plt.gca().get_yticklabels(): label.set_alpha(display_deviation)

            out_path = f'plots/_report/perturbation/deviation/{error.id}_by_sparsifier'
            perturbation_nb = P_NAME.index(perturbation)
            save_file(out_path, perturbation_nb, eps = True)

        else:
            plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small', 
                   handleheight=3, handlelength=3)
            plt.title(f'{perturbation} test')
            plt.ylabel(f'Average normalized {error.id}')

            out_path = f'plots/perturbation/_Deviation/{error.id}'
            save_file(out_path, f'_{perturbation} by sparsifier', clf = False)

            out_path = f'plots/perturbation/{perturbation}'
            save_file(out_path, f'_{error.id} by sparsifier')


    def perturbation_deviation_by_metric(
            self, D_dd, perturbation, error):
        
        MET = METRICS.copy()
        TS = TS_NAME.copy()
        
        MET.sort(key=lambda m: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for s in TS))
        maximum = np.max([abs(D_dd[m][s][('MAN', error.id)]) for m, s in product(M_ID, TS)])

        plt.figure(figsize=(8, 4))

        for i, metric in enumerate(MET):
            for sparse in TS:
                mean = D_dd[metric.id][sparse][('MAN', error.id)] / maximum
                plt.errorbar(mean, i, fmt='o', color=S_COLORS[sparse], markersize=5)

        plt.axvline(x=0, linestyle='-', color='black', label='Full')
                
        legend_handles = []
        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for m in MET))
        for sparse in TS:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        plt.title(f'{perturbation} test')
        plt.xlabel(f'Average normalized {error.id}')
        plt.xlim([-1.02, 1.02])
        plt.yticks(np.arange(len(MET)), labels=
                   [m.name.replace(' ', '\n') for m in MET], fontsize='x-small')
        plt.ylabel('Metrics')
        plt.ylim([-0.6, len(MET)-1 + 0.6])
        plt.grid(axis='x')

        out_path = f'plots/perturbation/_Deviation/{error.id}'
        save_file(out_path, f'_{perturbation} by metric', clf = False)

        out_path = f'plots/perturbation/{perturbation}'
        save_file(out_path, f'_{error.id} by metric')


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

        if report:
            display_y_label = metric == METRICS[0]
            plt.title(f'{metric.name}\n', fontsize = REPORT_FONTSIZE)
            plt.ylabel('Distance', fontsize = REPORT_FONTSIZE, alpha = display_y_label)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)
            plt.xlabel('σ', fontsize = REPORT_FONTSIZE)
            if display_y_label:
                legend = plt.legend(loc='upper left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            out_path = f'plots/_report/gaussian_noise/distances'
            metric_nb = METRICS.index(metric)
            save_file(out_path, metric_nb, eps = True)

        else:
            if y_axis_range is not None: plt.ylim(y_axis_range)

            plt.title(f'Gaussian Noise N(0, σ) on {weight} {graph}')
            plt.ylabel(metric.name)
            plt.legend(loc='upper left')
            plt.xlabel('σ')

            out_path = f'plots/gaussian_noise/{metric.name}'
            save_file(out_path, f'{weight} {graph}')

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
        save_file(out_path, f'{weight} {graph}')

    def gaussian_noise_deviation_by_graph(
            self, D_dd, metric, error, report = False):

        TS = TS_NAME.copy()
        if not report: TS.sort(key=lambda s: abs(D_dd[s][('MAN', error.id)]))

        _, ax = plt.subplots(figsize=(6, 4))
        #plt.subplots_adjust(left=0.15)

        for s in TS:
            for i, dev in enumerate(D_dd[s][error.id]):
                if i == 0:
                    ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5, label = s)
                else:
                    ax.errorbar(i, dev, fmt='o', color=S_COLORS[s], markersize=5)

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

        ylim = np.max([np.max(np.abs(D_dd[s][error.id])) for s in TS])
        ylim = 10 ** np.ceil(np.log10(ylim))
        ax.set_ylim([-ylim, ylim]) if error.negative else ax.set_ylim([0, ylim])
        ax.set_yscale('symlog', linthresh = ylim/100, subs = list(range(2, 10, 2)))
        ax.yaxis.set_major_formatter(FuncFormatter(custom_symlog_formatter))
        ax.grid(axis='y', linestyle=':')
        
        if report:
            display_y_label = metric == METRICS[0]
            ax.set_title(f'{metric.name}\n', fontsize = REPORT_FONTSIZE)
            ax.set_ylabel(f'{error.name}', fontsize = REPORT_FONTSIZE, alpha = display_y_label)
            ax.yaxis.set_label_coords(-0.125, 0.5)
            if display_y_label:
                legend = ax.legend(loc='upper left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            out_path = f'plots/_report/gaussian_noise/deviation/{error.id}_by_graph'
            metric_nb = METRICS.index(metric)
            save_file(out_path, metric_nb, eps = True)

        else:
            ax.set_title(f'Gaussian noise test')
            ax.set_xlabel('Graphs')
            ax.set_ylabel(f'{metric.name}\n{error.id}', fontsize='x-small')
            ax.legend(loc='lower left', fontsize='x-small')

            out_path = f'plots/gaussian_noise/_Deviation/{error.id}'
            save_file(out_path, metric.name, clf = False)

            out_path = f'plots/gaussian_noise/{metric.name}'
            save_file(out_path, f'_{error.id}')

    def gaussian_noise_deviation_by_sparsifier(
            self, D_dd, error, report = False):
        
        plt.figure(figsize=(8, 4))
        #plt.subplots_adjust(left=0.15)

        MET = METRICS.copy()
        TS = TS_NAME.copy()

        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for m in MET))
        maximum = max(abs(D_dd[m][s][('MAN', error.id)]) for m, s in product(M_ID, TS))

        for i, sparse in enumerate(TS):
            MET.sort(key=lambda m: abs(D_dd[m.id][sparse][('MAN', error.id)]))
            for j, metric in enumerate(MET):
                mean = D_dd[metric.id][sparse][('MAN', error.id)] / maximum
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
        plt.ylim([0, 1.1])

        if report:
            legend = plt.legend(handles=legend_handles, loc='upper left', framealpha = 0.5, 
                   handleheight=3, handlelength=3, fontsize = 'small')
            #for handle in legend.legendHandles: handle.set_alpha(0.5)
            #for text in legend.get_texts(): text.set_alpha(0.5)
            
            plt.title('Gaussian noise\n', fontsize=REPORT_FONTSIZE)
            plt.ylabel(f'Average normalized {error.id}', fontsize=REPORT_FONTSIZE)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)

            out_path = f'plots/_report/gaussian_noise/deviation'
            save_file(out_path, f'{error.id}_by_sparsifier', eps = True)

        else:
            plt.title(f'Gaussian noise test')
            plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small', 
                   handleheight=3, handlelength=3)
            plt.ylabel(f'Average normalized {error.id}')

            out_path = f'plots/gaussian_noise/_Deviation/{error.id}'
            save_file(out_path, '_by sparsifier')


    def gaussian_noise_deviation_by_metric(
            self, D_dd, error):
        
        MET = METRICS.copy()
        TS = TS_NAME.copy()

        MET.sort(key=lambda m: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for s in TS))
        maximum = np.max([abs(D_dd[m][s][('MAN', error.id)]) for m, s in product(M_ID, TS)])

        plt.figure(figsize=(8, 4))

        for i, metric in enumerate(MET):
            for sparse in TS:
                mean = D_dd[metric.id][sparse][('MAN', error.id)] / maximum
                plt.errorbar(mean, i, fmt='o', color=S_COLORS[sparse], markersize=5)

        plt.axvline(x=0, linestyle='-', color='black', label='Full')
                
        legend_handles = []
        TS.sort(key=lambda s: sum(abs(D_dd[m.id][s][('MAN', error.id)]) for m in MET))
        for sparse in TS:
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))
        plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

        plt.title(f'Gaussian noise test')
        plt.xlabel(f'Average normalized {error.id}')
        plt.xlim([-1.02, 1.02])
        plt.yticks(np.arange(len(MET)), labels=
                   [m.name.replace(' ', '\n') for m in MET], fontsize='x-small')
        plt.ylabel('Metrics')
        plt.ylim([-0.6, len(MET)-1 + 0.6])
        plt.grid(axis='x')

        out_path = f'plots/gaussian_noise/_Deviation/{error.id}'
        save_file(out_path, '_by metric')


    def clustering(
            self, distances_matrix, metric, labels, report = False):

        linkage = {}
        for sparse in S_NAME:
            linkage[sparse] = hierarchy.linkage(distances_matrix[sparse][metric.id].to_numpy(),
                                     method='ward', optimal_ordering=True)

        display_labels = [" ".join(str(i) for i in list(l.values())) for l in labels]
        colors_dict = {"Graph" : [G_COLORS[l['graph']] for l in labels],
                       "Weight": [W_COLORS[l['weight']] for l in labels]}

        for sparse in S_NAME:
            plt.figure(figsize=(25, 12))
            if report:
                D = hierarchy.dendrogram(linkage[sparse], 
                                        color_threshold=0.3*max(linkage[sparse][:, 2]),
                                        labels = [""] * len(labels),
                                        leaf_font_size=2)
                rt.plot(D, colorlabels = colors_dict)
                
                out_path = f'plots/_report/clustering/dendrograms'
                save_file(out_path, sparse, eps = True, dpi = 400)

            else:
                plt.subplots_adjust(bottom=0.15, top=0.85)
                D = hierarchy.dendrogram(linkage[sparse], 
                                        labels = display_labels, 
                                        color_threshold=0.3*max(linkage[sparse][:, 2]),
                                        leaf_font_size=1)
                rt.plot(D, colorlabels = colors_dict, colorlabels_legend = COLORS_LEGENDS)
                #plt.title(f'Clustering on {sparse} graphs\nMetric: {metric.name}')

                out_path = f'plots/clustering/{metric.name}/Dendrogram'
                save_file(out_path, sparse, dpi = 400)

    def pedagogic_clustering(self, s):
        from scipy.spatial.distance import squareform

        np.random.seed(s)
        distance_matrix = np.random.rand(8, 8)
        distance_matrix = squareform(distance_matrix, checks=False, force='tovector')
        linkage = hierarchy.linkage(distance_matrix,
                                    method='ward', optimal_ordering=True)

        colors_dict = {"Graph" : ["black"] * 4 + ["gray"] * 4}

        plt.figure(figsize=(10, 5))
        D = hierarchy.dendrogram(linkage,
                                color_threshold= 0,
                                labels = [""] * 8,
                                leaf_font_size=2)
        rt.plot(D, colorlabels = colors_dict)
        
        out_path = f'plots/_report/clustering/pedagogic'
        save_file(out_path, 'dendrogram', eps = True, dpi = 400)

    def clustering_label(self):
        _, axs = plt.subplots(1, 2, figsize=(5, 3))  # adjusted size to fit wider boxes
        plt.subplots_adjust(wspace=-0.4, hspace=0)  # adjust spacing if needed

        legends = {'Graph': G_COLORS, 'Weight': W_COLORS}

        for ax, (key, colors) in zip(axs, legends.items()):
            box_width = 0.4  # wider box
            for i, (label, color) in enumerate(colors.items()):
                # Drawing a wider rectangle
                ax.add_patch(patches.Rectangle((0.6, (3-i) * 0.2 + 0.2), box_width, 0.1, edgecolor='black', facecolor=color))
                # Adjusting text alignment to the top of the box
                ax.text(1.15, (3-i) * 0.2 + 0.215, label, verticalalignment='bottom', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(key, fontsize=14, fontweight='bold')

        out_path = 'plots/_report/clustering/dendrograms'
        save_file(out_path, 'legend', eps = True)

    def clustering_precision_recall_curve(
            self, D_d, metric, labels, class_characterisation, report = False):
        
        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)
        k = len(class_characterisation)
        rdm_chance = prevalence(class_m, k)

        plt.figure(figsize=(6, 4))

        pra = {}
        
        for sparse in S_NAME:
            distance_m = D_d[sparse][metric.id].to_numpy()
            distance_m = distance_m / np.max(distance_m)

            precisions, recalls = precisions_recalls(distance_m, class_m, thresholds, k)
            aupr = auc(recalls, precisions)

            pra[sparse] = (precisions, recalls, aupr)

        if not report: S_NAME.sort(key=lambda s: pra[s][2], reverse=True)

        for sparse in S_NAME:
            precisions, recalls, aupr = pra[sparse]

            aupr_string = f'(AUPR = {aupr:.3f}) ' if not report else ''

            plt.plot(recalls, precisions, marker=',', 
                     label=f'{aupr_string}{sparse}', color=S_COLORS[sparse])

        aupr_string = f'(AUPR = {rdm_chance:.3f}) ' if not report else ''
        plt.plot([0, 1], [rdm_chance, rdm_chance], 
                 linestyle=':', color='black',
                 label=f'{aupr_string}Random classification')

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)

        if report:
            plt.subplots_adjust(left=0.2, bottom=0.15)
            display_title   = metric == METRICS[0]
            display_x_label = metric == METRICS[-1]
            display_y_label = class_characterisation == CLASSIFICATION_MODES[0]
            plt.title(f'Classification by {class_str}\n', alpha = display_title, fontsize = REPORT_FONTSIZE)
            if display_x_label and display_y_label:
                legend = plt.legend(loc='lower left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            plt.xlabel('Recall', alpha = display_x_label, fontsize = REPORT_FONTSIZE)
            plt.ylabel(f'{metric.name}\n\nPrecision', alpha = display_y_label, fontsize = REPORT_FONTSIZE)

            plt.gca().xaxis.set_label_coords(0.5, -0.125)
            plt.gca().yaxis.set_label_coords(-0.125, 0.5)

            for label in plt.gca().get_xticklabels(): label.set_alpha(display_x_label)
            for label in plt.gca().get_yticklabels(): label.set_alpha(display_y_label)

            out_path = f'plots/_report/clustering/precision-recall'
            classification_mode_nb = CLASSIFICATION_MODES.index(class_characterisation)
            save_file(out_path, f'{metric.name} {classification_mode_nb}', eps = True)

        else:
            plt.title(f'Precision-Recall curve\n'
                    +f'Class: {class_str}\n'
                    +f'Metric: {metric.name}', fontsize='small')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            
            plt.legend(loc='lower left', fontsize='x-small')

            out_path = f'plots/clustering/{metric.name}/Precision-Recall'
            save_file(out_path, class_str, clf = False)

            out_path = f'plots/clustering/_Precision-Recall/{class_str}'
            save_file(out_path, metric.name)

    def clustering_precision_recall_curve_3D(
            self, D_d, metric, labels, class_characterisation, report = False):

        class_m = class_matrix(labels, class_characterisation)
        thresholds = np.linspace(0, 1, 10**4)
        k = len(class_characterisation)
        rdm_chance = prevalence(class_m, k)

        pra = {}

        for sparse in S_NAME:
            distance_m = D_d[sparse][metric.id].to_numpy()
            distance_m = distance_m / np.max(distance_m)

            precisions, recalls = precisions_recalls(distance_m, class_m, thresholds, k)
            aupr = auc(recalls, precisions)

            pra[sparse] = (precisions, recalls, aupr)

        if not report: S_NAME.sort(key=lambda s: pra[s][2], reverse=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for sparse in S_NAME:
            precisions, recalls, aupr = pra[sparse]

            ax.plot(recalls, thresholds, precisions, linestyle='-', 
                    label=f'(AUPR = {aupr:.3f}) {sparse}', color=S_COLORS[sparse])
            ax.plot(recalls, thresholds, np.zeros_like(precisions), 
                    linestyle='-', color=S_COLORS[sparse], alpha=0.3)
                
        ax.plot([0, 1], [0, 1], [rdm_chance, rdm_chance],
                linestyle=':', color='black',
                label=f'(AUPR = {rdm_chance:.3f}) Random classification')
        ax.plot([0, 1], [0, 1], [0, 0],
                linestyle=':', color='black', alpha=0.3)

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        ax.set_xlabel('Recall')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.legend(loc='upper right', fontsize='xx-small')
        ax.grid(True)

        if report:
            plt.subplots_adjust(top=0.85)
            ax.set_title(f'Classification by {class_str}\n'
                +f'with {metric.name}\n')
            
            out_path = f'plots/_report/clustering/precision-recall'
            save_file(out_path, f'3D', eps = True)

        else:
            ax.set_title(f'Precision-Recall curve\n'
                +f'Classification by {class_str}\n'
                +f'with {metric.name}', fontsize='small')

            out_path = f'plots/clustering/{metric.name}/Precision-Recall'
            save_file(out_path, f'3D {class_str}', clf = False)

            out_path = f'plots/clustering/_Precision-Recall/{class_str}'
            save_file(out_path, f'3D {metric.name}')

    def clustering_aupr_by_sparsifier(
            self, D_a, class_characterisation, display_str = True, report = False):
        
        plt.figure(figsize=(8, 4))

        S_NAME.sort(key=lambda s: sum(D_a[m.id][s]['mean'] for m in METRICS), reverse=True)

        for i, sparse in enumerate(S_NAME):
            MT = METRICS.copy()
            MT.sort(key=lambda m: D_a[m.id][sparse]['mean'], reverse=True)
            for j, metric in enumerate(MT):
                mean = D_a[metric.id][sparse]['mean']
                std = D_a[metric.id][sparse]['std'] if display_str else 0
                offset = -0.3 + j*(0.6/(len(MT)-1))
                plt.errorbar(i + offset, mean, yerr=std, fmt=M_MARKERS[metric], markersize = 3, capsize=1,
                             elinewidth=0.5, capthick=0.5, color = S_COLORS[sparse])
                
        rdm_chance = plt.axhline(y=D_a['prevalence'], linestyle=':', label = 'Random classification', color='black')
                
        legend_handles = [rdm_chance]
        for metric in METRICS:
            legend_handles.append(plt.Line2D([0], [0], marker = M_MARKERS[metric], label=metric.name, color = 'black'))


        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.ylim([0, 1.05])
        plt.grid(axis='y')
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xticks(np.arange(len(S_NAME)), labels=S_NAME,  fontsize='x-small')

        if report:
            display_y_label = class_characterisation == CLASSIFICATION_MODES[0]
            if display_y_label:
                legend = plt.legend(handles=legend_handles, loc='lower left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            plt.title(f'Classifcation by {class_str}\n', fontsize=REPORT_FONTSIZE)
            plt.ylabel('AUPR', fontsize=REPORT_FONTSIZE, alpha = display_y_label)

            plt.gca().yaxis.set_label_coords(-0.125, 0.5)
            for label in plt.gca().get_yticklabels(): label.set_alpha(display_y_label)

            out_path = f'plots/_report/clustering/aupr_by_sparsifier'
            classification_mode_nb = CLASSIFICATION_MODES.index(class_characterisation)
            save_file(out_path, classification_mode_nb, eps = True)

        else:
            plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')
            
            plt.title(f'Area Under Precision-Recall Curves\nClass: {class_str}')
            plt.xlabel('Sparsifier')
            plt.ylabel('AUPR')

            out_path = f'plots/clustering/_Precision-Recall/{class_str}'
            save_file(out_path, '_AUPRs by sparsifier')


    def clustering_aupr_by_metric(
            self, D_a, class_characterisation, display_str = True, report = False):
        
        MET = METRICS.copy()
        if not report: MET.sort(key=lambda m: sum(D_a[m.id][s]['mean'] for s in S_NAME), reverse=True)

        plt.figure(figsize=(10, 6))

        for i, metric in enumerate(MET):
            S_NAME.sort(key=lambda s: D_a[metric.id][s]['mean'], reverse=True)
            for j, sparse in enumerate(S_NAME):
                mean = D_a[metric.id][sparse]['mean']
                std = D_a[metric.id][sparse]['std'] if display_str else 0
                offset = -0.4 + j*(0.8/(len(S_NAME)-1))
                plt.errorbar(mean, i + offset, xerr=std, fmt='o', color=S_COLORS[sparse], markersize=3,
                             linewidth=0.5, capthick = 0.5, capsize=2)
                plt.annotate(f'{mean:.2f} ± {std:.2f}', xy=(mean, i + offset), xytext=(-41, 10),
                             textcoords='offset pixels', fontsize=5)
                
        rdm_chance = plt.axvline(x=D_a['prevalence'], linestyle=':', label = 'Random classification', color='black')
                
        legend_handles = [rdm_chance]
        S_NAME.sort(key=lambda s: sum(D_a[m.id][s]['mean'] for m in MET), reverse=True)
        for sparse in S_NAME:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color=S_COLORS[sparse], markersize=3, label=sparse))

        class_str = '(' + ', '.join(s.capitalize() for s in class_characterisation) + ')'
        plt.xlim([0, 1.02])
        plt.ylim([-0.6, len(MET)-1 + 0.6])

        if report:
            display_y_label = class_characterisation == CLASSIFICATION_MODES[0]
            if display_y_label:
                legend = plt.legend(handles=legend_handles, loc='lower left', framealpha = 0.5)
                #for handle in legend.legendHandles: handle.set_alpha(0.5)
                #for text in legend.get_texts(): text.set_alpha(0.5)

            plt.title(f'Classifcation by {class_str}\n', fontsize=REPORT_FONTSIZE)
            plt.xlabel('AUPR', fontsize=REPORT_FONTSIZE)
            plt.yticks(np.arange(len(METRICS)), labels=
                [m.name.replace(' ', '\n') + '\n' for m in METRICS], fontsize=REPORT_FONTSIZE-2,
                alpha = display_y_label, rotation = 90)

            out_path = f'plots/_report/clustering/aupr_by_metric'
            classification_mode_nb = CLASSIFICATION_MODES.index(class_characterisation)
            save_file(out_path, classification_mode_nb, eps = True)

        else:
            plt.legend(handles=legend_handles, loc='lower left', fontsize='x-small')

            plt.title(f'Area Under Precision-Recall Curves\n'
                    +f'Class: {class_str}')
            plt.xlabel('AUPR')
            plt.ylabel('Metrics')
            plt.yticks(np.arange(len(MET)), labels=
                [m.name.replace(' ', '\n') for m in MET], fontsize='x-small')

            out_path = f'plots/clustering/_Precision-Recall/{class_str}'
            save_file(out_path, '_AUPRs by metric')


    def graph(
            self, G, G_name, pos = None, sparse = None, e_width_from_weight = True, 
            node_color = 'black', size = 3/4, alpha_full = None, highlighting_factor = 1,
            explicit_out_path = None):
        
        plt.figure(figsize=(30, 24))

        if pos is None: pos = nx.spring_layout(G)

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
            e_path, file_name = explicit_out_path
            out_path = f'plots/_report/graphs/{e_path}'
            save_file(out_path, file_name, no_bbox=True)

        else:
            out_path = f'plots/_report/graphs/{G_name[0]}/{G_name[1]}'
            save_file(out_path, sparse_name, no_bbox=True)

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

    def metric_space(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plane with adjusted settings
        x = np.linspace(-0.5, 1.5, 10)
        y = np.linspace(-0.1, 0.1, 10)
        x, y = np.meshgrid(x, y)
        z_plane = np.zeros(x.shape)  # Plane at z = 0

        # Plotting the plane with z = 0
        ax.plot_surface(x, y, z_plane, alpha=0.5, color='cyan')  # Cyan colored plane

        # Plotting the specific curve
        t = np.linspace(0, 1, 100)
        x_curve = t
        y_curve = np.zeros_like(t)
        z_curve = t**2

        ax.plot(x_curve, y_curve, z_curve, color='r', linewidth=2, linestyle = 'dotted')  # Red line for the curve

        # Plotting the projection of the curve onto the plane
        ax.plot(x_curve, y_curve, np.zeros_like(z_curve), color='magenta', linewidth=1,
                linestyle = 'dotted')  # Magenta line for the projection

        # Removing x and y ticks and y label, setting x and z labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0, 1])  # Set some z ticks for reference

        ax.set_xlabel("Graphs")
        ax.set_ylabel("")
        ax.set_zlabel("Distance")

        # Setting the limits to emphasize the curve
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([0, 1])

        out_path = f'plots/_report'
        save_file(out_path, 'metric_space')


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
            
        out_path = f'plots/_report/sparsifiers'
        save_file(out_path, f'{sparse_name}', no_bbox=True)

    def portrait_shell(self, G, pos_seed = 42, save = False):
        
        pos = nx.spring_layout(G, seed=pos_seed)

        B = portrait(G)
        Blk = B[2][3]

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
        
        if save:
            out_path = f'plots/_report'
            save_file(out_path, 'portrait', no_bbox=True)
        else:
            plt.show()

    def perturbation_distances_toy(
            self, dfs, graph, weight, metric, N = N_PERTURBATIONS, g_first = True):

        #plt.subplots_adjust(left=0.15)
        
        df = dfs[(graph, weight)]
        for sparse, label in zip(['Full', 'APSP'], ['Full', 'Sparsifier']):
            m = df[sparse][metric.id]['mean'].to_numpy()
            x = np.concatenate(([0], np.arange(1, N+1, 5)))

            plt.plot(x, m, linestyle='-',
                         label = label, color=S_COLORS[sparse])
            
        if g_first: graph, weight = weight, graph

        for label in plt.gca().get_yticklabels(): label.set_alpha(0)

        plt.xlabel(f'# perturbation')
        plt.ylabel('Distance')
        plt.gca().yaxis.set_label_coords(-0.05, 0.5)
        plt.legend(loc='upper left')

        out_path = f'plots/_report'
        save_file(out_path, 'toy example perturbation distance', eps = True)


def load_perturbation_dfs(compute = False):
    dico_names = ['D_p_dists', 'D_p_edges', 'M_p_dists', 'M_p_edges', 'D_p_dd']
    if compute:
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

        D_p_dd = compute_perturbation_distances_deviation(D_p_dists)

        dicos = [D_p_dists, D_p_edges, M_p_dists, M_p_edges, D_p_dd]

        out_path = f'results/perturbation/json'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        """
        for name, dico in zip(dico_names, dicos):
            with open(f'{out_path}/{name}.json', "w") as json_file:
                json.dump(dico, json_file)
                """

    else:
        dicos = []
        for name in dico_names:
            with open(f'results/perturbation/json/{name}.json', "r") as json_file:
                dicos.append(json.load(json_file))

    return dicos

def load_gaussian_noise_dfs(compute = False):
    dico_names = ['D_g_dists', 'D_g_edges', 'M_g_dists', 'M_g_edges', 'D_g_dd']
    if compute:
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

        D_g_dd = compute_gaussian_noise_distances_deviation(D_g_dists)

        dicos = [D_g_dists, D_g_edges, M_g_dists, M_g_edges, D_g_dd]

        """
        out_path = f'results/gaussian_noise/json'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        for name, dico in zip(dico_names, dicos):
            with open(f'{out_path}/{name}.json', "w") as json_file:
                json.dump(dico, json_file)
                """

    else:
        dicos = []
        for name in dico_names:
            with open(f'results/gaussian_noise/json/{name}.json', "r") as json_file:
                dicos.append(json.load(json_file))

    return dicos

def load_clustering_dfs(compute = False, K = K_TEST_REP):
    D_c = {}

    for i in range(K):
        D_c[i] = {}
        for s in S_NAME:
            D_c[i][s] = pd.read_csv(f'results/clustering/{i}/{s}.csv')

    with open(f'results/clustering/{0}/labels.json', "r") as json_file:
        labels = json.load(json_file)

    if compute:
        D_a_gw = compute_mean_aupr(D_c, labels, ['graph', 'weight'], N = K)
        D_a_g  = compute_mean_aupr(D_c, labels, ['graph'], N = K)
        D_a_w  = compute_mean_aupr(D_c, labels, ['weight'],  N = K)
    else:
        with open(f'results/clustering/(Graph, Weight)/mean_aupr.json', "r") as json_file:
            D_a_gw = json.load(json_file)
        with open(f'results/clustering/(Graph)/mean_aupr.json', "r") as json_file:
            D_a_g = json.load(json_file)
        with open(f'results/clustering/(Weight)/mean_aupr.json', "r") as json_file:
            D_a_w = json.load(json_file)

    best_approx_indices = [D_a_gw['best_approx_index'], D_a_g['best_approx_index'], D_a_w['best_approx_index']]
    best_approx_index = Counter(best_approx_indices).most_common(1)[0][0]

    D_a = {'graph, weight': D_a_gw, 'graph': D_a_g, 'weight': D_a_w}

    return D_c, labels, D_a, best_approx_index
