import numpy as np
import timeit
import time as ti
import scipy.cluster.hierarchy as hierarchy
from utils.helper import *

from utils.graphs import *

def distance_vs_perturbation_test(
        G, weight, perturbation, metrics, K = 100, N = 1000, step = 1, time_printing = False):
    
    if time_printing:
        time = []
        t_iter = K*N

    w_g, w_e = weight
    G = w_g(G)
    apsp_G = apsp(G)
    
    distances_full = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    distances_apsp = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    n_edges_full = [[G.number_of_edges()] for _ in range(K)]
    n_edges_apsp = [[apsp_G.number_of_edges()] for _ in range(K)]
    
    prec_full = {}
    prec_apsp = {}
    
    # Precomputation phase for G
    for m_id, md, m in metrics:
        if m is not None:
            prec_full[m_id] = m(G)
            prec_apsp[m_id] = m(apsp_G)
            md_f = md(G, G, prec_full[m_id])
            md_a = md(apsp_G, apsp_G, prec_apsp[m_id])

        else:
            md_f = md(G, G)
            md_a = md(apsp_G, apsp_G)

        for i in range(K):
            distances_full[m_id][i].append(md_f)
            distances_apsp[m_id][i].append(md_a)

    for i in range(K):
        H = G.copy()
        for j in range(N):
            perturbation(H, w_e)
            n_edges_full[i].append(H.number_of_edges())
            if j % step == 0:
                start = timeit.default_timer()
                apsp_H = apsp(H)
                n_edges_apsp[i].append(apsp_H.number_of_edges())
                
                for m_id, md, m in metrics:
                    if m is not None:
                        distances_full[m_id][i].append(md(H, G, prec_full[m_id]))
                        distances_apsp[m_id][i].append(md(apsp_H, apsp_G, prec_apsp[m_id]))
                    else:
                        distances_full[m_id][i].append(md(H, G))
                        distances_apsp[m_id][i].append(md(apsp_H, apsp_G))

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    c_iter = i*N + j
                    print(f'Iteration {i}, Perturbation nb {j}')
                    print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                    print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter) / step))))
                    print()
    
    return distances_full, distances_apsp, n_edges_full, n_edges_apsp


def gaussian_noise_test(
        G, weight, metrics, sigmas = np.linspace(0, 0.1, 20+1).tolist(), K = 100, time_printing = False):

    G = weight(G)
    N = len(sigmas)
    
    if time_printing:
        time = []
        t_iter = N*K

    distances_full = {σ : {m_id : [] for m_id, _, _ in metrics} for σ in sigmas}
    distances_apsp = {σ : {m_id : [] for m_id, _, _ in metrics} for σ in sigmas}
    n_edges_apsp   = {σ : [] for σ in sigmas}

    for i, σ in enumerate(sigmas):
        for j in range(K):
            start = timeit.default_timer()
            H1 = add_gaussian_noise(G.copy(), σ, weight)
            H2 = add_gaussian_noise(G.copy(), σ, weight)
            apsp_H1 = apsp(H1)
            apsp_H2 = apsp(H2)

            n_edges_apsp[σ].append(apsp_H1.number_of_edges())
            n_edges_apsp[σ].append(apsp_H2.number_of_edges())

            for m_id, md, _ in metrics:
                distances_full[σ][m_id].append(md(H1, H2))
                distances_apsp[σ][m_id].append(md(apsp_H1, apsp_H2))

            if time_printing:
                time.append(timeit.default_timer() - start)
                c_iter = i*K + j
                print(f'Sigma nb {i}, Iteration nb {j}')
                print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                print()

    return distances_full, distances_apsp, n_edges_apsp


def distance_matrix(graphs, metric, metric_id, time_printing = False):

    N = len(graphs)

    if time_printing:
        time = []
        print(f'Distance matrix computation with {metric_id}')
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

def hierarchical_clustering(graphs, metrics, time_printing = False):

    clusters = {}

    for m_id, md, _ in metrics:
        clusters[m_id] = hierarchy.linkage(
            distance_matrix(graphs, md, m_id, time_printing), 'ward', optimal_ordering=True).flatten()

    return clusters

def clustering_gaussian_noise_test(G, weights, metrics, sigma, K = 3, N = 6, time_printing = False):

    if time_printing:
        time = []
        t_iter = len(weights)*K*N
        print("Graph initialization")

    graphs_full = []
    graphs_apsp = []
    graphs_label = []
    for i, w in enumerate(weights):
        for j in range(K):
            H = w(G.copy())
            for k in range(N):
                start = timeit.default_timer()
                H_full = add_gaussian_noise(H.copy(), sigma, w)
                H_apsp = apsp(H_full)
                graphs_full.append(H_full)
                graphs_apsp.append(H_apsp)
                graphs_label.append(f'{f_str(w)} {j}')

                if time_printing:
                    time.append(timeit.default_timer() - start)
                    c_iter = i*K*N + j*N + k
                    print(f'{f_str(w)}, Iteration (weight) nb {j}, Iteration (noise) nb {k}')
                    print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                    print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (t_iter - 1 - c_iter)))))
                    print()

    h_clustering_full = hierarchical_clustering(graphs_full, metrics, time_printing)
    h_clustering_apsp = hierarchical_clustering(graphs_apsp, metrics, time_printing)

    return h_clustering_full, h_clustering_apsp, graphs_label
    
    