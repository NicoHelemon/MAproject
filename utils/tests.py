import numpy as np
import distanceclosure as dc
import timeit
import time as ti

from utils.graphs import *

def distance_vs_perturbation_test(G, perturbation, metrics, K = 10, N = 500, step = 5):
    time = []
    apsp_G = dc.metric_backbone(G, weight='weight')
    
    distances_full = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    distances_apsp = {m_id : [ [] for _ in range(K) ] for m_id, _, _ in metrics}
    
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
            perturbation(H)
            if j % step == 0:
                start = timeit.default_timer()
                apsp_H = dc.metric_backbone(H, weight='weight')
                
                for m_id, md, m in metrics:
                    if m is not None:
                        distances_full[m_id][i].append(md(H, G, prec_full[m_id]))
                        distances_apsp[m_id][i].append(md(apsp_H, apsp_G, prec_apsp[m_id]))
                    else:
                        distances_full[m_id][i].append(md(H, G))
                        distances_apsp[m_id][i].append(md(apsp_H, apsp_G))

                time.append(timeit.default_timer() - start)
                print(f'Iteration {i}, Perturbation nb {j}')
                print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
                print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (K*N - 1 - i*N - j) / step))))
                print()
    
    return distances_full, distances_apsp


def gaussian_noise_test(G, metrics, sigmas, K = 50):
    time = []
    
    distances_full = {σ : {m_id : [] for m_id, _, _ in metrics} for σ in sigmas}
    distances_apsp = {σ : {m_id : [] for m_id, _, _ in metrics} for σ in sigmas}

    N = len(sigmas)
    for i, σ in enumerate(sigmas):
        for j in range(K):
            start = timeit.default_timer()
            H1 = add_gaussian_noise_w(G.copy(), σ)
            H2 = add_gaussian_noise_w(G.copy(), σ)
            apsp_H1 = dc.metric_backbone(H1, weight='weight')
            apsp_H2 = dc.metric_backbone(H2, weight='weight')

            for m_id, md, _ in metrics:
                distances_full[σ][m_id].append(md(H1, H2))
                distances_apsp[σ][m_id].append(md(apsp_H1, apsp_H2))

            time.append(timeit.default_timer() - start)
            print(f'Sigma nb {i}, Iteration nb {j}')
            print(f'Time spent               = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.sum(time)))))
            print(f'Estimated time remaining = ' + ti.strftime('%H:%M:%S', ti.gmtime(int(np.mean(time) * (N*K - 1 - i*K - j)))))
            print()

    return distances_full, distances_apsp
