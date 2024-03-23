import pandas as pd
from pathlib import Path

from utils.graphs import *
from utils.distances import *
from utils.tests import *
from utils.portrait_divergence import *
from utils.perturbations import *
from utils.helper import *

def distance_vs_perturbation_test_execution(
          G, G_name, weight, perturbation, metrics, K = 20, N = 1000, step = 5, time_printing = False):

        d_full, d_apsp, n_edges_full, n_edges_apsp = distance_vs_perturbation_test(
             G, weight, perturbation, metrics, K, N, step, time_printing)

        df_full = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d_full.items()}, axis=1)
        df_apsp = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d_apsp.items()}, axis=1)
        df_n_edges_full = pd.DataFrame(n_edges_full).T.agg(['mean', 'std'], axis=1)
        df_n_edges_apsp = pd.DataFrame(n_edges_apsp).T.agg(['mean', 'std'], axis=1)

        out_path = f'results/perturbation/{f_str(weight[0])}/'
        Path(out_path).mkdir(parents = True, exist_ok = True)
        out_path += f'{f_str(perturbation)} {G_name}'

        df_full.to_csv(f'{out_path} full.csv', index=False)
        df_apsp.to_csv(f'{out_path} apsp.csv', index=False)
        df_n_edges_full.to_csv(f'{out_path} n edges full.csv', index=False)
        df_n_edges_apsp.to_csv(f'{out_path} n edges apsp.csv', index=False)

def gaussian_noise_test_execution(
        G, G_name, weight, metrics, 
        sigmas = np.linspace(0, 0.1, 20+1).tolist(), K = 100, time_printing = False):
    
    d_full, d_apsp, n_edges_apsp = gaussian_noise_test(
        G, weight, metrics, sigmas, K, time_printing)

    df_full = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in d_full.items()}, axis=0).reset_index(level=1, drop=True)
    df_apsp = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in d_apsp.items()}, axis=0).reset_index(level=1, drop=True)
    df_n_edges_apsp = pd.DataFrame.from_dict(n_edges_apsp, orient='index').agg(['mean', 'std'], axis=1)

    out_path = f'results/gaussian_noise/'
    Path(out_path).mkdir(parents = True, exist_ok = True)
    out_path += f'{f_str(weight)} {G_name}'

    df_full.to_csv(f'{out_path} full.csv')
    df_apsp.to_csv(f'{out_path} apsp.csv')
    df_n_edges_apsp.to_csv(f'{out_path} n edges apsp.csv')

def clustering_gaussian_noise_test_execution(
          G, G_name, weights, metrics, sigma, K = 3, N = 6, time_printing = False):
     
    clustering_full, clustering_apsp, graphs_labels = clustering_gaussian_noise_test(
          G, weights, metrics, sigma, K, N, time_printing)
    
    w_string = ' '.join(f'{f_str(w)}' for w, _ in weights)
    out_path = f'results/clustering/gaussian_noise/'
    Path(out_path).mkdir(parents = True, exist_ok = True)
    out_path += f'{G_name} {w_string} {sigma}'
     
    pd.DataFrame.from_dict(clustering_full).to_csv(f'{out_path} full.csv', index=False)
    pd.DataFrame.from_dict(clustering_apsp).to_csv(f'{out_path} apsp.csv', index=False)
    pd.DataFrame(graphs_labels).to_csv(f'{out_path} labels.csv', index=False)



graphs = list(zip([BA(), ER(), ABCD()],
                  ["BA", "ER", "ABCD"]))

weights = [uni, exp, log]
weights_e = [np.random.uniform, np.random.exponential, np.random.lognormal]
weights_p = [1, 0.5, 0.5]

metrics = list(zip(
    ["lap", "nlap", "netlsd", "portrait"],
    [lap_spec_d, nlap_spec_d, netlsd_heat_d, portrait_div_d],
    [nx.laplacian_spectrum, nx.normalized_laplacian_spectrum, netlsd.heat, None]
))

perturbations = [edge_removal, 
                 edge_addition, 
                 random_edge_switching, 
                 degree_preserving_edge_switching]



"""
for G, G_name in graphs:
    for w in zip(weights, weights_e):
        for p in perturbations:
            distance_vs_perturbation_test_execution(
            G, G_name, w, p, metrics)

for G, G_name in graphs:
    for w in weights:
        gaussian_noise_test_execution(G, G_name, w, metrics)

for G, G_name in graphs:
     clustering_gaussian_noise_test_execution(
          G, G_name, zip(weights, weights_p), metrics, 0.05)

for G, G_name in graphs:
    for w in list(zip(weights, weights_e))[0:1]:
        for p in perturbations[0:1]:
            distance_vs_perturbation_test_execution(
                G, G_name, w, p, metrics[0:3], K = 10, N = 500, step = 5, time_printing = True)

for G, G_name in graphs:
    for w in weights[0:1]:
        gaussian_noise_test_execution(
             G, G_name, w, metrics[0:3], sigmas = np.linspace(0, 0.1, 10+1).tolist(), 
             K = 20, time_printing = True)

for G, G_name in graphs[0:1]:
     clustering_gaussian_noise_test_execution(
          G, G_name, weights, metrics[0:3], 0.1, K = 3, N = 6, time_printing = True)
          """

for G, G_name in graphs:
     clustering_gaussian_noise_test_execution(
          G, G_name, zip(weights, weights_p), metrics, 0.05)

         