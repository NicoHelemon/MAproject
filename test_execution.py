import pandas as pd

from utils.graphs import *
from utils.distances import *
from utils.tests import *
from utils.portrait_divergence import *
from utils.perturbations import *
from utils.helper import *

def distance_vs_perturbation_test_execution(
          G, G_name, weight, perturbation, metrics, K = 100, N = 1000, step = 1):

        d_full, d_apsp = distance_vs_perturbation_test(
             weight(G), perturbation, metrics, K, N, step)

        df_full = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d_full.items()}, axis=1)
        df_apsp = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in d_apsp.items()}, axis=1)

        df_full.to_csv(f'results/perturbation/{f_str(weight)}/{f_str(perturbation)} {f_str(weight)} {G_name} full.csv', index=False)
        df_apsp.to_csv(f'results/perturbation/{f_str(weight)}/{f_str(perturbation)} {f_str(weight)} {G_name} apsp.csv', index=False)

def gaussian_noise_test_execution(
        G, G_name, weight, metrics, sigmas, K = 40, min = 0, max = None, absolute = False):
    
    d_full, d_apsp = gaussian_noise_test(
        weight(G), metrics, sigmas, K, min, max, absolute)

    df_full = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in d_full.items()}, axis=0).reset_index(level=1, drop=True)
    df_apsp = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in d_apsp.items()}, axis=0).reset_index(level=1, drop=True)

    df_full.to_csv(f'results/gaussian_noise/{f_str(weight)} {G_name} full.csv')
    df_apsp.to_csv(f'results/gaussian_noise/{f_str(weight)} {G_name} apsp.csv')


graphs = list(zip([BA(), ER(), ABCD()],
                  ["BA", "ER", "ABCD"]))

weights = [uniform, exp, log_normal]

metrics = list(zip(
    ["lap", "nlap", "netlsd", "portrait"],
    [lap_spec_d, nlap_spec_d, netlsd_heat_d, portrait_div_d],
    [nx.laplacian_spectrum, nx.normalized_laplacian_spectrum, netlsd.heat, None]
))

perturbations = [edge_removal, 
                 edge_addition, 
                 random_edge_switching, 
                 degree_preserving_edge_switching]

sigmas = np.linspace(0, 0.1, 10+1).tolist()

"""
for G, G_name in graphs:
    for w in weights:
        for p in perturbations:
            distance_vs_perturbation_test_execution(
            G, G_name, w, p, metrics, K = 10, N = 500, step = 5)
            """

for G, G_name in graphs:
    gaussian_noise_test_execution(
        G, G_name, uniform, metrics, sigmas, K = 40, min = 0, max = 1, absolute = False)
    #gaussian_noise_test_execution(
    #    G, G_name, exp, metrics, sigmas, K = 40, min = 0, max = None, absolute = False)
         