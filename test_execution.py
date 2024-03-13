import pandas as pd

from utils.graphs import *
from utils.distances import *
from utils.tests import *
from utils.portrait_divergence import *
from utils.perturbations import *

perturbations = list(zip(
    ["Edge removal", "Edge addition", "Rdm edge switching", "Deg preserving edge switching"],
    [edge_removal, edge_addition, random_edge_switching, degree_preserving_edge_switching]
))

graphs = list(zip(
    ["BA", "ER", "ABCD"],
    [uniform_w(BA()), uniform_w(ER()), uniform_w(ABCD())]
))

metrics = list(zip(
    ["lap", "nlap", "netlsd", "portrait"],
    [lap_spec_d, nlap_spec_d, netlsd_heat_d, portrait_div_d],
    [nx.laplacian_spectrum, nx.normalized_laplacian_spectrum, netlsd.heat, None]
))

sigmas = np.linspace(0, 4, 10).tolist()[1:]

"""
for graph_name, G in graphs:
    for perturbation_name, perturbation in perturbations:
        print(graph_name + " : " + perturbation_name)
        distances_full, distances_apsp = distance_vs_perturbation_test(G, perturbation, metrics, K = 100, N = 1000, step = 1)

        df_full = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in distances_full.items()}, axis=1)
        df_apsp = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in distances_apsp.items()}, axis=1)

        df_full.to_csv(f'results/{perturbation_name} {graph_name} full.csv', index=False)
        df_apsp.to_csv(f'results/{perturbation_name} {graph_name} apsp.csv', index=False)
        """

for graph_name, G in graphs:
    print(graph_name)
    distances_full, distances_apsp = gaussian_noise_test(G, metrics, sigmas, 50)

    df_full = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in distances_full.items()}, axis=0).reset_index(level=1, drop=True)
    df_apsp = pd.concat({σ : pd.DataFrame(pd.DataFrame(a).agg(['mean', 'std']).unstack()).T 
                     for σ, a in distances_apsp.items()}, axis=0).reset_index(level=1, drop=True)

    df_full.to_csv(f'results/Gaussian noise {graph_name} full.csv')
    df_apsp.to_csv(f'results/Gaussian noise {graph_name} apsp.csv')
