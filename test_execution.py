import pandas as pd

from utils.graphs import *
from utils.distances import *
from utils.dist_vs_pert_test import *
from utils.portrait_divergence import *
from utils.perturbations import *


perturbations = list(zip(
    ["Edge removal", "Edge addition", "Rdm edge switching", "Deg preserving edge switching"],
    [edge_removal, edge_addition, random_edge_switching, degree_preserving_edge_switching]
))

graphs = list(zip(
    ["BA", "ER", "ABCD"],
    [weighted_BA(), weighted_ER(), weighted_ABCD]
))

metrics = list(zip(
    ["lap", "adj", "nlap", "netlsd", "portrait"],
    [lap_spec_d, adj_spec_d, nlap_spec_d, netlsd_heat_d, portrait_div_d],
    [nx.laplacian_spectrum, nx.adjacency_spectrum, nx.normalized_laplacian_spectrum, netlsd.heat, None]
))


for graph_name, G in graphs[2:3]:
    for perturbation_name, perturbation in perturbations[0:2]:
        print(graph_name + " : " + perturbation_name)
        distances_full, distances_apsp = test(G, perturbation, metrics)

        df_full = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in distances_full.items()}, axis=1)
        df_apsp = pd.concat({k : pd.DataFrame(a).T.agg(['mean', 'std'], axis=1) for k, a in distances_apsp.items()}, axis=1)

        df_full.to_csv("results/" + perturbation_name + " " + graph_name + " full.csv", index=False)
        df_apsp.to_csv("results/" + perturbation_name + " " + graph_name + " apsp.csv", index=False)

