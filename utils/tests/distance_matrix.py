import argparse
import pandas as pd
import networkx as nx
from utils.static import METRICS
from utils.tests.functions import *
from pathlib import Path

def str_to_bool(v):
    if v in ['True', 'False', '']:
        return v == 'True'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graphs_path', metavar='graphs_path', type=str, nargs=1,
                        help='Path to the graphs file')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    
    args = parser.parse_args()

    args.graphs_path = args.graphs_path[0]

    return args


args = args()

df_graphs = pd.read_csv(f'graphs.csv') # Given by condor
graphs = [nx.from_pandas_edgelist(g_edges, edge_attr=True) for _, g_edges in df_graphs.groupby('graph_index')]

if args.toy:
    graphs = graphs[:3]

distance_matrices = {}

clustering = Clustering()

for m in METRICS:
    distance_matrices[m.id] = clustering.distance_matrix(graphs, m, args.print)

if args.save:
    Path(args.graphs_path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(distance_matrices).to_csv(f'{args.graphs_path}/distance_matrices.csv', index=False)