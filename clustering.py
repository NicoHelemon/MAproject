from utils.static import *
from utils.tests.functions import *
import argparse
from utils.condor.condor import *
from pathlib import Path

def str_to_bool(v):
    if v in ['True', 'False', '']:
        return v == 'True'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-create_graphs', metavar='graphs', type=str_to_bool, nargs='?',
                        const=True, default=False, help='Create graphs lists')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    
    args = parser.parse_args()

    return args

def save_graphs(graph_s, mode, path, time_printing):
    def to_df(graphs):
        dfs = []
        for i, graph in enumerate(graphs):
            df = nx.to_pandas_edgelist(graph)
            df['graph_index'] = i
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    graphs_full, graphs_apsp, graphs_label = mode(graph_s, WEIGHTS, time_printing = time_printing)

    Path(f'results/clustering/{path}/full').mkdir(parents=True, exist_ok=True)
    Path(f'results/clustering/{path}/apsp').mkdir(parents=True, exist_ok=True)

    to_df(graphs_full).to_csv(f'results/clustering/{path}/full/graphs.csv', index=False)
    to_df(graphs_apsp).to_csv(f'results/clustering/{path}/apsp/graphs.csv', index=False)
    pd.DataFrame(graphs_label).to_csv(f'results/clustering/{path}/labels.csv', index=False)


args = args()

if args.create_graphs:
    clustering = Clustering()

    for g, g_name in zip(GRAPHS, G_NAME):
        save_graphs((g, g_name), clustering.DWG_graphs, f'DWG/{g_name}', args.print)
        save_graphs((g, g_name), clustering.GDW_graphs, f'GDW/{g_name}', args.print)
    save_graphs(zip(GRAPHS, G_NAME), clustering.GGD_graphs, 'GGD', args.print)


DWG_args_list = []
GDW_args_list = []
GGD_args_list = []

def distance_matrix_args(graphs_path):
    return (f'-graphs_path results/clustering/{graphs_path} -toy {args.toy} -print {args.print} -save {args.save}',
            ['net.dat', f'results/clustering/{graphs_path}/graphs.csv'])

for g in G_NAME:
    DWG_args_list.append(distance_matrix_args(f'DWG/{g}/full'))
    DWG_args_list.append(distance_matrix_args(f'DWG/{g}/apsp'))
    GDW_args_list.append(distance_matrix_args(f'GDW/{g}/full'))
    GDW_args_list.append(distance_matrix_args(f'GDW/{g}/apsp'))

GGD_args_list.append(distance_matrix_args('GGD/full'))
GGD_args_list.append(distance_matrix_args('GGD/apsp'))

for args_list, mode in [(DWG_args_list, 'clustering_DWG'), (GDW_args_list, 'clustering_GDW'), (GGD_args_list, 'clustering_GGD')]:
    build_condor(mode, args_list)
    build_sh(f'{mode}.sh', 'distance_matrix.py', len(args_list[0][0].split()))
    
    Path(f'logs/{mode}').mkdir(parents=True, exist_ok=True)
    os.system(f'condor_submit utils/condor/{mode}.condor')

