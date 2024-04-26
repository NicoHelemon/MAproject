import argparse
from pathlib import Path

from utils.static import *
from utils.tests import *
from utils.condor.condor import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph_gen', metavar='graph_gen', type=str_to_bool, nargs='?',
                        const=True, default=False, help='Graph generation')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    parser.add_argument('-I', metavar='I', type=int, nargs='?', 
                        const = 0, default= 0, help='Iteration chunk')
    
    args = parser.parse_args()

    return args

args = args()

if args.graph_gen:
    args_list = []

    for i in range(20):
        transfer_input_files = []
        args_list.append(
            (f'-graph_gen -toy {args.toy} -print {args.print} -save {args.save} -i {i}',
            transfer_input_files))

else:
    args_list = []

    for i in range(args.I*10, (args.I+1)*10):
        for s in S_ID:
            transfer_input_files = [f'{Clustering(i).out_path_root}/graphs.csv']
            args_list.append(
                (f'-S {s} -toy {args.toy} -print {args.print} -save {args.save} -i {i}',
                transfer_input_files))
            


build_condor('clustering', args_list)
build_sh('clustering.sh', 'clust.py', len(args_list[0][0].split()))

Path(f'logs/clustering').mkdir(parents=True, exist_ok=True)
os.system(f'condor_submit utils/condor/clustering.condor')

