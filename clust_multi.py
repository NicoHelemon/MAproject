import argparse
from pathlib import Path

from utils.static import *
from utils.tests import *
from utils.condor.condor import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    
    args = parser.parse_args()

    return args

args = args()

args_list = []

for s in S_NAME:
    transfer_input_files = ['results/clustering/graphs.csv']
    args_list.append(
        (f'-S {s} -toy {args.toy} -print {args.print} -save {args.save} -mode compute_matrices',
        transfer_input_files)
    )

build_condor('clustering', args_list)
build_sh('clustering.sh', 'clust.py', len(args_list[0][0].split()))

Path(f'logs/clustering').mkdir(parents=True, exist_ok=True)
os.system(f'condor_submit utils/condor/clustering.condor')

