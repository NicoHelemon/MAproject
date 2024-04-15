from utils.static import *
from utils.tests import *
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
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    parser.add_argument('-mode', metavar='mode', type=str, nargs=1,
                        help='Graph generation mode or distance computation mode')
    
    args = parser.parse_args()

    args.mode = args.mode[0]

    return args


args = args()

args_list = []

for g in G_NAME:
    for w in W_NAME:
        for p, p_name in zip(P_ID, P_NAME):
            path = f'results/perturbation/{p_name}/{g}/{w}'
            
            Path(path).mkdir(parents=True, exist_ok=True)
            args_list.append(
                (f'-G {g} -W {w} -P {p} -toy {args.toy} -print {args.print} -save {args.save} -mode {args.mode}',
                ['net.dat', path])
            )
        

build_condor('perturbation', args_list)
build_sh('perturbation.sh', 'pert.py', len(args_list[0][0].split()))

Path(f'logs/perturbation').mkdir(parents=True, exist_ok=True)
os.system(f'condor_submit utils/condor/perturbation.condor')