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

for g in G_NAME:
    for w in W_NAME:
        args_list.append(
            (f'-G {g} -W {w} -toy {args.toy} -print {args.print} -save {args.save}',
            ['net.dat'])
        )

build_condor('gaussian_noise', args_list)
build_sh('gaussian_noise.sh', 'gaussian_noise.py', len(args_list[0][0].split()))

Path(f'logs/gaussian_noise').mkdir(parents=True, exist_ok=True)
os.system(f'condor_submit utils/condor/gaussian_noise.condor')