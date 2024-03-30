import argparse
from .functions import *
from utils.static import *

def str_to_bool(v):
    if v in ['True', 'False', '']:
        return v == 'True'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', metavar='P', type=str, nargs=1,
                        help='Perturbation')
    parser.add_argument('-G', metavar='G', type=str, nargs=1, 
                        help='Graph')
    parser.add_argument('-W', metavar='W', type=str, nargs=1, 
                        help='Weight')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    
    args = parser.parse_args()

    return args

args = args()

perturbation = Perturbation

p = P_MAP[args.P]
g = G_MAP[args.G]()
w = W_MAP[args.W]

if args.toy:
    perturbation(g, w, p, METRICS, K = 2, N = 10, 
            time_printing = args.print, save = args.save)
else:
    perturbation(g, w, p, METRICS, 
            time_printing = args.print, save = args.save)
