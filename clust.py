import argparse

from utils.static import *
from utils.tests import *
from utils.condor.condor import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', metavar='S', type=str, nargs='?', 
                        const = '', default= '', help='Sparsifier')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    parser.add_argument('-mode', metavar='mode', type=str, nargs=1,
                        help='Graph generation mode or distance matrices computation mode')
    
    args = parser.parse_args()

    args.mode = args.mode[0]

    return args


args = args()

clustering = Clustering()

if args.S: sparsifier = S_MAP[args.S]

if args.mode == 'write_graphs':
    if args.toy:
        clustering.generate_graphs(
            weight_n_sample = 1, graph_n_sample = 1, gn_n_sample = 1, 
            time_printing = args.print, save = args.save)
    else:
        clustering.generate_graphs(
            time_printing = args.print, save = args.save)

elif args.mode == 'compute_matrices':
    if args.toy:
        clustering(sparsifier, time_printing = args.print, save = args.save, N = 9)
    else:
        clustering(sparsifier, time_printing = args.print, save = args.save)

else:
    raise ValueError('Invalid mode')