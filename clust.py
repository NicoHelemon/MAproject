import argparse

from utils.static import *
from utils.tests import *
from utils.condor.condor import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph_gen', metavar='graph_gen', type=str_to_bool, nargs='?',
                        const=True, default=False, help='Graph generation')
    parser.add_argument('-S', metavar='S', type=str, nargs='?', 
                        const = '', default= '', help='Sparsifier')
    parser.add_argument('-toy', metavar='toy', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-print', metavar='print', type=str_to_bool, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=str_to_bool, nargs='?',
                        const=True, default=True, help='Save results')
    parser.add_argument('-i', metavar='i', type=int, nargs='?', 
                        const = 0, default= 0, help='Iteration number')
    
    args = parser.parse_args()

    return args


args = args()

clustering = Clustering(args.i)

if args.S: sparsifier = S_MAP[args.S]

if args.graph_gen:
    if args.toy:
        clustering.generate_graphs(
            weight_n_sample = 1, graph_n_sample = 1, gn_n_sample = 1, 
            time_printing = args.print, save = args.save)
    else:
        clustering.generate_graphs(
            time_printing = args.print, save = args.save)

else:
    if args.toy:
        clustering(sparsifier, time_printing = args.print, save = args.save, N = 9)
    else:
        clustering(sparsifier, time_printing = args.print, save = args.save)