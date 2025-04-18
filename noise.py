import argparse
from utils.tests import *
from utils.static import *

def str_to_bool(v):
    if v in ['True', 'False', '']:
        return v == 'True'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args():
    parser = argparse.ArgumentParser()
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

    args.G = args.G[0]
    args.W = args.W[0]

    return args

args = args()

gaussian_noise = GaussianNoise()

g = G_MAP[args.G]()
w = W_MAP[args.W]

if args.toy:
    gaussian_noise(g, w, sigmas = np.linspace(0, 0.1, 1+1).tolist(), K = 2,
            time_printing = args.print, save = args.save)
else:
    gaussian_noise(g, w,
            time_printing = args.print, save = args.save)
