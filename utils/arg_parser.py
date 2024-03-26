import argparse
from utils.static import G_NAME, W_NAME, P_ID, T_NAME

def print_testing(args):
    toy = 'toy ' if args.toy else ''
    out = f'About to run a {toy}{args.test} test\n'
    out += f'  On graph(s):\t\t{args.G}\n'

    if args.test == 'perturbation':
        out += f'  With weight(s):\t{args.W}\n'
        out += f'  And perturbation(s):\t{args.P}\n'

    elif args.test == 'gaussian_noise':
        out += f'  With weight(s):\t{args.W}\n'

    elif args.test == 'clustering_gaussian_noise':
        out += f'  With sigma:\t\t{args.sigma}\n'

    out += f'Time printing:\t{args.print}\n'
    out += f'Saving results:\t{args.save}\n'
    
    print(out)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', metavar='test', type=str, nargs=1, 
                        help='Test to run')
    parser.add_argument('-toy', metavar='toy', type=bool, nargs='?', 
                        const=True, default=False, help='Run a toy test i.e. with a small number of computations')
    parser.add_argument('-G', metavar='G', type=str, nargs='*', 
                        default=G_NAME, help='List of graphs')
    parser.add_argument('-W', metavar='W', type=str, nargs='*', 
                        default=W_NAME, help='List of weights')
    parser.add_argument('-P', metavar='P', type=str, nargs='*', 
                        default=P_ID, help='List of perturbations')
    parser.add_argument('-sigma', metavar='sigma', type=float, nargs='?', 
                        const=0.05, default=0.05, help='Gaussian noise variance')
    parser.add_argument('-print', metavar='print', type=str, nargs='?', 
                        const=True, default=False, help='Time printing')
    parser.add_argument('-save', metavar='save', type=bool, nargs='?',
                        const=True, default=False, help='Save results')
    
    args = parser.parse_args()
    
    if not args.toy and not args.save:
        args.save = True

    args.test = args.test[0].replace('_', ' ')
    if args.test not in T_NAME:
        raise ValueError(f'Invalid test. Test must be one of:\n\t{T_NAME}')

    return args

def parse_args_to_string(args):
    if not isinstance(args, dict):
        args = vars(args)

    out = ''
    for arg, value in args.items():
        if isinstance(value, str):
            value = value.replace(' ', '_')
        if isinstance(value, list):
            value = ' '.join(value)
        out += f' -{arg} {value}'
    return out
