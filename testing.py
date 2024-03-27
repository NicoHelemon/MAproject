from utils.arg_parser import *
from utils.tests import *
from utils.static import *


args = args()
print("About to run the following test:")
print(test_description_str(args))
print()

graphs        =  [G_MAP[g] for g in args.G]
weights       =  [W_MAP[w] for w in args.W]
perturbations =  [P_MAP[p] for p in args.P]


if args.test == 'perturbation':
    Test = Perturbation()
    for g in graphs:
        g = g()
        for w in weights:
            for p in perturbations:
                if args.toy:
                    Test(g, w, p, METRICS, K = 2, N = 10, 
                            time_printing = args.print, save = args.save)
                else:
                    Test(g, w, p, METRICS, 
                            time_printing = args.print, save = args.save)

elif args.test == 'gaussian noise':
    Test = GaussianNoise()
    for g in graphs:
        g = g()
        for w in weights:
            if args.toy:
                Test(g, w, METRICS, sigmas = np.linspace(0, 0.1, 1+1).tolist(), K = 2,
                        time_printing = args.print, save = args.save)
            else:
                Test(g, w, METRICS,
                        time_printing = args.print, save = args.save)

elif args.test == 'clustering gaussian noise':
    Test = ClusteringGaussianNoise()
    for g in graphs:
        if args.toy:
            Test(g(), weights, METRICS, args.sigma, K = 1, N = 1,
                    time_printing = args.print, save = args.save)
        else:
            Test(g(), weights, METRICS, args.sigma,
                    time_printing = args.print, save = args.save)