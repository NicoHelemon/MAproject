from utils.tests import *
from utils.static import *


test = Test()

"""
for g in GRAPHS:
    for w in WEIGHTS:
        for p in PERTURBATIONS:
            test.perturbation(
                g, w, p, METRICS, K = 1, N = 5, time_printing = True)
    

for g in GRAPHS:
    for w in WEIGHTS:
        test.gaussian_noise(
            g, w, METRICS, sigmas = np.linspace(0, 0.1, 1+1).tolist(), K = 1, time_printing = True) 
"""
            
            
for g in GRAPHS:
     test.clustering_gaussian_noise(
          g, WEIGHTS, METRICS, 0.1, K = 1, N = 1, time_printing = True)
          

         