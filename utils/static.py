
from utils.graphs import *
from utils.metrics import *
from utils.perturbations import *

MODES = ['full', 'apsp']

E_MES = ['count', 'size']

GRAPHS = [BA, ER, ABCD]

WEIGHTS = [Uniform(), Exponential(), Lognormal()]

METRICS = [LaplacianSpectrum(), NormalizedLaplacianSpectrum(), NetlsdHeat(), PortraitDivergence()]

PERTURBATIONS = [EdgeRemoval(), EdgeAddition(), RandomEdgeSwitching(), DegreePreservingEdgeSwitching()]


G_NAME = [g.__name__ for g in GRAPHS]

W_NAME = [w.name for w in WEIGHTS]

P_ID   = [p.id for p in PERTURBATIONS]


G_MAP = dict(zip(G_NAME, GRAPHS))

W_MAP = dict(zip(W_NAME, WEIGHTS))

P_MAP = dict(zip(P_ID, PERTURBATIONS))


# To obtain graphs s.t. |E| \in [4970, 4985] with the subsequent graph generators
# (Fixing densities or power-law exponents do not guarantee a precise enough number of edges)
FIXED_SEED = {
    'BA' :   range(6),
    'ER' :   [10, 39, 40, 59, 77, 93],
    'ABCD' : [5, 10, 24, 44, 64, 95]
}


LABEL_COLORS = {
    '3 x 3' : ['lightcoral', 'red', 'darkred', 
            'lightblue', 'blue', 'darkblue', 
            'lightgreen', 'green', 'darkgreen'],
    '3 x 6' : []    
}

