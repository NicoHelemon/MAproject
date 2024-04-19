
from utils.graphs import *
from utils.metrics import *
from utils.perturbations import *
from utils.sparsifiers import *


E_MES = ['count', 'size']

GRAPHS = [BA, ER, RG, ABCD]

SPARSIFIERS = [Full(), APSP(), LocalDegree(), kNeighbor(), Random(), Threshold(), EffectiveResistance()]

WEIGHTS = [Uniform(), Exponential(), Lognormal()]

METRICS = [LaplacianSpectrum(), NormalizedLaplacianSpectrum(), NetlsdHeat(), PortraitDivergence()]

PERTURBATIONS = [EdgeRemoval(), EdgeAddition(), RandomEdgeSwitching(), DegreePreservingEdgeSwitching()]


G_NAME = [g.__name__ for g in GRAPHS]

S_NAME = [s.name for s in SPARSIFIERS]

S_ID   = [s.id for s in SPARSIFIERS]

W_NAME = [w.name for w in WEIGHTS]

P_NAME = [p.name for p in PERTURBATIONS]

P_ID   = [p.id for p in PERTURBATIONS]


G_MAP = dict(zip(G_NAME, GRAPHS))

S_MAP = dict(zip(S_ID, SPARSIFIERS))

W_MAP = dict(zip(W_NAME, WEIGHTS))

P_MAP = dict(zip(P_ID, PERTURBATIONS))


# To obtain graphs s.t. k(G) = 1, |E| \in [4970, 4985] with the subsequent graph generators
# (Fixing densities or power-law exponents do not guarantee a precise enough number of edges)
FIXED_SEED = {
    'BA' :   range(6),
    'ER' :   [10, 39, 40, 59, 77, 93],
    'RG' :   [30, 64, 75, 90, 111, 172],
    'ABCD' : [5, 10, 24, 44, 64, 95]
}

G_COLORS = dict(zip(G_NAME, ['green', 'blue', 'pink', 'violet']))

W_COLORS = dict(zip(W_NAME, ['red', 'orange', 'yellow']))

S_COLORS = dict(zip(S_NAME, ['black', 'blue', 'darkgreen', 'lightgreen', 'red', 'orange', 'pink']))

COLORS_LEGENDS = {"Weight": {"colors" : W_COLORS.values(),
                             "labels" : W_COLORS.keys()},
                  "Graph":  {"colors" : G_COLORS.values(),
                             "labels" : G_COLORS.keys()}}