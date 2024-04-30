
from utils.graphs import *
from utils.metrics import *
from utils.perturbations import *
from utils.sparsifiers import *

N_PERTURBATIONS = 250

E_MES = ['count', 'size', 'components']

GRAPHS = [BA, ER, RG, ABCD]

#SPARSIFIERS = [Full(), APSP(), LocalDegree(), kNeighbor(), Random(), Threshold(), EffectiveResistance()]
SPARSIFIERS = [Full(), Random(), 
               EffectiveResistance(random = True, qf_preserving = True),
               EffectiveResistance(random = True, qf_preserving = False),
               EffectiveResistance(random = False, qf_preserving = True),
               EffectiveResistance(random = False, qf_preserving = False)]

WEIGHTS = [Uniform(), Exponential(), Lognormal()]

METRICS = [LaplacianSpectrum(), NormalizedLaplacianSpectrum(), NetlsdHeat(), PortraitDivergence()]

PERTURBATIONS = [EdgeRemoval(), EdgeAddition(), RandomEdgeSwitching(), DegreePreservingEdgeSwitching()]

E_NAME = ['# Edges', 'Size', '# Components']

G_NAME = [g.__name__ for g in GRAPHS]

S_NAME = [s.name for s in SPARSIFIERS]

S_ID   = [s.id for s in SPARSIFIERS]

W_NAME = [w.name for w in WEIGHTS]

M_NAME = [m.name for m in METRICS]

M_ID   = [m.id for m in METRICS]

P_NAME = [p.name for p in PERTURBATIONS]

P_ID   = [p.id for p in PERTURBATIONS]

def union(d1, d2):
    return {**d1, **d2}

E_MAP = dict(zip(E_MES, E_NAME))

G_MAP = dict(zip(G_NAME, GRAPHS))

S_MAP = union(dict(zip(S_ID, SPARSIFIERS)), dict(zip(S_NAME, SPARSIFIERS)))

W_MAP = dict(zip(W_NAME, WEIGHTS))

P_MAP = union(dict(zip(P_ID, PERTURBATIONS)), dict(zip(P_NAME, PERTURBATIONS)))


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

#S_COLORS = dict(zip(S_NAME, ['black', 'blue', 'darkgreen', 'lightgreen', 'red', 'orange', 'pink']))
S_COLORS = dict(zip(S_NAME, ['black', 'red', 'pink', 'blue', 'green', 'orange']))

M_MARKERS = dict(zip(METRICS, ['o', 's', 'D', '^']))
M_HATCHES = dict(zip(METRICS, ['-', 'x', 'o', '*']))

COLORS_LEGENDS = {"Weight": {"colors" : W_COLORS.values(),
                             "labels" : W_COLORS.keys()},
                  "Graph":  {"colors" : G_COLORS.values(),
                             "labels" : G_COLORS.keys()}}

TS_NAME = S_NAME[1:]


VISU_GRAPHS_ARGS = {
    'BA'   : [500, None, 2000],
    'ER'   : [500, None, 2000, 0],
    'RG'   : [500, 0.075, 33],
    'ABCD' : [500, 2.55, 1.5, 10, 0.1]
}