
from utils.graphs import *
from utils.metrics import *
from utils.perturbations import *

MODES = ['full', 'apsp']

E_MES = ['count', 'size']

GRAPHS = [BA, ER, ABCD]

WEIGHTS = [Uniform(), Exponential(), Lognormal()]

METRICS = [LaplacianSpectrum(), NormalizedLaplacianSpectrum(), NetlsdHeat(), PortraitDivergence()]

PERTURBATIONS = [EdgeRemoval(), EdgeAddition(), RandomEdgeSwitching(), DegreePreservingEdgeSwitching()]

from utils.tests import *

TESTS = [Perturbation(), GaussianNoise(), ClusteringGaussianNoise()]


G_NAME = [g.__name__ for g in GRAPHS]

W_NAME = [w.name for w in WEIGHTS]

P_ID   = [p.id for p in PERTURBATIONS]

T_NAME = [t.name for t in TESTS]


G_MAP = dict(zip(G_NAME, GRAPHS))

W_MAP = dict(zip(W_NAME, WEIGHTS))

P_MAP = dict(zip(P_ID, PERTURBATIONS))

