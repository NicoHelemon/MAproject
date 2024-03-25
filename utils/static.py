
from utils.graphs import *
from utils.distances import *
from utils.perturbations import *

MODES = ['full', 'apsp']

E_MES = ['count', 'size']

GRAPHS = [BA(), ER(), ABCD()]

WEIGHTS = [Uniform(), Exponential(), Lognormal()]

METRICS = [LaplacianSpectrum(), NormalizedLaplacianSpectrum(), NetlsdHeat(), PortraitDivergence()]

PERTURBATIONS = [EdgeRemoval(), EdgeAddition(), RandomEdgeSwitching(), DegreePreservingEdgeSwitching()]