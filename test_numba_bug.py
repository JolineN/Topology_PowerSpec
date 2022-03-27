from src.topology import Topology
from src.tools import *
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb

l_max = 5
n_max = 5
L = 1

a = Topology(debug=False, l_max = l_max, n_max=n_max, L = L * 28 * 1e3, use_numba=True, fig_name = 'l_max_{}_n_max_{}'.format(l_max, n_max))


alm_parallel = a.do_numba_bug(with_parallel = True)
print('2')
alm_no_parallel = a.do_numba_bug(with_parallel = False)
print(np.sum(alm_parallel))
print(np.sum(alm_no_parallel))