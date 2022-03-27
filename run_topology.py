from src.topology import Topology
from src.E1 import E1
from src.tools import *
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import matplotlib.pyplot as plt

# Set parameter file
import parameter_files.default as parameter_file

param = parameter_file.parameter

a = E1(debug=True, param=param)

c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False, it=4)

a.calculate_c_lmlpmp(only_diag=True, with_parallel = False)

a.plot_c_l_and_realizations(c_l_a)