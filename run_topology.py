import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import matplotlib.pyplot as plt

# Set parameter file
import parameter_files.default as parameter_file

param = parameter_file.parameter

a = E1(debug=False, param=param)

c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = True, it=1000)

a.calculate_c_lmlpmp(only_diag=True)

a.plot_c_l_and_realizations(c_l_a)