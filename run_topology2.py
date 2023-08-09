#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E6 import E6
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E4 as parameter_file


param = parameter_file.parameter
param['topology'] = 'E2'
param['Lx'] = 1.1
param['Ly'] = 1.1
param['Lz'] = 1.1
param['beta'] = 70.0
param['alpha'] = 90.0
param['gamma'] = 0.0
param['l_max'] = 20
param['x0'] = np.array([0.213, 0, 0], dtype=np.float64)

if param['topology'] == 'E1':
  a = E1(param=param, make_run_folder=True)
elif param['topology'] == 'E2':
  a = E2(param=param, make_run_folder=True)
elif param['topology'] == 'E3':
  a = E3(param=param, make_run_folder=True)
elif param['topology'] == 'E4':
  a = E4(param=param, make_run_folder=True)
elif param['topology'] == 'E6':
  a = E6(param=param, make_run_folder=True)
else:
  exit()

a.calculate_c_lmlpmp(
  only_diag=True
)

# Plot the diagonal power spectrum and the realizations
# Good to see if there are any obvious bugs
a.plot_c_l_and_realizations(c_l_a=None)