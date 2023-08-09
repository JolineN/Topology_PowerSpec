import sys
sys.path.append("../..")
from src.E2 import E2
import numpy as np


import parameter_files.default_E2 as parameter_file


param = parameter_file.parameter

param['l_max'] = 20
param['Lx'] = 1.4
param['Ly'] = 1.4
param['Lz'] = 0.714*0.9
param['beta'] = 90
param['alpha'] = 90
param['gamma'] = 0
param['x0'] = np.array([0.35, 0, 0], dtype=np.float64)

a = E2(param=param, make_run_folder=True)

_, cor = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 20]]),
    'lp_ranges': np.array([[2, 20]])
  },
  save_cov = True)