#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E1 import E1

import numpy as np
import time

import parameter_files.default_E1 as parameter_file_E1

# Set parameter file
l_max_list = np.array([10, 20, 30, 40, 50])

param_E1 = parameter_file_E1.parameter

kl = np.zeros(5)
a_t = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  param_E1['l_max'] = l_max
  param_E1['Lx'] = 1.4
  param_E1['Ly'] = 1.4
  param_E1['Lz'] = 1.1
  param_E1['beta'] = 90
  param_E1['alpha'] = 90
  param_E1['gamma'] = 0
  param_E1['x0'] = np.array([0, 0, 0], dtype=np.float64)

  a = E1(param=param_E1, make_run_folder = False)

  start = time.time()
  cur_kl, cur_a_t = a.calculate_exact_kl_divergence(parallel_cov = True)
  end = time.time()
  print('Elapsed time parallel:', end-start)

  kl[i] = cur_kl
  a_t[i] = cur_a_t
  print('E1 Currenct kl!')
  print(kl)
  print('Arthur')
  print(a_t)

  np.save('kl_Lz11_lmax_10_50.npy', kl)
  np.save('a_Lz11_lmax_10_50.npy', a_t)