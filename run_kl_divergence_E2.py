#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E2 import E2

import numpy as np
import time

import parameter_files.default_E2 as parameter_file_E2

# Set parameter file

root = 'kl_runs/non_cubic_E2_no_tilt_no_shift/'

l_max_list = np.array([10, 20, 30, 40, 50])

param_E2 = parameter_file_E2.parameter

kl = np.zeros(5)
a_t = np.zeros(5)

for i, l_max in enumerate(l_max_list):
  print(i, l_max)
  param_E2['l_max'] = l_max
  param_E2['Lx'] = 1.2
  param_E2['Ly'] = 1.2
  param_E2['Lz'] = 1.2
  param_E2['x0'] = np.array([0.0, 0.0, 0], dtype=np.float64)

  a = E2(param=param_E2)

  start = time.time()
  cur_kl, cur_a_t = a.calculate_exact_kl_divergence(parallel_cov = True)
  end = time.time()
  print('Elapsed time parallel:', end-start)

  kl[i] = cur_kl
  a_t[i] = cur_a_t
  print('E2 Currenct kl!')
  print(kl)
  print('Arthur')
  print(a_t)

  np.save(root+'kl_E2_func_lmax_10_50_L12.npy', kl)
  np.save(root+'a_t_E2_func_lmax_10_50_L12.npy', a_t)