#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("..")
from src.E2 import E2

import numpy as np
import time

import parameter_files.default_E2 as parameter_file_E2

# Set parameter file
num = 11
L = np.linspace(0.4, 1.4, num)
root = '../kl_runs/cubic_E2_no_tilt_shift_x0_y0_02_015/'

l_max_list = np.array([50])

param_E2 = parameter_file_E2.parameter
print(L)

for l_max in l_max_list:
  kl = np.load(root+'kl_E2_lmax{}.npy'.format(l_max))
  a_t = np.load(root+'a_t_E2_lmax{}.npy'.format(l_max))

  #for i in range(L.size):
  i = 10
  param_E2['l_max'] = l_max
  param_E2['Lx'] = L[i]
  param_E2['Ly'] = L[i]
  param_E2['Lz'] = L[i]
  param_E2['x0'] = np.array([0.2, 0.15, 0], dtype=np.float64)

  a = E2(param=param_E2, make_run_folder = False)

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

  np.save(root+'kl_E2_lmax{}.npy'.format(l_max), kl)
  np.save(root+'a_t_E2_lmax{}.npy'.format(l_max), a_t)