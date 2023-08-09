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
num = 12
L = np.linspace(0.3, 1.4, num)
root = '../kl_runs/E2_noncubic_LxLy14_x035_y035/'

l_max_list = np.array([30])

param_E2 = parameter_file_E2.parameter
print(L)

for l_max in l_max_list:
  kl = np.zeros(num)
  a_t = np.zeros(num)

  for i in range(L.size):
    param_E2['l_max'] = l_max
    param_E2['Lx'] = 1.4
    param_E2['Ly'] = 1.4
    param_E2['Lz'] = L[i]
    param_E2['beta'] = 90
    param_E2['alpha'] = 90
    param_E2['gamma'] = 0
    param_E2['x0'] = np.array([0.35, 0.35, 0], dtype=np.float64)

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