#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("..")
from src.E1 import E1
from src.E2 import E2

import numpy as np
import time

import parameter_files.default_E1 as parameter_file_E1

# Set parameter file
num = 9
L = np.linspace(0.6, 1.4, num)
root = 'kl_run_E1_tilt/'

l_max_list = np.array([10, 20, 30, 40, 50])

param_E1 = parameter_file_E1.parameter
print(L)

for l_max in l_max_list:
  kl = np.zeros(num)
  a_t = np.zeros(num)

  for i in range(L.size):
    param_E1['l_max'] = l_max
    param_E1['Lx'] = L[i]
    param_E1['Ly'] = L[i]
    param_E1['Lz'] = L[i]
    param_E1['beta'] = 50.0

    a = E1(param=param_E1)

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

    np.save(root+'kl_E1_lmax{}.npy'.format(l_max), kl)
    np.save(root+'a_t_E1_lmax{}.npy'.format(l_max), a_t)