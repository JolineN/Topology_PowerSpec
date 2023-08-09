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
num = 9
L = np.linspace(0.6, 1.4, num)

l_max_list = np.array([10, 20, 30])

param = parameter_file_E1.parameter
print(L)

for l_max in l_max_list:
  kl_P_assuming_Q = np.zeros(num)
  kl_Q_assuming_P = np.zeros(num)
  a_t = np.zeros(num)

  for i in range(L.size):
    param['l_max'] = l_max
    param['Lx'] = L[i]
    param['Ly'] = L[i]
    param['Lz'] = L[i]
    param['beta'] = 90
    param['alpha'] = 90
    param['gamma'] = 0

    a = E1(param=param, make_run_folder = False)

    start = time.time()
    cur_kl_P_assuming_Q, cur_kl_Q_assuming_P, cur_a_t = a.calculate_exact_kl_divergence(parallel_cov = True)
    end = time.time()
    print('Elapsed time parallel:', end-start)

    kl_P_assuming_Q[i] = cur_kl_P_assuming_Q
    kl_Q_assuming_P[i] = cur_kl_Q_assuming_P
    a_t[i] = cur_a_t
    print('Currenct kl!')
    print(kl_P_assuming_Q)
    print(kl_Q_assuming_P)
    print('Arthur')
    print(a_t)

    np.save('kl_P_assuming_Q_lmax{}.npy'.format(l_max), kl_P_assuming_Q)
    np.save('kl_Q_assuming_P_lmax{}.npy'.format(l_max), kl_Q_assuming_P)
    np.save('a_t_lmax{}.npy'.format(l_max), a_t)