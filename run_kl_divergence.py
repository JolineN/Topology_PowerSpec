#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
from src.E2 import E2

import numpy as np

import parameter_files.default as parameter_file_E1
import parameter_files.default_E2 as parameter_file_E2

# Set parameter file
num = 8
L = np.linspace(0.6, 1.3, num)
root = 'kl_run/'

l_max_list = np.array([10, 20, 30, 40])

param_E1 = parameter_file_E1.parameter
param_E2 = parameter_file_E2.parameter
print(L)

for l_max in l_max_list:
  kl = np.zeros(num)
  a_t = np.zeros(num)

  for i in range(L.size):
    param_E1['l_max'] = l_max
    param_E1['Lx'] = L[i]
    param_E1['Ly'] = L[i]
    param_E1['Lz'] = L[i]

    a = E1(param=param_E1)

    cur_kl, cur_a_t = a.calculate_kl_divergence()
    kl[i] = cur_kl
    a_t[i] = cur_a_t
    print('E1 Currenct kl!')
    print(cur_kl, cur_a_t)
    print(kl)
    print('Arthur')
    print(a_t)

    np.save(root+'kl_E1_lmax{}.npy'.format(l_max), kl)
    np.save(root+'a_t_E1_lmax{}.npy'.format(l_max), a_t)


  kl = np.zeros(num)
  a_t = np.zeros(num)

  for i in range(L.size):
    param_E2['l_max'] = l_max
    param_E2['Lx'] = L[i]
    param_E2['Ly'] = L[i]
    param_E2['Lz'] = L[i]

    a = E2(param=param_E2)

    cur_kl, cur_a_t = a.calculate_kl_divergence()
    kl[i] = cur_kl
    a_t[i] = cur_a_t
    print('E2 Currenct kl!')
    print(cur_kl, cur_a_t)
    print(kl)
    print('Arthur')
    print(a_t)

    np.save(root+'kl_E2_lmax{}.npy'.format(l_max), kl)
    np.save(root+'a_t_E2_lmax{}.npy'.format(l_max), a_t)