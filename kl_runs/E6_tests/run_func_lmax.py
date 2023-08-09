#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E6 import E6

import numpy as np
import time

import parameter_files.default_E1 as parameter_file

# Set parameter file
L_list = np.linspace(1.4, 2.5, 12)
length = len(L_list)
print(L_list)

param = parameter_file.parameter

kl = np.zeros(length)
kl_Q_ass_P = np.zeros(length)
a_t = np.zeros(length)
for i, L in enumerate(L_list):
  param['topology'] = 'E6'
  param['l_max'] = 20
  param['Lx'] = L
  param['Ly'] = L
  param['Lz'] = L
  param['beta'] = 90
  param['alpha'] = 90
  param['gamma'] = 0
  param['x0'] = np.array([0, 0, 0], dtype=np.float64)

  a = E6(param=param, make_run_folder = False)

  start = time.time()
  cur_kl, cur_kl_Q_ass_P, cur_a_t = a.calculate_exact_kl_divergence()
  end = time.time()
  print('Elapsed time parallel:', end-start)

  kl[i] = cur_kl
  kl_Q_ass_P[i] = cur_kl_Q_ass_P
  a_t[i] = cur_a_t
  print('E6 Currenct kl!')
  print(kl)
  print('Arthur')
  print(a_t)

  np.save('kl_L_14_25_lmax20.npy', kl)
  np.save('kl_Q_ass_P_L_14_25_lmax20.npy', kl_Q_ass_P)
  np.save('a_t_L_14_25_lmax20.npy', a_t)