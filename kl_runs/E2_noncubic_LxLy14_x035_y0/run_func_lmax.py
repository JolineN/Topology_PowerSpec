#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E2 import E2

import numpy as np
import time

import parameter_files.default_E2 as parameter_file

# Set parameter file
l_max_list = np.array([10, 20, 30, 40, 50])

param = parameter_file.parameter

kl = np.zeros(5)
kl_Q_ass_P = np.zeros(5)
a_t = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  param['l_max'] = l_max
  param['Lx'] = 1.4
  param['Ly'] = 1.4
  param['Lz'] = 0.714*0.9
  param['beta'] = 90
  param['alpha'] = 90
  param['gamma'] = 0
  param['x0'] = np.array([0.35, 0, 0], dtype=np.float64)

  a = E2(param=param, make_run_folder = False)

  start = time.time()
  cur_kl, cur_kl_Q_ass_P, cur_a_t = a.calculate_exact_kl_divergence()
  end = time.time()
  print('Elapsed time parallel:', end-start)

  kl[i] = cur_kl
  kl_Q_ass_P[i] = cur_kl_Q_ass_P
  a_t[i] = cur_a_t
  print('E2 Currenct kl!')
  print(kl)
  print('Arthur')
  print(a_t)

  np.save('kl_Lz06426_lmax_10_50.npy', kl)
  np.save('kl_Q_ass_P_Lz06426_lmax_10_50.npy', kl_Q_ass_P)
  np.save('a_t_Lz06426_lmax_10_50.npy', a_t)