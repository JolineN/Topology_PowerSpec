#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E5 import E5
from src.E6 import E6

import numpy as np
import time

import parameter_files.default_E1 as parameter_file

num = 7
# Set parameter file
L_list = np.linspace(0.7, 1.3, num)
L_circle=np.sqrt(1-0.7**2)

param = parameter_file.parameter

topology_list = np.array(['E3'])
for top in topology_list:
  kl = np.zeros(num)
  kl_Q_ass_P = np.zeros(num)
  a_t = np.zeros(num)
  for i, L in enumerate(L_list):
    param['topology'] = top
    param['l_max'] = 30
    param['Lx'] = 1.4
    param['Ly'] = 1.4
    param['Lz'] = L*L_circle
    param['beta'] = 90
    param['alpha'] = 90
    param['gamma'] = 0
    param['x0'] = np.array([0.35, 0.35, 0], dtype=np.float64)

    if param['topology'] == 'E1':
      a = E1(param=param, make_run_folder=True)
    elif param['topology'] == 'E2':
      a = E2(param=param, make_run_folder=True)
    elif param['topology'] == 'E3':
      a = E3(param=param, make_run_folder=True)
    elif param['topology'] == 'E4':
      a = E4(param=param, make_run_folder=True)
    elif param['topology'] == 'E5':
      a = E5(param=param, make_run_folder=True)
    elif param['topology'] == 'E6':
      a = E6(param=param, make_run_folder=True)

    start = time.time()
    cur_kl, cur_kl_Q_ass_P, cur_a_t = a.calculate_exact_kl_divergence()
    end = time.time()
    print('Elapsed time parallel:', end-start)

    kl[i] = cur_kl
    kl_Q_ass_P[i] = cur_kl_Q_ass_P
    a_t[i] = cur_a_t

    np.save('kl_{}_x0_035_y0_035.npy'.format(top), kl)
    np.save('kl_Q_ass_P{}_x0_035_y0_035.npy'.format(top), kl_Q_ass_P)
    np.save('at_{}_x0_035_y0_035.npy'.format(top), a_t)