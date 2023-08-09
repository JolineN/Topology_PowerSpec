#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E1 import E1
import numpy as np
import time

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E1 as parameter_file



L_list = np.linspace(0.95, 1.05, 11)
print(L_list)
kl = np.zeros((len(L_list)))

for i, L in enumerate(L_list):
  param = parameter_file.parameter
  param['Lx'] = L
  param['Ly'] = L
  param['Lz'] = L
  param['l_max'] = 30

  if param['topology'] == 'E1':
    a = E1(param=param)

  time_s = time.time()
  kl1, _, _ = a.calculate_exact_kl_divergence()
  print('Exact KL:', kl1)
  kl[i] = kl1
  np.save('real_kl.npy', kl)
  print('Time taken for exact:', time.time()-time_s)
