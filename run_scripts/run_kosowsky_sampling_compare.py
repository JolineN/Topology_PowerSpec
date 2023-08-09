#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("..")
from src.E1 import E1
from src.E2 import E2
import numpy as np
import time
import matplotlib.pyplot as plt

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E2 as parameter_file


N_s = 100
num_times = 20
L = 1.2

get_exact = False


sampled_kosowsky = np.zeros((5, 2))
lmax_list = [10, 20, 30, 40, 50]

for i, lmax in enumerate(lmax_list):
  param = parameter_file.parameter
  param['Lx'] = L
  param['Ly'] = L
  param['Lz'] = L
  param['l_max'] = lmax

  if param['topology'] == 'E1':
    a = E1(param=param)
  elif param['topology'] == 'E2':
    a = E2(param=param)
  else:
    exit()
      
  time_s = time.time()

  if get_exact:
    cur_kl, cur_a_t = a.calculate_exact_kl_divergence()
    print('Exact Kosowsky:', cur_a_t)
  print('Time taken:', time.time()-time_s)
  kosowsky_list = a.sampled_kosowsky_statistics(N_s=N_s, num_times=num_times)

  mean = np.mean(kosowsky_list)
  std = np.std(kosowsky_list)
  print(mean, std)

  sampled_kosowsky[i, 0] = mean
  sampled_kosowsky[i, 1] = std
  np.save('tmp/kosowsky_monte_carlo_{}_L{}_ns_{}_ntimes_{}.pdf'.format(param['topology'], L, N_s, num_times), sampled_kosowsky)

  plt.figure()
  plt.hist(kosowsky_list)
  if get_exact: plt.axvline(cur_a_t, color='blue', linestyle='-')
  plt.axvline(mean, color='k', linestyle='-')
  plt.axvline(mean-std, color='k', linestyle='--')
  plt.axvline(mean+std, color='k', linestyle='--')
  plt.title(r'$\ell_{{max}}={{{0}}}, N_s={{{1}}}$ Mean:{2:.4f} Std:{3:.4f}'.format(lmax, N_s, mean, std))
  plt.savefig('tmp/tmp_kosowsky_{}_L{}_lmax{}_ns_{}_ntimes_{}.pdf'.format(param['topology'], L, lmax, N_s, num_times))
