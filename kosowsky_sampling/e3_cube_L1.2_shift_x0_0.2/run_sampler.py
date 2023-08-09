#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append("../..")
from src.E3 import E3
import numpy as np
import time

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E3 as parameter_file


N_s = 100
num_times = 20
L = 1.2

i = 6
lmax_list = [10, 20, 30, 40, 50, 100, 130]
lmax = 130
sampled_kosowsky = np.load('kosowsky_monte_carlo_E3_L1.2_ns_100_ntimes_20.npy')
print(sampled_kosowsky)

param = parameter_file.parameter
param['Lx'] = L
param['Ly'] = L
param['Lz'] = L
param['l_max'] = lmax
param['x0'] = np.array([0.2, 0, 0], dtype=np.float64)

if param['topology'] == 'E3':
  a = E3(param=param)

time_s = time.time() 
kosowsky_list = a.sampled_kosowsky_statistics(N_s=N_s, num_times=num_times)
print('Time taken for mc:', time.time()-time_s)

mean = np.mean(kosowsky_list)
std = np.std(kosowsky_list)

sampled_kosowsky[i, 0] = mean
sampled_kosowsky[i, 1] = std

print(sampled_kosowsky)
np.save('kosowsky_monte_carlo_{}_L{}_ns_{}_ntimes_{}.npy'.format(param['topology'], L, N_s, num_times), sampled_kosowsky)

'''
sampled_kosowsky = np.zeros((len(lmax_list), 2))
real_kosowsky = np.zeros(5)

for i, lmax in enumerate(lmax_list):
  param = parameter_file.parameter
  param['Lx'] = L
  param['Ly'] = L
  param['Lz'] = L
  param['l_max'] = lmax
  param['x0'] = np.array([0.2, 0, 0], dtype=np.float64)

  if param['topology'] == 'E3':
    a = E3(param=param)
      
  if lmax <= 50:
    time_s = time.time()
    _, _, cur_a_t = a.calculate_exact_kl_divergence()
    print('Exact Kosowsky:', cur_a_t)
    real_kosowsky[i] = cur_a_t
    print(real_kosowsky)
    np.save('real_kosowsky.npy', real_kosowsky)
    print('Time taken for exact:', time.time()-time_s)

  time_s = time.time() 
  kosowsky_list = a.sampled_kosowsky_statistics(N_s=N_s, num_times=num_times)
  print('Time taken for mc:', time.time()-time_s)

  mean = np.mean(kosowsky_list)
  std = np.std(kosowsky_list)

  sampled_kosowsky[i, 0] = mean
  sampled_kosowsky[i, 1] = std

  print(real_kosowsky)
  print(sampled_kosowsky)
  np.save('kosowsky_monte_carlo_{}_L{}_ns_{}_ntimes_{}.pdf'.format(param['topology'], L, N_s, num_times), sampled_kosowsky)
'''
