#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
import numpy as np

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E1 as parameter_file

param = parameter_file.parameter
param['topology'] = 'E1'
param['Lx'] = 1.2
param['Ly'] = 1.2
param['Lz'] = 1.2
param['beta'] = 90.0
param['alpha'] = 90.0
param['l_max'] = 20
param['x0'] = np.array([0.2, 0, 0], dtype=np.float64)

if param['topology'] == 'E1':
  a = E1(param=param, make_run_folder=True)
elif param['topology'] == 'E2':
  a = E2(param=param, make_run_folder=True)
elif param['topology'] == 'E3':
  a = E3(param=param, make_run_folder=True)
else:
  exit()
'''
# Create 2 realizations
c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False, it=2)

# Calculate the diagonal covariance matrix
a.calculate_c_lmlpmp(
  only_diag=True
)

# Plot the diagonal power spectrum and the realizations
# Good to see if there are any obvious bugs
a.plot_c_l_and_realizations(c_l_a)

'''
# Plot the covariance matrix for certain intervals
_, cor = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 10]]),
    'lp_ranges': np.array([[2, 10]])
  },
  save_cov = True)
kl1, kl2, at = a.calculate_exact_kl_divergence()
print(kl1, kl2, at)
#np.save('cor.npy', cor)
'''
a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 25], [100, 103],  [200, 200], [100, 101]]),
    'lp_ranges': np.array([[200, 201], [200, 201], [200, 200], [100, 101]])
  })

tic = time.perf_counter()
cur_kl = a.calculate_kl_divergence()
toc = time.perf_counter()
print(f"Calculated KL divergence in {toc - tic:0.4f} seconds")
print('KL divergence:', cur_kl)
'''