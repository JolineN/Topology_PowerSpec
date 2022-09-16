#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
from src.E2 import E2
import numpy as np

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E1 as parameter_file


param = parameter_file.parameter

if param['topology'] == 'E1':
  a = E1(param=param)
elif param['topology'] == 'E2':
  a = E2(param=param)
else:
  exit()

# Create 2 realizations
#c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False, it=2)

# Calculate the diagonal covariance matrix
#a.calculate_c_lmlpmp(
#  only_diag=True
#)

# Plot the diagonal power spectrum and the realizations
# Good to see if there are any obvious bugs
#a.plot_c_l_and_realizations(c_l_a)


# Plot the covariance matrix for certain intervals
a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 7], [10, 13],  [18, 19], [2, 7]]),
    'lp_ranges': np.array([[18, 19], [18, 19], [18, 19], [2, 7]])
  })
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