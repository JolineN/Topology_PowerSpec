#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E5 import E5
from src.E6 import E6
from src.E7 import E7
import numpy as np

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E2 as parameter_file
param = parameter_file.parameter
'''
L_circle = np.sqrt(1-0.7**2)
param = parameter_file.parameter
param['topology'] = 'E1'
param['Lx'] = 1
param['Ly'] = 1
param['Lz'] = 1
param['l_max'] = 30
param['beta'] = 90
param['alpha'] = 90
param['gamma'] = 0
param['x0'] = np.array([0, 0, 0], dtype=np.float64)
param['number_of_a_lm_realizations'] = 1
'''

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
elif param['topology'] == 'E7':
  a = E7(param=param, make_run_folder=True)
else:
  exit()

# Create 2 realizations
c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False)
#print(c_l_a.shape)
# Calculate the diagonal covariance matrix
a.calculate_c_lmlpmp(
  only_diag=True
)

# Plot the diagonal power spectrum and the realizations
# Good to see if there are any obvious bugs
a.plot_c_l_and_realizations(c_l_a=None)

_, norm_c = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  save_cov = True,
  plot_param={
    'l_ranges': np.array([[2, 10]]),
    'lp_ranges': np.array([[2, 10]]),
  }
)
cur_kl, _, _ = a.calculate_exact_kl_divergence()
print('KL:', cur_kl)