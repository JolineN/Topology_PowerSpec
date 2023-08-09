#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1
from src.E1 import E1
from src.tools import *
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from anomaly.anomaly_tools import *

# Set parameter file
import parameter_files.default_E1 as parameter_file
param = parameter_file.parameter

num_it = 7000

param['topology'] = 'E1'
param['Lx'] = 1.1
param['Ly'] = 1.1
param['Lz'] = 1.1
param['beta'] = 75.0
param['alpha'] = 90.0
param['gamma'] = 0.0
param['l_max'] = 8
param['x0'] = np.array([0, 0, 0], dtype=np.float64)
param['number_of_a_lm_realizations'] = num_it

if param['topology'] == 'E1':
  a = E1(param=param, make_run_folder=True)
else:
  exit()

# Create realizations
a_lm, c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False)
a.calculate_c_lmlpmp(
  only_diag=True
)

# Plot the diagonal power spectrum and the realizations
# Good to see if there are any obvious bugs
a.plot_c_l_and_realizations(c_l_a=c_l_a)
_, _ = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  save_cov = True,
  plot_param={
    'l_ranges': np.array([[2, 8]]),
    'lp_ranges': np.array([[2, 8]]),
  }
)

os.environ['OMP_NUM_THREADS'] = '128'

S_one_half, C_theta = get_S_one_half(a_lm)
np.save('anomaly/E1/S_one_half_E1_L_11_beta_75_it_{}.npy'.format(num_it), S_one_half)
np.save('anomaly/E1/C_theta_E1_L_11_beta_75_it_{}.npy'.format(num_it), C_theta)

t_3, S_2_coord, S_3_coord = test_octopole_planarity(a_lm)
print(t_3)
np.save('anomaly/E1/t_3_E1_L_11_beta_75_it_{}.npy'.format(num_it), t_3)
np.save('anomaly/E1/S_2_coords_E1_L_11_beta_75_it_{}.npy'.format(num_it), S_2_coord)
np.save('anomaly/E1/S_3_coords_E1_L_11_beta_75_it_{}.npy'.format(num_it), S_3_coord)

