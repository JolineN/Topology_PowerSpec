#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E1 import E1
from src.E2 import E2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E2 as parameter_file

param = parameter_file.parameter
param['Lx'] = 0.7
param['Ly'] = 0.7
param['Lz'] = 0.7
param['beta'] = 90.0
param['alpha'] = 90.0
param['l_max'] = 10
param['x0'] = np.array([0, 0, 0], dtype=np.float64)

if param['topology'] == 'E1':
  a = E1(param=param)
elif param['topology'] == 'E2':
  a = E2(param=param, make_run_folder=True)
else:
  exit()

# Plot the covariance matrix for certain intervals
_, no_shift = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 10]]),
    'lp_ranges': np.array([[2, 10]])
  },
  save_cov = True)

param['x0'] = np.array([0.4, 0.7, 0], dtype=np.float64)
if param['topology'] == 'E1':
  a = E1(param=param)
elif param['topology'] == 'E2':
  a = E2(param=param, make_run_folder=True)
else:
  exit()

# Plot the covariance matrix for certain intervals
_, with_shift = a.calculate_c_lmlpmp(
  only_diag=False,
  normalize=True,
  plot_param={
    'l_ranges': np.array([[2, 10]]),
    'lp_ranges': np.array([[2, 10]])
  },
  save_cov = True)

diff = np.abs(with_shift).T - np.abs(no_shift).T
pos = np.where(diff >= 1e-10, diff, 1e-10)
neg = np.where(diff < -1e-10, -diff, 1e-10)

l_min = 2
l_max = 10
ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])

plt.figure()
plt.imshow(pos, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
plt.colorbar()
plt.clim(1e-8, 1e0)
plt.title('When shift > no shift')
plt.savefig('tmp_pos.pdf')

plt.figure()
plt.imshow(neg, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
plt.colorbar()
plt.clim(1e-8, 1e0)
plt.title('When shift < no shift')
plt.savefig('tmp_neg.pdf')

plt.figure()
plt.imshow(diff, cmap='bwr', origin='lower', interpolation = 'nearest')
plt.colorbar()
plt.clim(-0.1, 0.1)
plt.title('Diff [shift - no shift]')
plt.savefig('tmp_diff.pdf')