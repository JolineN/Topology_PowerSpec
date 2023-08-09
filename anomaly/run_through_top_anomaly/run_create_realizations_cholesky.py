#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1
import sys
sys.path.append("../..")
sys.path.append("..")
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E5 import E5
from src.E6 import E6
from src.E7 import E7
import numpy as np
from anomaly_tools import *
import time

# Set parameter file
#import parameter_files.default as parameter_file
import parameter_files.default_E1 as parameter_file

param = parameter_file.parameter
param['topology'] = 'E2'
param['Lx'] = 1.0
param['Ly'] = 1.0
param['Lz'] = 1.0
param['beta'] = 80.0
param['alpha'] = 90.0
param['gamma'] = 0.0
param['l_max'] = 3
param['x0'] = np.array([0, 0, 0], dtype=np.float64)
param['number_of_a_lm_realizations'] = 1

def get_top(param):
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
  return a

topology_list = np.array(['E1', 'E2', 'E3', 'E4', 'E5'])

t_3_lcdm = np.load('../LCDM/t_3_50000_LCDM.npy')
t3_lcdm_PTE = np.sum(np.where(t_3_lcdm > 0.94, 1, 0)) / len(t_3_lcdm)
print('LCDM t_3 > 0.94:', t3_lcdm_PTE * 100)

S_2_lcdm = np.load('../LCDM/S_2_coords_100000_LCDM.npy')
S_3_lcdm = np.load('../LCDM/S_3_coords_100000_LCDM.npy')
n_2_lcdm = np.array([cos(S_2_lcdm[:, 1])*sin(S_2_lcdm[:, 0]), sin(S_2_lcdm[:, 1])*sin(S_2_lcdm[:, 0]), cos(S_2_lcdm[:, 0])])
n_3_lcdm = np.array([cos(S_3_lcdm[:, 1])*sin(S_3_lcdm[:, 0]), sin(S_3_lcdm[:, 1])*sin(S_3_lcdm[:, 0]), cos(S_3_lcdm[:, 0])])
C_ell_2_lcdm = np.load('../LCDM/C_ell_2_100000_LCDM.npy')
c_ell_2_lcdm_PTE = np.sum(np.where(C_ell_2_lcdm < 226, 1, 0)) / len(C_ell_2_lcdm)
print('LCDM C_2 < 226:', c_ell_2_lcdm_PTE * 100)

dot_lcdm = np.zeros(n_2_lcdm[0, :].size)
for i in range(len(dot_lcdm)):
  dot_lcdm[i] = np.dot(n_2_lcdm[:, i], n_3_lcdm[:, i])
dot_lcdm_PTE = np.sum(np.where(dot_lcdm > 0.98, 1, 0)) / len(dot_lcdm)
print('LCDM n_2 n_3 > 0.98:', dot_lcdm_PTE * 100)

anomaly_list = np.zeros((3, len(topology_list)))
for k, top in enumerate(topology_list):
  param['topology'] = top
  a = get_top(param)
  a.calculate_c_lmlpmp(
    only_diag=True
  )

  # Plot the diagonal power spectrum and the realizations
  # Good to see if there are any obvious bugs
  a.plot_c_l_and_realizations(c_l_a=None)

  _, _ = a.calculate_c_lmlpmp(
    only_diag=False,
    normalize=True,
    save_cov = True,
    plot_param={
      'l_ranges': np.array([[2, 3]]),
      'lp_ranges': np.array([[2, 3]]),
    }
  )
  a_lm_list, c_l_list = a.make_realization_c_lmlpmp_cholesky(50000)

  #S_one_half, C_theta = get_S_one_half(a_lm)
  t_3, S_2_coords, S_3_coords = run_octopole_quadrupole_alignment_octopole_planarity(a_lm_list, N_side=16)

  n_2 = np.array([cos(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), sin(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), cos(S_2_coords[:, 0])])
  n_3 = np.array([cos(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), sin(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), cos(S_3_coords[:, 0])])

  dot = np.zeros(n_2[0, :].size)
  for i in range(len(dot)):
    dot[i] = np.dot(n_2[:, i], n_3[:, i])

  print('------')
  print(top)
  print('------')
  
  #PTE t3 planarity
  t3_PTE = np.sum(np.where(t_3 > 0.94, 1, 0)) / len(t_3)
  anomaly_list[0, k] = t3_PTE
  print('PTE t_3 > 0.94:', t3_PTE*100, '% LCDM PTE:', t3_lcdm_PTE*100, '%')
  plt.figure()
  plt.title('t_3')
  plt.hist(t_3, label=top, density=True, bins=40)
  plt.hist(t_3_lcdm, label='LCDM', alpha=0.5, density=True, bins=40)
  plt.axvline(0.94, color='black', linestyle='--')
  plt.legend()
  plt.savefig('img/t_3_{}_L_1_beta_80.pdf'.format(top))


  #PTE quadrupole power
  c_ell_2_PTE = np.sum(np.where(c_l_list[:, 2] < 226, 1, 0)) / len(c_l_list[:, 2])
  anomaly_list[1, k] = c_ell_2_PTE
  print('PTE C_2 < 226 muK^2:', c_ell_2_PTE*100, '% LCDM PTE:', c_ell_2_lcdm_PTE*100, '%')
  plt.figure()
  plt.title('C_2')
  plt.hist(c_l_list[:, 2], label=top, density=True, bins=40)
  plt.hist(C_ell_2_lcdm, label='LCDM', alpha=0.5, density=True, bins=40)
  plt.axvline(226, color='black', linestyle='--')
  plt.legend()
  plt.savefig('img/c_2_{}_L_1_beta_80.pdf'.format(top))

  #PTE alignment
  alignment_PTE = np.sum(np.where(dot > 0.98, 1, 0)) / len(dot)
  anomaly_list[2, k] = alignment_PTE
  print('PTE |n_2 * n_3| > 0.98:', alignment_PTE*100, '% LCDM PTE:', dot_lcdm_PTE*100, '%')
  plt.figure()
  plt.title('|n_2 * n_3|')
  plt.hist(dot, label=top, density=True, bins=40)
  plt.hist(dot_lcdm, label='LCDM', alpha=0.5, density=True, bins=40)
  plt.axvline(0.98, color='black', linestyle='--')
  plt.legend()
  plt.savefig('img/align_{}_L_1_beta_80.pdf'.format(top))
  
  np.save('anomaly_top_50000_L_1_beta_80.npy', anomaly_list)