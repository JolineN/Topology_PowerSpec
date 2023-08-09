import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=15)
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16, labelsize=17)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=1600)

def plot_ax(filename, index, l_min, l_max, lp_min, lp_max, title=None):
  C_TT_order = np.load(filename)
  C_TT_order = np.abs(C_TT_order)
  C_TT_order = np.where(C_TT_order > 1e-8, C_TT_order, 1e-8)
  ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
  ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])
  
  ax = axs[index]

  axim = ax.imshow(C_TT_order.T, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
          
  if l_max-l_min > 15:
      jump = np.array([5, 10, 15, 20])-2
      ax.set_xticks(ell_to_s_map[jump])
      ax.set_xticklabels(np.arange(l_min, l_max+1)[jump])
  else:
      ax.set_xticks(ell_to_s_map)
      ax.set_xticklabels(np.arange(l_min, l_max+1))

  if lp_max-lp_min > 15:
      jump = np.array([5, 10, 15, 20])-2

      ax.set_yticks(ellp_to_s_map[jump])
      ax.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
  else:

    ax.set_yticks(ellp_to_s_map)
    ax.set_yticklabels(np.arange(lp_min, lp_max+1))


  ax.set_xlim([0, (l_max+1)*(l_max+2) - (l_max+1) - l_min**2 - 1])
  ax.set_ylim([0, (lp_max+1)*(lp_max+2) - (lp_max+1) - lp_min**2 - 1])


  ax.set_title(title)


  ax.set_xlabel(r'$\ell$')

  ax.set_ylabel(r"$\ell'$   ", Rotation=0)

  axim.set_clim(1e-8, 1e0)
  ax.set_aspect('equal')
  return axim

'''
l_max = 100
l_min = 99

lp_max = 101
lp_min = 100

filename = '../runs/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_121_accuracy_99_percent/corr_matrix_l_99_100_lp_100_101.npy'
axim = plot_ax(filename, (0, 0), l_min, l_max, lp_min, lp_max, title=r'$E_1$ - No tilt')
filename = '../runs/E3_L12_1.20_Lz_1.20_beta_90_alpha_90_x_0.20_y_0.00_z_0.00_l_max_121_accuracy_99_percent/corr_matrix_l_99_100_lp_100_101.npy'
axim = plot_ax(filename, (0, 1), l_min, l_max, lp_min, lp_max, title=r'$E_3$ - $x=0.2L_{\mathrm{LSS}}$')


l_max = 24
l_min = 2
lp_max = 121
lp_min = 120
filename = '../runs/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_121_accuracy_99_percent/corr_matrix_l_2_24_lp_120_121.npy'
axim = plot_ax(filename, (1, 0), l_min, l_max, lp_min, lp_max)
filename = '../runs/E3_L12_1.20_Lz_1.20_beta_90_alpha_90_x_0.20_y_0.00_z_0.00_l_max_121_accuracy_99_percent/corr_matrix_l_2_24_lp_120_121.npy'
axim = plot_ax(filename, (1, 1), l_min, l_max, lp_min, lp_max)
'''

l_max = 20
l_min = 2
lp_max = 20
lp_min = 2
filename = '../runs/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_121_accuracy_99_percent/corr_matrix_l_2_20_lp_2_20.npy'
axim = plot_ax(filename, 0, l_min, l_max, lp_min, lp_max)
filename = '../runs/E3_L12_1.20_Lz_1.20_beta_90_alpha_90_x_0.20_y_0.00_z_0.00_l_max_121_accuracy_99_percent/corr_matrix_l_2_20_lp_2_20.npy'
axim = plot_ax(filename, 1, l_min, l_max, lp_min, lp_max)


#clb = plt.colorbar(axim)

plt.savefig('tmp_revised_dec14.pdf', bbox_inches='tight')