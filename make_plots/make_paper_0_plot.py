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

fig = plt.figure(figsize=(10, 6), dpi=800)

subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
axsLeft = subfigs[1].subplots(nrows=2, ncols=1)

red = '#CC79A7'
blue = '#56B4E9'
green = '#009E73'

l_max_list = [10, 20, 30, 40, 50]
kl_lmax = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../kl_runs/E1_E2_no_tilt_no_shift_old/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E1_cubic[-3]
axsLeft[1].plot(l_max_list, kl_lmax, color=red, label=r'$E_1$ - No tilt')

kl_lmax = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  kl_E2_cubic = np.load('../kl_runs/kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E2_cubic[-3]
axsLeft[1].plot(l_max_list, kl_lmax, color=blue, label=r'$E_1$ - $\beta=50^\circ$')


kl_lmax = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  kl = np.load('../kl_runs/E1_E2_no_tilt_no_shift_old/kl_E2_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl[-3]
axsLeft[1].plot(l_max_list, kl_lmax, color=green, label=r'$E_2$ - On axis')



l_max_list = [10, 30, 50]
linestyles=['solid', 'dashed', 'dotted']

L = np.linspace(0.6, 1.4, 9)

for i, l_max in enumerate(l_max_list):
  kl_no_tilt = np.load('../kl_runs/cubic_E1_no_tilt/kl_E1_lmax{}.npy'.format(l_max))
  axsLeft[0].plot(L, kl_no_tilt[2:], color=red, label=r'$E_1$ - No tilt' if i == 0 else None, linestyle = linestyles[i])
for i, l_max in enumerate(l_max_list):
  kl_tilt = np.load('../kl_runs/kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  axsLeft[0].plot(L, kl_tilt, color=blue, label=r'$E_1$ - $\beta=50^\circ$' if i == 0 else None, linestyle = linestyles[i])

for i, l_max in enumerate(l_max_list):
  kl_E2 = np.load('../kl_runs/cubic_E2_no_tilt_no_shift/kl_E2_lmax{}.npy'.format(l_max))
  axsLeft[0].plot(L, kl_E2[2:], color=green, label=r'$E_2$ - On axis' if i == 0 else None, linestyle = linestyles[i])

first_legend = axsLeft[0].legend(frameon=False, loc='upper right')
axsLeft[0].figure.add_artist(first_legend)

black_line3 = mlines.Line2D([], [], color='black', marker='s',linestyle="-",markersize=0)
black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle="--",markersize=0)
black_line1 = mlines.Line2D([], [], color='black', marker='s',linestyle=":",markersize=0)
axsLeft[0].legend([black_line1, black_line2, black_line3],[r'$\ell_{\mathrm{max}}=50$', r'$\ell_{\mathrm{max}}=30$', r'$\ell_{\mathrm{max}}=10$'], loc='lower left', frameon=False)


axsLeft[1].axhline(1, linestyle='dashdot', color = 'black')
axsLeft[0].set_title('KL Divergence')
axsLeft[1].text(11.5, 3.5, r'$L/L_{\mathrm{LSS}}=1.2$')
axsLeft[1].set_xlabel(r'$\ell_{\mathrm{max}}$')
axsLeft[1].set_ylabel(r'$D_{\mathrm{KL}}$')
axsLeft[1].set_xlim([10, 50])
axsLeft[1].set_ylim([0, 4])
axsLeft[1].yaxis.set_ticks_position('both')

axsLeft[0].axvline(1, linestyle='dashdot', color = 'black')
axsLeft[0].axhline(1, linestyle='dashdot', color = 'black')
axsLeft[0].set_yscale('log')
axsLeft[0].set_xlabel(r'$L / L_{\mathrm{LSS}}$')
axsLeft[0].set_ylabel(r'$D_{\mathrm{KL}}$')
axsLeft[0].set_ylim([0.2*1e-1, 8*1e2])
axsLeft[0].set_xlim([0.6, 1.4])
axsLeft[0].yaxis.set_ticks_position('both')


axsRight = subfigs[0].subplots(nrows=3, ncols=2)

def plot_ax(filename, index, l_min, l_max, lp_min, lp_max, title=None):
  C_TT_order = np.load(filename)
  C_TT_order = np.abs(C_TT_order)
  C_TT_order = np.where(C_TT_order > 1e-8, C_TT_order, 1e-8)
  ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
  ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])
  
  ax = axsRight[index]

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
      if index[1] == 0:
        ax.set_yticks(ellp_to_s_map[jump])
        ax.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
      else:
        ax.set_yticks(np.array([]))
        ax.set_yticklabels(np.array([]))
  else:
    if index[1] == 0:
      ax.set_yticks(ellp_to_s_map)
      ax.set_yticklabels(np.arange(lp_min, lp_max+1))
    else:
      ax.set_yticks(np.array([]))
      ax.set_yticklabels(np.array([]))

  if index[0] == 1:
    ax.set_xlim([0, (24+1)*(24+2) - (24+1) - l_min**2 - 1])
  else:
    ax.set_xlim([0, (l_max+1)*(l_max+2) - (l_max+1) - l_min**2 - 1])
  ax.set_ylim([0, (lp_max+1)*(lp_max+2) - (lp_max+1) - lp_min**2 - 1])

  if title != None:
    ax.set_title(title)

  if index[0] == 2:
    ax.set_xlabel(r'$\ell$')
  if index[1] == 0:
    ax.set_ylabel(r"$\ell'$   ", Rotation=0)

  axim.set_clim(1e-8, 1e0)
  ax.set_aspect('equal')
  return axim

l_max = 100
l_min = 99

lp_max = 101
lp_min = 100

filename = '../runs/protected/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_151_accuracy_99_percent/corr_matrix_l_99_100_lp_100_101.npy'
axim = plot_ax(filename, (0, 0), l_min, l_max, lp_min, lp_max, title=r'$E_1$ - No tilt')
filename = '../runs/protected/E2_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_x_0.00_y_0.00_z_0.00_l_max_151_accuracy_99_percent/corr_matrix_l_99_100_lp_100_101.npy'
axim = plot_ax(filename, (0, 1), l_min, l_max, lp_min, lp_max, title=r'$E_2$ - On axis')


l_max = 25
l_min = 2
lp_max = 151
lp_min = 150
filename = '../runs/protected/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_151_accuracy_99_percent/corr_matrix_l_2_25_lp_150_151.npy'
axim = plot_ax(filename, (1, 0), l_min, l_max, lp_min, lp_max)
filename = '../runs/protected/E2_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_x_0.00_y_0.00_z_0.00_l_max_151_accuracy_99_percent/corr_matrix_l_2_25_lp_150_151.npy'
axim = plot_ax(filename, (1, 1), l_min, l_max, lp_min, lp_max)

l_max = 20
l_min = 2
lp_max = 20
lp_min = 2
filename = '../runs/protected/E1_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_l_max_20_accuracy_99_percent/cor.npy'
axim = plot_ax(filename, (2, 0), l_min, l_max, lp_min, lp_max)
filename = '../runs/protected/E2_Lx_1.20_Ly_1.20_Lz_1.20_beta_90_alpha_90_x_0.00_y_0.00_z_0.00_l_max_20_accuracy_99_percent/cor.npy'
axim = plot_ax(filename, (2, 1), l_min, l_max, lp_min, lp_max)

cbar_ax = subfigs[0].add_axes([0.825, 0.0, 0.05, 1])
clb = subfigs[0].colorbar(axim, cax=cbar_ax)
clb.ax.set_title(r"$\frac{C_{\ell m \ell' m'}}{\sqrt{C_\ell C_{\ell'}}}$")

subfigs[0].subplots_adjust(left = 0, hspace=.25, wspace=-.3, bottom = 0, top = 1)
subfigs[1].subplots_adjust(hspace=.3, wspace=-0.1, bottom = 0, top = 1, left=0.17)

plt.savefig('tmp.pdf', bbox_inches='tight')