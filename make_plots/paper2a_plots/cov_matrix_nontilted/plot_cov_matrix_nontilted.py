import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=14)
#plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)

l_min = 2
l_max = 10
ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
ell_to_s_map = np.delete(ell_to_s_map, [1, 2, 4, 5, 7])
ticks = np.delete(np.arange(l_min, l_max+1), [1, 2, 4, 5, 7])

nrow = 3
ncol = 2
fig = plt.figure(figsize=(6/1.3, 8.2/1.3), dpi=500) 
gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1],
         wspace=0.02, hspace=0.0, top=0.98, bottom=0.06, left=0.13, right=0.8) 

ax = plt.subplot(gs[0, 0])
tilde_C_abs = np.abs(np.load('E1_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_1$')
ax.set_xticklabels([])
ax.set_xticks(ell_to_s_map)
ax.set_yticks(ell_to_s_map)
ax.set_yticklabels(ticks)
ax.set_ylabel(r"$\ell'$")

ax = plt.subplot(gs[0, 1])
tilde_C_abs = np.abs(np.load('E2_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_2$')
ax.set_xticks(ell_to_s_map)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])

ax = plt.subplot(gs[1, 0])
tilde_C_abs = np.abs(np.load('E3_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_3$')
ax.set_xticklabels([])
ax.set_xticks(ell_to_s_map)
ax.set_yticks(ell_to_s_map)
ax.set_yticklabels(ticks)
ax.set_ylabel(r"$\ell'$")

ax = plt.subplot(gs[1, 1])
tilde_C_abs = np.abs(np.load('E4_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_4$')
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticks(ell_to_s_map)
ax.set_xticklabels([])

ax = plt.subplot(gs[2, 0])
tilde_C_abs = np.abs(np.load('E5_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
im = ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_5$')
ax.set_yticks(ell_to_s_map)
ax.set_yticklabels(ticks)
ax.set_ylabel(r"$\ell'$")
ax.set_xticks(ell_to_s_map)
ax.set_xticklabels(ticks)
ax.set_xlabel(r"$\ell$")

ax = plt.subplot(gs[2, 1])
tilde_C_abs = np.abs(np.load('E6_corr_matrix_l_2_10_lp_2_10.npy'))
tilde_C_abs = np.where(tilde_C_abs > 1e-6, tilde_C_abs, 1e-6)
ax.imshow(tilde_C_abs, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
ax.set_title(r'$E_6$')
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticks(ell_to_s_map)
ax.set_xticklabels(ticks)
ax.set_xlabel(r"$\ell$")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.05, 0.05, 0.9])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('cov_matrix_E1_E6_default_L_L_circle.pdf')