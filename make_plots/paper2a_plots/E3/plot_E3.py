import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=14)
#plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
num = 7
# Set parameter file
L_list = np.linspace(0.7, 1.3, num)
L_circle=np.sqrt(1-0.7**2)

E3_no_shift = np.load('kl_E3_x0_0_y0_0.npy')
E3_shift = np.load('kl_E3_x0_035_y0_035.npy')

fig = plt.figure(figsize=(5, 5.6), dpi=300)

plt.subplots_adjust(wspace= 0.1, hspace=0.2)

def plot_ax(cov, sub, i, l_min, l_max, lp_min, lp_max, title):
  C_TT_order = cov
  C_TT_order = np.abs(C_TT_order)
  C_TT_order = np.where(C_TT_order > 1e-6, C_TT_order, 1e-6)
  ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
  ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])

  axim = sub.imshow(C_TT_order.T, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
          

  jump = np.array([2, 5, 8, 10])-2
  sub.set_xticks(ell_to_s_map[jump])
  sub.set_xticklabels(np.arange(l_min, l_max+1)[jump])


  if i == 0:
    jump = np.array([2, 5, 8, 10])-2
    sub.set_yticks(ellp_to_s_map[jump])
    sub.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
    sub.set_ylabel(r"$\ell'$", Rotation=0)
  else:
    sub.set_yticks([])
    sub.set_yticklabels([])

  sub.set_xlim([0, (l_max+1)*(l_max+2) - (l_max+1) - l_min**2 - 1])
  sub.set_ylim([0, (lp_max+1)*(lp_max+2) - (lp_max+1) - lp_min**2 - 1])

  if title != None:
    sub.set_title(title, fontsize=12)


  sub.set_xlabel(r'$\ell$')
  

  axim.set_clim(1e-6, 1e0)
  #ax.set_aspect('equal')
  return axim


sub1 = fig.add_subplot(2,2,1) # two rows, two columns, fist cell
cov = np.load('E3_no_shift_corr_matrix_l_2_10_lp_2_10.npy')
axim = plot_ax(cov, sub1, 0, 2, 10, 2, 10, r'$\boldsymbol{x}_0=(0, 0, 0)L_{\mathrm{LSS}}$')


# Create second axes, the top-left plot with orange plot
sub2 = fig.add_subplot(2,2,2) # two rows, two columns, second cell
cov = np.load('E3_with_shift_corr_matrix_l_2_10_lp_2_10.npy')
axim = plot_ax(cov, sub2, 1, 2, 10, 2, 10, r'$\boldsymbol{x}_0=(0.35, 0.35, 0)L_{\mathrm{LSS}}$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.05, 0.04, 0.9])
fig.colorbar(axim, cax=cbar_ax)

# Create third axes, a combination of third and fourth cell
sub3 = fig.add_subplot(2, 2, (3,4)) # two rows, two colums, combined third and fourth cell

sub3.plot(L_list, E3_no_shift, label=r'$\boldsymbol{x}_0=(0, 0, 0)L_{\mathrm{LSS}}$', color='red')
sub3.plot(L_list, E3_shift, label=r'$\boldsymbol{x}_0=(0.35, 0.35, 0)L_{\mathrm{LSS}}$', color='blue')
sub3.set_ylabel(r'$D_{\mathrm{KL}}(p || q)$')
sub3.yaxis.set_label_coords(-0.128, 0.44)
sub3.set_xlabel(r'$L_B / L_{\mathrm{circle}}$')
sub3.set_xlim([0.7, 1.3])
sub3.set_ylim([0.1, 4*1e2])
sub3.axhline(1, color='black', linestyle='--')
sub3.axvline(1, ymin=0, ymax=0.7, color='black', linestyle='--')
sub3.set_yscale('log')
sub3.legend(frameon=False)
fig.suptitle(r'$E_3$')
plt.savefig('E3.pdf')