import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=15)
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16, labelsize=17)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)

l_max_list = [30]

L = np.linspace(0.3, 1.0, 8)


plt.figure()
for i, l_max in enumerate(l_max_list):
  kl = np.load('kl_E2_lmax{}.npy'.format(l_max))
  plt.plot(L, kl, color='red', label=r'E2 - $x_0=(0.5, 0, 0)L_{\mathrm{LSS}}$' if i == 0 else None, alpha = 1-0.3*i)
for i, l_max in enumerate(l_max_list):
  kl = np.load('../E2_noncubic_noshift_LxLy1/kl_E2_lmax{}.npy'.format(l_max))
  plt.plot(L, kl, color='blue', label=r'E2 - $x_0=(0, 0, 0)L_{\mathrm{LSS}}$' if i == 0 else None, alpha = 1-0.3*i)


plt.hlines(1, xmin = 0.3, xmax=1.0, linestyle=':', color = 'black')
plt.yscale('log')
plt.title(r'KL Divergence $L_x=L_y=1.0$')
plt.xlabel(r'$L_z / L_{\mathrm{LSS}}$')
plt.ylabel(r'$D_{KL}$')
plt.legend()
plt.savefig('kl.pdf', bbox_inches='tight')