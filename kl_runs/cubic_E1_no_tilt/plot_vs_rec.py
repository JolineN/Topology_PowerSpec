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

# Set parameter file
num = 9
L = np.linspace(0.6, 1.4, num)

l_max_list = np.array([30])


plt.figure()
for i, l_max in enumerate(l_max_list):
  kl = np.load('a_t_E1_lmax{}.npy'.format(l_max))
  plt.plot(L, kl[2:], color='blue', label=r'Cubic E1' if i == 0 else None, alpha = 1-0.3*i)

  kl = np.load('../E1_noncubic_LxLy14_notilt/kl_lmax{}.npy'.format(l_max))
  plt.plot(L, kl[3:], color='red', label=r'$L_x=L_y=1.4$ E1' if i == 0 else None, alpha = 1-0.3*i)

plt.hlines(1, xmin = np.min(L), xmax=np.max(L), linestyle=':', color = 'black')
plt.yscale('log')
plt.title(r'KL Divergence')
plt.xlabel(r'$L / L_{\mathrm{LSS}}$')
plt.ylabel(r'$D_{KL}$')
plt.legend()
plt.savefig('kl.pdf', bbox_inches='tight')