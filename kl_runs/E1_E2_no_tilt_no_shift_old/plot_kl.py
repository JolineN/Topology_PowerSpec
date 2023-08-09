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

l_max_list = [10, 30, 50]

L = np.linspace(0.6, 1.4, 9)
print(L)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1 = np.load('kl_E1_lmax{}.npy'.format(l_max))
  plt.plot(L, kl_E1, color='red', label=r'E1 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(l_max), alpha = 1-0.3*i)
for i, l_max in enumerate(l_max_list):
  kl_E2 = np.load('kl_E2_lmax{}.npy'.format(l_max))
  plt.plot(L, kl_E2, color='blue', label=r'E2 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(l_max), alpha = 1-0.3*i)

plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
plt.hlines(1, xmin = 0.5, xmax=1.4, linestyle=':', color = 'black')
plt.yscale('log')
plt.title('KL Divergence')
plt.xlabel(r'$L / L_{\mathrm{LSS}}$')
plt.ylabel(r'$D_{KL}$')
plt.ylim([0.5*1e-1, 1e3])
plt.xlim([0.6, 1.4])
plt.legend(ncol=2)
plt.savefig('kl.pdf', bbox_inches='tight')

plt.figure()
for i, l_max in enumerate(l_max_list):
  at_E1 = np.load('a_t_E1_lmax{}.npy'.format(l_max))
  plt.plot(L, at_E1, color='red', label=r'E1 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(l_max), alpha = 1-0.3*i)

for i, l_max in enumerate(l_max_list):
  at_E2 = np.load('a_t_E2_lmax{}.npy'.format(l_max))
  plt.plot(L, at_E2, color='blue', label=r'E2 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(l_max), alpha = 1-0.3*i)

plt.vlines(1, ymin = 0.5*1e-1, ymax=15, color = 'black')
plt.hlines(1, xmin = 0.5, xmax=1.4, linestyle=':', color = 'black')
plt.title('S/N Inverse-variance weighted')
plt.xlabel(r'$L / L_{\mathrm{LSS}}$')
plt.ylabel(r'$S/N$')
plt.yscale('log')
plt.gca().tick_params(right=True)
plt.xlim([0.6, 1.4])
plt.ylim([0.2, 100])
plt.legend(ncol=2)
plt.savefig('at.pdf', bbox_inches='tight')
