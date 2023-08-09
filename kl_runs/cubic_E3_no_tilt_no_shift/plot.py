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

L = np.linspace(0.6, 1.4, 9)
lmax_list = [50]



for lmax in lmax_list:
  E3 = np.load('a_t_E3_lmax{}.npy'.format(lmax))
  plt.plot(L, E3, label=r'E3 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))

  E1 = np.load('../cubic_E1_no_tilt/a_t_E1_lmax{}.npy'.format(lmax))
  plt.plot(L, E1[2:], label=r'E1 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))

  plt.plot(L, E3-E1[2:], label=r'E3-E1 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))
plt.axhline(1, color='grey')
plt.xlabel('$L/L_{\mathrm{max}}$')
plt.yscale('log')
plt.ylabel('S/N')
plt.legend()
plt.savefig('kosowsky.pdf')
print(E3[-1], E1[-1])

plt.figure()
for lmax in lmax_list:
  E3 = np.load('kl_E3_lmax{}.npy'.format(lmax))
  plt.plot(L, E3, label=r'E3 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))

  E1 = np.load('../cubic_E1_no_tilt/kl_E1_lmax{}.npy'.format(lmax))
  plt.plot(L, E1[2:], label=r'E1 $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))

  plt.plot(L, np.abs(E3-E1[2:]), label=r'|E3-E1| $\ell_{{\mathrm{{max}}}}={{{}}}$'.format(lmax))
plt.axhline(1, color='grey')
plt.xlabel('$L/L_{\mathrm{max}}$')
plt.yscale('log')
plt.ylabel(r'$D_{KL}$')
plt.legend()
plt.savefig('kl.pdf')
