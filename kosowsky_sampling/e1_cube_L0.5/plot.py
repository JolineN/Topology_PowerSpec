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

l_max = np.array([10, 30, 50, 100, 150, 200])

exact = np.load('real_kosowsky.npy')
print(exact)
sampling = np.load('kosowsky_monte_carlo_E1_L0.5_ns_100_ntimes_6.npy')
print(sampling.shape)
print(exact.shape)
plt.scatter(l_max[:3]+2, exact[:3], label='Exact', color='black')
plt.errorbar(l_max, sampling[:, 0], yerr=sampling[:6, 1], label='MC', ls='none', elinewidth=2)
plt.axhline(1, color='grey', linestyle='--')

plt.title('Kosowsky Statistics - E1 cubic L=0.5')
plt.xlabel(r'$\ell_{max}$')
plt.legend()
plt.savefig('tmp.pdf')