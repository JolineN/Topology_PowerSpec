import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=15)
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16, labelsize=17)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)

data = np.load('a_t_E2_func_lmax_10_50_L12.npy')
print(data)
errorbar = np.array([[3.39, 0.046], [3.70, 0.077]])

monte_carlo = np.load('kosowsky_monte_carlo_E2_L1.2_ns_100_ntimes_20.npy')
print(monte_carlo.shape)
l_max = np.array([10, 20, 30, 40, 50, 100, 150])

plt.figure()
plt.scatter(l_max[:5], data[:5], color='black', label='Exact')
plt.errorbar(l_max[5:], errorbar[:, 0], yerr=errorbar[:, 1], fmt='.b', label='Monte Carlo')
plt.errorbar(l_max[:5]+2, monte_carlo[:, 0], yerr=monte_carlo[:, 1], fmt='.b')
plt.axhline(1, color='grey', linestyle='--')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$S/N$')
plt.legend()
plt.title('E2 - Cubic - L=1.2')
plt.savefig('tmp.pdf')