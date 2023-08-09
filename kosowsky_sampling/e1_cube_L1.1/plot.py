import matplotlib.pyplot as plt
import numpy as np

lmax_list = [50, 100, 150, 200]

exact = np.load('real_kosowsky.npy')
print(exact)
sampling = np.load('kosowsky_monte_carlo_E1_L1.1_ns_100_ntimes_6.npy')
print(sampling.shape)

plt.scatter(lmax_list[0]+2, exact, label='Exact', color='black')
plt.errorbar(lmax_list[:3], sampling[:3, 0], yerr=sampling[:3, 1], label='MC', ls='none')
plt.axhline(1, color='grey', linestyle='--')

plt.title('Kosowsky Statistics - E1 cubic L=1.1')
plt.xlabel(r'$\ell_{max}$')
plt.ylim([0, 2.5])
plt.legend()
plt.savefig('tmp.pdf')