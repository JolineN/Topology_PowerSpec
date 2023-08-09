import matplotlib.pyplot as plt
import numpy as np

l_max = np.array([10, 20, 30, 40, 50, 100])

exact = np.load('real_kosowsky.npy')
print(exact)
sampling = np.load('kosowsky_monte_carlo_E3_L1.2_ns_100_ntimes_20.npy')
print(sampling.shape)

plt.scatter(l_max[:5]+2, exact, label='Exact', color='black')
plt.errorbar(l_max, sampling[:6, 0], yerr=sampling[:6, 1], label='MC', ls='none')
plt.axhline(1, color='grey', linestyle='--')

plt.title('Kosowsky Statistics - E3 cubic L=1.2 x_0=[0.2, 0, 0]')
plt.xlabel(r'$\ell_{max}$')
plt.ylim([0, 1.2])
plt.legend()
plt.savefig('tmp.pdf')