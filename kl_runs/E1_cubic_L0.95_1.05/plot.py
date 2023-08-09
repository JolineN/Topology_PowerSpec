import matplotlib.pyplot as plt
import numpy as np

L_list = np.linspace(0.95, 1.05, 11)

exact = np.load('real_kl.npy')

plt.plot(L_list, exact)
plt.axhline(1, color='grey', linestyle='--')
plt.axvline(1, color='grey', linestyle='--')

plt.title('KL - E1 cubic, l_max=30')
plt.xlabel(r'$L/L_{LSS}$')
plt.legend()
plt.savefig('tmp.pdf')