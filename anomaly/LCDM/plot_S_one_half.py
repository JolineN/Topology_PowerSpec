import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

num=100000
C_theta = np.load('C_theta_{}_LCDM.npy'.format(num))
S_one_half = np.load('S_one_half_{}_LCDM.npy'.format(num))
num, theta_steps = C_theta.shape
C_theta_mean_std = np.zeros((theta_steps, 3))
for i in range(theta_steps):
  mean_sigma = np.percentile(C_theta[:, i], [16, 50, 84])

  # Mean
  C_theta_mean_std[i, 0] = mean_sigma[1]

  # Upper
  C_theta_mean_std[i, 1] = mean_sigma[2]

  # Lower
  C_theta_mean_std[i, 2] = mean_sigma[0]

theta = np.linspace(0, np.pi, theta_steps)
plt.plot(theta*180/np.pi, C_theta_mean_std[:, 0])
plt.axhline(0, xmin=0, xmax=180, linestyle='--', color='black')
plt.fill_between(theta*180/np.pi, C_theta_mean_std[:, 2], C_theta_mean_std[:, 1], color='grey', alpha=0.5)
plt.savefig('C_theta_it_{}.pdf'.format(num))

plt.figure()
plt.hist(S_one_half, bins=5000)
plt.xlim([0, 5000])
plt.savefig('S_one_half_it_{}.pdf'.format(num))

print((S_one_half<1000).sum()/len(S_one_half) * 100, '%', (S_one_half<1000).sum(), 'out of', len(S_one_half))
