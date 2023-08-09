import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

num=7000
C_theta = np.load('C_theta_E1_L_11_beta_75_it_{}.npy'.format(num))
S_one_half = np.load('S_one_half_E1_L_11_beta_75_it_{}.npy'.format(num))
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
plt.plot(theta*180/np.pi, C_theta_mean_std[:, 0], color='blue', label=r'E1, $\beta=75^\circ, L=1.1$')
plt.text(70, 1000, r'E1: p-value of $S_{1/2}<1000 (\mu K)^4$ is 0.31%')
plt.text(62, 800, r'LCDM: p-value of $S_{1/2}<1000 (\mu K)^4$ is 0.11%')
plt.axhline(0, xmin=0, xmax=180, linestyle='--', color='black')
plt.fill_between(theta*180/np.pi, C_theta_mean_std[:, 2], C_theta_mean_std[:, 1], color='blue', alpha=0.5)

C_theta_LCDM = np.load('../LCDM/C_theta_100000_LCDM.npy')
num, theta_steps = C_theta_LCDM.shape
C_theta_LCDM_mean_std = np.zeros((theta_steps, 3))
for i in range(theta_steps):
  mean_sigma = np.percentile(C_theta_LCDM[:, i], [16, 50, 84])

  # Mean
  C_theta_LCDM_mean_std[i, 0] = mean_sigma[1]

  # Upper
  C_theta_LCDM_mean_std[i, 1] = mean_sigma[2]

  # Lower
  C_theta_LCDM_mean_std[i, 2] = mean_sigma[0]

theta = np.linspace(0, np.pi, theta_steps)
plt.plot(theta*180/np.pi, C_theta_LCDM_mean_std[:, 0], color='black', label=r'LCDM')
plt.fill_between(theta*180/np.pi, C_theta_LCDM_mean_std[:, 2], C_theta_LCDM_mean_std[:, 1], color='grey', alpha=0.3)
plt.legend()
plt.title(r'$C(\theta)$ - $\ell_{max}=8$')
plt.xlim([0, 180])
plt.savefig('C_theta_compared_with_LCDM_it_{}.pdf'.format(num))










plt.figure()
plt.hist(S_one_half, bins=2000)
plt.xlim([0, 10000])
plt.savefig('plots/E1_beta_75_L_11_S_one_half.pdf')

print((S_one_half<1000).sum()/len(S_one_half) * 100, '%', (S_one_half<1000).sum(), 'out of', len(S_one_half))