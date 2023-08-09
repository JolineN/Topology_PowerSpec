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

l_max_list = [30]

L = np.linspace(0.4, 1.4, 11)


plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../cubic_E1_no_tilt/kl_E1_lmax{}.npy'.format(l_max))
  plt.plot(L, kl_E1_cubic, color='red', label=r'E1 - $x_0=(0, 0, 0)$' if i == 0 else None, alpha = 1-0.3*i)
for i, l_max in enumerate(l_max_list):
  kl_E1_tilt = np.load('../kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  plt.plot(L[2:], kl_E1_tilt, color='blue', label=r'E1 - $\beta=50^\circ$' if i == 0 else None, alpha = 1-0.3*i)
for i, l_max in enumerate(l_max_list):
  kl_E2_cubic_with_shift = np.load('../E2_tilt_beta50/kl_E2_lmax{}.npy'.format(l_max))
  plt.plot(L[2:], kl_E2_cubic_with_shift, color='green', label=r'E2 - $\beta=50^\circ$' if i == 0 else None, alpha = 1-0.3*i)
for i, l_max in enumerate(l_max_list):
  kl_E2_with_shift = np.load('../E2_shift_x0_y0_04_07/kl_E2_lmax{}.npy'.format(l_max))
  plt.plot(L[2:], kl_E2_with_shift, color='cyan', label=r'E2 - $x_0=(0.4, 0.7,0)L_{\mathrm{LSS}}$' if i == 0 else None, alpha = 1-0.3*i)


plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.yscale('log')
plt.title('Cubic - KL Divergence')
plt.xlabel(r'$L / L_{\mathrm{LSS}}$')
plt.ylabel(r'$D_{KL}$')
plt.ylim([0.5*1e-1, 1e3])
plt.xlim([0.6, 1.4])
plt.legend()
plt.savefig('kl_cubic.pdf', bbox_inches='tight')