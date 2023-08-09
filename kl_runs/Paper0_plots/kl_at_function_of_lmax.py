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

l_max_list = [10, 20, 30, 40, 50]

kl_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../E1_E2_no_tilt_no_shift_old/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E1_cubic[-1]
plt.plot(l_max_list, kl_lmax, color='red', label=r'E1 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title('E1 Cubic - KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_kl_E1_cubic.pdf', bbox_inches='tight')

at_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  at = np.load('../E1_E2_no_tilt_no_shift_old/a_t_E1_lmax{}.npy'.format(l_max))
  at_lmax[i] = at[-1]
plt.plot(l_max_list, at_lmax, color='red', label=r'E1 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title('E1 Cubic - Kosowsky Statistics')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$S/N$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_at_E1_cubic.pdf', bbox_inches='tight')

kl_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E2_cubic = np.load('../E1_E2_no_tilt_no_shift_old/kl_E2_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E2_cubic[-1]
plt.plot(l_max_list, kl_lmax, color='red', label=r'E2 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title('E2 Cubic - KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_kl_E2_cubic.pdf', bbox_inches='tight')

at_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  at = np.load('../E1_E2_no_tilt_no_shift_old/a_t_E2_lmax{}.npy'.format(l_max))
  at_lmax[i] = at[-1]
plt.plot(l_max_list, at_lmax, color='red', label=r'E2 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title('E2 Cubic - Kosowsky Statistics')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$S/N$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_at_E2_cubic.pdf', bbox_inches='tight')

kl_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E1_cubic[-1]
plt.plot(l_max_list, kl_lmax, color='red', label=r'E1 - $\beta = 50^\circ$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title(r'E1 Cubic $\beta=50^\circ$ $L=1.4$ - KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_kl_E1_cubic_tilt_L14.pdf', bbox_inches='tight')

at_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  at = np.load('../kl_run_E1_tilt_beta_50/a_t_E1_lmax{}.npy'.format(l_max))
  at_lmax[i] = at[-1]
plt.plot(l_max_list, at_lmax, color='red', label=r'E1 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title(r'E1 Cubic $\beta=50^\circ$ $L=1.4$ - Kosowsky Statistics')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$S/N$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_at_E1_cubic_tilt_L14.pdf', bbox_inches='tight')

kl_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E1_cubic[-3]
plt.plot(l_max_list, kl_lmax, color='red', label=r'E1 - $\beta = 50^\circ$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title(r'E1 Cubic $\beta=50^\circ$ $L=1.2$ - KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_kl_E1_cubic_tilt_L12.pdf', bbox_inches='tight')

at_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  at = np.load('../kl_run_E1_tilt_beta_50/a_t_E1_lmax{}.npy'.format(l_max))
  at_lmax[i] = at[-3]
plt.plot(l_max_list, at_lmax, color='red', label=r'E1 - $x_0=(0, 0, 0)$')

#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
#plt.hlines(1, xmin = 0.4, xmax=1.4, linestyle=':', color = 'black')
plt.title(r'E1 Cubic $\beta=50^\circ$ $L=1.2$ - Kosowsky Statistics')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$S/N$')
#plt.ylim([0.5*1e-1, 1e3])
#plt.xlim([0.4, 1.4])
plt.legend()
plt.savefig('lmax_at_E1_cubic_tilt_L12.pdf', bbox_inches='tight')






kl_lmax = np.zeros(5)
plt.figure()
for i, l_max in enumerate(l_max_list):
  kl_E1_cubic = np.load('../E1_E2_no_tilt_no_shift_old/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E1_cubic[-3]
plt.plot(l_max_list, kl_lmax, color='red', label=r'E1 - No tilt')

kl_lmax = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  kl_E2_cubic = np.load('../kl_run_E1_tilt_beta_50/kl_E1_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl_E2_cubic[-3]
plt.plot(l_max_list, kl_lmax, color='blue', label=r'E1 - $\beta=50^\circ$')


kl_lmax = np.zeros(5)
for i, l_max in enumerate(l_max_list):
  kl = np.load('../E1_E2_no_tilt_no_shift_old/kl_E2_lmax{}.npy'.format(l_max))
  kl_lmax[i] = kl[-3]
plt.plot(l_max_list, at_lmax, color='green', label=r'E2 - No tilt')



#plt.vlines(1, ymin = 0.5*1e-1, ymax=70, color = 'black')
plt.hlines(1, xmin = 10, xmax=50, linestyle=':', color = 'black')
plt.title('KL Divergence - $L/L_{\mathrm{LSS}} = 1.2$')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{\mathrm{KL}}$')
#plt.ylim([0.5*1e-1, 1e3])
plt.xlim([0, 4.1])
plt.legend()
plt.savefig('kl_lmax_3cases.pdf', bbox_inches='tight')