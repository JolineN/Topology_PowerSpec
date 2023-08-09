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

plt.figure()
E1_list = np.zeros((2,5))
E2_list = np.zeros((2,5))
for i, l_max in enumerate(l_max_list):
  kl_E1 = np.load('kl_E1_lmax{}.npy'.format(l_max))
  E1_list[0, i] = kl_E1[0]
  E1_list[1, i] = kl_E1[-1]

for i, l_max in enumerate(l_max_list):
  kl_E2 = np.load('kl_E2_lmax{}.npy'.format(l_max))
  E2_list[0, i] = kl_E2[0]
  E2_list[1, i] = kl_E2[-1]

plt.plot(l_max_list, E1_list[0, :], label='E1 L=0.6', color='red', linestyle='--') 
plt.plot(l_max_list, E2_list[0, :], label='E2 L=0.6', color='blue', linestyle='--')
#plt.plot(l_max_list, 100*np.log(l_max_list)-200)
plt.title('KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
plt.ylim([0.5*1e-1, 250])
plt.legend(ncol=2)
plt.savefig('kl_func_of_l_max_L06.pdf', bbox_inches='tight')

plt.figure()
plt.plot(l_max_list, E2_list[1, :], label='E1 L=1.4', color='blue', linestyle='-')
plt.title('KL Divergence')
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'$D_{KL}$')
plt.ylim([0.5*1e-1, 2])
plt.legend(ncol=2)
plt.savefig('kl_func_of_l_max_L14.pdf', bbox_inches='tight')


