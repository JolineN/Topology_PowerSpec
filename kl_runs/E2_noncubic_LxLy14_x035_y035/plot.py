import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=15)
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16, labelsize=17)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)


linestyles = ['solid', 'dashed', 'dotted']

num = 12
L = np.linspace(0.3, 1.4, num)


plt.figure()
l_max_list = [30, 20, 10]
for i, l_max in enumerate(l_max_list):
  kl = np.load('../E1_noncubic_LxLy14_notilt/kl_lmax{}.npy'.format(l_max))
  plt.plot(L, kl, color='red', label=r'E1 - No tilt' if i == 0 else None, alpha = 1-0.3*i, linestyle = linestyles[i])

for i, l_max in enumerate(l_max_list):
  kl = np.load('kl_lmax{}.npy'.format(l_max))
  plt.plot(L, kl, color='green', label=r'E2 - $x_0=(0.35, 0.35, 0)L_{LSS}$' if i == 0 else None, alpha = 1-0.3*i, linestyle = linestyles[i])
for i, l_max in enumerate(l_max_list):
  kl = np.load('../E2_noncubic_LxLy14_x035_y0/kl_lmax{}.npy'.format(l_max))
  plt.plot(L, kl, color='blue', label=r'E2 - $x_0=(0.35, 0, 0)L_{LSS}$' if i == 0 else None, alpha = 1-0.3*i, linestyle = linestyles[i])


black_line3 = mlines.Line2D([], [], color='black', marker='s',linestyle=linestyles[0],markersize=0)
black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle=linestyles[1],markersize=0)
black_line1 = mlines.Line2D([], [], color='black', marker='s',linestyle=linestyles[2],markersize=0)
legend2 = plt.legend([black_line1, black_line2, black_line3],[r'$\ell_{\mathrm{max}}=10$', r'$\ell_{\mathrm{max}}=20$', r'$\ell_{\mathrm{max}}=30$'], loc='lower left', frameon=False)
plt.legend(frameon=False)
plt.gca().add_artist(legend2)

plt.axvline(0.6, linestyle=':', color = 'black')
plt.axvline(1, linestyle=':', color = 'black')
plt.hlines(1, xmin = np.min(L), xmax=np.max(L), linestyle=':', color = 'black')
plt.yscale('log')
plt.title(r'KL Divergence $L_x=L_y=1.4$')
plt.xlabel(r'$L_z / L_{\mathrm{LSS}}$')
plt.ylabel(r'$D_{KL}$')
#plt.legend()
plt.savefig('kl.pdf', bbox_inches='tight')