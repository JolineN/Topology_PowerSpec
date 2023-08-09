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
plt.rc('legend', fontsize=10)

root = '../../../kl_runs/compare_EX/'
num = 10
# Set parameter file
L_list = np.linspace(0.4, 1.3, num)
topology_list = np.array(['E1', 'E2', 'E3', 'E4', 'E5', 'E6'])
labels = np.array([r'$E_1$', r'$E_2$', r'$E_3$', r'$E_4$', r'$E_5$', r'$E_6$'])

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6/1.3, 8.2/1.3))

for i, top in enumerate(topology_list):
  
  kl = np.load(root+'kl_{}.npy'.format(top))
  if top != 'E6':
    kl_lowL = np.load(root+'kl_{}_lowL.npy'.format(top))
    tot_kl = np.zeros(len(kl) + len(kl_lowL))
    tot_kl[:len(kl_lowL)] = kl_lowL
    tot_kl[len(kl_lowL):] = kl
    axes[0].plot(L_list[3:], tot_kl[3:], label=labels[i], alpha = 0.3)
  else:
    axes[0].plot(L_list[3:], kl, label=top, alpha = 0.3)


  #axes[0].set_xlabel(r'$L / L_{\mathrm{LSS}}$')
  axes[0].set_ylabel(r'$D^{P||Q}_{\mathrm{KL}}$')
  axes[0].set_xlim([0.7, 1.3])
  axes[0].set_ylim([0.1, 1.5*1e2])
  axes[0].set_yscale('log')
  axes[0].set_xticklabels([])
  
  kl = np.load(root+'kl_Q_ass_P{}.npy'.format(top))
  if top != 'E6':
    kl_lowL = np.load(root+'kl_Q_ass_P{}_lowL.npy'.format(top))
    tot_kl = np.zeros(len(kl) + len(kl_lowL))
    tot_kl[:len(kl_lowL)] = kl_lowL
    tot_kl[len(kl_lowL):] = kl
    axes[1].plot(L_list[3:], tot_kl[3:], label=top, alpha = 0.3)
  else:
    axes[1].plot(L_list[3:], kl, label=top, alpha = 0.3)
  axes[1].set_ylabel(r'$D^{Q||P}_{\mathrm{KL}}$')
  axes[1].set_xlim([0.7, 1.3])
  axes[1].set_ylim([0.1, 1.5*1e2])
  axes[1].set_yscale('log')
  axes[1].set_xticklabels([])

  kos = np.load(root+'at_{}.npy'.format(top))
  if top != 'E6':
    kosL = np.load(root+'at_{}_lowL.npy'.format(top))
    tot_kos = np.zeros(len(kos) + len(kosL))
    tot_kos[:len(kosL)] = kosL
    tot_kos[len(kosL):] = kos
    tot_kos = (tot_kos/np.sqrt(2))**2
    axes[2].plot(L_list[3:], tot_kos[3:], label=labels[i], alpha = 0.3)
  else:
    kos = (kos/np.sqrt(2))**2
    axes[2].plot(L_list[3:], kos, label=labels[i], alpha = 0.3)
  axes[2].set_xlabel(r'$L / L_{\mathrm{LSS}}$')
  axes[2].set_ylabel(r'Off-diagonal $(\mathrm{S}/\mathrm{N})^2$')
  axes[2].set_xlim([0.7, 1.3])
  axes[2].set_ylim([0.1, 1.5*1e2])
  axes[2].set_yscale('log')

  axes[0].axvline(1, linestyle=':', color = 'black')
  axes[0].axhline(1, linestyle=':', color = 'black')
  axes[1].axvline(1, linestyle=':', color = 'black')
  axes[1].axhline(1, linestyle=':', color = 'black')
  axes[2].axvline(1, ymin=0, ymax=1, linestyle=':', color = 'black')
  axes[2].axhline(1, linestyle=':', color = 'black')
  axes[2].legend(frameon=False, loc='upper right', ncol=3)

plt.savefig('kl_and_kosowsky_default.pdf')

plt.figure()

top = 'E1'
kl = np.load(root+'kl_{}.npy'.format(top))
kl_lowL = np.load(root+'kl_{}_lowL.npy'.format(top))
tot_kl = np.zeros(len(kl) + len(kl_lowL))
tot_kl[:len(kl_lowL)] = kl_lowL
tot_kl[len(kl_lowL):] = kl
plt.plot(L_list[3:], tot_kl[3:], alpha = 0.3, label=r'$D^{P||Q}_{\mathrm{KL}}$')

kl = np.load(root+'kl_Q_ass_P{}.npy'.format(top))
kl_lowL = np.load(root+'kl_Q_ass_P{}_lowL.npy'.format(top))
tot_kl = np.zeros(len(kl) + len(kl_lowL))
tot_kl[:len(kl_lowL)] = kl_lowL
tot_kl[len(kl_lowL):] = kl
plt.plot(L_list[3:], tot_kl[3:], alpha = 0.3, label=r'$D^{Q||P}_{\mathrm{KL}}$')

kos = np.load(root+'at_{}.npy'.format(top))
kosL = np.load(root+'at_{}_lowL.npy'.format(top))
tot_kos = np.zeros(len(kos) + len(kosL))
tot_kos[:len(kosL)] = kosL
tot_kos[len(kosL):] = kos
tot_kos = (tot_kos/np.sqrt(2))**2
plt.plot(L_list[3:], tot_kos[3:], alpha = 0.3, label = r'Off-diagonal $(\mathrm{S}/\mathrm{N})^2$')
plt.legend(frameon=False)

plt.xlabel(r'$L / L_{\mathrm{LSS}}$')
plt.xlim([0.7, 1.3])
plt.ylim([0.1, 1.5*1e2])
plt.axvline(1, linestyle=':', color = 'black')
plt.axhline(1, linestyle=':', color = 'black')
plt.yscale('log')
plt.title(r'Cubic $E_1$')
plt.savefig('E_1_all_stats.pdf')