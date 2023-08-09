import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt

t_3_lcdm = np.load('../LCDM/t_3_50000_LCDM.npy')
t3_lcdm_PTE = np.sum(np.where(t_3_lcdm > 0.94, 1, 0)) / len(t_3_lcdm)

S_2_lcdm = np.load('../LCDM/S_2_coords_100000_LCDM.npy')
S_3_lcdm = np.load('../LCDM/S_3_coords_100000_LCDM.npy')
n_2_lcdm = np.array([cos(S_2_lcdm[:, 1])*sin(S_2_lcdm[:, 0]), sin(S_2_lcdm[:, 1])*sin(S_2_lcdm[:, 0]), cos(S_2_lcdm[:, 0])])
n_3_lcdm = np.array([cos(S_3_lcdm[:, 1])*sin(S_3_lcdm[:, 0]), sin(S_3_lcdm[:, 1])*sin(S_3_lcdm[:, 0]), cos(S_3_lcdm[:, 0])])
C_ell_2_lcdm = np.load('../LCDM/C_ell_2_100000_LCDM.npy')
c_ell_2_lcdm_PTE = np.sum(np.where(C_ell_2_lcdm < 226, 1, 0)) / len(C_ell_2_lcdm)

dot_lcdm = np.zeros(n_2_lcdm[0, :].size)
for i in range(len(dot_lcdm)):
  dot_lcdm[i] = np.dot(n_2_lcdm[:, i], n_3_lcdm[:, i])
dot_lcdm_PTE = np.sum(np.where(dot_lcdm > 0.98, 1, 0)) / len(dot_lcdm)

print('LCDM - t_3:', t3_lcdm_PTE*100, 'C_2:', c_ell_2_lcdm_PTE*100, 'n_2 n_3:', dot_lcdm_PTE*100)

top_data = np.load('anomaly_top.npy')
topology_list = np.array(['E1', 'E2', 'E3', 'E4', 'E5'])

fig, ax1 = plt.subplots(3, 1, figsize=(4, 6))
fig.tight_layout()
ax1[0].set_title(r'$t_3 > 0.94$')
ax1[0].bar(topology_list,  top_data[0, :]*100)
ax1[0].set_ylabel('PTE %')
ax1[0].axhline(t3_lcdm_PTE*100, color='black', linestyle='--')

ax1[1].set_title(r'$C_2 < 226 \mu K^2$')
ax1[1].bar(topology_list,  top_data[1, :]*100)
ax1[1].set_ylabel('PTE %')
ax1[1].axhline(c_ell_2_lcdm_PTE*100, color='black', linestyle='--')

ax1[2].set_title(r'$|n_2 \cdot n_3| > 0.98$')
ax1[2].bar(topology_list,  top_data[2, :]*100)
ax1[2].set_ylabel('PTE %')
ax1[2].axhline(dot_lcdm_PTE*100, color='black', linestyle='--')

plt.savefig('tmp.pdf', bbox_inches='tight')
for i, top in enumerate(topology_list):
  print('{} - t_3: {} C_2: {} n_2 n_3: {}'.format(top, top_data[0, i]*100, top_data[1, i]*100, top_data[2, i]*100))