import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import quaternionic
import spherical
from tqdm import tqdm
from matplotlib import gridspec

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=14)
#plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)

data = np.load('../E2/E2_no_shift_corr_matrix_l_2_10_lp_2_10.npy')

ell_min = 2
ell_max = 10

wigner = spherical.Wigner(ell_max)

theta_rot = 1.53
phi_rot = 2.12
R = quaternionic.array.from_spherical_coordinates(theta_rot, phi_rot)
D = wigner.D(R)

print(D[wigner.Dindex(5, 2, 3)])
'''
rot_cov = np.zeros(data.shape, dtype=np.complex128)

for l in tqdm(range(ell_min, ell_max+1)):
  for m in range(-l, l + 1):
      lm_index_cur = l * (l+1) + m - ell_min**2
      for l_p in range(ell_min, ell_max+1): 
        for m_p in range(-l_p, l_p + 1):
            lm_p_index_cur = l_p * (l_p + 1) + m_p - ell_min**2
            cur_element = 0
            for m_bar in range(-l, l+1):
              index_m_bar = l * (l+1) + m_bar - ell_min**2
              for m_bar_p in range(-l_p, l_p+1):
                #D[wigner.Dindex(l, m, mp)]
                index_m_bar_p = l_p * (l_p + 1) + m_bar_p - ell_min**2
                cur_element += D[wigner.Dindex(l, m, m_bar)] * np.conjugate(D[wigner.Dindex(l_p, m_p, m_bar_p)]) * data[index_m_bar, index_m_bar_p]
            rot_cov[lm_index_cur, lm_p_index_cur] = cur_element
np.save('rot_cov.npy', rot_cov)
'''
rot_cov = np.load('rot_cov.npy')
def plot_ax(cov, sub, i, l_min, l_max, lp_min, lp_max, title):
  C_TT_order = cov
  C_TT_order = np.abs(C_TT_order)
  C_TT_order = np.where(C_TT_order > 1e-6, C_TT_order, 1e-6)
  ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
  ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])

  axim = sub.imshow(C_TT_order.T, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
          
  jump = np.array([2, 5, 8, 10])-2


  if i == 0:
    sub.set_yticks(ellp_to_s_map[jump])
    sub.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
    sub.set_ylabel(r"$\ell'$", Rotation=0)
  else:
    sub.set_yticks([])
    sub.set_yticklabels([])

  sub.set_xticks(ell_to_s_map[jump])
  sub.set_xticklabels(np.arange(l_min, l_max+1)[jump])
  sub.set_xlabel(r'$\ell$')

  sub.set_xlim([0, (l_max+1)*(l_max+2) - (l_max+1) - l_min**2 - 1])
  sub.set_ylim([0, (lp_max+1)*(lp_max+2) - (lp_max+1) - lp_min**2 - 1])

  if title != None:
    sub.set_title(title)

  axim.set_clim(1e-6, 1e0)
  #ax.set_aspect('equal')
  return axim

fig = plt.figure(figsize=(7, 4), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1],
         wspace=0.02, hspace=0.0, top=1, bottom=0, left=0.07, right=0.8) 

sub1 = plt.subplot(gs[0])
axim = plot_ax(data, sub1, 0, 2, 10, 2, 10, r'Unrotated coordinate system')

sub2 = plt.subplot(gs[1])
axim = plot_ax(rot_cov, sub2, 1, 2, 10, 2, 10, 'Randomly rotated\ncoordinate system')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.05, 0.04, 0.9])
fig.colorbar(axim, cax=cbar_ax)

plt.savefig('rot_C.pdf')

w, _ = np.linalg.eig(data)
kl_P_assuming_Q = 0
kl_Q_assuming_P = 0
for eig in w:
    kl_P_assuming_Q += (np.log(np.abs(eig)) + 1/eig - 1)/2
    kl_Q_assuming_P += (-np.log(np.abs(eig)) + eig - 1)/2
np.fill_diagonal(data, 0)
a_t = np.sqrt(np.sum(np.abs(data)**2))
print('Unrotated:', kl_P_assuming_Q, kl_Q_assuming_P, a_t)

w, _ = np.linalg.eig(rot_cov)
kl_P_assuming_Q = 0
kl_Q_assuming_P = 0
for eig in w:
    kl_P_assuming_Q += (np.log(np.abs(eig)) + 1/eig - 1)/2
    kl_Q_assuming_P += (-np.log(np.abs(eig)) + eig - 1)/2
np.fill_diagonal(rot_cov, 0)
a_t = np.sqrt(np.sum(np.abs(rot_cov)**2))
print('Unrotated:', kl_P_assuming_Q, kl_Q_assuming_P, a_t)