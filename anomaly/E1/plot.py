import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

num=7000
S_2_coords = np.load('S_2_coords_E1_L_11_beta_75_it_{}.npy'.format(num))
S_3_coords = np.load('S_3_coords_E1_L_11_beta_75_it_{}.npy'.format(num))
t_3 = np.load('t_3_E1_L_11_beta_75_it_{}.npy'.format(num))
t3_PTE = np.sum(np.where(t_3 > 0.94, 1, 0)) / len(t_3)
print(t_3, t3_PTE*100)
plt.figure()
plt.hist(t_3, bins=30)
plt.savefig('planarity_t_3_{}.pdf'.format(num))

n_2 = np.array([cos(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), sin(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), cos(S_2_coords[:, 0])])
n_3 = np.array([cos(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), sin(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), cos(S_3_coords[:, 0])])

plt.figure()
dot = np.zeros(n_2[0, :].size)
for i in range(len(dot)):
 dot[i] = np.dot(n_2[:, i], n_3[:, i])
plt.hist(dot, bins=30)
plt.savefig('octopole_quadrupole_dot_{}.pdf'.format(num))