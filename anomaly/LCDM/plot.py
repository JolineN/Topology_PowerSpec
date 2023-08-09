import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

num=1000
S_2_coords = np.load('S_2_coords_{}_LCDM.npy'.format(num))
S_3_coords = np.load('S_3_coords_{}_LCDM.npy'.format(num))
t_3 = np.load('t_3_{}_LCDM.npy'.format(num))

plt.figure()
plt.hist(t_3, bins=30)
plt.savefig('t_3_LCDM_{}.pdf'.format(num))

n_2 = np.array([cos(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), sin(S_2_coords[:, 1])*sin(S_2_coords[:, 0]), cos(S_2_coords[:, 0])])
n_3 = np.array([cos(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), sin(S_3_coords[:, 1])*sin(S_3_coords[:, 0]), cos(S_3_coords[:, 0])])

plt.figure()
print(n_2)
dot = np.zeros(n_2[0, :].size)
for i in range(len(dot)):
 dot[i] = np.dot(n_2[:, i], n_3[:, i])
print(dot)
plt.hist(dot, bins=30)
plt.savefig('quadrupole_octopole_alignment_LCDM_{}.pdf'.format(num))