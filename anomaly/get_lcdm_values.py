import camb
import healpy as hp
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from anomaly_tools import *

l_max = 3
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(200)

# Get the CAMB functions and save them
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['lensed_scalar']
TT_power_spectra = powers[:l_max+1, 0]

num_alm = 50000
a_lm_size = int((l_max+1)*(l_max+2)/2)
lcdm_alm_realizations = np.zeros((num_alm, a_lm_size), dtype=np.complex128)
c_ell_realizations = np.zeros((num_alm, l_max+1))
for i in tqdm(range(num_alm)):
  _, a_lm = hp.synfast(TT_power_spectra, nside=16, alm=True, pol=False)
  lcdm_alm_realizations[i, :] = a_lm
  c_ell_realizations[i, :] = hp.alm2cl(a_lm)
#np.save('LCDM/C_ell_2_{}_LCDM.npy'.format(num_alm), c_ell_realizations[:, 2])

t_3, S_2_coords, S_3_coords = test_octopole_planarity(lcdm_alm_realizations, rotate=True, N_side=16)

plt.figure()
plt.hist(t_3, bins=30)
plt.savefig('LCDM/t_3_LCDM_{}.pdf'.format(num_alm))
np.save('LCDM/t_3_{}_LCDM.npy'.format(num_alm), t_3)
np.save('LCDM/S_2_coords_{}_LCDM.npy'.format(num_alm), S_2_coords)
np.save('LCDM/S_3_coords_{}_LCDM.npy'.format(num_alm), S_3_coords)

#S_one_half, C_theta = get_S_one_half(lcdm_alm_realizations)
#np.save('LCDM/S_one_half_{}_LCDM.npy'.format(num_alm), S_one_half)
#np.save('LCDM/C_theta_{}_LCDM.npy'.format(num_alm), C_theta)
