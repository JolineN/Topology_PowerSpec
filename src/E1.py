from .E1_tools import *
from .topology import *
import numpy as np
import numba as nb

class E1(Topology):
  def __init__(self, param, debug=True):
    Topology.__init__(self, param, debug)

  def do_pre_processing(self):
    # Calculates the transfer functions, primordial power spectrum,
    # and spherical harmonics and stores them for future use

    # Pre-computes the correct healpy index for (l,m) and spherical harmonics for future use
    l_max = self.l_max
    n_max = max(self.n_max_list)
    num_l_m = int((l_max + 1)*(l_max + 2)/2)

    # k is discrete, so caluclate all possible integers nx^2+ny^2+nz^2
    # Also find all possible nx^2+ny^2, we will use this later to find theta for the
    # spherical harmonics
    
    start_time = time.time()
    list_n_xy_squared, list_n_xyz_squared = get_list_of_n_squared(n_max)
    print('Time to get list of n_xyz:', time.time()-start_time)

    self.list_n_xyz_squared = list_n_xyz_squared
    self.list_n_xy_squared = list_n_xy_squared

    # Number of unique nx^2+ny^2(+nz^2) we can have
    num_xyz_squared = list_n_xyz_squared.size
    num_xy_squared = list_n_xy_squared.size

    # This will be the P(k)/k^3 * transfer_k(l) * conj(transfer_k(l')). 

    scalar_pk_k3 = np.zeros(num_xyz_squared)

    start_time = time.time()

    # So instead of for looping over all nx, ny, nz, we instead loop over possible n^2 values
    transfer_interpolate_k_l = scipy.interpolate.interp2d(self.k_list, self.ell_list, self.transfer_data[0, :, :], kind='quintic')
    transfer_delta_kl = np.zeros((num_xyz_squared, self.l_max+1))
    print('Transfer is', round(getsizeof(transfer_delta_kl) / 1024 / 1024,2), 'MB')
    for n_squared_index in range(1, num_xyz_squared):
        n_squared = list_n_xyz_squared[n_squared_index]

        n = sqrt(n_squared)
        k = 2*pi / self.L * n
        scalar_pk_k3[n_squared_index] = self.pars.scalar_power(k) / k**3
    
        for l in range(self.l_max+1):
            if n > self.n_max_list[l]:
                continue
            transfer_delta_kl[n_squared_index, l] = transfer_interpolate_k_l(k, l)

    print('Time get transfer_delta_kl:', time.time()-start_time)

    # Store them
    self.scalar_pk_k3 = scalar_pk_k3
    self.transfer_delta_kl = transfer_delta_kl
    self.transfer_interpolate_k_l = transfer_interpolate_k_l
    
    
    if self.is_nmax_high_enough() == False:
        # n_max is not high enough. So we start pre-processing again with a higher n_max
        print('Starting the pre-process again with higher n_max')
        print('\n')
        self.do_pre_processing()
        return

    start_time = time.time()
    integrand = do_integrand_pre_processing(list_n_xyz_squared, scalar_pk_k3, transfer_delta_kl, self.L, self.l_max)
    self.integrand = integrand
    print('Time get calculate integrand:', time.time()-start_time)

    # Healpy ordering of a_lm
    lm_index = np.zeros((l_max+1, l_max+1), dtype=int)
    for l in range(l_max+1):
        for m in range(l+1):
            cur_index = hp.Alm.getidx(l_max, l, m)
            lm_index[l, m] = cur_index

    # Get the spherical harmonics without the phase (without exp(i*m*phi))
    # We store these in an array of size (all lm, all n_z, all n*x^2+n_y^2)
    # This is because theta can be found from nz and nx^2+ny^2, and we do not
    # care about phi since we can add the phase in the sum
    sph_harm_no_phase = np.zeros((num_l_m, n_max+1, num_xy_squared))
    print('Spherical harmonics array is', round(getsizeof(sph_harm_no_phase) / 1024 / 1024,2), 'MB')
    for n_z in tqdm(range(n_max+1)):
        for n_squared_index in range(num_xy_squared):
            n_xy_squared = list_n_xy_squared[n_squared_index]

            n_squared = n_xy_squared + n_z**2
            n = sqrt(n_squared)
            if n_squared==0 or n > n_max:
                continue

            # Get theta for the sph_harm function
            theta = cart2theta(n_xy_squared, n_z)
            all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, theta, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
            for l in range(l_max+1):
                if n > self.n_max_list[l]:
                    continue
                for m in range(l+1):
                    cur_index = lm_index[l, m]
                    sph_harm_no_phase[cur_index, n_z, n_squared_index] = all_sph_harm_no_phase[0, l, m]
                    
    # Save healpy indices, legendre functions, and the pre-factor
    self.lm_index = lm_index
    self.sph_harm_no_phase = sph_harm_no_phase