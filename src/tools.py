from numba import njit, prange
import numba
import numpy as np
from numpy import pi
import pyshtools as pysh

@njit
def cart2phi(x, y):
    return np.arctan2(y, x)

@njit
def cart2theta(xy_squared, z):
    return np.arctan2(np.sqrt(xy_squared), z)

@njit
def cart2spherical(xyz):
    # Calculates spherical coordinates from cartesian coordinates
    xy = xyz[0]**2 + xyz[1]**2

    # phi
    phi = np.arctan2(xyz[1], xyz[0])
    # theta
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    
    return phi, theta

def get_D_l(c_l):
    return np.array([c_l[l] * l * (l+1) / (2*np.pi) for l in range(c_l.size)])

@njit
def isclose(param_list, param):
  # Check if a parameter is already in a list of that parameter
  # We use this for example to not recalculate spherical harmonics twice for same theta
  param_abs = np.abs(param)
  for i in range(param_list.size):
    if param_abs < 1e-8:
      if np.abs(param_list[i] - param) < 1e-8:
        return i
    elif np.abs((param_list[i] - param) / param)  < 1e-8:
      return i
  return -1

@njit
def get_idx(l_max, l, m):
    # From the Healpy library. But we copy it here so that Numba can use it
    return m * (2 * l_max + 1 - m) // 2 + l

@njit(parallel=True)
def get_unique_array_indices(full_param_array, param_repeat):
    # The point of this function is to save all unique parameters of a list
    # into a new list. For example:
    # There are many numbers in the theta list that are the same
    # but we do not want to calculate the spherical harmonics for same theta twice or more
    # So we find only the spherical harmonics for each unique theta
    # We also save a lot of memory by saving sperical harmonics for unique theta

    assert(full_param_array.size == param_repeat.size)
    num_indices = full_param_array.size
    unique_param_length = np.count_nonzero(param_repeat==-1)
    param_unique, param_unique_index = np.zeros(unique_param_length), np.zeros(num_indices, dtype=numba.int32)

    max_index = 0
    for i in prange(num_indices):
      if param_repeat[i] != -1:
        # This theta is not unique
        index = isclose(param_unique[:max_index], full_param_array[i])
        assert (index != -1)
        param_unique_index[i] = index
      else:
        # We found a new unique parameter in the list
        # Save it
        param_unique_index[i] = max_index
        param_unique[max_index] = full_param_array[i]
        max_index += 1
    return param_unique, param_unique_index

@njit
def get_c_l_from_a_lm(a_lm, l_max):
    c_l = np.zeros(l_max+1)
    for l in range(2, l_max+1):
        # m = 0
        id = get_idx(l_max, l, 0)
        c_l[l] += np.abs(a_lm[id]) ** 2
        for m in range(1, l+1):
            id = get_idx(l_max, l, m)
            alm = a_lm[id]
            c_l[l] += 2 * np.abs(alm) ** 2
        c_l[l] /= 2*l+1
    return c_l

@njit
def get_c_l_from_c_lmlpmp(c_lmlpmp, l_max):
    c_l = np.zeros(l_max+1, dtype=np.complex128)
    for l in range(l_max+1):
        for m in range(-l, l+1):
            lm_id = l * (l+1) + m
            
            if c_lmlpmp.ndim == 1:
                c_l[l] += c_lmlpmp[lm_id]
            else:
                c_l[l] += c_lmlpmp[lm_id, lm_id]
        c_l[l] /= 2*l + 1
    return np.real(c_l)

def get_sph_harm_parallel(i, l_max, theta, lm_index, num_l_m):
  # Get the spherical harmonics with no phase (phi=0) for a given index i
  sph_harm_no_phase_i = np.zeros(num_l_m, dtype=np.float64)
  theta_cur = theta[i]
  all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, theta_cur, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
  for l in range(l_max+1):
      for m in range(l+1):
          cur_index = lm_index[l, m]
          sph_harm_no_phase_i[cur_index] = all_sph_harm_no_phase[0, l, m]      
  return sph_harm_no_phase_i

def transfer_parallel(i, l_max, k_amp, transfer_interpolate_k_l_list):
  # Get the transfer function for a given index i
  cur_transfer_i = np.zeros(l_max+1)
  k = k_amp[i]
  for l in range(2, l_max+1):
    cur_transfer_i[l] = transfer_interpolate_k_l_list[l](k)
  return cur_transfer_i

def get_k_theta_index_repeat(k_amp, theta):
    # k and theta often repeats themselves in the full list of allowed wavenumber list
    # So we want to know when they have repeated values so that we dont have to
    # recalculate spherical harmonics for example.

    # The unique lists are the lists of only unique elements
    # The unique_index lists are the indices going form the full parameter list to the
    # unique list.
    # For example:
    # k = [1, 2, 3, 1, 2, 5]
    # k_unique = [1, 2, 3, 5]
    # k_unique_index = [0, 1, 2, 0, 1, 3]
    
    length = theta.size

    print('Getting repeated theta and k elements')
    k_amp_unique, k_amp_unique_index = np.unique(np.round(k_amp, decimals=7), return_inverse=True)
    theta_unique, theta_unique_index = np.unique(np.round(theta, decimals=7), return_inverse=True)
    print('Ratio of unique theta:', theta_unique.size / length)
    print('Ratio of unique |k|:', k_amp_unique.size / length)

    return k_amp_unique, k_amp_unique_index, theta_unique, theta_unique_index

@njit
def do_integrand_pre_processing(unique_k_amp, scalar_pk_k3, transfer_delta_kl, l_max):
    # Calculating P(k) / k^3 * Delta_ell(k) * Delta_ell'(k) 
    num_k_amp = unique_k_amp.size
    integrand = np.zeros((num_k_amp, l_max+1, l_max+1))
    for i in prange(num_k_amp):
        scalar_pk_k3_cur = scalar_pk_k3[i]
        for l in range(2, l_max+1):
            scalar_pk_k3_transfer_delta_kl = scalar_pk_k3_cur * transfer_delta_kl[i, l]
            for lp in range(2, l_max+1):
                integrand[i, l, lp] = scalar_pk_k3_transfer_delta_kl * transfer_delta_kl[i, lp]
    return integrand

@njit
def normalize_c_lmlpmp(c_lmlpmp, camb_c_l, l_min, l_max, lp_min, lp_max, cl_accuracy=1):
    # Normalize the covariance matrix by dividing sqrt(c_l * c_l'). This is used in the KL divergence.
    # I also divide by cl_accuracy as an approximation of the lost power along the diagonal.
    # Wihtout it we get that the diagonal in the L->infinity limit becomes cl_accuracy and not 1. This
    # adds a contribution to the KL divergence which should not be there.
    normalized_c_lmlpmp = np.zeros(c_lmlpmp.shape, dtype=np.complex128)
    for l in range(l_min, l_max+1):
        for m in range(-l, l+1):
            index = l * (l+1) + m - l_min**2
            for lp in range(lp_min, lp_max+1):
                for mp in range(-lp, lp+1):
                    index_p = lp * (lp+1) + mp - lp_min**2                    
                    normalized_c_lmlpmp[index, index_p] = c_lmlpmp[index, index_p] / (np.sqrt(camb_c_l[l]*camb_c_l[lp]) * cl_accuracy)
    return normalized_c_lmlpmp