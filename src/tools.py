from numba import njit, prange
import numba
import numpy as np
from numpy import sqrt, pi
import scipy.special as spc
import healpy as hp
import math
import time
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
    spherical_coords = np.zeros(2)
    xy = xyz[0]**2 + xyz[1]**2

    #phi
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
    for l in range(1, l_max+1):
        id = get_idx(l_max, l, 0)
        c_l[l] += np.abs(a_lm[id]) ** 2
        for m in range(1, l+1):
            id = get_idx(l_max, l, m)
            alm = a_lm[id]
            c_l[l] += 2 * np.abs(alm) ** 2
        c_l[l] /= 2*l+1
    return c_l


###TEST
def numba_bug(L, l_max, n_max, lm_index, sph_harm_no_phase, list_n_xy_squared, list_n_xyz_squared, delta_k_n, transfer_delta_kl):
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    a_lm = np.zeros(num_l_m, dtype=np.complex128)
    for n_x in range(-n_max, n_max+1):
        for n_y in range(-n_max, n_max+1):
            n_xy_squared = n_x**2 + n_y**2
            i = np.where(list_n_xy_squared == n_xy_squared)[0][0]
            phi = cart2phi(n_x, n_y)

            m_list = np.arange(0, l_max+1)
            phase_list = np.exp(-1j * phi * m_list)
            
            for n_z in range(-n_max, n_max+1):
                n_squared = n_xy_squared + n_z**2
                if n_squared==0 or n_squared > n_max**2:
                    # We dont do k=0 or k > k_max
                    continue
                j = np.where(list_n_xyz_squared == n_squared)[0][0]

                random_delta_k_n = 1

                for l in range(l_max+1):
                    delta_k_n_mul_transfer = random_delta_k_n * transfer_delta_kl[j, l]
                    for m in range(l+1):
                        lm_index_cur = lm_index[l, m]
                        #sph_harm = np.exp(-1j * phi * m) * sph_harm_no_phase[lm_index_cur, np.abs(n_z), i]
                        sph_harm = phase_list[m] * sph_harm_no_phase[lm_index_cur, np.abs(n_z), i]

                        if n_z < 0: sph_harm *= (-1) ** (l+m)

                        a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm
    
    return a_lm