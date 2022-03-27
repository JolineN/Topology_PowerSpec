from numba import njit, prange, jit, objmode
import numpy as np
from numpy import sqrt, pi
import scipy.special as spc
import healpy as hp
import math
import time
import pyshtools as pysh

# Numba actually makes this slower.
# parallel = True does not work since we are appending to the list as we go
# Not sure how to make this faster
#@njit
def get_list_of_n_squared(n_max):
    # Returns all possible integers of
    # n_x^2 + n_y^2 and n_x^2 + n_y^2 + n_z^2
    #
    # Since P(k) is a function of P(sqrt(n_x^2 + n_y^2 + n_z^2)), we can make
    # P(k) into an array for all possible values of sum of three squares.
    # This is to make the P(k) array as small as possible

    list_n_xy_squared = np.zeros((1,), dtype=np.int64)
    list_n_xyz_squared = np.zeros((1,), dtype=np.int64)
    for nx in range(n_max+1):
        for ny in range(nx, n_max+1):
            n_xy_squared = nx**2 + ny**2
            if n_xy_squared not in list_n_xy_squared:
                    list_n_xy_squared = np.append(list_n_xy_squared, n_xy_squared)
            for nz in range(ny, n_max+1):
                n_xyz_squared = n_xy_squared + nz**2
                if n_xyz_squared not in list_n_xyz_squared:
                    list_n_xyz_squared = np.append(list_n_xyz_squared, n_xyz_squared)
    list_n_xy_squared = np.sort(list_n_xy_squared)
    list_n_xyz_squared = np.sort(list_n_xyz_squared)

    return list_n_xy_squared, list_n_xyz_squared

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
    spherical_coords[0] = np.arctan2(xyz[1], xyz[0])
    # theta
    spherical_coords[1] = np.arctan2(np.sqrt(xy), xyz[2])
    
    return spherical_coords

#@njit(parallel=False)
def get_c_lmlpmp(L, l_max, n_max_list, lm_index, integrand, sph_harm_no_phase, list_n_xy_squared, list_n_xyz_squared, only_diag=False):
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    C_TT_lmlpmp = np.zeros((num_l_m, num_l_m), dtype=np.complex128)
    n_max = max(n_max_list)

    for n_x in prange(-n_max, n_max+1):
        for n_y in range(-n_max, n_max+1):
            n_xy_squared = n_x**2 + n_y**2
            i = np.where(list_n_xy_squared == n_xy_squared)[0][0]
            phi = cart2phi(n_x, n_y)
            
            for n_z in range(-n_max, n_max+1):
                n_squared = n_xy_squared + n_z**2
                if n_squared==0 or n_squared > n_max**2:
                    # We dont do k=0 or k > k_max
                    continue

                # k is discrete, so j gives the array number
                j = np.where(list_n_xyz_squared == n_squared)[0][0]

                #theta = cart2theta(n_xy_squared, n_z)

                #all_sph_harm_no_phase = np.zeros((l_max+1, l_max+1))
                #with objmode(all_sph_harm_no_phase='float64[:, :]'):
                #all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, theta, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))[0, :, :]
                #all_sph_harm_no_phase[0, 0]
                #print(all_sph_harm_no_phase.shape)
                for l in range(l_max+1):
                    if sqrt(n_squared) > n_max_list[l]:
                        continue

                    for m in range(l + 1):
                        lm_index_cur = lm_index[l, m]

                        sph_harm_no_phase_lm = sph_harm_no_phase[lm_index_cur, np.abs(n_z), i]

                        if only_diag:
                            C_TT_lmlpmp[lm_index_cur, lm_index_cur] += integrand[j, l, l] * sph_harm_no_phase_lm **2

                        else:
                            if n_z < 0: sph_harm_no_phase_lm *= (-1) ** (l+m)

                            for l_p in range(l_max+1):
                                integrand_il = integrand[j, l, l_p] * np.power(1j, l_p-l)
                                
                                for m_p in range(l_p + 1):
                                    lm_p_index_cur = lm_index[l_p, m_p]

                                    sph_harm_no_phase_lm_p = sph_harm_no_phase[lm_p_index_cur, np.abs(n_z), i]
                                    
                                    if n_z < 0: sph_harm_no_phase_lm_p *= (-1) ** (l_p+m_p)

                                    C_TT_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * sph_harm_no_phase_lm * sph_harm_no_phase_lm_p \
                                    * np.exp(1j * (m-m_p) * phi)
    C_TT_lmlpmp *= 2*pi**2 * (4*pi)**2 / L**3

    return C_TT_lmlpmp

#@njit(parallel=True)
def get_alm_numba(L, l_max, n_max_list, lm_index, sph_harm_no_phase, list_n_xy_squared, list_n_xyz_squared, delta_k_n, transfer_delta_kl):
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    a_lm = np.zeros(num_l_m, dtype=np.complex128)
    n_max = max(n_max_list)
    for n_x in prange(-n_max, n_max+1):
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

                random_delta_k_n = np.random.normal(loc=0, scale = delta_k_n[j])
                uniform = np.random.uniform(0.0, np.pi*2)
                random_delta_k_n *= np.exp(1j * uniform)
                #random_delta_k_n = delta_k_n[j]

                for l in range(l_max+1):
                    if sqrt(n_squared) > n_max_list[l]:
                        continue
                    delta_k_n_mul_transfer = random_delta_k_n * transfer_delta_kl[j, l]
                    for m in range(l+1):
                        lm_index_cur = lm_index[l, m]
                        
                        sph_harm = phase_list[m] * sph_harm_no_phase[lm_index_cur, np.abs(n_z), i]
                        if n_z < 0: sph_harm *= (-1) ** (l+m)

                        a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm

    for l in range(l_max+1):
        for m in range(l+1):
            lm_index_cur = lm_index[l, m]
            a_lm[lm_index_cur] *= np.power(1j, l)
    
    a_lm *= np.sqrt(2*np.pi**2) * 4*np.pi / np.power(L, 1.5)

    c_l = get_c_l_from_a_lm(a_lm, l_max)
    
    return a_lm, c_l

def get_D_l(c_l):
    return np.array([c_l[l] * l * (l+1) / (2*np.pi) for l in range(c_l.size)])

@njit
def get_idx(l_max, l, m):
    # From the Healpy library. But we copy it here so that Numba can use it
    return m * (2 * l_max + 1 - m) // 2 + l

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

@njit(parallel = True)
def do_integrand_pre_processing(list_n_xyz_squared, scalar_pk_k3, transfer_delta_kl, L, l_max):
    num_xyz_squared = list_n_xyz_squared.size
    integrand = np.zeros((num_xyz_squared, l_max+1, l_max+1))
    for n_squared_index in prange(1, num_xyz_squared):
        n_squared = list_n_xyz_squared[n_squared_index]
        k = 2*pi / L * sqrt(n_squared)
        scalar_pk_k3_cur = scalar_pk_k3[n_squared_index]
        for l in range(2, l_max+1):
            scalar_pk_k3_transfer_delta_kl = scalar_pk_k3_cur * transfer_delta_kl[n_squared_index, l]
            for lp in range(l_max+1):
                integrand[n_squared_index, l, lp] = scalar_pk_k3_transfer_delta_kl * transfer_delta_kl[n_squared_index, lp]
    return integrand









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