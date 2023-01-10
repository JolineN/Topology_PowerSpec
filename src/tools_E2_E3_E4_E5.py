from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan
from numba import njit, prange
from numba_progress import ProgressBar

@njit(nogil=True, parallel=False)
def E2_E3_E4_E5_get_c_lmlpmp(
    min_index,
    max_index,
    V,
    k_amp, 
    phi,
    theta,
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    lm_index,
    sph_harm_no_phase,
    integrand,
    ell_range,
    ell_p_range,
    tilde_xi,
    tilde_xi_delta_m,
    progress
    ):

    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - ell_p_range[0]**2 
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    

    for i in range(min_index, max_index):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      cur_tilde_xi = tilde_xi[i, :]

      # Powers of i so we can get them as an array call instead of recalculating them every time.
      # 1j**n == ipow[n%4], even for n<0.
      ipow = 1j**np.arange(4)

      for l in range(ell_range[0], ell_range[1]+1):
        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m - ell_range[0]**2
            sph_cur_index = lm_index[l, np.abs(m)]

            # Do Y_lm^* first
            conj_Y_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[np.abs(m)]
            if m < 0:
              conj_Y_lm = (-1)**m * np.conjugate(conj_Y_lm)
            # Then tilde xi
            xi_lm = cur_tilde_xi[m % tilde_xi_delta_m] * conj_Y_lm

            for l_p in range(ell_p_range[0], ell_p_range[1]+1):
              if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
                continue
              integrand_il = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
              
              for m_p in range(-l_p, l_p + 1):
                  lm_p_index_cur = l_p * (l_p + 1) + m_p - ell_p_range[0]**2
                  sph_p_cur_index = lm_index[l_p, np.abs(m_p)]

                  Y_lpmp = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[np.abs(m_p)])
                  if m_p < 0:
                    Y_lpmp = (-1)**m_p * np.conjugate(Y_lpmp)

                  xi_lm_p_conj = np.conjugate(cur_tilde_xi[m_p % tilde_xi_delta_m]) * Y_lpmp

                  C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * xi_lm * xi_lm_p_conj
    
      if progress != None: progress.update(1)
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel = False)
def E2_E3_E4_E5_sample_kosowsky_statistics(
      N_s,
      sampled_m_mp_fixed_ell_ellp,
      min_index,
      max_index,
      V,
      k_amp, 
      phi, 
      theta_unique_index,
      k_amp_unique_index,
      k_max_list, 
      l_max,
      lm_index,
      sph_harm_no_phase,
      integrand,
      tilde_xi,
      tilde_xi_delta_m,
      progress
    ):

    # This calculates only the diagonal of the covariance.
    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    C_lmlpmp = np.zeros((l_max+1, l_max+1, N_s), dtype=np.complex128)    

    min_k_amp = np.min(k_amp)

    for i in range(min_index, max_index):
      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]
      cur_tilde_xi = tilde_xi[i, :]

      for l in range(2, l_max+1):
        for l_p in range(l, l_max+1):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]) and k_amp_cur > min_k_amp:
            continue
          if l == l_p:
            num_m_m_p = (2*l+1)*l
          else:
            num_m_m_p = (2*l+1)*(2*l_p+1)

          for s in range(N_s if num_m_m_p > N_s else num_m_m_p):
            m = sampled_m_mp_fixed_ell_ellp[l, l_p, s, 0]
            m_p = sampled_m_mp_fixed_ell_ellp[l, l_p, s, 1]
            # Dont do diagonal
            if l == l_p and m == m_p:
              continue
            
            abs_m = np.abs(m)
            abs_m_p = np.abs(m_p)

            sph_cur_index = lm_index[l, abs_m]
            xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[np.abs(m)]
            if m < 0:
              xi_lm = (-1)**m * np.conjugate(xi_lm)
            # Then tilde xi
            xi_lm *= cur_tilde_xi[m%tilde_xi_delta_m]

            sph_p_cur_index = lm_index[l_p, abs_m_p]
            Y_lpmp = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[np.abs(m_p)])
            if m_p < 0:
              Y_lpmp = (-1)**m_p * np.conjugate(Y_lpmp)
            xi_lm_p_conj = np.conjugate(cur_tilde_xi[m_p%tilde_xi_delta_m]) * Y_lpmp

            C_lmlpmp[l, l_p, s] += integrand[k_unique_index_cur, l, l_p] * xi_lm * xi_lm_p_conj

      if progress != None: progress.update(1)
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit()
def E2_E3_E4_E5_get_alm_per_process_numba(
    min_index,
    max_index,
    k_amp,
    phi,
    k_amp_unique_index,
    theta_unique_index,
    k_max_list,
    l_max,
    lm_index,
    sph_harm_no_phase,
    delta_k_n,
    transfer_delta_kl,
    tilde_xi,
    tilde_xi_delta_m
): 
    # This function returns parts of the summation over wavenumber k to get a_lm
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    a_lm = np.zeros(num_l_m, dtype=np.complex128)
    for i in range(min_index, max_index):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]
      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      random_delta_k_n = np.random.normal(loc=0, scale = delta_k_n[k_unique_index_cur])
      uniform = np.random.uniform(0.0, np.pi*2)
      random_delta_k_n *= np.exp(1j * uniform)

      cur_tilde_xi = tilde_xi[i, :]

      for l in range(2, l_max+1):
          if k_amp_cur > k_max_list[l]:
              continue
          delta_k_n_mul_transfer = random_delta_k_n * transfer_delta_kl[k_unique_index_cur, l]
          for m in range(l+1):
              lm_index_cur = lm_index[l, m]
              
              sph_harm_conj = phase_list[m] * sph_harm_no_phase[sph_harm_index, lm_index_cur]

              a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm_conj * cur_tilde_xi[m%tilde_xi_delta_m]

    return a_lm

@njit(nogil=True, parallel=False, fastmath=True)
def E2_E3_E4_E5_get_c_lmlpmp_diag(
    V,
    k_amp,
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    lm_index,
    sph_harm_no_phase,
    integrand,
    ell_range,
    tilde_xi,
    tilde_xi_delta_m,
    progress
    ):

    # This calculates parts of the covariance. If only_diag==True
    # then it only finds the diagonal values, l, m = lp, mp

    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    num_l_m = l_max * (l_max+1) + l_max + 1
    C_lmlpmp = np.zeros(num_l_m, dtype=np.complex128)

    num_indices = k_amp.size
    for i in range(num_indices):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]
      cur_tilde_xi = tilde_xi[i, :]

      for l in range(ell_range[0], ell_range[1]+1):
        if k_amp_cur > k_max_list[l]:
            continue

        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m
            sph_cur_index = lm_index[l, np.abs(m)]

            # Do Y_lm^* first
            xi_lm_abs = sph_harm_no_phase[sph_harm_index, sph_cur_index]
            # Then tilde xi
            xi_lm_abs *= cur_tilde_xi[m % tilde_xi_delta_m]

            C_lmlpmp[lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(xi_lm_abs)**2
           
      progress.update(1)
    
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp