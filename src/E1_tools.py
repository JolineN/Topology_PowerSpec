import numpy as np
import numba as nb

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