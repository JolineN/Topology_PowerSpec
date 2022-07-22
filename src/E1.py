from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, tan, sqrt
from numba import njit, prange
from numba_progress import ProgressBar
import multiprocessing
import os

class E1(Topology):
  def __init__(self, param, debug=True):
    L_LSS = 13824.9 * 2
    self.Lx = param['Lx'] * L_LSS
    self.Ly = param['Ly'] * L_LSS
    self.Lz = param['Lz'] * L_LSS
    self.alpha = param['alpha'] * np.pi / 180
    self.beta = param['beta'] * np.pi / 180
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta) * sin(self.alpha)
    self.l_max = param['l_max']
    self.param = param

    if param['Lx'] == param['Ly'] and param['Ly'] == param['Lz'] and param['beta'] == 90 and param['alpha'] == 90:
      print('Cubic Torus')
      self.cubic = True
    else:
      self.cubic = False

    print('Running - E1 l_max={}, Lx={}'.format(self.l_max, int(self.Lx)))
    self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_alpha_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        self.l_max,
        int(param['c_l_accuracy']*100)
    )

    Topology.__init__(self, param, debug)

  def get_alm_per_process(
      self,
      process_i,
      return_dict,
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
      transfer_delta_kl
  ):
      # This function seems unnecessary, but Numba does not allow return_dict
      # which is of type multiprocessing.Manager
      return_dict[process_i] = get_alm_per_process_numba(
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
          transfer_delta_kl
      )

  def get_c_lmlpmp_top(self, ell_range, ell_p_range):
    #ncpus = multiprocessing.cpu_count()        
    #os.environ['OMP_NUM_THREADS'] = str(ncpus)

    with ProgressBar(total=self.k_amp.size) as progress:
      c_lmlpmp = get_c_lmlpmp(
        V=self.V,
        k_amp=self.k_amp,
        phi=self.phi,
        theta_unique_index=self.theta_unique_index,
        k_amp_unique_index=self.k_amp_unique_index,
        k_max_list = self.k_max_list[0, :],
        l_max=self.l_max,
        lm_index = self.lm_index,
        sph_harm_no_phase = self.sph_harm_no_phase,
        integrand=self.integrand_TT,
        ell_range = ell_range,
        ell_p_range = ell_p_range,
        cubic = self.cubic,
        progress = progress
      )
    return c_lmlpmp


  def get_list_of_k_phi_theta(self):
    return get_list_of_k_phi_theta(max(self.k_max_list[0, :]), self.Lx, self.Ly, self.Lz, self.beta, self.alpha)

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, beta, alpha):
    # Returns list of k, phi, and theta for this topology

    tan_b_inv = 1/tan(beta)
    sin_b_inv = 1/sin(beta)
    tan_a_inv = 1/tan(alpha)
    sin_a_inv = 1/sin(alpha)
    print('inv sin and tan', sin_b_inv, tan_b_inv)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z / (2*pi)))
 
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    cur_index = 0

    for n_x in prange(-n_x_max, n_x_max + 1):
      k_x = 2*pi * n_x / L_x

      for n_z in range(-n_z_max, n_z_max+1):
        
        k_z = 2*pi * n_z * sin_b_inv / L_z - k_x * tan_b_inv


        k_xz_squared = k_x**2 + k_z**2

        if k_xz_squared > k_max**2:
          continue

        for n_y in range(-n_y_max, n_y_max+1):
          k_y = 2*pi * n_y * sin_a_inv / L_y - k_x * tan_a_inv


          k_xyz = sqrt(k_xz_squared + k_y**2)
          if k_xyz > k_max or k_xyz < 1e-5:
            continue

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
          cur_index += 1

    k_amp = k_amp[:cur_index-1]
    phi = phi[:cur_index-1]   
    theta = theta[:cur_index-1]

    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta

@njit(fastmath=True)
def get_alm_per_process_numba(
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
    transfer_delta_kl
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

      for l in range(2, l_max+1):
          if k_amp_cur > k_max_list[l]:
              continue
          delta_k_n_mul_transfer = random_delta_k_n * transfer_delta_kl[k_unique_index_cur, l]
          for m in range(l+1):
              lm_index_cur = lm_index[l, m]
              
              sph_harm = phase_list[m] * sph_harm_no_phase[sph_harm_index, lm_index_cur]

              a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm

    return a_lm

@njit(nogil=True, fastmath=True, parallel = False)
def get_c_lmlpmp(
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
    ell_range,
    ell_p_range,
    cubic,
    progress
    ):

    # This calculates parts of the covariance. If only_diag==True
    # then it only finds the diagonal values, l, m = lp, mp

    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    only_diag = False
    if ell_p_range.size == 0:
        print('Only doing diagonal')
        # Only do diagonal
        only_diag = True

        #To make Numba compile
        ell_p_range = np.array([0, 0])

    num_l_m = l_max * (l_max+1) + l_max + 1
    C_lmlpmp = np.zeros((num_l_m, num_l_m), dtype=np.complex128)

    num_indices = k_amp.size

    min_k_amp = np.min(k_amp)
    for i in range(num_indices):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      for l in range(ell_range[0], ell_range[1]+1):
        if k_amp_cur > k_max_list[l] and k_amp_cur > min_k_amp:
            continue
            
        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m
            sph_cur_index = lm_index[l, np.abs(m)]

            sph_harm_lm_conj = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[np.abs(m)]
            if m<0:
              sph_harm_lm_conj = (-1)**m * np.conjugate(sph_harm_lm_conj)

            if only_diag:
                C_lmlpmp[lm_index_cur, lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(sph_harm_lm_conj)**2
            else:
                # Only do l-lp = 0 mod 2. Holds for general torus
                for l_p in range(l%2 + ell_p_range[0], ell_p_range[1]+1, 2):
                    
                    integrand_il_sphm = integrand[k_unique_index_cur, l, l_p] * np.power(1j, l-l_p) * sph_harm_lm_conj
                    
                    # Negative m_p are calculated later
                    # Only do m-mp = 0 mod 2
                    if cubic:
                      start_m = m%4
                      m_step = 4
                    else:
                      start_m = m%2
                      m_step = 2
                    for m_p in range(start_m, l_p + 1, m_step):
                        lm_p_index_cur = l_p * (l_p+1) + m_p
                        sph_p_cur_index = lm_index[l_p, m_p]

                        sph_harm_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[m_p])

                        C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il_sphm * sph_harm_lm_p
      progress.update(1)
    
    # Negative m_p are calculated here using symmetry arguments for general torus
    for l in range(ell_range[0], ell_range[1]+1):
        for m in range(-l, l + 1):
          lm_index_cur = l * (l+1) + m
          new_lm_index = l * (l+1) - m

          for l_p in range(ell_p_range[0], ell_p_range[1]+1):
              for m_p in range(-l_p, 0):
                lm_p_index_cur = l_p * (l_p+1) + m_p
                new_lm_p_index_cur = l_p * (l_p+1) - m_p

                C_lmlpmp[lm_index_cur, lm_p_index_cur] = C_lmlpmp[new_lm_index, new_lm_p_index_cur]

    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp