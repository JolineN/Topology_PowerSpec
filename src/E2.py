from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan
from numba import njit, prange
from numba_progress import ProgressBar

class E2(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.Lx = param['Lx'] * L_LSS
    self.Ly = param['Ly'] * L_LSS
    self.Lz = param['Lz'] * L_LSS
    
    self.x0 = param['x0'] * L_LSS
    if np.linalg.norm(param['x0']) < 1e-6 and np.abs(param['beta'] - 90) < 1e-6  and np.abs(param['alpha'] - 90) < 1e-6:
      self.no_shift = True
    else:
      self.no_shift = False
    print('No shift:', self.no_shift)
    self.beta = param['beta'] * np.pi / 180
    self.alpha = param['alpha'] * np.pi / 180
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta) * sin(self.alpha)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_alpha_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        "{:.2f}".format(param['x0'][0]),
        "{:.2f}".format(param['x0'][1]),
        "{:.2f}".format(param['x0'][2]),
        self.l_max,
        int(param['c_l_accuracy']*100)
    )

    Topology.__init__(self, param, debug, make_run_folder)

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
        transfer_delta_kl,
        tilde_xi = self.tilde_xi,
    )

  def get_kosowsky_per_process(
    self,
    N_s,
    sampled_m_mp_fixed_ell_ellp,
    process_i,
    return_dict,
    min_index,
    max_index,
  ):
    # This function seems unnecessary, but Numba does not allow return_dict
    # which is of type multiprocessing.Manager

    return_dict[process_i] = sample_kosowsky_statistics(
      N_s = N_s,
      sampled_m_mp_fixed_ell_ellp = sampled_m_mp_fixed_ell_ellp,
      min_index = min_index,
      max_index = max_index,
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
      tilde_xi = self.tilde_xi,
      progress = None
    )

  def get_c_lmlpmp_per_process(
    self,
    process_i,
    return_dict,
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
    ell_range,
    ell_p_range
  ):
    # This function seems unnecessary, but Numba does not allow return_dict
    # which is of type multiprocessing.Manager

    return_dict[process_i] = get_c_lmlpmp(
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
        ell_range,
        ell_p_range,
        tilde_xi = self.tilde_xi,
        no_shift = self.no_shift,
        progress = None
    )

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi = get_list_of_k_phi_theta(
      max(self.k_max_list[0, :]),
      self.Lx,
      self.Ly,
      self.Lz,
      self.x0,
      self.beta,
      self.alpha
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    return k_amp, phi, theta
  
  def get_c_lmlpmp_top(self, ell_range, ell_p_range):
    with ProgressBar(total=self.k_amp.size) as progress:
      if ell_p_range.size == 0:
        print('Only calculating diagonal')
        c_lmlpmp = get_c_lmlpmp_diag(
          V=self.V,
          k_amp=self.k_amp,
          theta_unique_index=self.theta_unique_index,
          k_amp_unique_index=self.k_amp_unique_index,
          k_max_list = self.k_max_list[0, :],
          l_max=self.l_max,
          lm_index = self.lm_index,
          sph_harm_no_phase = self.sph_harm_no_phase,
          integrand=self.integrand_TT,
          ell_range = ell_range,
          tilde_xi = self.tilde_xi,
          progress = progress
        )
      else:
        c_lmlpmp = get_c_lmlpmp(
          min_index = 0,
          max_index = self.k_amp.size,
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
          tilde_xi = self.tilde_xi,
          no_shift = self.no_shift,
          progress = progress
        )
    return c_lmlpmp

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, x0, beta, alpha):
    # Returns list of k, phi, and theta for this topology

    sin_b_inv = 1/sin(beta)
    tan_a_inv = 1/tan(alpha)
    sin_a_inv = 1/sin(alpha)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z * 2 / (2*pi))) # Because of eigenmode 1
 
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    tilde_xi = np.ones((n_x_max * n_y_max * n_z_max * 8, 2), dtype=np.complex128)

    T_B = L_z * np.array([cos(beta), 0, sin(beta)])

    cur_index = 0

    # Eigenmode 1
    k_x = 0
    k_y = 0
    # Only n_z = 0 mod 2
    for n_z in range(-n_z_max, n_z_max+1):
      if n_z % 2 == 1 or n_z == 0:
        continue
      k_z = 2*pi * n_z * sin_b_inv / (2* L_z)
      k_xyz = sqrt(k_z**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      tilde_xi[cur_index, :] = exp(- 1j * k_z * x0[2])
      cur_index += 1

    # Eigenmode 2
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = 2*pi * n_x / L_x
      for n_y in range(0, n_y_max+1):
        if n_x == 0 and n_y == 0:
          continue
        k_y = 2*pi * n_y * sin_a_inv/ L_y - k_x * tan_a_inv

        k_xy_squared = k_x**2 + k_y**2
        if k_xy_squared > k_max**2:
          continue
        
        for n_z in range(-n_z_max, n_z_max+1):
          k_z = 2*pi * n_z * sin_b_inv / (2*L_z)
          
          k_xyz = sqrt(k_xy_squared + k_z**2)

          if k_xyz > k_max or k_xyz < 1e-6:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
          for m in range(2):
            tilde_xi[cur_index, m] = exp(- 1j * np.dot(k_vec, x0))/sqrt(2) \
            * (1 + (-1)**m * exp(2*1j*(k_x*x0[0] + k_y*x0[1])) * exp(1j*np.dot(k_vec, T_B)))
          #print(tilde_xi[cur_index, :], n_x, n_y, n_z)
          cur_index += 1

    k_amp = k_amp[:cur_index-1]
    phi = phi[:cur_index-1]   
    theta = theta[:cur_index-1]
    tilde_xi = tilde_xi[:cur_index-1, :]

    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi

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
    transfer_delta_kl,
    tilde_xi
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

              a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm_conj * cur_tilde_xi[m%2]

    return a_lm

@njit(nogil=True, parallel=False, fastmath=True)
def get_c_lmlpmp(
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
    ell_range,
    ell_p_range,
    tilde_xi,
    no_shift,
    progress
    ):

    # This calculates parts of the covariance. If only_diag==True
    # then it only finds the diagonal values, l, m = lp, mp

    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure        

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
        start_lp = 0#l%2 if no_shift else 0
        lp_step = 1#2 if no_shift else 1
        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m - ell_range[0]**2
            sph_cur_index = lm_index[l, np.abs(m)]

            # Do Y_lm^* first
            xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[np.abs(m)]
            if m < 0:
              xi_lm = (-1)**m * np.conjugate(xi_lm)
            # Then tilde xi
            xi_lm *= cur_tilde_xi[m%2]

            
            #for l_p in range(ell_p_range[0], ell_p_range[1]+1):
            for l_p in range(start_lp + ell_p_range[0], ell_p_range[1]+1, lp_step):
              if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
                continue
              integrand_il = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
              
              for m_p in range(-l_p, l_p + 1):
                  lm_p_index_cur = l_p * (l_p + 1) + m_p - ell_p_range[0]**2
                  sph_p_cur_index = lm_index[l_p, np.abs(m_p)]

                  Y_lpmp = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[np.abs(m_p)])
                  if m_p < 0:
                    Y_lpmp = (-1)**m_p * np.conjugate(Y_lpmp)

                  xi_lm_p_conj = np.conjugate(cur_tilde_xi[m_p%2]) * Y_lpmp

                  C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * xi_lm * xi_lm_p_conj
    
      if progress != None: progress.update(1)
    
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel=False, fastmath=True)
def get_c_lmlpmp_diag(
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
            xi_lm_abs *= cur_tilde_xi[m%2]

            C_lmlpmp[lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(xi_lm_abs)**2
           
      progress.update(1)
    
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel = False)
def sample_kosowsky_statistics(
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
        for l_p in range(l%2 + 2, l_max+1, 2):
          if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]) and k_amp_cur > min_k_amp:
            continue
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
            xi_lm *= cur_tilde_xi[m%2]

            sph_p_cur_index = lm_index[l_p, abs_m_p]
            Y_lpmp = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[np.abs(m_p)])
            if m_p < 0:
              Y_lpmp = (-1)**m_p * np.conjugate(Y_lpmp)
            xi_lm_p_conj = np.conjugate(cur_tilde_xi[m_p%2]) * Y_lpmp

            C_lmlpmp[l, l_p, s] += integrand[k_unique_index_cur, l, l_p] * xi_lm * xi_lm_p_conj

      if progress != None: progress.update(1)
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp
