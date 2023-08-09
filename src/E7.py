from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, cos, tan, exp, sqrt
from numba import njit, prange
from numba_progress import ProgressBar

class E7(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.Lx = param['Lx'] * L_LSS
    self.Ly = param['Ly'] * L_LSS
    self.Lz = param['Lz'] * L_LSS
    self.alpha = param['alpha'] * np.pi / 180
    self.beta = param['beta'] * np.pi / 180
    self.gamma = param['gamma'] * np.pi / 180

    L_1x = self.Lx * cos(self.alpha)
    L_1y = self.Lx * sin(self.alpha)
    L_2y = -L_1y
    L_Bz = self.Lz * sin(self.beta)

    self.V = np.abs(L_1x * (L_1y - L_2y) * L_Bz)
    self.l_max = param['l_max']
    self.param = param
    
    self.x0 = param['x0'] * L_LSS

    self.l_max = param['l_max']
    self.param = param

    print('Running - E7 l_max={}, Lx={}'.format(self.l_max, int(self.Lx)))
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
          transfer_delta_kl
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
        eigenmode_index_split = self.eigenmode_index_split,
        progress = None
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
      eigenmode_index_split = self.eigenmode_index_split,
      progress = None
    )

  def get_c_lmlpmp_top(self, ell_range, ell_p_range):
    #ncpus = multiprocessing.cpu_count()        
    #os.environ['OMP_NUM_THREADS'] = str(ncpus)

    with ProgressBar(total=self.k_amp.size) as progress:
      if ell_p_range.size == 0:
        print('Only calculating diagonal')
        c_lmlpmp = get_c_lmlpmp_diag(
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
          tilde_xi = self.tilde_xi,
          eigenmode_index_split = self.eigenmode_index_split,
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
          eigenmode_index_split = self.eigenmode_index_split,
          progress = progress
        )
    return c_lmlpmp

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi, eigenmode_index_split = get_list_of_k_phi_theta(max(self.k_max_list[0, :]), self.Lx, self.Ly, self.Lz, self.beta, self.alpha, self.gamma, self.x0)

    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    self.eigenmode_index_split = eigenmode_index_split
    return k_amp, phi, theta

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, beta, alpha, gamma, x0):
    # Returns list of k, phi, and theta for this topology

    sin_b_inv = 1/sin(beta)
    cot_b = 1/tan(beta)

    L_1x = L_x * cos(alpha)
    L_1y = L_x * sin(alpha)
    L_2y = -L_1y

    n_x_max = int(np.ceil(k_max * L_1x * 2/ (2*pi)))
    n_y_max = int(np.ceil(k_max * (L_1y - L_2y) / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z / (2*pi))) # Because of eigenmode 1
 
    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 2), dtype=np.complex128)

    #L_Bz = L_z * sin(beta)
    #L_Bx = L_z * cos(beta)

    M_A_E7 = np.array([
      [1.0, 0.0, 0.0],
      [0.0, -1.0, 0.0],
      [0.0, 0.0, 1.0]
    ])
    M_A_E7_dot_x_0 = np.dot(M_A_E7, x0)

    T_A1_E7 = np.array([L_1x, L_1y, 0.0])

    cur_index = 0

    # Eigenmode 1
    # n_y = 0
    # Sum over all n_x and n_z
    k_y = 0
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = 2 * np.pi * n_x/(2 * L_1x)
      
      for n_z in range(-n_z_max, n_z_max+1):
        if n_x == 0 and n_z == 0:
          continue

        k_z = 2*pi * n_z * sin_b_inv / L_z - cot_b * k_x
        k_xyz = sqrt(k_x**2 + k_z**2)
        if k_xyz > k_max:
          continue
        
        k_amp[cur_index] = k_xyz
        cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
        phi[cur_index] = cur_phi
        theta[cur_index] = cur_theta
        k_vec = np.array([k_x, k_y, k_z])
        tilde_xi[cur_index, :] = exp(- 1j * np.dot(k_vec, x0))
        cur_index += 1
    print('Eigenmode 1:', cur_index)

    eigenmode_index_split = cur_index
    
    # Eigenmode 2
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = 2 * np.pi * n_x/(2 * L_1x)
      for n_y in range(1, n_y_max+1):
        k_y =  2 * np.pi * n_y/(L_1y - L_2y)

        k_xy_squared = k_x**2 + k_y**2
        if k_xy_squared > k_max**2:
          continue
        
        for n_z in range(-n_z_max, n_z_max+1):
          k_z = 2*pi * n_z * sin_b_inv / L_z - cot_b * k_x
          
          k_xyz = sqrt(k_xy_squared + k_z**2)
          if k_xyz > k_max:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
        
          tilde_xi[cur_index, 0] = exp(- 1j * np.dot(k_vec, x0))
          tilde_xi[cur_index, 1] = np.exp(-1j * np.dot(k_vec, M_A_E7_dot_x_0)) * np.exp(1j * np.dot(k_vec, T_A1_E7))
          tilde_xi[cur_index, :] *= 1 / np.sqrt(2)

          cur_index += 1
    print('Eigenmode 2:', cur_index)
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :]

    print('E7 Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, eigenmode_index_split

@njit(nogil=True, parallel = False)
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
    eigenmode_index_split,
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
        for m in range(-l, l + 1):
          lm_index_cur = l * (l+1) + m - ell_range[0]**2
          abs_m = np.abs(m)
          sph_cur_index = lm_index[l, abs_m]

          if i <= eigenmode_index_split:
            # First eigenmode
            conj_Y_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[np.abs(m)]
            if m < 0:
              conj_Y_lm = (-1)**m * np.conjugate(conj_Y_lm)
            # Then tilde xi
            xi_lm = cur_tilde_xi[0] * conj_Y_lm
          else:
            # Second eigenmode
            phase_1 = phase_list[abs_m] if m >= 0 else np.conjugate(phase_list[abs_m])
            phase_2 = np.conjugate(phase_1)

            xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * (cur_tilde_xi[0]*phase_1 + cur_tilde_xi[1]*phase_2)
            
            # Y_lm = (-1)^m Y^*_{l, -m} - The complex conjugate part of this equation was done above. Here we do the (-1)^m part
            if m < 0 and m%2 == 1: xi_lm *= -1

          for l_p in range(ell_p_range[0], ell_p_range[1]+1):
              if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
                continue
              integrand_il = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
              
              for m_p in range(-l_p, l_p + 1):
                  lm_p_index_cur = l_p * (l_p + 1) + m_p - ell_p_range[0]**2
                  abs_m_p = np.abs(m_p)
                  sph_p_cur_index = lm_index[l_p, abs_m_p]

                  if i <= eigenmode_index_split:
                    # First eigenmode
                    Y_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[abs_m_p])
                    if m_p < 0:
                      Y_lm_p = (-1)**m_p * np.conjugate(Y_lm_p)
                    # Then tilde xi
                    conj_xi_lm_p = np.conjugate(cur_tilde_xi[0]) * Y_lm_p
                  else:
                    # Second eigenmode
                    phase_1 = phase_list[abs_m_p] if m_p >= 0 else np.conjugate(phase_list[abs_m_p])
                    phase_2 = np.conjugate(phase_1)

                    xi_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * (cur_tilde_xi[0]*phase_1 + cur_tilde_xi[1]*phase_2)
                    if m_p < 0 and m_p%2 == 1: xi_lm_p *= -1

                    conj_xi_lm_p = np.conjugate(xi_lm_p)
                    
                  C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * xi_lm * conj_xi_lm_p

      if progress != None: progress.update(1)

    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel = False)
def get_c_lmlpmp_diag(
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
    tilde_xi,
    eigenmode_index_split,
    progress
    ):

    # This calculates only the diagonal of the covariance.
    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    num_l_m = l_max * (l_max+1) + l_max + 1
    C_lmlpmp = np.zeros(num_l_m, dtype=np.complex128)    
    num_indices = k_amp.size

    min_k_amp = np.min(k_amp)
    for i in prange(num_indices):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)
      cur_tilde_xi = tilde_xi[i, :]

      for l in range(ell_range[0], ell_range[1]+1):
        if k_amp_cur > k_max_list[l] and k_amp_cur > min_k_amp:
            continue
            
        for m in range(-l, l + 1):
          abs_m = np.abs(m)
          lm_index_cur = l * (l+1) + m
          sph_cur_index = lm_index[l, abs_m]

          if i <= eigenmode_index_split:
            # First eigenmode
            conj_Y_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[abs_m]
            
            # Dont need this since we are doing absolute value
            #if m < 0:
            #  conj_Y_lm = (-1)**m * np.conjugate(conj_Y_lm)
            # Then tilde xi
            xi_lm = cur_tilde_xi[0] * conj_Y_lm
          else:
            # Second eigenmode
            phase_1 = phase_list[abs_m] if m >= 0 else np.conjugate(phase_list[abs_m])
            phase_2 = np.conjugate(phase_1)

            xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * (cur_tilde_xi[0]*phase_1 + cur_tilde_xi[1]*phase_2)
            
            # Dont need this since we are doing absolute value
            #if m < 0 and m%2 == 1: xi_lm *= -1


          C_lmlpmp[lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(xi_lm)**2
            
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
    eigenmode_index_split,
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

    for l in range(2, l_max+1):
      for l_p in range(l, l_max+1, 2):
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
          sph_harm_lm_conj = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[abs_m]
          if m<0:
            sph_harm_lm_conj = np.conjugate(sph_harm_lm_conj)

          integrand_il_sphm = integrand[k_unique_index_cur, l, l_p] * sph_harm_lm_conj
          
          sph_p_cur_index = lm_index[l_p, abs_m_p]
          sph_harm_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[abs_m_p])
          if m_p<0:
            sph_harm_lm_p = np.conjugate(sph_harm_lm_p)

          C_lmlpmp[l, l_p, s] += integrand_il_sphm * sph_harm_lm_p

    if progress != None: progress.update(1)
  C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
  return C_lmlpmp