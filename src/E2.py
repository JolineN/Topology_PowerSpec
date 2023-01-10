from .topology import Topology
from .tools import *
from .tools_E2_E3_E4_E5 import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan
from numba import njit
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
    self.gamma = param['gamma'] * np.pi / 180
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta) * sin(self.alpha)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_alpha_{}_gamma_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        int(param['gamma']),
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
    return_dict[process_i] = E2_E3_E4_E5_get_alm_per_process_numba(
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
        tilde_xi_delta_m = 2
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

    return_dict[process_i] = E2_E3_E4_E5_sample_kosowsky_statistics(
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
      tilde_xi_delta_m = 2,
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

    return_dict[process_i] = E2_E3_E4_E5_get_c_lmlpmp(
        min_index,
        max_index,
        V,
        k_amp, 
        phi,
        self.theta,
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
        tilde_xi_delta_m = 2,
        progress = None
    )

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi = get_list_of_k_phi_theta(
      k_max = max(self.k_max_list[0, :]),
      L_x = self.Lx,
      L_y = self.Ly,
      L_z = self.Lz,
      x0 = self.x0,
      beta = self.beta,
      alpha = self.alpha,
      gamma = self.gamma
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    return k_amp, phi, theta
  
  def get_c_lmlpmp_top(self, ell_range, ell_p_range):
    with ProgressBar(total=self.k_amp.size) as progress:
      if ell_p_range.size == 0:
        print('Only calculating diagonal')
        c_lmlpmp = E2_E3_E4_E5_get_c_lmlpmp_diag(
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
          tilde_xi_delta_m = 2,
          progress = progress
        )
    return c_lmlpmp

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, x0, beta, alpha, gamma):
    # Returns list of k, phi, and theta for this topology

    sin_b_inv = 1/sin(beta)
    tan_a_inv = 1/tan(alpha)
    sin_a_inv = 1/sin(alpha)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))*2
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))*2
    n_z_max = int(np.ceil(k_max * L_z * 2 / (2*pi)))*2 # Because of eigenmode 1
 
    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 2), dtype=np.complex128)

    T_B = L_z * np.array([cos(beta)*cos(gamma), cos(beta)*sin(gamma), sin(beta)])

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
    print('Eigenmode 1:', cur_index)
    
    # Eigenmode 2
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = 2*pi * n_x / L_x
      for n_y in range(0, n_y_max+1):
        if n_x <= 0 and n_y == 0:
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
            #tilde_xi[cur_index, m] = (1 + (-1)**(m+n_z))/sqrt(2)
            tilde_xi[cur_index, m] = exp(- 1j * np.dot(k_vec, x0))/sqrt(2) \
            * (1 + (-1)**m * exp(2*1j*(k_x*x0[0] + k_y*x0[1])) * exp(1j*np.dot(k_vec, T_B)))

          cur_index += 1
    print('Eigenmode 2:', cur_index)
    k_amp = k_amp[:cur_index-1]
    phi = phi[:cur_index-1]   
    theta = theta[:cur_index-1]
    tilde_xi = tilde_xi[:cur_index-1, :]

    print('E2 Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi