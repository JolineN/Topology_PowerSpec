from .topology import Topology
from .tools import *
import numpy as np
import healpy as hp
from numpy import pi, sin, cos, exp, sqrt, tan
from numba import njit, prange
from numba_progress import ProgressBar
import pyshtools as pysh

class E6(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.LAx = param['Lx'] * L_LSS
    self.LBy = param['Ly'] * L_LSS
    self.LCz = param['Lz'] * L_LSS

    self.LCx = self.LAx
    self.LAy = self.LBy
    self.LBz = self.LCz

    self.LAz = 0.0
    self.LBx = 0.0
    self.LCy = 0.0

    tilde_P_x = (self.LBx - self.LCx) / (2*self.LAx)
    tilde_P_y = (self.LCy - self.LAy) / (2*self.LBy)
    tilde_P_z = (self.LAz - self.LBz) / (2*self.LCz)

    assert(np.isclose(tilde_P_x-0.5, np.rint(tilde_P_x-0.5)))
    assert(np.isclose(tilde_P_y-0.5, np.rint(tilde_P_y-0.5)))
    assert(np.isclose(tilde_P_z-0.5, np.rint(tilde_P_z-0.5)))

    alpha_x = (self.LBx + self.LCx) / (2*self.LAx)
    alpha_y = (self.LCy + self.LAy) / (2*self.LBy)
    alpha_z = (self.LAz + self.LBz) / (2*self.LCz)

    self.alpha_x = alpha_x
    self.alpha_y = alpha_y
    self.alpha_z = alpha_z

    self.V = self.LAx * self.LBy * self.LCz / 8 * np.abs(
      1 + \
      alpha_x * (tilde_P_y - tilde_P_z + 2 * tilde_P_y * tilde_P_z) + \
      alpha_y * (tilde_P_z - tilde_P_x + 2 * tilde_P_z * tilde_P_x) + \
      alpha_z * (tilde_P_x - tilde_P_y + 2 * tilde_P_x * tilde_P_y) + \
      - alpha_x*alpha_y - alpha_y*alpha_z - alpha_z*alpha_x + 2*alpha_x*alpha_y*alpha_z
    )**3
    print(self.V / L_LSS**3)

    l_max = param['l_max']
    lm_index = np.zeros((l_max+1, l_max+1), dtype=int)
    for l in range(l_max+1):
        for m in range(l+1):
            cur_index = hp.Alm.getidx(l_max, l, m)
            lm_index[l, m] = cur_index
    
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    sph_harm_no_phase_theta_0_pi_over_2 = np.zeros((2, num_l_m), dtype=np.float64)

    all_sph_harm_theta_0_no_phase = np.real(pysh.expand.spharm(l_max, 0, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
    all_sph_harm_theta_pi_over_2_no_phase = np.real(pysh.expand.spharm(l_max, pi/2, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
    for l in range(l_max+1):
        for m in range(l+1):
            cur_index = lm_index[l, m]
            sph_harm_no_phase_theta_0_pi_over_2[0, cur_index] = all_sph_harm_theta_0_no_phase[0, l, m]
            sph_harm_no_phase_theta_0_pi_over_2[1, cur_index] = all_sph_harm_theta_pi_over_2_no_phase[0, l, m]  
    self.sph_harm_no_phase_theta_0_pi_over_2 = sph_harm_no_phase_theta_0_pi_over_2

    self.x0 = param['x0'] * L_LSS
    if np.linalg.norm(param['x0']) < 1e-6 and np.abs(param['beta'] - 90) < 1e-6  and np.abs(param['alpha'] - 90) < 1e-6:
      self.no_shift = True
    else:
      self.no_shift = False
    print('No shift:', self.no_shift)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_LAx_{}_LBy_{}_LCz_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
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
        eigenmode_index_split = self.eigenmode_index_split
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
        no_shift = self.no_shift,
        sph_harm_no_phase_theta_0_pi_over_2 = self.sph_harm_no_phase_theta_0_pi_over_2,
        progress = None
    )

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi, eigenmode_index_split = get_list_of_k_phi_theta(
      max(self.k_max_list[0, :]),
      self.LAx,
      self.LBy,
      self.LCz,
      self.alpha_x,
      self.alpha_y,
      self.alpha_z,
      self.x0,
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    self.eigenmode_index_split = eigenmode_index_split
    return k_amp, phi, theta
  
  def get_c_lmlpmp_top(self, ell_range, ell_p_range):
    with ProgressBar(total=self.k_amp.size) as progress:
      if ell_p_range.size == 0:
        print('Only calculating diagonal')
        c_lmlpmp = get_c_lmlpmp_diag(
          V=self.V,
          k_amp=self.k_amp,
          phi = self.phi,
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
          sph_harm_no_phase_theta_0_pi_over_2 = self.sph_harm_no_phase_theta_0_pi_over_2,
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
          sph_harm_no_phase_theta_0_pi_over_2 = self.sph_harm_no_phase_theta_0_pi_over_2,
          progress = progress
        )
    return c_lmlpmp

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_Ax, L_By, L_Cz, alpha_x, alpha_y, alpha_z, x0):
    # Returns list of k, phi, and theta for this topology

    n_x_max = int(np.ceil(k_max * L_Ax / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_By / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_Cz / (2*pi)))

    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    # First index is i, second is l%2, third is m%2. The fourth index splits the fourth eigenmode into two pieces
    tilde_xi = np.zeros((n_x_max * n_y_max * n_z_max * 8, 2, 2, 2), dtype=np.complex128)

    T_A = 0.5 * np.array([L_Ax, L_By, 0])
    T_B = 0.5 * np.array([0, L_By, L_Cz])
    T_C = 0.5 * np.array([L_Ax, 0, L_Cz])

    M_A_minus_id = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ], dtype=np.float64) - np.identity(3)

    M_B_minus_id = np.array([
      [-1, 0, 0],
      [0, 1, 0],
      [0, 0, -1]
    ], dtype=np.float64) - np.identity(3)

    M_C_minus_id = np.array([
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1]
    ], dtype=np.float64) - np.identity(3)

    cur_index = 0
    eigenmode_index_split = np.zeros(3)

    # Eigenmode 1
    k_y = 0
    k_z = 0
    for n_x in range(2, n_x_max+1, 2):
      k_x = 2*pi * n_x / L_Ax
      k_xyz = sqrt(k_x**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_x * x0[0]) + (-1)**l_mod_2 * exp(1j * k_x * x0[0]) * exp(1j * 2*pi*n_x*alpha_x))
      cur_index += 1
    eigenmode_index_split[0] = cur_index

    # Eigenmode 2
    k_x = 0
    k_z = 0
    for n_y in range(2, n_y_max+1, 2):
      k_y = 2*pi * n_y / L_By
      k_xyz = sqrt(k_y**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_y * x0[1]) + (-1)**l_mod_2 * exp(1j * k_y * x0[1]) * exp(1j * 2*pi*n_y*alpha_y))
      cur_index += 1
    eigenmode_index_split[1] = cur_index

    # Eigenmode 3
    k_x = 0
    k_y = 0
    for n_z in range(2, n_z_max+1, 2):
      k_z = 2*pi * n_z / L_Cz
      k_xyz = sqrt(k_z**2)
      if k_xyz > k_max:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      for l_mod_2 in range(2):
        tilde_xi[cur_index, l_mod_2, :, :] = 1/sqrt(2) * (exp(- 1j * k_z * x0[2]) + (-1)**l_mod_2 * exp(1j * k_z * x0[2]) * exp(1j * 2*pi*n_z*alpha_z))
      cur_index += 1
    eigenmode_index_split[2] = cur_index

    # Eigenmode 4
    for n_x in range(-n_x_max, n_x_max+1):
      k_x = 2*pi * n_x / L_Ax

      if n_x < 0:
        n_y_start = -n_y_max
        n_y_end = -1

        n_z_start = -n_z_max
        n_z_end = 0
      elif n_x > 0:
        n_y_start = 1
        n_y_end = n_y_max

        n_z_start = 0
        n_z_end = n_z_max
      else:
        continue

      for n_y in range(n_y_start, n_y_end+1):
        k_y = 2*pi * n_y / L_By

        k_xy_squared = k_x**2 + k_y**2
        if k_xy_squared > k_max**2:
          continue
        
        for n_z in range(n_z_start, n_z_end+1):
          k_z = 2*pi * n_z / L_Cz
          
          k_xyz = sqrt(k_xy_squared + k_z**2)

          if k_xyz > k_max or k_xyz < 1e-6:
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
         
          for l_mod_2 in range(2):
            for m_mod_2 in range(2):
              tilde_xi[cur_index, l_mod_2, m_mod_2, 0] = 1 + (-1)**m_mod_2 * exp(-1j * np.dot(k_vec, np.dot(M_C_minus_id, x0))) * exp(1j * np.dot(k_vec, T_C))
              tilde_xi[cur_index, l_mod_2, m_mod_2, 1] = (-1)**(l_mod_2+m_mod_2) * exp(-1j * np.dot(k_vec, np.dot(M_A_minus_id, x0))) * exp(1j * np.dot(k_vec, T_A)) + (-1)**m_mod_2 * exp(-1j * np.dot(k_vec, np.dot(M_B_minus_id, x0))) * exp(1j * np.dot(k_vec, T_B))
          tilde_xi[cur_index, :, :, :] *= exp(- 1j * np.dot(k_vec, x0))/2
          #print(np.sum(tilde_xi[cur_index, 0, 0, :]), tilde_xi[cur_index, 0, 0, 0], tilde_xi[cur_index, 0, 0, 1], n_x, n_y, n_z, )
          cur_index += 1
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :, :, :]

    print(cur_index, eigenmode_index_split, 'split')
    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, eigenmode_index_split

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
    eigenmode_index_split,
    no_shift,
    sph_harm_no_phase_theta_0_pi_over_2,
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

      m_list = np.arange(-l_max, l_max+1)
      phase_list = np.exp(1j * phi[i] * m_list)

      cur_tilde_xi = tilde_xi[i, :, :, :]

      # Powers of i so we can get them as an array call instead of recalculating them every time.
      # 1j**n == ipow[n%4], even for n<0.
      ipow = 1j**np.arange(4)

      for l in range(ell_range[0], ell_range[1]+1):
        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m - ell_range[0]**2
            sph_cur_index = lm_index[l, np.abs(m)]

            if i <= eigenmode_index_split[0]:
              # xi depends on Y^*
              # xi^* depends on Y
              xi_lm = sph_harm_no_phase_theta_0_pi_over_2[1, sph_cur_index] * cur_tilde_xi[l%2, m%2, 0]
            elif i <= eigenmode_index_split[1]:
              xi_lm = sph_harm_no_phase_theta_0_pi_over_2[1, sph_cur_index] * exp(-1j * m * pi/2) * cur_tilde_xi[l%2, m%2, 0]
            elif i <= eigenmode_index_split[2]:
              if m == 0:
                xi_lm = sph_harm_no_phase_theta_0_pi_over_2[0, sph_cur_index] * cur_tilde_xi[l%2, m%2, 0]
              else:
                xi_lm = 0
            else:
              xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * (phase_list[-m + l_max] * cur_tilde_xi[l%2, m%2, 0] + phase_list[m + l_max] * cur_tilde_xi[l%2, m%2, 1])
            
            #for l_p in range(ell_p_range[0], ell_p_range[1]+1):
            for l_p in range(ell_p_range[0], ell_p_range[1]+1):
              if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
                continue
              integrand_il = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4]
              
              for m_p in range(-l_p, l_p + 1):
                  lm_p_index_cur = l_p * (l_p + 1) + m_p - ell_p_range[0]**2
                  sph_p_cur_index = lm_index[l_p, np.abs(m_p)]

                  if i <= eigenmode_index_split[0]:
                    # xi depends on Y^*
                    # xi^* depends on Y
                    xi_lm_p_conj = sph_harm_no_phase_theta_0_pi_over_2[1, sph_p_cur_index] * cur_tilde_xi[l_p%2, m_p%2, 0]
                  elif i <= eigenmode_index_split[1]:
                    xi_lm_p_conj = sph_harm_no_phase_theta_0_pi_over_2[1, sph_p_cur_index] * exp(-1j * m_p * pi/2) * cur_tilde_xi[l_p%2, m_p%2, 0]
                  elif i <= eigenmode_index_split[2]:
                    if m == 0:
                      xi_lm_p_conj = sph_harm_no_phase_theta_0_pi_over_2[0, sph_p_cur_index] * cur_tilde_xi[l_p%2, m_p%2, 0]
                    else:
                      xi_lm_p_conj = 0
                  else:
                    xi_lm_p_conj = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * (phase_list[-m_p + l_max] * cur_tilde_xi[l_p%2, m_p%2, 0] + phase_list[m_p + l_max] * cur_tilde_xi[l_p%2, m_p%2, 1])

                  xi_lm_p_conj = np.conjugate(xi_lm_p_conj)

                  C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * xi_lm * xi_lm_p_conj
    
      if progress != None: progress.update(1)
    
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel=False, fastmath=True)
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
    sph_harm_no_phase_theta_0_pi_over_2,
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
      cur_tilde_xi = tilde_xi[i, :, :, :]
      m_list = np.arange(-l_max, l_max+1)
      phase_list = np.exp(1j * phi[i] * m_list)

      for l in range(ell_range[0], ell_range[1]+1):
        if k_amp_cur > k_max_list[l]:
            continue

        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m
            sph_cur_index = lm_index[l, np.abs(m)]

            if i <= eigenmode_index_split[0]:
              # xi depends on Y^*
              # xi^* depends on Y
              xi_lm = sph_harm_no_phase_theta_0_pi_over_2[1, sph_cur_index] * cur_tilde_xi[l%2, m%2, 0]
            elif i <= eigenmode_index_split[1]:
              xi_lm = sph_harm_no_phase_theta_0_pi_over_2[1, sph_cur_index] * exp(-1j * m * pi/2) * cur_tilde_xi[l%2, m%2, 0]
            elif i <= eigenmode_index_split[2]:
              if m == 0:
                xi_lm = sph_harm_no_phase_theta_0_pi_over_2[0, sph_cur_index] * cur_tilde_xi[l%2, m%2, 0]
              else:
                xi_lm = 0
            else:
              xi_lm = sph_harm_no_phase[sph_harm_index, sph_cur_index] * (phase_list[-m + l_max] * cur_tilde_xi[l%2, m%2, 0] + phase_list[m + l_max] * cur_tilde_xi[l%2, m%2, 1])

            C_lmlpmp[lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(xi_lm)**2
           
      progress.update(1)
    
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp