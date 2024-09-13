from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, tan, sqrt
from numba import njit, prange
from numba_progress import ProgressBar

class E1(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
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
      transfer_delta_kl,
      #random_phase
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
          #random_phase
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
        cubic = self.cubic,
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
      cubic = self.cubic,
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
          theta_unique_index=self.theta_unique_index,
          k_amp_unique_index=self.k_amp_unique_index,
          k_max_list = self.k_max_list[0, :],
          l_max=self.l_max,
          lm_index = self.lm_index,
          sph_harm_no_phase = self.sph_harm_no_phase,
          integrand=self.integrand_TT,
          ell_range = ell_range,
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
          cubic = self.cubic,
          progress = progress
        )
    return c_lmlpmp

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta= get_list_of_k_phi_theta(max(self.k_max_list[0, :]), self.Lx, self.Ly, self.Lz, self.beta, self.alpha, self.number_of_a_lm_realizations)
    '''
    if self.number_of_a_lm_realizations >= 1:
      self.random_phase = random_phase
    '''

    return k_amp, phi, theta

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, beta, alpha, number_of_a_lm_realizations=0):
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
    
    num_alm_real = number_of_a_lm_realizations if number_of_a_lm_realizations>=1 else 1
    '''
    random_phase = np.zeros((n_x_max * n_y_max * n_z_max * 8, num_alm_real), dtype=np.complex128)
    if number_of_a_lm_realizations >= 1:
      random_phase_index = -1 * np.ones((2 * n_x_max+1, 2 * n_y_max+1, 2 * n_z_max+1), dtype=np.intc)
    '''

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
          '''
          if number_of_a_lm_realizations >= 1:
            # Find opposite phase index (-k)
            
            opposite_phase_index = random_phase_index[-n_x + n_x_max, -n_y + n_y_max, -n_z + n_z_max]

            if opposite_phase_index == -1:
              # Phase has not been set for -vec(k), so we set it here
              for i in range(number_of_a_lm_realizations):
                random_phase[cur_index, i] = np.random.normal(loc=0.0, scale=1/sqrt(2)) + 1j*np.random.normal(loc=0.0, scale=1/sqrt(2))
                #random_phase[cur_index, i] = np.exp(1j*np.random.uniform(0.0, np.pi*2)) * np.random.normal(loc=0.0, scale=1.0)
              random_phase_index[n_x + n_x_max, n_y + n_y_max, n_z + n_z_max] = cur_index
            else:
              # Phase has been set for -vec(k), so we set the phase for k to be minus this
              random_phase[cur_index, :] = np.conjugate(random_phase[opposite_phase_index, :])
              random_phase_index[n_x + n_x_max, n_y + n_y_max, n_z + n_z_max] = cur_index
          '''
            
          cur_index += 1
    
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    #random_phase = random_phase[:cur_index, :]

    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta #, random_phase

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
    #random_phase
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

      #random_delta_k_n = np.random.normal(loc=0, scale = delta_k_n[k_unique_index_cur])
      #uniform = np.random.uniform(0.0, np.pi*2)
      #random_delta_k_n *= np.exp(1j * random_phase[i])
      #random_delta_k_n = delta_k_n[k_unique_index_cur] * random_phase[i]

      for l in range(2, l_max+1):
          if k_amp_cur > k_max_list[l]:
              continue
          delta_k_n_mul_transfer = transfer_delta_kl[k_unique_index_cur, l] #random_delta_k_n * 
          for m in range(l+1):
              lm_index_cur = lm_index[l, m]
              
              sph_harm = phase_list[m] * sph_harm_no_phase[sph_harm_index, lm_index_cur]

              a_lm[lm_index_cur] += delta_k_n_mul_transfer * sph_harm
    
    return a_lm

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
    cubic,
    progress
    ):

    # This calculates parts of the covariance. If only_diag==True
    # then it only finds the diagonal values, l, m = lp, mp

    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0]**2     
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - ell_p_range[0]**2 
    C_lmlpmp = np.zeros((num_l_m, num_l_m_p), dtype=np.complex128)    

    min_k_amp = np.min(k_amp)
    for i in range(min_index, max_index):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      # Powers of i so we can get them as an array call instead of recalculating them every time.
      # 1j**n == ipow[n%4], even for n<0.
      ipow = 1j**np.arange(4)

      for l in range(ell_range[0], ell_range[1]+1):
        for m in range(-l, l + 1):
          lm_index_cur = l * (l+1) + m - ell_range[0]**2
          abs_m = np.abs(m)
          sph_cur_index = lm_index[l, abs_m]

          sph_harm_lm_conj = sph_harm_no_phase[sph_harm_index, sph_cur_index] * phase_list[abs_m]
          if m<0:
            sph_harm_lm_conj = (-1)**m * np.conjugate(sph_harm_lm_conj)

          # Only do l-lp = 0 mod 2. Holds for general torus
          for l_p in range(l%2 + ell_p_range[0], ell_p_range[1]+1, 2):
            if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]) and k_amp_cur > min_k_amp:
              continue
            integrand_il_sphm = integrand[k_unique_index_cur, l, l_p] * ipow[(l-l_p)%4] * sph_harm_lm_conj
            
            # Negative m_p are calculated later
            # Only do m-mp = 0 mod 2
            if cubic:
              start_m = m%4
              m_step = 4
            else:
              start_m = m%2
              m_step = 2

            lm_p_index_cur = l_p * (l_p+1) - ell_p_range[0]**2
            for m_p in range(start_m, l_p + 1, m_step):
              #lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_p_range[0]**2
              sph_p_cur_index = lm_index[l_p, m_p]

              sph_harm_lm_p = sph_harm_no_phase[sph_harm_index, sph_p_cur_index] * np.conjugate(phase_list[m_p])

              C_lmlpmp[lm_index_cur, lm_p_index_cur + m_p] += integrand_il_sphm * sph_harm_lm_p

      if progress != None: progress.update(1)
    

    # Negative m_p are calculated here using symmetry arguments for general torus
    for l in range(ell_range[0], ell_range[1]+1):
        for m in range(-l, l + 1):
          lm_index_cur = l * (l+1) + m - ell_range[0]**2
          new_lm_index = l * (l+1) - m - ell_range[0]**2

          for l_p in range(ell_p_range[0], ell_p_range[1]+1):
              for m_p in range(-l_p, 0):
                lm_p_index_cur = l_p * (l_p+1) + m_p - ell_p_range[0]**2
                new_lm_p_index_cur = l_p * (l_p+1) - m_p - ell_p_range[0]**2

                C_lmlpmp[lm_index_cur, lm_p_index_cur] = C_lmlpmp[new_lm_index, new_lm_p_index_cur]

    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

@njit(nogil=True, parallel = False)
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

      for l in range(ell_range[0], ell_range[1]+1):
        if k_amp_cur > k_max_list[l] and k_amp_cur > min_k_amp:
            continue
            
        for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m
            sph_cur_index = lm_index[l, np.abs(m)]

            sph_harm_lm_abs = sph_harm_no_phase[sph_harm_index, sph_cur_index]

            C_lmlpmp[lm_index_cur] += integrand[k_unique_index_cur, l, l] * np.abs(sph_harm_lm_abs)**2
            
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
      cubic,
      progress
    ):

    # This calculates only the diagonal of the covariance.
    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    C_lmlpmp = np.zeros((l_max+1, l_max+1, N_s), dtype=np.complex128)    

    min_k_amp = np.min(k_amp)

    if cubic:
      m_mp_diff = 4
    else:
      m_mp_diff = 2

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
