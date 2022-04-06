from .topology import Topology
from .tools import *
import numpy as np
from numpy import pi, sin, tan, sqrt
from numba import njit, prange

class E1(Topology):
  def __init__(self, param, debug=True):
    self.Lx = param['Lx'] * 28*1e3
    self.Ly = param['Ly'] * 28*1e3
    self.Lz = param['Lz'] * 28*1e3
    self.beta = param['beta']
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta)
    self.l_max = param['l_max']
    self.param = param


    self.n_max_list = self.get_initial_nmax_list()
    k_max_list = 2*pi * self.n_max_list / np.power(self.V, 1/3)
    self.k_max_list = k_max_list

    Topology.__init__(self, param, debug)

  def get_list_of_k_phi_theta(self):
    return get_list_of_k_phi_theta(max(self.k_max_list), self.Lx, self.Ly, self.Lz, self.beta)

@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, L_x, L_y, L_z, beta):
    # Returns list of k, phi, and theta for this topology

    tan_b_inv = 1/tan(beta)
    sin_b_inv = 1/sin(beta)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z / (2*pi) * (sin_b_inv + tan_b_inv))) # Multiply by 2 since we have something like kz = a * nz - b * nx
 
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    cur_index = 0

    for n_x_amp in prange(n_x_max+1):
      k_x = 2*pi * n_x_amp / L_x

      for n_x_sign in range(2):
        # Dont do n_x=0 twice
        if n_x_amp == 0 and n_x_sign == 1:
          continue

        if n_x_sign == 1: k_x *= -1

        for n_z_amp in range(n_z_max+1):
          for n_z_sign in range(2):
            # Dont do n_z=0 twice
            if n_z_amp == 0 and n_z_sign == 1:
              continue

            if n_z_sign == 0:
              k_z = 2*pi * n_z_amp * sin_b_inv / L_z - k_x * tan_b_inv
            else:
              k_z = - 2*pi * n_z_amp * sin_b_inv/ L_z - k_x * tan_b_inv

            k_xz_squared = k_x**2 + k_z**2

            if k_xz_squared > k_max**2:
              continue

            for n_y_amp in range(n_y_max+1):
              k_y = 2*pi * n_y_amp / L_y
              
              for n_y_sign in range(2):
                # Dont do n_y=0 twice
                if n_y_amp == 0 and n_y_sign == 1:
                  continue
                  
                if n_y_sign == 1: k_y *= -1

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