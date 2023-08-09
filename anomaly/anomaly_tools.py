import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi
from tqdm import tqdm
from scipy import special, integrate


def run_octopole_quadrupole_alignment_octopole_planarity(a_lm, rotate=True, N_side=16):
  # This functions returns the octopole planarity t_3
  # And the normal vector of ell=2 and ell=3 (S_2_coord and S_3_coord)

  # rotate = True rotates a map of N_side to all possible pixels in the northern hemisphere
  # to find the point where t_3 is max. See https://arxiv.org/abs/astro-ph/0307282

  num_alm = a_lm[:, 0].size
  ell_max = hp.Alm.getlmax(a_lm[0, :].size)
  m = np.arange(0, ell_max+1)
  indices_ell3 = hp.Alm.getidx(lmax=ell_max, l = 3, m=m)
  indices_ell2 = hp.Alm.getidx(lmax=ell_max, l = 2, m=m)

  t_3 = np.zeros(num_alm)
  S_2_coord = np.zeros((num_alm, 2))
  S_3_coord = np.zeros((num_alm, 2))

  if rotate:
    pixels = 12*N_side**2
    pixel_ids = np.arange(pixels)
    theta_list, phi_list = hp.pix2ang(N_side, pixel_ids)

    k = 100
    vec_1 = np.array([cos(phi_list[k])*sin(theta_list[k]), sin(phi_list[k])*sin(theta_list[k]), cos(theta_list[k])])
    vec_2 = np.array([cos(phi_list[k+1])*sin(theta_list[k+1]), sin(phi_list[k+1])*sin(theta_list[k+1]), cos(theta_list[k+1])])
    print('Pixel resolution for Nside={}. Max dot product of neighboring pixels: '.format(N_side), np.dot(vec_1, vec_2))

    r_rotator = [None] * pixels

    # Only northern hemisphere is good enough
    pixels = int(pixels/2)
    for i in range(pixels):
      r_rotator[i] = hp.Rotator(rot=[pi + phi_list[i], theta_list[i], 0],  deg=False, eulertype='Y')

    for i in tqdm(range(num_alm)):
      S_2_largest = 0
      S_3_largest = 0
      for j in range(pixels):
        new_alm = r_rotator[j].rotate_alm(a_lm[i, :])
        a_3m = new_alm[indices_ell3]
        a_2m = new_alm[indices_ell2]
        sigma_3_times_7 = np.abs(a_3m[0])**2
        sigma_3_times_7 += 2*np.sum( np.abs(a_3m[1:])**2 )

        t_3_cur = 2*np.abs(a_3m[3])**2 / sigma_3_times_7
        S_2_cur = 2*(abs(a_2m[1])**2 + 2**2 * abs(a_2m[2])**2)
        S_3_cur = 2*(abs(a_3m[1])**2 + 2**2 * abs(a_3m[2])**2 + 3**2 * abs(a_3m[3])**2)

        if S_2_cur > S_2_largest:
          S_2_coord[i, 0] = theta_list[j]
          S_2_coord[i, 1] = phi_list[j]
          S_2_largest = S_2_cur
        
        if S_3_cur > S_3_largest:
          S_3_coord[i, 0] = theta_list[j]
          S_3_coord[i, 1] = phi_list[j]
          S_3_largest = S_3_cur

        if t_3_cur > t_3[i]:
          t_3[i] = t_3_cur          
  else:
    a_3m = a_lm[i, indices_ell3]
    sigma_3_times_7 = np.abs(a_3m[0])**2
    sigma_3_times_7 += 2*np.sum( np.abs(a_3m[1:])**2 )
    t_3[i] = 2*np.abs(a_3m[3])**2 / sigma_3_times_7
  
  return t_3, S_2_coord, S_3_coord

def get_S_one_half(a_lm):
  num_alm = a_lm[:, 0].size
  lmax = hp.Alm.getlmax(a_lm[0, :].size)
  c_l = np.zeros((num_alm, lmax+1))
  for i in range(num_alm):
    c_l[i, :] = hp.alm2cl(a_lm[i, :])
  
  steps = 3000
  
  theta = np.linspace(0, pi, steps)

  C_theta = np.zeros((num_alm, steps))
  S_one_half =  np.zeros((num_alm))

  P_l_cos_theta = np.zeros((lmax+1, steps))
  for l in range(2, lmax+1):
    P_l_cos_theta[l, :] =  special.eval_legendre(l, cos(theta))

  for i in tqdm(range(num_alm)):
    C_theta_curr = np.zeros(steps)
    for l in range(2, lmax+1):
      C_theta_curr += (2*l+1)/(4*np.pi) * P_l_cos_theta[l, :] * c_l[i, l]
    C_theta[i, :] = C_theta_curr

    integrand = sin(theta) * C_theta_curr**2
    
    start = int(steps/3)
    S_one_half_cur = integrate.simpson(integrand[start:], theta[start:])
    S_one_half[i] = S_one_half_cur
  
  return S_one_half, C_theta

  