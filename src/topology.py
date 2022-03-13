import camb
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import time
import healpy as hp
from tools import *
from numba import njit, prange, jit
from joblib import Parallel, delayed
import pyshtools as pysh

class Topology:
    def __init__(self, l_max = 5, n_max = 2, L = 1000, use_numba = True, fig_name='', debug=True):
        self.l_max = l_max
        self.n_max = n_max
        self.L = L #in Mpc
        self.use_numba = use_numba
        self.fig_name = fig_name
        self.debug = debug

        #Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(l_max)
        pars.set_accuracy(lSampleBoost = 50)
        if debug == False:
            # More accurate transfer functions
            # lSampleBoost >= 50 gives all multipoles
            pars.set_accuracy(AccuracyBoost = 3, lAccuracyBoost = 3, lSampleBoost = 50)
            pars.Accuracy.IntkAccuracyBoost = 3
            pars.Accuracy.SourcekAccuracyBoost = 3
        self.pars = pars

        # Get the CAMB functions and save them
        data = camb.get_transfer_functions(pars)
        results = camb.get_results(self.pars)
        self.powers = results.get_cmb_power_spectra(self.pars, raw_cl=True, CMB_unit='muK')['lensed_scalar']
        transfer = data.get_cmb_transfer_data(tp='scalar')

        # To get C_\ell in units of umK, we multiply by 1e6 (K to micro K) and the temperature of the CMB in K
        self.transfer_data = np.array(transfer.delta_p_l_k) * 1e6 * 2.7255
        print(self.transfer_data.shape)

        # CAMB gives the transfer data for a set of k and ell. Store these values
        # and later we will use interpolation for other k/ell values.
        #k is in Mpc^{-1}
        self.k_list = np.array(transfer.q)
        self.ell_list = np.array(transfer.L)
        #print(self.ell_list)

        time_start = time.time()
        self.do_pre_processing()
        print('Time to pre-process with l_max={} and n_max={}:'.format(l_max, n_max), time.time()-time_start, 'seconds')

    def do_pre_processing(self):
        # Calculates the transfer functions, primordial power spectrum,
        # and spherical harmonics and stores them for future use

        # Pre-computes the correct healpy index for (l,m) and spherical harmonics for future use
        l_max = self.l_max
        n_max = self.n_max
        num_l_m = int((l_max + 1)*(l_max + 2)/2)

        # k is discrete, so caluclate all possible integers nx^2+ny^2+nz^2
        # Also find nx^2+ny^2, we will use this later to find phi for the
        # spherical harmonics
        list_n_xy_squared, list_n_xyz_squared = get_list_of_n_squared(n_max)
        self.list_n_xyz_squared = list_n_xyz_squared
        self.list_n_xy_squared = list_n_xy_squared

        # Number of unique nx^2+ny^2(+nz^2) we can have
        num_xyz_squared = list_n_xyz_squared.size
        num_xy_squared = list_n_xy_squared.size

        # This will be the P(k)/k^3 * transfer_k(l) * conj(transfer_k(l')). 

        scalar_pk_k3 = np.zeros(num_xyz_squared)

        start_time = time.time()

        # So instead of for looping over all nx, ny, nz, we instead loop over possible n^2 values
        transfer_interpolate_k_l = scipy.interpolate.interp2d(self.k_list, self.ell_list, self.transfer_data[0, :, :], kind='quintic')
        transfer_delta_kl = np.zeros((num_xyz_squared, self.l_max+1))
        for n_squared_index in range(1, num_xyz_squared):
            n_squared = list_n_xyz_squared[n_squared_index]

            k = 2*pi / self.L * sqrt(n_squared)
            scalar_pk_k3[n_squared_index] = self.pars.scalar_power(k) / k**3
        
            for l in range(self.l_max+1):
                transfer_delta_kl[n_squared_index, l] = transfer_interpolate_k_l(k, l)
        
        integrand = do_integrand_pre_processing(list_n_xyz_squared, scalar_pk_k3, transfer_delta_kl, self.L, self.l_max)

        print('Time taken for first part of preprocessing:', time.time()-start_time)
        # Store them
        self.scalar_pk_k3 = scalar_pk_k3
        self.transfer_delta_kl = transfer_delta_kl
        self.transfer_interpolate_k_l = transfer_interpolate_k_l
        self.integrand = integrand

        # Plot the integrand for \ell=\ell_max to see that we are going to high enough n_max
        self.plot_integrand(l = l_max)

        # Healpy ordering of a_lm
        lm_index = np.zeros((l_max+1, l_max+1), dtype=int)
        for l in range(l_max+1):
            for m in range(l+1):
                cur_index = hp.Alm.getidx(l_max, l, m)
                lm_index[l, m] = cur_index

        # Get the spherical harmonics without the phase (without exp(i*m*phi))
        # We store these in an array of size (all lm, all n_z, all n*x^2+n_y^2)
        # This is because theta can be found from nz and nx^2+ny^2, and we do not
        # care about phi since we can add the phase in the sum
        sph_harm_no_phase = np.zeros((num_l_m, 2*n_max+1, num_xy_squared))
        for n_z in tqdm(range(-n_max, n_max+1)):
            for n_squared_index in range(num_xy_squared):
                n_xy_squared = list_n_xy_squared[n_squared_index]

                n_squared = n_xy_squared + n_z**2
                n = sqrt(n_squared)
                if n_squared==0 or n > n_max:
                    continue

                # Get theta for the sph_harm function
                theta = cart2theta(n_xy_squared, n_z)
                all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, theta, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
                for l in range(l_max+1):
                    for m in range(l+1):
                        cur_index = lm_index[l, m]
                        sph_harm_no_phase[cur_index, n_z+n_max, n_squared_index] = all_sph_harm_no_phase[0, l, m]

        # Save healpy indices, legendre functions, and the pre-factor
        self.lm_index = lm_index
        self.sph_harm_no_phase = sph_harm_no_phase

    def plot_integrand(self, l):
        # Plot the integrand up to n_max and finds the
        # continous integral to get C_l. This is to see
        # if the chosen n_max is large enough.
        plt.figure()
        n_max = self.n_max
        k_max = 2*pi * n_max / self.L
        k_list = np.linspace(2*pi * 0.1 / self.L, k_max, 1000)
        integrand = 4*pi * self.pars.scalar_power(k_list) * self.transfer_interpolate_k_l(k_list, l)**2 / k_list
        plt.plot(k_list * self.L/(2*pi), integrand)
        plt.savefig('figs/transfer_integrand/transfer_squared_P_k_k_ell{}_L_{}.pdf'.format(l, int(self.L/1e3)))

        c_l = scipy.integrate.simpson(integrand, x=k_list)
        print('Continous integration up to n_max={} ratio for l={}:'.format(n_max, l), c_l / self.powers[l, 0])

    def calculate_c_lmlpmp(self, only_diag=False):
        # Calculates the off-diagonal and on-diagonal power spectrum

        l_max = self.l_max
        n_max = self.n_max
        L = self.L

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        # Numba makes stuff 10x faster
        # But things need to be pre-computed for it to work
        if (self.use_numba):
            print('Done pre-processing. Starting Numba')
            
            os.environ["OMP_NUM_THREADS"] = "1"
            time_start = time.time()
            
            #test_unpara = njit(test, parallel=False)(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, integrand=self.integrand, legendre = self.legendre, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared)
            #test_para = njit(test, parallel=True)(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, integrand=self.integrand, legendre = self.legendre, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared)
            
            C_TT_lmlpmp = get_C_numba(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, integrand=self.integrand, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, only_diag = only_diag)
            print('Time to calculate C_TT with l_max={} and n_max={}:'.format(l_max, n_max), time.time()-time_start, 'seconds with Numba')            

        C_TT_diag = np.zeros(l_max+1)
        for l in range(l_max+1):
            for m in range(l+1):
                lm_index = hp.Alm.getidx(l_max, l, m)
                if m==0:
                    C_TT_diag[l] += np.real(C_TT_lmlpmp[lm_index, lm_index])
                else:
                    # Twice because we are putting -m and +m together.
                    C_TT_diag[l] += 2*np.real(C_TT_lmlpmp[lm_index, lm_index])
            C_TT_diag[l] /= 2*l + 1
        self.C_TT_diag = C_TT_diag
        self.C_TT_lmlpmp = C_TT_lmlpmp

    def calculate_alm_realization(self):
        l_max = self.l_max
        n_max = self.n_max
        L = self.L
        print('Caclulating a_lm realization')
        time_start = time.time()

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        a_lm, c_l = get_alm_numba(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, delta_k_n = np.sqrt(self.scalar_pk_k3), transfer_delta_kl = self.transfer_delta_kl)
        print('Time to calculate one a_lm realization:', time.time()-time_start)
        return a_lm, c_l

    def make_alm_realizations(self):
        fig1, ax1 = plt.subplots()
        ell = np.arange(3, l_max+1)

        it = 4
        cl_list = np.zeros((l_max+1, it))
        for i in range(it):
            alm, cl = self.calculate_alm_realization()
            cl_list[:, i] = cl
            ax1.plot(ell, get_D_l(cl)[3:])

            fig2 = plt.figure()
            map = hp.alm2map(alm, 1024, pol=False)
            hp.mollview(map, fig=fig2, remove_dip=True)
            fig2.savefig('realizations/tmp_real/map_L{}_{}_nmax_{}.pdf'.format(self.L, i, n_max))

        correct_D_l = get_D_l(self.powers[:l_max+1, 0])
        ax1.plot(ell, correct_D_l[3:l_max+1], label='CAMB spectrum')
        cosmic_variance = np.array([2 * self.powers[l, 0]**2 / (2*l+1) for l in range(l_max+1)])
        D_l_cv = get_D_l(sqrt(cosmic_variance))
        ax1.fill_between(ell, (correct_D_l - D_l_cv)[3:], (correct_D_l + D_l_cv)[3:], color='grey', alpha=0.5)
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$\ell$')
        ax1.set_ylabel(r'$\ell (\ell+1)C^{TT}_\ell / 2\pi \, [\mu K^2]$ ')
        fig1.savefig('tmp_L{}_nmax_{}.pdf'.format(int(self.L), n_max))

        return cl_list

    def plot_c_lmlpmp(self):
        L = self.L
        l_max = self.l_max
        plt.figure()
        plt.plot(self.C_TT_diag, label='Topology C_l (Discrete sum)')
        plt.plot(self.powers[:l_max+1, 0], label='CAMB C_l (Continuous integration)')
        #np.save('C_TT_top.npy', self.C_TT_diag)
        np.save('realizations/c_TT_L_{}_lmax_{}.npy'.format(int(self.L/1000), self.l_max), self.C_TT)
        plt.legend()
        plt.yscale('log')
        plt.ylabel(r'$C^{TT}_{\ell} \,\, [\mu K^2]$ ')
        plt.xlabel(r'$\ell$')
        plt.title(r'$L={{{}}}$ Gpc'.format(int(L/1e3)))
        plt.savefig('figs/tmp_{}_L_{}.pdf'.format(self.fig_name, int(self.L/1e3)))


        # Plot covariance
        lmax = self.l_max
        C_TT_order = np.zeros(self.C_TT.shape, dtype=np.complex128)
        for l in range(self.l_max+1):
            for m in range(l+1):
                id = hp.Alm.getidx(self.l_max, l, m)
                for lp in range(self.l_max+1):
                    for mp in range(lp+1):
                        idp = hp.Alm.getidx(self.l_max, lp, mp)
                        #lm = 00, lm = 10, lm = 11, lm=20, lm=21, lm=22, lm=30, lm=31
                        #i =0        1        2      3       4       5       6     7

                        C_TT_order[int(l * (l+1) / 2) + m, int(lp * (lp+1) / 2) + mp] = self.C_TT[id, idp]

        ell_ranges = np.array([int(l * (l+1) / 2) for l in range(1, lmax+1)])
        ell_labels = np.arange(1, l_max+1)
        print(ell_labels)

        plt.figure()
        plt.imshow(np.abs(C_TT_order), norm=LogNorm())
        plt.xlabel(r'$\ell$')
        plt.xticks(ticks=ell_ranges, label=ell_labels)
        plt.yticks(ticks=ell_ranges, label=ell_labels)
        ax = plt.gca() # grab the current axis
        ax.set_xticklabels(ell_labels)
        ax.set_yticklabels(ell_labels)

        plt.ylabel(r'$\ell$')
        plt.clim(1e-3, 1e3)
        plt.colorbar()
        plt.savefig('figs/c_tt_off_diag_L_{}_lmax_{}_nmax_{}.pdf'.format(int(self.L/1000), self.l_max, self.n_max))

    def make_realization_with_c_lmlp(self):
        C_TT = self.C_TT
        _, dim = C_TT.shape
        #print(C_TT)
        v1 = np.ones(dim)
        print(np.dot(v1, np.dot(C_TT,v1)))
        L = np.linalg.cholesky(C_TT)
        #L = scipy.linalg.cholesky(C_TT)
        
        # This looks scary, but its just a random COMPLEX vector of mean 0 and sigma=1
        # for both the real part and imaginary part
        uncorr_alm = np.random.normal(0, 1, (L.shape[0], 2)).view(np.complex128)[:, 0]

        corr_alm = np.dot(L, uncorr_alm)

        map = hp.alm2map(corr_alm, 512, pol=False)
        plt.figure()
        hp.mollview(map, remove_dip=True)
        plt.savefig('realizations/map_L_{}_lmax_{}_nmax_{}.pdf'.format(int(self.L/1000), self.l_max, self.n_max))

def __init__():
    print('hei')
#Things to add:
# 1) Calculating all off-diagonal elements take time, maybe add a bool that if true only calculates diagonal elements
# 2) A way to generalize for different topologies. Class inheritance?