import camb
import os
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import time
import healpy as hp
from .tools import *
import pyshtools as pysh
from sys import getsizeof
import numba as nb

class Topology:
    def __init__(self, param, debug=True):
        self.param = param
        self.topology = param['topology']
        self.l_max = param['l_max']
        self.min_cl_ratio = param['c_l_accuracy']
        self.L = param['Lx'] * 28 * 1e3 # Defined in MPc

        self.n_max_list = self.get_initial_nmax_list()
        self.fig_name = 'l_max_{}_n_max_{}'.format(self.l_max, max(self.n_max_list))
        self.debug = debug

        print('Running - n_max={}, l_max={}, L={}'.format(max(self.n_max_list), self.l_max, int(self.L)))
        self.root = 'runs/top_{}_L_{}_l_max_{}_accuracy_{}_percent/'.format(
            self.topology,
            int(self.L),
            self.l_max,
            int(self.min_cl_ratio*100)
        )
        if os.path.exists(self.root) == False:
            print('Making run folder:', self.root)
            os.makedirs(self.root)
            os.makedirs(self.root+'figs/')
            os.makedirs(self.root+'realizations/')

        #Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(self.l_max)
        pars.set_accuracy(lSampleBoost = 50)
        if debug == False:
            # More accurate transfer functions. Takes longer time to run
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
        print('Shape of transfer function from CAMB:', self.transfer_data.shape)

        # CAMB gives the transfer data for a set of k and ell. Store these values
        # and later we will use interpolation for other k/ell values.
        #k is in Mpc^{-1}
        self.k_list = np.array(transfer.q)
        self.ell_list = np.array(transfer.L)

        time_start = time.time()
        self.do_pre_processing()
        print('Time to pre-process with l_max={} and n_max={}:'.format(self.l_max, max(self.n_max_list)), time.time()-time_start, 'seconds')

    def get_initial_nmax_list(self):
        # Returns n_max as a function of ell.
        # This is just an initial guess. But make sure that self.param['initial_n_max_for_l_max']
        # does not make the CAMB c_l ratio for l_max to large
        n_max = self.param['initial_n_max_for_l_max']
        l_max = self.l_max

        n_max_list_initial = np.ceil((np.arange(0, l_max+1)+4) * (n_max-1) / l_max)
        print(n_max_list_initial)
        return np.array(list(map(int, n_max_list_initial)))

    def is_nmax_high_enough(self, plot_lmax_integrand=False):
        # Calculate c_l up to k_max(n_max) and compare it to CAMB to see
        # if the chosen n_max is large enough.
        l_max = self.l_max
        ratio_lmax = np.ones(l_max+1)
        for l in tqdm(range(2, l_max+1)):
            n_max = self.n_max_list[l]
            k_max = 2 * pi * n_max / self.L
            k_list = np.linspace(2*pi * 1e-8 / self.L, k_max, 2**16+1)
            integrand = 4*pi * self.pars.scalar_power(k_list) * self.transfer_interpolate_k_l(k_list, l)**2 / k_list
            c_l = scipy.integrate.romb(integrand) * (k_list[1]-k_list[0])
            ratio_lmax[l] = c_l / self.powers[l, 0]

            if ratio_lmax[l] < self.min_cl_ratio:
                self.n_max_list[l] = int(np.ceil(self.n_max_list[l] * 1 / ratio_lmax[l]))

        for l in range(l_max+1):
            if self.n_max_list[l] > self.n_max_list[l_max]: self.n_max_list[l] = self.n_max_list[l_max]

        if plot_lmax_integrand:
            plt.figure()
            plt.plot(k_list * self.L/(2*pi), integrand)
            plt.savefig(root+'figs/transfer_integrand/transfer_squared_P_k_k_ell{}_L_{}.pdf'.format(l, int(self.L)))
        
        print('Continous integration up to n_max={} ratio for l={}:'.format(n_max, l), ratio_lmax)
        print('n_max list as a function of ell:', self.n_max_list)
        if min(ratio_lmax) < self.min_cl_ratio:
            # n_max is not high enough. We need to go to increase n_max
            print('n_max of was not high enough for one or more ell')
            return False
        else:
            print('n_max is high enough (make sure it is not too high, otherwise it will be slow)')
            np.save(self.root+'n_max_list_lmax_{}_L_{}.npy'.format(self.l_max, int(self.L)), self.n_max_list)
            return True

    def calculate_c_lmlpmp(self, only_diag=False, with_parallel = False):
        # Calculates the off-diagonal and on-diagonal power spectrum

        l_max = self.l_max
        L = self.L

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        print('Done pre-processing. Starting Numba')

        time_start = time.time()

        if with_parallel:
            C_TT_lmlpmp = nb.njit(parallel=True)(get_c_lmlpmp)(L=L, l_max=l_max, n_max_list=self.n_max_list, lm_index = lm_index, integrand=self.integrand, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, only_diag = only_diag)
        else:
            C_TT_lmlpmp = nb.njit(parallel=False)(get_c_lmlpmp)(L=L, l_max=l_max, n_max_list=self.n_max_list, lm_index = lm_index, integrand=self.integrand, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, only_diag = only_diag)
        print('Time to calculate C_TT with l_max={} and n_max={}:'.format(l_max, max(self.n_max_list)), time.time()-time_start, 'seconds with Numba')            

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

    def calculate_alm_realization(self, timer=True, with_parallel = False):
        l_max = self.l_max
        n_max = max(self.n_max_list)
        L = self.L
        print('Caclulating a_lm realization')
        time_start = time.time()

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        if with_parallel:
            a_lm, c_l = nb.njit(parallel=True)(get_alm_numba)(L=L, l_max=l_max, n_max_list=self.n_max_list, lm_index = lm_index, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, delta_k_n = np.sqrt(self.scalar_pk_k3), transfer_delta_kl = self.transfer_delta_kl)
            get_alm_numba.parallel_diagnostics(level=4)
        else:
            a_lm, c_l = nb.njit(parallel=False)(get_alm_numba)(L=L, l_max=l_max, n_max_list=self.n_max_list, lm_index = lm_index, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, delta_k_n = np.sqrt(self.scalar_pk_k3), transfer_delta_kl = self.transfer_delta_kl)
        if timer: print('Time to calculate one a_lm realization:', time.time()-time_start)
        return a_lm, c_l

    def make_alm_realizations(self, plot_alm=True, save_alm = False, it=2):
        l_max = self.l_max

        cl_list = np.zeros((l_max+1, it))

        if save_alm: alm_list = np.zeros((it, int((l_max + 1)*(l_max + 2)/2)), dtype=np.complex128)

        for i in tqdm(range(it)):
            alm, cl = self.calculate_alm_realization(timer=False, with_parallel = False)
            #print(alm[100:120])
            #alm, cl = self.calculate_alm_realization(timer=False, with_parallel = True)
            #print(alm[100:120])

            cl_list[:, i] = cl

            if save_alm: alm_list[i, :] = alm

            # Plot only first 4 realizations for check
            if i < 4 and plot_alm:
                fig2 = plt.figure()
                map = hp.alm2map(alm, 128, pol=False)
                hp.mollview(map, fig=fig2, remove_dip=True)
                fig2.savefig(self.root+'realizations/map_L{}_lmax_{}_nmax_{}_{}.pdf'.format(int(self.L), l_max, max(self.n_max_list), i))

        if save_alm: np.save(self.root+'realizations/realizations_L_{}_lmax_{}_num_{}.npy'.format(int(self.L), l_max, it), alm_list)

        return cl_list

    def make_euclidean_realizations(plot_alms=False):
        it=1000
        alm_list = np.zeros((it, int((l_max + 1)*(l_max + 2)/2)), dtype=np.complex128)
        for i in range(it):
            map, alm = hp.synfast(self.powers[:, 0], nside=128, lmax=l_max, alm=True, pol=False)
            alm_list[i, :] = alm

            if i < 4 and plot_alms:
                plt.figure()
                hp.mollview(map, remove_dip=True)
                plt.savefig(self.root+'realizations/map_L_infty_lmax_{}_{}.pdf'.format(l_max, i))

        np.save(self.root+'realizations/realizations_L_infty_lmax_{}_num_{}.npy'.format(l_max, it), alm_list)


    def plot_c_lmlpmp(self):
        # This function plots the off-diagonal covariance matrix
        L = self.L
        l_max = self.l_max
        plt.figure()
        plt.plot(self.C_TT_diag, label='Topology C_l (Discrete sum)')
        plt.plot(self.powers[:l_max+1, 0], label='CAMB C_l (Continuous integration)')
        #np.save('C_TT_top.npy', self.C_TT_diag)
        np.save(self.root+'realizations/c_TT_L_{}_lmax_{}.npy'.format(int(self.L), self.l_max), self.C_TT)
        plt.legend()
        plt.yscale('log')
        plt.ylabel(r'$C^{TT}_{\ell} \,\, [\mu K^2]$ ')
        plt.xlabel(r'$\ell$')
        plt.title(r'$L={{{}}}$ Mpc'.format(int(L)))
        plt.savefig(self.root+'figs/tmp_{}_L_{}.pdf'.format(self.fig_name, int(self.L)))


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
        plt.savefig(self.root+'figs/c_tt_off_diag_L_{}_lmax_{}_nmax_{}.pdf'.format(int(self.L/1000), self.l_max, self.n_max))

    def make_realization_with_c_lmlp(self):
        # THIS CODE SEEMS TO BE BUGGY!
        # DO NOT TRUST YET

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
        plt.savefig(self.root+'realizations/map_L_{}_lmax_{}_nmax_{}.pdf'.format(int(self.L/1000), self.l_max, self.n_max))

    def plot_c_l_and_realizations(self, c_l_a):
        l_min=3
        l_max = self.l_max
        ell = np.arange(l_min, self.l_max+1)
        correct_D_l = get_D_l(self.powers[:self.l_max+1, 0])

        cosmic_variance = np.array([2 * self.powers[l, 0]**2 / (2*l+1) for l in range(self.l_max+1)])
        D_l_cv = get_D_l(sqrt(cosmic_variance))

        plt.figure()
        plt.plot(ell, get_D_l(self.C_TT_diag)[l_min:self.l_max+1], linewidth=4, label='L={} True C_l'.format(int(self.L)))
        plt.plot(ell, correct_D_l[l_min:self.l_max+1], linewidth=4, label='CAMB')

        plt.fill_between(ell, (correct_D_l - D_l_cv)[l_min:], (correct_D_l + D_l_cv)[3:], color='grey', alpha=0.5)

        for i in range(2):
            plt.plot(ell, get_D_l(c_l_a[:, i])[l_min:self.l_max+1], label='Realization {}'.format(i))
        #plt.plot(get_D_l(b.C_TT_diag)[5:l_max], label='L=0.4')
        #plt.plot(get_D_l(c.C_TT_diag)[5:l_max], label='L=0.6')
        plt.legend()
        plt.ylabel(r'$\ell (\ell+1)C^{TT}_\ell / 2\pi \, [\mu K^2]$')
        plt.xlabel(r'$\ell$')
        #plt.xscale('log')
        plt.savefig(self.root+'figs/power_spectrum_L_{}_l_max_{}.pdf'.format(int(self.L), l_max))






    def do_numba_bug(self, timer=True, with_parallel = True):
        l_max = self.l_max
        n_max = self.n_max
        L = self.L
        print('Finding numba bug')
        time_start = time.time()

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        if with_parallel:
            a_lm = nb.njit(parallel=True)(numba_bug)(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, delta_k_n = np.sqrt(self.scalar_pk_k3), transfer_delta_kl = self.transfer_delta_kl)
        else:
            a_lm = nb.njit(parallel=False)(numba_bug)(L=L, l_max=l_max, n_max=n_max, lm_index = lm_index, sph_harm_no_phase = self.sph_harm_no_phase, list_n_xy_squared = self.list_n_xy_squared, list_n_xyz_squared = self.list_n_xyz_squared, delta_k_n = np.sqrt(self.scalar_pk_k3), transfer_delta_kl = self.transfer_delta_kl)
        return a_lm

#Things to add:
# 1) Calculating all off-diagonal elements take time, maybe add a bool that if true only calculates diagonal elements
# 2) A way to generalize for different topologies. Class inheritance?