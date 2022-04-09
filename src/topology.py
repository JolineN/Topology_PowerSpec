import camb
import os
import numpy as np
from numpy import pi, sqrt, cos
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import time
import healpy as hp
from .tools import *
import pyshtools as pysh
from sys import getsizeof
import numba
from numba import njit, prange
from collections import defaultdict

class Topology:
    def __init__(self, param, debug=True):
        self.param = param
        self.topology = param['topology']
        self.l_max = param['l_max']
        self.c_l_accuracy = param['c_l_accuracy']

        self.get_initial_kmax_list()

        self.fig_name = 'l_max_{}'.format(self.l_max)
        self.debug = debug

        print('Running - l_max={}, Lx={}'.format(self.l_max, int(self.Lx)))
        self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_l_max_{}_accuracy_{}_percent/'.format(
            self.topology,
            int(self.Lx),
            int(self.Ly),
            int(self.Lz),
            int(self.beta*180/np.pi),
            self.l_max,
            int(self.c_l_accuracy*100)
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

        # More accurate transfer functions. Takes longer time to run
        pars.set_accuracy(AccuracyBoost = 2, lAccuracyBoost = 2, lSampleBoost = 50)
        pars.Accuracy.IntkAccuracyBoost = 2
        pars.Accuracy.SourcekAccuracyBoost = 2
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
        print('Time to pre-process with l_max={} and accuracy={}:'.format(self.l_max, self.c_l_accuracy), time.time()-time_start, 'seconds')

    def do_pre_processing(self):
        # This function does all the preprocessing

        # Calculates the transfer functions, primordial power spectrum,
        # and spherical harmonics and stores them with as little memory as possible

        l_max = self.l_max
        num_l_m = int((l_max + 1)*(l_max + 2)/2)

        # Get the transfer functions and put them into lists of interpolate objects.
        # We can therefore evaluate the transfer function for all possibel |k| later
        assert(self.ell_list[0] == 2 and self.ell_list[l_max-2] == l_max)
        transfer_interpolate_k_l_list = np.empty_like(scipy.interpolate.interp1d, shape=l_max+1)
        for l in range(2, l_max+1):
            transfer_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, self.transfer_data[0, l-2, :], kind='quadratic') 
        self.transfer_interpolate_k_l_list = transfer_interpolate_k_l_list

        # We have a list of k_max as a function of ell. We need to make sure this is large enough
        self.get_kmax_as_function_of_ell()
        #if self.is_kmax_high_enough() == False:
        #    # k_max is not high enough. So we start pre-processing again with a higher k_max for one or more ell
        #    print('Starting the pre-process again with higher k_max')
        #    print('\n')
        #    self.do_pre_processing()
        #    return
        
        # We find all allowed |k|, phi, theta and put them in big lists
        # The function get_list_of_k_phi_theta is specific to each topology
        start_time = time.time()
        k_amp, phi, theta = self.get_list_of_k_phi_theta()
        print('Time to get list of k, phi, theta:', time.time()-start_time, 'seconds')

        self.k_amp = k_amp
        self.phi = phi
        self.theta = theta

        # Number of indices on the k, phi, theta vectors
        # Same as number of sums we have to do for each alm
        num_k_indices = k_amp.size

        # |k|, phi and theta often repeats themselves. We do not want to recalculate spherical harmonics
        # twice or more so we store a list of all unique thetas. Same for |k| to quickly find transfer functions later
        start_time = time.time()
        k_amp_unique, k_amp_unique_index, theta_unique, theta_unique_index = get_k_theta_index_repeat(k_amp, theta)
        print('Time to get unique k and theta:', time.time()-start_time, 'seconds')

        self.k_amp_unique = k_amp_unique
        self.k_amp_unique_index = k_amp_unique_index

        self.theta_unique = theta_unique
        self.theta_unique_index = theta_unique_index

        # Get P(k) / k^3 for all unique |k| values
        scalar_pk_k3 = self.pars.scalar_power(k_amp_unique) / k_amp_unique**3

        # Get the transfer function for all unique |k| values
        start_time = time.time()
        transfer_delta_kl = self.get_transfer_functions()
        self.scalar_pk_k3 = scalar_pk_k3
        self.transfer_delta_kl = transfer_delta_kl

        # Get P(k)/k^3 * Delta_ell(k) * Delta_ell'(k)
        start_time = time.time()
        integrand = do_integrand_pre_processing(k_amp_unique, scalar_pk_k3, transfer_delta_kl, self.l_max)
        self.integrand = integrand
        print('Size of integrand: {} MB. Time to calculate integrand: {} s'.format(round(getsizeof(integrand) / 1024 / 1024,2), time.time()-start_time))
        print(integrand[10, 10, 10])
        # Store the Healpy ordering of a_lm
        lm_index = np.zeros((l_max+1, l_max+1), dtype=int)
        for l in range(l_max+1):
            for m in range(l+1):
                cur_index = hp.Alm.getidx(l_max, l, m)
                lm_index[l, m] = cur_index
        self.lm_index = lm_index

        # Get the spherical harmonics without the phase (without exp(i*m*phi))
        # We store these in an array of size (all lm, all n_z, all n*x^2+n_y^2)
        # This is because theta can be found from nz and nx^2+ny^2, and we do not
        # care about phi since we can add the phase in the sum
        start_time = time.time()
        sph_harm_no_phase = self.get_sph_harm()
        self.sph_harm_no_phase = sph_harm_no_phase

        print('\n**************')
        print('Done with all preprocessing')
        print('**************\n')

    def get_initial_kmax_list(self):
        # Returns k_max as a function of ell.
        # This is just an initial guess of k_max assuming ell_max = 250
        if np.isclose(self.c_l_accuracy, 0.99):
            k_max = 0.061
        elif np.isclose(self.c_l_accuracy, 0.95):
            k_max = 0.043
        elif np.isclose(self.c_l_accuracy, 0.90):
            k_max = 0.032
        else:
            # Random guess. This should be improved in the future
            k_max = 0.035 * self.c_l_accuracy

        l_max = self.l_max

        self.k_max_list = np.arange(0, l_max+1) * k_max / l_max

    def get_kmax_as_function_of_ell(self):
        # Get k_max as a function of multipole ell
        # We use cumulative trapezoid to find the k_value where we reach
        # the wanted accuracy

        l_max = self.l_max

        # Do the integration up to k=0.08. This should be fine for ell=<250 and accuracy<=0.99
        k_max = 0.08
        print('\nFinding k_max as a function of ell')
        for l in tqdm(range(2, l_max+1)):            
            k_list = np.linspace(min(self.k_list), k_max, 2**16+1)
            integrand = 4*pi * self.pars.scalar_power(k_list) * self.transfer_interpolate_k_l_list[l](k_list)**2 / k_list
            
            cumulative_c_l_ratio = scipy.integrate.cumulative_trapezoid(y=integrand, x=k_list) / self.powers[l, 0]
            index_closest_to_accuracy_target = (np.abs(cumulative_c_l_ratio -  self.c_l_accuracy)).argmin()

            self.k_max_list[l] = k_list[index_closest_to_accuracy_target]
            #print(cumulative_c_l_ratio[index_closest_to_accuracy_target], 'for ell =', l, 'k=', self.k_max_list[l])
        print('\n********************************************')
        print('Done. k_max for ell_max =', self.k_max_list[l_max])
        print('********************************************\n')
    
    def calculate_c_lmlpmp(self, only_diag=False):
        # Calculates the off-diagonal and on-diagonal power spectrum
        l_max = self.l_max

        # We use healpy ordering of the a_{lm}. We use a 1d array, instead of 2d.
        num_l_m = int((l_max + 1)*(l_max + 2)/2)
        lm_index = self.lm_index

        time_start = time.time()
        
        print('\nDone pre-processing. Starting Numba')
        C_TT_lmlpmp = get_c_lmlpmp(
            V=self.V,
            k_amp=self.k_amp,
            phi=self.phi,
            theta_unique_index=self.theta_unique_index,
            k_amp_unique_index=self.k_amp_unique_index,
            k_max_list = self.k_max_list,
            l_max=l_max,
            lm_index = self.lm_index,
            sph_harm_no_phase = self.sph_harm_no_phase,
            integrand=self.integrand,
            only_diag = only_diag
        )
        print('Time to calculate C_TT with l_max={} and c_l_ratio={}:'.format(l_max, self.c_l_accuracy), time.time()-time_start, 'seconds with Numba')            

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

    def make_alm_realizations(self, plot_alm=True, save_alm = False, it=2):
        # Make it number of alm realizations. These can be saved in .npy files
        l_max = self.l_max

        cl_list = np.zeros((l_max+1, it))

        if save_alm: alm_list = np.zeros((it, int((l_max + 1)*(l_max + 2)/2)), dtype=np.complex128)

        print('')
        print('***********')
        print('Calculating a_lm realizations')
        print('***********')
        print('')
        for i in tqdm(range(it)):
            a_lm, c_l = get_alm_numba(
                V=self.V,
                k_amp=self.k_amp,
                phi=self.phi,
                k_amp_unique_index = self.k_amp_unique_index,
                theta_unique_index = self.theta_unique_index,
                k_max_list = self.k_max_list,
                l_max=self.l_max,
                lm_index = self.lm_index,
                sph_harm_no_phase = self.sph_harm_no_phase,
                delta_k_n = np.sqrt(self.scalar_pk_k3),
                transfer_delta_kl = self.transfer_delta_kl
            )

            cl_list[:, i] = c_l

            if save_alm: alm_list[i, :] = a_lm

            # Plot only first 4 realizations for check
            if i < 4 and plot_alm:
                fig2 = plt.figure()
                map = hp.alm2map(a_lm, 128, pol=False)
                hp.mollview(map, fig=fig2, remove_dip=True)
                fig2.savefig(self.root+'realizations/map_{}.pdf'.format(i))

        if save_alm: np.save(self.root+'realizations/realizations_Lx_{}_Ly_{}_Lz_{}_lmax_{}_num_{}.npy'.format(
            int(self.Lx),
            int(self.Ly),
            int(self.Lz),
            l_max,
            it),
            alm_list)

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
        np.save(self.root+'realizations/c_TT_Lx_{}_lmax_{}.npy'.format(int(self.Lx), self.l_max), self.C_TT)
        plt.legend()
        plt.yscale('log')
        plt.ylabel(r'$C^{TT}_{\ell} \,\, [\mu K^2]$ ')
        plt.xlabel(r'$\ell$')
        plt.title(r'$L={{{}}}$ Mpc'.format(int(L)))
        plt.savefig(self.root+'figs/tmp_{}_Lx_{}.pdf'.format(self.fig_name, int(self.Lx)))

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
        plt.savefig(self.root+'figs/c_tt_off_diag_L_{}_lmax_{}.pdf'.format(int(self.L/1000), self.l_max))

    def make_realization_c_lmlpmp_cholesky(self):
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
        plt.savefig(self.root+'realizations/map_L_{}_lmax_{}.pdf'.format(int(self.L/1000), self.l_max))

    def plot_c_l_and_realizations(self, c_l_a=None):
        l_min=3
        l_max = self.l_max
        ell = np.arange(l_min, self.l_max+1)
        correct_D_l = get_D_l(self.powers[:self.l_max+1, 0])

        cosmic_variance = np.array([2 * self.powers[l, 0]**2 / (2*l+1) for l in range(self.l_max+1)])
        D_l_cv = get_D_l(sqrt(cosmic_variance))

        plt.figure()
        plt.plot(ell, get_D_l(self.C_TT_diag)[l_min:self.l_max+1], linewidth=4, label='Lx={} True C_l'.format(int(self.Lx)))
        plt.plot(ell, correct_D_l[l_min:self.l_max+1], linewidth=4, label='CAMB')

        plt.fill_between(ell, (correct_D_l - D_l_cv)[l_min:], (correct_D_l + D_l_cv)[3:], color='grey', alpha=0.5)

        for i in range(2):
            plt.plot(ell, get_D_l(c_l_a[:, i])[l_min:self.l_max+1], label='Realization {}'.format(i), alpha=0.3)

        plt.legend()
        plt.ylabel(r'$\ell (\ell+1)C^{TT}_\ell / 2\pi \, [\mu K^2]$')
        plt.xlabel(r'$\ell$')
        plt.xscale('log')
        plt.savefig(self.root+'figs/power_spectrum.pdf')

    def get_transfer_functions(self):
        # Get all the transfer functions
        
        num_k_amp_unique = self.k_amp_unique.size
        transfer_delta_kl = np.zeros((num_k_amp_unique, self.l_max+1))

        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting transfer functions')
        args = zip(np.arange(num_k_amp_unique), repeat(self.l_max), repeat(self.k_amp_unique), repeat(self.k_max_list), repeat(self.transfer_interpolate_k_l_list), repeat(self.k_list))
        with multiprocessing.Pool(processes=ncpus) as pool:
            transfer_delta_kl = np.array(pool.starmap(transfer_parallel, tqdm(args, total=num_k_amp_unique)))
            print('Size of transfer function: {} MB.'.format(round(getsizeof(transfer_delta_kl) / 1024 / 1024,2)), '\n')
            return transfer_delta_kl

    def get_sph_harm(self):
        # Get all the spherical harmonics without phase (phi=0)
        # We use multiprocessing to make this fast
        num_k_indices = self.k_amp.size
        num_l_m = int((self.l_max + 1)*(self.l_max + 2)/2)
        theta = self.theta

        # We only find Y_lm for unique theta elements. We don't want to recalculate Y_lm unnecessarily
        unique_theta_length = self.theta_unique.size
        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting spherical harmonics')
        args = zip(np.arange(unique_theta_length), repeat(self.l_max), repeat(self.theta_unique), repeat(self.lm_index), repeat(num_l_m))
        with multiprocessing.Pool(processes=ncpus) as pool:
            sph_harm_no_phase = np.array(pool.starmap(get_sph_harm_parallel, tqdm(args, total=unique_theta_length)), dtype=np.float32)
            print('The spherical harmonics array is', round(getsizeof(sph_harm_no_phase) / 1024 / 1024,2), 'MB \n')
            return sph_harm_no_phase


def get_sph_harm_parallel(i, l_max, theta, lm_index, num_l_m):
  # Get the spherical harmonics with no phase (phi=0) for a given index i
  sph_harm_no_phase_i = np.zeros(num_l_m, dtype=np.float64)
  theta_cur = theta[i]
  all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, theta_cur, 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
  for l in range(l_max+1):
      for m in range(l+1):
          cur_index = lm_index[l, m]
          sph_harm_no_phase_i[cur_index] = all_sph_harm_no_phase[0, l, m]      
  return sph_harm_no_phase_i

def transfer_parallel(i, l_max, k_amp, k_max_list, transfer_interpolate_k_l_list, k_list):
  # Get the transfer function for a given index i
  cur_transfer_i = np.zeros(l_max+1)
  k = k_amp[i]
  min_k = min(k_list)
  for l in range(2, l_max+1):  
    cur_transfer_i[l] = transfer_interpolate_k_l_list[l](k)
  return cur_transfer_i

#@njit(parallel=True)
def get_k_theta_index_repeat(k_amp, theta):
    # k and theta often repeats themselves in the full list of allowed wavenumber list
    # So we want to know when they have repeated values so that we dont have to
    # recalculate spherical harmonics for example.

    # The unique lists are the lists of only unique elements
    # The unique_index lists are the indices going form the full parameter list to the
    # unique list.
    # For example:
    # k = [1, 2, 3, 1, 2, 5]
    # k_unique = [1, 2, 3, 5]
    # k_unique_index = [0, 1, 2, 0, 1, 3]
    
    k_repeat_dict = defaultdict(list)
    theta_repeat_dict = defaultdict(list)
    length = theta.size

    print('Getting repeated theta and k elements')
    for i in tqdm(range(length)):
        theta_repeat_dict[np.round(theta[i], decimals=7)].append(i)
        k_repeat_dict[np.round(k_amp[i], decimals=7)].append(i)

    theta_unique_dict = np.zeros(np.fromiter(theta_repeat_dict.keys(), dtype=float).size)
    theta_unique_index_dict = np.zeros(length, dtype=int)
    index = 0
    for key, value in theta_repeat_dict.items():
        theta_unique_index_dict[value] = index
        theta_unique_dict[index] = key
        index += 1

    k_amp_unique_dict = np.zeros(np.fromiter(k_repeat_dict.keys(), dtype=float).size)
    k_amp_unique_index_dict = np.zeros(length, dtype=int)
    index = 0
    for key, value in k_repeat_dict.items():
        k_amp_unique_index_dict[value] = index
        k_amp_unique_dict[index] = key
        index += 1

    print('Ratio of unique theta:', theta_unique_dict.size / length)
    print('Ratio of unique |k|:', k_amp_unique_dict.size / length)

    return k_amp_unique_dict, k_amp_unique_index_dict, theta_unique_dict, theta_unique_index_dict
@njit
def do_integrand_pre_processing(unique_k_amp, scalar_pk_k3, transfer_delta_kl, l_max):
    # Calculating P(k) / k^3 * Delta_ell(k) * Delta_ell'(k) 
    num_k_amp = unique_k_amp.size
    integrand = np.zeros((num_k_amp, l_max+1, l_max+1))
    for i in prange(num_k_amp):
        scalar_pk_k3_cur = scalar_pk_k3[i]
        for l in range(2, l_max+1):
            scalar_pk_k3_transfer_delta_kl = scalar_pk_k3_cur * transfer_delta_kl[i, l]
            for lp in range(l_max+1):
                integrand[i, l, lp] = scalar_pk_k3_transfer_delta_kl * transfer_delta_kl[i, lp]
    return integrand

def get_alm_numba(
    V,
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
    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    a_lm = np.zeros(num_l_m, dtype=np.complex128)
    num_indices = k_amp.size

    
    ncpus = multiprocessing.cpu_count()
    index_thread_split = np.arange(0, num_indices, int(np.ceil(num_indices/ncpus)))
    size = index_thread_split.size
    
    os.environ['OMP_NUM_THREADS'] = '1'
    
    index_thread_split = np.append(index_thread_split, num_indices)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(size):
        # Spawn a process for each cpu that goes through parts of the summation each
        min_index_list = index_thread_split[i]
        max_index_list = index_thread_split[i+1]

        args = (
            i,
            return_dict,
            min_index_list,
            max_index_list,
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
        p = multiprocessing.Process(target=get_alm_per_process, args=args)
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # The final a_lm that is the sum of contribution from each process    
    a_lm = sum(return_dict.values())

    # Add i^\ell
    for l in range(2, l_max+1):
        for m in range(l+1):
            lm_index_cur = lm_index[l, m]
            a_lm[lm_index_cur] *= np.power(1j, l)
    
    # Add prefactor
    a_lm *= np.sqrt(2*np.pi**2) * 4*np.pi / np.sqrt(V)

    # Get corresponding observed c_l (sigma_l?)
    c_l = get_c_l_from_a_lm(a_lm, l_max)
    return a_lm, c_l

def get_alm_per_process(
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

@njit(parallel=False, fastmath=True)
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
    only_diag
    ):
    # This calculates the full covariance. If only_diag==True
    # then it only finds the diagonal values, l, m = lp, mp

    # Not multiprocessed yet, but implementation would be similar to
    # the a_lm realization procedure

    num_l_m = int((l_max + 1)*(l_max + 2)/2)
    C_lmlpmp = np.zeros((num_l_m, num_l_m), dtype=np.complex128)

    num_indices = k_amp.size
    for i in range(num_indices):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      sph_harm_index = theta_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list = np.exp(-1j * phi[i] * m_list)

      for l in range(l_max+1):
        if k_amp_cur > k_max_list[l]:
            continue

        for m in range(l + 1):
            lm_index_cur = lm_index[l, m]

            sph_harm_no_phase_lm = sph_harm_no_phase[sph_harm_index, lm_index_cur]

            if only_diag:
                C_lmlpmp[lm_index_cur, lm_index_cur] += integrand[k_unique_index_cur, l, l] * sph_harm_no_phase_lm**2
            else:
                for l_p in range(l_max+1):
                    integrand_il = integrand[k_unique_index_cur, l, l_p] * np.power(1j, l_p-l)
                    
                    for m_p in range(l_p + 1):
                        lm_p_index_cur = lm_index[l_p, m_p]

                        sph_harm_no_phase_lm_p = sph_harm_no_phase[sph_harm_index, lm_p_index_cur]


                        C_lmlpmp[lm_index_cur, lm_p_index_cur] += integrand_il * sph_harm_no_phase_lm * sph_harm_no_phase_lm_p \
                        * phase_list[m_p] / phase_list[m]
    C_lmlpmp *= 2*pi**2 * (4*pi)**2 / V
    return C_lmlpmp

#Things to add:
# 1) Calculating all off-diagonal elements take time, maybe add a bool that if true only calculates diagonal elements
# 2) A way to generalize for different topologies. Class inheritance?