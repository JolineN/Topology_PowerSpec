from locale import normalize
import camb
import os
import numpy as np
from numpy import pi, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import time
import healpy as hp
from .tools import *
from sys import getsizeof

class Topology:
    def __init__(self, param, debug=True):
        self.param = param
        self.topology = param['topology']
        self.l_max = param['l_max']
        self.c_l_accuracy = param['c_l_accuracy']
        self.do_polarization = param['do_polarization']
        self.get_initial_kmax_list()

        self.fig_name = 'l_max_{}'.format(self.l_max)
        self.debug = debug

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

        # Get the transfer functions and put them into lists of interpolate objects.
        # We can therefore evaluate the transfer function for all possibel |k| later
        assert(self.ell_list[0] == 2 and self.ell_list[l_max-2] == l_max)
        transfer_T_interpolate_k_l_list = np.empty_like(scipy.interpolate.interp1d, shape=l_max+1)
        transfer_E_interpolate_k_l_list = np.empty_like(scipy.interpolate.interp1d, shape=l_max+1)
        for l in range(2, l_max+1):
            transfer_T_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, self.transfer_data[0, l-2, :], kind='cubic') 
            transfer_E_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, self.transfer_data[1, l-2, :], kind='cubic') 
        self.transfer_T_interpolate_k_l_list = transfer_T_interpolate_k_l_list
        self.transfer_E_interpolate_k_l_list = transfer_E_interpolate_k_l_list

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
        transfer_T_delta_kl = self.get_transfer_functions(transfer_T_interpolate_k_l_list)
        self.transfer_T_delta_kl = transfer_T_delta_kl
        if self.do_polarization:
            transfer_E_delta_kl = self.get_transfer_functions(transfer_E_interpolate_k_l_list)
            self.transfer_E_delta_kl = transfer_E_delta_kl
        self.scalar_pk_k3 = scalar_pk_k3

        

        # Get P(k)/k^3 * Delta_ell(k) * Delta_ell'(k)
        start_time = time.time()
        integrand_TT = do_integrand_pre_processing(k_amp_unique, scalar_pk_k3, transfer_T_delta_kl, self.l_max)
        self.integrand_TT = integrand_TT
        if self.do_polarization:
            integrand_EE = do_integrand_pre_processing(k_amp_unique, scalar_pk_k3, transfer_E_delta_kl, self.l_max)
            self.integrand_EE = integrand_EE
        print('Size of integrand: {} MB. Time to calculate integrand: {} s'.format(round(integrand_TT.size * integrand_TT.itemsize / 1024 / 1024, 2), time.time()-start_time))
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
        
        k_max_list_one = np.arange(0, l_max+1) * k_max / l_max
        self.k_max_list = np.stack((k_max_list_one, k_max_list_one))

    def get_kmax_as_function_of_ell(self):
        # Get k_max as a function of multipole ell
        # We use cumulative trapezoid to find the k_value where we reach
        # the wanted accuracy

        l_max = self.l_max

        # Do the integration up to k=0.08. This should be fine for ell=<250 and accuracy<=0.99
        print('\nFinding k_max as a function of ell')
        plt.figure()
        for l in tqdm(range(2, l_max+1)):            
            k_list = np.linspace(min(self.k_list), max(self.k_list), 200000)
            integrand = 4*pi * self.pars.scalar_power(k_list) * self.transfer_T_interpolate_k_l_list[l](k_list)**2 / k_list
            integrand_EE = 4*pi * self.pars.scalar_power(k_list) * self.transfer_E_interpolate_k_l_list[l](k_list)**2 / k_list

            cumulative_c_l_ratio = scipy.integrate.cumulative_trapezoid(y=integrand, x=k_list) / self.powers[l, 0]
            cumulative_c_l_EE_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_EE, x=k_list) / self.powers[l, 1]

            #print(max(cumulative_c_l_EE_ratio), max(scipy.integrate.cumulative_trapezoid(y=integrand_EE, x=k_list)), self.powers[l, 1], l)
            
            index_closest_to_accuracy_target_TT = (np.abs(cumulative_c_l_ratio -  self.c_l_accuracy)).argmin()
            index_closest_to_accuracy_target_EE = (np.abs(cumulative_c_l_EE_ratio -  self.c_l_accuracy)).argmin()
           
            self.k_max_list[0, l] = k_list[index_closest_to_accuracy_target_TT]
            self.k_max_list[1, l] = k_list[index_closest_to_accuracy_target_EE]

            if l == 2 or l == 3 or l == 4 or l == 5:
                #plt.figure()
                
                plt.plot(k_list[:index_closest_to_accuracy_target_TT]*self.Lx/(2*np.pi), (integrand/max(integrand))[:index_closest_to_accuracy_target_TT], label=r'TT $\ell$={}'.format(l), alpha=0.5)
                
                plt.scatter(k_list[index_closest_to_accuracy_target_EE], 1)
            #print(cumulative_c_l_ratio[index_closest_to_accuracy_target], 'for ell =', l, 'k=', self.k_max_list[l])
        plt.title('Integrand for $C_\ell$')
        plt.legend()
        plt.savefig(self.root+'figs/transfer_squared_P_k_k_ell_space_100.pdf')
        np.save(self.root+'k_max_list.npy', self.k_max_list)

        plt.figure()
        plt.plot(self.k_max_list[0, :])
        plt.ylabel(r'$k_{max}$')
        plt.xlabel(r'$\ell$')
        plt.savefig(self.root+'k_max.pdf')

        plt.figure()
        plt.title('Integrand for $C_\ell$')
        plt.plot(k_list[:index_closest_to_accuracy_target_TT]*self.Lx/(2*np.pi), integrand[:index_closest_to_accuracy_target_TT])
        plt.savefig(self.root+'figs/transfer_squared_P_k_k_ell{}.pdf'.format(l))

        print('\n********************************************')
        print('Done. k_max for TT ell_max =', self.k_max_list[0, l_max], 'k_max for EE ell_max =', self.k_max_list[1, l_max])
        print('********************************************\n')
    
    def calculate_c_lmlpmp(self, only_diag=False, normalize=False, plotting = True, plot_param={}):
        # Calculaget_c_lmlpmp, the off-diagonal and on-diagonal power spectrum
        l_max = self.l_max

        time_start = time.time()
        
        print('\nCalculating covariance matrix')


        if only_diag:
            # Only care about the diagonal elements
            # Probably because you want the diagonal power spectrum
            ell_range = np.array([0, self.l_max])
            ell_p_range = np.array([], dtype=np.int64)

            C_TT_lmlpmp = self.get_c_lmlpmp_top(
                ell_range = ell_range,
                ell_p_range = ell_p_range
            )
            print('Time to calculate C_TT with l_max={} and c_l_ratio={}:'.format(l_max, self.c_l_accuracy), time.time()-time_start, 'seconds with Numba')            
            
            self.C_TT_diag = get_c_l_from_c_lmlpmp(C_TT_lmlpmp, self.l_max)
        elif plotting == False:
            ell_range = np.array([2, self.l_max])
            ell_p_range = np.array([2, self.l_max])
            C_TT_lmlpmp = self.get_c_lmlpmp_top(
                ell_range = ell_range,
                ell_p_range = ell_p_range
            )
        else:
            num_plots = plot_param['l_ranges'][:, 0].size
            ell = np.arange(2, self.l_max+1, dtype=np.int32)
            plt.figure()
            plt.plot(ell, ell, color='black', linestyle='--')
            for i in range(num_plots):
                rectangle = plt.Rectangle(
                    (plot_param['l_ranges'][i, 0], plot_param['lp_ranges'][i, 0]),
                    width = plot_param['l_ranges'][i, 1]-plot_param['l_ranges'][i, 0],
                    height = plot_param['lp_ranges'][i, 1]-plot_param['lp_ranges'][i, 0],
                    edgecolor='black',
                    facecolor=(0, 0, 0, 0),
                    )
                plt.gca().add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width()/2.0
                cy = ry + rectangle.get_height()
                plt.gca().annotate(str(i+1), (cx, cy), color='black', weight='bold', fontsize=20, ha='center', va='bottom')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r"$\ell'$")
            plt.xticks([2, 5, 10, 15, 20])
            plt.yticks([2, 5, 10, 15, 20])
            plt.savefig(self.root+'figs/cov_border.pdf')

            if num_plots == 4:
                ncols=2
                nrows=2
            else:
                ncols=1
                nrows = num_plots
            fig, ax = plt.subplots(ncols = ncols, nrows = nrows, dpi=500)
            axs = np.array(ax)

            for i, ax in enumerate(tqdm(axs.reshape(-1))):
                l_min = plot_param['l_ranges'][i, 0]
                l_max = plot_param['l_ranges'][i, 1]
                lp_min = plot_param['lp_ranges'][i, 0]
                lp_max = plot_param['lp_ranges'][i, 1]
                # Make sure the l_ranges do not overlap!
                ell_range = np.array(plot_param['l_ranges'][i, :])
                ell_p_range = np.array(plot_param['lp_ranges'][i, :])
                C_TT_lmlpmp = self.get_c_lmlpmp_top(
                    ell_range = ell_range,
                    ell_p_range = ell_p_range
                )
                if l_max > 30 or lp_max > 30:
                    normalized_clmlpmp = normalize_c_lmlpmp(
                        C_TT_lmlpmp, 
                        self.powers[:, 0], 
                        l_min=l_min, 
                        l_max=l_max, 
                        lp_min = lp_min,
                        lp_max = lp_max)

                    np.save(self.root+'corr_matrix_l_{}_{}_lp_{}_{}.npy'.format(
                        l_min, l_max,
                        lp_min, lp_max,
                    ), normalized_clmlpmp)
                
                im = self.do_cov_sub_plot(ax, normalize, i, C_TT_lmlpmp, ell_range, ell_p_range)
                
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.8, 0.05, 0.05, 0.85])
            fig.colorbar(im, cax=cbar_ax)
            fig.subplots_adjust(hspace=.4, wspace=-0.1)
            #fig.tight_layout()
            plt.savefig(
                self.root+'figs/c_tt_offdiagonal_{}.pdf'.format('high_ell' if ell_range[1]>50 or ell_p_range[1]> 50 else 'low_ell'),
                bbox_inches='tight')
        return C_TT_lmlpmp

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
            a_lm, c_l = self.get_alm_numba(
                V=self.V,
                k_amp=self.k_amp,
                phi=self.phi,
                k_amp_unique_index = self.k_amp_unique_index,
                theta_unique_index = self.theta_unique_index,
                k_max_list = self.k_max_list[0, :],
                l_max=self.l_max,
                lm_index = self.lm_index,
                sph_harm_no_phase = self.sph_harm_no_phase,
                delta_k_n = np.sqrt(self.scalar_pk_k3),
                transfer_T_delta_kl = self.transfer_T_delta_kl
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

    def make_euclidean_realizations(self, plot_alms=False):
        it=1000
        l_max = self.l_max
        alm_list = np.zeros((it, int((l_max + 1)*(l_max + 2)/2)), dtype=np.complex128)
        for i in range(it):
            map, alm = hp.synfast(self.powers[:, 0], nside=128, lmax=l_max, alm=True, pol=False)
            alm_list[i, :] = alm

            if i < 4 and plot_alms:
                plt.figure()
                hp.mollview(map, remove_dip=True)
                plt.savefig(self.root+'realizations/map_L_infty_lmax_{}_{}.pdf'.format(l_max, i))

        np.save(self.root+'realizations/realizations_L_infty_lmax_{}_num_{}.npy'.format(l_max, it), alm_list)

    def calculate_exact_kl_divergence(self):
        print('Calculating KL divergence')

        c_lmlpmp_ordered = self.calculate_c_lmlpmp(only_diag=False, normalize=False, plotting = False)
        A_ssp = normalize_c_lmlpmp(c_lmlpmp_ordered, self.powers[:, 0], cl_accuracy = self.c_l_accuracy, l_min=2, lp_min=2, l_max=self.l_max, lp_max=self.l_max)

        w, _ = np.linalg.eig(A_ssp)
        t = 0
        for eig in w:
            t += (np.log(np.abs(eig)) + 1/eig - 1)/2
            
        fig = plt.figure()
        A_ssp_abs = np.where(np.abs(A_ssp) < 1e-4, 1e-4, np.abs(A_ssp))
        im = plt.imshow(A_ssp_abs.T, norm=LogNorm(), origin='lower')
        
        cbar_ax = fig.add_axes([0.8, 0.05, 0.05, 0.85])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.savefig(self.root+'tmp_ani.pdf')

        (_, logdet_norm) = np.linalg.slogdet(A_ssp)
        logdet_norm = np.abs(logdet_norm)

        res = np.sum(np.abs(1/np.diag(A_ssp)))

        tot = logdet_norm / 2  + res / 2 - self.l_max * (self.l_max+2)/2 + 3/2
        print('tot, det, res, leftovers', tot, logdet_norm/2, res/2, - self.l_max * (self.l_max+2)/2 + 3/2)
        print('old way vs new way', tot, t)

        np.fill_diagonal(A_ssp, 0)

        fig = plt.figure()
        A_ssp_abs = np.where(np.abs(A_ssp) < 1e-4, 1e-4, np.abs(A_ssp))
        im = plt.imshow(A_ssp_abs.T, norm=LogNorm(), origin='lower')
        
        cbar_ax = fig.add_axes([0.8, 0.05, 0.05, 0.85])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.savefig(self.root+'tmp_ani_no_diag.pdf')

        a_t = np.sqrt(np.sum(np.abs(A_ssp)**2))
        print('arthur statistics:', a_t)
        return np.real(t), np.real(a_t)

    def sampled_kosowski_statistics(self, N_s = 400, num_times=1):
        print('Sampling Kosowski statistics')

        kosowski_list = np.zeros(num_times)
        for i in tqdm(range(num_times)):
            kosowski_list[i] = self.get_kosowski_stat_top(N_s)
            print(kosowski_list[i])

        return kosowski_list

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
        l_min=2
        l_max = self.l_max
        ell = np.arange(l_min, l_max+1)
        correct_D_l = get_D_l(self.powers[:l_max+1, 0])

        cosmic_variance = np.array([2 * self.powers[l, 0]**2 / (2*l+1) for l in range(self.l_max+1)])
        D_l_cv = get_D_l(sqrt(cosmic_variance))

        plt.figure()
        plt.plot(ell, get_D_l(self.C_TT_diag)[l_min:self.l_max+1], linewidth=4, label='Lx={} True C_l'.format(int(self.Lx)))
        plt.plot(ell, correct_D_l[l_min:self.l_max+1], linewidth=4, label='CAMB')

        plt.fill_between(ell, (correct_D_l - D_l_cv)[l_min:], (correct_D_l + D_l_cv)[l_min:], color='grey', alpha=0.5)

        for i in range(2):
            plt.plot(ell, get_D_l(c_l_a[:, i])[l_min:self.l_max+1], label='Realization {}'.format(i), alpha=0.3)

        plt.legend()
        plt.ylabel(r'$\ell (\ell+1)C^{TT}_\ell / 2\pi \, [\mu K^2]$')
        plt.xlabel(r'$\ell$')
        plt.xscale('log')
        plt.savefig(self.root+'figs/power_spectrum.pdf')

    def get_transfer_functions(self, transfer_interpolate_k_l_list):
        # Get all the transfer functions
        
        num_k_amp_unique = self.k_amp_unique.size
        transfer_delta_kl = np.zeros((num_k_amp_unique, self.l_max+1))

        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting transfer functions')
        args = zip(np.arange(num_k_amp_unique), repeat(self.l_max), repeat(self.k_amp_unique), repeat(transfer_interpolate_k_l_list))
        with multiprocessing.Pool(processes=ncpus) as pool:
            transfer_T_delta_kl = np.array(pool.starmap(transfer_parallel, tqdm(args, total=num_k_amp_unique)))
            print('Size of transfer function: {} MB.'.format(round(getsizeof(transfer_delta_kl) / 1024 / 1024,2)), '\n')
            return transfer_T_delta_kl

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

    def do_cov_sub_plot(self, ax, normalize, ax_index, C_TT_order, ell_range, ell_p_range):
        if normalize:
            C_TT_order = normalize_c_lmlpmp(C_TT_order, self.powers[:, 0], l_min=ell_range[0], l_max =ell_range[1], lp_min=ell_p_range[0], lp_max=ell_p_range[1])
        
        l_min = ell_range[0]
        l_max = ell_range[1]
        lp_min = ell_p_range[0]
        lp_max = ell_p_range[1]
        C_TT_order = np.where(np.abs(C_TT_order) < 1e-12, 1e-12, np.abs(C_TT_order))

        ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
        ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])

        axim = ax.imshow(C_TT_order.T, cmap='inferno', norm=LogNorm(), origin='lower', interpolation = 'nearest')
        
        if l_max-l_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_xticks(ell_to_s_map[jump])
            ax.set_xticklabels(np.arange(l_min, l_max+1)[jump])
        else:
            ax.set_xticks(ell_to_s_map)
            ax.set_xticklabels(np.arange(l_min, l_max+1))

        if lp_max-lp_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_yticks(ellp_to_s_map[jump])
            ax.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
        else:
            ax.set_yticks(ellp_to_s_map)
            ax.set_yticklabels(np.arange(lp_min, lp_max+1))

        ax.set_xlim([0, (l_max+1)*(l_max+2) - (l_max+1) - l_min**2 - 1])
        ax.set_ylim([0, (lp_max+1)*(lp_max+2) - (lp_max+1) - lp_min**2 - 1])
        
        if lp_max > 50 or l_max > 50:
            ax.set_title(str(ax_index+5), weight='bold', fontsize='20')
        else:
            ax.set_title(str(ax_index+1), weight='bold', fontsize='20')
        
        if ax_index == 3 or ax_index == 2:
            ax.set_xlabel(r'$\ell$')
        if ax_index == 0 or ax_index == 2:  
            ax.set_ylabel(r"$\ell'$")
        if normalize:
            axim.set_clim(1e-8, 1e0)
        else:
            axim.set_clim(1e-5, 1e2)
        return axim

    def get_alm_numba(
        self,
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
        transfer_T_delta_kl
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
                transfer_T_delta_kl
            )
            p = multiprocessing.Process(target=self.get_alm_per_process, args=args)
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