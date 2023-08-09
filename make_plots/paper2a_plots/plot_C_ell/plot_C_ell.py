import camb
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
plt.rc('font', size=15)
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16, labelsize=17)
plt.rc('lines', linewidth=2, markersize=6)
plt.rc('legend', fontsize=12)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

l_max = 30
l_min = 2

def get_D_l(c_l):
    return np.array([c_l[l] * l * (l+1) / (2*np.pi) for l in range(c_l.size)])

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(l_max)

# More accurate transfer functions. Takes longer time to run
pars.set_accuracy(AccuracyBoost = 2, lAccuracyBoost = 2, lSampleBoost = 50)
pars.Accuracy.IntkAccuracyBoost = 2
pars.Accuracy.SourcekAccuracyBoost = 2

# Get the CAMB functions and save them
data = camb.get_transfer_functions(pars)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['lensed_scalar']

correct_D_l = get_D_l(powers[:l_max+1, 0])

cosmic_variance = np.array([2 * powers[l, 0]**2 / (2*l+1) for l in range(l_max+1)])
D_l_cv = get_D_l(sqrt(cosmic_variance))

ell = np.arange(l_min, l_max+1)

plt.figure(figsize=(6, 3.5))



labels=np.array([r'$E_1$', r'$E_2$',r'$E_3$', r'$E_4$', r'$E_5$', r'$E_6$'])
topology_list = np.array(['E1', 'E2', 'E3', 'E4', 'E5', 'E6'])
for i, top in enumerate(topology_list):
        C_TT_diag = np.load('{}_C_ell_diag.npy'.format(top))
        plt.plot(ell, get_D_l(C_TT_diag/0.99)[l_min:l_max+1], linewidth=4, label=labels[i], alpha=0.4)
plt.plot(ell, correct_D_l[l_min:l_max+1], linewidth=4, label=r'$\Lambda$CDM', linestyle='--', color='black')
plt.fill_between(ell, (correct_D_l - D_l_cv)[l_min:], (correct_D_l + D_l_cv)[l_min:], color='grey', alpha=0.3)


plt.legend(frameon=False, ncol=4)
plt.ylabel(r'$C^{E_i;TT}_{\ell} \ell(\ell+1)/2\pi\,\, [\mu\mathrm{K}^2]$')
plt.xlabel(r'$\ell$')
plt.xlim([l_min, l_max])
plt.savefig('C_ell.pdf')