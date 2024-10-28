#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1
from src.E1 import E1
from src.E2 import E2
from src.E3 import E3
from src.E4 import E4
from src.E5 import E5
from src.E6 import E6
from src.E7 import E7
import numpy as np

# Set parameter file
import parameter_files.default_E1 as parameter_file
param = parameter_file.parameter

length=0.5
l_maxx=30

ampl=[1., 5.]
kmin= [4.5e-4, 5e-4, 7e-4, 9e-4]
width=2.e-4


#specify the initial power spectrum of the E_k topology (E18 uses the standard power law)
#available power spec are: powlaw (default), local (amp, location, width), 
#wavepacket (amp, freq, width), logosci (amp,freq)
for amp in ampl:
  for k_min in kmin:
    powerparam={
    'powerspec': 'local',
    'amp': amp,
    'pow_min': 0.0001,
    'pow_max': 0.0002,
    'location': k_min,
    'width': 0.05,
    'freq': 10,
    'k_cutoff': 0.001,
    'alpha_cutoff': 3.,
   }
    param['topology'] = 'E1'
    param['Lx'] = length
    param['Ly'] = length
    param['Lz'] = length
    param['l_max'] = l_maxx

    path_for_a=f"./a_lm_E1_L{amp}_{k_min}.npy"
    path_for_c=f"./c_l_E1_L{amp}_{k_min}.npy"

    if param['topology'] == 'E1':
      a = E1(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E2':
      a = E2(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E3':
      a = E3(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E4':
      a = E4(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E5':
      a = E5(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E6':
      a = E6(param=param,powerparam=powerparam, make_run_folder=True)
    elif param['topology'] == 'E7':
      a = E7(param=param,powerparam=powerparam, make_run_folder=True)
    else:
      exit()

        
    #c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = True)
    # Calculate the diagonal covariance matrix
    a.calculate_c_lmlpmp(
      only_diag=True
    )


    _, _ = a.calculate_c_lmlpmp(
      only_diag=False,
      normalize=True,
      save_cov = True,
      plot_param={
        'l_ranges': np.array([[2, l_maxx]]),
        'lp_ranges': np.array([[2, l_maxx]]),
        'powerspec': 'powlaw_mov',
        'amplitude': amp,
        'k_value': k_min+width/2,
      }
    )
    a_lm_list, c_l_list = a.make_realization_c_lmlpmp_cholesky(10000)

    np.save(path_for_a,a_lm_list)
    np.save(path_for_c,c_l_list)


