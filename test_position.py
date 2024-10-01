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
#import parameter_files.default as parameter_file
import parameter_files.default_E1 as parameter_file
param = parameter_file.parameter

#specify the initial power spectrum of the E_k topology (E18 uses the standard power law)
#available power spec are: powlaw (default), local (amp, location, width), 
#wavepacket (amp, freq, width), logosci (amp,freq)
for loc in [1e-4, 5e-4, 1e-3, 5e-3]:
  powerparam={
  'powerspec': 'local',
  'amp': 1.,
  'location': loc,
  'width': 0.03,
  'freq': 10,
  }

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

  # Create 2 realizations
  c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = False)
  #print(c_l_a.shape)
  # Calculate the diagonal covariance matrix
  a.calculate_c_lmlpmp(
    only_diag=True
  )

  # Plot the diagonal power spectrum and the realizations
  # Good to see if there are any obvious bugs
  a.plot_c_l_and_realizations(c_l_a=None)

  _, norm_c = a.calculate_c_lmlpmp(
    only_diag=False,
    normalize=True,
    save_cov = True,
    plot_param={
      'l_ranges': np.array([[2, 10]]),
      'lp_ranges': np.array([[2, 10]]),
    }
  )
  cur_kl, _, _ = a.calculate_exact_kl_divergence()
  print('KL:', cur_kl)


