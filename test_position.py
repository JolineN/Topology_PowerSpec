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


ampl=[0.2,0.6,1.]
kmin= [1e-4,2e-4,3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,1e-3, 1.5e-3, 2e-3]
width=0.2e-4
length=0.5

KLs=np.zeros((len(ampl), len(kmin),2))
path_for_KL="./MoveE1_l20_05_width02.npy"

#specify the initial power spectrum of the E_k topology (E18 uses the standard power law)
#available power spec are: powlaw (default), local (amp, location, width), 
#wavepacket (amp, freq, width), logosci (amp,freq)
for amp in ampl:
  for k_min in kmin:
    powerparam={
    'powerspec': 'powlaw_mov',
    'amp': amp,
    'pow_min': k_min,
    'pow_max': k_min+width,
    'location': 0.0001,
    'width': 0.01,
    'freq': 10,
    'k_cutoff': 0.001,
    'alpha_cutoff': 3.,
   }
    param['Lx'] = length
    param['Ly'] = length
    param['Lz'] = length

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
    
    _, _ = a.calculate_c_lmlpmp(
    only_diag=False,
    normalize=True,
    save_cov = True,
    plot_param={
      'l_ranges': np.array([[2, 20]]),
      'lp_ranges': np.array([[2, 20]]),
      'powerspec': 'powlaw_mov',
      'amplitude': amp,
      'k_value': k_min+width/2,
    }
    )
    
    forward_kl, backward_kl, _ = a.calculate_exact_kl_divergence()
    KLs[ampl.index(amp), kmin.index(k_min),0]=forward_kl
    KLs[ampl.index(amp), kmin.index(k_min),1]=backward_kl
    np.save(path_for_KL,KLs)
    #print('KL:', cur_kl)