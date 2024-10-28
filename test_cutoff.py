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

k_cut=[0.0001, 0.0004, 0.0006, 0.001]
alph=[2., 3.35, 4., 8.]

KLs=np.zeros((len(k_cut),len(alph),1))
path_for_KL="./CutoffE1_l20_1_5.npy"

#specify the initial power spectrum of the E_k topology (E18 uses the standard power law)
#available power spec are: powlaw (default), local (amp, location, width), 
#wavepacket (amp, freq, width), logosci (amp,freq), cutoff (k_cutoff, alpha)
for cut in k_cut:
  for alp in alph:
    powerparam={
    'powerspec': 'cutoff',
    'amp': 1.,
    'location': 0.001,
    'width': 0.03,
    'freq': 10.,
    'k_cutoff': cut,
    'alpha_cutoff': alp,
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
    
    cur_kl, _, _ = a.calculate_exact_kl_divergence()
    #print('KL:', cur_kl)
    KLs[k_cut.index(cut),alph.index(alp),0]=cur_kl
    np.save(path_for_KL,KLs)

    
