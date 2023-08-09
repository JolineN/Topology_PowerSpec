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
param['topology'] = 'E1'
param['Lx'] = 0.4
param['Ly'] = 0.4
param['Lz'] = 0.4
param['beta'] = 90.0
param['alpha'] = 90.0
param['gamma'] = 0.0
param['l_max'] = 200
param['x0'] = np.array([0, 0, 0], dtype=np.float64)
param['number_of_a_lm_realizations'] = 2

if param['topology'] == 'E1':
  a = E1(param=param, make_run_folder=True)
elif param['topology'] == 'E2':
  a = E2(param=param, make_run_folder=True)
elif param['topology'] == 'E3':
  a = E3(param=param, make_run_folder=True)
elif param['topology'] == 'E4':
  a = E4(param=param, make_run_folder=True)
elif param['topology'] == 'E5':
  a = E5(param=param, make_run_folder=True)
elif param['topology'] == 'E6':
  a = E6(param=param, make_run_folder=True)
elif param['topology'] == 'E7':
  a = E7(param=param, make_run_folder=True)
else:
  exit()

# Create 2 realizations
_, c_l_a = a.make_alm_realizations(plot_alm=True, save_alm = True)
print(c_l_a.shape)
a.calculate_c_lmlpmp(
  only_diag=True
)
a.plot_c_l_and_realizations(c_l_a=c_l_a, plot_average_real=True)
