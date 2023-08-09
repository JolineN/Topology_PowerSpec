#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'
from src.E2 import E2
import numpy as np

import parameter_files.default_E2 as parameter_file_E2

param_E2 = parameter_file_E2.parameter


a = E2(param=param_E2)

import time

start = time.time()
cur_kl, cur_a_t = a.calculate_exact_kl_divergence(parallel_cov = True)
end = time.time()
print(cur_kl, cur_a_t, 'Elapsed time parallel:', end-start)

start = time.time()
cur_kl, cur_a_t = a.calculate_exact_kl_divergence(parallel_cov = False)
end = time.time()
print(cur_kl, cur_a_t, 'Elapsed time non-parallel:', end-start)


    