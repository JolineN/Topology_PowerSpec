# Parameter file
import numpy as np

parameter = {
  # TOPOLOGY PARAMETERS
  'topology': 'E1',               # Self-explanatory
  'Lx': 1.15,                     # In diameter of LSS
  'Ly': 1.15,                     # In diameter of LSS
  'Lz': 1.15,                     # In diameter of LSS

  'beta': 90,
  'alpha': 90,


  # OTHER PARAMETERS
  'c_l_accuracy': 0.99,           # Determines what k_max is. Determined by doing the integration up to a k_max
                                  # that gives 'c_l_accuracy' times CAMB c_l ouput
  'l_max': 20,                    # Self-explanatory
  'do_polarization': False,

  #GENERAL POWER SPECTRUM PARAMETERS
  'A_s': 2e-9,
  'n_s': 0.965,

  'number_of_a_lm_realizations': 1,
}