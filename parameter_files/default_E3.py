# Parameter file
import numpy as np

parameter = {
  # TOPOLOGY PARAMETERS
  'topology': 'E3',              # Self-explanatory
  'Lx': 1,                     # In diameter of LSS
  'Ly': 1,                     # In diameter of LSS
  'Lz': 1,                     # In diameter of LSS

  'x0': np.array([0,0,0], dtype=np.float64),
  'beta': 90,
  'alpha': 90,
  'gamma': 0,


  # OTHER PARAMETERS
  'c_l_accuracy': 0.95,          # Determines what k_max is. Determined by doing the integration up to a k_max
                                  # that gives 'c_l_accuracy' times CAMB c_l ouput
  'l_max': 20,                    # Self-explanatory
  'do_polarization': False,

  #POWER SPECTRUM PARAMETERS
  'A_s': 2e-9,
  'n_s': 0.965,

  'number_of_a_lm_realizations': 1,
}