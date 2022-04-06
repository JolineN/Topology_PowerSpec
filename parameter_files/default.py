# Parameter file
import numpy as np

parameter = {
  # TOPOLOGY PARAMETERS
  'topology': 'E1',              # Self-explanatory
  'Lx': 0.3,                     # In diameter of LSS
  'Ly': 0.3,                     # In diameter of LSS
  'Lz': 0.3,                     # In diameter of LSS
  'beta': 90 * np.pi/180,


  # OTHER PARAMETERS
  'c_l_accuracy': 0.99,           # Determines what k_max is. Determined by doing the integration up to a k_max
                                  # that gives 'c_l_accuracy' times CAMB c_l ouput
  'l_max': 250,                   # Self-explanatory
  'initial_n_max_for_l_max': 82,  # This is your best guess for n_max at l_max.
                                  # Code run fastest when this parameter is such that 'c_l_accuracy'
                                  # is barely achieved for l_max
}