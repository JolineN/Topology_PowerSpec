# Parameter file
import numpy as np

parameter = {
  # TOPOLOGY PARAMETERS
  'topology': 'E1',              # Self-explanatory
  'Lx': 0.12,                     # In diameter of LSS
  'Ly': 0.12,                     # In diameter of LSS
  'Lz': 0.12,                     # In diameter of LSS
  'beta': 90 * np.pi/180,


  # OTHER PARAMETERS
  'c_l_accuracy': 0.99,            # Determines what k_max is. Determined by doing the integration up to a k_max
                                  # that gives 'c_l_accuracy' times CAMB c_l ouput
  'l_max': 250,                   # Self-explanatory
}