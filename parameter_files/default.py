# Parameter file
parameter = {
  'topology': 'E1',               # Self-explanatory

  'Lx': 0.5,                       # In diameter of LSS
  'Ly': 0.5,                       # In diameter of LSS
  'Lz': 0.5,                       # In diameter of LSS

  'c_l_accuracy': 0.90,           # Determines what n_max is. Determined by doing the integration up to a k_max
                                  # that gives 'c_l_accuracy' times CAMB c_l ouput

  'l_max': 150,                   # Self-explanatory

  'initial_n_max_for_l_max': 30,  # This is your best guess for n_max at l_max.
                                  # Code run fastest when this is such that 'c_l_accuracy' is barely achieved
                                  # for l_max

  'debug': False                  # Could be used in the future
}