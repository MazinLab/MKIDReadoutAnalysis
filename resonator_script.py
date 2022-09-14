from resonator import Resonator

res_params = {'f0': 4.0012e9,  # resonance frequency [Hz]
              'qi': 200000,  # internal quality factor
              'qc': 15000,  # coupling quality factor
              'xa': 0.5,  # resonance fractional asymmetry
              'a': 0,  # inductive nonlinearity
              'alpha': 1.,  # IQ mixer amplitude imbalance
              'beta': 0.,  # IQ mixer phase imbalance
              'gain0': 3.0,  # gain polynomial coefficients
              'gain1': 0,  # linear gain coefficient
              'gain2': 0,  # quadratic gain coefficient
              'phase0': 0,  # total loop rotation in radians
              'tau': 50e-9}  # cable delay

fsweep_params = {'fc': 4.0012e9,  # center frequency [Hz]
                 'points': 1000,  # frequency sweep points
                 'span': 500e6,  # frequency sweep bandwidth [Hz]
                 'increasing': True}  # frequency sweep direction

res = Resonator(**res_params)
res.fsweep(**fsweep_params)
s21 = res.s21
res.plot_trans()