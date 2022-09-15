# Initialize Objects for Investigation
from phasetimestream import PhaseTimeStream
from resonator import *

phase_timestream = PhaseTimeStream(fs = 2e6, ts = 20e3)
phase_timestream.set_tls_noise()
res = Resonator()
mixer = MixerImbalance()
rf = RFElectronics()
freq = FrequencyGrid()
measurment = MeasureResonator(res, mixer, freq, rf)
response = ResonatorResponse(phase_timestream,res,mixer,freq,rf)
