import logging

import matplotlib.pyplot as plt

from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidreadout import MKIDReadout
from mkidreadoutanalysis.optimal_filters.make_filters import Calculator
from mkidcore.config import ConfigThing
import copy

# Generate a timestream proportional to the change in quasiparticle density
quasiparticle_timestream = QuasiparticleTimeStream(fs = 2e6, ts = 5)

# Define a sudden change in quasiparticle density (caused by a photon)
quasiparticle_timestream.gen_quasiparticle_pulse(tf=30);

# Generate photon arrival times
quasiparticle_timestream.gen_photon_arrivals(cps=650)

# Populate phase data with photon pulses
quasiparticle_timestream.populate_photons()

# Create resonator and compute S21
resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1e2)
rf = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9)
freq = FrequencyGrid( fc=4.0012e9, points=1000, span=500e6)
sweep = ResonatorSweep(resonator, freq, rf)

# Measure the Resonator with Photons
lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)

# toggle white noise and line noise
lit_res_measurment.noise_on = True

# adjust white noise scale
lit_res_measurment.rf.noise_scale = 10

# configure line noise
lit_res_measurment.rf.line_noise.freqs = ([60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3]) # Hz and relative to center of bin (MKID we are reading out)
lit_res_measurment.rf.line_noise.amplitudes = ([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01])
lit_res_measurment.rf.line_noise.phases = ([0, 0.5, 0,1.3,0.5, 0.2, 2.4])

# coordintate transform
theta1, d1 = lit_res_measurment.basic_coordinate_transformation()

# Current 8-Tap Equirippple Lowpass Exported from MATLAB
coe = np.array([-0.08066211966627938, 0.02032901400427789, 0.21182262325068868, 0.38968583545138658, 0.38968583545138658, 0.21182262325068868, 0.02032901400427789, -0.08066211966627938])

fine_channel = apply_lowpass_filter(coe, theta1)

# Make a noise trace
quasiparticle_timestream_noise = QuasiparticleTimeStream(fs = 2e6, ts = 5)
noise_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream_noise, freq, rf)
noise_measurment.noise_on = True
noise_measurment.rf.noise_scale = 10
noise_measurment.rf.line_noise.freqs = ([60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3]) # Hz and relative to center of bin (MKID we are reading out)
noise_measurment.rf.line_noise.amplitudes = ([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01])
noise_measurment.rf.line_noise.phases = ([0, 0.5, 0,1.3,0.5, 0.2, 2.4])
theta1_noise, _ = noise_measurment.basic_coordinate_transformation()
noise = apply_lowpass_filter(coe, theta1_noise)


# Optimal Filters
cfg=ConfigThing()
cfg.registerfromkvlist((('dt', 2e-6),
('summary_plot', True),
('pulses.unwrap', False),
('pulses.fallback_template', 'default'),
('pulses.ntemplate', 1000), # need to set this larger to calculate covariance matrix in the time domain "accurately" for the number of selected filter coefficients
('pulses.offset', 20),
('pulses.threshold', 4),
('pulses.separation', 50),
('pulses.min_pulses', 500),
('noise.nwindow', 1000), #1000
('noise.isolation', 100),
('noise.max_windows', 2000), # maximum number of nwindows of samples needed before moving on [int]
('noise.max_noise', 5000), #2000
('template.percent', 80),
('template.cutoff', .2),
('template.min_tau', 5),
('template.max_tau', 100),
('template.fit', 'triple_exponential'),
('filter.filter_type',  'wiener'), #dc_orthogonal filter doesn't use bin 0
('filter.nfilter', 100), # for messing around this should be closer to 1000 and ntemplate should be increased to be 5-10x nfilter
                        # need to make sure filter is periodic and this gets hard when the filter is short
('filter.normalize', True)), namespace='')

ofc = Calculator(fine_channel, noise, config=cfg, name='simulated')

ofc.calculate(clear=False)

# Apply Optimal Filter
result = np.convolve(fine_channel, ofc.result["filter"])
result = result[:lit_res_measurment.photons.points]

# Trigger
readout = MKIDReadout()
data = result
readout.trigger(lit_res_measurment.photons, data, threshold=-1.3, deadtime=50)

# Record Energies
readout.record_energies(data)

# Compute Energy Resolution
r_val = compute_r(readout.photon_energies, plot=True)
print(f"R Value: {r_val}")


