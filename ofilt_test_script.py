import logging

import matplotlib.pyplot as plt
import numpy as np

from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidnoiseanalysis import plot_psd
from mkidreadoutanalysis.mkidreadout import MKIDReadout
from mkidreadoutanalysis.optimal_filters.make_filters import Calculator
from mkidreadoutanalysis.optimal_filters.config import ConfigThing
import scipy as sp




data = np.load(f'/work/jpsmith/Gen3/Fridge_Tests/r_testing/data/white_fridge/10_18_23/wf_ellison3_6000_650GHz.npz')
blue_phase=data['red_phase']*np.pi
del data

#blue_phase_readout = MKIDReadout()
#blue_phase_readout.trigger(blue_phase, fs = 1e6, threshold=-1, deadtime=60)
#plt.xlim([60000,1000000]);
#x = slice(16000, 19000)
#blue_phase_dark_mean = blue_phase[x].mean()
#blue_phase_max_location, blue_phase_fwhm = compute_r(blue_phase_readout.photon_energies - blue_phase_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()


cfg=ConfigThing()
cfg.registerfromkvlist((('dt', 1e-6),
('fit', False),
('summary_plot', True),
('pulses.unwrap', False),
#('pulses.fallback_template', '/work/jpsmith/R_Analysis/MKIDReadoutAnalysis/mkidreadoutanalysis/optimal_filters/template_15us.txt'), # specify a pre-computed fallback template,
                        # will be sliced according to offset and n_template
('pulses.fallback_template', None), # specify a pre-computed fallback template,

('pulses.tf', 30), # pre filter pulse fall time in microseconds
('pulses.ntemplate', 5000), # need to set this larger to calculate covariance matrix in the time domain "accurately" for the number of selected filter coefficients
('pulses.offset', 10),
('pulses.threshold', 6), # sigma above noise
('pulses.separation', 500),
('pulses.min_pulses', 10000),
('noise.nwindow', 1000), #1000
('noise.isolation', 200),
('noise.max_windows', 2000), # maximum number of nwindows of samples needed before moving on [int]
('noise.max_noise', 2000), #2000
('template.percent', 80),
('template.cutoff', .2),
('filter.cutoff', .5),
('template.min_tau', 5),
('template.max_tau', 100),
('template.fit', 'triple_exponential'),
('filter.filter_type', 'wiener'),
('filter.nfilter', 50), # for messing around this should be closer to 1000 and ntemplate should be increased to be 5-10x nfilter
                        # need to make sure filter is periodic and this gets hard when the filter is short
('filter.normalize', True)), namespace='')

ofc = Calculator(phase=blue_phase, config=cfg, name='simulated')

ofc.calculate(clear=False)
plt.show()

blue_phase_ofilt = sp.signal.convolve(blue_phase, ofc.result['filter'], mode='same')
blue_phase_readout_ofilt = MKIDReadout()
blue_phase_readout_ofilt.trigger(blue_phase_ofilt, fs = 1e6, threshold=-1.2, deadtime=60)
x = slice(16000, 19000)
blue_phase_ofilt_dark_mean = blue_phase_ofilt[x].mean()
blue_phase_ofilt_max_location, blue_phase_ofilt_fwhm = compute_r(blue_phase_readout_ofilt.photon_energies - blue_phase_ofilt_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()



fig, axs = plt.subplots(2,2, figsize=(10,10))
fig.suptitle("Nic's Ofilt")
tvec = np.arange(blue_phase.size)
axs[0,0].plot(tvec, blue_phase, label='phase timestream', color='blue')
axs[0,0].plot(tvec[blue_phase_readout.trig], blue_phase[blue_phase_readout.trig], '.', label='trigger')
axs[0,0].set_xlabel('time (us)')
axs[0,0].set_ylabel('phase (radians)')
axs[0,0].plot(tvec[blue_phase_readout.photon_energy_idx], blue_phase[blue_phase_readout.photon_energy_idx], 'o', label='energy')
axs[0,0].set_title("Raw Data")
axs[0,0].set_xlim(60000,1000000)
axs[0,0].set_ylim(-4, 3.15)
axs[0,1].plot(tvec, blue_phase_ofilt, label='phase timestream', color='blue')
axs[0,1].plot(tvec[blue_phase_readout_ofilt.trig], blue_phase_ofilt[blue_phase_readout_ofilt.trig], '.', label='trigger')
axs[0,1].set_xlabel('time (us)')
axs[0,1].set_ylabel('phase (radians)')
axs[0,1].plot(tvec[blue_phase_readout_ofilt.photon_energy_idx], blue_phase_ofilt[blue_phase_readout_ofilt.photon_energy_idx], 'o', label='energy')
axs[0,1].set_title("Filtered Data")
axs[0,1].set_xlim(60000,1000000)
axs[0,1].set_ylim(-4, 3.15)
axs[1,0].hist(blue_phase_readout.photon_energies, color='blue', bins='auto', density=True, label=f'Center: {blue_phase_max_location:1f} FWHM:{blue_phase_fwhm:1f}')
axs[1,0].set_xlabel('Phase peak')
axs[1,0].legend(loc='upper right')
axs[1,0].set_ylabel('Counts')
axs[1,1].hist(blue_phase_readout_ofilt.photon_energies, color='blue', bins='auto', density=True, label=f'Center: {blue_phase_ofilt_max_location:1f} FWHM:{blue_phase_ofilt_fwhm:1f}')
axs[1,1].set_xlabel('Phase peak')
axs[1,1].set_ylabel('Counts')
axs[1,1].set_xlim(-6, 0)
axs[1,1].set_ylim(0, 9)
axs[1,1].legend(loc='upper right')
plt.show()
print('hi')