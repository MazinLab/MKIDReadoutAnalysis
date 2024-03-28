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
from mkidreadoutanalysis.jenny_ofilt_helpers import generate_fake_data, ofilt_plot_comparison



## IMPORT / GENERATE DATA
data_key = 'ir' # options: 'red', 'ir', 'None' (generates fake data)
if data_key == 'None':
    phase_data = generate_fake_data()
else:
    data = np.load(f'/work/jpsmith/Gen3/Fridge_Tests/r_testing/data/white_fridge/10_18_23/wf_ellison3_6000_650GHz.npz')
    phase_data = data[data_key+'_phase'] * np.pi
    del data

## CONFIG PARAMETERS
cfg=ConfigThing()
cfg.registerfromkvlist((('dt', 1e-6),
('fit', False),
('summary_plot', True),
('pulses.unwrap', False),
#('pulses.fallback_template', '/work/jpsmith/R_Analysis/MKIDReadoutAnalysis/mkidreadoutanalysis/optimal_filters/template_15us.txt'), # specify a pre-computed fallback template,
                        # will be sliced according to offset and n_template
('pulses.fallback_template', None), # specify a pre-computed fallback template,

('pulses.tf', 30), # pre filter pulse fall time in microseconds
('pulses.ntemplate', 500), # need to set this larger to calculate covariance matrix in the time domain "accurately" for the number of selected filter coefficients
('pulses.offset', 10),
('pulses.threshold', 6), # sigma above noise
('pulses.separation', 500),
('pulses.min_pulses', 500),
('noise.nwindow', 500), #1000
('noise.isolation', 200),
('noise.max_windows', 2000), # maximum number of nwindows of samples needed before moving on [int]
#('noise.max_noise', 2000), #2000
('template.percent', 80),
('filter.cutoff', .1),
('template.min_tau', 5),
('template.max_tau', 100),
('template.fit', 'triple_exponential'),
('filter.filter_type', 'wiener'),
('filter.nfilter', 50), # for messing around this should be closer to 1000 and ntemplate should be increased to be 5-10x nfilter
                        # need to make sure filter is periodic and this gets hard when the filter is short
('filter.normalize', True)), namespace='')

ofc = Calculator(phase=phase_data, config=cfg, name='simulated')

ofc.calculate(clear=False)
ofc.plot(); plt.show();

optimal_filter = ofc.result['filter']

## APPLY FILTER
phase_ofilt = sp.signal.convolve(phase_data,optimal_filter, mode='same')
readout_ofilt = MKIDReadout()
readout_ofilt.trigger(phase_ofilt, fs = 1e6, threshold=-1.2, deadtime=60)
phase_ofilt_dark_mean = phase_data[25000:60000].mean() #TODO: Make automatic
phase_ofilt_max_location, phase_ofilt_fwhm = compute_r(readout_ofilt.photon_energies - phase_ofilt_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()

plot_comparison=True
if plot_comparison:
    ## PROCESS W/O OFILT
    readout = MKIDReadout()
    readout.trigger(phase_data, fs=1e6, threshold=0.0, deadtime=60)
    readout.plot_triggers(phase_data, energies=True, xlim=(60000, 100000));
    plt.show();
    ofilt_plot_comparison(phase_data, readout, phase_ofilt, readout_ofilt, data_key, xlim=(60000, 100000))

