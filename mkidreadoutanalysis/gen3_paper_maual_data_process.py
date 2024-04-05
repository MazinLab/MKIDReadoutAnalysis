from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidnoiseanalysis import plot_psd
from mkidreadoutanalysis.mkidro import MKIDReadout
from mkidreadoutanalysis.optimal_filters.make_filters import Calculator
from mkidreadoutanalysis.optimal_filters.config import ConfigThing
import copy
import scipy as sp
import os
import matplotlib.pyplot as plt
import numpy as np
from gen3_paper_analysis_helpers import *
import matplotlib.ticker as tck


colors = ['blue_405_9', 'red_663_1', 'ir_808_0']
filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_03_30_24/r_single_res'
filenames = [f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity']


for i, filename in enumerate(filenames):

    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color + '_')
        phase_data = get_data(filedir, fname)
        phase_readout = MKIDReadout()
        phase_readout.trigger(phase_data, fs=1e6, threshold=-1.6, deadtime=254)
        phase_readout.plot_triggers(phase_data, fs=1e6, energies=True, color='red', xlim=(60000, 200000))
        plt.show()
        energies = phase_readout.photon_energies - phase_data.mean()
        energies = np.sort(energies)
        max_location, fwhm, pdf = fit_histogram(energies)
        plt.hist(energies, bins=50)
        plt.show()
        fprocessedname = os.path.join(filedir, fname)+'_processed.npz'
        np.savez(fprocessedname, normalized_energies=energies, max_location=max_location, fwhm=fwhm, pdf=pdf)
