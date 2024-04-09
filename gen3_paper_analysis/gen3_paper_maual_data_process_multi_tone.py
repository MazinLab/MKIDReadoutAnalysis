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
from scipy.signal import savgol_filter
import numpy as np
from gen3_paper_analysis_helpers import *
import matplotlib.ticker as tck


colors = ['blue_405_9', 'red_663_1', 'ir_808_0']
#colors = ['red_663_1']
#colors = ['ir_808_0']


filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/r_multi_tone'
#filenames = [f'wf_ellison_5_739_GHz_500_tone_phase_unity',
#             f'wf_ellison_5_739_GHz_1024_tone_phase_unity',
#             f'wf_ellison_5_739_GHz_2048_tone_phase_unity']
filenames = [f'wf_ellison_5_739_GHz_1024_tone_phase_unity']

for i, filename in enumerate(filenames):

    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color + '_')
        phase_data = get_data(filedir, fname)
        dark_data = get_data(filedir, filename)
        #phase_data = phase_data[:phase_data.size//4]
        phase_readout = MKIDReadout()
        phase_readout.trigger(phase_data, fs=1e6, threshold=-0.5, deadtime=254)
        phase_readout.plot_triggers(phase_data, fs=1e6, energies=True, color='red', xlim=(60000, 200000))
        plt.show()
        energies = phase_readout.photon_energies - dark_data.mean()
        plt.plot(dark_data[:1000])
        plt.plot(phase_data[:1000])
        plt.show()
        max_location, fwhm, pdf = fit_histogram(energies)
        plt.hist(energies, bins='auto')
        plt.show()
        pdf_x = np.linspace(energies.min(), energies.max(), 1000)
        pdf_y = pdf(pdf_x)
        yhat = savgol_filter(pdf_y, 101, 3) # smooth out quantization effects
        #yhat = savgol_filter(savgol_filter(pdf_y, 131, 3), 71, 3)
        yhat = savgol_filter(savgol_filter(pdf_y, 221, 3), 191, 3)
        plt.plot(pdf_x, yhat)
        plt.show()
        fprocessedname = os.path.join(filedir, fname)+'_processed.npz'
        np.savez(fprocessedname, normalized_energies=energies, max_location=max_location, fwhm=fwhm, pdf_x=pdf_x,
                 pdf_y=yhat)
