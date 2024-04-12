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
from helpers import *
import matplotlib.ticker as tck


colors = ['blue_405_9', 'red_663_1', 'ir_808_0']

filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/r_multi_tone'
filenames = [f'wf_ellison_5_739_GHz_500_tone_phase_unity',
             f'wf_ellison_5_739_GHz_1024_tone_phase_unity',
             f'wf_ellison_5_739_GHz_2048_tone_phase_unity',
             ]


for i, filename in enumerate(filenames):
    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color + '_')

        path = os.path.join(filedir, fname)+'_processed.npz'
        data = np.load(path)
        pdf_x = data['pdf_x']
        pdf_y = data['pdf_y']
        energies = data['normalized_energies']

        max_location, fwhm, pdf = fit_univariate_spline(pdf_x, pdf_y, k=3, ext=1)
        plt.plot(pdf_x, pdf_y)
        plt.plot(pdf_x, pdf(pdf_x))
        plt.show()
        np.savez(path, normalized_energies=energies, max_location=max_location, fwhm=fwhm, pdf_x=pdf_x,
                 pdf_y=pdf_y)
