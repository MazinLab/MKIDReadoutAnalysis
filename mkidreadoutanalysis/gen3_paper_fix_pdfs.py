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
filenames = [f'wf_ellison_5_739_GHz_single_tone_500dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity',
             ]


for i, filename in enumerate(filenames):
    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color + '_')

        path = os.path.join(filedir, fname)+'_processed.npz'
        data = np.load(path)
        max_location = data['max_location']
        fwhm = data['fwhm']
        energies = data['normalized_energies']
        _, _, pdf = fit_histogram(energies)
        pdf_x = np.linspace(energies.min(), energies.max(), 1000)
        pdf_y = pdf(pdf_x)
        np.savez(path, normalized_energies=energies, max_location=max_location, fwhm=fwhm, pdf_x=pdf_x,
                 pdf_y=pdf_y)
