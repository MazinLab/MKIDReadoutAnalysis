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
filenames = [f'wf_ellison_5_739_GHz_2048_tone_phase_unity']


for i, filename in enumerate(filenames):
    fname = filename + '_ecal_processed_nm.npz'
    path = os.path.join(filedir, fname)
    data = np.load(path)
    pdfs_x = data['pdfs_x']
    pdfs_yo = data['pdfs_y']

    energy_dist_centers = np.empty(3)
    fwhms = np.empty(3)
    pdfs_y = []
    for i in range(3):
        max_location, fwhm, pdf = fit_univariate_spline(pdfs_x[i], pdfs_yo[i], k=3, ext=1)
        plt.plot(pdfs_x[i], pdfs_yo[i])
        plt.plot(pdfs_x[i], pdf(pdfs_x[i]))
        plt.show()

        pdfs_y.append(pdf(pdfs_x[i]))
        fwhms[i] = fwhm
        energy_dist_centers[i] = max_location

    np.savez(path, energy_dist_centers=np.abs(energy_dist_centers), fwhm=fwhms, pdfs_x=pdfs_x,
             pdfs_y=pdfs_y)