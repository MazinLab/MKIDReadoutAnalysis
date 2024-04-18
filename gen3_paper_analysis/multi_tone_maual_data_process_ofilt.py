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
from helpers import *
import matplotlib.ticker as tck


colors = ['red_663_1', 'blue_405_9', 'ir_808_0']
#colors = ['red_663_1']
#colors = ['ir_808_0']
#colors = ['blue_405_9']


filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_15_24/adc_dr_test'
filenames = [f'wf_ellison_5_739_GHz_single_tone_adc_attn_2048_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_adc_attn_1024_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_adc_attn_500_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_adc_attn_1_phase_unity']
#filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity']
filenames=[f'wf_ellison_5_739_GHz_single_tone_adc_attn_1_phase_unity']

filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_03_30_24/software_ofilt'  # DAC DR Test (software ofilt)
filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity']

filenames=[f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity']


for i, filename in enumerate(filenames):

    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color + '_')
        phase_data = get_data(filedir, fname, exists=True)
        dark_data = get_data(filedir, filename)
        if color == 'red_663_1':
            ofc = compute_ofilt(phase_data)
            ofc.plot()
            plt.show()
            optimal_filter = ofc.result['filter']
        try:
            phase_data = np.convolve(phase_data, optimal_filter)
        except NameError:
            ffilt = filename.replace('phase_unity', 'phase_red_663_1_unity_processed.npz')
            data = np.load(os.path.join(filedir, ffilt))
            optimal_filter = data['ofilt']
            phase_data = np.convolve(phase_data, optimal_filter)
        dark_data = np.convolve(dark_data, optimal_filter)
        phase_readout = MKIDReadout()
        phase_readout.trigger(phase_data, fs=1e6, threshold=-1.5, deadtime=254)
        phase_readout.plot_triggers(phase_data, fs=1e6, energies=True, color='red', xlim=(60000, 200000))
        plt.show()
#        energies = phase_readout.photon_energies - dark_data.mean()
        energies = phase_readout.photon_energies - phase_data[100:200000].mean()
#        plt.plot(dark_data[:200000])
        plt.plot(phase_data[:200000])
        plt.plot(np.ones(200000 - 100) * phase_data[100:200000].mean())
        plt.show()
        plt.hist(energies, bins='auto')
        plt.show()
        max_location, fwhm, pdf = estimate_pdf(energies)
        pdf_x = np.linspace(energies.min(), energies.max(), 1000)
        pdf_y = pdf(pdf_x)
        yhat = pdf_y
#        yhat = savgol_filter(pdf_y, 101, 3) # smooth out quantization effects
#        yhat = savgol_filter(savgol_filter(pdf_y, 131, 3), 101, 3)
#        yhat = savgol_filter(savgol_filter(pdf_y, 221, 3), 191, 3)
        plt.plot(pdf_x, yhat)
        plt.show()
        fprocessedname = os.path.join(filedir, fname)+'_processed.npz'
        ofilt_name = os.path.join(filedir, fname)+'_ofilt.npz'
        np.savez(ofilt_name, phase_data=phase_data, dark_data=dark_data)
        np.savez(fprocessedname, normalized_energies=energies, max_location=max_location, fwhm=fwhm, pdf_x=pdf_x,
                 pdf_y=yhat, ofilt=optimal_filter)
