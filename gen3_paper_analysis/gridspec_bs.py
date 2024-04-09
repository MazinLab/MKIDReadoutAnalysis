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
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tck


filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_03_30_24/r_single_res'

dac_filenames = [f'dac_table_fulldr.npz',
                 f'dac_table_500dr.npz',
                 f'dac_table_1024dr.npz',
                 f'dac_table_2048dr.npz']
titles = ['Full Dynamic Range', '500 Tone Equivalent Dynamic Range', '1024 Tone Equivalent Dynamic Range',
          '2048 Tone Equivalent Dyanic Range']

noise = np.random.normal(0, 1000, 2**19)

fig, axs = plt.subplots(2,4, figsize=(40,20), sharey='row')
gs = fig.add_gridspec(3, 4)

f_ax00 = fig.add_subplot(gs[0, 0])
f_ax01 = fig.add_subplot(gs[0, 1], sharey=f_ax00)
f_ax02 = fig.add_subplot(gs[0, 2], sharey=f_ax00)
f_ax03 = fig.add_subplot(gs[0, 3], sharey=f_ax00)
row_0_ax = [f_ax00, f_ax01, f_ax02, f_ax03]

f_ax10 = fig.add_subplot(gs[1, 0])
f_ax11 = fig.add_subplot(gs[1, 1],  sharey=f_ax10)
f_ax12 = fig.add_subplot(gs[1, 2],  sharey=f_ax10)
f_ax13 = fig.add_subplot(gs[1, 3],  sharey=f_ax10)
row_1_ax = [f_ax10, f_ax11, f_ax12, f_ax13]

f_ax2 = fig.add_subplot(gs[2, :])

max_val = (20*np.log10(get_dac_fft(filedir, dac_filenames[0], noise))).max()
row_0_ax[0].set_ylabel('DAC Output', fontsize=20)

for i, filename in enumerate(dac_filenames):
    waveform_fft = get_dac_fft(filedir, filename, noise)
    plot_dac_output(row_0_ax[i], waveform_fft, max_val)
    row_0_ax[i].set_title(titles[i], fontsize=25)



filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_']
colors = ['blue_405_9', 'red_663_1', 'ir_808_0']


for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, normalized_energies, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors)
    make_r_hist_plt(row_1_ax[i], phase_dist_centers, raw_r, pdfs_x, pdfs_y)
row_1_ax[i].set_ylim(row_1_ax[i].get_ylim()[0], row_1_ax[i].get_ylim()[1]+0.5)

f_ax2.plot(np.arange(10))

#plt.tight_layout()


yticks = np.linspace(0, -100, 11)
ylabels = [str(x) + ' dB' for x in yticks]
ylabels[0] = 'DAC Max Output'
row_0_ax[0].set_yticks(yticks, labels=ylabels, fontsize=18)
row_1_ax[0].set_ylabel('Counts', fontsize=20)
plt.show()



print('hi')



