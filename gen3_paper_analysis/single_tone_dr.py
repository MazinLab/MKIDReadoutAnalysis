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

fig, axs = plt.subplots(2,4, figsize=(30,10), sharey='row')

max_val = (20*np.log10(get_dac_fft(filedir, dac_filenames[0], noise))).max()
axs[0, 0].set_ylabel('DAC Output', fontsize=20)
for i, filename in enumerate(dac_filenames):
    waveform_fft = get_dac_fft(filedir, filename, noise)
    plot_dac_output(axs[0,i], waveform_fft, max_val)
    axs[0,i].set_title(titles[i], fontsize=25)



filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_']
colors = ['blue_405_9', 'red_663_1', 'ir_808_0']


axs[1,0].set_ylabel('Counts', fontsize=20)
blue_r = []
red_r = []
ir_r = []
for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, normalized_energies, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors)
    blue_r.append(raw_r[0])
    red_r.append(raw_r[1])
    ir_r.append(raw_r[2])
    make_r_hist_plt(axs[1,i], phase_dist_centers, raw_r, pdfs_x, pdfs_y)
axs[1,i].set_ylim(axs[1,i].get_ylim()[0], axs[1,i].get_ylim()[1]+0.5)

plt.tight_layout()
#plt.savefig('single_hist_dac.png')
plt.show()

fig, ax = plt.subplots(1,1, figsize=(30,5))

mkr_plt = [3,5,7,9]
ax.errorbar(mkr_plt, blue_r, 1, color='blue', linewidth=5)
ax.plot(mkr_plt, blue_r,  color='blue', linewidth=5, marker='o', markersize=20, markerfacecolor='#0015B0', markeredgecolor='#0015B0', label='Blue 409.5 nm')
ax.errorbar(mkr_plt, red_r, 1, color='red', linewidth=5)
ax.plot(mkr_plt, red_r,color='red', linewidth=5,  marker='d', markersize=20, markerfacecolor='#AB1A00', markeredgecolor='#AB1A00', label='Red 633.1 nm')
ax.errorbar(mkr_plt, ir_r, 1, color='lightcoral', linewidth=5)
ax.plot(mkr_plt, ir_r, color='#AF49A0', linewidth=5, marker='v', markersize=20, markerfacecolor='#AF49A0', markeredgecolor='#AF49A0', label='IR 808.0 nm')
ax.set_xlim([2,10])
xlabels = ['1', '500', '1000', '2000']
xticks =[3,5,7,9]
ax.set_xticks(xticks, labels=xlabels, fontsize=14)
ax.legend(loc='center right', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel('Energy Resolution', fontsize=20)
plt.xlabel('N Tones Equivalent Dynamic Range', fontsize=20)
plt.tight_layout()
plt.savefig('single_res_r_line.png')
plt.show()
print('hi')



