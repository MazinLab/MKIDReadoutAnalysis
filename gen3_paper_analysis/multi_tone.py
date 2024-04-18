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

filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/software_ofilt'  # Multitone (software ofilt)
filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity',
             f'wf_ellison_5_739_GHz_500_tone_phase_unity',
             f'wf_ellison_5_739_GHz_1024_tone_phase_unity',
             f'wf_ellison_5_739_GHz_2048_tone_phase_unity']

#filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_15_24/adc_dr_test'  # ADC DR Test (software ofilt)
#filenames = [f'wf_ellison_5_739_GHz_single_tone_adc_attn_2048_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_adc_attn_1024_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_adc_attn_500_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_adc_attn_1_phase_unity']

#filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_03_30_24/software_ofilt'  # DAC DR Test (software ofilt)
#filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_unity',
#             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity']

titles = ['Single Tone', '500 Tones', '1024 Tones', '2048 Tones']
fig, axs = plt.subplots(1,4, figsize=(30,10), sharey='row')
fig, axs = plt.subplots(1,1, figsize=(3.37,3.37), sharey='row')


#filenames = [f'wf_ellison_5_739_GHz_single_tone_adc_attn_1_phase_unity']

colors = ['blue_405_9', 'red_663_1', 'ir_808_0']

#axs[1,0].set_ylabel('Counts', fontsize=20)
#blue_r = []
#red_r = []
#ir_r = []
#xoffset = [0, 0.6, -0.1, -0.4]
xoffset = [0, 0, 0, 0]
#plt.show()
#plt.show()
#fig, ax = plt.subplots(figsize=(30,10))
#dist_centers, raw_r, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filenames[0], colors)
#make_r_hist_plt(ax, dist_centers, raw_r, pdfs_x, pdfs_y)
#plt.show()
#print('hi')
#plt.show()
filenames =  [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity']
for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors, advanced=True)
#    blue_r.append(raw_r[0])
#    red_r.append(raw_r[1])
#    ir_r.append(raw_r[2])
    make_r_hist_plt(axs, phase_dist_centers, raw_r, pdfs_x, pdfs_y,  rxoffset=10, ryoffset=10, rlblsz=10, xoffset=xoffset[i])

axs.set_ylabel('Probability Density [nm$^{-1}$]', fontsize=12)
axs.yaxis.get_major_formatter().set_scientific(True)
axs.yaxis.get_major_formatter().set_powerlimits((1,2))
axs.yaxis.offsetText.set_fontsize(8)
axs.set_ylim(axs.get_ylim()[0], axs.get_ylim()[1]+0.005)

axs.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("single_r_pretty.pdf", format="pdf", bbox_inches="tight")
plt.show()
print('hi')
fig, ax = plt.subplots(1,1, figsize=(30,5))

mkr_plt = [3,5,7,9]
#ax.errorbar(mkr_plt, blue_r, 1, color='blue', linewidth=5)
ax.plot(mkr_plt, blue_r,  color='blue', linewidth=5, marker='o', markersize=20, markerfacecolor='#0015B0', markeredgecolor='#0015B0', label='Blue 409.5 nm')
#ax.errorbar(mkr_plt, red_r, 1, color='red', linewidth=5)
ax.plot(mkr_plt, red_r,color='red', linewidth=5,  marker='d', markersize=20, markerfacecolor='#AB1A00', markeredgecolor='#AB1A00', label='Red 633.1 nm')
#ax.errorbar(mkr_plt, ir_r, 1, color='lightcoral', linewidth=5)
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
plt.savefig("single_r_pretty.pdf", format="pdf", bbox_inches="tight")
plt.show()
print('hi')
