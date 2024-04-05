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


filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_03_30_24/r_single_res'
filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_']
colors = ['blue_405_9', 'red_663_1', 'ir_808_0']


titles = ['Full Dynamic Range', '500 Tone Equivalent Dynamic Range', '1024 Tone Equivalent Dynamic Range',
          '2048 Tone Equivalent Dyanic Range']


fig, axs = plt.subplots(1,4, figsize=(20,7), sharey='row')
axs[0].set_ylabel('Counts', fontsize=16)
for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, normalized_energies, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors)
    make_r_hist_plt(axs[i], phase_dist_centers, raw_r, pdfs_x, pdfs_y)

    axs[i].set_title(titles[i], fontsize=16)

axs[i].set_ylim(axs[i].get_ylim()[0], axs[i].get_ylim()[1]+0.5)
plt.tight_layout()

plt.show()
print('hi')



