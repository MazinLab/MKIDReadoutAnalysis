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
filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_2048dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_1024dr_phase_unity',
             f'wf_ellison_5_739_GHz_single_tone_500dr_phase_unity']


titles = ['Full Dynamic Range', '500 Tone Equivalent Dynamic Range', '1024 Tone Equivalent Dynamic Range',
          '2048 Tone Equivalent Dyanic Range']

fig, axs = plt.subplots(1,4, figsize=(20,7), sharey='row')
axs[0].set_ylabel('Counts', fontsize=16)

for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, normalized_energies = get_energy_hist_points(filedir, filename, colors)
    pdfs = compute_pdfs(normalized_energies)
    make_r_hist_plt(axs[i], phase_dist_centers, raw_r, pdfs, normalized_energies)
    axs[i].set_title(titles[i], fontsize=16)

plt.tight_layout()
plt.show()

print('hi')



