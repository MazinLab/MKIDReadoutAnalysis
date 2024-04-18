from helpers import *

filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/software_ofilt'  # Multitone (software ofilt)
filenames = [f'wf_ellison_5_739_GHz_single_tone_fulldr_phase_unity',
             f'wf_ellison_5_739_GHz_500_tone_phase_unity',
             f'wf_ellison_5_739_GHz_1024_tone_phase_unity',
             f'wf_ellison_5_739_GHz_2048_tone_phase_unity']

colors = ['blue_405_9', 'red_663_1', 'ir_808_0']

for i, filename in enumerate(filenames):
    phase_dist_centers, raw_r, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors, advanced=True)
