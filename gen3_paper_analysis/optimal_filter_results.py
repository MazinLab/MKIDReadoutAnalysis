import matplotlib.pyplot as plt
from scipy.io import savemat
from helpers import *


filedir_o = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/software_ofilt'  # Filtered
filedir = '/nfs/wheatley/work/rfsocs/j_whitefridge_data_04_02_24/r_multi_tone'  # Unfiltered
filename = f'wf_ellison_5_739_GHz_2048_tone_phase_unity'  # 2048 tone case
color = 'red_663_1'
colors = ['blue_405_9', 'red_663_1', 'ir_808_0']



fname = filename.replace('phase_', 'phase_' + color + '_')
rawphase_data = get_data(filedir, fname, skip=1)
filtphase_data = np.load(os.path.join(filedir_o,fname+'_ofilt.npz'))['phase_data']

fig = plt.figure(figsize=(3.37,3.37), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1,2])
f_ax1 = fig.add_subplot(gs[0, :])
sl = slice(51000,55000)
f_ax1.plot(rawphase_data[sl], label='unfiltered')
f_ax1.plot(filtphase_data[sl], label='filtered')
f_ax1.set_title('Phase Timeseries', fontsize=8)
f_ax1.set_ylabel('Phase [radians]', fontsize=8)
f_ax1.set_xlabel('Time [microseconds]', fontsize=8)
f_ax1.legend(loc='lower center', fontsize=8)
f_ax1.tick_params(axis='both', which='major', labelsize=8)

f_ax2 = fig.add_subplot(gs[1, 0])
#plt.show()
phase_dist_centers, raw_r, pdfs_x, pdfs_y = get_energy_hist_points(filedir, filename, colors, advanced=True)
make_r_hist_plt(f_ax2, phase_dist_centers, raw_r, pdfs_x, pdfs_y, rxoffset=8, ryoffset=5, rlblsz=8)
#f_ax2.set_ylim(-5,5)
f_ax2.set_title('Unfiltered', fontsize=8)
f_ax2.set_ylabel('Probability Density [nm$^{-1}$]', fontsize=8)
f_ax2.tick_params(axis='both', which='major', labelsize=8)

f_ax3 = fig.add_subplot(gs[1, 1], sharey=f_ax2)
phase_dist_centers, raw_r, pdfs_xo, pdfs_yo = get_energy_hist_points(filedir_o, filename, colors, advanced=True)
make_r_hist_plt(f_ax3, phase_dist_centers, raw_r, pdfs_xo, pdfs_yo,  rxoffset=8, ryoffset=5, rlblsz=8)
f_ax3.tick_params(axis='both', which='major', labelsize=8)
plt.setp(f_ax3.get_yticklabels(), visible=False)
f_ax3.set_title('Filtered', fontsize=8)

plt.tight_layout()
plt.show()

mdic = {"unfiltered_timeseries": rawphase_data[sl],
        "filtered_timeseries": filtphase_data[sl],
        "pdfs_x_blue_unfiltered": pdfs_x[0],
        "pdfs_x_red_unfiltered": pdfs_x[1],
        "pdfs_x_ir_unfiltered": pdfs_x[2],
        "pdfs_y_blue_unfiltered": pdfs_y[0],
        "pdfs_y_red_unfiltered": pdfs_y[1],
        "pdfs_y_ir_unfiltered": pdfs_y[2],
        "pdfs_x_blue_filtered": pdfs_xo[0],
        "pdfs_x_red_filtered": pdfs_xo[1],
        "pdfs_x_ir_filtered": pdfs_xo[2],
        "pdfs_y_blue_filtered": pdfs_yo[0],
        "pdfs_y_red_filtered": pdfs_yo[1],
        "pdfs_y_ir_filtered": pdfs_yo[2]
        }
savemat("optimal_filter_results_workspace.mat", mdic)
print('hi')