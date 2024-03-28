
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger

from mkidreadoutanalysis.optimal_filters.filters import wiener
from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidnoiseanalysis import plot_psd
from mkidreadoutanalysis.mkidreadout import MKIDReadout
from mkidreadoutanalysis.jenny_ofilt_helpers import generate_fake_data, shift_and_normalize, ofilt_plot_comparison, ofilt_summary_plot, normalize_only
import scipy as sp
import matplotlib.pyplot as plt



data = np.load(f'/nfs/exoserver/volume1/homes/baileyji/mecdatajenny_20220930/data/phasesnaps/220/snap_220_resID60068_20220930-102016.npz')
#data = np.load(f'/nfs/exoserver/volume1/homes/baileyji/mecdatajenny_20220930/data/phasesnaps/220/snap_220_resID60694_20220930-121347.npz')
phase_data = data['arr_0']

soln = np.load('/nfs/exoserver/volume1/homes/baileyji/mecdatajenny_20220930/data/phasesnaps/220/filter_solution_coefficients.npz')
res_ids = soln['res_ids']
kind = soln['kind']
filters = soln['filters']
n_filt = filters[np.where(res_ids == 60068),:][0,0,:]

## CONFIG PARAMETERS
fs = 1e6 # Sampling rate in Hz
threshold = -1.0 # Phase threshold in radians below which to trigger
deadtime = 60 # Deadtime in microseconds of time to wait before another trigger is possible
n_template=500 # Number of samples in the template, this should be 5-10x the length of the filter for best accuracy
offset=10 #5 samples to include before pulse minimum. Useful for fine-tuning ~short filters
n_filter = 50 # number of taps in final optimal filter
cutoff = 0.1 # lowpass cutoff used during filter generation to compensate for effects of lowpass filters in digital readout?
dt = 1/fs


readout = MKIDReadout()
readout.trigger(phase_data, fs=fs, threshold=threshold, deadtime=deadtime)
readout.plot_triggers(phase_data, energies=True, xlim=(60000, 100000)); plt.show();
window = np.arange(-offset, n_template - offset)
usable_idx = readout.photon_energy_idx[readout.photon_energy_idx+window.max()<phase_data.size]
readout.photon_energy_idx = usable_idx

raw_pulses = phase_data[(readout.photon_energy_idx + np.arange(-offset, n_template - offset)[:, np.newaxis]).T] # n_pulsesfound x n_template array where each row is one pulse

## PROCESS W/O OPTIMAL FILTERS
process_raw = True
if process_raw:
    dark_sl = slice(16000, 19000) # TODO: make this automatic
    phase_dark_mean = phase_data[dark_sl].mean()
    phase_max_location, phase_fwhm = compute_r(readout.photon_energies - phase_dark_mean, color=None, plot=True)
    print(f'Max Phase: {-phase_max_location} FWHM: {phase_fwhm} radians')
    plt.title('No Optimal Filter, FPGA Phase');
    plt.show()

## CUT OUT BAD PULSES

# cut out noise triggers
min_idxs = raw_pulses.argmin(axis=1)
mode, count = sp.stats.mode(min_idxs)
if count / raw_pulses.shape[0] < 0.8:
    getLogger(__name__).warning(f'Warning: {(count / raw_pulses.shape[0])*100:.1f}% of pulse minima differ from the mode. '
                                f'These may be noise triggers. Check deadtime and holdoff settings. Proceeding without noise cuts.')
else:
    getLogger(__name__).debug(
        f'Eliminating {(1-(int(count) / raw_pulses.shape[0])) * 100:.1f}% of pulses which appear to be noise.')
    pulses_noisecut = raw_pulses[min_idxs == mode]
#com_idxs = np.round(np.matmul(pulses, (np.arange(n_template)[np.newaxis,:]).T) / pulses.sum(axis=1))[1,:].astype(int) # compute pulse center of masses

# cut out multi-photon events
integrals = pulses_noisecut.sum(axis=1)
typical_integrals = np.abs((integrals-integrals.mean())/integrals.mean()) < 0.12
getLogger(__name__).debug(
        f'Eliminating {pulses_noisecut.shape[0] - typical_integrals.sum()} pulses which appear to be doubles.')
pulses_final = pulses_noisecut[typical_integrals]

pulse_psd = np.zeros(int(500 / 2. + 1))

## MAKE TEMPLATE PULSE
for i in range(pulses_final.shape[0]):
    pulse_psd += sp.signal.welch(pulses_final[i,:] - pulses_final[i,:].mean(), fs=1. / dt, nperseg=500, detrend="constant",
                               return_onesided=True, scaling="density")[1]

pulse_psd = pulse_psd / pulses_final.shape[0]


raw_template = pulses_final.sum(axis=0)
shifted_template = raw_template-raw_template[0:offset//2].mean() # make pulse start at 0


#lp_filter_template = sp.signal.convolve(shifted_template,lowpass, mode='same')

# window template
#template_window = -sp.signal.windows.hamming(shifted_template.size, sym=False)
#windowed_template = -shifted_template*template_window
normalized_template = shift_and_normalize(shifted_template, n_template, offset)
#normalized_template = shift_and_normalize(lp_filter_template, n_template, offset)

#normalized_template = normalize_only(windowed_template)


#a = sp.interpolate.UnivariateSpline(np.arange(normalized_template.size), normalized_template, s=0)
#interp_x = np.linspace(0,normalized_template.size-1,normalized_template.size*2-1)
#interp_temp = a(interp_x)
#filtered_temp = sp.signal.convolve(interp_temp,lowpass, mode='same')
#ds_temp = filtered_temp[::2]

## MAKE NOISE
# noise-specific configs
isolation = 100 # number of samples after threshold is crossed to ignore (allows pulse to recover to baseline)
noise_npoints = 500#50 # psd is calcualted from overlapping segments of this length (defualt is 50% ovlp).
n_ovl = 2 # factor by which noise_npoints is multiplied by to determine the minimum noise window
noise_threshold = 0.4 # value above which will be considered noise (usually set more conservativly than pulse threshold)
max_noise_windows = 2000 # maximum number of windows noise_npoints long to calculate


thresh_all = phase_data < noise_threshold
thresh_diff = np.diff(thresh_all, append=thresh_all[-1])
crossing_idx = np.arange(phase_data.size)[thresh_diff]

noise_data = np.array([0])
psd = np.zeros(int(noise_npoints / 2. + 1))
windows_used = 0
n_psd = 0
for start, stop in zip(crossing_idx[:-1], crossing_idx[1:]):
    if stop - start < noise_npoints*n_ovl+isolation+offset:
        continue # not enough data points in noise region
    else:
        if windows_used > max_noise_windows:
            break  # no more noise is needed
        noise_data = phase_data[start + isolation: stop - offset]
        psd += sp.signal.welch(noise_data - noise_data.mean(), fs=1. / dt, nperseg=noise_npoints, detrend="constant",
                               return_onesided=True, scaling="density")[1]
        n_psd += 1
        windows_used += noise_data.size // noise_npoints
if windows_used < max_noise_windows:
    getLogger(__name__).warning(f'Failed to reach {max_noise_windows} noise windows, only found {windows_used}. '
                                f'Consider reducing noise window size with noise_points or n_ovlp.')
if n_psd == 0:
    getLogger(__name__).warning(f'There was no noise data. Assuming white noise.')
    psd[:] = 1.
else:
    psd /= n_psd

## MAKE OPTIMAL FILTER
optimal_filter = wiener(normalized_template, psd, noise_npoints, nfilter=n_filter, cutoff=cutoff, dt=dt, normalize=True)



optimal_filter = np.zeros(50)
optimal_filter = optimal_filter - optimal_filter.mean()
optimal_filter = -1*optimal_filter


## PLOT OPTIMAL FILTER SUMMARY
#ofilt_summary_plot(psd, dt, noise_npoints, optimal_filter, normalized_template, cutoff)

## APPLY FILTER
phase_ofilt = sp.signal.convolve(phase_data,n_filt, mode='same')
readout_ofilt = MKIDReadout()
readout_ofilt.trigger(phase_ofilt, fs = 1e6, threshold=-1.0, deadtime=60)
phase_ofilt_dark_mean = 0#noise_data.mean()
phase_ofilt_max_location, phase_ofilt_fwhm = compute_r(readout_ofilt.photon_energies - phase_ofilt_dark_mean, color='lightcoral', plot=True)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()
data_key='black'
plot_comparison=True
if plot_comparison:
    ofilt_plot_comparison(phase_data, readout, phase_ofilt, readout_ofilt, data_key, xlim=(0, 100000))


print('hi')