
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


## IMPORT / GENERATE DATA
data_key = 'blue' # options: 'red', 'ir', 'None' (generates fake data)
if data_key == 'None':
    phase_data = generate_fake_data()
else:
    data = np.load(f'/work/jpsmith/Gen3/Fridge_Tests/r_testing/data/white_fridge/10_18_23/wf_ellison3_6000_650GHz.npz')
    phase_data = data[data_key+'_phase'] * np.pi
    del data

## CONFIG PARAMETERS
fs = 1e6 # Sampling rate in Hz
threshold = -0.8 # Phase threshold in radians below which to trigger
deadtime = 80 # Deadtime in microseconds of time to wait before another trigger is possible
n_template=1000 # Number of samples in the template, this should be 5-10x the length of the filter for best accuracy
offset=5 # samples to include before pulse minimum. Useful for fine-tuning ~short filters
n_filter = 1000 # number of taps in final optimal filter
cutoff = 0.1 # lowpass cutoff used during filter generation to compensate for effects of lowpass filters in digital readout?
dt = 1/fs
window_filter = True

## FIND PHOTONS
readout = MKIDReadout()
readout.trigger(phase_data, fs=fs, threshold=threshold, deadtime=deadtime)
readout.plot_triggers(phase_data, energies=True, xlim=(60000, 100000)); plt.show();
raw_pulses = phase_data[(readout.photon_energy_idx + np.arange(-offset, n_template - offset)[:, np.newaxis]).T] # n_pulsesfound x n_template array where each row is one pulse


## PROCESS W/O OPTIMAL FILTERS
process_raw = False
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

## MAKE TEMPLATE PULSE
raw_template = pulses_final.sum(axis=0)
shifted_template = raw_template-raw_template[0:offset//2].mean() # make pulse start at 0
# window template
template_window = -sp.signal.windows.hamming(shifted_template.size, sym=False)
windowed_template = -shifted_template*template_window
normalized_template = shift_and_normalize(shifted_template, n_template, offset)
#normalized_template = normalize_only(windowed_template)


## MAKE NOISE
# noise-specific configs
isolation = 200 # number of samples after threshold is crossed to ignore (allows pulse to recover to baseline)
noise_npoints = 1000#50 # psd is calcualted from overlapping segments of this length (defualt is 50% ovlp).
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
if window_filter:
    window = sp.signal.windows.hamming(optimal_filter.size, sym=False)
    optimal_filter_w = window * optimal_filter


## PLOT OPTIMAL FILTER SUMMARY
ofilt_summary_plot(psd, dt, noise_npoints, optimal_filter, normalized_template)

## APPLY FILTER
phase_ofilt = sp.signal.convolve(phase_data,optimal_filter, mode='same')
readout_ofilt = MKIDReadout()
readout_ofilt.trigger(phase_ofilt, fs = 1e6, threshold=-0.5, deadtime=60)
phase_ofilt_dark_mean = noise_data.mean()
phase_ofilt_max_location, phase_ofilt_fwhm = compute_r(readout_ofilt.photon_energies - phase_ofilt_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()

plot_comparison=True
if plot_comparison:
    ofilt_plot_comparison(phase_data, readout, phase_ofilt, readout_ofilt, data_key, xlim=(60000, 100000))