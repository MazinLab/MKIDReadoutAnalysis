import logging

import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger

from mkidreadoutanalysis.optimal_filters.filters import wiener
from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidnoiseanalysis import plot_psd
from mkidreadoutanalysis.mkidreadout import MKIDReadout
import scipy as sp



#quasiparticle_timestream = QuasiparticleTimeStream(fs = 1e6, ts = 10)
#quasiparticle_timestream.gen_quasiparticle_pulse(tf=30);
#quasiparticle_timestream.gen_photon_arrivals(cps=1551)
#quasiparticle_timestream.populate_photons()
# Create resonator and compute S21
#resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1) #1e2
#rf = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9)
#freq = FrequencyGrid( fc=4.0012e9, points=1000, span=500e6)
#sweep = ResonatorSweep(resonator, freq, rf)
#lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)
# toggle white noise and line noise
#lit_res_measurment.noise_on = True

# adjust white noise scale
#lit_res_measurment.rf.noise_scale = 10

# configure line noise
#lit_res_measurment.rf.line_noise.freqs = ([60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3]) # Hz and relative to center of bin (MKID we are reading out)
#lit_res_measurment.rf.line_noise.amplitudes = ([0.5, 0.1, 0.5, 0.3, 0.1, 0.5, 0.01])
#lit_res_measurment.rf.line_noise.phases = ([0, 0.5, 0,1.3,0.5, 0.2, 2.4])

#lit_res_measurment.rf.line_noise.freqs = ([500e3])
#lit_res_measurment.rf.line_noise.amplitudes = ([0.00001])
#lit_res_measurment.rf.line_noise.phases = ([0])
#phase_data, _ = lit_res_measurment.basic_coordinate_transformation()

def _shift_and_normalize(template, ntemplate, offset):
    template = template.copy()
    if template.min() != 0:  # all weights could be zero
        template /= np.abs(template.min())  # correct over all template height

    # shift template (max may not be exactly at offset due to filtering and imperfect default template)
    start = 10 + np.argmin(template) - offset
    stop = start + ntemplate
    template = np.pad(template, 10, mode='wrap')[start:stop]  # use wrap to not change the frequency content
    return template

fs = 1e6
threshold = -0.8
deadtime = 80
n_template=500
offset=30
n_filter = 50
cutoff = 0.2
dt = 1/1e6



data = np.load(f'/work/jpsmith/Gen3/Fridge_Tests/r_testing/data/white_fridge/10_18_23/wf_ellison3_6000_650GHz.npz')
blue_phase=data['red_phase']*np.pi
del data



blue_phase_readout = MKIDReadout()
blue_phase_readout.trigger(blue_phase, fs = 1e6, threshold=-1.0, deadtime=60)
#plt.xlim([60000,1000000]);
x = slice(16000, 19000)
blue_phase_dark_mean = blue_phase[x].mean()
blue_phase_max_location, blue_phase_fwhm = compute_r(blue_phase_readout.photon_energies - blue_phase_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()


pulses = blue_phase[(blue_phase_readout.photon_energy_idx + np.arange(-offset, n_template - offset)[:, np.newaxis]).T]
min_idxs = pulses.argmin(axis=1)
# cuts / check stats
mode, count = sp.stats.mode(min_idxs)
if count / pulses.shape[0] < 0.8:
    getLogger(__name__).warning(f'Warning: {(count / pulses.shape[0])*100:.1f}% of pulse minima differ from the mode. '
                                f'These may be noise triggers. Check deadtime and holdoff settings')
else:
    getLogger(__name__).debug(
        f'Eliminating {(1-(int(count) / pulses.shape[0])) * 100:.1f}% of pulses which appear to be noise.')
    pulses = pulses[min_idxs == mode]
#com_idxs = np.round(np.matmul(pulses, (np.arange(n_template)[np.newaxis,:]).T) / pulses.sum(axis=1))[1,:].astype(int)
integrals = pulses.sum(axis=1)
typical_integrals = np.abs((integrals-integrals.mean())/integrals.mean()) < 0.12
getLogger(__name__).debug(
        f'Eliminating {pulses.shape[0] - typical_integrals.sum()} pulses which appear to be doubles.')
pulses = pulses[typical_integrals]

raw_template = pulses.sum(axis=0)
shifted_template = raw_template-raw_template[0:offset//2].mean() #make pulse start at 0
normalized_template = _shift_and_normalize(shifted_template, n_template, offset)

# make noise
isolation = 200
noise_npoints = 1000
noise_threshold = 0.4
half_pulse_width = deadtime
max_noise_windows = 2000
thresh_all = blue_phase<noise_threshold
thresh_diff = np.diff(thresh_all, append=thresh_all[-1])
crossing_idx = np.arange(blue_phase.size)[thresh_diff]
noise_regions = np.diff(crossing_idx, append=crossing_idx[-1]) > noise_npoints+isolation+offset
noise_idxs = crossing_idx[noise_regions]

start_idx = noise_idxs[:-1]
end_idx = noise_idxs[1:]
#getLogger(__name__).warning(f'User requested {max_noise_windows} but only found {start_idx.size}.'
#                                f'Proceeding with fewer noise windows.')

noise_data = np.array([0])
psd = np.zeros(int(noise_npoints / 2. + 1))
windows_used = 0
n_psd = 0

for start, stop in zip(start_idx, end_idx):
    if windows_used > max_noise_windows:
        break # no more noise is needed
    data = blue_phase[start + isolation: stop - offset]
    if (data < threshold).any():
        continue # pulse in noise chunk
    psd += sp.signal.welch(data - data.mean(), fs=1. / dt, nperseg=noise_npoints, detrend="constant",
                           return_onesided=True, scaling="density")[1]
    n_psd += 1
    windows_used += data.size // noise_npoints
    # finish the average, assume white noise if there was no data
    if n_psd == 0:
        psd[:] = 1.
    else:
        psd /= n_psd

optimal_filter = wiener(normalized_template, psd, noise_npoints, nfilter=n_filter, cutoff=cutoff, normalize=True)

blue_phase_ofilt = sp.signal.convolve(blue_phase,optimal_filter, mode='same')
blue_phase_readout_ofilt = MKIDReadout()
blue_phase_readout_ofilt.trigger(blue_phase_ofilt, fs = 1e6, threshold=-1.2, deadtime=60)
x = slice(16000, 19000)
blue_phase_ofilt_dark_mean = blue_phase_ofilt[x].mean()
blue_phase_ofilt_max_location, blue_phase_ofilt_fwhm = compute_r(blue_phase_readout_ofilt.photon_energies - blue_phase_ofilt_dark_mean, color='blue', plot=False)
#print(f'Max Phase: {-blue_phase_ofilt_max_location} FWHM: {blue_phase_ofilt_fwhm} radians')
#plt.title('Blue Photons Optimal Filter (409.5 nm), FPGA Phase');
#plt.show()



fig, axs = plt.subplots(2,2, figsize=(10,10))
fig.suptitle("Jenny's Ofilt")
tvec = np.arange(blue_phase.size)
axs[0,0].plot(tvec, blue_phase, label='phase timestream', color='blue')
axs[0,0].plot(tvec[blue_phase_readout.trig], blue_phase[blue_phase_readout.trig], '.', label='trigger')
axs[0,0].set_xlabel('time (us)')
axs[0,0].set_ylabel('phase (radians)')
axs[0,0].plot(tvec[blue_phase_readout.photon_energy_idx], blue_phase[blue_phase_readout.photon_energy_idx], 'o', label='energy')
axs[0,0].set_title("Raw Data")
axs[0,0].set_xlim(60000,1000000)
axs[0,0].set_ylim(-4, 3.15)
axs[0,1].plot(tvec, blue_phase_ofilt, label='phase timestream', color='blue')
axs[0,1].plot(tvec[blue_phase_readout_ofilt.trig], blue_phase_ofilt[blue_phase_readout_ofilt.trig], '.', label='trigger')
axs[0,1].set_xlabel('time (us)')
axs[0,1].set_ylabel('phase (radians)')
axs[0,1].plot(tvec[blue_phase_readout_ofilt.photon_energy_idx], blue_phase_ofilt[blue_phase_readout_ofilt.photon_energy_idx], 'o', label='energy')
axs[0,1].set_title("Filtered Data")
axs[0,1].set_xlim(60000,1000000)
axs[0,1].set_ylim(-4, 3.15)
axs[1,0].hist(blue_phase_readout.photon_energies, color='blue', bins='auto', density=True, label=f'Center: {blue_phase_max_location:1f} FWHM:{blue_phase_fwhm:1f}')
axs[1,0].set_xlabel('Phase peak')
axs[1,0].legend(loc='upper right')
axs[1,0].set_ylabel('Counts')
axs[1,1].hist(blue_phase_readout_ofilt.photon_energies, color='blue', bins='auto', density=True, label=f'Center: {blue_phase_ofilt_max_location:1f} FWHM:{blue_phase_ofilt_fwhm:1f}')
axs[1,1].set_xlabel('Phase peak')
axs[1,1].set_ylabel('Counts')
axs[1,1].set_xlim(-6, 0)
axs[1,1].set_ylim(0, 9)
axs[1,1].legend(loc='upper right')
plt.show()
print('hi')