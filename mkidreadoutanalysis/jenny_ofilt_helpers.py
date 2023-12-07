import logging

import matplotlib.pyplot as plt
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidreadout import *
from scipy import signal

def shift_and_normalize(template, ntemplate, offset):
    template = template.copy()
    if template.min() != 0:  # all weights could be zero
        template /= np.abs(template.min())  # correct over all template height

    # shift template (max may not be exactly at offset due to filtering and imperfect default template)
    start = 10 + np.argmin(template) - offset
    stop = start + ntemplate
    template = np.pad(template, 10, mode='wrap')[start:stop]  # use wrap to not change the frequency content
    return template


def normalize_only(template):
    template = template.copy()
    if template.min() != 0:  # all weights could be zero
        template /= np.abs(template.min())  # correct over all template height
    return template


def generate_fake_data(res: Resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1), fs=1e6,
                       ts=10, tf=30, cps=1551,
                       rf: RFElectronics = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9),
                       noise=True, noise_scale=10, line_noise_freqs=[60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3],
                       line_noise_amplitudes=[0.005, 0.001, 0.005, 0.003, 0.001, 0.005, 0.001],
                       line_noise_phases=[0, 0.5, 0, 1.3, 0.5, 0.2, 2.4]):
    quasiparticle_timestream = QuasiparticleTimeStream(fs=1e6, ts=10)
    quasiparticle_timestream.gen_quasiparticle_pulse(tf=30);
    quasiparticle_timestream.gen_photon_arrivals(cps=1551)
    quasiparticle_timestream.populate_photons()
    # Create resonator and compute S21
    resonator = res  # 1e2
    rf = rf
    freq = FrequencyGrid(fc=res.f0_0, points=1000, span=500e6)
    sweep = ResonatorSweep(resonator, freq, rf)
    lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)
    # toggle white noise and line noise
    lit_res_measurment.noise_on = noise

    # adjust white noise scale
    lit_res_measurment.rf.noise_scale = noise_scale

    # configure line noise
    lit_res_measurment.rf.line_noise.freqs = line_noise_freqs  # Hz and relative to center of bin (MKID we are reading out)
    lit_res_measurment.rf.line_noise.amplitudes = line_noise_amplitudes
    lit_res_measurment.rf.line_noise.phases = line_noise_phases

    phase_data, _ = lit_res_measurment.basic_coordinate_transformation()

    return phase_data


def plot_lowpass(fs, filter_coe):
    w, h = signal.freqz(b=filter_coe, a=1)
    x = w * fs * 1.0 / (2 * np.pi)
    y = 20 * np.log10(abs(h))
    plt.figure(figsize=(10, 5))
    plt.semilogx(x, y)
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.title('Frequency response')
    plt.grid(which='both', linestyle='-', color='grey')
    plt.show()

def ofilt_summary_plot(psd, dt, noise_nwindow, filter, template, cutoff, fs=1e6):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    # PSD
    ax[0, 0].set_xlabel("frequency [Hz]")
    ax[0, 0].set_ylabel("PSD [dBc / Hz]")
    ax[0, 0].axvline(cutoff*fs/2, color='black', linestyle='--')
    f_noise = np.fft.rfftfreq(noise_nwindow, d=dt)
    ax[0, 0].semilogx(f_noise, 10 * np.log10(psd), label='noise')
    f_template = np.fft.rfftfreq(template.size, d=dt)
    template_fft = 1e-8*np.abs(np.fft.rfft(template))
    ax[0, 0].semilogx(f_template, 10 * np.log10(template_fft), label='template')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_title('PSD')
    # FOURIER QUOTIENT
    ax[0, 1].semilogx(f_noise, 10 * np.log10(np.abs(np.conj(np.fft.rfft(template)) / psd)))
    ax[0, 1].set_xlabel("frequncy [Hz]")
    ax[0, 1].set_ylabel("dB")
    ax[0, 1].axvline(cutoff*fs/2, color='black', linestyle='--')
    ax[0, 1].set_title('Fourier Quotient')
    # TEMPLATE
    ax[1, 0].set_xlabel(r"time [$\mu$s]")
    ax[1, 0].set_ylabel("template [arb.]")
    ax[1, 0].plot(np.arange(template.size) * dt * 1e6, template)
    ax[1, 0].plot(np.arange(filter.size) * dt * 1e6, template[:filter.size])
    ax[1, 0].set_title('Template')
    # FILTER
    ax[1, 1].set_xlabel(r"time [$\mu$s]")
    ax[1, 1].set_ylabel("filter coefficient [radians]")
    ax[1, 1].plot(np.arange(filter.size) * dt * 1e6, filter)
    ax[1, 1].set_title('Filter')
    plt.tight_layout()
    plt.show()


def ofilt_plot_comparison(phase_data: np.ndarray, readout: MKIDReadout, phase_ofilt: np.ndarray,
                          readout_ofilt: MKIDReadout, data_key: str, xlim: (int, int) = (60000, 100000)):
    """
    Comparison plot for phase data before and after filtering.
    Args:
        phase_data: raw phase data
        readout: MKIDReadout object containing trigger locations, energies, etc. for raw data
        phase_ofilt: filtered phase data
        readout_ofilt: MKIDReadout object containing trigger locations, energies, etc. for filtered data
        data_key: 'red', 'blue', or 'ir' for plot labeling and color coding
        xlim: timeseries plot x limits

    Returns:
        2x2 plot showing time series and resulting phase histograms for filtered and unfiltered data.

    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex='row', sharey='row')
    fig.suptitle("Jenny's Ofilt")
    tvec = np.arange(phase_data.size)
    if data_key == 'ir':
        data_color = 'lightcoral'
    else:
        data_color = data_key

    axs[0, 0].plot(tvec[xlim[0]:xlim[1]], phase_data[xlim[0]:xlim[1]], label='phase timestream', color=data_color)
    axs[0, 0].plot(np.ma.masked_outside(tvec[readout.trig], xlim[0], xlim[1]), phase_data[readout.trig], '.',
                   label='trigger')
    axs[0, 0].set_xlabel('time (us)')
    axs[0, 0].set_ylabel('phase (radians)')
    axs[0, 0].plot(np.ma.masked_outside(tvec[readout.photon_energy_idx], xlim[0], xlim[1]),
                   phase_data[readout.photon_energy_idx], 'o', label='energy')
    axs[0, 0].legend(loc='upper right')

    axs[0, 0].set_title("Raw Data")
    axs[0, 1].plot(tvec[xlim[0]:xlim[1]], phase_ofilt[xlim[0]:xlim[1]], label='phase timestream', color=data_color)
    axs[0, 1].plot(np.ma.masked_outside(tvec[readout_ofilt.trig], xlim[0], xlim[1]), phase_ofilt[readout_ofilt.trig],
                   '.', label='trigger')
    axs[0, 1].set_xlabel('time (us)')
    axs[0, 1].set_ylabel('phase (radians)')
    axs[0, 1].plot(np.ma.masked_outside(tvec[readout_ofilt.photon_energy_idx], xlim[0], xlim[1]),
                   phase_ofilt[readout_ofilt.photon_energy_idx], 'o', label='energy')
    axs[0, 1].set_title("Filtered Data")
    axs[0, 1].legend(loc='upper right')
    axs[1, 0].hist(readout.photon_energies, color=data_color, bins='auto', density=True)
    axs[1, 0].set_xlabel('Phase peak')
    axs[1, 0].set_ylabel('Counts')
    axs[1, 1].hist(readout_ofilt.photon_energies, color=data_color, bins='auto', density=True)
    axs[1, 1].set_xlabel('Phase peak')
    axs[1, 1].set_ylabel('Counts')
    plt.show()
