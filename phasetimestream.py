import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.signal import welch
from logging import getLogger


class PhaseTimeStream:
    """ A time series containing photon pulses.
    Attributes:
     - fs: float, sample rate [Hz]
     - ts: float, sample time [uSec]
     - tvec: 1D np.array, time vector [Sec]
     - points: number of samples [None]
     - raw_phase_data: phase timestream
     - photon_arrivals: boolean 1D np.array, whether or not a photon arrived in that time step."""

    def __init__(self, fs, ts, seed=2):
        self.fs = fs
        self.ts = ts
        self.points = int(self.ts * 1e-6 * self.fs)
        self.tvec = np.arange(0, self.points) / self.fs
        self.data_nonoise = np.zeros(self.points)
        self.rng = np.random.default_rng(seed=seed)
        self.data = None  # add other
        self._holdoff = None
        self.psd = None
        self.optimal_filter = None
        self.photon_arrivals = None
        self.tls_noise = None
        self.photon_pulse = None

    def plot_phasetime(self, data):
        plt.figure()
        plt.plot(self.tvec * 1e6, data)
        plt.xlabel('time (usec)')
        plt.ylabel('phase (radians?)')

    def gen_photon_pulse(self, tr=4, tf=30):
        """ pulse with tr rise time in usec and tf fall time in usec."""
        tp = np.linspace(0, 10 * tf, int(10 * tf + 1))  # pulse duration
        self.photon_pulse = -tf * (np.exp(-tp / tf) - np.exp(-tp / tr)) / (tf - tr)

    def plot_pulse(self):
        plt.figure()
        plt.plot(self.photon_pulse)
        plt.xlabel('Time (usec)')
        plt.ylabel('Phase (radians?)')

    def gen_photon_arrivals(self, cps=500):
        """ generate boolean list corresponding to poisson-distributed photon arrival events.
        Inputs:
        - cps: int, photon co
        unts per second.
        """
        photon_events = self.rng.poisson(cps / self.fs, self.tvec.shape[0])
        self.photon_arrivals = np.array(photon_events, dtype=bool)
        if sum(photon_events) > sum(self.photon_arrivals):
            getLogger(__name__).warning(f'More than 1 photon arriving per time step. Lower the count rate?')
        if sum(photon_events) == 0:
            getLogger(__name__).warning(f"Warning: No photons arrived. :'(")
        return self.photon_arrivals

    def populate_photons(self):
        for i in range(self.data_nonoise.size):
            if self.photon_arrivals[i]:
                self.data_nonoise[i:i + self.photon_pulse.shape[0]] = self.photon_pulse
        return self.data_nonoise

    def set_tls_noise(self, scale=1e-3, fr=6e9, q=15e3, **kwargs):
        """ two-level system noise"""
        psd_freqs = np.fft.rfftfreq(self.data_nonoise.size, d=1 / self.fs)
        fc = fr / (2 * q)
        psd = np.zeros_like(psd_freqs)
        nonzero = psd_freqs != 0
        psd[nonzero] = scale / (1 + (psd_freqs[nonzero] / fc) ** 2) / psd_freqs[nonzero]
        noise_phi = 2 * np.pi * self.rng.random(psd_freqs.size)
        noise_fft = np.exp(1j * noise_phi)  # n_traces x n_frequencies
        # rescale the noise to the covariance
        a = np.sqrt(self.data_nonoise.size * psd * self.fs / 2)
        noise_fft = a * noise_fft
        self.tls_noise = np.fft.irfft(noise_fft, self.data_nonoise.size)
        self.data = self.data_nonoise + self.tls_noise
