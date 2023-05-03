import numpy as np
from matplotlib import pyplot as plt
from .mkidnoiseanalysis import swenson_formula
from .quasiparticletimestream import QuasiparticleTimeStream
import copy


def compute_s21(readout_freq, fc, increasing, f0, qi, qc, xa, a, rf_phase_delay, rf_gain, cable_delay_phase):
    """
    Compute forward scattering matrix element for a given mesurment setup.
    -------------
    @param:
    - readout_freq: 1D np.array
        the frequency or frequencies at which to compute S21.
        for a lit resonator this should be the intrinsic resonance frequency
        for a dark resonator this should be a sweep around the intrinsic resonance frequency
    - fc: float
        center frequency of sweep
    - increasing: bool
        True if the frequency sweep is done from low frequency to high frequency
    - f0: 1D np.array
        resonant resonant frequency in Hz. For a lit resonator this will be a 1D np.array.
        for a dark resonator it will be a single value.
    - qi: 1D np.array
        resonator internal quality factor.  For a lit resonator this will be a 1D np.array.
        for a dark resonator it will be a single value.
    - qc: float
         resonator coupling quality factor.
    - xa: float
        resonator inductive nonlinearity
    - a: float
        resonance fractional asymmetry
    - rf_phase_delay: np.poly1D
        phase polynomial coefficients (total loop rotation) [radians]
    - rf_gain: np.poly1D
        gain polynomial coefficients [units?]
    - cable_delay_phase: float
        phase offset caused by rf cable delay [radians]
    @return:
    - s21: np.array
        s21 evaluated across frequency sweep (dark resonator) or time (lit resonator)

    """
    xm = (readout_freq - fc) / fc
    xg = (readout_freq - f0) / f0
    q = (qi ** -1 + qc ** -1) ** -1
    xn = swenson_formula(q * xg, a, increasing) / q

    gain = rf_gain(xm)
    phase = np.exp(1j * (rf_phase_delay + cable_delay_phase * xm))
    q_num = qc + 2 * 1j * qi * qc * (xn + xa)  # use xn instead of xg
    q_den = qi + qc + 2 * 1j * qi * qc * xn
    return gain * phase * (q_num / q_den)  # xm, xg, q, xn, gain, phase, q_num, q_den


def generate_tls_noise(fs, size, scale, seed=4):
    """ two-level system noise
    inputs:
    @param fs: float
        sampling frequency (should match with whatever signal you will apply it to)
    @param size: int
        size of 1D array of points to generate
    @param scale: float
        magnitude of tls noise
    @param seed: int?
        random seed
    scale should be the value of the psd at 1 Hz"""
    random_seed = np.random.default_rng(seed=seed)
    psd_freqs = np.fft.rfftfreq(size, d=1 / fs)
    psd = np.zeros_like(psd_freqs)
    nonzero = psd_freqs != 0
    psd[nonzero] = scale / psd_freqs[nonzero]
    noise_phi = 2 * np.pi * random_seed.random(psd_freqs.size)
    noise_fft = np.exp(1j * noise_phi)  # n_traces x n_frequencies
    # rescale the noise to the covariance
    a = np.sqrt(size * psd * fs / 2)
    noise_fft = a * noise_fft
    return np.fft.irfft(noise_fft, size)


def gen_amp_noise(snr, points, seed=2):
    """ Flat PSD, white-noise generated from voltage fluctuations"""
    random_number_generator = np.random.default_rng(seed=seed)
    a_noise = 10 ** ((20 * np.log10(1 / np.sqrt(2)) - snr) / 10)  # input dBm of noise
    real_noise = np.sqrt(a_noise) * random_number_generator.normal(size=points)
    imag_noise = np.sqrt(a_noise) * random_number_generator.normal(size=points)
    return real_noise + 1j*imag_noise


def compute_phase1(fc, cable_delay):
    """need to ask Nick what this is called. He called it "phase1"
    Inputs:
    fc - readout center frequency in Hz
    cable_delay - RF cable delay in sec"""
    return -2 * np.pi * fc * cable_delay


def compute_background(f, fc, rf_gain, rf_phase_delay, phase1):
    """ Resonator background??"""
    xm = (f - fc) / fc
    gain = rf_gain(xm)
    phase = np.exp(1j * (rf_phase_delay + phase1 * xm))
    return gain * phase


def gen_line_noise(freqs, amps, phases, n_samples, fs):
    """
    Generate time series representing line noise in a single MKID coarse channel (MKID has been centered).
    @param freqs: 1D np.array or list
        frequencies of line noise
    @param amps: 1D np.array or list
        amplitudes of line noise
    @param phases: 1D np.array or list
        phases of line noise
    @param n_samples: int
        number of timeseries samples to produce
    @param fs: float
        sample rate of channel in Hz
    @return:
    """
    freqs = np.asarray(freqs)  # Hz and relative to center of bin (MKID we are reading out)
    amps = np.asarray(amps)
    phases = np.asarray(phases)

    n_samples = n_samples
    sample_rate = fs

    line_noise = np.zeros(n_samples, dtype=np.complex64)
    t = 2 * np.pi * np.arange(n_samples) / sample_rate
    for i in range(freqs.size):
        phi = t * freqs[i]
        exp = amps[i] * np.exp(1j * (phi + phases[i]))
        line_noise += exp
    return line_noise


def lowpass(s21, tau_r, dt):
    """
    Causal lowpass filter which determines the IQ response of an MKID resonator
    ----------------------------------------------------------------------
    @param s21: 1D np.array
        forward scattering matrix element timeseries measured at a single frequency (detector resonance frequency)
    @param tau_r: float
        characteristic timescale for resonator ring up (units must match dt!)
        For an MKID resonator is it ususally:
        [total quality factor (no photon)] / (pi * [original resonance frequency])
    @param dt: float
        time step between S21 sample points (units must match tau_r!)

    @return: 1D np.array
        iq response lowpass filtered by resonator
    """
    # tau_r = q_tot_0/(pi*f0_0)
    # tau_r needs to be in units of dt
    t = np.arange(0, 10 * tau_r, dt)  # filter time series
    t = t[:int(np.round(t.size / 2) * 2)]  # make t an even number of elements
    pad = t.size // 2
    causal_filter = np.exp(-t / tau_r) / tau_r
    full_convolve = np.convolve(np.pad(s21, pad, mode='edge'), causal_filter, mode='same') * dt
    return full_convolve[pad:-pad]


class LineNoise:
    """
    A class to represent the line noise in an MKID readout setup. Line noise is comprised of individual
    extraneous frequencies and usually arises from imperfections in analog and digital electronics.

    Attributes:
    --------------
    """

    def __init__(self, freqs, amplitudes, phases, n_samples, fs):
        self.freqs = freqs
        self.amplitudes = amplitudes
        self.phases = phases
        self.n_samples = n_samples
        self.fs = fs

    @property
    def values(self):
        return gen_line_noise(self.freqs, self.amplitudes, self.phases, self.n_samples, self.fs)


class FrequencyGrid:
    """
    A class to represent the frequency sweep settings used to read out an MKID resonator.
    ...
    Attributes:
    --------------
    @type fc: float
        center frequency [Hz]
    @type points: int
        sweep points
    @type span: float
        sweep bandwidth [Hz]
    """

    def __init__(self, fc=4.0012e9, points=1000, span=500e6):
        self.fc = fc  # center frequency [Hz]
        self.points = int(points)  # sweep points
        self.span = span  # sweep bandwidth [points]
        self.grid = np.linspace(self.fc - 2 * self.span / self.points,
                                self.fc + 2 * self.span / self.points,
                                self.points)

    @property
    def xm(self):
        return self.grid / self.fc - 1

    @property
    def increasing(self):
        if self.grid[1] > self.grid[0]:
            return True
        else:
            return False


class RFElectronics:
    def __init__(self, gain: (np.poly1d, tuple) = (3.0, 0, 0), phase_delay=0, cable_delay=50e-9, white_noise_scale=30,
                 line_noise: LineNoise = LineNoise([500e3], [0.01], [0], 100, 1e3)):
        """
        A class to represent effects of RF cabling and amplifiers on MKID readout.
        ...
        Attributes:
        -----------------
        @type gain: np.poly1D
            gain polynomial coefficients
        @type phase_delay: float
            total loop rotation [radians]
        @type cable_delay: float
            cable delay [sec]
        @type white_noise_scale: float
            dimensionless parameter similar to SNR indicating system white noise
            nominal values are 10 (bad SNR, very noisy) and 30 (good SNR, not too noisy)
        """
        if isinstance(gain, tuple):
            gain = np.poly1d(*gain)
        self.gain = gain
        self.phase_delay = phase_delay  # phase polynomial coefficients (total loop rotation) [radians]
        self.cable_delay = cable_delay
        self.noise_scale = white_noise_scale
        self.line_noise = line_noise


class Resonator:
    def __init__(self, f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1e4):
        """
        A class to represent one MKID resonator.
        ...
        Attributes:
        -----------------
        @type f0: float
            resonance frequency [Hz]
        @type qi: float
            total loop rotation [radians]
        @type qc: float
            cable delay [sec]
        @type xa: float
            inductive nonlinearlity (reference??)
        @type a: float
            resonance fractional asymmetry
        """
        self.f0 = f0
        self.qi = qi
        self.qc = qc
        self.xa = xa
        self.a = a
        self.q_tot = (self.qi ** -1 + self.qc ** -1) ** -1
        self.f0_0 = f0
        self.qi_0 = qi
        self.q_tot_0 = self.q_tot
        self.tls_scale = tls_scale


class ResonatorSweep:
    """
    No photons.
    """

    def __init__(self, res: Resonator, freq: FrequencyGrid, rf: RFElectronics):
        self.res = res
        self.freq = freq
        self.rf = rf

    @property
    def phase1(self):
        """need to ask Nick what this is called. He called it "phase1" """
        return compute_phase1(self.freq.fc, self.rf.cable_delay)

    @property
    def background(self):
        """ Resonator background??"""
        return compute_background(self.res.f0, self.freq.fc, self.rf.gain, self.rf.phase_delay, self.phase1)

    @property
    def s21(self):
        return compute_s21(self.freq.grid, self.freq.fc, self.freq.increasing, self.res.f0, self.res.qi,
                           self.res.qc, self.res.xa, self.res.a, self.rf.phase_delay, self.rf.gain, self.phase1)

    def plot_sweep(self, ax=None, fig=None):
        plt.rcParams.update({'font.size': 12})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{self.res.f0 * 1e-9} GHz Simulated Resonator', fontsize=15)

        ax1.plot(self.freq.grid * 1e-9, 20 * np.log10(np.abs(self.s21)), linewidth=4)
        ax1.set_ylabel('|S21| [dB]')
        ax1.set_xlabel('Frequency [GHz]')
        ax1.set_title('Transmission')
        ax2.plot(self.s21.real, self.s21.imag, 'o')
        ax2.set_xlabel('Real(S21)')
        ax2.set_ylabel('Imag(S21)')
        ax2.set_title('IQ Loop')


class ReadoutPhotonResonator:
    def __init__(self, res: Resonator, photons: QuasiparticleTimeStream, freq: FrequencyGrid, rf: RFElectronics,
                 seed=2, noise_on=False):
        self.res = copy.deepcopy(res)
        self.photons = photons
        self.noise_rng = np.random.default_rng(seed=seed)
        self.freq = freq
        self.rf = rf
        self.rf.line_noise.fs = self.photons.fs
        self.rf.line_noise.n_samples = self.photons.points
        self.tls_noise = generate_tls_noise(self.photons.fs, self.photons.data.size, self.res.tls_scale)
        dfr = -photons.data * 1e5 + self.tls_noise
        dqi_inv = photons.data * 2e-5
        self.res.f0 = res.f0 + dfr
        self.res.qi = (res.qi ** -1 + dqi_inv) ** -1
        self.res.q_tot = (self.res.qi ** -1 + res.qc ** -1) ** -1
        self.res.f0_0 = res.f0  # original resonance frequency
        self.res.q_tot_0 = res.q_tot
        self.noise_on = noise_on

    @property
    def phase1(self):
        """need to ask Nick what this is called. He called it "phase1" """
        return compute_phase1(self.freq.fc, self.rf.cable_delay)

    @property
    def amp_noise(self):
        """ White amplifier noise"""
        return gen_amp_noise(self.rf.noise_scale, self.photons.points)

    @property
    def line_noise(self):
        """ Line noise"""
        return self.rf.line_noise.values

    @property
    def background(self):
        """ Resonator background??"""
        return compute_background(self.res.f0, self.freq.fc, self.rf.gain, self.rf.phase_delay, self.phase1)

    @property
    def s21(self):  # maybe change name to reflect 1/2 noise weirdness
        return compute_s21(self.res.f0_0, self.freq.fc, self.freq.increasing, self.res.f0, self.res.qi,
                           self.res.qc, self.res.xa, self.res.a, self.rf.phase_delay, self.rf.gain, self.phase1)

    @property
    def iq_response(self):
        if self.noise_on:
            return lowpass(self.s21, self.res.q_tot_0 / (np.pi * self.res.f0_0), self.photons.dt)\
                   + self.amp_noise + self.line_noise
        return lowpass(self.s21, self.res.q_tot_0 / (np.pi * self.res.f0_0), self.photons.dt)

    # Add amplifier and line noise after lowpass

    @property
    def normalized_s21(self):
        s21_dark_on_res = compute_s21(self.res.f0_0, self.freq.fc, self.freq.increasing, self.res.f0_0, self.res.qi_0,
                                      self.res.qc, self.res.xa, self.res.a, self.rf.phase_delay, self.rf.gain,
                                      self.phase1)
        return s21_dark_on_res / self.background

    @property
    def normalized_iq(self):
        return self.iq_response / self.background

    def gen2_coordinate_transformation(self):
        i_center = (np.percentile(self.iq_response.real, 95) + np.percentile(self.iq_response.real, 5)) / 2.
        q_center = (np.percentile(self.iq_response.imag, 95) + np.percentile(self.iq_response.imag, 5)) / 2.
        #TODO add loop rotation
        return np.angle(self.iq_response.real - i_center + 1j*(self.iq_response.imag - q_center))

    def basic_coordinate_transformation(self):  # implement a more basic coordinate transformation
        z1 = (1 - self.iq_response / self.background - self.res.q_tot_0 / (2 * self.res.qc) + 1j *
              self.res.q_tot_0 * self.res.xa) / (1 - self.normalized_s21 - self.res.q_tot_0 /
                                                 (2 * self.res.qc) + 1j * self.res.q_tot_0 * self.res.xa)
        theta1 = np.arctan2(z1.imag, z1.real)
        d1 = (np.abs(1 - self.iq_response / self.background - self.res.q_tot_0 / (2 * self.res.qc) +
                     1j * self.res.q_tot_0 * self.res.xa) / np.abs(self.res.q_tot_0 / (2 * self.res.qc) -
                                                                   1j * self.res.q_tot_0 * self.res.xa)) - 1
        return theta1, d1

    def nick_coordinate_transformation(self):
        xn = swenson_formula(0, self.res.a, self.freq.increasing) / self.res.q_tot_0
        theta_2 = -4 * self.res.q_tot_0 / (1 + 4 * self.res.q_tot_0 ** 2 * xn ** 2) * \
                  ((self.normalized_iq.imag + 2 * self.res.qc * self.res.xa * (self.normalized_iq.real - 1)) /
                   (2 * self.res.qc * np.abs(1 - self.normalized_iq) ** 2) - xn)
        d2 = -2 * self.res.q_tot_0 / (1 + 4 * self.res.q_tot_0 ** 2 * xn ** 2) * ((self.normalized_iq.real -
                                                                                   np.abs(
                                                                                       self.normalized_iq) ** 2 + 2 * self.res.qc * self.res.xa *
                                                                                   self.normalized_iq.imag) / (
                                                                                          self.res.qc * np.abs(
                                                                                      1 - self.normalized_iq) ** 2) - 1 / self.res.qi_0)
        return theta_2, d2

    def plot_photon_response(self, s21_dark):
        fig, axes = plt.subplots()
        axes.axis('equal')
        axes.plot(s21_dark.real, s21_dark.imag, 'o')
        axes.plot(self.s21.real, self.s21.imag, '-')
        axes.plot(self.iq_response.real, self.iq_response.imag)
