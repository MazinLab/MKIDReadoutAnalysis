import numpy as np
from matplotlib import pyplot as plt
from mkidnoiseanalysis import swenson_formula
from phasetimestream import PhaseTimeStream
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


def lowpass(s21, f0, q, dt):
    f = np.fft.fftfreq(s21.size, d=dt)
    z = np.fft.ifft(np.fft.fft(s21) / (1 + 2j * q * f / f0))
    return z


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
    def __init__(self, gain: (np.poly1d, tuple) = (3.0, 0, 0), phase_delay=0, cable_delay=50e-9):
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
        """
        if isinstance(gain, tuple):
            gain = np.poly1d(*gain)
        self.gain = gain  # gain polynomial coefficients
        self.phase_delay = phase_delay  # phase polynomial coefficients (total loop rotation) [radians]
        self.cable_delay = cable_delay  # cable delay


class Resonator:
    def __init__(self, f0=4.0012e9, qi=200000, qc=15000, xa=0.5, a=0):
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
        self.tls_noise = None
        self.f0_0 = f0
        self.qi_0 = qi
        self.q_tot_0 = self.q_tot


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
    def __init__(self, res: Resonator, photons: PhaseTimeStream, freq: FrequencyGrid, rf: RFElectronics):
        self.res = copy.deepcopy(res)
        self.photons = photons
        self.freq = freq
        self.rf = rf

        dfr = photons.data_nonoise * 1e5 + photons.tls_noise
        dqi_inv = -photons.data_nonoise * 2e-5
        self.res.f0 = res.f0 + dfr
        self.res.qi = (res.qi ** -1 + dqi_inv) ** -1
        self.res.q_tot = (self.res.qi ** -1 + res.qc ** -1) ** -1
        self.res.tls_noise = photons.tls_noise
        self.res.f0_0 = res.f0  # original resonance frequency
        self.res.q_tot_0 = res.q_tot

        # res.q_tot = res.q_tot[:freq.points]
        # res.qi = res.qi[:freq.points]
        # res.f0 = res.f0[:freq.points]

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
        return compute_s21(self.res.f0_0, self.freq.fc, self.freq.increasing, self.res.f0, self.res.qi,
                           self.res.qc, self.res.xa, self.res.a, self.rf.phase_delay, self.rf.gain, self.phase1)

    @property
    def iq_response_nonoise(self):
        return lowpass(self.s21, self.res.f0_0, self.res.q_tot_0, self.photons.dt)

    @property
    def noisy_iq_response(self):
        response = self.iq_response_nonoise
        return response.real + self.res.tls_noise + 1j * (response.imag + self.res.tls_noise)

    @property
    def normalized_s21(self):
        s21_dark_on_res = compute_s21(self.res.f0_0, self.freq.fc, self.freq.increasing, self.res.f0_0, self.res.qi_0,
                                      self.res.qc, self.res.xa, self.res.a, self.rf.phase_delay, self.rf.gain,
                                      self.phase1)
        return s21_dark_on_res / self.background

    @property
    def normalized_iq(self):
        return self.noisy_iq_response / self.background

    def basic_coordinate_transformation(self):
        z1 = (1 - self.noisy_iq_response / self.background - self.res.q_tot_0 / (2 * self.res.qc) + 1j *
              self.res.q_tot_0 * self.res.xa) / (1 - self.normalized_s21 - self.res.q_tot_0 /
                                                 (2 * self.res.qc) + 1j * self.res.q_tot_0 * self.res.xa)
        theta1 = np.arctan2(z1.imag, z1.real)
        d1 = (np.abs(1 - self.noisy_iq_response / self.background - self.res.q_tot_0 / (2 * self.res.qc) +
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
        axes.plot(self.iq_response_nonoise.real, self.iq_response_nonoise.imag)

        # fig, axes = plt.subplots()
        # axes.plot(theta1, color='C0')
        # axes.plot(d1, color='C1')
        # axes.plot(theta2, linestyle=":", color='C0')
        # axes.plot(d2, linestyle=':', color='C1')
