import numpy as np
from matplotlib import pyplot as plt
from mkidnoiseanalysis import swenson_formula
from phasetimestream import PhaseTimeStream


class Resonator:
    def __init__(self, f0=4.0012e9, qi=200000, qc=15000, xa=0.5, a=0):
        self.f0 = f0  # resonance frequency
        self.qi = qi  # internal quality factor
        self.qc = qc  # coupling quality factor
        self.xa = xa  # resonance fractional asymmetry
        self.a = a  # inductive nonlinearity


class MixerImbalance:
    def __init__(self, alpha=1., beta=0.):
        self.alpha = alpha  # amplitude
        self.beta = beta  # phase


class RFElectronics:
    def __init__(self, gain: (np.poly1d, tuple) = (3.0, 0, 0), phase_delay=0, cable_delay=50e-9):
        """
        gain: np.poly1D (3 coefficients is sufficient)
        phase0: total loop rotation [radians]
        cable_delay: cable delay [sec]
        """
        if isinstance(gain, tuple):
            gain = np.poly1d(*gain)
        self.gain = gain  # gain polynomial coefficients
        self.phase_delay = phase_delay  # phase polynomial coefficients (total loop rotation) [radians]
        self.cable_delay = cable_delay  # cable delay


class FrequencyGrid:
    """
    fc: center frequency [Hz]
    points: sweep points
    span: sweep bandwidth [Hz]
    """

    def __init__(self, fc=4.0012e9, points=1000, span=500e6):
        self.fc = fc  # center frequency [Hz]
        self.points = points  # sweep points
        self.span = span  # sweep bandwidth [points]
        self.grid = np.linspace(self.fc - 2 * self.span / self.points,
                                self.fc + 2 * self.span / self.points,
                                self.points)

    @property
    def increasing(self):
        return self.grid[1] > self.grid[0]


class Noise:
    """noise"""

    def __init__(self, tls, amplifier, line):
        self.tls = tls
        self.amplifier = amplifier
        self.line = line


class MeasureResonator:
    def __init__(self, res: Resonator,
                 mixer: MixerImbalance,
                 freq: FrequencyGrid,
                 rf: RFElectronics):
        """Note that arguments aren't copied and may be mutated by method calls!"""
        self.freq = freq
        self.mixer = mixer
        self.rf = rf
        self.res = res

    @property
    def res_phase(self):
        """need to ask Nick what this is called. He called it "phase1" """
        return -2 * np.pi * self.freq.fc * self.rf.cable_delay

    @property
    def background(self):
        xm = self.freq.grid / self.res.f0 - 1
        gain = self.rf.gain(xm)
        phase = np.exp(1j * (self.rf.phase_delay + self.res_phase * xm))
        return gain * phase

    @property
    def s21(self):
        xm = (self.freq.grid - self.freq.fc) / self.freq.fc
        xg = (self.freq.grid - self.res.f0) / self.res.f0
        q = (self.res.qi ** -1 + self.res.qc ** -1) ** -1
        xn = swenson_formula(q * xg, self.res.a, self.freq.increasing) / q

        phase = np.exp(1j * (self.rf.phase_delay + self.res_phase * xm))
        q_num = self.res.qc + 2 * 1j * self.res.qi * self.res.qc * (xn + self.res.xa)
        q_den = self.res.qi + self.res.qc + 2 * 1j * self.res.qi * self.res.qc * xn
        return self.rf.gain(xm) * phase * (q_num / q_den)

    def plot_trans(self, ax=None, fig=None):
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

        fig.tight_layout()


class ResonatorResponse(MeasureResonator):
    def __init__(self, phase_timestream: PhaseTimeStream, *args):
        super().__init__(*args),
        self.dfr = phase_timestream.data * 1e5  # check with nick fractional detuning of the resonance frequency?
        self.f0 = self.res.f0 + self.dfr  # change in resonance frequency
        self.dqi_inv = -phase_timestream.data * 2e-5  # quasiparticle density change
        self.qi = (self.res.qi ** -1 + self.dqi_inv) ** -1
        self.q0 = (self.qi[0] ** -1 + self.res.qc ** -1) ** -1
        self._tlsnoise = phase_timestream.tls_noise

    @property
    def s21_0(self):
        return super().s21

    @property
    def iq_response_nonoise(self):
        """Effectivly a lowpass filter."""
        dt = (self.freq.grid[1] - self.freq.grid[0]) * 1e-6
        f = np.fft.fftfreq(self.s21.size, d=dt)
        return np.fft.ifft(np.fft.fft(self.s21) / (1 + 2j * self.q0 * f / self.f0))

    @property
    def noisy_iq_response(self):
        return self.iq_response_nonoise.real + self._tlsnoise + 1j*(self.iq_response_nonoise.imag + self._tlsnoise)

    @property
    def normalized_s21(self):
        return self.s21_0 / self.background(self.f0)

    @property
    def normalized_iq(self):
        return self.noisy_iq_response / self.background(self.f0)

    @property
    def basic_coordinate_transform(self):
        z1 = (1 - self.normalized_iq - self.q0 / (2 * self.res.qc) + 1j * self.q0 * self.res.xa) / \
             (1 - self.normalized_s21 - self.q0 / (2 * self.res.qc) + 1j * self.q0 * self.res.xa)
        theta1 = np.arctan2(z1.imag, z1.real)
        d1 = (np.abs(1 - self.normalized_iq - self.q0 / (2 * self.res.qc) + 1j * self.q0 * self.res.xa) /
              np.abs(self.q0 / (2 * self.res.qc) - 1j * self.q0 * self.res.xa)) - 1
        return theta1, d1

    @property
    def nick_coordinate_transformation(self):
        xn = swenson_formula(0, self.res.a, self.freq.increasing) / self.q0
        theta2 = -4 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * ((self.normalized_iq.imag + 2 * self.res.qc * self.res.xa * (self.normalized_iq.real - 1)) / (2 * self.res.qc * np.abs(1 - (self.normalized_iq / self.background(self.f0))) ** 2) - xn)
        d2 = -2 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * ((self.normalized_iq.real - np.abs(
            self.normalized_iq) ** 2 + 2 * self.res.qc * self.res.xa * self.normalized_iq.imag) /
                                                                (self.res.qc * np.abs(
                                                                    1 - self.normalized_iq) ** 2) - 1 /
                                                                self.qi[0])
        return theta2, d2


