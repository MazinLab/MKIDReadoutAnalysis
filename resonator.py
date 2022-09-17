import numpy as np
from matplotlib import pyplot as plt
from mkidnoiseanalysis import swenson_formula
from phasetimestream import PhaseTimeStream
import copy


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
        self._f0_dark = f0
        self._qi_dark = qi
        self._lit = False


def illuminate_resonator(photons: PhaseTimeStream, res: Resonator):

    lit_res = copy.deepcopy(res)
    dfr = photons.data_nonoise * 1e5 + photons.tls_noise
    dqi_inv = -photons.data_nonoise * 2e-5
    lit_res.f0 = res.f0 + dfr
    lit_res.qi = (res.qi ** -1 + dqi_inv) ** -1
    lit_res.q_tot = (lit_res.qi ** -1 + res.qc ** -1) ** -1
    lit_res.tls_noise = photons.tls_noise
    lit_res._lit = True
    return lit_res


class MeasureResonator:
    def __init__(self, res: Resonator, freq: FrequencyGrid, rf: RFElectronics):
        self.res = res
        self.freq = freq
        self.rf = rf
        if res._lit: # TODO not sure how to handle this
            res.q_tot = res.q_tot[:freq.points]
            res.qi = res.qi[:freq.points]
            res.f0 = res.f0[:freq.points]

    @property
    def phase1(self):
        """need to ask Nick what this is called. He called it "phase1" """
        return -2 * np.pi * self.freq.fc * self.rf.cable_delay

    @property
    def background(self):
        """ Resonator background??"""
        xm = self.freq.grid / self.res.f0 - 1
        gain = self.rf.gain(xm)
        phase = np.exp(1j * (self.rf.phase_delay + self.phase1 * xm))
        return gain * phase

    @property
    def s21(self):
        xg = self.freq.grid / self.res.f0 - 1
        q = (self.res.qi ** -1 + self.res.qc ** -1) ** -1
        xn = swenson_formula(q * xg, self.res.a) / q
        res_phase = -2 * np.pi * self.freq.fc * self.rf.cable_delay
        phase = np.exp(1j * (self.rf.phase_delay + res_phase * self.freq.xm))
        q_num = self.res.qc + 2 * 1j * self.res.qi * self.res.qc * (xn + self.res.xa)
        q_den = self.res.qi + self.res.qc + 2 * 1j * self.res.qi * self.res.qc * xn
        return self.rf.gain(self.freq.xm) * phase * (q_num / q_den)

    @property
    def _s21_0(self):
        xg = self.freq.grid / self.res._f0_dark - 1
        q = (self.res._qi_dark ** -1 + self.res.qc ** -1) ** -1
        xn = swenson_formula(q * xg, self.res.a) / q
        res_phase = -2 * np.pi * self.freq.fc * self.rf.cable_delay
        phase = np.exp(1j * (self.rf.phase_delay + res_phase * self.freq.xm))
        q_num = self.res.qc + 2 * 1j * self.res.qi * self.res.qc * (xn + self.res.xa)
        q_den = self.res._qi_dark + self.res.qc + 2 * 1j * self.res._qi_dark * self.res.qc * xn
        return self.rf.gain(self.freq.xm) * phase * (q_num / q_den)

    @property
    def iq_response_nonoise(self):
        """Effectivly a lowpass filter."""
        dt = (self.freq.grid[1] - self.freq.grid[0]) * 1e-6
        f = np.fft.fftfreq(self.s21.size, d=dt)
        return np.fft.ifft(np.fft.fft(self.s21) / (1 + 2j * self.res.q_tot * f / self.res.f0[0]))

    @property
    def noisy_iq_response(self):
        response = self.iq_response_nonoise()
        return response.real + self.res.tls_noise + 1j * (response.imag + self.res.tls_noise)

    @property
    def normalized_s21(self):
        return self.s21 / self.background()

    @property
    def normalized_iq(self):
        return self.noisy_iq_response / self.background()

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