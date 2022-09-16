import numpy as np
from matplotlib import pyplot as plt
from mkidnoiseanalysis import swenson_formula
from phasetimestream import PhaseTimeStream


def compute_s21(rf: RFElectronics, freq: FrequencyGrid, res: Resonator):
    xg = freq.grid / res.f0 - 1
    q = (res.qi ** -1 + res.qc ** -1) ** -1
    xn = swenson_formula(q * xg, res.a) / q
    res_phase = -2 * np.pi * freq.fc * rf.cable_delay
    phase = np.exp(1j * (rf.phase_delay + res_phase * freq.xm))
    q_num = res.qc + 2 * 1j * res.qi * res.qc * (xn + res.xa)
    q_den = res.qi + res.qc + 2 * 1j * res.qi * res.qc * xn
    return rf.gain(freq.xm) * phase * (q_num / q_den)


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
        self.points = points  # sweep points
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

    def phase1(self, freq: FrequencyGrid):
        """need to ask Nick what this is called. He called it "phase1" """
        return -2 * np.pi * freq.fc * self.cable_delay

    def background(self, f0, freq: FrequencyGrid):
        xm = freq.grid / f0 - 1
        gain = self.gain(xm)
        phase = np.exp(1j * (self.phase_delay + self.phase1(freq) * xm))
        return gain * phase


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

    def s21(self, freq: FrequencyGrid, rf: RFElectronics):
        return compute_s21(rf, freq, self)

    def background(self, freq: FrequencyGrid, rf: RFElectronics):
        xm = freq.grid / self.f0 - 1
        gain = rf.gain(xm)
        phase = np.exp(1j * (rf.phase_delay + self.res_phase(freq, rf) * xm))
        return gain * phase

    def plot_trans(self, freq: FrequencyGrid, rf: RFElectronics, ax=None, fig=None):
        plt.rcParams.update({'font.size': 12})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{self.f0 * 1e-9} GHz Simulated Resonator', fontsize=15)

        ax1.plot(freq.grid * 1e-9, 20 * np.log10(np.abs(self.s21(self, freq, rf))), linewidth=4)
        ax1.set_ylabel('|S21| [dB]')
        ax1.set_xlabel('Frequency [GHz]')
        ax1.set_title('Transmission')

        ax2.plot(self.s21.real, self.s21.imag, 'o')
        ax2.set_xlabel('Real(S21)')
        ax2.set_ylabel('Imag(S21)')
        ax2.set_title('IQ Loop')

        fig.tight_layout()


class LitResonator(Resonator):
    def __init__(self, photons: PhaseTimeStream, res: Resonator):
        super().__init__()
        self.dfr = photons.data_nonoise * 1e5 + photons.tls_noise
        self.dqi_inv = -photons.data_nonoise * 2e-5
        self.f0 = res.f0 + self.dfr
        self.qi = (res.qi ** -1 + self.dqi_inv) ** -1
        self.q0 = (res.qi ** -1 + res.qc ** -1) ** -1
        self.tls_noise = photons.tls_noise

    def iq_response_nonoise(self, freq: FrequencyGrid, rf: RFElectronics):
        """Effectivly a lowpass filter."""
        dt = (freq.grid[1] - freq.grid[0]) * 1e-6
        f = np.fft.fftfreq(self.s21(self, freq, rf).size, d=dt)
        return np.fft.ifft(np.fft.fft(self.s21(self, freq, rf)) / (1 + 2j * self.q0 * f / self.f0[0]))

    def noisy_iq_response(self, freq: FrequencyGrid, rf: RFElectronics):
        response = self.iq_response_nonoise(freq, rf)
        return response.real + self.tls_noise + 1j * (response.imag + self.tls_noise)

    def normalized_s21(self):
        return self.s21_0 / self.background(self.f0)

    @property
    def normalized_iq(self):
        return self.noisy_iq_response / self.background(self.f0)


"""
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
        theta2 = -4 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * (
                (self.normalized_iq.imag + 2 * self.res.qc * self.res.xa * (self.normalized_iq.real - 1)) / (
                2 * self.res.qc * np.abs(1 - (self.normalized_iq / self.background(self.f0))) ** 2) - xn)
        d2 = -2 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * ((self.normalized_iq.real - np.abs(
            self.normalized_iq) ** 2 + 2 * self.res.qc * self.res.xa * self.normalized_iq.imag) /
                                                                (self.res.qc * np.abs(
                                                                    1 - self.normalized_iq) ** 2) - 1 /
                                                                self.qi[0])
        return theta2, d2
"""
