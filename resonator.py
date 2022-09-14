from logging import getLogger
import numpy as np
from matplotlib import pyplot as plt
from mkidnoiseanalysis import swenson_formula
from phasetimestream import PhaseTimeStream


class Resonator:
    def __init__(self, **res_params):
        self.f0 = res_params['f0']  # resonance frequency
        self.qi = res_params['qi']  # internal quality factor
        self.qc = res_params['qc']  # coupling quality factor
        self.xa = res_params['xa']  # resonance fractional asymmetry
        self.a = res_params['a']  # inductive nonlinearity
        self.alpha = res_params['alpha']  # IQ mixer amplitude imbalance
        self.beta = res_params['beta']  # IQ mixer phase imbalance
        self.gain0 = res_params['gain0']  # gain polynomial coefficients
        self.gain1 = res_params['gain1']
        self.gain2 = res_params['gain2']
        self.phase0 = res_params['phase0']  # phase polynomial coefficients
        self.tau = res_params['tau']

        self.increasing = None
        self.phase1 = None
        self.points = None
        self.span = None
        self.fvec = None
        self.fc = None

    def fsweep(self, **fsweep_params):
        self.increasing = fsweep_params['increasing']  # fsweep direction
        self.fc = fsweep_params['fc']  # center frequency [Hz]
        self.points = fsweep_params['points']  # sweep points
        self.span = fsweep_params['span']  # sweep bandwidth [points]
        self.fvec = np.linspace(self.fc - 2 * self.span / self.points,
                                self.fc + 2 * self.span / self.points,
                                self.points)
        self.phase1 = -2 * np.pi * self.fc * self.tau

        return self.fvec

    def background(self, f):
        xm = (f - self.fc) / self.fc
        xg = (f - self.f0) / self.f0
        q = (self.qi ** -1 + self.qc ** -1) ** -1
        gain = self.gain0 + self.gain1 * xm + self.gain2 * xm ** 2
        phase = np.exp(1j * (self.phase0 + self.phase1 * xm))
        return gain * phase

    @property
    def s21(self):
        xm = (self.fvec - self.fc) / self.fc
        xg = (self.fvec - self.f0) / self.f0
        q = (self.qi ** -1 + self.qc ** -1) ** -1
        xn = swenson_formula(q * xg, self.a, increasing=self.increasing) / q

        gain = self.gain0 + self.gain1 * xm + self.gain2 * xm ** 2
        phase = np.exp(1j * (self.phase0 + self.phase1 * xm))
        q_num = self.qc + 2 * 1j * self.qi * self.qc * (xn + self.xa)
        q_den = self.qi + self.qc + 2 * 1j * self.qi * self.qc * xn
        return gain * phase * (q_num / q_den)

    def plot_trans(self):
        plt.rcParams.update({'font.size': 12})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{self.f0 * 1e-9} GHz Simulated Resonator', fontsize=15)

        ax1.plot(self.fvec * 1e-9, 20 * np.log10(np.abs(self.s21)), linewidth=4)
        ax1.set_ylabel('|S21| [dB]')
        ax1.set_xlabel('Frequency [GHz]')
        ax1.set_title('Transmission')

        ax2.plot(self.s21.real, self.s21.imag, 'o')
        ax2.set_xlabel('Real(S21)')
        ax2.set_ylabel('Imag(S21)')
        ax2.set_title('IQ Loop')

        fig.tight_layout()


class ResonatorResponse(Resonator):
    def __init__(self, resonator: Resonator, phase_timestream: PhaseTimeStream, **res_params):
        super().__init__(**res_params)  # might not need this?
        self.dfr = phase_timestream.data * 1e5  # fractional detuning of the resonance frequency?
        self.f0 = resonator.f0 + self.dfr  # change in resonance frequency
        self.dqi_inv = -phase_timestream.data * 2e-5  # quasiparticle density change
        self.qi = (resonator.qi ** -1 + self.dqi_inv) ** -1
        self.q0 = (self.qi[0] ** -1 + resonator.qc ** -1) ** -1
        self.i_noise = None
        self.q_noise = None
        self.s21_0 = resonator.s21
        self.normalized_s21 = None
        self.normalized_iq = None

    @property
    def iq_response_nonoise(self):
        """Effectivly a lowpass filter."""
        dt = (self.fvec[1] - self.fvec[0]) * 1e-6
        f = np.fft.fftfreq(self.s21.size, d=dt)
        return np.fft.ifft(np.fft.fft(self.s21) / (1 + 2j * self.q0 * f / self.f0))

    def set_iq_noise(self, noise):
        self.i_noise = self.iq_response_nonoise.real + noise
        self.q_noise = self.iq_response_nonoise.imag + noise

    def set_data(self):
        self.normalized_iq = (self.i_noise + 1j * self.q_noise) / self.background(self.f0)
        self.normalized_s21 = self.s21_0 / self.background(self.f0)

    def nick_coordinate_transform(self):
        z1 = (1 - self.normalized_iq - self.q0 / (2 * self.qc) + 1j * self.q0 * self.xa) / \
             (1 - self.normalized_s21 - self.q0 / (2 * self.qc) + 1j * self.q0 * self.xa)
        theta1 = np.arctan2(z1.imag, z1.real)
        d1 = (np.abs(1 - self.normalized_iq - self.q0 / (2 * self.qc) + 1j * self.q0 * self.xa) /
              np.abs(self.q0 / (2 * self.qc) - 1j * self.q0 * self.xa)) - 1
        return theta1, d1

    def basic_coordinate_transformation(self):
        xn = swenson_formula(0, self.a, increasing=self.increasing) / self.q0
        theta2 = -4 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * \
                 ((self.normalized_iq.imag + 2 * self.qc * self.xa *
                   (self.normalized_iq.real - 1)) / (2 * self.qc *
                                                     np.abs(1 - (self.normalized_iq / self.background(
                                                         self.f0))) ** 2) - xn)
        d2 = -2 * self.q0 / (1 + 4 * self.q0 ** 2 * xn ** 2) * ((self.normalized_iq.real - np.abs(
            self.normalized_iq) ** 2 + 2 * self.qc * self.xa * self.normalized_iq.imag) /
                                                                (self.qc * np.abs(1 - self.normalized_iq) ** 2) - 1 /
                                                                self.qi[0])
        return theta2, d2
