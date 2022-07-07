import matplotlib.pyplot as plt
import numpy as np


def swenson_formula(y0, a, increasing=True):
    """doi: 10.1063/1.4903855"""
    y0 = np.atleast_1d(y0)
    y = np.empty_like(y0)
    for i, y0_i in enumerate(y0):
        roots = np.roots([4, -4 * y0_i, 1, -(y0_i + a)])
        if increasing:
            y[i] = np.min(roots[np.isreal(roots)].real)
        else:
            y[i] = np.max(roots[np.isreal(roots)].real)
    return y


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

        self.s21 = None
        self.increasing = None
        self.phase1 = None
        self.points = None
        self.span = None
        self.fvec = None

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
        q = (self.qi**-1 + self.qc**-1)**-1
        gain = self.gain0 + self.gain1 * xm + self.gain2 * xm ** 2
        phase = np.exp(1j * (self.phase0 + self.phase1 * xm))
        return gain * phase

    def compute_s21(self):
        xm = (self.fvec - self.fc) / self.fc
        xg = (self.fvec - self.f0) / self.f0
        q = (self.qi**-1 + self.qc**-1)**-1
        xn = swenson_formula(q * xg, self.a, increasing=self.increasing) / q

        gain = self.gain0 + self.gain1 * xm + self.gain2 * xm ** 2
        phase = np.exp(1j * (self.phase0 + self.phase1 * xm))
        q_num = self.qc + 2 * 1j * self.qi * self.qc * (xn + self.xa)  # use xn instead of xg
        q_den = self.qi + self.qc + 2 * 1j * self.qi * self.qc * xn
        self.s21 = gain * phase * (q_num / q_den)
        return self.s21

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


if __name__ == "__main__":
    # Resonator Properties
    res_params = {'f0': 4.0012e9,  # resonance frequency [Hz]
                   'qi': 100000,  # internal quality factor
                   'qc': 30000,  # coupling quality factor
                   'xa': 5e-6,  # resonance fractional asymmetry
                   'a': 0,  # inductive nonlinearity
                   'alpha': 1.,  # IQ mixer amplitude imbalance
                   'beta': 0.,  # IQ mixer phase imbalance
                   'gain0': 3.0,  # gain polynomial coefficients
                   'gain1': 0,  # linear gain coefficient
                   'gain2': 0,  # quadratic gain coefficient
                   'phase0': 0,  # total loop rotation in radians
                   'tau': 50e-9}  # cable delay

    fsweep_params = {'fc': 4.0012e9,   # center frequency [Hz]
                     'points': 1000,   # frequency sweep points
                     'span': 500e6, # frequency sweep bandwidth [Hz]
                     'increasing': True}   # frequency sweep direction

    res = Resonator(**res_params)
    res.fsweep(**fsweep_params)
    res.compute_s21()
    res.plot_trans()