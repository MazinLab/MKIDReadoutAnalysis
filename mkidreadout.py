import numpy as np
import matplotlib.pyplot as plt
import skimage
from phasetimestream import PhaseTimeStream
from resonator import MeasureResonator
from mkidnoiseanalysis import swenson_formula


def basic_coordinate_transform(meas: MeasureResonator):
    z1 = (1 - meas.normalized_iq - meas.res.q_tot / (2 * meas.res.qc) + 1j * meas.res.q_tot * meas.res.xa) / \
         (1 - meas.normalized_s21 - meas.res.q_tot / (2 * meas.res.qc) + 1j * meas.res.q_tot * meas.res.xa)
    theta1 = np.arctan2(z1.imag, z1.real)
    d1 = (np.abs(1 - meas.normalized_iq - meas.res.q_tot / (2 * meas.res.qc) + 1j * meas.res.q_tot * meas.res.xa) /
          np.abs(meas.res.q_tot / (2 * meas.res.qc) - 1j * meas.res.q_tot * meas.res.xa)) - 1
    return theta1, d1


def nick_coordinate_transformation(meas: MeasureResonator):
    xn = swenson_formula(0, meas.res.a) / meas.res.q_tot
    theta2 = -4 * meas.res.q_tot / (1 + 4 * meas.res.q_tot ** 2 * xn ** 2) * (
                (meas.normalized_iq.imag + 2 * meas.res.qc * meas.res.xa * (meas.normalized_iq.real - 1)) / (
                    2 * meas.res.qc * np.abs(1 - (meas.normalized_iq / meas.background(meas.res.f0))) ** 2) - xn)
    d2 = -2 * meas.res.q_tot / (1 + 4 * meas.res.q_tot ** 2 * xn ** 2) * ((meas.normalized_iq.real - np.abs(
        meas.normalized_iq) ** 2 + 2 * meas.res.qc * meas.res.xa * meas.normalized_iq.imag) / (meas.res.qc * np.abs(
        1 - meas.normalized_iq) ** 2) - 1 / meas.res._qi_dark)
    return theta2, d2


class MKIDReadout:
    """ A class containing readout functions and their specifications.
"""

    def __init__(self, coordinate_transform: str):
        self.optimal_filter = None
        self.trig = None
        self.photon_energies = None
        self.photon_energy_idx = None
        self._trig_holdoff = None
        self.coordinate_transform = coordinate_transform

    def trigger(self, timestream: PhaseTimeStream, data, threshold=-0.7, deadtime=30):
        """ threshold = phase value (really density of quasiparticles in the inductor) one must exceed to trigger
            holdoff: samples to wait before triggering again.
            deadtime: minimum time in microseconds between triggers"""

        self._trig_holdoff = int(deadtime * 1e-6 * timestream.fs)
        all_trig = (data < threshold)  # & (np.diff(phase_data.phase_data, prepend=0)>0)
        trig = all_trig
        for i in range(all_trig.size):
            if all_trig[i]:
                trig[i + 1:i + 1 + self._trig_holdoff] = 0  # impose holdoff
        self.trig = trig
        return self.trig

    def plot_triggers(self, phase_data: PhaseTimeStream, data, energies=False, ax=None, fig=None):
        plt.figure()
        plt.plot(phase_data.tvec * 1e6, data)
        plt.plot(phase_data.tvec[self.trig] * 1e6, data[self.trig], '.')
        plt.xlabel('time (us)')
        plt.ylabel('phase (radians)')
        # plt.xticks(rotation = 45)
        if energies:
            plt.plot(phase_data.tvec[self.photon_energy_idx] * 1e6, data[self.photon_energy_idx], 'o')

    def record_energies(self, data):
        holdoff_views = skimage.util.view_as_windows(data, self._trig_holdoff)  # n_data x holdoff
        trig_views = holdoff_views[self.trig[:holdoff_views.shape[0]]]  # n_triggers x holdoff
        self.photon_energies = np.min(trig_views, axis=1)
        self.photon_energy_idx = np.argmin(trig_views, axis=1) + np.nonzero(self.trig[:holdoff_views.shape[0]])[0]
        return self.photon_energies

    def plot_energies(self, ax=None, fig=None):
        plt.figure()
        plt.hist(self.photon_energies, bins='auto', density=True)
        plt.xlabel('Phase peak (need to change this to energy?)')
        plt.ylabel('counts')
