import numpy as np
import matplotlib.pyplot as plt
import skimage
from .quasiparticletimestream import QuasiparticleTimeStream


class MKIDReadout:
    """ A class containing readout functions and their specifications.
"""

    def __init__(self):
        self.trig = None
        self.photon_energies = None
        self.photon_energy_idx = None
        self._trig_holdoff = None

    def trigger(self, timestream: QuasiparticleTimeStream, data, threshold=-0.7, deadtime=30):
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

    def plot_triggers(self, phase_data: QuasiparticleTimeStream, data, energies=False, ax=None, fig=None):
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
