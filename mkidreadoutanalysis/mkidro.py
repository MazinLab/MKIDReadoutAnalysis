import numpy as np
import matplotlib.pyplot as plt
import skimage


class MKIDReadout:
    """ A class containing readout functions and their specifications.
    """

    def __init__(self):
        self.trig = None
        self.photon_energies = None
        self.photon_energy_idx = None
        self._trig_holdoff = None

    def record_energies(self, data: np.ndarray):
        """
        Records photon energies.

        Args:
            data: phase timestream data

        Returns:

        Uses a rolling window to create holdoff-long windows of data starting at every possible index.
        Windows which begin with a trigger event are selected as photons.
        The minimum of these windows is computed and recorded as the photon energy.
        """
        holdoff_views = skimage.util.view_as_windows(data, self._trig_holdoff)  # n_data x holdoff
        trig_views = holdoff_views[self.trig[:holdoff_views.shape[0]]]  # n_triggers x holdoff
        self.photon_energies = np.min(trig_views, axis=1)
        self.photon_energy_idx = np.argmin(trig_views, axis=1) + np.nonzero(self.trig[:holdoff_views.shape[0]])[0]
        return

    def trigger(self, data, fs, threshold=-0.7, deadtime=30):
        """ threshold = phase value one must exceed to trigger
            holdoff: samples to wait before triggering again.
            deadtime: minimum time in microseconds between triggers"""

        self._trig_holdoff = int(deadtime * 1e-6 * fs)
        all_trig = (data < threshold)  # & (np.diff(phase_data.phase_data, prepend=0)>0)
        trig = all_trig
        for i in range(all_trig.size):
            if all_trig[i]:
                trig[i + 1:i + 1 + self._trig_holdoff] = 0  # impose holdoff
        self.trig = trig
        self.record_energies(data)
        return

    def plot_triggers(self, data, fs=1e6, energies=False, color=None, xlim=(60000,100000), ax=None, fig=None):
        tvec = (np.arange(data.shape[0])*1/fs)*1e6

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(tvec[xlim[0]:xlim[1]], data[xlim[0]:xlim[1]], label='phase timestream', color=color)
        ax.plot(np.ma.masked_outside(tvec[self.trig], xlim[0], xlim[1]), data[self.trig], '.', label='trigger')
        ax.set_xlabel('time (us)')
        ax.set_ylabel('phase (radians)')
        if energies:
            ax.plot(np.ma.masked_outside(tvec[self.photon_energy_idx], xlim[0], xlim[1]), data[self.photon_energy_idx], 'o', label='energy')
        ax.legend(loc='upper right')


    def plot_energies(self, color=None, ax=None, fig=None):
        plt.figure()
        plt.hist(self.photon_energies, color=color, bins='auto', density=True)
        plt.xlabel('Phase peak (need to change this to energy?)')
        plt.ylabel('counts')
