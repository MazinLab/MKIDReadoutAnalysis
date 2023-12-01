import os
import logging
import numpy as np
from scipy.stats import norm
from scipy.signal import welch, csd
from scipy.ndimage import find_objects, binary_dilation, label as binary_label

from mkidcalculator.io.data import AnalogReadoutNoise, NoData
from mkidcalculator.io.utils import (compute_phase_and_dissipation, offload_data, _loaded_npz_files, dump, load,
                                     setup_axes, finalize_axes)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    """A class for manipulating the noise data."""
    def __init__(self):
        # original file name
        self.name = None
        self._data = NoData()  # dummy class replaced by from_file()
        # loop reference for computing phase and dissipation
        self._loop = None
        # phase and dissipation data
        self._p_trace = None
        self._d_trace = None
        # noise data
        self._f_psd = None
        self._ii_psd = None
        self._qq_psd = None
        self._iq_psd = None
        self._pp_psd = None
        self._dd_psd = None
        self._pd_psd = None
        self._n_samples = None  # for generate_noise()
        # for holding large data
        self._npz = None
        self._directory = None
        log.debug("Noise object created. ID: {}".format(id(self)))

    def __getstate__(self):
        return offload_data(self, excluded_keys=("_d_trace", "_p_trace"), prefix="noise_data_")

    @property
    def f_bias(self):
        """The bias frequency for the data set."""
        return self._data["f_bias"]

    @property
    def i_trace(self):
        """The mixer I output traces."""
        return self._data["i_trace"]

    @property
    def q_trace(self):
        """The mixer Q output traces."""
        return self._data["q_trace"]

    @property
    def metadata(self):
        """A dictionary containing metadata about the noise."""
        return self._data["metadata"]

    @property
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def sample_rate(self):
        """The sample rate of the IQ data."""
        return self._data['sample_rate']

    @property
    def loop(self):
        """
        A settable property that contains the Loop object required for doing
        noise calculations like computing the phase and dissipation traces. If
        the loop has not been set, it will raise an AttributeError. When the
        loop is set, all information created from the previous loop is deleted.
        """
        if self._loop is None:
            raise AttributeError("The loop object for this noise has not been set yet.")
        return self._loop

    @loop.setter
    def loop(self, loop):
        if self._loop is not loop:
            self.clear_loop_data()
        self._loop = loop

    @property
    def p_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_dissipation() is run.
        """
        if self._p_trace is None:
            raise AttributeError("The phase information has not been computed yet.")
        if isinstance(self._p_trace, str):
            return _loaded_npz_files[self._npz][self._p_trace]
        else:
            return self._p_trace

    @p_trace.setter
    def p_trace(self, phase_trace):
        self._p_trace = phase_trace

    @property
    def d_trace(self):
        """
        A settable property that contains the dissipation trace information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_dissipation() is run.
        """
        if self._d_trace is None:
            raise AttributeError("The dissipation information has not been computed yet.")
        if isinstance(self._d_trace, str):
            return _loaded_npz_files[self._npz][self._d_trace]
        else:
            return self._d_trace

    @d_trace.setter
    def d_trace(self, dissipation_trace):
        self._d_trace = dissipation_trace

    @property
    def f_psd(self):
        """
        A settable property that contains the noise frequency information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._f_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._f_psd

    @f_psd.setter
    def f_psd(self, f_psd):
        self._f_psd = f_psd

    @property
    def ii_psd(self):
        """
        A settable property that contains the II noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._ii_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._ii_psd

    @ii_psd.setter
    def ii_psd(self, ii_psd):
        self._ii_psd = ii_psd

    @property
    def qq_psd(self):
        """
        A settable property that contains the QQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._qq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._qq_psd

    @qq_psd.setter
    def qq_psd(self, qq_psd):
        self._qq_psd = qq_psd

    @property
    def iq_psd(self):
        """
        A settable property that contains the IQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._iq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._iq_psd

    @iq_psd.setter
    def iq_psd(self, iq_psd):
        self._iq_psd = iq_psd

    @property
    def pp_psd(self):
        """
        A settable property that contains the PP noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_dissipation() and noise.compute_psd() are run.
        """
        if self._pp_psd is None:
            raise AttributeError("The phase and dissipation noise has not been computed yet.")
        return self._pp_psd

    @pp_psd.setter
    def pp_psd(self, pp_psd):
        self._pp_psd = pp_psd

    @property
    def dd_psd(self):
        """
        A settable property that contains the DD noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_dissipation() and noise.compute_psd() are run.
        """
        if self._dd_psd is None:
            raise AttributeError("The phase and dissipation noise has not been computed yet.")
        return self._dd_psd

    @dd_psd.setter
    def dd_psd(self, dd_psd):
        self._dd_psd = dd_psd

    @property
    def pd_psd(self):
        """
        A settable property that contains the PD noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_dissipation() and noise.compute_psd() are run.
        """
        if self._pd_psd is None:
            raise AttributeError("The phase and dissipation noise has not been computed yet.")
        return self._pd_psd

    @pd_psd.setter
    def pd_psd(self, pd_psd):
        self._pd_psd = pd_psd

    def clear_loop_data(self):
        """Remove all data calculated from the noise.loop attribute."""
        self.clear_traces()
        self.dd_psd = None
        self.pp_psd = None
        self.pd_psd = None

    def clear_traces(self):
        """
        Remove all trace data calculated from noise.i_trace and noise.q_trace.
        """
        self.d_trace = None
        self.p_trace = None
        self._npz = None

    def free_memory(self, directory=None):
        """
        Offloads d_traces and p_traces to an npz file if they haven't been
        offloaded already and removes any npz file objects from memory, keeping
        just the file name. It doesn't do anything if they don't exist.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the noise was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        offload_data(self, excluded_keys=("_d_trace", "_p_trace"), prefix="noise_data_")
        if isinstance(self._npz, str):  # there might not be an npz file yet
            _loaded_npz_files.free_memory(self._npz)
        try:
            self._data.free_memory()
        except AttributeError:
            pass

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        dump(self, file_name)
        log.info("saved noise as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Noise class from the pickle file 'file_name'."""
        noise = load(file_name)
        if not isinstance(noise, cls):
            raise ValueError(f"'{file_name}' does not contain a Noise class.")
        log.info("loaded noise from '{}'".format(file_name))
        return noise

    @classmethod
    def from_file(cls, noise_file_name, data=AnalogReadoutNoise, **kwargs):
        """
        Noise class factory method that returns a Noise() with the data loaded.
        Args:
            noise_file_name: string
                The file name for the noise data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module. The
                return value may also be a list of these objects.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            noise: object or list
                A Noise() or list of Noise() objects containing the loaded
                data.
        """
        _data = data(noise_file_name, **kwargs)
        if not isinstance(_data, list):
            _data = [_data]
        noise = []
        for d in _data:
            noise.append(cls())
            noise[-1]._data = d
            noise[-1].name = os.path.basename(noise_file_name) + ", " + str(kwargs)
        if len(noise) == 1:
            noise = noise[0]
        return noise

    def compute_phase_and_dissipation(self, label="best", fit_type="lmfit", **kwargs):
        """
        Compute the phase and dissipation traces stored in noise.p_trace and
        noise.d_trace.
        Args:
            label: string
                Corresponds to the label in the loop.lmfit_results or
                loop.emcee_results dictionaries where the fit parameters are.
                The default is "best", which gets the parameters from the best
                fits.
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            kwargs: optional keyword arguments
                Optional keyword arguments to send to
                model.phase_and_dissipation().
        """
        compute_phase_and_dissipation(self, label=label, fit_type=fit_type, **kwargs)

    def compute_psd(self, sigma_threshold=np.inf, grow=0, **kwargs):
        """
        Compute the noise power spectral density of the noise data in this
        object.
        Args:
            sigma_threshold: float (optional)
                Exclude data that deviates from the median at a level that
                exceeds sigma_threshold * (noise standard deviation) assuming
                stationary gaussian noise. The default is numpy.inf, and all
                data is used. This keyword argument is useful for removing
                pulse contamination from the computed noise.
            grow: integer (optional)
                If p_threshold is used, the regions of excluded points can be
                expanded by grow time steps.
            kwargs: optional keyword arguments
                keywords for the scipy.signal.welch and scipy.signal.csd
                methods. The spectrum scaling and two-sided spectrum of the PSD
                can not be changed since they are assumed in other methods.
        """
        # mask data
        if not np.isinf(sigma_threshold):
            # compute the standard deviation from the median absolute deviation
            abs_di = np.abs(self.i_trace - np.median(self.i_trace))
            abs_dq = np.abs(self.q_trace - np.median(self.q_trace))
            std_i = np.median(abs_di) / norm.ppf(0.75)
            std_q = np.median(abs_dq) / norm.ppf(0.75)
            exclude = (abs_di > sigma_threshold * std_i) | (abs_dq > sigma_threshold * std_q)
            # grow the exclusion regions
            if grow > 0:
                exclude = binary_dilation(exclude, structure=[[1] * (1 + 2 * grow)])
        else:
            exclude = np.zeros_like(self.i_trace, dtype=bool)  # no exclusions
        # get regions from excluded mask
        include, n_regions = binary_label(~exclude, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        regions = np.array(find_objects(include))
        region_size = np.array([region[-1].stop - region[-1].start for region in regions])
        if region_size.size == 0:
            raise ValueError("There are no regions in the data with any good points")
        # update keyword arguments
        noise_kwargs = {'nperseg': max(region_size), 'fs': self.sample_rate, 'return_onesided': True,
                        'detrend': 'constant', 'scaling': 'density'}
        noise_kwargs.update(kwargs)
        if noise_kwargs['scaling'] != 'density':
            raise ValueError("The PSD scaling is not an allowed keyword.")
        if not noise_kwargs['return_onesided']:
            raise ValueError("A two-sided PSD is not an allowed keyword.")
        # remove regions smaller than nperseg
        nperseg = noise_kwargs['nperseg']
        regions = regions[region_size >= nperseg, :]
        if regions.size == 0:
            raise ValueError("There are no regions in the data with at least {} (nperseg) good points".format(nperseg))
        # compute I/Q noise in V^2 / Hz
        ii_psd, qq_psd, iq_psd = [], [], []
        for region in regions:
            self.f_psd, psd = welch(self.i_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)
            ii_psd.append(psd)
            qq_psd.append(welch(self.q_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)[1])
            # scipy has different order convention we use equation 5.2 from J. Gao's 2008 thesis.
            # noise_iq = F(I) conj(F(Q))
            iq_psd.append(csd(self.q_trace[tuple(region.tolist())].squeeze(),
                              self.i_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)[1])
        # average multiple PSDs together
        self.ii_psd = np.mean(ii_psd, axis=0)
        self.qq_psd = np.mean(qq_psd, axis=0)
        self.iq_psd = np.mean(iq_psd, axis=0)
        # record n_samples for generate_noise()
        self._n_samples = nperseg
        # compute phase and dissipation noise in rad^2 / Hz
        try:
            pp_psd, dd_psd, pd_psd = [], [], []
            for region in regions:
                pp_psd.append(welch(self.p_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)[1])
                dd_psd.append(welch(self.d_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)[1])
                pd_psd.append(csd(self.d_trace[tuple(region.tolist())].squeeze(),
                                  self.p_trace[tuple(region.tolist())].squeeze(), **noise_kwargs)[1])
            # average multiple PSDs together
            self.pp_psd = np.mean(pp_psd, axis=0)
            self.dd_psd = np.mean(dd_psd, axis=0)
            self.pd_psd = np.mean(pd_psd, axis=0)
        except AttributeError:
            pass

    def generate_noise(self, noise_type="pd", n_traces=10000, psd=None):
        """
        Generate fake noise traces from the computed PSDs.
        Args:
            noise_type: string
                The type of noise to generate. Valid options are "pd", "p",
                "d", "iq", "i", "q", which correspond to phase, dissipation, I,
                and Q.
            n_traces: integer
                The number of noise traces to make.
            psd: length 3 iterable of numpy.arrays
                PSD to use to generate the noise in the form
                (PSD_00, PSD_11, PSD_01). Components that aren't used can be
                set to None.
        Returns:
            noise: np.ndarray
                If noise_type == "pd" (or "iq"), a 2 x n_traces x N array of
                noise is made where the first dimension is phase then
                dissipation or (I then Q).
                If noise_type == "p" or "d" or "i" or "q" a n_traces x N array
                of noise is made.
        """
        # check parameters
        noise_types = ["pd", "p", "d", "iq", "i", "q"]
        if noise_type not in noise_types:
            raise ValueError("'noise_type' is not in {}".format(noise_types))
        # get constants
        dt = 1 / self.sample_rate
        if psd is not None:
            psd_00, psd_11, psd_01 = psd
            if hasattr(psd_00, "size"):
                n_frequencies = psd_00.size
            else:
                n_frequencies = psd_11.size
        elif noise_type in ["pd", "p", "d"]:
            psd_00 = self.pp_psd
            psd_01 = self.pd_psd
            psd_11 = self.dd_psd
            n_frequencies = self.f_psd.size
        else:
            psd_00 = self.ii_psd
            psd_01 = self.iq_psd
            psd_11 = self.qq_psd
            n_frequencies = self.f_psd.size

        if noise_type in ["pd", "iq"]:
            # compute square root of covariance
            c = np.array([[psd_00, psd_01],  # 2 x 2 x n_frequencies
                          [np.conj(psd_01), psd_11]])
            c = np.moveaxis(c, 2, 0)  # n_frequencies x 2 x 2
            u, s, vh = np.linalg.svd(c)
            s = np.array([[s[:, 0], np.zeros(s[:, 0].shape)],
                          [np.zeros(s[:, 0].shape), s[:, 1]]])
            s = np.moveaxis(s, -1, 0)
            a = u @ np.sqrt(self._n_samples * s / (2 * dt)) @ vh  # divide by 2 for single sided noise
            # get unit dissipation random phase noise in both quadratures
            phase_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            phase_fft = np.exp(1j * phase_phi)
            amp_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            amp_fft = np.exp(1j * amp_phi)
            # rescale the noise to the covariance
            noise_fft = np.array([[phase_fft],  # 2 x 1 x n_traces x n_frequencies
                                  [amp_fft]])
            noise_fft = np.moveaxis(noise_fft, [0, 1], [-2, -1])  # n_traces x n_frequencies x 2 x 1
            noise_fft = (a @ noise_fft).squeeze()  # n_traces x n_frequencies x 2
            noise_fft = np.moveaxis(noise_fft, -1, 0)  # 2 x n_traces x n_frequencies
        else:
            # compute square root of covariance
            psd = psd_00 if noise_type in ["p", "i"] else psd_11
            a = np.sqrt(self._n_samples * psd / (2 * dt))
            # get unit dissipation random phase noise
            noise_phi = 2 * np.pi * np.random.rand(n_traces, n_frequencies)
            noise_fft = np.exp(1j * noise_phi)  # n_traces x n_frequencies
            # rescale the noise to the covariance
            noise_fft = a * noise_fft
        noise = np.fft.irfft(noise_fft, self._n_samples)
        return noise

    def plot_psd(self, noise_type="iq", x_label=None, y_label=None, label_kwargs=None, legend=True, legend_kwargs=None,
                 title=False, title_kwargs=None, tick_kwargs=None, tighten=True, axes=None):
        """
        Plot the power spectral density of the trace data.
        Args:
            noise_type: string
                Either "iq" or "pd" for I and Q data or phase and dissipation
                data.
            x_label: string
                The label for the x axis. The default is None which uses the
                default label. If x_label evaluates to False, parameter_kwargs
                is ignored.
            y_label: string
                The label for the y axis. The default is None which uses the
                default label. If y_label evaluates to False, parameter_kwargs
                is ignored.
            label_kwargs: dictionary
                Keyword arguments for the axes labels in axes.set_*label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            legend: boolean
                Determines whether the legend is used or not. The default is
                True. If False, legend_kwargs is ignored.
            legend_kwargs: dictionary
                Keyword arguments for the legend in axes.legend(). The default
                is None which uses default options. Keywords in this
                dictionary override the default options.
            title: boolean or string
                If it is a boolean, it determines whether or not to add the
                default title. If it is a string, that string is used as the
                title. If False, title_kwargs is ignored. The default is False.
            title_kwargs: dictionary
                Keyword arguments for the axes title in axes.set_title(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            tick_kwargs: dictionary
                Keyword arguments for the ticks using axes.tick_params(). The
                default is None which uses the default options. Keywords in
                this dictionary override the default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            axes: matplotlib.axes.Axes class
                An Axes class on which to put the plot. The default is None and
                a new figure is made.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted noise.
        """
        if noise_type.lower() not in ["iq", "pd"]:
            raise ValueError("Noise type must be one of 'iq' or 'pd'.")
        iq = (noise_type.lower() == "iq")
        _, axes = setup_axes(axes, x_label, y_label, label_kwargs, 'frequency  [Hz]',
                             'PSD [V² / Hz]' if iq else 'PSD [dBc / Hz]')
        psd11 = self.ii_psd if iq else 10 * np.log10(self.pp_psd)
        psd22 = self.qq_psd if iq else 10 * np.log10(self.dd_psd)

        axes.step(self.f_psd[1:-1], psd11[1:-1], where='mid', label="I" if iq else "phase", color="C0")
        axes.step(self.f_psd[1:-1], psd22[1:-1], where='mid', label="Q" if iq else "dissipation", color="C1")

        axes.set_xlim(self.f_psd[1:-1].min(), self.f_psd[1:-1].max())
        axes.set_xscale('log')
        if iq:
            axes.set_yscale('log')
        s = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
        title = s.format(self.loop.power, self.loop.field, self.loop.temperature * 1000) if title is True else title
        finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def _set_directory(self, directory):
        self._directory = directory
        if isinstance(self._npz, str):
            self._npz = os.path.join(self._directory, os.path.basename(self._npz))
        try:
            self.loop._directory = self._directory
        except AttributeError:
            pass
