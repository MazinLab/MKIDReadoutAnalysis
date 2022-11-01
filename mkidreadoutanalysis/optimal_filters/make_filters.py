import logging
import numpy as np
import scipy as sp
import pkg_resources as pkg
from astropy.stats import mad_std
from skimage.restoration import unwrap_phase
from matplotlib import ticker

import mkidcore.config

# Flags for the optimal filter routine
flags = {'not_started': 0,  # calculation has not been started.
         'pulses_computed': 1,  # finished finding the pulses.
         'noise_computed': 2,  # finished the noise calculation.
         'template_computed': 4,  # finished the template calculation.
         'filter_computed': 8,  # finished the filter calculation.
         'bad_pulses': 16,  # not enough pulses found satisfying the configuration conditions.
         'bad_noise': 32,  # noise calculation failed. Assuming white noise.
         'bad_template': 64,  # template calculation failed. Using the fallback template.
         'bad_template_fit': 128,  # the template fit failed. Using the raw data.
         'bad_filter': 256}  # filter calculation failed. Using the template as a filter.
from . import utils
from . import filters as filter_functions
from . import templates as template_functions

log = logging.getLogger(__name__)


# log.addHandler(logging.NullHandler())
#
# DEFAULT_SAVE_NAME = "filter_solution.p"
# TIMEOUT = 0.001


class Calculator:
    """
    Class for manipulating a resonator's phase time-stream and computing
    filters.

    Args:
        stream: string, mkidcore.objects.TimeStream
            The name for the file containing the phase time-stream or a
            TimeStream object.
        config: yaml config object (optional)
            The filter configuration object for the calculation loaded by
            mkidcore.config.load(). If not supplied, the default configuration
            will be used.
        fallback_template: numpy.ndarray (optional)
            A 1D numpy array of size config.pulses.ntemplate with the correct
            config.pulses.offset that will be used for the time stream template
            if it cannot be computed from the phase time-stream. If
            supplied, the template is not updated if the config changes.
        name: any (optional)
            An object used to identify the time stream. It is not used directly
            by this class except for when printing log messages.
    """

    def __init__(self, phase, noise_data=None, config=None, fallback_template=None, name=None):
        self._cfg = config if config is not None else None
        self._user_input_phase = phase
        self._user_input_noise_data = noise_data
        if fallback_template is not None:
            utils.check_template(self.cfg.pulses, fallback_template)
            self._fallback_template = fallback_template
            self._reload_fallback = False
        else:
            self._fallback_template = None
            self._reload_fallback = True

        self.name = name

        self.phase = None
        self.noise_data = None

        self._init_result()

    def __getstate__(self):
        self.clear_file_properties()
        return self.__dict__

    @property
    def phase(self):
        """The phase time-stream of the resonator."""
        if self._phase is None:
            self._phase = self._user_input_phase
            # unwrap the phase
            if self.cfg.pulses.unwrap:
                self._phase = unwrap_phase(self._phase)  # much faster than np.unwrap
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def noise_data(self):
        """The phase time-stream of the resonator with no photon source."""
        if self._user_input_noise_data is not None:
            self._noise_data = self._user_input_noise_data
        return self._noise_data

    @noise_data.setter
    def noise_data(self, value):
        self._noise_data = value

    @property
    def cfg(self):
        """
        The filter computation configuration. Resetting it will clear parts of
        the filter result that are not consistent with the new settings.
        """
        if self._cfg is None:
            self._cfg = mkidcore.config.load(pkg.resource_filename(__name__, 'filter.yml')).filters
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        # overload pulses, noise, template & filter if the pulse finding configuration changed
        if any([getattr(self.cfg.pulses, key) != item for key, item in value.pulses.items()]):
            self.clear_results()
            if self._reload_fallback:  # if the fallback wasn't supplied on the init, reset it
                self._fallback_template = None
        # overload noise, template & filter the results if the noise configuration changed
        if any([getattr(self.cfg.noise, key) != item for key, item in value.noise.items()]):
            self.clear_noise()
            self.clear_template()
            self.clear_filter()
        # overload template & filter if the template configuration changed
        if any([getattr(self.cfg.template, key) != item for key, item in value.template.items()]):
            self.clear_template()
            self.clear_filter()
        # overload filter if the filter configuration changed
        if any([getattr(self.cfg.filter, key) != item for key, item in value.filter.items()]):
            self.clear_filter()
        # set the config
        self._cfg = value

    @property
    def fallback_template(self):
        if self._fallback_template is None:
            self._fallback_template = utils.load_fallback_template(self.cfg.pulses)
        return self._fallback_template

    @fallback_template.setter
    def fallback_template(self, value):
        self._fallback_template = value

    @property
    def characteristic_time(self):
        """Approximate time constant of the response in units of dt."""
        if self.result['template'] is not None:
            return -np.trapz(self.result['template'])
        else:
            return -np.trapz(self.fallback_template)

    def clear_file_properties(self):
        """Free up memory by removing properties that can be reloaded from files."""
        self.phase = None

    def clear_results(self):
        """Delete computed results from the time stream."""
        self._init_result()
        self.phase = None  # technically computed since it is unwrapped
        log.debug("Calculator {}: results reset.".format(self.name))

    def clear_noise(self):
        """Delete computed noise from the time stream."""
        self.result["psd"] = None
        log.debug("Calculator {}: noise reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_noise"]:
            self.result["flag"] ^= flags["bad_noise"]
            log.debug("Calculator {}: noise problem flag reset.".format(self.name))
        if self.result["flag"] & flags["noise_computed"]:
            self.result["flag"] ^= flags["noise_computed"]
            log.debug("Calculator {}: noise status flag reset.".format(self.name))

    def clear_template(self):
        """Delete computed template from the time stream."""
        self.result["template"] = None
        self.result["template_fit"] = None
        log.debug("Calculator {}: template reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_template"]:
            self.result["flag"] ^= flags["bad_template"]
            log.debug("Calculator {}: template problem flag reset.".format(self.name))
        if self.result["flag"] & flags["bad_template_fit"]:
            self.result["flag"] ^= flags["bad_template_fit"]
            log.debug("Calculator {}: template fit problem flag reset.".format(self.name))
        if self.result["flag"] & flags["template_computed"]:
            self.result["flag"] ^= flags["template_computed"]
            log.debug("Calculator {}: template status flag reset.".format(self.name))

    def clear_filter(self):
        """Delete computed filter from the time stream."""
        self.result["filter"] = None
        log.debug("Calculator {}: filter reset.".format(self.name))
        # if the filter was flagged reset the flag bitmask
        if self.result["flag"] & flags["bad_filter"]:
            self.result["flag"] ^= flags["bad_filter"]
            log.debug("Calculator {}: filter problem flag reset.".format(self.name))
        if self.result["flag"] & flags["filter_computed"]:
            self.result["flag"] ^= flags["filter_computed"]
            log.debug("Calculator {}: filter status flag reset.".format(self.name))

    def make_pulses(self, save=True, use_filter=False):
        """
        Find the pulse index locations in the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            use_filter: boolean, numpy.ndarray (optional)
                Use a pre-computed filter to find the pulse indices. If True,
                the filter from the result attribute is used. If False, the
                fallback template is used to make a filter. Otherwise,
                use_filter is assumed to be a pre-computed filter and is used.
                Any precomputed filter should have the same offset and size as
                specified in the configuration to get the right peak indices.

        Returns:
            pulses: numpy.ndarray
                An array of indices corresponding to the pulse peak locations.
            mask: numpy.ndarray
                A boolean mask that filters out bad pulses from the pulse
                indices.
        """
        if self.result['flag'] & flags['pulses_computed'] and save:
            return
        cfg = self.cfg.pulses

        # make a filter to use on the time stream (ignore DC component)
        if use_filter is True:
            filter_ = self.result["filter"]
        elif use_filter is not False:
            filter_ = use_filter
        else:
            filter_ = filter_functions.matched(self.fallback_template, nfilter=cfg.ntemplate, dc=True)

        # find pulse indices
        _, pulses = self.compute_responses(cfg.threshold, filter_=filter_)

        # correct for correlation offset and save results
        pulses += cfg.offset - filter_.size // 2
        mask = np.ones_like(pulses, dtype=bool)

        # mask bad pulses
        diff = np.diff(np.insert(np.append(pulses, self.phase.size + 1), 0, 0))  # assume pulses are at the ends
        bad_previous = (diff < cfg.separation)[:-1]  # far from previous previous pulse (remove last)
        bad_next = (diff < cfg.ntemplate - cfg.offset)[1:]  # far from next pulse  (remove first)
        end = (pulses + cfg.ntemplate - cfg.offset - 1 >= self.phase.size)  # close to the end
        beginning = (pulses - cfg.offset < 0)  # close to the beginning
        mask[bad_next | bad_previous | end | beginning] = False

        # save and return the pulse indices and mask
        if save:
            self.result["pulses"] = pulses
            self.result["mask"] = mask

            # set flags
            self.result['flag'] |= flags['pulses_computed']
            if self.result["mask"].sum() < cfg.min_pulses:  # not enough good pulses to make a reliable template
                self.result['template'] = self.fallback_template
                self.result['flag'] |= flags['bad_pulses'] | flags['bad_template'] | flags['template_computed']
        return pulses, mask

    def make_noise(self, save=True):
        """
        Make the noise spectrum for the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.

        Returns:
            psd: numpy.ndarray
                The computed power spectral density.
        """
        if self.result['flag'] & flags['noise_computed'] and save:
            return
        self._flag_checks(pulses=True)
        cfg = self.cfg.noise
        pulse_cfg = self.cfg.pulses

        if self.noise_data is not None:
            psd = sp.signal.welch(self.noise_data, fs=1. / self.cfg.dt, nperseg=cfg.nwindow, detrend="constant", return_onesided=True, scaling="density")[1]
            n_psd = 1
        else:
            # add pulses to the ends so that the bounds are treated correctly
            pulses = np.insert(np.append(self.result["pulses"], self.phase.size + 1), 0, 0)

            # loop space between peaks and compute noise
            n_psd = 0
            windows_used = 0
            psd = np.zeros(int(cfg.nwindow / 2. + 1))
            for peak1, peak2 in zip(pulses[:-1], pulses[1:]):
                if windows_used > cfg.max_windows:
                    break  # no more noise is needed
                if peak2 - peak1 < cfg.isolation + pulse_cfg.offset + cfg.nwindow:
                    continue  # not enough space between peaks
                data = self.phase[peak1 + cfg.isolation: peak2 - pulse_cfg.offset]
                psd += sp.signal.welch(data-data.mean(), fs=1. / self.cfg.dt, nperseg=cfg.nwindow, detrend="constant",
                                   return_onesided=True, scaling="density")[1]
                n_psd += 1
                windows_used += data.size // cfg.nwindow
        # finish the average, assume white noise if there was no data
            if n_psd == 0:
                psd[:] = 1.
            else:
                psd /= n_psd

        if save:
            # set flags and results
            self.result['psd'] = psd
            if n_psd == 0:
                self.result['flag'] |= flags['bad_noise']
            self.result['flag'] |= flags['noise_computed']
        return psd

    def make_template(self, save=True, refilter=True, pulses=None, mask=None, phase=None):
        """
        Make the template for the photon pulse.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            refilter: boolean (optional)
                Recompute the pulse indices and remake the template with a
                filter computed from the data.
            pulses: numpy.ndarray (optional)
                A list of pulse indices corresponding to phase to use in
                computing the template. The default is to use the value in
                the result.
            mask: numpy.ndarray (optional)
                A boolean array that picks out good indices from pulses.
            phase: numpy.ndarray (optional)
                The phase data to use to compute the template. The default
                is to use the median subtracted phase time-stream.

        Returns:
            template: numpy.ndarray
                A template for the pulse shape.
            template_fit: lmfit.ModelResult, None
                The template fit result. If the template was not fit, None is
                returned.
        """
        if self.result['flag'] & flags['template_computed'] and save:
            return
        self._flag_checks(pulses=True, noise=True)
        cfg = self.cfg.template
        if pulses is None:
            pulses = self.result['pulses']
        if mask is None:
            mask = self.result['mask']
        if phase is None:
            phase = self.phase - np.median(self.phase)

        # compute the template
        template = self._average_pulses(phase, pulses, mask)

        # shift and normalize template
        template = self._shift_and_normalize(template)

        # fit the template
        if cfg.fit is not False:
            template, fit_result = self._fit_template(template)
        else:
            fit_result = None

        # filter again to get the best possible pulse locations
        if self._good_template(template) and refilter:
            filter_ = filter_functions.dc_orthogonal(template, self.result['psd'], self.cfg.noise.nwindow,
                                                     cutoff=cfg.cutoff)
            pulses, mask = self.make_pulses(save=False, use_filter=filter_)
            if mask.any():
                template, fit_result = self.make_template(save=False, refilter=False, pulses=pulses, mask=mask,
                                                          phase=phase)

        # save and return the template
        if save:
            if self._good_template(template):
                self.result['template'] = template
            else:
                self.result['flag'] |= flags['bad_template']
                self.result['template'] = self.fallback_template
            self.result['template_fit'] = fit_result
            if cfg.fit is not False and (fit_result is None or fit_result.success is False):
                self.result['flag'] |= flags['bad_template_fit']
            self.result['flag'] |= flags['template_computed']
        return template, fit_result

    def make_filter(self, save=True, filter_type=None):
        """
        Make the filter for the time stream.

        Args:
            save: boolean (optional)
                Save the result in the result attribute when done. If save is
                False, none of the flags or results will change.
            filter_type: string (optional)
                Create a filter of this type. The default is to use the value
                in the configuration file. Valid filters are function names in
                filters.py.

        Returns:
            filter_: numpy.ndarray
                The computed filter.
        """
        if self.result['flag'] & flags['filter_computed'] and save:
            return
        self._flag_checks(pulses=True, noise=True, template=True)
        cfg = self.cfg.filter
        if filter_type is None:
            filter_type = cfg.filter_type

        # compute filter (some options may be ignored by the filter function)
        if filter_type not in filter_functions.__all__:
            raise ValueError("{} must be one of {}".format(filter_type, filter_functions.__all__))
        filter_ = getattr(filter_functions, filter_type)(
            self.result["template"],
            self.result["psd"],
            self.cfg.noise.nwindow,
            nfilter=cfg.nfilter,
            cutoff=self.cfg.template.cutoff,
            normalize=cfg.normalize)

        # save and return the filter
        if save:
            self.result['filter'] = filter_
            self.result['flag'] |= flags['filter_computed']
        return filter_

    def apply_filter(self, filter_=None, positive=False):
        """
        Apply a filter to the time-stream.

        Args:
            filter_: numpy.ndarray (optional)
                The filter to apply. If not provided, the filter from the saved
                result is used.
            positive: boolean (optional)
                If True, the filter will be negated so that the filtered pulses
                have a positive amplitude.

        Returns:
            filtered_phase: numpy.ndarray
                The filtered phase time-stream.
        """
        if filter_ is None:
            filter_ = self.result["filter"]
        if positive:
            filter_ = -filter_  # "-" so that pulses are positive
        filtered_phase = sp.signal.convolve(self.phase, filter_, mode='same')
        return filtered_phase

    def compute_responses(self, threshold, filter_=None, no_filter=False):
        """
        Computes the pulse responses in the time stream.

        Args:
            threshold: float
                Only pulses above threshold sigma in the filtered time stream
                are included.
            filter_: numpy.ndarray (optional)
                A filter to use to calculate the responses. If None, the filter
                from the result attribute is used.
            no_filter: boolean (optional)
                If true, don't apply the filter to the trace. Useful for
                pre-filtered time streams.
        Returns:
            responses: numpy.ndarray
                The response values of each pulse found in the time stream.
            pulses: numpy.ndarray
                The indices of each pulse found in the time stream.
        """
        if threshold is None:
            threshold = self.cfg.pulses.threshold
        filtered_phase = -self.phase if no_filter else self.apply_filter(filter_=filter_, positive=True)
        sigma = mad_std(filtered_phase)
        pulses, _ = sp.signal.find_peaks(filtered_phase, height=threshold * sigma,
                                         distance=10 * self.characteristic_time)
        responses = -filtered_phase[pulses]
        return responses, pulses

    def plot(self, tighten=True, blit=False, filter_=None, flag=None, axes_list=None):
        """
        Plot the results of the calculation.

        Args:
            tighten: boolean (optional)
                If true, figure.tight_layout() is called after the plot is
                generated.
            blit: boolean (optional)
                Set to True if supplying axes and rewriting the data.
            filter_: boolean (optional)
                Use with flag to plot a different filter than saved in the
                result.
            flag: boolean (optional)
                Use with filter_ to plot a different filter than saved in the
                result.
            axes_list: iterable of matplotlib.axes.Axes (optional)
                The axes on which to plot. If not provided, they are generated
                using pyplot.

        Returns:
            axes_list: iterable of matplotlib.axes.Axes
                The axes with the plotted results.
        """
        if axes_list is None:
            from matplotlib import pyplot as plt
            figure, axes_list = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
            axes_list = axes_list.flatten()
        else:
            figure = axes_list[0].figure

        self.plot_noise(axes=axes_list[0], blit=blit, tighten=False)
        self.plot_filter(axes=axes_list[1], blit=blit, filter_=filter_, flag=flag, tighten=False)
        self.plot_template(axes=axes_list[2], blit=blit, tighten=False)
        if len(axes_list) > 3:
            self.plot_template_fit(axes=axes_list[3], blit=blit, tighten=False)

        if tighten and not blit:
            figure.tight_layout()
        return axes_list

    def plot_noise(self, tighten=True, blit=False, axes=None):
        """
        Plot the results of the noise calculation.

        Args:
            tighten: boolean (optional)
                If true, figure.tight_layout() is called after the plot is
                generated.
            blit: boolean (optional)
                Set to True if supplying axes and rewriting the data.
            axes:  matplotlib.axes.Axes (optional)
                The axes on which to plot. If not provided, they are generated
                using pyplot.

        Returns:
            axes: matplotlib.axes.Axes
                An Axes class with the plotted noise.
        """
        if not blit:
            figure, axes = utils.init_plot(axes)
            axes.set_xlabel("frequency [Hz]")
            axes.set_ylabel("PSD [dBc / Hz]")
        if self.result['psd'] is not None:
            psd = self.result['psd']
            if not blit:
                f = np.fft.rfftfreq(self.cfg.noise.nwindow, d=self.cfg.dt)
                axes.semilogx(f, 10 * np.log10(psd))
            else:
                line = axes.get_lines()[0]
                line.set_ydata(10 * np.log10(psd))
            if self.result["flag"] & flags["bad_noise"]:
                axes.set_title("failed: using white noise", color='red')
            else:
                axes.set_title("successful", color='green')
        else:
            if blit:
                line = axes.get_lines()[0]
                line.remove()
                del line
            axes.set_title("noise not computed", color='red')
        utils.finish_plot(axes, tighten=tighten, blit=blit)
        return axes

    def plot_template(self, tighten=True, blit=False, axes=None):
        """
        Plot the results of the template calculation.

        Args:
            tighten: boolean (optional)
                If true, figure.tight_layout() is called after the plot is
                generated.
            blit: boolean (optional)
                Set to True if supplying axes and rewriting the data.
            axes:  matplotlib.axes.Axes (optional)
                The axes on which to plot. If not provided, they are generated
                using pyplot.

        Returns:
            axes: matplotlib.axes.Axes
                An Axes class with the plotted template.
        """
        if not blit:
            figure, axes = utils.init_plot(axes)
            axes.set_xlabel(r"time [$\mu$s]")
            axes.set_ylabel("template [arb.]")
        if self.result['template'] is not None:
            template = self.result['template']
            if not blit:
                t = np.arange(template.size) * self.cfg.dt * 1e6
                axes.plot(t, template)
            else:
                line = axes.get_lines()[0]
                line.set_ydata(template)
            if self.result["flag"] & flags["bad_template"]:
                axes.set_title("failed: using fallback template", color='red')
            else:
                axes.set_title("successful", color='green')
        else:
            if blit:
                line = axes.get_lines()[0]
                line.remove()
                del line
            axes.set_title("template not computed", color='red')
        utils.finish_plot(axes, tighten=tighten, blit=blit)
        return axes

    def plot_template_fit(self, tighten=True, blit=False, axes=None):
        """
        Plot the results of the template fit.

        Args:
            tighten: boolean (optional)
                If true, figure.tight_layout() is called after the plot is
                generated.
            blit: boolean (optional)
                Set to True if supplying axes and rewriting the data.
            axes:  matplotlib.axes.Axes (optional)
                The axes on which to plot. If not provided, they are generated
                using pyplot.

        Returns:
            axes: matplotlib.axes.Axes
                An Axes class with the plotted template fit.
        """
        if blit:
            axes.clear()
        else:
            figure, axes = utils.init_plot(axes)

        if self.result['template_fit'] is not None:
            fit = self.result['template_fit']
            formatter = ticker.FuncFormatter(lambda x, y: "{:g}".format(x * self.cfg.dt * 1e6))
            axes.xaxis.set_major_formatter(formatter)
            fit.plot_fit(ax=axes, show_init=True, xlabel=r"time [$\mu$s]", ylabel="template [arb.]")
            if self.result["flag"] & flags["bad_template_fit"]:
                axes.set_title("failed: using data", color='red')
            else:
                axes.set_title("successful", color='green')
        else:
            axes.set_xlabel(r"time [$\mu$s]")
            axes.set_ylabel("template [arb.]")
            axes.set_title("template not fit", color='red')
        utils.finish_plot(axes, tighten=tighten)
        return axes

    def plot_filter(self, tighten=True, blit=False, filter_=None, flag=None, axes=None):
        """
        Plot the results of the filter calculation.

        Args:
            tighten: boolean (optional)
                If true, figure.tight_layout() is called after the plot is
                generated.
            blit: boolean (optional)
                Set to True if supplying axes and rewriting the data.
            filter_: boolean (optional)
                Use with flag to plot a different filter than saved in the
                result.
            flag: boolean (optional)
                Use with filter_ to plot a different filter than saved in the
                result.
            axes:  matplotlib.axes.Axes (optional)
                The axes on which to plot. If not provided, they are generated
                using pyplot.

        Returns:
            axes: matplotlib.axes.Axes
                An Axes class with the plotted filter.
        """
        if flag is None:
            flag = self.result['flag']
        if filter_ is None:
            filter_ = self.result['filter']
        if not blit:
            figure, axes = utils.init_plot(axes)
            axes.set_xlabel(r"time [$\mu$s]")
            axes.set_ylabel("filter coefficient [radians]")
        if filter_ is not None:
            if not blit:
                t = np.arange(filter_.size) * self.cfg.dt * 1e6
                axes.plot(t, filter_)
            else:
                line = axes.get_lines()[0]
                line.set_ydata(filter_)
            if flag & flags["bad_filter"]:
                axes.set_title("failed", color='red')
            else:
                axes.set_title("successful", color='green')
        else:
            if blit:
                line = axes.get_lines()[0]
                line.remove()
                del line
            axes.set_title("filter not computed", color='red')
        utils.finish_plot(axes, tighten=tighten, blit=blit)
        return axes

    def _init_result(self):
        self.result = {"pulses": None, "mask": None, "template": None, "template_fit": None, "filter": None,
                       "psd": None, "flag": flags["not_started"]}

    def _flag_checks(self, pulses=False, noise=False, template=False):
        if pulses:
            assert self.result['flag'] & flags['pulses_computed'], "run self.make_pulses() first."
        if noise:
            assert self.result['flag'] & flags['noise_computed'], "run self.make_noise() first."
        if template:
            assert self.result['flag'] & flags['template_computed'], "run self.make_template first."

    def _average_pulses(self, phase, indices, mask):
        # get parameters
        offset = self.cfg.pulses.offset
        ntemplate = self.cfg.pulses.ntemplate
        percent = self.cfg.template.percent

        # make a pulse array
        index_array = (indices[mask] + np.arange(-offset, ntemplate - offset)[:, np.newaxis]).T
        pulses = phase[index_array]

        # weight the pulses by pulse height and remove those that are outside the middle percent of the data
        pulse_heights = np.abs(np.min(pulses, axis=1))
        percentiles = np.percentile(pulse_heights, [(100. - percent) / 2., (100. + percent) / 2.])
        logic = (pulse_heights <= percentiles[1]) & (percentiles[0] <= pulse_heights)
        weights = np.zeros_like(pulse_heights)
        weights[logic] = pulse_heights[logic]

        # compute the template
        template = np.sum(pulses * weights[:, np.newaxis], axis=0)
        return template

    def _shift_and_normalize(self, template):
        template = template.copy()
        if template.min() != 0:  # all weights could be zero
            template /= np.abs(template.min())  # correct over all template height

        # shift template (max may not be exactly at offset due to filtering and imperfect default template)
        start = 10 + np.argmin(template) - self.cfg.pulses.offset
        stop = start + self.cfg.pulses.ntemplate
        template = np.pad(template, 10, mode='wrap')[start:stop]  # use wrap to not change the frequency content
        return template

    def _fit_template(self, template):
        if self.cfg.template.fit not in template_functions.__all__:
            raise ValueError("{} must be one of {}".format(cfg.fit, template_functions.__all__))
        model = template_functions.Model(self.cfg.template.fit)
        guess = model.guess(template)
        t = np.arange(template.size)
        result = model.fit(template, guess, t=t)
        # get the template data vector
        template_fit = result.eval(t=t)
        good_peak_height = -1.2 < template_fit.min() < -0.8
        peak = np.argmin(template_fit)
        # ensure the template is properly shifted and normalized
        template_fit = result.eval(t=t - peak + self.cfg.pulses.offset)
        if template_fit.min() != 0:
            template_fit /= np.abs(np.min(template_fit))
        # only modify template if it was a good fit
        success = False
        if result.success and result.errorbars and good_peak_height and self._good_template(template_fit):
            template = template_fit
            success = True
        result.success = success
        return template, result

    def _good_template(self, template):
        tau = np.trapz(template / template.min())  # normalize because fit may not be
        return self.cfg.template.min_tau < tau < self.cfg.template.max_tau

    def calculate(self, clear=True):
        self.make_pulses()
        self.make_noise()
        self.make_template()
        self.make_filter()
        if clear:
            self.clear_file_properties()

#
#     def report(self, filter_type, return_string=False):
#         """
#         Print a summary report of the filter calculation.
#
#         Args:
#             filter_type: string
#                 The filter type for which to plot the summary.
#             return_string: boolean (optional)
#                 If True, the report isn't printed and is returned as a string.
#         """
#         filter_flags = self.flags[filter_type]
#         default_template_as_filter = 0
#         default_template_and_noise_as_filter = 0
#         calculated_template_as_filter = 0
#         calculated_template_and_noise_as_filter = 0
#         bad_template_fits = 0
#         low_counts = 0
#         n_filters = 0
#         for flag in filter_flags:
#             if flag & flags['filter_computed']:
#                 n_filters += 1
#                 bad_pulses = flag & flags['bad_pulses']
#                 bad_noise = flag & flags['bad_noise']
#                 bad_template = flag & flags['bad_template']
#                 bad_template_fit = flag & flags['bad_template_fit']
#                 if bad_template and bad_noise:
#                     default_template_as_filter += 1
#                 elif bad_template and not bad_noise:
#                     default_template_and_noise_as_filter += 1
#                 elif not bad_template and bad_noise:
#                 elif not bad_template and bad_noise:
#                     calculated_template_as_filter += 1
#                 else:
#                     calculated_template_and_noise_as_filter += 1
#                 if bad_template_fit:
#                     bad_template_fits += 1
#                 if bad_pulses:
#                     low_counts += 1
#         percent_optimal = calculated_template_and_noise_as_filter / max(1, n_filters) * 100
#         percent_no_template = default_template_and_noise_as_filter / max(1, n_filters) * 100
#         percent_no_noise = calculated_template_as_filter / max(1, n_filters) * 100
#         percent_default = default_template_as_filter / max(1, n_filters) * 100
#         percent_bad_fits = bad_template_fits / max(1, n_filters) * 100
#         percent_low_counts = low_counts / max(1, n_filters) * 100
#         s = ("{:.2f}% of pixels are using optimal filters \n".format(percent_optimal) +
#              "{:.2f}% of pixels are using the default template with real noise as the filter \n"
#              .format(percent_no_template) +
#              "{:.2f}% of pixels are using the calculated template with white noise as the filter \n"
#              .format(percent_no_noise) +
#              "{:.2f}% of pixels are using the default template as the filter \n".format(percent_default) +
#              "--------------------------------------------------------------------------------\n" +
#              "{:.2f}% of pixels didn't have enough photons to make a template\n".format(percent_low_counts) +
#              "{:.2f}% of pixels had bad template fits".format(percent_bad_fits))
#         if return_string:
#             return s
#         else:
#             print(s)
#
#     def _collect_data(self):
#         filter_array = np.empty((self.res_ids.size, self.cfg.filters.filter.nfilter))
#         for index, calculator in enumerate(self.calculators):
#             filter_array[index, :] = calculator.result["filter"]
#         self.filters.update({self.cfg.filters.filter.filter_type: filter_array})
#         self.flags.update({self.cfg.filters.filter.filter_type: [c.result["flag"] for c in self.calculators]})
#
