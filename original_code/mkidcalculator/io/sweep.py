import os
import logging
import numpy as np
from operator import itemgetter
from matplotlib import gridspec
from collections.abc import Collection

from mkidcalculator.io.loop import Loop
from mkidcalculator.io.resonator import Resonator
from mkidcalculator.plotting import plot_parameter_vs_f, plot_parameter_hist
from mkidcalculator.io.data import (analogreadout_sweep,
                                    labview_segmented_widesweep)
from mkidcalculator.io.utils import (find_resonators, collect_resonances,
                                     _loop_fit_data, dump, load)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep:
    """A class for organizing data from multiple resonators."""
    def __init__(self):
        self.name = None
        self.resonators = []

    @property  # @property so that self.f not put into memory on load
    def f_centers(self):
        """
        A list of the median frequencies for each resonator corresponding to
        sweep.resonators. This is a useful rough proxy for the resonant
        frequencies that depends only on the data and not the fit.
        """
        return [resonator.f_center for resonator in self.resonators]

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # _set_directory so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        dump(self, file_name)
        log.info("saved sweep as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Sweep class from the pickle file 'file_name'."""
        sweep = load(file_name)
        if not isinstance(sweep, cls):
            raise ValueError(f"'{file_name}' does not contain a Sweep class.")
        log.info("loaded sweep from '{}'".format(file_name))
        return sweep

    def add_resonators(self, resonators, sort=True):
        """
        Add Resonator objects to the sweep.
        Args:
            resonators: Resonator class or iterable of Resonator classes
                The resonators that are to be added to the Sweep.
            sort: boolean (optional)
                Sort the resonator list by its median frequency.
                The default is True. If False, the order of the resonator list
                is preserved.
        """
        if isinstance(resonators, Resonator):
            resonators = [resonators]
        # append resonator data
        for resonator in resonators:
            resonator.sweep = self
            self.resonators.append(resonator)
        # sort
        if sort and self.resonators:
            self.resonators = [
                r for _, r in sorted(zip(self.f_centers, self.resonators),
                                     key=itemgetter(0,))]

    def remove_resonators(self, indices):
        """
        Remove resonators from the sweep.
        Args:
            indices: integer or iterable of integers
                The indices in sweep.resonators that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.resonators.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Resonator objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for resonator in self.resonators:
            resonator.free_memory(directory=directory)

    @classmethod
    def from_widesweep(cls, sweep_file_name, df, sort=True,
                       data=labview_segmented_widesweep,
                       indices=find_resonators, indices_kwargs=None,
                       loop_kwargs=None, **kwargs):
        """
        Sweep class factory method that returns a Sweep() from widesweep data
        (continuous data in which the resonator locations need to be
        identified).
        Args:
            sweep_file_name: string
                The file name for the widesweep data.
            df: float
                The frequency bandwidth for each resonator in the units of the
                data in the file.
            sort: boolean (optional)
                Sort the loop data in each resonator by its power, field, and
                temperature. Also sort noise data and pulse data lists for each
                loop by their bias frequencies. The resonator list will be
                sorted by the median frequency of its loops. The default is
                True. If False the input order is preserved.

                Note
                    Sorting requires loading data and computing medians. The
                    process could be slow for very large datasets. In this case
                    set this keyword argument to False.
            data: object (optional)
                Function whose return value is a tuple of the frequencies,
                complex scattering data, attenuation, field, and temperature of
                the widesweep. The dimensions of the returned data arrays
                should conform to the shape (K temperatures, L fields,
                M attenuations, N frequencies), where singleton dimensions to
                the left can be left out.
            indices: iterable of integers, function, or string (optional)
                If an iterable, indices is interpreted as starting peak
                frequency locations from the values returned by data. If a
                function, it must return an iterable of resonator peak indices
                corresponding to the data returned by data. The mandatory input
                arguments are f, z. If a string, the indices are unpickled.
            indices_kwargs: dictionary (optional)
                Extra keyword arguments to pass to the indices function. The
                default is None.
            loop_kwargs: dictionary (optional)
                Extra keyword arguments to pass to loop.from_python().
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            sweep: object
                A Sweep() object containing the loaded data.
        """
        # get the data and add in missing dimensions
        f, z, attenuation, field, temperature = data(sweep_file_name, **kwargs)

        # get the indices
        if callable(indices):
            kws = {"df": df} if indices is find_resonators else {}
            if indices_kwargs is not None:
                kws.update(indices_kwargs)
            peaks = indices(f, z, **kws)
        elif isinstance(indices, str):
            peaks = load(indices)
        else:
            peaks = indices

        # collect the data into (temp, field, atten, resonator, data) shape
        f_array, z_array = collect_resonances(f, z, peaks, df)

        # make the resonator objects
        kws = {}
        if loop_kwargs is not None:
            kws.update(loop_kwargs)
        resonators = []
        for ri in range(f_array.shape[-2]):
            resonators.append(Resonator())
            for ti, tv in enumerate(np.atleast_1d(temperature)):
                for fi, fv in enumerate(np.atleast_1d(field)):
                    for ai, av in enumerate(np.atleast_1d(attenuation)):
                        # add the loop for each temperature, field, attenuation
                        resonators[-1].add_loops(
                            Loop.from_python(z_array[ti, fi, ai, ri, :],
                                             f_array[ti, fi, ai, ri, :],
                                             av, fv, tv, **kws))
        # create the sweep
        sweep = cls()
        sweep.add_resonators(resonators, sort=sort)
        return sweep

    @classmethod
    def from_file(cls, sweep_file_name, data=analogreadout_sweep, sort=True,
                  **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the resonator
        data loaded.
        Args:
            sweep_file_name: string
                The file name for the sweep data.
            data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to
                Resonator.from_file().
            sort: boolean (optional)
                Sort the loop data in each resonator by its power, field, and
                temperature. Also sort noise data and pulse data lists for each
                loop by their bias frequencies. The resonator list will be
                sorted by the median frequency of its loops. The default is
                True. If False the input order is preserved.

                Note:
                    Sorting requires loading data and computing medians. The
                    process could be slow for very large datasets. In this case
                    set this keyword argument to False.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            sweep: object
                A Sweep() object containing the loaded data.
        """
        sweep = cls()
        res_kwarg_list = data(sweep_file_name, **kwargs)
        resonators = []
        for kws in res_kwarg_list:
            kws.update({"sort": sort})
            resonators.append(Resonator.from_file(**kws))
        sweep.add_resonators(resonators, sort=sort)
        sweep.name = os.path.basename(sweep_file_name) + ", " + str(kwargs)
        return sweep

    def _set_directory(self, directory):
        self._directory = directory
        for resonator in self.resonators:
            resonator._set_directory(self._directory)

    def plot_loop_fits(self, parameters=("chi2",), fit_type="lmfit", fr="f0",
                       bounds=None, errorbars=True, success=True, power=None,
                       field=None, temperature=None, title=True, tighten=True,
                       label='best', plot_kwargs=None, axes_list=None):
        """
        Plot a summary of all the loop fits.
        Args:
            parameters: tuple of strings
                The fit parameters to plot. "chi2" can be used to plot the
                reduced chi squared value. The default is just "chi2".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit" and
                "loopfit". The default is "lmfit".
            fr: string
                The parameter name that corresponds to the resonance frequency.
                The default is "f0" which gives the resonance frequency for the
                mkidcalculator.S21 model. If this parameter is used in the
                parameters list, a histogram and a nearest-neighbor scatter
                plot is shown instead of the usual histogram and scatter plot.
            bounds: tuple of numbers or tuples
                The bounds for the parameters. It must be a tuple of the same
                length as the parameters keyword argument. Each element is
                either an upper bound on the parameter or a tuple,
                e.g. (lower bound, upper bound). Only data points that satisfy
                all of the bounds are plotted. None can be used as a
                placeholder to skip a bound. The default is None and no
                bounds are used. A bound for the fr parameter is used as a
                bound on |∆fr| in MHz but does not act like a filter for
                the other parameters.
            errorbars: boolean
                If errorbars is True, only data from loop fits that could
                compute errorbars on the fit parameters is included. If
                errorbars is False, only data from loop fits that could not
                compute errorbars on the fit parameters is included. The
                default is True. None may be used to enforce no filtering on
                the errorbars. This keyword has no effect if the fit_type is
                "loopfit" since no error bars are computed.
            success: boolean
                If success is True, only data from successful loop fits is
                included. If False, only data from failed loop fits is
                included. The default is True. None may be used
                to enforce no filtering on success. Note: fit success is
                typically a bad indicator on fit quality. It only ever fails
                when something really bad happens.
            power: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of powers to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the powers.
            field: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of fields to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the fields.
            temperature: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of temperatures to plot. A single
                number will cause only that value to be plotted. The default is
                to include all of the temperatures.
            title: string or boolean (optional)
                The title to use for the summary plot. The default is True and
                the default title will be applied. If False, no title is
                applied.
            tighten: boolean (optional)
                Whether or not to apply figure.tight_layout() at the end of the
                plot. The default is True.
            label: string (optional)
                The fit label to use for the plots. The default is 'best'.
            plot_kwargs: dictionary or list of dictionaries (optional)
                A dictionary or list of dictionaries containing plot options.
                If only one is provided, it is used for all of the plots. If a
                list is provided, it must be of the same length as the
                number of plots. No kwargs are passed by default.
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes on which to put the plots. The default
                is None and a new figure is made.
        Returns:
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes with the plotted data.
        """
        # get the data
        loops = []
        for resonator in self.resonators:
            loops += resonator.loops
        parameters = list(parameters)
        if not isinstance(bounds, Collection):
            bounds = [bounds] * len(parameters)
        parameters = [fr] + parameters
        bounds = [None] + list(bounds)
        # replace fr bound with None so we can just set the plot bound
        dfr_bound = None
        for index, bnd in enumerate(bounds):
            if parameters[index] == fr and index != 0 and bnd is not None:
                dfr_bound = bnd if isinstance(bnd, Collection) else (0, bnd)
                bounds[index] = None
        outputs = _loop_fit_data(loops, parameters=parameters,
                                 fit_type=fit_type, label=label, bounds=bounds,
                                 success=success, errorbars=errorbars,
                                 power=power, field=field,
                                 temperature=temperature)
        # create figure if needed
        if axes_list is None:
            from matplotlib import pyplot as plt
            figure = plt.figure(figsize=(8.5, 11 / 5 * (len(parameters) - 1)))
            # setup figure axes
            gs = gridspec.GridSpec(len(parameters) - 1, 2)
            axes_list = np.array([figure.add_subplot(gs_ii) for gs_ii in gs])
        else:
            figure = axes_list[0].figure
        # check plot kwargs
        if plot_kwargs is None:
            plot_kwargs = {}
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [plot_kwargs] * len(axes_list)
        # add plots
        for index in range(len(axes_list) // 2):
            kws = {"x_label": parameters[index + 1] + " [GHz]"
                   if parameters[index + 1] == fr else parameters[index + 1]}
            if plot_kwargs[2 * index]:
                kws.update(plot_kwargs[2 * index])
            output = outputs[index + 1][~np.isinf(outputs[index + 1])]
            plot_parameter_hist(output, axes=axes_list[2 * index], **kws)
            kws = {"y_label": parameters[index + 1], "x_label": fr + " [GHz]",
                   "title": True, "title_kwargs": {"fontsize": "medium"}}
            if index == 0:
                kws.update({"legend": True})
            factor = 1
            if parameters[index + 1] == fr:
                kws.update(
                    {"absolute_delta": True,
                     "y_label": "|∆" + parameters[index + 1] + "| [MHz]"})
                factor = 1e3
            if plot_kwargs[2 * index + 1]:
                kws.update(plot_kwargs[2 * index + 1])
            plot_parameter_vs_f(outputs[index + 1] * factor, outputs[0],
                                axes=axes_list[2 * index + 1], **kws)
            if parameters[index + 1] == fr and dfr_bound is not None:
                axes_list[2 * index + 1].set_ylim(dfr_bound[0], dfr_bound[1])
        # add title
        if title:
            title = f"loop fit summary: '{label}'" if title is True else title
            figure.suptitle(title, fontsize=15)
        figure.align_labels()
        # tighten
        if tighten:
            figure.tight_layout()
        return axes_list
