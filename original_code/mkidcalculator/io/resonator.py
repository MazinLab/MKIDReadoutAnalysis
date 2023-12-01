import os
import copy
import logging
import inspect
import matplotlib
import lmfit as lm
import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.cluster.vq import kmeans2, ClusterError

from mkidcalculator.io.loop import Loop
from mkidcalculator.io.data import analogreadout_resonator
from mkidcalculator.io.utils import (lmfit, create_ranges, valid_ranges, save_lmfit, subplots_colorbar, dump, load,
                                     _loop_fit_data)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Resonator:
    """A class for manipulating resonator parameter data."""
    def __init__(self):
        # resonator reference
        self.name = None
        self._sweep = None
        self.loops = []
        self.powers = []
        self.fields = []
        self.temperatures = []
        self.temperature_groups = []
        # analysis results
        self.lmfit_results = {}
        self.loop_parameters = {}
        # directory of the data
        self._directory = None

    @property
    def sweep(self):
        """
        A settable property that contains the Resonator object that this loop
        has been assigned to. If the resonator has not been set, it will raise
        an AttributeError.
        """
        if self._sweep is None:
            raise AttributeError("The sweep object for this resonator has not been set yet.")
        return self._sweep

    @sweep.setter
    def sweep(self, sweep):
        self._sweep = sweep

    @property  # @property so that self.f not put into memory on load
    def f_center(self):
        """
        The median frequency of all of the loop.f_center frequencies. This is a
        useful rough proxy for the resonant frequency that depends only on the
        data and not the fit.
        """
        return np.median([loop.f_center for loop in self.loops])

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        dump(self, file_name)
        log.info("saved resonator as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Resonator class from the pickle file 'file_name'."""
        resonator = load(file_name)
        if not isinstance(resonator, cls):
            raise ValueError(f"'{file_name}' does not contain a Resonator "
                             "class.")
        log.info("loaded resonator from '{}'".format(file_name))
        return resonator

    def group_temperatures(self, n_groups=None):
        """
        Groups temperatures together into temperature_groups attribute since
        they aren't ever exactly equal.
           n_groups: integer
               An integer that determines how many temperature groups to
               include. The default is None, and n_groups is calculated. This
               procedure only works if the data is 'square' (same number of
               temperature points per unique power and field combination).
        Raises:
           scipy.cluster.vq.ClusterError:
               The temperature data is too disordered to cluster into the
               specified number of groups.
        """
        if np.isnan(self.temperatures).any():
            raise ValueError("Can't group NaN temperatures")
        temperatures = np.array(self.temperatures)
        if n_groups is None:
            n_groups = temperatures.size // (np.unique(self.powers).size * np.unique(self.fields).size)
        k = np.linspace(temperatures.min(), temperatures.max(), n_groups)
        try:
            centroids, groups = kmeans2(temperatures, k=k, minit='matrix', missing='raise')
        except ClusterError:
            message = "The temperature data is too disordered to cluster into {} groups".format(n_groups)
            raise ClusterError(message)
        self.temperature_groups = np.empty_like(self.temperatures)
        for index, centroid in enumerate(centroids):
            self.temperature_groups[groups == index] = centroid
        self.temperature_groups = list(self.temperature_groups)
        for index, loop in enumerate(self.loops):
            loop.temperature_group = self.temperature_groups[index]

    def add_loops(self, loops, sort=True):
        """
        Add Loop objects to the resonator.
        Args:
            loops: Loop class or iterable of Loop classes
                The loop objects that are to be added to the Resonator.
            sort: boolean (optional)
                Sort the loop list by its power, field, and temperature.
                The default is True. If False, the order of the loop list
                is preserved.
        """
        if isinstance(loops, Loop):
            loops = [loops]
        # append loop data
        for loop in loops:
            loop.resonator = self
            self.loops.append(loop)
            self.powers.append(loop.power)
            self.fields.append(loop.field)
            self.temperatures.append(loop.temperature)
        # sort
        if sort and self.loops:
            lp = zip(*sorted(zip(self.powers, self.fields, self.temperatures, self.loops), key=itemgetter(0, 1, 2)))
            self.powers, self.fields, self.temperatures, self.loops = (list(t) for t in lp)

    def remove_loops(self, indices):
        """
        Remove loops from the resonator.
        Args:
            indices: integer or iterable of integers
                The indices in resonator.loops that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.loops.pop(ii)
            self.powers.pop(ii)
            self.fields.pop(ii)
            self.temperatures.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Loop objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for loop in self.loops:
            loop.free_memory(directory=directory)

    @classmethod
    def from_file(cls, resonator_file_name, data=analogreadout_resonator, sort=True, **kwargs):
        """
        Resonator class factory method that returns a Resonator() with the loop,
        noise and pulse data loaded.
        Args:
            resonator_file_name: string
                The file name for the resonator data.
            data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to
                Loop.from_file().
            sort: boolean (optional)
                Sort the loop data by its power, field, and temperature. Also
                sort noise data and pulse data lists by their bias frequencies.
                The default is True. If False, the input order is preserved.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            resonator: object
                A Resonator() object containing the loaded data.
        """
        # create resonator
        resonator = cls()
        # load loop kwargs based on the resonator file
        loop_kwargs_list = data(resonator_file_name, **kwargs)
        loops = []
        # load loops
        for kws in loop_kwargs_list:
            kws.update({"sort": sort})
            loops.append(Loop.from_file(**kws))
        resonator.add_loops(loops, sort=sort)
        resonator.name = os.path.basename(resonator_file_name) + ", " + str(kwargs)
        return resonator

    def lmfit(self, parameter, model, guess, label='default', keep=True,
              residual_args=(), residual_kwargs=None, data_kwargs=None,
              **kwargs):
        """
        Compute a least squares fit using the supplied residual function and
        guess.
        Args:
            parameter: string or list of strings
                The loop parameters to fit. They must be a columns in the loop
                parameters table. If more than one parameter is specified a
                joint fit will be performed and the mkidcalculator.models.Joint
                class should be used.
            model: object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first two arguments must
                be a lmfit.Parameters object, and the parameter data. Other
                arguments can be passed in through the residual_args and
                residual_kwargs arguments.
            guess: lmfit.Parameters object
                A parameters object containing starting values (and bounds if
                desired) for all of the parameters needed for the residual
                function.
            index: tuple of 3 slices or a list of those tuples
                A list of slices for power, field and temperature which specify
                which data from the loop parameters table should be fit. The
                default is None and all data is fit. If a list of slices is
                used, the slices are concatenated.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.lmfit_results dictionary. The default is 'default'.
            keep: boolean (optional)
                Store the fit result in the object. The default is True. If
                False, the fit will only be stored if it is the best so far.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first two. The default is an empty tuple.
            residual_kwargs: dictionary (optional)
                A dictionary of arguments to be passed to the residual
                function. The default is None, which corresponds to an empty
                dictionary.
            kwargs: optional keyword arguments
                Additional keyword arguments are sent to the
                lmfit.Minimizer.minimize() method.
        Returns:
            result: lmfit.MinimizerResult
                An object containing the results of the minimization. It is
                also stored in self.lmfit_results[label]['result'].
        """
        # get the data to fit
        if isinstance(parameter, str):
            parameter = [parameter]
        args_list = []
        kws_list = []
        # collect the arguments for each parameter
        for p in parameter:
            data, sigmas, temperatures, powers = _loop_fit_data(
                self.loops, [p, p + "_sigma", 'temperature', 'power'],
                **data_kwargs)
            if p in ['fr', 'f0']:
                data = data * 1e9  # convert to Hz for model
            args = (data, *residual_args)
            args_list.append(args)
            kws = {"temperatures": temperatures, "powers": powers}
            if (~np.isnan(sigmas)).all():
                if p in ['fr', 'f0']:
                    sigmas = sigmas * 1e9  # convert to Hz for model
                kws.update({"sigmas": sigmas})
            if residual_kwargs is not None:
                kws.update(residual_kwargs)
            kws_list.append(kws)
        # reformat the arguments to work with one or many parameters
        if len(parameter) == 1:
            args = args_list[0]
            kws = kws_list[0]
        else:
            args = [tuple([args_list[ind][index]
                           for ind, _ in enumerate(args_list)])
                    for index, _ in enumerate(args_list[0])]
            kws = {key: tuple(kws_list[ind][key]
                              for ind, _ in enumerate(kws_list))
                   for key in kws_list[0].keys()}
        # make sure the dictionary exists for each parameter
        for p in parameter:
            if p not in self.lmfit_results.keys():
                self.lmfit_results[p] = {}
        # do the fit for the first parameter
        result = lmfit(self.lmfit_results[parameter[0]], model, guess,
                       label=label, keep=keep, residual_args=args,
                       residual_kwargs=kws,
                       model_index=0 if len(parameter) != 1 else None,
                       **kwargs)
        # copy the result to the other parameters
        for ind, p in enumerate(parameter[1:]):
            save_lmfit(self.lmfit_results[p], model.models[ind + 1], result,
                       label=label, keep=keep,
                       residual_args=args_list[ind + 1],
                       residual_kwargs=kws_list[ind + 1])
        return result

    def emcee(self):
        raise NotImplementedError

    def fit_report(self, parameter, label='best', fit_type='lmfit', return_string=False):
        """
        Print a string summarizing a resonator fit.
        Args:
            parameter: string
                The parameter on which the fit was done.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            return_string: boolean
                Return a string with the fit report instead of printing. The
                default is False.

        Returns:
            string: string
                A string containing the fit report. None is output if
                return_string is False.
        """
        _, result_dict = self._get_model(parameter, fit_type, label)
        string = lm.fit_report(result_dict['result'])
        if return_string:
            return string
        else:
            print(string)

    def _set_directory(self, directory):
        self._directory = directory
        for loop in self.loops:
            loop._set_directory(self._directory)

    def _get_model(self, parameter, fit_type, label):
        if fit_type not in ['lmfit', 'emcee', 'emcee_mle']:
            raise ValueError("'fit_type' must be either 'lmfit', 'emcee', or 'emcee_mle'")
        if fit_type == "lmfit" and label in self.lmfit_results[parameter].keys():
            result_dict = self.lmfit_results[parameter][label]
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        elif fit_type == "emcee" and label in self.emcee_results[parameter].keys():
            result_dict = self.emcee_results[parameter][label]
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        elif fit_type == "emcee_mle" and label in self.emcee_results[parameter].keys():
            result_dict = copy.deepcopy(self.emcee_results[parameter][label])
            for name in result_dict['result'].params.keys():
                result_dict['result'].params[name].set(value=self.emcee_results[parameter][label]["mle"][name])
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        else:
            result_dict = None
            original_label = None
        return original_label, result_dict

    def plot_loops(self, power=None, field=None, temperature=None, color_data='power', colormap=None,
                   colorbar=True, colorbar_kwargs=None, colorbar_label=True, colorbar_limits=None,
                   colorbar_label_kwargs=None, colorbar_tick_kwargs=None, tighten=True, **loop_kwargs):
        """
        Plot a subset of the loops in the resonator by combining multiple
        loop.plot() calls.
        Args:
            power: tuple of two number tuples or numbers
                Inclusive range or ranges of powers to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the powers.
            field: tuple of two number tuples or numbers
                Inclusive range or ranges of fields to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the fields.
            temperature: tuple of two number tuples or numbers
                Inclusive range or ranges of temperatures to plot. A single
                number will cause only that value to be plotted. The default is
                to include all of the temperatures.
            color_data: string
                Either 'temperature', 'field', or 'power' indicating off what
                type of data to base the colormap. The default is
                'power'.
            colormap: matplotlib.colors.Colormap
                A matplotlib colormap for coloring the data. If the default
                None is used, a colormap is chosen based on color_data.
            colorbar: boolean
                Determines whether to include a colorbar. The default is True.
                If False, colorbar_kwargs, colorbar_label, and
                colorbar_label_kwargs are ignored.
            colorbar_kwargs: dictionary
                Keyword arguments for the colorbar in figure.colorbar(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            colorbar_label: boolean or string
                If it is a boolean, it determines whether or not to add the
                default colorbar label. If it is a string, that string is used
                as the colorbar label. If False, colorbar_label_kwargs is
                ignored. The default is True.
            colorbar_limits: tuple of floats
                The limits of the colorbar to use. The default is None and the
                maximum and minimum of color_data is used.
            colorbar_label_kwargs: dictionary
                Keyword arguments for the colorbar in colorbar.set_label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            colorbar_tick_kwargs: dictionary
                Keyword arguments for the colorbar ticks using
                colorbar_axes.tick_params(). The default is None which uses the
                default options. Keywords in this dictionary override the
                default options.
            tighten: boolean
                Determines whether figure.tight_layout() is called. The default
                is True.
            loop_kwargs: optional keyword arguments
                Extra keyword arguments to send to loop.plot().
        Returns:
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes with the plotted data.
        """
        # parse inputs
        if "fit_parameters" in loop_kwargs.keys():
            raise TypeError("'fit_parameters' is not a valid keyword argument")
        if "parameters_kwargs" in loop_kwargs.keys():
            raise TypeError("'parameters_kwargs' is not a valid keyword argument")
        power, field, temperature = create_ranges(power, field, temperature)
        if color_data == 'temperature':
            cmap = matplotlib.cm.get_cmap('coolwarm') if colormap is None else colormap
            cdata = np.array(self.temperatures[::-1]) * 1000
        elif color_data == 'field':
            cmap = matplotlib.cm.get_cmap('viridis') if colormap is None else colormap
            cdata = self.fields[::-1]
        elif color_data == 'power':
            cmap = matplotlib.cm.get_cmap('plasma') if colormap is None else colormap
            cdata = self.powers[::-1]
        else:
            raise ValueError("'{}' is not a valid value of color_data.".format(color_data))
        if colorbar_limits is None:
            colorbar_limits = (min(cdata), max(cdata))
        norm = matplotlib.colors.Normalize(vmin=colorbar_limits[0], vmax=colorbar_limits[1])
        n_plots = 3 if 'plot_types' not in loop_kwargs.keys() else len(loop_kwargs['plot_types'])
        axes_list = None
        # format title
        title = loop_kwargs.get("title", True)
        if title is True:
            # power
            if len(power) == 1 and np.isinf(power[0]).all():
                title = "All Powers, "
            elif all(x == power[0] for x in power) and power[0][0] == power[0][1]:
                title = "{:.0f} dBm, ".format(power[0][0])
            else:
                title = "({:.0f}, {:.0f}) dBm, ".format(np.min(power[0]), np.max(power[-1]))
            # field
            if len(field) == 1 and np.isinf(field[0]).all():
                title += "All Fields, "
            elif all(x == field[0] for x in field) and field[0][0] == field[0][1]:
                title += "{:.0f} V, ".format(field[0][0])
            else:
                title += "({:.0f}, {:.0f}) V, ".format(np.min(field[0]), np.max(field[-1]))
            # temperature
            if len(temperature) == 1 and np.isinf(temperature[0]).all():
                title += "All Temperatures"
            elif all(x == temperature[0] for x in temperature) and temperature[0][0] == temperature[0][1]:
                title += "{:.0f} mK".format(temperature[0][0] * 1000)
            else:
                title += "({:.0f}, {:.0f}) mK".format(np.min(temperature[0]) * 1000, np.max(temperature[-1]) * 1000)
        # store key word options
        user_plot_kwargs = loop_kwargs.get('plot_kwargs', [])
        if isinstance(user_plot_kwargs, dict):
            user_plot_kwargs = [user_plot_kwargs] * n_plots
        user_data_kwargs = []
        for kw in user_plot_kwargs:
            user_data_kwargs.append(kw.get("data_kwargs", {}))
        user_fit_kwargs = []
        for kw in user_plot_kwargs:
            user_fit_kwargs.append(kw.get("fit_kwargs", {}))
        # make a plot for each loop
        plot_index = 0
        for index, loop in enumerate(self.loops[::-1]):
            if valid_ranges(loop, power, field, temperature):
                # default plot key words
                if plot_index == 0:
                    plot_kwargs = [{'data_kwargs': {'color': cmap(norm(cdata[index]))},
                                    'fit_kwargs': {'color': 'k'}}] * n_plots
                else:
                    plot_kwargs = [{'x_label': '', 'y_label': '', 'data_kwargs': {'color': cmap(norm(cdata[index]))},
                                    'fit_kwargs': {'color': 'k'}}] * n_plots
                # update data key words with user defaults
                for kw_index, data_kw in enumerate(user_data_kwargs):
                    plot_kwargs[kw_index]['data_kwargs'].update(data_kw)
                # update fit key words with user defaults
                for kw_index, fit_kw in enumerate(user_fit_kwargs):
                    plot_kwargs[kw_index]['fit_kwargs'].update(fit_kw)
                # update plot key words with user defaults
                for kw_index, kws in enumerate(user_plot_kwargs):
                    kws = kws.copy()
                    kws.pop('data_kwargs', None)
                    kws.pop('fit_kwargs', None)
                    plot_kwargs[kw_index].update(kws)
                # update loop kwargs
                if plot_index == 0:
                    loop_kwargs.update({"plot_kwargs": plot_kwargs, "title": title, "tighten": False})
                else:
                    loop_kwargs.update({"axes_list": axes_list, "title": False, "legend": False, "tighten": False,
                                        "plot_kwargs": plot_kwargs})
                axes_list = loop.plot(**loop_kwargs)
                plot_index += 1
        # if we didn't plot anything exit the function
        if axes_list is None:
            return
        gs = None
        if colorbar:
            mappable = matplotlib.cm.ScalarMappable(norm, cmap)
            mappable.set_array([])
            kwargs = {}
            if colorbar_kwargs is not None:
                kwargs.update(colorbar_kwargs)

            if all([isinstance(axes, matplotlib.axes.SubplotBase) for axes in axes_list]) and len(axes_list) != 1:
                gs_kwargs = {"top": 0.9 if title else 1}
                cbar, gs = subplots_colorbar(mappable, axes_list, gridspec_kwargs=gs_kwargs, **kwargs)
            else:
                ax = axes_list[0] if len(axes_list) == 1 else axes_list
                cbar = axes_list[0].figure.colorbar(mappable, ax=ax, **kwargs)

            if colorbar_label:
                if color_data == 'temperature':
                    label = "temperature [mK]" if colorbar_label is True else colorbar_label
                elif color_data == 'field':
                    label = "field [V]" if colorbar_label is True else colorbar_label
                else:
                    label = "power [dBm]" if colorbar_label is True else colorbar_label
                kwargs = {"rotation": 270, 'va': 'bottom'}
                if colorbar_label_kwargs is not None:
                    kwargs.update(colorbar_label_kwargs)
                cbar.set_label(label, **kwargs)
                if colorbar_tick_kwargs is not None:
                    cbar.ax.tick_params(**colorbar_tick_kwargs)
        if tighten:
            if gs is not None:
                gs.tight_layout(axes_list[0].figure, rect=[0, 0, 1, 0.9 if title else 1])
            else:
                axes_list[0].figure.tight_layout(rect=[0, 0, 1, 0.9 if title else 1])
        return axes_list

    def plot_parameters(self, parameters, x="power", n_rows=1, n_sigma=2, plot_fit=False, label="best", axes_list=None,
                        **loop_kwargs):
        # set up axes and input arguments
        if isinstance(parameters, str):
            parameters = [parameters]
        if axes_list is None:
            from matplotlib import pyplot as plt
            n_columns = int(np.ceil(len(parameters) / n_rows))
            figure, axes_list = plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False,
                                             figsize=(4.3 * n_columns, 4.0 * n_rows))
            axes_list = axes_list.ravel()
        else:
            if not isinstance(axes_list, np.ndarray):
                axes_list = np.atleast_1d(axes_list)
            figure = axes_list[0].figure
        # set up loop attributes to iterate over
        levels = ["power", "field", "temperature"]
        if x not in levels:
            raise ValueError("x must be in {}".format(levels))
        values_dict = {"power": np.unique(self.powers), "field": np.unique(self.fields)}
        if x != "temperature" and self.temperature_groups:
            levels[2] = "temperature_group"
            values_dict.update({"temperature_group": np.unique(self.temperature_groups)})
        else:
            values_dict.update({"temperature": np.unique(self.temperatures)})
        for key, values in values_dict.items():  # unique doesn't work with nan
            if np.isnan(values).any():
                values_dict[key] = np.concatenate((values[~np.isnan(values)], [np.nan]))
        levels.remove(x)
        # make one set of axes per parameter
        kwargs = {"label": "best"}
        for index, parameter in enumerate(parameters):
            # collect data for parameter
            kwargs.update({"parameters": [parameter, parameter + "_sigma", x, levels[0], levels[1]]})
            kwargs.update(loop_kwargs)
            data, error_bars, x_vals, values0, values1 = _loop_fit_data(self.loops, **kwargs)
            # plot in mK
            if x == "temperature":
                x_vals = x_vals * 1000
            # plot a line for each level
            for value0 in values_dict[levels[0]]:
                for value1 in values_dict[levels[1]]:
                    logic = np.isclose(value0, values0, equal_nan=True) & np.isclose(value1, values1, equal_nan=True)
                    # plot data
                    if ~np.isnan(data[logic]).any():
                        if ~np.isnan(error_bars[logic]).any():
                            axes_list[index].errorbar(x_vals[logic], data[logic], error_bars[logic] * n_sigma, fmt='o')
                        else:
                            axes_list[index].plot(x_vals[logic], data[logic], 'o')
                        if parameter.endswith("_sigma"):
                            axes_list[index].set_ylabel("_".join(parameter.split("_")[:-1]) + " sigma")
                        else:
                            axes_list[index].set_ylabel(parameter)
                        x_label = {"power": "power [dBm]", "field": "field [V]", "temperature": "temperature [mK]"}
                        axes_list[index].set_xlabel(x_label[x])
                        # plot a fit if it's been done
                        if plot_fit and parameter in self.lmfit_results.keys():
                            if label in self.lmfit_results[parameter].keys():
                                result_dict = self.lmfit_results[parameter][label]
                                result = result_dict['result']
                                model = result_dict['model']
                                parameters = inspect.signature(model.model).parameters
                                residual_kwargs = result_dict['kwargs']
                                kwargs = {}
                                for key in parameters.keys():
                                    if key in residual_kwargs.keys():
                                        kwargs.update({key: residual_kwargs[key]})
                                if 'parallel' in kwargs.keys():
                                    kwargs['parallel'] = bool(kwargs['parallel'])
                                args = result_dict['args'][1:]
                                m = model.model(result.params, *args, **kwargs)
                                x_m = kwargs[x + "s"] if x != "temperature" else 1000 * kwargs[x + "s"]
                                if parameter == "fr":
                                    m *= 1e-9  # Convert to GHz
                                axes_list[index].plot(x_m, m)
        figure.tight_layout()
        return axes_list
