import os
import signal
import pickle
import numbers
import logging
import tempfile
import matplotlib
import numpy as np
import lmfit as lm
from collections import OrderedDict
import scipy.constants as c
from scipy.signal import find_peaks, detrend

from mkidcalculator.models.s21 import S21

try:
    import loopfit
    HAS_LOOPFIT = True
except ImportError:
    loopfit = None
    HAS_LOOPFIT = False

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class NpzCache:
    """
    Cached lazy loading of npz file. Reads from disk only on the first access.
    """
    def __init__(self, npz):
        self._npz = npz
        self._dict = {}

    def __getitem__(self, item):
        if item in self._dict.keys():
            return self._dict[item]
        else:
            value = self._npz[item]
            self._dict[item] = value
            return value


class NpzHolder:
    """Loads npz file when requested and saves them."""
    MAX_SIZE = 200

    def __init__(self):
        self._files = OrderedDict()

    def __getitem__(self, item):
        # if string load and save to cache
        if isinstance(item, str):
            item = os.path.abspath(item)
            # check if already loaded
            if item in self._files.keys():
                log.debug("loaded from cache: {}".format(item))
                return self._files[item]
            else:
                self._check_size()
                npz = np.load(item, allow_pickle=True)
                log.debug("loaded: {}".format(item))
                self._files[item] = NpzCache(npz)
                log.debug("saved to cache: {}".format(item))
                return self._files[item]
        # if NpzFile skip loading but save if it hasn't been loaded before
        elif isinstance(item, np.lib.npyio.NpzFile):
            file_name = os.path.abspath(item.fid.name)
            if file_name not in self._files.keys():
                self._check_size()
                log.debug("loaded: {}".format(file_name))
                self._files[file_name] = NpzCache(item)
                log.debug("saved to cache: {}".format(file_name))
            else:
                log.debug("loaded from cache: {}".format(file_name))
                item = self._files[file_name]
            return item
        elif item is None:
            return None
        else:
            raise ValueError("'item' must be a valid file name or a numpy npz file object.")

    def free_memory(self, file_names=None):
        """
        Removes file names in file_names from active memory. If file_names is
        None, all are removed (default).
        """
        if file_names is None:
            file_names = list(self._files.keys())
        elif isinstance(file_names, str):
            file_names = [file_names]
        for file_name in file_names:
            npz = self._files.pop(file_name, None)
            del npz

    def _check_size(self):
        for _ in range(max(len(self._files) - self.MAX_SIZE + 1, 0)):
            item = self._files.popitem(last=False)
            log.debug("Max cache size reached. Removed from cache: {}".format(item[0]))


_loaded_npz_files = NpzHolder()  # cache of already loaded files


def compute_phase_and_dissipation(cls, label="best", fit_type="lmfit", **kwargs):
    """
    Compute the phase and dissipation traces stored in pulse.p_trace and
    pulse.d_trace.
    Args:
        cls: Pulse or Noise class
            The Pulse or Noise class used to create the phase and dissipation
            data.
        label: string
            Corresponds to the label in the loop.lmfit_results or
            loop.emcee_results dictionaries where the fit parameters are.
            The resulting DataFrame is stored in
            object.loop_parameters[label]. The default is "best", which gets
            the parameters from the best fits.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit", "emcee",
            and "emcee_mle" where MLE estimates are used instead of the
            medians. The default is "lmfit".
        kwargs: optional keyword arguments
            Optional keyword arguments to send to
            model.phase_and_dissipation(). If using the loopfit fit_type,
            S21.phase_and_dissipation() is used.
    """
    # clear prior data
    cls.clear_traces()
    # get the model and parameters
    _, result_dict = cls.loop._get_model(fit_type, label)
    if fit_type in ["lmfit", "emcee", "emcee_mle"]:
        model = result_dict["model"]
        params = result_dict["result"].params
        # compute phase and dissipation  # TODO: input I and Q separately to save memory
        phase, dissipation = model.phase_and_dissipation(params, cls.i_trace + 1j * cls.q_trace, cls.f_bias, **kwargs)
    elif fit_type == "loopfit":
        if not HAS_LOOPFIT:
            raise ImportError("The loopfit package is not installed.")
        # use the S21.compute_phase_and_dissipation algorithm by making a params object
        params = lm.Parameters()
        for key, value in result_dict.items():
            if isinstance(value, float):
                params.add(key, value=value)
        params.add("phase2", value=0.)
        params.add("q0", expr="1 / (1 / qi + 1 / qc)")
        params.add("fr", expr="f0 * (1 - a / q0)")
        phase, dissipation = S21.phase_and_dissipation(params, cls.i_trace + 1j * cls.q_trace, cls.f_bias, **kwargs)
    else:
        raise ValueError("{} is not a valid fit type".format(fit_type))
    cls.p_trace = phase
    cls.d_trace = dissipation


def offload_data(cls, excluded_keys=(), npz_key="_npz", prefix="", directory_key="_directory"):
    """
    Offload data in excluded_keys from the class to an npz file. The npz file
    name is stored in cls.npz_key.
    Args:
        cls: class
            The class being unpickled
        excluded_keys: iterable of strings
            Keys to force into npz format. The underlying attributes must be
            numpy arrays. The default is to not exclude any keys.
        npz_key: string
            The class attribute name that corresponds to where the npz file was
            stored. The default is "_npz".
        prefix: string
            File name prefix for the class npz file if a new one needs to be
            made. The default is no prefix.
        directory_key: string
            The class attribute that corresponds to the data directory. If it
            doesn't exist, than the current directory is used.
    Returns:
        cls.__dict__: dictionary
            The new class dict which can be used for pickling.
    """
    # get the directory
    directory = "." if getattr(cls, directory_key, None) is None else getattr(cls, directory_key)
    directory = os.path.abspath(directory)
    # default is to not export to a new npz file
    make_npz = False
    # if any of the excluded keys are numpy arrays we will export to a new npz file
    for key in excluded_keys:
        make_npz = make_npz or isinstance(getattr(cls, key), np.ndarray)
    if isinstance(getattr(cls, npz_key), str):
        file_name = getattr(cls, npz_key)
        # if the npz_key directory doesn't match we will export the excluded keys to a new npz file
        if os.path.dirname(file_name) != directory:
            make_npz = True
        # if we are exporting and there already exists a npz file, ensure the real data is loaded into the class
        if make_npz:
            npz = _loaded_npz_files[file_name]
            for key in excluded_keys:
                if isinstance(getattr(cls, key), str):  # don't load in data that's been overloaded by the user
                    setattr(cls, key, npz[key])
    if make_npz:
        # get the data to save
        excluded_data = {}
        for key in excluded_keys:
            if getattr(cls, key) is not None:
                excluded_data[key] = getattr(cls, key)
        # if there is data to save, save it
        if excluded_data:
            # get the npz file name
            file_name = tempfile.mkstemp(prefix=prefix, suffix=".npz", dir=directory)[1]
            np.savez(file_name, **excluded_data)
            setattr(cls, npz_key, file_name)
    # change the excluded keys in the dict to the key for the npz_file if it exists
    if getattr(cls, npz_key) is not None:
        for key in excluded_keys:
            cls.__dict__[key] = key
    return cls.__dict__


def quadratic_spline_roots(spline):
    """Returns the roots of a scipy spline."""
    roots = []
    knots = spline.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spline(a), spline((a + b) / 2), spline(b)
        t = np.roots([u + w - 2 * v, w - u, 2 * v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t * (b - a) / 2 + (b + a) / 2)
    return np.array(roots)


def ev_nm_convert(x):
    """
    If x is a wavelength in nm, the corresponding energy in eV is returned.
    If x is an energy in eV, the corresponding wavelength in nm is returned.
    """
    return c.speed_of_light * c.h / c.eV * 1e9 / x


def load_legacy_binary_data(binary_file, channel, n_points, noise=True, offset=0, chunk=None):
    """
    Load data from legacy Matlab code binary files (.ns or .dat).
    Args:
        binary_file: string
            The full file name and path to the binary data.
        channel: integer
            Either 0 or 1 specifying which channel to load data from.
        n_points: integer
            The number of points per trigger per trace. Both I and Q traces
            should have the same number of points.
        noise: boolean (optional)
            A flag specifying if the data is in the noise or pulse format.
        offset: integer (optional)
            Ignore this many of the first triggers in the data set. The default
            is 0, and all triggers are used.
        chunk: integer (optional)
            Only load this many triggers from the data set. The default is
            None, and all of the triggers are loaded.
    Returns:
        i_trace: numpy.ndarray
            An N x n_points numpy array with N traces.
        q_trace: numpy.ndarray
            An N x n_points numpy array with N traces.
        f: float
            The bias frequency for the data.
    """
    # get the binary data from the file
    data = np.memmap(binary_file, dtype=np.int16, mode='r')
    # grab the tone frequency (not a 16 bit integer)
    if noise:
        f = np.frombuffer(data[4 * channel: 4 * (channel + 1)].tobytes(), dtype=np.float64)[0]
    else:
        f = np.frombuffer(data[4 * (channel + 2): 4 * (channel + 3)].tobytes(), dtype=np.float64)[0]
    # remove the header from the array
    data = data[4 * 12:] if noise else data[4 * 14:]
    # check that we have an integer number of triggers
    n_triggers = data.size / n_points / 4.0
    if not n_triggers.is_integer():
        raise ValueError("non-integer number of traces found in {0}".format(binary_file))
    n_triggers = int(n_triggers)
    if chunk is None:
        chunk = n_triggers - offset
    if offset + chunk > n_triggers:  # don't try to return chunk triggers if there aren't that many
        chunk = n_triggers - offset
    if chunk <= 0:
        raise ValueError("offset exceeds the number of traces found in {0}".format(binary_file))
    # break noise data into I and Q data
    i_trace = np.zeros((chunk, n_points), dtype=np.float16)
    q_trace = np.zeros((chunk, n_points), dtype=np.float16)
    convert = 0.2 / 32767.0  # convert the data to voltages * 0.2 V / (2**15 - 1)
    for trigger_num in range(chunk):
        trace_num = 4 * (trigger_num + offset)
        i_trace[trigger_num, :] = data[(trace_num + 2 * channel) * n_points:
                                       (trace_num + 2 * channel + 1) * n_points].astype(np.float16) * convert
        q_trace[trigger_num, :] = data[(trace_num + 2 * channel + 1) * n_points:
                                       (trace_num + 2 * channel + 2) * n_points].astype(np.float16) * convert
    return i_trace, q_trace, f


def structured_to_complex(array):
    if array is None or array.dtype == np.complex or array.dtype == np.complex64:
        return array
    else:
        return array["I"] + 1j * array["Q"]


def lmfit(lmfit_results, model, guess, label='default', residual_args=(), residual_kwargs=None, model_index=None,
          keep=True, **kwargs):
    if label == 'best':
        raise ValueError("'best' is a reserved label and cannot be used")
    # set up and do minimization
    minimizer = lm.Minimizer(model.residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
    result = minimizer.minimize(**kwargs)
    # save the results
    if model_index is not None:
        model = model.models[model_index]  # if the fit was done with a joint model only save the relevant part
        residual_args = tuple([arg[model_index] if isinstance(arg, tuple) else arg for arg in residual_args])
        residual_kwargs = {key: value[model_index] if isinstance(value, tuple) else value
                           for key, value in residual_kwargs.items()}
    save_lmfit(lmfit_results, model, result, label=label, residual_args=residual_args, residual_kwargs=residual_kwargs,
               keep=keep)
    return result


def save_lmfit(lmfit_results, model, result, label='default', residual_args=(), residual_kwargs=None, keep=True):
    save_dict = {'result': result, 'model': model, 'kwargs': residual_kwargs, 'args': residual_args}
    if keep:
        if label in lmfit_results:
            message = "'{}' has already been used as an lmfit label. The old data has been overwritten."
            log.warning(message.format(label))
        lmfit_results[label] = save_dict
    # if the result is better than has been previously computed, add it to the 'best' key
    save = ('best' not in lmfit_results.keys()
            or (result.aic < lmfit_results['best']['result'].aic
                and (result.errorbars # don't save if we already have errors
                     or not lmfit_results['best']['result'].errorbars)))
    if 'best' not in lmfit_results.keys() or result.aic < lmfit_results['best']['result'].aic:
        lmfit_results['best'] = save_dict
        lmfit_results['best']['label'] = label


def create_range(value):
    # single input case
    if value is None:
        value = [(-np.inf, np.inf)]
    elif not isinstance(value, (tuple, list, np.ndarray)):
        value = [(value, value)]
    value = list(value)
    # check each span for single elements
    for index, span in enumerate(value):
        if isinstance(span, numbers.Number):
            value[index] = (span, span)
    return tuple(value)


def create_ranges(power, field, temperature):
    power = create_range(power)
    field = create_range(field)
    temperature = create_range(temperature)
    return power, field, temperature


def valid_range(value, spans):
    return any(spans[i][0] <= value <= spans[i][1] for i in range(len(spans))) or np.isnan(value)


def valid_ranges(loop, power, field, temperature):
    condition = (valid_range(loop.power, power) and valid_range(loop.field, field) and
                 valid_range(loop.temperature, temperature))
    return condition


def sort_and_fix(data, energies, fix_zero):
    data = [0] + data if fix_zero else data
    data, energies = np.array(data), np.array(energies)
    energies, indices = np.unique(energies, return_index=True)
    data = data[indices]
    return data, energies


def setup_axes(axes, x_label, y_label, label_kwargs=None, x_label_default="", y_label_default="", equal=False):
    if axes is None:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    if x_label is None:
        x_label = x_label_default
    if y_label is None:
        y_label = y_label_default
    # setup axes
    kwargs = {}
    if label_kwargs is not None:
        kwargs.update(label_kwargs)
    if x_label:
        axes.set_xlabel(x_label, **kwargs)
    if y_label:
        axes.set_ylabel(y_label, **kwargs)
    if equal:
        axes.axis('equal')
    return figure, axes


def finalize_axes(axes, title=False, title_kwargs=None, legend=False, legend_kwargs=None, tick_kwargs=None,
                  tighten=False):
    if legend:
        kwargs = {}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        axes.legend(**kwargs)
    # make the title
    if title:
        kwargs = {"fontsize": 11}
        if title_kwargs is not None:
            kwargs.update(title_kwargs)
        axes.set_title(title, **kwargs)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if tighten:
        axes.figure.tight_layout()


def get_plot_model(self, fit_type, label, params=None, calibrate=False, default_kwargs=None, plot_kwargs=None,
                   center=False, use_mask=True, n_factor=10):
    # get the model
    fit_name, result_dict = self._get_model(fit_type, label)
    if fit_name is None:
        raise ValueError("No fit of type '{}' with the label '{}' has been done".format(fit_type, label))
    # calculate the model values
    if use_mask:
        n = (np.max(self.f[self.mask]) -
             np.min(self.f[self.mask])) / np.min(np.diff(self.f[self.mask])) + 1
        f = np.linspace(np.min(self.f[self.mask]), np.max(self.f[self.mask]), int(n * n_factor))
    else:
        n = (np.max(self.f) - np.min(self.f)) / np.min(np.diff(self.f)) + 1
        f = np.linspace(np.min(self.f), np.max(self.f), int(n * n_factor))
    if fit_type in ["lmfit", "emcee", "emcee_mle"]:
        if params is None:
            params = result_dict['result'].params
        model = result_dict['model']
        m = model.model(params, f)
        if calibrate:
            m = model.calibrate(result_dict['result'].params, m, f, center=center)  # must calibrate wrt the fit
    else:
        if not HAS_LOOPFIT:
            raise ImportError("The loopfit package is not installed.")
        if params is None:
            params = result_dict
        m = loopfit.model(f, **params)
        if calibrate:
            m = loopfit.calibrate(f, z=m, center=center, **result_dict)
    # add the plot
    kwargs = {} if default_kwargs is None else default_kwargs
    if plot_kwargs is not None:
        kwargs.update(plot_kwargs)
    return f, m, kwargs


def _integer_bandwidth(f, df):
    return int(np.round(df / (f[1] - f[0])))


def find_resonators(f, z, df=5e-4, index=None, **kwargs):
    """
    Find resonators in a S21 trace.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to the magnitude array.
        z: numpy.ndarray
            The complex scattering data.
        df: float (optional)
            The frequency bandwidth for each resonator. df / 2 will be used as
            the max peak width unless overridden. Resonators separated by less
            than df / 2 are also discarded. If None, no max peak width or
            frequency cut is used. The default is 5e-4 (0.5 MHz).
        index: tuple of integers (optional)
            A tuple corresponding to all but the last index in f to use for
            finding resonators.
        kwargs: optional keyword arguments
            Optional keyword arguments to scipy.signal.find_peaks. Values here
            will override the defaults.
    Returns:
        index_array: tuple of (numpy.ndarray, dtype=integer)
            An array of peak locations
    """
    # find peaks for only the first temp, field, atten if index isn't given
    if index is None:
        index = tuple([0] * (f.ndim - 1))
    z = z[index]
    f = f[index]
    # resonator bandwidth in indices
    dfii = _integer_bandwidth(f, df) if df is not None else None
    # detrend magnitude data for peak finding
    magnitude = detrend(20 * np.log10(np.abs(z)))
    fit = np.argsort(magnitude)[:int(3 * len(magnitude) / 4):-1]
    poly = np.polyfit(f[fit], magnitude[fit], 1)
    magnitude = magnitude - np.polyval(poly, f)
    # find peaks
    kws = {"prominence": 1, "height": 5}
    if dfii is not None:
        kws.update({"width": (None, int(dfii / 2)), "distance": int(dfii / 2)})
    kws.update(kwargs)
    peaks, _ = find_peaks(-magnitude, **kws)
    peaks = peaks[(peaks > dfii // 2) & ((f.size - peaks) > dfii // 2)]
    return peaks


def collect_resonances(f, z, peaks, df):
    """
    Collect all of the resonances from a data into an array.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to i and q.
        z: numpy.ndarray
            The S21 complex scattering data.
        peaks: iterable of integers
            The indices corresponding to the resonator locations in frequency.
        df: float
            The final bandwidth of all of the outputs.
    Returns:
        f_array: numpy.ndarray
            A JxKxLxMxN array for the frequencies where J is the number of
            temperatures, K is the number of fields, L is the number of
            attenuations, M is the number of resonators and N is the number
            of frequencies.
        z_array: numpy.ndarray
            An array for the S21 data of the same size as f_array.
    """
    f = np.array(f, ndmin=4, copy=False)  # (temp, field, atten, frequencies)
    z = np.array(z, ndmin=4, copy=False)  # (temp, field, atten, z)
    peaks = np.array(peaks)

    # resonator bandwidth in indices
    dfii = _integer_bandwidth(f[(0,) * (f.ndim - 1)], df)

    # collect resonance data into arrays
    offset = np.arange(-dfii // 2 + 1, dfii // 2 + 1)[:, np.newaxis]
    index_array = (peaks + offset).T
    try:
        f_array = f[..., index_array]
        z_array = z[..., index_array]
    except IndexError:
        raise IndexError("Some frequency windows are outside of the supplied data range. "
                         "Either remove the offending peaks or make the bandwidth smaller.")
    return f_array, z_array


def _red_chi(fit_type, result):
    if fit_type == 'lmfit':
        if isinstance(result, dict):
            return result['result'].redchi
        else:
            return result.redchi
    elif fit_type == 'loopfit':
        return result['chi_squared'] / (result['size'] - result['varied'])
    else:
        raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")


def _loop_fit_data(loops, parameters=("chi2",), fit_type="lmfit", label='best',
                   bounds=None, errorbars=None, success=None, power=None,
                   field=None, temperature=None):
    if isinstance(parameters, str):
        parameters = [parameters]
    power, field, temperature = create_ranges(power, field, temperature)
    outputs = []
    for parameter in parameters:
        outputs.append([])
        for loop in loops:
            if valid_ranges(loop, power, field, temperature):
                try:
                    result = getattr(loop, fit_type + "_results")[label]
                except KeyError:
                    continue  # no fit for this label
                if fit_type == "lmfit":
                    minim = result['result']
                    if errorbars is not None and minim.errorbars != errorbars:
                        continue  # skip if wrong errorbars setting
                    if success is not None and minim.success != success:
                        continue  # skip if wrong success setting
                try:
                    if fit_type == "lmfit":
                        outputs[-1].append(result['result'].params[parameter].value)
                    elif fit_type == "loopfit":
                        outputs[-1].append(result[parameter])
                except KeyError as error:
                    if parameter.endswith("_sigma"):
                        try:
                            error = result['result'].params[parameter[:-6]].stderr  # only allowed for lmfit
                            if error is None:
                                print(result['result'].params[parameter[:-6]],
                                      result['result'].params[parameter[
                                                              :-6]].stderr)
                                error = np.nan
                            outputs[-1].append(error)
                        except KeyError:
                            outputs[-1].append(np.nan)
                    elif parameter.startswith("chi2") or parameter.startswith("redchi"):
                        outputs[-1].append(_red_chi(fit_type, result))
                    elif parameter == "q0" and fit_type == "loopfit":
                        outputs[-1].append(1 / (1 / result["qi"] + 1 / result["qc"]))
                    elif parameter == "power":
                        outputs[-1].append(loop.power)
                    elif parameter == "field":
                        outputs[-1].append(loop.field)
                    elif parameter == "temperature":
                        outputs[-1].append(loop.temperature)
                    elif parameter == "temperature_group":
                        outputs[-1].append(loop.temperature_group)
                    else:
                        raise error
    # turn outputs into a list of numpy arrays
    for index, output in enumerate(outputs):
        outputs[index] = np.array(output)
    # format bounds if None
    if bounds is None:
        bounds = [None] * len(outputs)
    # make filtering logic
    logic = np.ones(outputs[-1].shape, dtype=bool)
    for index, output in enumerate(outputs):
        if bounds[index] is None:
            continue
        elif isinstance(bounds[index], (list, tuple)):
            logic = logic & (output >= bounds[index][0]) & (output <= bounds[index][1])
        else:
            logic = logic & (output <= bounds[index])
    # filter outputs
    for index, output in enumerate(outputs):
        outputs[index] = output[logic]
    return tuple(outputs)


def subplots_colorbar(mappable, axes_list, gridspec_kwargs=None, **kwargs):
    fraction = kwargs.pop('fraction', 0.05)  # fraction of total width to give to colorbar
    shrink = kwargs.pop('shrink', 1.0)  # colorbar height multiplicative factor
    aspect = kwargs.pop('aspect', 20)  # colorbar height / width
    pad = kwargs.pop('pad', 0.1)  # distance between right plot and colorbar / figure width
    subplot_pad = 0.2  # pad between subplots (not colorbar and over-ridden with tight_layout)
    n_rows = axes_list[0].get_gridspec().nrows
    n_cols = axes_list[0].get_gridspec().ncols

    x1 = 1 - fraction
    width_ratios = [x1 / n_rows] * n_cols
    width_ratios.append(fraction)
    pad_s = (1 - shrink) * 0.5
    wh_ratios = [pad_s, shrink, pad_s]
    wh_space = subplot_pad * (n_rows + 1)  # subplot_pad in units of avg axes width
    gs_kwargs = {'figure': axes_list[0].figure, "wspace": wh_space, "width_ratios": width_ratios}
    if gridspec_kwargs is not None:
        gs_kwargs.update(gridspec_kwargs)
    gs = matplotlib.gridspec.GridSpec(n_rows, n_cols + 1, **gs_kwargs)
    gs2 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2,
                                                      subplot_spec=gs[:, -1],
                                                      hspace=0.,
                                                      wspace=0,
                                                      height_ratios=wh_ratios,
                                                      width_ratios=[pad - subplot_pad,
                                                                    fraction - pad + subplot_pad])
    for axes in axes_list:
        n_row = axes.get_subplotspec().rowspan.start
        n_col = axes.get_subplotspec().colspan.start
        axes.set_position(gs[n_row, n_col].get_position(axes.figure))
        axes.set_subplotspec(gs[n_row, n_col])
    cax = axes_list[0].figure.add_subplot(gs2[1, 1])
    cax.set_aspect(aspect, anchor=(0.0, 0.5), adjustable='box')
    cbar = axes_list[0].figure.colorbar(mappable, cax=cax, **kwargs)
    return cbar, gs


def dump(obj, file_name):
    with open(file_name, "wb") as f:
        try:
            import cloudpickle
            cloudpickle.dump(obj, f)
        except ImportError:
            pickle.dump(obj, f)


def load(file_name):
    with open(file_name, "rb") as f:
        try:
            import cloudpickle
        except ImportError:
            pass
        data = pickle.load(f)
    return data


def initialize_worker():
    """Initialize multiprocessing.pool worker to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt in worker process


def map_async_stoppable(pool, func, iterable, callback=None):
    results = MapResult()
    for item in iterable:
        results.append(pool.apply_async(func, (item,), callback=callback))
    return results


class MapResult(list):
    def get(self, *args, **kwargs):
        results = []
        for r in self:
            if r.ready():
                results.append(r.get(*args, **kwargs))
            else:
                results.append(None)
        return results

    def wait(self, timeout=None):
        for r in self:
            r.wait(timeout)
