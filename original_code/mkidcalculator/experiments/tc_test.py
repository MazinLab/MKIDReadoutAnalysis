import calendar
import numpy as np
from datetime import datetime
from scipy.signal import detrend
from skimage.util import view_as_windows
from scipy.stats import median_abs_deviation
from scipy.interpolate import UnivariateSpline, interp1d
from mkidcalculator.io.utils import setup_axes, finalize_axes


def load_lakeshore_log(filename, start=None, stop=None, timestamp=False,
                       absolute=True, remove_outliers=False, nsigma=None):
    """
    Load the log file and return the times and values.
    Args:
        filename: string or list of strings
            The log filename(s) including the path.
        start: datetime.datetime
            The start time for the experiment. If not provided, the whole
            log file is used.
        stop: datetime.datetime
            The stop time for the experiment. If not provided, the whole
            log file is used.
        timestamp: boolean
            If True, the dates are returned as timestamps. The default is
            False and datetime objects are returned.
        absolute: boolean
            The default is True and the absolute value of the data is
            returned. If False, the original data is returned instead.
        remove_outliers: boolean
            Remove outliers from the data if True. The default is False.
        nsigma: float
            If 'remove_outliers' is True, specifying 'nsigma' will mask
            the detrended data outside nsigma. If 'remove_outliers' is
            True, but 'nsigma' is not specified only extremely bad values
            are removed.
    Returns:
        time: numpy.ndarray
            Times corresponding to the values.
        values: numpy.ndarray
            Values at the corresponding times.
    """
    # Load the file.
    if isinstance(filename, str):
        filename = [filename]
    data = []
    for name in filename:
        data.append(np.loadtxt(name, dtype=np.object, delimiter=","))
    data = np.vstack(data)

    # Reformat the time into a datetime.
    time_strings = [t[0].strip() + " " + t[1].strip() for t in data]
    fmt = "%d-%m-%y %H:%M:%S"
    time = np.array([datetime.strptime(t, fmt) for t in time_strings])
    # Coerce the values into floats.
    values = data[:, 2].astype(float)
    # Mask the data.
    mask = np.ones(time.shape, dtype=bool)
    if start is not None:
        mask *= (time >= start)
    if stop is not None:
        mask *= (time <= stop)
    time = time[mask]
    values = values[mask]
    # Convert times to timestamps for easy manipulation.
    if timestamp:
        time = np.array([t.timestamp() for t in time])
    # Remove outliers.
    if remove_outliers:
        # Get rid of values greater than nsigma from the detrended data.
        if nsigma is not None:
            v = detrend(values)
            mask = np.abs(v) < nsigma * median_abs_deviation(v)
        # Get rid of large values of the opposite sign w.r.t the measurement.
        else:
            sign = np.sign(np.median(values))
            v_max = np.max(np.abs(values[np.sign(values) == sign]))
            mask = np.abs(values) < v_max
        values = values[mask]
        time = time[mask]
    # Return absolute values.
    if absolute:
        values = np.abs(values)
    return time, values


def t_vs_r(t_filename, r_filename, start=None, stop=None, interp='smooth',
           t_kwargs=None, r_kwargs=None):
    """
    Return the resistance as a function of temperature.
    Args:
        t_filename: string or list of strings
            The filename(s) for the temperature log data.
        r_filename: string or list of strings
            The filename(s) for the resistance log data.
        start: datetime.datetime
            The start time for the experiment. If not provided, the whole
            log file is used.
        stop: datetime.datetime
            The stop time for the experiment. If not provided, the whole
            log file is used.
        interp: str
            If interp is 'smooth', a smoothing spline is used. If
            'linear', a linear interpolation is used.
        t_kwargs: dictionary
            Keyword arguments to use when loading the temperature with
            the load_lakeshore_log() function.
        r_kwargs: dictionary
            Keyword arguments to use when loading the resistance with
            the load_lakeshore_log() function.
    Returns:
        temperature: numpy.ndarray
            Temperature at the same time as the resistance.
        resistance: numpy.ndarray
            Resistance at the same time as the temperature.
    """
    # Load the files.
    kwargs = {'remove_outliers': True, 'nsigma': 10}
    if t_kwargs is not None:
        kwargs.update(t_kwargs)
    time_t, t = load_lakeshore_log(t_filename, start=start, stop=stop,
                                   timestamp=True, **kwargs)
    kwargs = {'remove_outliers': True}
    if r_kwargs is not None:
        kwargs.update(r_kwargs)
    time_r, resistance = load_lakeshore_log(r_filename, start=start, stop=stop,
                                            timestamp=True, **kwargs)
    # Get the temperature when the resistance was measured by interpolation.
    if interp.lower().startswith('linear'):
        time_to_t = interp1d(time_t, t, fill_value='extrapolate')
    elif interp.lower().startswith('smooth'):
        sigma = np.pad(np.std(detrend(view_as_windows(t, 40), axis=-1),
                              ddof=1, axis=-1), (20, 19), 'edge')
        time_to_t = UnivariateSpline(time_t, t, w=1 / sigma
                                     if (sigma != 0).all() else None)
    else:
        raise ValueError("Invalid value given to 'interp' keyword argument.")
    temperature = time_to_t(time_r)
    return temperature, resistance


def plot_transition(t_filename, r_filename, plot_kwargs=None, x_label=None,
                    y_label=None, label_kwargs=None, legend=False,
                    legend_kwargs=None, title=False, title_kwargs=None,
                    tick_kwargs=None, tighten=True, axes=None, **kwargs):
    """
    Plot the transition data.
    Args:
        t_filename: string or list of strings
            The filename(s) for the temperature log data.
        r_filename: string or list of strings
            The file name(s) for the resistance log data.
        x_label: string
            The label for the x axis. The default is None which uses the
            default label. If x_label evaluates to False, parameter_kwargs
            is ignored.
        y_label: string
            The label for the y axis. The default is None which uses the
            default label. If y_label evaluates to False, parameter_kwargs
            is ignored.
        plot_kwargs: dictionary
            Keyword arguments for the data in axes.plot(). The default is
            None which uses default options. Keywords in this dictionary
            override the default options.
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
        kwargs: optional keyword arguments
            These keyword arguments are set to the t_vs_r() function.
    Returns:
        axes: matplotlib.axes.Axes class
            An Axes class with the plotted loop.
    """
    t, r = t_vs_r(t_filename, r_filename, **kwargs)
    _, axes = setup_axes(axes, x_label, y_label, label_kwargs,
                         'T [mK]', r'R [$\Omega$]')
    kwargs = {"marker": 'o', "markersize": 4, "linestyle": "none",
              "markerfacecolor": "k", "markeredgecolor": "none"}
    if plot_kwargs is not None:
        kwargs.update(plot_kwargs)
    axes.plot(t * 1e3, r, **kwargs)
    title = "superconducting transition" if title is True else title
    finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend,
                  legend_kwargs=legend_kwargs, tick_kwargs=tick_kwargs,
                  tighten=tighten)
    return axes
