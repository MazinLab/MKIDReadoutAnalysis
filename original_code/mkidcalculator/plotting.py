import warnings
import numpy as np


def plot_parameter_vs_f(parameter, f, title=None, title_kwargs=None, x_label=True, y_label=True, label_kwargs=None,
                        tick_kwargs=None, tighten=True, scatter=True, median=True, bins=30, extend=True,
                        return_bin=False, axes=None, median_kwargs=None, scatter_kwargs=None, delta=False,
                        absolute_delta=False, legend=False, legend_kwargs=None):
    """
    Plot a parameter vs frequency.
    Args:
        parameter: numpy.ndarray
            An array of parameter values.
        f: numpy.ndarray
            The frequencies corresponding to the parameter values.
        title: string (optional)
            The title for the plot. The default is None and no title is made.
            If title is True, a default title is used.
        title_kwargs: dictionary (optional)
            Keyword arguments for the title as used in axes.title(). The
            default is None and no keyword arguements are passed.
        x_label: string, boolean (optional)
            The x label for the plot. The default is True and the default label
            is used. If False, no label is used.
        y_label: string, boolean (optional)
            The y label for the plot. The default is True and the default label
            is used. If False, no label is used.
        label_kwargs: dictionary
            Keyword arguments for the axes labels in axes.set_*label(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        tick_kwargs: dictionary
            Keyword arguments for the ticks using axes.tick_params(). The
            default is None which uses the default options. Keywords in
            this dictionary override the default options.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        scatter: boolean (optional)
            Whether to plot a scatter of the data as a function of frequency.
        median: boolean (optional)
            Whether to plot the median as a function of frequency.
        bins: integer (optional)
            The number of bins to use in the median plot. The default is 30.
        extend: boolean (optional)
            Determines whether or not to extend the median data so that there
            is a bin with zero values on either side of the frequency range.
            The default is True.
        return_bin: boolean (optional)
            Whether or not to include the binned median information in the
            returned values. The default is False and only the axes are
            returned. The bin values are not returned if median is False.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        median_kwargs: dictionary (optional)
            Extra keyword arguments to send to axes.step().
        scatter_kwargs: dictionary (optional)
            Extra keyword arguments to send to axes.plot().
        delta: boolean (optional)
            If True, the difference between the parameter value and its nearest
            neighbor in frequency is plotted instead of its parameter value.
            The default is False.
        absolute_delta: boolean (optional)
            If True, the absolute difference between the parameter value and
            its nearest neighbor in frequency is plotted instead of its
            parameter value. The default is False.
        legend: boolean (optional)
            If True, the legend is displayed. The default is False.
        legend_kwargs: dictionary (optional)
            Keyword arguments for the legend in axes.legend(). The default
            is None which uses default options. Keywords in this
            dictionary override the default options.
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
            centers: numpy.ndarray
                The bin centers. Only returned if return_bin and median are
                True.
            medians: numpy.ndarray
                The median values in each bin. Only returned if return_bin and
                median are True.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    if delta or absolute_delta:
        df = np.abs(f - np.atleast_2d(f).T)
        df[df == 0] = np.nan
        nearest_neighbor = np.nanargmin(df, axis=1)
        parameter = parameter - parameter[nearest_neighbor]
        if absolute_delta:
            parameter = np.abs(parameter)
    if scatter:
        kws = {"linestyle": "none", "marker": "o", "markersize": 1}
        if scatter_kwargs is not None:
            kws.update(scatter_kwargs)
        axes.plot(f, parameter, **kws)
    if median:
        edges = np.linspace(f.min(), f.max(), bins)
        centers = 0.5 * (edges[1:] + edges[:-1])
        medians = np.empty(bins - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # median of empty slice
            for ii in range(len(medians) - 1):
                medians[ii] = np.nanmedian(parameter[(f >= edges[ii]) & (f < edges[ii + 1])])
            medians[-1] = np.nanmedian(parameter[(f >= edges[-2]) & (f <= edges[-1])])  # last bin is fully closed
        medians[np.isnan(medians)] = 0
        if extend:
            dx = centers[1] - centers[0]
            centers = np.hstack([centers[0] - dx, centers, centers[-1] + dx])
            medians = np.hstack([0, medians, 0])
        kws = {"where": "mid", "label": "median"}
        if median_kwargs is not None:
            kws.update(median_kwargs)
        axes.step(centers, medians, **kws)
    else:
        centers = None
        medians = None
    try:
        axes.set_xlim(f.min() - 0.05 * (f.max() - f.min()), f.max() + 0.05 * (f.max() - f.min()))
        axes.set_ylim(parameter.min() - 0.05 * (parameter.max() - parameter.min()),
                      parameter.max() + 0.05 * (parameter.max() - parameter.min()))
    except ValueError:
        pass  # f and/or parameter have no data
    kws = {}
    if label_kwargs is not None:
        kws.update(label_kwargs)
    if x_label is not False:
        axes.set_xlabel("frequency [GHz]" if x_label is True else x_label, **kws)
    if y_label is not False:
        axes.set_ylabel("median parameter" if y_label is True else y_label, **kws)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if title is True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            title = "median = {:g}".format(np.nanmedian(parameter))
    if title and title is not None:
        kwargs = {}
        if title_kwargs is not None:
            kwargs.update(title_kwargs)
        axes.set_title(title, **kwargs)
    if legend:
        kwargs = {"frameon": False}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        axes.legend(**kwargs)
    if tighten:
        figure.tight_layout()
    if return_bin and median:
        return axes, centers, medians
    return axes


def plot_parameter_hist(parameter, title=None, x_label=True, y_label=True, label_kwargs=None, tick_kwargs=None,
                        tighten=True, return_bin=False, axes=None, **kwargs):
    """
    Plot a parameter histogram.
    Args:
        parameter: numpy.ndarray
            An array of parameter values.
        title: string (optional)
            The title for the plot. The default is None and no title is made.
        x_label: string, boolean (optional)
            The x label for the plot. The default is True and the default label
            is used. If False, no label is used.
        y_label: string, boolean (optional)
            The y label for the plot. The default is True and the default label
            is used. If False, no label is used.
        label_kwargs: dictionary
            Keyword arguments for the axes labels in axes.set_*label(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        tick_kwargs: dictionary
            Keyword arguments for the ticks using axes.tick_params(). The
            default is None which uses the default options. Keywords in
            this dictionary override the default options.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        return_bin: boolean (optional)
            Whether or not to include the binned information in the returned
            values. The default is False and only the axes are returned.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        kwargs: optional keyword arguments
            Extra keyword arguments to send to axes.hist().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
            centers: numpy.ndarray
                The histogram bin centers. Only returned if return_bin is True.
            counts: numpy.ndarray
                The histogram bin counts. Only returned if return_bin is True.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    kws = {"bins": 50}
    if kwargs:
        kws.update(kwargs)
    counts, edges, _ = axes.hist(parameter, **kws)
    kws = {}
    if label_kwargs is not None:
        kws.update(label_kwargs)
    if x_label is not False:
        axes.set_xlabel("parameter values" if x_label is True else x_label, **kws)
    if y_label is not False:
        axes.set_ylabel("counts per bin" if y_label is True else y_label, **kws)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if title is not False and title is not None:
        axes.set_title(title)
    if tighten:
        figure.tight_layout()
    if return_bin:
        centers = 0.5 * (edges[1:] + edges[:-1])
        return axes, centers, counts
    return axes
