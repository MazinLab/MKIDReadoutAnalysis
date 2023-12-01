import os
import pickle
import logging
import numpy as np
from scipy.signal import argrelmin
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mkidcalculator.io.utils import find_resonators
from mkidcalculator.io.data import labview_segmented_widesweep

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def frequency_selector(file_name, delta_f=0.025, data=labview_segmented_widesweep,
                       indices=find_resonators, indices_kwargs=None, figure_kwargs=None, **kwargs):
    """
    A simple GUI for finding resonant frequencies in a widesweep.
    Args:
        file_name: string
            The widesweep file name.
        delta_f: float (optional)
            The frequency span to use for each window range. The default is
            0.025.
        data: object (optional)
            Function whose return value is a tuple of the frequencies
            (numpy.ndarray), complex scattering data (numpy.ndarray),
            attenuation (float), field (float), and temperature (float) of
            the widesweep.
        indices: string, iterable of integers or function (optional)
            A string may be used to specify an indices file name. Data will be
            loaded and then re-saved to this file on closing. If the file does
            not exist, no indices are assumed and the file will be created on
            closing.
            If an iterable, indices is interpreted as starting peak locations.
            If a function, it must return an iterable of resonator peak indices
            corresponding to the data returned by 'data'. The manditory
            input arguments are f, z.
        indices_kwargs: dictionary (optional)
            Keyword arguments for indices if it is a function.
        figure_kwargs: dictionary (optional)
            Keyword arguments for plt.subplots().
        kwargs: optional keyword arguments
            Extra keyword arguments to send to data.
    """
    from matplotlib import pyplot as plt
    # get output file name
    if isinstance(indices, str):
        output_file = indices
    else:
        output_file = ".".join(file_name.split('.')[:-1])
        while os.path.isfile(output_file + ".p"):
            output_file += "_new"
        output_file += ".p"
    # get data from file
    f, z, attenuation, field, temperature = data(file_name, **kwargs)
    magnitude = 20 * np.log10(np.abs(z))
    # find peaks if we were given a method or a file
    peaks = np.full(f.shape, False)
    peak_handles = np.full(f.shape, None)
    if isinstance(indices, str) and os.path.isfile(output_file):
        with open(output_file, "rb") as file_:
            indices = pickle.load(file_)
    elif indices is not None and callable(indices):
        kws = {}
        if indices_kwargs is not None:
            kws.update(indices_kwargs)
        indices = indices(f, z, **kws)
    else:
        indices = np.zeros(f.shape, dtype=bool)
    peaks[np.array(indices)] = True

    # setup figure
    kws = {"figsize": (15, 5)}
    if figure_kwargs is not None:
        kws.update(figure_kwargs)
    figure, axes = plt.subplots(**kws)
    axes.plot(f, magnitude)
    axes.set_title("left click: add frequency, right click: remove frequency, saves data on close")
    axes.set_xlabel("frequency")
    axes.set_ylabel(r"|S$_{21}$|")
    axes.set_xlim(np.min(f) - delta_f / 2, np.min(f) + delta_f / 2)
    min_data_height = 5
    height = 0.1
    pad = 0.05
    figure.tight_layout(rect=(0, 2 * (height + pad), 1, 1))
    figure.canvas.draw()

    # setup buttons
    ax_color = 'lightgoldenrodyellow'
    bbox = axes.get_tightbbox(renderer=figure.canvas.renderer)
    bbox = bbox.transformed(axes.transAxes.inverted())
    y_min = bbox.extents[1]  # bottom of axes including y label
    ax_slider = inset_axes(axes, width="100%", height="100%", axes_kwargs={'facecolor': ax_color},
                           bbox_to_anchor=(0, y_min - pad - height, 1, height),
                           bbox_transform=axes.transAxes, loc=4, borderpad=0)
    slider = Slider(ax_slider, '', np.min(f), np.max(f), valinit=np.min(f), valstep=delta_f / 2)
    slider.valtext.set_ha('right')
    slider.valtext.set_va('center')
    slider.valtext.set_position((0.99, 0.5))

    width = 0.2
    ax_next = inset_axes(axes, width="100%", height="100%", axes_kwargs={'facecolor': ax_color},
                         bbox_to_anchor=(1 - width, y_min - 2 * (pad + height), width, height),
                         bbox_transform=axes.transAxes, loc=4, borderpad=0)
    button_next = Button(ax_next, 'next', color=ax_color, hovercolor='0.975')
    ax_previous = inset_axes(axes, width="100%", height="100%", axes_kwargs={'facecolor': ax_color},
                             bbox_to_anchor=(1 - 2 * width - pad, y_min - 2 * (pad + height), width, height),
                             bbox_transform=axes.transAxes, loc=4, borderpad=0)
    button_previous = Button(ax_previous, 'previous', color=ax_color, hovercolor='0.975')

    # define updating functions
    def add_vline(index, value):
        if peak_handles[index] is None:
            peak_handles[index] = axes.axvline(value, color='C2')

    def update_frequency(value):
        # set x lim
        axes.set_xlim(value - delta_f / 2, value + delta_f / 2)
        try:
            d = magnitude[(f >= value - delta_f / 2) & (f <= value + delta_f / 2)]
            axes.set_ylim(d.min() if d.max() - d.min() > min_data_height else d.max() - min_data_height, d.max())
        except ValueError:  # data out of range
            pass

    def previous_frequency(event):
        value = slider.val - delta_f / 2
        slider.set_val(value)
        update_frequency(value)

    def next_frequency(event):
        value = slider.val + delta_f / 2
        slider.set_val(value)
        update_frequency(value)

    def key_press(event):
        if event.key == "right" or event.key == "d":
            next_frequency(event)
        elif event.key == "left" or event.key == "a":
            previous_frequency(event)

    def closest_index(value):
        peak_init = np.argmin(np.abs(value - f))
        peak = argrelmin(magnitude[peak_init - 5: peak_init + 5])[0][:1]  # first dimension only first item
        peak = peak_init - 5 + peak[0] if peak else None
        return peak

    def is_close(value):
        index = np.argmin(np.abs(value - f))
        close_indices = np.r_[index - 5: index + 5]
        close_indices = close_indices[close_indices >= 0]
        return close_indices

    def add_frequency(value):
        index = closest_index(value)
        if index is not None and not peaks[index]:
            log.info("adding peak at frequency: {:f}".format(f[index]))
            peaks[index] = True
            add_vline(index, f[index])
            figure.canvas.draw()

    def remove_frequency(value):
        close_indices = is_close(value)
        subset = peak_handles[close_indices]
        for index, handle in enumerate(subset):
            if handle is not None:
                handle.remove()
                peak_handles[close_indices[index]] = None
                del handle
                figure.canvas.draw()
                break  # only remove one
        subset = peaks[close_indices]
        for index, peak in enumerate(subset):
            if peak:
                log.info("removing peak at frequency: {:f}".format(f[close_indices[index]]))
                peaks[close_indices[index]] = False
                break

    def click(event):
        if not event.inaxes == axes:
            return
        if event.button == 1:  # left click
            add_frequency(event.xdata)
        elif event.button == 3:  # right click
            remove_frequency(event.xdata)

    def close(event):
        frequency_indices = np.unique(np.nonzero(peaks)[0])  # first dimension and unique
        with open(output_file, "wb") as f_:
            pickle.dump(frequency_indices, f_)
        log.info("data saved to '{:s}'".format(output_file))

    # initialize data to plot
    update_frequency(f.min())
    for ind, p in enumerate(peaks):
        if p:
            add_vline(ind, f[ind])

    # connect buttons to function
    slider.on_changed(update_frequency)
    button_next.on_clicked(next_frequency)
    button_previous.on_clicked(previous_frequency)
    figure.canvas.mpl_connect('key_press_event', key_press)
    figure.canvas.mpl_connect('button_press_event', click)
    figure.canvas.mpl_connect('close_event', close)
    plt.show(block=True)
