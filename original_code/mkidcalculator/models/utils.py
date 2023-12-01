import logging
import numpy as np
from scipy.signal import detrend
from skimage.util import view_as_windows

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def scaled_alpha_inv(s_alpha):
    return np.arctan(s_alpha) / np.pi + 1 / 2


def scaled_alpha(alpha):
    return np.tan(np.pi * (alpha - 1 / 2))


def bandpass(data):
    fft_data = np.fft.rfft(data)
    fft_data[:, 0] = 0
    indices = np.array([np.arange(fft_data[0, :].size)] * fft_data[:, 0].size)
    f_data_ind = np.argmax(np.abs(fft_data), axis=-1)[:, np.newaxis]
    fft_data[np.logical_or(indices < f_data_ind - 1, indices > f_data_ind + 1)] = 0
    data_new = np.fft.irfft(fft_data, data[0, :].size)
    return data_new, f_data_ind


def _compute_sigma(z):
    eps_real = np.std(detrend(view_as_windows(z.real, 10), axis=-1), ddof=1, axis=-1)
    eps_imag = np.std(detrend(view_as_windows(z.imag, 10), axis=-1), ddof=1, axis=-1)
    # We use the minimum standard deviation because data with multiple and/or saturated resonators can corrupt more
    # than half of the measured data using this method.
    index = np.argmin(np.sqrt(eps_real**2 + eps_imag**2))
    eps_real = eps_real[index]
    eps_imag = eps_imag[index]
    # make sure there are no zeros
    if eps_real == 0:
        log.warning("zero variance calculated and set to 1 when detrending I data")
        eps_real = 1
    if eps_imag == 0:
        log.warning("zero variance calculated and set to 1 when detrending Q data")
        eps_imag = 1
    return eps_real + 1j * eps_imag
