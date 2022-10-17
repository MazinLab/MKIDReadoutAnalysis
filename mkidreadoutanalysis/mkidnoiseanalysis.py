import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import welch, lfilter


def swenson_formula(y0, a, increasing: bool):
    """doi: 10.1063/1.4903855"""
    if a == 0:
        return y0
    y0 = np.atleast_1d(y0)
    y = np.empty_like(y0)
    for i, y0_i in enumerate(y0):
        roots = np.roots([4, -4 * y0_i, 1, -(y0_i + a)])
        if increasing:
            y[i] = np.min(roots[np.isreal(roots)].real)
        else:
            y[i] = np.max(roots[np.isreal(roots)].real)
    return y


def plot_psd(data, fs=2e6, fres=1e3, ax=None, fig=None, **kwargs):
    plt.figure()
    default = {'fs': fs, 'nperseg': fs / fres}
    default.update(kwargs)
    f, psd = welch(data, **default)
    plt.semilogx(f, 10 * np.log10(psd))
    plt.xlabel(f'Frequency [Hz] ({fres * 1e-3:g} kHz resolution)')
    plt.ylabel('dB/Hz')
    plt.grid()
    plt.title('Power Spectral Density')
    # add axis later (include res)


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


def compute_r(amplitudes, plot=False):
    # Using Scott's method to compute bandwidth but with MAD
    sigma = amplitudes.std(ddof=1)
    sigma_mad = stats.median_abs_deviation(amplitudes, scale='normal')
    bandwidth = amplitudes.size ** (-1 / 5) * sigma_mad

    # Compute the PDF using a KDE and then converting to a spline
    maximum, minimum = amplitudes.max(), amplitudes.min()
    x = np.linspace(minimum, maximum,  # sample at 100x the bandwidth
                    int(100 * (maximum - minimum) / bandwidth))
    pdf_kde = stats.gaussian_kde(amplitudes, bw_method=bandwidth / sigma)
    pdf_data = pdf_kde(x)
    pdf = InterpolatedUnivariateSpline(x, pdf_data, k=3, ext=1)

    # compute the maximum of the distribution
    pdf_max = 0
    max_location = 0
    for root in quadratic_spline_roots(pdf.derivative()):
        if pdf(root) > pdf_max:
            pdf_max = pdf(root)
            max_location = root.item()
    if pdf_max == 0 and max_location == 0:
        raise RuntimeError("Could not find distribution maximum.")

    # compute the FWHM
    pdf_approx_shifted = InterpolatedUnivariateSpline(
        x, pdf_data - pdf_max / 2, k=3, ext=1)

    roots = pdf_approx_shifted.roots()
    if roots.size >= 2:
        indices = np.argsort(np.abs(roots - max_location))
        roots = roots[indices[:2]]
        fwhm = roots.max() - roots.min()
    else:
        raise ValueError("Could not find distribution FWHM")

    if plot:
        plt.figure()
        plt.hist(amplitudes, bins='auto', density=True)
        x = np.linspace(amplitudes.min(), amplitudes.max(), 1000)
        plt.plot(x, pdf(x))
        plt.xlabel('Phase peak (need to change this to energy?)')
        plt.ylabel('counts')

    return -max_location / fwhm


def plot_channel_fft(data, fs):
    """Plot the power spectrum in dB of a channel"""
    fft_data = 20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(data))))
    plt.plot(np.linspace(-fs / 2, fs / 2, int(data.size)), fft_data - fft_data.max())
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')


def gen_line_noise(freqs, amps, phases, n_samples, fs):
    """
    Generate time series representing line noise in a single MKID coarse channel (MKID has been centered).
    @param freqs: 1D np.array or list
        frequencies of line noise
    @param amps: 1D np.array or list
        amplitudes of line noise
    @param phases: 1D np.array or list
        phases of line noise
    @param n_samples: int
        number of timeseries samples to produce
    @param fs: float
        sample rate of channel in Hz
    @return:
    """
    freqs = np.asarray(freqs)  # Hz and relative to center of bin (MKID we are reading out)
    amps = np.asarray(amps)
    phases = np.asarray(phases)

    n_samples = n_samples
    sample_rate = fs

    line_noise = np.zeros(n_samples, dtype=np.complex64)
    t = 2 * np.pi * np.arange(n_samples) / sample_rate
    for i in range(freqs.size):
        phi = t * freqs[i]
        exp = amps[i] * np.exp(1j * (phi + phases[i]))
        line_noise += exp
    return line_noise


def apply_lowpass_filter(coe, data):
    """

    @param coe: 1D np.array
        lowpass filter coefficients
    @param data: 1D np.array
        data to be filtered
    """

    return lfilter(coe, 1, data)
