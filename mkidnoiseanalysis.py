import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import welch


def swenson_formula(y0, a, increasing: bool):
    """doi: 10.1063/1.4903855"""
    y0 = np.atleast_1d(y0)
    y = np.empty_like(y0)
    for i, y0_i in enumerate(y0):
        roots = np.roots([4, -4 * y0_i, 1, -(y0_i + a)])
        if increasing:
            y[i] = np.min(roots[np.isreal(roots)].real)
        else:
            y[i] = np.max(roots[np.isreal(roots)].real)
    return y


def gen_amp_noise(points, snr):
    """ Flat PSD, white-noise generated from voltage fluctuations"""
    a_noise = 10 ** ((20 * np.log10(1 / np.sqrt(2)) - snr) / 10);  # input dBm of noise
    amp_noise = np.sqrt(a_noise) * np.rng.normal(points)
    return amp_noise


# def gen_tls_noise(points, fs, scale=1e-3, fr=6e9, q=15e3):
#    """ two-level system noise"""
#    psd_freqs = np.fft.rfftfreq(points, d=1 / fs)
#    fc = fr / (2 * q)
#    psd = np.zeros_like(psd_freqs)
#    nonzero = psd_freqs != 0
#    psd[nonzero] = scale / (1 + (psd_freqs[nonzero] / fc) ** 2) / psd_freqs[nonzero]
#    noise_phi = 2 * np.pi * np.rng.random(psd_freqs.size)
#    noise_fft = np.exp(1j * noise_phi)
#    # rescale the noise to the covariance
#    a = np.sqrt(points * psd * fs / 2)
#    noise_fft = a * noise_fft
#    tls_noise = np.fft.irfft(noise_fft, points)
#    return tls_noise


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
