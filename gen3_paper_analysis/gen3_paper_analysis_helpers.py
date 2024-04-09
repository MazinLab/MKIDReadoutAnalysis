import numpy.typing as nt
import os
import numpy as np
import subprocess
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import welch, lfilter
from mkidreadoutanalysis.mkidro import MKIDReadout
from mkidreadoutanalysis.mkidnoiseanalysis import quadratic_spline_roots


def get_dac_fft(filedir, filename, noise):
    data = np.load(os.path.join(filedir, filename))
    waveform = data['dac_waveform']
    waveform_noise = waveform + noise
    waveform_fft = np.abs(np.fft.fftshift(np.fft.fft(waveform_noise)))
    return waveform_fft

def get_data(filedir: str, filename: str) -> nt.NDArray:
    n_files = int(
        subprocess.check_output(f"ls -d {os.path.join(filedir, filename) + '*'} -1 | wc -l", shell=True, text=True)[
        :-1])
    if n_files == 0:
        raise FileNotFoundError('No files found for given dir and fname.')
    fnames = []
    for suffix in range(0, n_files):
        fname = filename + f'{suffix:002d}' + '.npz'
        fnames.append(os.path.join(filedir, fname))

    tmp = np.load(fnames[0])
    n_cap_points = tmp['phase_data'].size
    phase = np.empty(n_cap_points * n_files)
    for i, f in enumerate(fnames):
        data = np.load(f)
        phase[i * n_cap_points: (i + 1) * n_cap_points] = data['phase_data']
    del data
    return phase


def get_energies(phase_data, dark_data, threshold, holdoff):
    phase_readout = MKIDReadout()
    phase_readout.trigger(phase_data, fs=1e6, threshold=threshold, deadtime=holdoff)
    return phase_readout.photon_energies - dark_data.mean()


def fit_histogram(normalized_energies):
    amplitudes = normalized_energies
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
    return -max_location, fwhm, pdf


def compute_pdfs(energies):
    pdfs = []
    for i in energies:
        _, _, pdf = fit_histogram(i)
        pdfs.append(pdf)
    return pdfs


def get_processed_data(filedir, fname):
    data = np.load(os.path.join(filedir, fname))
    phase_dist_centers = data['phase_dist_centers']
    raw_r = data['raw_r']
    pdf_x = data['pdf_x']
    pdf_y = data['pdf_y']
    return phase_dist_centers, raw_r, pdf_x, pdf_y


def get_energy_hist_points(filedir, filename, colors):
    phase_dist_centers = np.empty(3)
    phase_dist_fwhm = np.empty(3)
    normalized_energies = []
    pdfs_x = []
    pdfs_y = []

    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color+'_')
        fname += '_processed.npz'
        data = np.load(os.path.join(filedir, fname))
        normalized_energies.append(data['normalized_energies'])
        phase_dist_centers[i] = -data['max_location']
        phase_dist_fwhm[i] = data['fwhm']
        pdfs_x.append(data['pdf_x'])
        pdfs_y.append(data['pdf_y'])

    lasers = np.array([405.9, 663.1, 808.0])
    hc = 1240  # eV
    energies = hc / lasers  # eV

    slope = np.polyfit(phase_dist_centers, energies, 1)[0]
    intercept = np.polyfit(phase_dist_centers, energies, 1)[1]
    x = np.linspace(-3.5, -1.5, 10)
    y = slope * x + intercept
    delta_e = slope * phase_dist_fwhm
    raw_r = -energies / delta_e

    return phase_dist_centers, raw_r, normalized_energies, pdfs_x, pdfs_y


def place_annotations(pdf_x, pdf_y, phase_dist_centers, raw_r, colors, ax):
    for i in range(len(pdf_x)):
        y = pdf_y[i].max()
        x = phase_dist_centers[i]

        ax.annotate(f'R={np.round(raw_r[i])}',
                    xy=(x, y), xycoords='data',
                    xytext=(-30, 10), textcoords='offset points', fontsize=16, color=colors[i])


def plot_dac_output(ax, waveform_fft, max_val):
    waveform_fft = waveform_fft
    yticks = np.linspace(0, -100, 11)
    ylabels = [str(x) + ' dB' for x in yticks]
    ylabels[0] = 'DAC Max Output'
    freqs = np.linspace(-2048,2048,2**19)
    ax.plot(freqs, 20*np.log10(waveform_fft)-max_val, linewidth=4, color='#04AA25')
    ax.set_yticks(yticks, labels=ylabels, fontsize=18)
    ax.grid()
    ax.set_ylim([-100, None])
    ax.set_xlabel('Frequency [MHz]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)

def make_r_hist_plt(ax, phase_dist_centers, raw_r, pdf_x, pdf_y):
    xticks = np.sort(np.concatenate((np.round(phase_dist_centers, decimals=1), np.array([-np.pi, -np.pi / 2]))))

    xlabels = []
    custom_labels = [r'-$\pi$', r'-$\pi$/2']
    for i in xticks:
        if i == -np.pi:
            xlabels.append(custom_labels[0])
        elif i == -np.pi / 2:
            xlabels.append(custom_labels[1])
        else:
            xlabels.append(str(i))

    ax.set_xticks(xticks, labels=xlabels, fontsize=14)

    place_annotations(pdf_x, pdf_y, phase_dist_centers, raw_r, ['#0015B0', '#AB1A00', '#AF49A0'], ax)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(pdf_x[0], pdf_y[0], marker='o', markevery=5, markerfacecolor='#0015B0', markeredgecolor='#0015B0', color='blue',
            linewidth=3)
    ax.fill_between(pdf_x[0], pdf_y[0], 0, color='blue', alpha=0.7)
    ax.plot(pdf_x[1], pdf_y[1], marker='d', markevery=5, markerfacecolor='#AB1A00', markeredgecolor='#AB1A00', color='red',
            linewidth=3)
    ax.fill_between(pdf_x[1], pdf_y[1], 0, color='red', alpha=0.7)
    ax.plot(pdf_x[2], pdf_y[2], marker='v', markevery=5, markerfacecolor='#AF49A0', markeredgecolor='#AF49A0',
            color='lightcoral', linewidth=3)
    ax.fill_between(pdf_x[2], pdf_y[2], 0, color='lightcoral', alpha=0.7)


    ax.set_xlabel('Phase', fontsize=18)
    ax.set_xlim([-np.pi, -1.5])