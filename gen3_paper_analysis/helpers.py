import numpy.typing as nt
import os
import numpy as np
import subprocess
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import welch, lfilter, savgol_filter

from mkidreadoutanalysis.mkidro import MKIDReadout
from mkidreadoutanalysis.mkidnoiseanalysis import quadratic_spline_roots
from mkidreadoutanalysis.optimal_filters.make_filters import Calculator
from mkidreadoutanalysis.optimal_filters.config import ConfigThing
import matplotlib.pyplot as plt



def get_dac_fft(filedir, filename, noise):
    data = np.load(os.path.join(filedir, filename))
    waveform = data['dac_waveform']
    waveform_noise = waveform + noise
    waveform_fft = np.abs(np.fft.fftshift(np.fft.fft(waveform_noise)))
    return waveform_fft

def get_data(filedir: str, filename: str, skip=2) -> nt.NDArray:
    n_files = int(
        subprocess.check_output(f"ls -d {os.path.join(filedir, filename) + '*'} -1 | wc -l", shell=True, text=True)[
        :-1])
    if skip:
        n_files -= skip
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


def fit_univariate_spline(x, pdf_data, k=3, ext=1):
    pdf = InterpolatedUnivariateSpline(x, pdf_data, k=k, ext=ext)
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


def estimate_pdf(normalized_energies):
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
    return fit_univariate_spline(x, pdf_data, k=3, ext=1)



def compute_pdfs(energies):
    pdfs = []
    for i in energies:
        _, _, pdf = estimate_pdf(i)
        pdfs.append(pdf)
    return pdfs


def get_processed_data(filedir, fname):
    data = np.load(os.path.join(filedir, fname))
    phase_dist_centers = data['phase_dist_centers']
    raw_r = data['raw_r']
    pdf_x = data['pdf_x']
    pdf_y = data['pdf_y']
    return phase_dist_centers, raw_r, pdf_x, pdf_y

def compute_ofilt(phase_data):
    cfg = ConfigThing()
    cfg.registerfromkvlist((('dt', 1e-6),
                            ('fit', True),
                            ('summary_plot', True),
                            ('pulses.unwrap', False),
                            ('pulses.fallback_template', 'default'),
                            ('pulses.tf', 60),
                            ('pulses.ntemplate', 500),
                            # need to set this larger to calculate covariance matrix in the time domain "accurately" for the number of selected filter coefficients
                            ('pulses.offset', 10),
                            ('pulses.threshold', 6),  # sigma above noise
                            ('pulses.separation', 500),
                            ('pulses.min_pulses', 1000),
                            ('noise.nwindow', 1000),  # 1000
                            ('noise.isolation', 200),
                            ('noise.max_windows', 2000),
                            # maximum number of nwindows of samples needed before moving on [int]
                            ('noise.max_noise', 5000),  # 2000
                            ('template.percent', 80),
                            ('template.min_tau', 5),
                            ('template.max_tau', 100),
                            ('template.fit', 'triple_exponential'),
                            ('filter.cutoff', .1),
                            ('filter.filter_type', 'wiener'),
                            ('filter.nfilter', 30),
                            # for messing around this should be closer to 1000 and ntemplate should be increased to be 5-10x nfilter
                            # need to make sure filter is periodic and this gets hard when the filter is short
                            ('filter.normalize', True)), namespace='')

    ofc = Calculator(phase=phase_data, config=cfg, name='simulated')

    ofc.calculate(clear=False)

    return ofc


def redo_hist_energy(filedir, filename, phase_dist_centers, normalized_phases, plot=False):
    fprocessedname = os.path.join(filedir, filename) + '_ecal_processed_nm.npz'
    try:
        data = np.load(fprocessedname)
        energy_dist_centers = data['energy_dist_centers']
        fwhm = data['fwhm']
        pdfs_x = data['pdfs_x']
        pdfs_y = data['pdfs_y']
        return energy_dist_centers, fwhm, pdfs_x, pdfs_y

    except FileNotFoundError:
        compute = True
        pass

    lasers = np.array([405.9, 663.1, 808.0])
    hc = 1240  # eV
    lzr_energies = hc / lasers  # eV
    phase_dist_centers = np.append(phase_dist_centers, 0)
    lzr_energies = np.append(lzr_energies, 0)

    ecal = np.poly1d(np.polyfit(phase_dist_centers, lzr_energies, 2))
    x = np.linspace(-3.5, 0, 10)
    y = ecal(x)
    if plot:
        plt.plot(phase_dist_centers, lzr_energies, 'o', label='data')
        plt.plot(x, y, label='quadratic fit')
        plt.legend(loc='upper right')
        plt.xlabel('Phase (radians)')
        plt.ylabel('Energy (eV)')
        plt.title("Energy Calibration")
        plt.show()

    if compute:
        energy_dist_centers = np.empty(3)
        fwhm = np.empty(3)
        pdfs_x = []
        pdfs_y = []
        for i, x in enumerate(normalized_phases):
            energies = ecal(x)
            energies_nm = 1239.8/energies
            energy_dist_centers[i], fwhm[i], pdf = estimate_pdf(energies_nm)
            plt.hist(energies_nm, bins='auto')
            plt.show()
            pdf_x = np.linspace(energies_nm.min(), energies_nm.max(), 1000)
            pdf_y = pdf(pdf_x)
            if i == 0:
                yhat = savgol_filter(savgol_filter(pdf_y, 81, 3), 31, 3)
            elif i == 1:
                yhat = savgol_filter(savgol_filter(pdf_y, 221, 3), 291, 3)
            else:
                yhat = savgol_filter(savgol_filter(savgol_filter(pdf_y, 351, 3), 351, 3), 301, 3)
            plt.plot(pdf_x, yhat)
            plt.show()
            pdfs_x.append(pdf_x)
            pdfs_y.append(yhat)
        np.savez(fprocessedname, energy_dist_centers=np.abs(energy_dist_centers), fwhm=fwhm, pdfs_x=pdfs_x,
                 pdfs_y=pdfs_y)
    raw_r = (1239.8/lzr_energies[:-1]) / fwhm
    return energy_dist_centers, fwhm, pdfs_x, pdfs_y

def get_energy_hist_points(filedir, filename, colors, advanced=True):
    dist_centers = np.empty(3)
    dist_fwhm = np.empty(3)
    normalized_phases = []
    pdfs_x = []
    pdfs_y = []

    for i, color in enumerate(colors):
        fname = filename.replace('phase_', 'phase_' + color+'_')
        fname += '_processed.npz'
        data = np.load(os.path.join(filedir, fname))
        normalized_phases.append(data['normalized_energies'])
        dist_centers[i] = -data['max_location']
        dist_fwhm[i] = data['fwhm']
        pdfs_x.append(data['pdf_x'])
        pdfs_y.append(data['pdf_y'])

    # compute R
    lasers = np.array([405.9, 663.1, 808.0])
    hc = 1240  # eV
    lzr_energies = hc / lasers  # eV

    if advanced:
        dist_centers_nm, fwhm_nm, pdfs_x, pdfs_y = redo_hist_energy(filedir, filename, dist_centers, normalized_phases, plot=False)
        lzr_energies_nm = 1239.8/lzr_energies
        raw_r = lzr_energies_nm / fwhm_nm
        return dist_centers_nm, raw_r, pdfs_x, pdfs_y

    else:
        ecal = np.poly1d(np.polyfit(dist_centers, lzr_energies, 1))
        x = np.linspace(-3.5, -1.5, 10)
        y = ecal(x)
        delta_e = ecal[1] * dist_fwhm
        raw_r = -lzr_energies / delta_e

    return dist_centers, raw_r, pdfs_x, pdfs_y


def place_annotations(pdf_x, pdf_y, dist_centers, raw_r, colors, rxoffset, ryoffset, rlblsz, ax):
    for i in range(len(pdf_x)):
        y = pdf_y[i].max()
        x = dist_centers[i]

        ax.annotate(f'R={np.round(raw_r[i])}',
                    xy=(x, y), xycoords='data',
                    xytext=(-rxoffset, ryoffset), textcoords='offset points', fontsize=rlblsz, color=colors[i])


def plot_dac_output(ax, waveform_fft, max_val):
    waveform_fft = waveform_fft
    yticks = np.linspace(0, -100, 11)
    ylabels = [str(x) + ' dB' for x in yticks]
    ylabels[0] = 'DAC Max Output'
    freqs = np.linspace(-2048,2048,2**19)
    ax.plot(freqs, 20*np.log10(waveform_fft)-max_val, linewidth=4, color='#04AA25')
    ax.set_yticks(yticks, labels=ylabels, fontsize=8)
    ax.grid()
    ax.set_ylim([-100, None])
    ax.set_xlabel('Frequency [MHz]', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)


def set_x_tick_label(dist_centers, phase, fwhm=None):
    if phase:
        xticks = np.sort(np.concatenate((np.round(dist_centers, decimals=1), np.array([-np.pi, -np.pi / 2]))))

        xlabels = []
        custom_labels = [r'-$\pi$', r'-$\pi$/2']
        for i in xticks:
            if i == -np.pi:
                xlabels.append(custom_labels[0])
            elif i == -np.pi / 2:
                xlabels.append(custom_labels[1])
            else:
                xlabels.append(str(i))
    else:
        centers_nm = 1239.8 / dist_centers
        xticks = np.sort(np.concatenate((np.round(centers_nm, decimals=1), np.array([300.0, 900.0]))))
        xlabels = ['300.0', '405.9', '633.1', '808.0', '900.0']
    return xticks, xlabels


def make_r_hist_plt(ax, dist_centers, raw_r, pdf_x, pdf_y,  rxoffset, ryoffset, rlblsz, xoffset=None, phase=False):
    if phase:
        if xoffset:
            dist_centers = [x - xoffset for x in dist_centers]
            pdf_x = [x - xoffset for x in pdf_x]

        xticks, xlabels = set_x_tick_label(dist_centers, phase)
        ax.set_xticks(xticks, labels=xlabels, fontsize=8)
        ax.set_xlabel('Phase', fontsize=12)
        ax.set_xlim([-np.pi, -1.5])
    else:
        ax.set_xlim(350, 1050)
        ax.set_xlabel('Wavelength [nm]', fontsize=12)

    place_annotations(pdf_x, pdf_y, dist_centers, raw_r, ['#0015B0', '#AC5818', '#9C2419'], rxoffset, ryoffset, rlblsz, ax)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.plot(pdf_x[0], pdf_y[0], marker='o', markevery=20, markersize=3, markerfacecolor='#0015B0', markeredgecolor='#0015B0', color='blue',
            linewidth=2, label='Blue 405.9 nm')
    ax.fill_between(pdf_x[0], pdf_y[0], 0, color='blue', alpha=0.7)
    ax.plot(pdf_x[1], pdf_y[1], marker='d', markevery=20, markersize=3, markerfacecolor='#FF9300', markeredgecolor='#FF9300', color='#FF9300',
            linewidth=2, label='Red 663.1 nm')
    ax.fill_between(pdf_x[1], pdf_y[1], 0, color='#FF9300', alpha=0.7)
    ax.plot(pdf_x[2], pdf_y[2], marker='v', markevery=20, markersize=3, markerfacecolor='#AB1A00', markeredgecolor='#AB1A00',
            color='red', linewidth=2, label='IR 808.0 nm')
    ax.fill_between(pdf_x[2], pdf_y[2], 0, color='red', alpha=0.7)
