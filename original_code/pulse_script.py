import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import find_peaks, welch
from scipy.linalg import toeplitz
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
import skimage


class PhaseTimeStream:
    """ A time series containing photon pulses.
    Attributes:
     - fs: float, sample rate [Hz]
     - ts: float, sample time [uSec]
     - tvec: 1D np.array, time vector [Sec]
     - points: number of samples [None]
     - raw_phase_data: phase timestream
     - photon_arrivals: boolean 1D np.array, whether or not a photon arrived in that time step."""

    def __init__(self, fs, ts, seed=2):
        self.fs = fs
        self.ts = ts
        self.points = int(self.ts * 1e-6 * self.fs)
        self.tvec = np.arange(0, self.points) / self.fs  # J: now I think fs is the real sample rate
        self.raw_phase_data = np.zeros(self.points)
        self.rng = np.random.default_rng(seed=seed)
        self.phase_data = None  # add other
        self.holdoff = None
        self.f_psd = None
        self.psd = None
        self.optimal_filter = None

    def plot_phasetime(self, data):
        plt.figure()
        plt.plot(self.tvec * 1e6, data)
        plt.xlabel('time (usec)')
        plt.ylabel('phase (radians?)')

    def gen_pulse(self, tr=4, tf=30):
        """ pulse with tr rise time in usec and tf fall time in usec."""
        tp = np.linspace(0, 10 * tf, int(10 * tf + 1))  # pulse duration
        self.pulse = -tf * (np.exp(-tp / tf) - np.exp(-(tp) / tr)) / (tf - tr)
        return (self.pulse)

    def plot_pulse(self):
        plt.figure()
        plt.plot(self.pulse)
        plt.xlabel('Time (usec)')
        plt.ylabel('Phase (radians?)')

    def gen_photon_arrivals(self, cps=500):
        """ genetrate boolean list corresponding to poisson-distributed photon arrival events.
        Inputs:
        - cps: int, photon co
        unts per second.
        """
        photon_events = self.rng.poisson(cps / self.fs, self.tvec.shape[0])
        self.photon_arrivals = np.array(photon_events, dtype=bool)
        if sum(photon_events) > sum(self.photon_arrivals):
            print('Warning: More than 1 photon arriving per time step. Recommendation: Lower the count rate.')
        if sum(photon_events) == 0:
            print("Warning: No photons arrived. :'(")
        self.total_photons = sum(self.photon_arrivals)
        return (self.photon_arrivals)

    def populate_photons(self):
        for i in range(self.raw_phase_data.size):
            if self.photon_arrivals[i]:
                self.raw_phase_data[i:i + self.pulse.shape[0]] = self.pulse
        return (self.raw_phase_data)

    def trigger(self, threshold=-0.7, deadtime=30):
        """ threshold = phase value (really density of quasiparticles in the inductor) one must exceed to trigger
        holdoff: samples to wait before triggering again.
        deadtime: minimum time in microseconds between triggers"""

        self.holdoff = int(deadtime * 1e-6 * self.fs)

        all_trig = (phase_data.phase_data < threshold)  # & (np.diff(phase_data.phase_data, prepend=0)>0)
        trig = all_trig
        # impose holdoff
        for i in range(all_trig.size):
            if all_trig[i]:
                trig[i + 1:i + 1 + self.holdoff] = 0
        self.trig = trig
        self.total_triggers = sum(self.trig)
        return (self.trig)

    def plot_triggers(self, energies=False):
        plt.figure()
        plt.plot(self.tvec * 1e6, self.phase_data)
        plt.plot(self.tvec[self.trig] * 1e6, self.phase_data[self.trig], '.')
        plt.xlabel('time (us)')
        plt.ylabel('phase (radians)')
        # plt.xticks(rotation = 45)
        if energies == True:
            plt.plot(self.tvec[self.photon_energy_idx] * 1e6, self.phase_data[self.photon_energy_idx], 'o')

    def filter_phase(self, ntaps=None):
        if ntaps is None:
            ntaps = self.pulse.size
        # make filter
        template = self.pulse[:ntaps] / self.pulse[:ntaps].min()
        autocovariance = np.real(np.fft.irfft(self.psd / 2, 10 * self.pulse.size) * self.fs)
        autocovariance = autocovariance[:self.pulse.size]
        covariance = toeplitz(autocovariance)
        h = np.linalg.solve(covariance, template)[::-1]
        h -= h.mean()
        h /= template @ h[::-1]
        self.optimal_filter = h
        # apply the filter
        self.phase_data = np.convolve(h, self.phase_data, mode='same')

    def record_energies(self):
        holdoff_views = skimage.util.view_as_windows(self.phase_data, self.holdoff)  # n_data x holdoff
        trig_views = holdoff_views[self.trig[:holdoff_views.shape[0]]]  # n_triggers x holdoff
        self.photon_energies = np.min(trig_views, axis=1)
        self.photon_energy_idx = np.argmin(trig_views, axis=1) + np.nonzero(self.trig[:holdoff_views.shape[0]])[0]
        return (self.photon_energies)

    def gen_amp_noise(self, snr):
        """ Flat PSD, white-noise generated from voltage fluctuations"""
        a_noise = 10 ** ((20 * np.log10(1 / np.sqrt(2)) - snr) / 10);  # input dBm of noise
        noise = np.sqrt(a_noise) * self.rng.normal(size=self.points)
        self.amp_noise = noise

    def gen_tls_noise(self, scale=1e-3, fr=6e9, q=15e3):
        """ two-level system noise"""
        psd_freqs = np.fft.rfftfreq(self.phase_data.size, d=1 / self.fs)
        fc = fr/(2*q)
        psd = np.zeros_like(psd_freqs)
        nonzero = psd_freqs != 0
        psd[nonzero] = scale / (1 + (psd_freqs[nonzero] / fc)**2) / psd_freqs[nonzero]
        noise_phi = 2 * np.pi * self.rng.random(psd_freqs.size)
        noise_fft = np.exp(1j * noise_phi)  # n_traces x n_frequencies
        # rescale the noise to the covariance
        a = np.sqrt(self.phase_data.size * psd * self.fs / 2)
        noise_fft = a * noise_fft
        self.tls_noise =  np.fft.irfft(noise_fft, self.phase_data.size)


    def plot_psd(self, data, fres=1e3, **kwargs):
        plt.figure()
        default = {'fs': self.fs, 'nperseg': self.fs / fres}
        default.update(kwargs)
        f, psd = welch(data, **default)
        plt.semilogx(f, 10 * np.log10(psd))
        plt.xlabel(f'Frequency [Hz] ({fres * 1e-3:g} kHz resolution)')
        plt.ylabel('dB/Hz')
        plt.grid()
        plt.title('Power Spectral Density')
        # add axis later (include res)

    def set_noise(self, amp=False, tls=False, **kwargs):
        noise = np.zeros_like(self.raw_phase_data)
        if amp:
            noise += self.amp_noise
        if tls:
           noise += self.tls_noise
        self.phase_data = self.raw_phase_data + noise
        default = {'fs': self.fs, 'nperseg': 10 * self.pulse.size}
        default.update(kwargs)
        self.f_psd, self.psd = welch(noise, **default)


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
    bandwidth = amplitudes.size**(-1 / 5) * sigma_mad

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


if __name__ == "__main__":
    # Generate a phase time stream
    phase_data = PhaseTimeStream(fs=2e6, ts=1e6)

    # Plot the raw data
#    phase_data.plot_phasetime(phase_data.raw_phase_data)

    # Define a photon pulse
    phase_data.gen_pulse()

    # Plot the pulse
#    phase_data.plot_pulse()

    # Generate photon arrival times
    phase_data.gen_photon_arrivals(500)

    # Verify how many photons we got
    print(phase_data.total_photons)

    # Populate phase data with photon pulses
    phase_data.populate_photons()

    # Plot new phase data
#    phase_data.plot_phasetime(phase_data.raw_phase_data)

    # Set Phase Data (No noise)
    phase_data.set_noise()

    # Trigger on photons
    phase_data.trigger(threshold=-0.7, deadtime=30)

    # Count Triggers
    print('Photons:', phase_data.total_photons, 'Triggers:', phase_data.total_triggers)

    # Plot trigger events
#    phase_data.plot_triggers()

    # Record triggered values
    phase_data.record_energies()

    # Plot trigger events
#    phase_data.plot_triggers(energies=True)

    # Plot histogram of "energies"
#    phase_data.energy_histogram()

    phase_data.gen_amp_noise(10)
    phase_data.gen_tls_noise(scale=1e-3)

#    phase_data.plot_psd(phase_data.amp_noise)

    phase_data.set_noise(amp=True, tls=True)
    phase_data.plot_psd(phase_data.tls_noise + phase_data.amp_noise)

    phase_data.filter_phase()

# Plot new phase data
#    phase_data.plot_phasetime(phase_data.phase_data)

    # Trigger on photons
    phase_data.trigger(threshold=-0.3, deadtime=60)

    # Plot trigger events
#    phase_data.plot_triggers()

    phase_data.record_energies()

    # Plot trigger events
    phase_data.plot_triggers(energies=True)

    # Plot histogram of "energies"
#    phase_data.energy_histogram()

    print("R = ", compute_r(phase_data.photon_energies, plot=True))



