from mkidreadoutanalysis.mkidnoiseanalysis import plot_channel_fft, gen_line_noise, plot_psd, apply_lowpass_filter, compute_r
from mkidreadoutanalysis.resonator import *
from mkidreadoutanalysis.mkidreadout import MKIDReadout

if __name__ == "__main__":

    # Generate a timestream proportional to the change in quasiparticle density
    quasiparticle_timestream = QuasiparticleTimeStream(fs = 2e6, ts = 600e-3)
    # Define a sudden change in quasiparticle density (caused by a photon)
    quasiparticle_timestream.gen_quasiparticle_pulse(tf=30);
    quasiparticle_timestream.plot_pulse()
    # Generate photon arrival times
    quasiparticle_timestream.gen_photon_arrivals(cps=900)
    # Populate phase data with photon pulses
    quasiparticle_timestream.populate_photons()
    # Create resonator and compute S21
    resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=100e4)
    rf = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9)
    freq = FrequencyGrid(fc=4.0012e9, points=1000, span=500e6)
    sweep = ResonatorSweep(resonator, freq, rf)
    lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)
    theta1, d1 = lit_res_measurment.basic_coordinate_transformation()
    theta2, d2 = lit_res_measurment.nick_coordinate_transformation()
    amp_noise = gen_amp_noise(snr=30, points=quasiparticle_timestream.points)
    freqs = np.array([500e3])  # Hz and relative to center of bin (MKID we are reading out)
    amps = np.array([0.01])
    phases = np.array([0])
    line_noise = gen_line_noise(freqs, amps, phases, theta1.size, lit_res_measurment.photons.fs)
    fine_channel = theta1 + amp_noise + line_noise.real
    readout = MKIDReadout()
    readout.trigger(lit_res_measurment.photons, -fine_channel, threshold=-0.9, deadtime=45)
    readout.record_energies(-fine_channel)
    compute_r(readout.photon_energies, plot=False)