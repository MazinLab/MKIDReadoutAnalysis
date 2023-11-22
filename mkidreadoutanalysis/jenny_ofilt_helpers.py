import logging

import matplotlib.pyplot as plt
from mkidreadoutanalysis.resonator import *


def generate_fake_data(res: Resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1), fs=1e6,
                       ts=10, tf=30, cps=1551,
                       rf: RFElectronics = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9),
                       noise=True, noise_scale=10, line_noise_freqs=[60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3],
                       line_noise_amplitudes=[0.005, 0.001, 0.005, 0.003, 0.001, 0.005, 0.001],
                       line_noise_phases=[0, 0.5, 0, 1.3, 0.5, 0.2, 2.4]):
    quasiparticle_timestream = QuasiparticleTimeStream(fs = 1e6, ts = 10)
    quasiparticle_timestream.gen_quasiparticle_pulse(tf=30);
    quasiparticle_timestream.gen_photon_arrivals(cps=1551)
    quasiparticle_timestream.populate_photons()
    # Create resonator and compute S21
    resonator = res #1e2
    rf = rf
    freq = FrequencyGrid( fc=res.f0_0, points=1000, span=500e6)
    sweep = ResonatorSweep(resonator, freq, rf)
    lit_res_measurment = ReadoutPhotonResonator(resonator, quasiparticle_timestream, freq, rf)
    # toggle white noise and line noise
    lit_res_measurment.noise_on = noise

    # adjust white noise scale
    lit_res_measurment.rf.noise_scale = noise_scale

    # configure line noise
    lit_res_measurment.rf.line_noise.freqs = line_noise_freqs # Hz and relative to center of bin (MKID we are reading out)
    lit_res_measurment.rf.line_noise.amplitudes = line_noise_amplitudes
    lit_res_measurment.rf.line_noise.phases = line_noise_phases

    phase_data, _ = lit_res_measurment.basic_coordinate_transformation()

    return phase_data

