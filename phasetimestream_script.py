from phasetimestream import PhaseTimeStream
import mkidnoiseanalysis

if __name__ == "__main__":
    # Generate a phase time stream
    phase_data = PhaseTimeStream(fs=2e6, ts=1e6)

    # Plot the raw data
    phase_data.plot_phasetime(phase_data.data_nonoise)

    # Define a photon pulse
    phase_data.gen_pulse(tr=4, tf=30)

    # Plot the pulse
    phase_data.plot_pulse()

    # Generate photon arrival times
    phase_data.gen_photon_arrivals(cps=500)

    # Verify how many photons we got
    print(phase_data.photon_arrivals.sum())

    # Populate phase data with photon pulses
    phase_data.populate_photons()

    # Plot new phase data
    phase_data.plot_phasetime(phase_data.data_nonoise)

    # Set TLS Noise
    phase_data.set_tls_noise(scale=1e-3)

    # View TLS Noise
    mkidnoiseanalysis.plot_psd(phase_data.tls_noise)

    # Trigger on photons
    phase_data.trigger(phase_data.data, threshold=-0.7, deadtime=30)

    # Count Triggers
    print('Photons:', phase_data.photon_arrivals.sum(), 'Triggers:', phase_data.trig.sum())

    # Record energies of triggered values
    phase_data.record_energies(phase_data.data)

    # Plot trigger events with energies
    phase_data.plot_triggers(phase_data.data, energies=True)

    phase_data.plot_energies()

  #  print("R = ", mkidnoiseanalysis.compute_r(phase_data.photon_energies, plot=True))