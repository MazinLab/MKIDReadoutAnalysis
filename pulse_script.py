from phasetimestream import PhaseTimeStream

if __name__ == "__main__":
    # Generate a phase time stream
    phase_data = PhaseTimeStream(fs=2e6, ts=1e6)

    # Plot the raw data
    phase_data.plot_phasetime(phase_data.raw_phase_data)

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



