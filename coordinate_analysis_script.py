from mkidnoiseanalysis import swenson_formula
from resonator import Resonator
import numpy as np
from matplotlib import pyplot as plt

# Resonator Properties
res_params = {'f0': 4.0012e9,  # resonance frequency [Hz]
              'qi': 200000,  # internal quality factor
              'qc': 15000,  # coupling quality factor
              'xa': 0.5,  # resonance fractional asymmetry
              'a': 0,  # inductive nonlinearity
              'alpha': 1.,  # IQ mixer amplitude imbalance
              'beta': 0.,  # IQ mixer phase imbalance
              'gain0': 3.0,  # gain polynomial coefficients
              'gain1': 0,  # linear gain coefficient
              'gain2': 0,  # quadratic gain coefficient
              'phase0': 0,  # total loop rotation in radians
              'tau': 50e-9}  # cable delay

fsweep_params = {'fc': 4.0012e9,  # center frequency [Hz]
                 'points': 1000,  # frequency sweep points
                 'span': 500e6,  # frequency sweep bandwidth [Hz]
                 'increasing': True}  # frequency sweep direction

res = Resonator(**res_params)
res.fsweep(**fsweep_params)
s21 = res.s21
res.fvec = res.f0
s210 = res.s21


def pulse(t, t0, tf=30):
    p = np.zeros_like(t)
    p[t >= t0] = -np.exp(-(t[t >= t0] - t0) / tf)
    return p

t = np.linspace(0, 500, 1000)  # in us
photon = pulse(t, 20, tf=30)

#make photon into a series of photons

tls_noise_without_lowpass = 0
dfr = photon * 1e5 + tls_noise_without_lowpass
dqi_inv = -photon * 2e-5
res.fvec = res_params['f0']
res.f0 += dfr
res.qi = (res.qi**-1 + dqi_inv)**-1
s21_photon = res.s21


def lowpass(s21, f0, q, dt):
    f = np.fft.fftfreq(s21.size, d=dt)
    z = np.fft.ifft(np.fft.fft(s21) / (1 + 2j * q * f / f0))
    return z.real, z.imag


q0 = (res.qi[0]**-1 + res.qc**-1)**-1

i, q = lowpass(s21_photon, res.f0[0], q0, (t[1] - t[0]) * 1e-6)
amp_and_line_noise1 = 0
amp_and_line_noise2 = 0
i += amp_and_line_noise1
q += amp_and_line_noise2

z = i + 1j * q
zb = z / res.background(res.f0)
ib, qb = zb.real, zb.imag

s21b = s210 / res.background(res.f0)

xn = swenson_formula(0, res.a, increasing=res.increasing) / q0

z1 = (1 - zb - q0 / (2 * res.qc) + 1j * q0 * res.xa) / (1 - s21b - q0 / (2 * res.qc) + 1j * q0 * res.xa)
theta1 = np.arctan2(z1.imag, z1.real)

d1 = (np.abs(1 - zb - q0 / (2 * res.qc) + 1j * q0 * res.xa) /
      np.abs(q0 / (2 * res.qc) - 1j * q0 * res.xa)) - 1

theta2 = -4 * q0 / (1 + 4 * q0**2 * xn**2) * ((qb + 2 * res.qc * res.xa * (ib - 1)) /
                                              (2 * res.qc * np.abs(1 - zb)**2) - xn)
d2 = -2 * q0 / (1 + 4 * q0**2 * xn**2) * ((ib - np.abs(zb)**2 + 2 * res.qc * res.xa * qb) /
                                          (res.qc * np.abs(1 - zb)**2) - 1 / res.qi[0])

fig, axes = plt.subplots()
axes.axis('equal')
axes.plot(s21.real, s21.imag, 'o')
axes.plot(s21_photon.real, s21_photon.imag, '-')
axes.plot(i, q)

fig, axes = plt.subplots()
axes.plot(theta1, color='C0')
axes.plot(d1, color='C1')
axes.plot(theta2, linestyle=":", color='C0')
axes.plot(d2, linestyle=':', color='C1')
