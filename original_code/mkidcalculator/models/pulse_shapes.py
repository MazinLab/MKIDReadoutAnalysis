import numpy as np
import lmfit as lm
from scipy.signal import find_peaks

EPS = np.finfo(np.float64).eps


class PulseShapeABC:
    RESPONSE = None

    def __init__(self, pulse, fit_type, weight=True):
        self.pulse = pulse
        self.loop = pulse.loop
        self.noise = pulse.noise
        self.fit_type = fit_type
        self.shrink = self.pulse.p_trace.shape[1] - self.pulse.template.shape[1]

        # initialize to template shape
        self.n_points = self.pulse.template.shape[1]
        self.f = np.fft.rfftfreq(self.n_points)[:, np.newaxis, np.newaxis]  # n_points x 1 x 1
        self.t = np.linspace(0, self.n_points / self.pulse.sample_rate * 1e6, self.n_points)
        self.weight = weight
        # initialize weights
        if weight:
            if self.fit_type == "optimal_fit":
                s = np.array([[self.noise.pp_psd, self.noise.pd_psd],
                              [np.conj(self.noise.pd_psd), self.noise.dd_psd]], dtype=np.complex)
                s = s.transpose((2, 0, 1))  # n_frequencies x 2 x 2
            elif self.fit_type == "phase_fit":
                s = self.noise.pp_psd[..., np.newaxis, np.newaxis]  # n_frequencies x 1 x 1
            elif self.fit_type == "dissipation_fit":
                s = self.noise.dd_psd[..., np.newaxis, np.newaxis]  # n_frequencies x 1 x 1
            else:
                raise ValueError("'{}' is not a valid fit_type".format(fit_type))
            self.s_inv = np.linalg.inv(s)
        else:
            if self.fit_type == "optimal_fit":
                self.s_inv = np.ones((self.f.size, 2, 2))
            else:
                self.s_inv = np.ones((self.f.size, 1, 1))

    def fit(self, data, guess):
        result = lm.minimize(self.chi2, guess, scale_covar=True, args=(data,))
        return result

    def guess(self):
        raise NotImplementedError

    def model_fft(self, params):
        raise NotImplementedError

    def chi2(self, params, data):
        model_fft = self.model_fft(params)
        if self.shrink != 0 and data.shape[0] != self.pulse.template.shape[1]:
            data = data[self.shrink // 2: -self.shrink // 2, :, :]
        data_fft = np.fft.rfft(data, axis=0)
        x = (data_fft - model_fft)  # n_frequencies x 2 x 1
        return np.sqrt((np.conj(x.transpose(0, 2, 1)) @ self.s_inv @ x).real)


class Template(PulseShapeABC):
    """A model to fit pulses to an energy dependent template."""
    RESPONSE = "energy"

    def __init__(self, pulse, fit_type):
        super().__init__(pulse, fit_type)
        self.s_inv[0, :] = 0  # don't use DC component for the fit

    def model_fft(self, params):
        energy = params["energy"].value
        index = params['index'].value
        if self.fit_type == "optimal_fit":
            calibration = np.array([[self.loop.phase_calibration(energy)], [self.loop.dissipation_calibration(energy)]])
            template_fft = self.loop.template_fft(energy).T[..., np.newaxis]  # n_frequencies x 2 x 1
        elif self.fit_type == "phase_fit":
            calibration = self.loop.phase_calibration(energy)
            template_fft = self.loop.template_fft(energy)[0, ..., np.newaxis, np.newaxis]  # n_frequencies x 1 x 1
        else:
            calibration = self.loop.dissipation_calibration(energy)
            template_fft = self.loop.template_fft(energy)[1, ..., np.newaxis, np.newaxis]  # n_frequencies x 1 x 1
        fft = template_fft * calibration
        fft_shifted = fft * np.exp(-2j * np.pi * self.f * (index - (self.n_points - self.n_points // 2)))
        return fft_shifted

    def guess(self):
        params = lm.Parameters()
        params.add("energy", value=self.pulse.energies[0] if len(self.pulse.energies) == 1 else 0)
        params.add("index", self.pulse._peak_index[self.mask].mean())
        return params


class TripleExponential(PulseShapeABC):
    """A model to fit pulses to a triple exponential function"""
    RESPONSE = "a * (1 + b) + c * (1 + d)"

    def model(self, params, **kwargs):
        t0 = kwargs.get("t0", params['t0'].value)
        p = np.empty((self.t.size, 2 if self.fit_type == "optimal_fit" else 1, 1))
        index = 0
        if self.fit_type in ["optimal_fit", "phase_fit"]:
            a = kwargs.get('a', params['a'].value)
            b = kwargs.get('b', params['b'].value)
            rise_time1 = kwargs.get('rise_time1', params['rise_time1'].value)
            fall_time1 = kwargs.get('fall_time1', params['fall_time1'].value)
            fall_time2 = kwargs.get('fall_time2', params['fall_time2'].value)
            phase_offset = kwargs.get('phase_offset', params['phase_offset'].value)
            arg0 = -(self.t[self.t >= t0] - t0) / max(rise_time1, EPS)
            arg1 = -(self.t[self.t >= t0] - t0) / max(fall_time1, EPS)
            arg2 = -(self.t[self.t >= t0] - t0) / max(fall_time2, EPS)
            p[self.t >= t0, index, 0] = -a * (1 - np.exp(arg0)) * (np.exp(arg1) + b * np.exp(arg2)) + phase_offset
            p[self.t < t0, index, 0] = phase_offset
            index += 1
        if self.fit_type in ["optimal_fit", "dissipation_fit"]:
            c = kwargs.get('c', params['c'].value)
            d = kwargs.get('d', params['d'].value)
            rise_time2 = kwargs.get('rise_time2', params['rise_time2'].value)
            fall_time3 = kwargs.get('fall_time3', params['fall_time3'].value)
            fall_time4 = kwargs.get('fall_time4', params['fall_time4'].value)
            dissipation_offset = kwargs.get('dissipation_offset', params['dissipation_offset'].value)
            arg3 = -(self.t[self.t >= t0] - t0) / max(rise_time2, EPS)
            arg4 = -(self.t[self.t >= t0] - t0) / max(fall_time3, EPS)
            arg5 = -(self.t[self.t >= t0] - t0) / max(fall_time4, EPS)
            p[self.t >= t0, index, 0] = -c * (1 - np.exp(arg3)) * (np.exp(arg4) + d * np.exp(arg5)) + dissipation_offset
            p[self.t < t0, index, 0] = dissipation_offset
        return p

    def model_fft(self, params):
        return np.fft.rfft(self.model(params), axis=0)

    def guess(self):
        params = lm.Parameters()
        params.add("t0", value=np.argmin(self.pulse.template[0]) / self.pulse.sample_rate * 1e6)
        index = []
        phase_amplitude = np.abs(np.median(np.min(self.pulse.p_trace, axis=1)))
        dissipation_amplitude = np.abs(np.median(np.min(self.pulse.d_trace, axis=1)))
        phase_fall_time = np.abs(np.trapz(self.pulse.template[0]) / self.pulse.sample_rate * 1e6)
        dissipation_fall_time = np.abs(np.trapz(self.pulse.template[1] / self.pulse.template[1].min()) /
                                       self.pulse.sample_rate * 1e6)
        peak = np.argmin(self.pulse.template[0])
        try:
            start = find_peaks(self.pulse.template[0][:peak], height=-0.5)[0][-1]  # nearest extrema
        except IndexError:
            start = 0  # no relative max before the peak
        rise_time = (self.t[peak] - self.t[start]) / 2
        if self.fit_type in ["phase_fit", "optimal_fit"]:
            params.add("a", value=phase_amplitude * 1.2, min=0)
            params.add("b", value=0.25, min=0)
            params.add("rise_time1", value=rise_time, min=0)
            params.add("fall_time1", value=phase_fall_time / 2, min=0)
            params.add("fall_time2", value=phase_fall_time * 2, min=0)
            params.add("phase_offset", value=0)
            index.append(0)

        if self.fit_type in ["dissipation_fit", "optimal_fit"]:
            params.add("c", value=dissipation_amplitude * 1.2, min=0)
            params.add("d", value=0.25, min=0)
            params.add("rise_time2", value=rise_time, min=0)
            params.add("fall_time3", value=dissipation_fall_time / 2, min=0)
            params.add("fall_time4", value=dissipation_fall_time * 2, min=0)
            params.add("dissipation_offset", value=0)
            index.append(1)

        # fit the template with all of the parameters varying to get the guess
        result = self.fit(np.atleast_3d(self.pulse.template.T[:, index]) * phase_amplitude, params)
        params = result.params

        # fix the exponential ratios and fall times based on the template fit
        if self.fit_type in ["phase_fit", "optimal_fit"]:
            params['b'].set(vary=False)
            params['fall_time1'].set(vary=False)
            params['fall_time2'].set(vary=False)
        if self.fit_type in ["dissipation_fit", "optimal_fit"]:
            params['d'].set(vary=False)
            params['fall_time3'].set(vary=False)
            params['fall_time4'].set(vary=False)
        # fix the relative amplitudes of the phase and dissipation signal
        if self.fit_type == "optimal_fit":
            params['c'].set(expr="a * {} / {}".format(params['c'].value, params['a'].value))
        elif self.fit_type == "phase_fit":
            params.add('c', value=0, vary=False)
            params.add('d', value=0, vary=False)
        else:
            params.add('a', value=0, vary=False)
            params.add('b', value=0, vary=False)
        return params

    def chi2(self, params, data):
        if self.weight:
            return super().chi2(params, data)
        else:  # override super to get a more robust residual vector (twice the length)
            model = self.model(params)
            if self.shrink != 0 and data.shape[0] != self.pulse.template.shape[1]:
                data = data[self.shrink // 2: -self.shrink // 2, :, :]
            x = (data - model)  # n_time x 2 x 1
            if self.fit_type == "optimal_fit":
                return np.concatenate([x[:, 0, 0], x[:, 1, 0]])
            else:
                return x[:, 0, 0]
