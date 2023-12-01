import logging
import numbers
import numpy as np
import lmfit as lm
import scipy.signal as sps
from scipy.ndimage import label, find_objects

from mkidcalculator.models.nonlinearity import swenson
from mkidcalculator.models.utils import bandpass, _compute_sigma

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
EPS = np.finfo(np.float).eps


class S21:
    """Basic S₂₁ model."""
    @classmethod
    def baseline(cls, params, f):
        """
        Return S21 at the frequencies f, ignoring the effect of the resonator,
        for the specified model parameters.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            f: numpy.ndarray, dtype=real
                Frequency points corresponding to z.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        # 0th, 1st, and 2nd terms in a power series to handle magnitude gain different than 1
        gain0 = params['gain0'].value
        gain1 = params['gain1'].value
        gain2 = params['gain2'].value

        # 0th and 1st terms in a power series to handle phase gain different than 1
        phase0 = params['phase0'].value
        phase1 = params['phase1'].value
        phase2 = params['phase2'].value

        # midpoint frequency
        fm = params['fm'].value

        # the gain should be referenced to the file midpoint so that the baseline
        # coefficients do not drift with changes in f0
        # this breaks down if different sweeps have different frequency ranges
        xm = (f - fm) / fm

        # Calculate magnitude and phase gain
        gain = gain0 + gain1 * xm + gain2 * xm**2
        phase = np.exp(1j * (phase0 + phase1 * xm + phase2 * xm**2))
        z = gain * phase
        return z

    @classmethod
    def x(cls, params, f):
        """
        Return the fractional frequency shift at the frequencies f for the
        specified model parameters. If the resonator is nonlinear, the
        treatment by Swenson et al. (doi: 10.1063/1.4794808) is used.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            f: number or numpy.ndarray, dtype=real
                Frequency points corresponding to z. If f is just a number, the
                sweep direction is chosen to be increasing.
        Returns:
            x: numpy.ndarray
                The fractional frequency shift.
        """
        # loop parameters
        f0 = params['f0'].value  # resonant frequency
        q0 = params['q0'].value  # total Q
        a = params['a'].value  # nonlinearity parameter (Swenson et al. 2013)
        # make sure that f is a numpy array
        f = np.atleast_1d(f)
        # Make everything referenced to the shifted, unitless, reduced frequency
        # accounting for nonlinearity
        if a != 0:
            y0 = q0 * (f - f0) / f0
            try:
                f1d = f.ravel()  # ensures f doesn't have a dimension >= 2 when indexing
                increasing = True if f1d[1] - f1d[0] >= 0 else False
            except IndexError:  # single value so assume we are increasing in frequency
                increasing = True
            y = swenson(y0, a, increasing=increasing)
            x = y / q0
        else:
            x = (f - f0) / f0
        return x

    @classmethod
    def resonance(cls, params, f):
        """
        Return S21 at the frequencies f, only considering the resonator, for
        the specified model parameters. Formula from Khalil et al. 2012.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            f: number or numpy.ndarray, dtype=real
                Frequency points corresponding to z.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        # loop parameters
        xa = params['xa'].value  # frequency shift due to mismatched impedances
        qi = params['qi'].value  # internal Q
        qc = params['qc'].value  # coupling Q
        # make sure that f is a numpy array
        f = np.atleast_1d(f)
        # find the fractional frequency shift
        x = cls.x(params, f)
        # find the scattering parameter
        z = (qc + 2j * qi * qc * (x + xa)) / (qi + qc + 2j * qi * qc * x)
        return z

    @classmethod
    def mixer(cls, params, z):
        """
        Apply the mixer correction specified in the model parameters to S21.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            z: numpy.ndarray, dtype=complex
                Complex resonator scattering parameter.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        alpha = params['alpha'].value
        beta = params['beta'].value
        offset = params['gamma'].value + 1j * params['delta'].value
        z = (z.real + 1j * alpha * (z.real * np.sin(beta) + z.imag * np.cos(beta))) + offset
        return z

    @classmethod
    def mixer_inverse(cls, params, z):
        """
        Undo the mixer correction specified by the parameters to S21. This is
        useful for removing the effect of the mixer on real data.
        Args:
            params: lmfit.Parameters() object or tuple
                The parameters for the model function or a tuple with
                (alpha, beta, offset) where offset is gamma + i * delta.
            z: numpy.ndarray, dtype=complex
                Complex resonator scattering parameter.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        if isinstance(params, lm.Parameters):
            alpha = params['alpha'].value
            beta = params['beta'].value
            offset = params['gamma'].value + 1j * params['delta'].value
        else:
            alpha, beta, offset = params
        z = z - offset
        z = (z.real + 1j * (-z.real * np.tan(beta) + z.imag / np.cos(beta) / alpha))
        return z

    @classmethod
    def calibrate(cls, params, z, f, mixer_correction=True, center=False):
        """
        Remove the baseline and mixer effects from the S21 data.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            z: numpy.ndarray, dtype=complex
                Complex resonator scattering parameter.
            f: numpy.ndarray, dtype=real
                Frequency points corresponding to z.
            mixer_correction: boolean (optional)
                Remove the mixer correction specified in the params object. The
                default is True.
            center: boolean (optional)
                Rotates and centers the loop if True. The default is False.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        if mixer_correction:
            z = cls.mixer_inverse(params, z) / cls.baseline(params, f)
        else:
            z /= cls.baseline(params, f)
        if center:
            center = params.eval("1 - q0 / (2 * qc) + 1j * q0 * xa")
            z = center - z
        return z

    @classmethod
    def model(cls, params, f, mixer_correction=True):
        """
        Return the model of S21 at the frequencies f for the specified model
        parameters.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            f: numpy.ndarray, dtype=real
                Frequency points corresponding to z.
            mixer_correction: boolean (optional)
                Apply the mixer correction specified in the params object. The
                default is True.
        Returns:
            z: numpy.ndarray
                The S21 scattering parameter.
        """
        z = cls.baseline(params, f) * cls.resonance(params, f)
        if mixer_correction:
            z = cls.mixer(params, z)
        return z

    @classmethod
    def residual(cls, params, z, f, sigma=None, return_real=True):
        """
        Return the normalized residual between the S21 data and model.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            z: numpy.ndarray, dtype=complex, shape=(N,)
                Complex resonator scattering parameter.
            f: numpy.ndarray, dtype=real, shape=(N,)
                Frequency points corresponding to z.
            sigma: numpy.ndarray, dtype=complex, shape=(N,) or complex number
                The standard deviation of the data z at f in the form
                std(z.real) + i std(z.imag). The default is None. If None is
                provided, the standard deviation is calculated from the first
                10 points after being detrended. If a complex number, the same
                sigma is used for each data point.
            return_real: boolean (optional)
                Concatenate the real and imaginary parts of the residual into a
                real 1D array of shape (2N,).
        Returns:
            res: numpy.ndarray, dtype=(complex or float)
                Either a complex N or a real 2N element 1D array (depending on
                return_real) with the normalized residuals.
        """
        # grab the model
        m = cls.model(params, f)
        # calculate constant error from standard deviation of the first 10 pts of the data if not supplied
        if sigma is None:
            sigma = _compute_sigma(z)
        if isinstance(sigma, numbers.Number):
            sigma = np.broadcast_to(sigma, z.shape)
        if return_real:
            # convert model, data, and error into a real vector
            m_1d = np.concatenate((m.real, m.imag), axis=0)
            z_1d = np.concatenate((z.real, z.imag), axis=0)
            sigma_1d = np.concatenate((sigma.real, sigma.imag), axis=0)
            res = (m_1d - z_1d) / sigma_1d
        else:
            # return the complex residual
            res = (m.real - z.real) / sigma.real + 1j * (m.imag - z.imag) / sigma.imag
        return res

    @classmethod
    def phase_and_dissipation(cls, params, z, f, form='geometric', unwrap=False, fr_reference=False):
        """
        Return the phase and dissipation signals from measured IQ data.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            z: numpy.ndarray, dtype=complex, shape=(N,)
                Complex resonator scattering parameter.
            f: numpy.ndarray, dtype=real, shape=(N,)
                Frequency points corresponding to z.
            form: string (optional)
                Either 'geometric' or 'analytic'. The default is 'geometric'.
                'geometric': use the polar decomposition from the loop center
                'analytic': solve for xr and dqi_inv from the loop equation
                            then scale to match the geometric version
            unwrap: boolean (optional)
                If form is 'geometric', unwrap can be set to True to remove
                wraps introduced by numpy.angle(). The default is False, and
                no unwrapping is done. This argument is ignored if form =
                'analytic'.
            fr_reference: boolean (optional)
                If True, the phase is referenced to the phase of the resonance
                frequency. Otherwise, the phase is referenced to the value
                given by the model at f. The default is False.
        Returns:
            phase: numpy.ndarray, dtype=real, shape=(N,)
                The phase signal.
            dissipation: numpy.ndarray, dtype=real, shape=(N,)
                The dissipation signal.
        """
        # make sure data is of the right type and shape
        z = np.array(z, ndmin=1, dtype=complex, copy=False)
        f = np.broadcast_to(f, z.shape)
        if form.lower().startswith('geometric'):
            # get loop z for the data at the reference frequency
            f_ref = params['fr'].value if fr_reference else f
            z_ref = cls.model(params, f_ref)
            # calibrate the IQ data
            z = cls.calibrate(params, z, f, center=True)
            z_ref = cls.calibrate(params, z_ref, f_ref, center=True)  # real if f_ref = fr and no loop asymmetry
            # compute the phase from the centered traces
            phase = np.angle(z)
            # make the wrap angle as far from each trace median as possible to minimize wraps
            wrap_angle = np.median(phase, axis=-1, keepdims=True) + np.pi
            phase = np.mod(phase - wrap_angle, 2 * np.pi) - (2 * np.pi - wrap_angle)
            # unwrap any data that is still crossing the wrap angle
            if unwrap:
                phase = np.unwrap(phase)
            # reference the angle to the (properly wrapped) z_ref angle
            phase -= np.mod(np.angle(z_ref) - wrap_angle, 2 * np.pi) - (2 * np.pi - wrap_angle)
            # compute the dissipation trace from the centered traces
            dissipation = np.abs(z) / np.abs(z_ref) - 1
        elif form.lower().startswith('analytic'):
            # grab parameter values
            q0 = params['q0'].value
            qc = params['qc'].value
            qi = params['qi'].value
            xa = params['xa'].value
            x = cls.x(params, f)
            # calibrate IQ data
            z = cls.calibrate(params, z, f, center=False)
            i = z.real
            q = z.imag
            z[np.abs(1 - z) < EPS] = 1 - EPS  # avoid zero denominator
            # compute phase
            dx = (q + 2 * qc * xa * (i - 1)) / (2 * qc * np.abs(1 - z)**2) - x
            phase = -4 * q0 / (1 + 4 * q0**2 * x**2) * dx
            if fr_reference:  # already referenced if not using resonance frequency
                f_ref = params['fr'].value
                z_ref = cls.calibrate(params, cls.model(params, f_ref), f_ref, center=True)
                phase -= np.angle(z_ref)
            # compute dissipation
            dqi_inv = (i - np.abs(z)**2 + 2 * qc * xa * q) / (qc * np.abs(1 - z)**2) - qi**-1
            dissipation = -2 * q0 / (1 + 4 * q0**2 * x**2) * dqi_inv
        else:
            raise ValueError("'form' must be one of ['geometric', 'analytic']")
        return phase, dissipation

    @classmethod
    def guess(cls, z, f, imbalance=None, offset=None, use_filter=False, filter_length=None, fit_resonance=True,
              nonlinear_resonance=False, fit_gain=True, quadratic_gain=True, fit_phase=True, quadratic_phase=False,
              fit_imbalance=False, fit_offset=False, **kwargs):
        """
        Guess the model parameters based on the data. Returns a
        lmfit.Parameters() object.
        Args:
            z: numpy.ndarray, dtype=complex, shape=(N,)
                Complex resonator scattering parameter.
            f: numpy.ndarray, dtype=real, shape=(N,)
                Frequency points corresponding to z.
            imbalance: numpy.ndarray, dtype=complex, shape=(M, L) (optional)
                Mixer imbalance calibration data (M data sets of I and Q
                beating). Each of the M data sets is it's own calibration,
                potentially taken at different frequencies and frequency
                offsets. The results of the M data sets are averaged together.
                The default is None, which means alpha and beta are taken from
                the keywords. The alpha and beta keywords are ignored if a
                value other than None is given.
            offset: complex, iterable (optional)
                A complex number corresponding to the I + iQ mixer offset. The
                default is 0, corresponding to no offset. If the input is
                iterable, a mean is taken to determine the mixer_offset value.
            use_filter: boolean (optional)
                Filter the phase and magnitude data of z before trying to guess
                the parameters. This can be helpful for noisy data, but can
                also result in poor guesses for clean data. The default is
                False.
            filter_length: int, odd >= 3 (optional)
                If use_filter==True, this is used as the filter length. Only
                odd numbers greater or equal to three are allowed. If None, a
                filter length is computed as roughly 1% of the number of points
                in z. The default is None.
            fit_resonance: boolean (optional)
                Allow the resonance parameters to vary in the fit. The default
                is True.
            nonlinear_resonance: float, boolean (optional)
                Allow the resonance model to fit for nonlinear behavior. The
                default is False. If a float, this value is used for 'a'.
                If True, the 'a' is set to 0.0025, since the fit has trouble
                if 'a' is initialized to 0.
            fit_gain: boolean (optional)
                Allow the gain parameters to vary in the fit. The default is
                True.
            quadratic_gain: boolean (optional)
                Allow for a quadratic gain component in the model. The default
                is True.
            fit_phase: boolean (optional)
                Allow the phase parameters to vary in the fit. The default is
                True.
            quadratic_phase: boolean (optional)
                Allow for a quadratic phase component in the model. The default
                is False since there isn't an obvious physical reason why there
                should be a quadratic term.
            fit_imbalance: boolean (optional)
                Allow the IQ mixer amplitude and phase imbalance to vary in the
                fit. The default is False. The imbalance is typically
                calibrated and not fit.
            fit_offset: boolean (optional)
                Allow the IQ mixer offset to vary in the fit. The default is
                False. The offset is highly correlated with the gain parameters
                and typically should not be allowed to vary unless the gain is
                properly calibrated.
            kwargs: (optional)
                Set the options of any of the parameters directly bypassing the
                calculated guess.
        Returns:
            params: lmfit.Parameters
                An object with guesses and bounds for each parameter.
        """
        # undo the mixer calibration for more accurate guess if known ahead of time
        offset = np.mean(offset) if offset is not None else 0.
        if imbalance is not None:
            # bandpass filter the I and Q signals
            imbalance = np.atleast_2d(imbalance)
            i, q = imbalance.real, imbalance.imag
            n = i.shape[0]
            ip, f_i_ind = bandpass(i)
            qp, f_q_ind = bandpass(q)
            # compute alpha and beta
            amp = np.sqrt(2 * np.mean(ip**2, axis=-1))
            alpha = np.sqrt(2 * np.mean(qp**2, axis=-1)) / amp
            ratio = np.angle(np.fft.rfft(ip)[np.arange(n), f_i_ind[:, 0]] /
                             np.fft.rfft(qp)[np.arange(n), f_q_ind[:, 0]])  # for arcsine branch
            beta = np.arcsin(np.sign(ratio) * 2 * np.mean(qp * ip, axis=-1) / (alpha * amp**2)) + np.pi * (ratio < 0)
            alpha = np.mean(alpha)
            beta = np.mean(beta)
        else:
            alpha = 1.
            beta = 0.
        if kwargs.get('alpha', None) is not None:
            alpha = kwargs['alpha']['value'] if isinstance(kwargs['alpha'], dict) else kwargs['alpha']
        if kwargs.get('beta', None) is not None:
            beta = kwargs['beta']['value'] if isinstance(kwargs['beta'], dict) else kwargs['beta']
        z = cls.mixer_inverse((alpha, beta, offset), z)
        # compute the magnitude and phase of the scattering parameter
        magnitude = np.abs(z)
        phase = np.unwrap(np.angle(z))
        # filter the magnitude and phase if requested
        if use_filter:
            if filter_length is None:
                filter_length = int(np.round(len(magnitude) / 100.0))
            if filter_length % 2 == 0:
                filter_length += 1
            if filter_length < 3:
                filter_length = 3
            magnitude = sps.savgol_filter(magnitude, filter_length, 1)
            phase = sps.savgol_filter(phase, filter_length, 1)

        # calculate useful indices
        f_index_end = len(f) - 1  # last frequency index
        f_index_5pc = max(int(len(f) * 0.05), 2)  # end of first 5% of data
        # set up a unitless, reduced, midpoint frequency for baselines
        f_midpoint = np.median(f)  # frequency at the center of the data

        def xm(fx):
            return (fx - f_midpoint) / f_midpoint

        # get the magnitude and phase data to fit
        mag_ends = np.concatenate((magnitude[:f_index_5pc], magnitude[-f_index_5pc + 1:]))
        phase_ends = np.concatenate((phase[:f_index_5pc], phase[-f_index_5pc + 1:]))
        freq_ends = xm(np.concatenate((f[:f_index_5pc], f[-f_index_5pc + 1:])))
        # calculate the gain polynomials
        gain_poly = np.polyfit(freq_ends, mag_ends, 2 if quadratic_gain else 1)
        if not quadratic_gain:
            gain_poly = np.concatenate(([0], gain_poly))
        phase_poly = np.polyfit(freq_ends, phase_ends, 2 if quadratic_phase else 1)
        if not quadratic_phase:
            phase_poly = np.concatenate(([0], phase_poly))

        # guess f0
        f_index_min = np.argmin(magnitude - np.polyval(gain_poly, xm(f)))
        f0_guess = f[f_index_min]
        # set some bounds (resonant frequency should not be within 5% of file end)
        f_min = min(f[f_index_5pc],  f[f_index_end - f_index_5pc])
        f_max = max(f[f_index_5pc],  f[f_index_end - f_index_5pc])
        if not f_min < f0_guess < f_max:
            f0_guess = f_midpoint

        # guess Q values
        mag_max = np.polyval(gain_poly, xm(f[f_index_min]))
        mag_min = magnitude[f_index_min]
        fwhm = np.sqrt((mag_max**2 + mag_min**2) / 2.)  # fwhm is for power not amplitude
        fwhm_mask = magnitude < fwhm
        regions, _ = label(fwhm_mask)  # find the regions where magnitude < fwhm
        region = regions[f_index_min]  # pick the one that includes the minimum
        try:
            f_masked = f[find_objects(regions, max_label=region)[-1]]  # mask f to only include that region
            bandwidth = f_masked.max() - f_masked.min()  # find the bandwidth
        except IndexError:  # no found region
            bandwidth = 0  # defer calculation
        # Q0 = f0 / fwhm bandwidth
        q0_guess = f0_guess / bandwidth if bandwidth != 0 else 1e4
        # Q0 / Qi = min(mag) / max(mag)
        qi_guess = q0_guess * mag_max / mag_min if mag_min != 0 else 1e5
        if qi_guess == 0:
            qi_guess = 1e5
        if q0_guess == 0:
            q0_guess = 1e4
        # 1 / Q0 = 1 / Qc + 1 / Qi
        qc_guess = 1. / (1. / q0_guess - 1. / qi_guess) if (1. / q0_guess - 1. / qi_guess) != 0 else 1e4

        # make the parameters object (coerce all values to float to avoid ints and numpy types)
        params = lm.Parameters()
        # resonance parameters
        params.add('xa', value=float(0), vary=fit_resonance)
        params.add('f0', value=float(f0_guess), min=f_min, max=f_max, vary=fit_resonance)
        params.add('qc', value=float(qc_guess), min=1, max=10**8, vary=fit_resonance)
        params.add('qi', value=float(qi_guess), min=1, max=10**8, vary=fit_resonance)
        a_sqrt = np.sqrt(0.0025) if nonlinear_resonance is True else nonlinear_resonance  # bifurcation at a=0.7698
        params.add('a_sqrt', value=float(a_sqrt), vary=bool(nonlinear_resonance) and fit_resonance)
        # polynomial gain parameters
        params.add('gain0', value=float(gain_poly[2]), min=0, vary=fit_gain)
        params.add('gain1', value=float(gain_poly[1]), vary=fit_gain)
        params.add('gain2', value=float(gain_poly[0]), vary=quadratic_gain and fit_gain)
        # polynomial phase parameters
        params.add('phase0', value=float(phase_poly[2]), vary=fit_phase)
        params.add('phase1', value=float(phase_poly[1]), vary=fit_phase)
        params.add('phase2', value=float(phase_poly[0]), vary=quadratic_phase and fit_phase)
        # IQ mixer parameters
        params.add('gamma', value=float(offset.real), vary=fit_offset)
        params.add('delta', value=float(offset.imag), vary=fit_offset)
        params.add('alpha', value=float(alpha), vary=fit_imbalance)
        params.add('beta', value=float(beta), min=beta - np.pi / 2, max=beta + np.pi / 2, vary=fit_imbalance)
        # add derived parameters
        params.add("a", expr="a_sqrt**2")  # nonlinearity parameter (Swenson et al. 2013)
        params.add("q0", expr="1 / (1 / qi + 1 / qc)")  # the total quality factor
        params.add("fm", value=float(f_midpoint), vary=False)  # the frequency midpoint used for fitting
        params.add("tau", expr="-phase1 / (2 * pi * fm)")  # the cable delay
        params.add("fr", expr="f0 * (1 - a / q0)")  # resonance frequency accounting for nonlinearity

        # override the guess
        for key, options in kwargs.items():
            if options is not None:
                if isinstance(options, dict):
                    params[key].set(**options)
                else:
                    params[key].set(value=options)
        return params
