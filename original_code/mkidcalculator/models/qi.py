import warnings
import numpy as np
import lmfit as lm
import scipy.constants as sc
from mkidcalculator.models.utils import scaled_alpha, scaled_alpha_inv

try:
    from superconductivity import complex_conductivity as cc
    HAS_SUPERCONDUCTIVITY = True
except ImportError:
    HAS_SUPERCONDUCTIVITY = False

# constants
pi = sc.pi
kb = sc.k  # [J / K] Boltzmann const
h = sc.h  # [J * s] Plank constant
BCS = pi / np.exp(np.euler_gamma)  # bcs constant


class Qi:
    @classmethod
    def mattis_bardeen(cls, params, temperatures, low_energy=False,
                       parallel=False):
        """
        Returns the inverse internal quality factor of the resonator
        with the specified model parameters for Mattis Bardeen effects.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray
                The temperatures at which the quality factor is
                evaluated. If None, an inverse quality factor of zero is
                returned.
            low_energy: boolean (optional)
                Use the low energy approximation to evaluate the complex
                conductivity. The default is False.
            parallel: multiprocessing.Pool or boolean (optional)
                A multiprocessing pool object to use for the
                computation. The default is False, and the computation
                is done in serial. If True, a Pool object is created
                with multiprocessing.cpu_count() CPUs. Only used if
                low_energy is False.
        Returns:
            q_inv: numpy.ndarray
                The inverse quality factor.
        """
        if not HAS_SUPERCONDUCTIVITY:
            raise ImportError("The superconductivity package is not "
                              "installed.")
        if temperatures is None:
            return 0.
        # unpack parameters
        alpha = params['alpha'].value
        tc = params['tc'].value
        bcs = params['bcs'].value
        dynes = params['dynes'].value
        gamma = np.abs(params['gamma'].value)
        f0 = params['f0'].value
        # calculate Qinv
        sigma = cc.value(temperatures, f0, tc, gamma=dynes, bcs=bcs,
                         low_energy=low_energy, parallel=parallel)
        q_inv = alpha * np.real(sigma**-gamma) / np.imag(sigma**-gamma)
        return q_inv

    @classmethod
    def two_level_systems(cls, params, temperatures, powers):
        """
        Returns the inverse quality factor of the resonator with the
        specified model parameters for two-level system effects.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray
                The temperatures at which the quality factor is
                evaluated. If None, only the power dependence is used.
            powers: numpy.ndarray
                The powers at which the quality factor is evaluated. If
                None, only the temperature dependence is used.
        Returns:
            q_inv: numpy.ndarray
                The inverse quality factor.
        """
        if powers is None and temperatures is None:
            return 0.
        # unpack parameters
        f0 = params['f0'].value
        fd = params['fd'].value
        pc = params['pc'].value
        # calculate Qinv
        q_inv = fd
        if temperatures is not None:
            xi = h * f0 / (2 * kb * temperatures)
            q_inv *= np.tanh(xi)
        if powers is not None:
            q_inv /= np.sqrt(1. + 10**((powers - pc) / 10.))
        return q_inv

    @classmethod
    def constant_loss(cls, params):
        """
        Returns the constant loss inverse quality factor.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
        Returns:
            q_inv: float
                The inverse quality factor.
        """
        return params['q0_inv'].value

    @classmethod
    def model(cls, params, temperatures=None, powers=None,
              low_energy=False, parallel=False):
        """
        Returns the model of Qi for the specified model parameters.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray (optional)
                The temperatures at which the quality factors are
                evaluated. The default is None and the temperature's
                effect on the quality factor is ignored.
            powers: numpy.ndarray (optional)
                The powers at which the quality factors are evaluated.
                The default is None and the power's effect on the
                quality factor is ignored.
            low_energy: boolean (optional)
                Use the low energy approximation to evaluate the complex
                conductivity. The default is False.
            parallel: multiprocessing.Pool or boolean (optional)
                A multiprocessing pool object to use for the
                Mattis-Bardeen computation. The default is False, and
                the computation is done in serial. If True, a Pool
                object is created with multiprocessing.cpu_count() CPUs.
                Only used if low_energy is False.
        Returns:
            q: numpy.ndarray
                The quality factor.
        """
        q_inv = cls.mattis_bardeen(params, temperatures, low_energy=low_energy,
                                   parallel=parallel)
        q_inv += cls.two_level_systems(params, temperatures, powers)
        q_inv += cls.constant_loss(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q = 1 / q_inv
        return q

    @classmethod
    def residual(cls, params, data, temperatures=None, powers=None,
                 sigmas=None, low_energy=False, parallel=False):
        """
        Return the normalized residual between the Qi data and model.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            data: numpy.ndarray
                The Qi data.
            temperatures: numpy.ndarray (optional)
                The temperatures at which the quality factors are
                evaluated. The default is None and the temperature's
                effect on the quality factor is ignored.
            powers: numpy.ndarray (optional)
                The powers at which the quality factors are evaluated.
                The default is None and the power's effect on the
                quality factor is ignored.
            sigmas: numpy.ndarray (optional)
                The error associated with each data point. The default
                is None and the residual is not normalized.
            low_energy: boolean (optional)
                Use the low energy approximation to evaluate the complex
                conductivity. The default is False.
            parallel: multiprocessing.Pool or boolean (optional)
                A multiprocessing pool object to use for the
                Mattis-Bardeen computation. The default is False, and
                the computation is done in serial. If True, a Pool
                object is created with multiprocessing.cpu_count() CPUs.
                Only used if low_energy is False.
        Returns:
            residual: numpy.ndarray
                the normalized residuals.
        """
        q = cls.model(params, temperatures=temperatures, powers=powers,
                      low_energy=low_energy, parallel=parallel)
        residual = (q - data) / sigmas if sigmas is not None else (q - data)
        return residual

    @classmethod
    def guess(cls, data, f0, tc, alpha=0.5, bcs=BCS, temperatures=None,
              powers=None, gamma=1., fit_resonance=False, fit_mb=True,
              fit_tc=False, fit_alpha=True, fit_dynes=False, fit_tls=True,
              fit_fd=True, fit_pc=False, fit_loss=False):
        """
        Guess the model parameters based on the data. Returns a
        lmfit.Parameters() object.
        Args:
            data: numpy.ndarray
                The fr data in Hz.
            f0: float
                The resonance frequency in Hz at zero temperature.
            tc: float
                The transition temperature for the resonator in Kelvin.
            alpha: float (optional)
                The kinetic inductance fraction. The default is 0.5.
            bcs: float (optional)
                ∆ = bcs * kB * tc. The default is the usual BCS value.
            temperatures: numpy.ndarray (optional)
                The temperatures at which the Qi data is taken. The
                default is None. If specified, this helps set the fd
                parameter to a reasonable value.
            powers: numpy.ndarray (optional)
                The powers at which the Qi data is taken. The default is
                None. If specified, this helps set the pc parameter to a
                reasonable value.
            gamma: float (optional)
                The float corresponding to the superconducting limit of
                the resonator. The default is 1 which corresponds to the
                thin film, local limit. 1/2 is the thick film, local
                limit. 1/3 is the thick film, extreme anomalous limit.
            fit_resonance: boolean (optional)
                A boolean specifying if the offset frequency, f0, should
                be varied during the fit. The default is False.
            fit_mb: boolean (optional)
                A boolean specifying whether the Mattis-Bardeen
                parameters should be varied during the fit. The default
                is True.
            fit_tc: boolean (optional)
                A boolean specifying whether to fit Tc in the fit. The
                default is False and bcs is varied instead. Tc and bcs
                may still be fixed if fit_mbd is False.
            fit_alpha: boolean (optional)
                A boolean specifying whether to vary alpha during the
                fit. The default is True. alpha may still be not fit if
                fit_mbd is False.
            fit_dynes: float, boolean (optional)
                A boolean specifying whether to vary the dynes parameter
                in the fit. The default is False. If a float, this value
                is used for 'dynes'. If True, the 'dynes' is set to
                0.01, since the fit has trouble if 'dynes' is
                initialized to 0.
            fit_tls: boolean (optional)
                A boolean specifying whether to vary the TLS parameters
                during the fit. The default is True.
            fit_fd: boolean (optional)
                A boolean specifying whether to vary the TLS fraction
                and loss tangent factor during the fit. The default is
                True. fd may still not be varied if fit_tls is False.
            fit_pc: boolean (optional)
                A boolean specifying whether to vary the critical power
                during the fit. The default is False and the low power
                limit is used. If True, pc may still not be varied if
                fit_tls is False.
            fit_loss: boolean (optional)
                A boolean specifying whether to fit a constant loss to
                the data. The default is False.
        Returns:
            params: lmfit.Parameters
             An object with guesses and bounds for each parameter.
        """
        scale = 2 if fit_tls and fit_loss else 1
        # guess constant loss
        qi_inv_min = np.min(1 / data)
        q0_inv = qi_inv_min / scale if fit_loss else 0
        # guess two level system values
        if temperatures is not None:
            xi = h * f0 / (2 * kb * temperatures)
            fd = np.min(qi_inv_min / np.tanh(xi)) / scale if fit_tls else 0
        else:
            fd = 0
        pc = np.mean(powers) if powers is not None else 0
        dynes_sqrt = np.sqrt(0.01) if fit_dynes is True else np.sqrt(fit_dynes)

        # make the parameters object
        # (coerce all values to float to avoid ints and numpy types)
        params = lm.Parameters(usersyms={'scaled_alpha_inv': scaled_alpha_inv})
        # resonator params
        params.add("f0", value=float(f0), vary=fit_resonance, min=0)
        params.add("gamma", value=float(gamma), vary=False)
        # constant loss parameters
        params.add("q0_inv", value=float(q0_inv), vary=fit_loss)
        # two level system params
        params.add("fd_scaled", value=float(np.sqrt(fd * 1e6)),
                   vary=fit_tls and fit_fd)
        params.add("pc", value=float(pc if fit_pc else np.inf),
                   vary=fit_tls and fit_pc)
        # Mattis-Bardeen params
        params.add("tc", value=float(tc), vary=fit_mb and fit_tc, min=0)
        params.add("bcs", value=float(bcs), vary=fit_mb and not fit_tc, min=0)
        params.add("scaled_alpha", value=float(scaled_alpha(alpha)),
                   vary=fit_mb and fit_alpha)
        # Dynes params
        params.add("dynes_sqrt", value=float(dynes_sqrt), vary=bool(fit_dynes))
        # derived params
        params.add("alpha", expr='scaled_alpha_inv(scaled_alpha)')
        params.add("fd", expr='fd_scaled**2 * 1e-6')
        params.add("dynes", expr="dynes_sqrt**2")

        return params
