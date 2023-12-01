import logging
import numbers
import numpy as np
import multiprocessing as mp
from functools import partial
from collections.abc import Collection

from mkidcalculator.models import S21
from mkidcalculator.io.loop import Loop
from mkidcalculator.io.sweep import Sweep
from mkidcalculator.io.resonator import Resonator
from mkidcalculator.io.utils import (_loop_fit_data, initialize_worker, map_async_stoppable, HAS_LOOPFIT, loopfit,
                                     _red_chi)
from mkidcalculator.models.utils import _compute_sigma

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
_loops = None

MAX_REDCHI = 100
FIT_MESSAGE = ("'{:s}' fit: label = '{:s}', "
               "\N{Mathematical Italic Small Chi}\N{Latin Subscript Small Letter V}\N{Superscript Two} = {:g}")


def _parallel(function, loops, n_cpu, fit_type, **kwargs):
    # If the memory is not freed, multiple child processes will try to open a zip container previously opened by the
    # parent process. When forking, the file id is not fully copied, so the child processes will error. This can be
    # avoided by removing any dependence on the npz files from the parent before the fork.
    for loop in loops:
        loop.free_memory()

    global _loops
    if n_cpu is True:
        n_cpu = mp.cpu_count() // 2
    try:
        # enforce fork since loop objects are too big to pickle over pipes
        _loops = loops  # global _loops gets copied efficiently over into subprocesses
        with mp.get_context("fork").Pool(n_cpu, initializer=initialize_worker) as pool:
            # do the fit
            fit = partial(function, parallel=False, return_dict=True, fit_type=fit_type, **kwargs)
            results = map_async_stoppable(pool, fit, np.arange(len(loops)))
            # wait for the results
            try:
                results.wait()
            except KeyboardInterrupt as error:
                log.error("Keyboard Interrupt encountered: retrieving computed fits before exiting")
                pool.terminate()
                pool.join()
                raise error
            finally:
                fit_results = fit_type + "_results"
                for index, result in enumerate(results.get()):
                    if result is not None:
                        setattr(loops[index], fit_results, result[0])  # data returned by pool are just the fit results
                log.info("Retrieved results from parallel computation")
    finally:
        _loops = None
    return loops


def _get_loops(data):
    if not isinstance(data, Collection):
        data = [data]
    loops = []
    for datum in data:
        if isinstance(datum, numbers.Integral):
            loops.append(_loops[datum])
        elif isinstance(datum, Loop):
            loops.append(datum)
        elif isinstance(datum, Resonator):
            loops += datum.loops
        elif isinstance(datum, Sweep):
            for resonator in datum.resonators:
                loops += resonator.loops
        else:
            message = "'data' object ({}) is not a Loop, Resonator, Sweep, or a collection of those objects."
            raise ValueError(message.format(type(data)))
    return loops


def _get_good_fits(loop, target, fit_type, fix=()):
    # find good fits from other loop
    good_guesses, powers, fields, temperatures = [], [], [], []
    try:
        loops = loop.resonator.loops
    except AttributeError:  # no resonator defined
        return good_guesses, powers, fields, temperatures
    for potential_loop in loops:
        if potential_loop is loop:  # don't use fits from this loop
            continue
        if np.isnan(getattr(potential_loop, target)):  # no nans in the target loop attribute
            continue
        different = [not np.isclose(getattr(potential_loop, fixed), getattr(loop, fixed), equal_nan=True)
                     for fixed in fix]
        if any(different):  # don't use fits with a different fixed attribute
            continue
        # only use fits that have redchi < MAX_REDCHI
        results_dict = getattr(potential_loop, fit_type + "_results")
        red_chi = _red_chi(fit_type, results_dict['best']) if "best" in results_dict.keys() else np.inf
        if red_chi < MAX_REDCHI:
            if fit_type == "lmfit":
                good_guesses.append(results_dict['best']['result'].params.copy())
            elif fit_type == "loopfit":
                good_guesses.append(results_dict['best'].copy())
            else:
                raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")
            powers.append(potential_loop.power)
            temperatures.append(potential_loop.field)
            fields.append(potential_loop.temperature)
    return good_guesses, powers, fields, temperatures


def _prepare_output(loops, fit_type, return_dict=False):
    fit_results = fit_type + "_results"
    if return_dict:
        return [getattr(loop, fit_results) for loop in loops]
    else:
        return loops


def _make_guess(loop, fit_type, kwargs, fit_kwargs):
    f = loop.f[loop.mask] if fit_kwargs.get('use_mask', False) else loop.f
    z = loop.z[loop.mask] if fit_kwargs.get('use_mask', False) else loop.z
    if fit_type == "lmfit":
        model = fit_kwargs.get("model", S21)
        guess = model.guess(z, f, **kwargs)
    elif fit_type == "loopfit":
        if not HAS_LOOPFIT:
            raise ImportError("The loopfit package is not installed.")
        guess = loopfit.guess(f, z=z, **kwargs)
    else:
        raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")
    return guess


def _do_fit(loop, guess, fit_type, fit_kwargs, sigma, label):
    z = loop.z[loop.mask] if fit_kwargs.get('use_mask', False) else loop.z
    if fit_type == "lmfit":
        kwargs = {"label": label}
        kwargs.update(fit_kwargs)
        model = fit_kwargs.get("model", S21)
        residual_kwargs = kwargs.pop("residual_kwargs", {})
        if sigma:
            residual_kwargs.update({"sigma": _compute_sigma(z) if sigma is True else sigma})
        result = loop.lmfit(model, guess, residual_kwargs=residual_kwargs, **kwargs)
    elif fit_type == "loopfit":
        kwargs = guess.copy()
        kwargs.update(fit_kwargs)
        kwargs.update({'label': label})
        if sigma:
            kwargs.update({"sigma": _compute_sigma(z) if sigma is True else sigma})
        result = loop.loopfit(**kwargs)
    else:
        raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")
    return result


def _set(fit_type, parameter, new, old, fit_kwargs, vary=None):
    if isinstance(old, dict):
        old = old[parameter]
    if fit_type == "lmfit":
        if vary is not None:
            new[parameter].set(value=old, vary=vary)
        else:
            new[parameter].set(value=old)
    elif fit_type == "loopfit":
        new[parameter] = old
        if vary is not None:
            if parameter == "a":
                fit_kwargs['nonlinear'] = vary
            elif parameter in ['gain0', 'gain1', 'gain2', 'phase0', 'phase1']:
                fit_kwargs['baseline'] = vary
            elif parameter in ['alpha', 'beta']:
                fit_kwargs['imbalance'] = vary
            elif parameter in ['gamma', 'delta']:
                fit_kwargs['offset'] = vary
    else:
        raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")


def basic_fit(data, fit_type="lmfit", label="basic_fit", calibration=True, sigma=True, guess=None,
              guess_kwargs=None, parallel=False, return_dict=False, callback=None, **fit_kwargs):
    """
    Fit the loop using the standard model guess.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The label to store the fit results under. The default is
            "basic_fit".
        calibration: boolean
            Automatically add 'offset' and 'imbalance' parameters to the guess
            keywords. The default is True, but if a model is used that doesn't
            have those keywords, it should be set to False.
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        guess: object
            The guess object for the 'fit_type'. Defaults to None and the guess
            is computed with 'guess_kwargs'. A different guess can be applied
            to each loop if a list of guesses is provided.
        guess_kwargs: dictionary
            A dictionary of keyword arguments that can overwrite the default
            options for model.guess().
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the fit results dictionary if True. The default is
            False.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the fitting function. E.g.
            loop.lmfit() or loop.loopfit().
    Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    loops = _get_loops(data)
    if parallel:
        loops = _parallel(basic_fit, loops, parallel, fit_type, label=label, calibration=calibration,
                          sigma=sigma, guess=guess, guess_kwargs=guess_kwargs, **fit_kwargs)
        return _prepare_output(loops, fit_type, return_dict=return_dict)
    for i, loop in enumerate(loops):
        # make guess
        if guess is None:
            kwargs = {"imbalance": loop.imbalance_calibration, "offset": loop.offset_calibration} if calibration else {}
            if guess_kwargs is not None:
                kwargs.update(guess_kwargs)
            g = _make_guess(loop, fit_type, kwargs, fit_kwargs)
        else:
            g = guess[i] if isinstance(guess, (tuple, list)) else guess
        # do fit
        result = _do_fit(loop, g, fit_type, fit_kwargs, sigma, label)
        log.info(FIT_MESSAGE.format(loop.name, label, _red_chi(fit_type, result)))
        if callback is not None:
            callback()
    return _prepare_output(loops, fit_type, return_dict=return_dict)


def power_fit(data, fit_type="lmfit", label="power_fit", sigma=True, baseline=None, parallel=False, return_dict=False,
              callback=None, **fit_kwargs):
    """
    Fit the loop using the two nearest power data points of similar
    temperature and same field in the resonator as guesses. If there are no
    good guesses, nothing will happen.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit. The loops must be associated
            with resonator objects.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The label to store the fit results under. The default is
            "power_fit".
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the lmfit_results dictionary if True. The default is
            False.
        baseline: tuple of strings (optional)
            A list of parameter names corresponding to the baseline. They will
            use the model guess from the best fit done so far on this loop as a
            starting point. If no fit has been done a AttributeError will be
            raised.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the fitting function. E.g.
            loop.lmfit() or loop.loopfit().
     Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    loops = _get_loops(data)
    if parallel:
        loops = _parallel(power_fit, loops, parallel, fit_type, label=label, sigma=sigma, baseline=baseline,
                          **fit_kwargs)
        return _prepare_output(loops, fit_type, return_dict=return_dict)
    for loop in loops:
        # check that at least one other fit has been done first
        if "best" not in getattr(loop, fit_type + "_results").keys():
            raise AttributeError("loop does not have a previous fit on which to base the power fit.")
        # find good fits from other loop
        good_guesses, powers, _, temperatures = _get_good_fits(loop, "power", fit_type, fix=("field",))
        # get the guesses nearest in power and temperature data sets
        distance = np.empty(len(good_guesses), dtype=[('power', np.float), ('temperature', np.float)])
        distance['power'] = np.abs(loop.power - np.array(powers))
        distance['temperature'] = np.abs(loop.temperature - np.array(temperatures))
        indices = np.argsort(distance, order=("temperature", "power"))  # closest in temperature then in power
        # fit the two nearest data sets
        used_powers = []
        for index in indices:
            if len(used_powers) >= 2:
                break
            if powers[index] not in used_powers:  # don't try the same power more than once
                used_powers.append(powers[index])
                # pick guess
                guess = good_guesses[index]
                # do fit
                fit_label = label + "_" + str(len(used_powers) - 1)
                result = _do_fit(loop, guess, fit_type, fit_kwargs, sigma, fit_label)
                log.info(FIT_MESSAGE.format(loop.name, fit_label, _red_chi(fit_type, result)))
        if callback is not None:
            callback()
    return _prepare_output(loops, fit_type, return_dict=return_dict)


def temperature_fit(data, fit_type="lmfit", label="temperature_fit", sigma=True, parallel=False, return_dict=False,
                    callback=None, **fit_kwargs):
    """
    Fit the loop using the two nearest temperature data points of the same
    power and field in the resonator as guesses. If there are no good guesses,
    nothing will happen.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit. The loops must be associated
            with resonator objects.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The label to store the fit results under. The default is
            "temperature_fit".
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the lmfit_results dictionary if True. The default is
            False.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the fitting function. E.g.
            loop.lmfit() or loop.loopfit().
     Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    loops = _get_loops(data)
    if parallel:
        loops = _parallel(temperature_fit, loops, parallel, fit_type, label=label, sigma=sigma, **fit_kwargs)
        return _prepare_output(loops, fit_type, return_dict=return_dict)
    for loop in loops:
        # find good fits from other loop
        good_guesses, _, _, temperatures = _get_good_fits(loop, "temperature", fit_type, fix=("power", "field"))
        # fit the two nearest temperature data sets
        indices = np.argsort(np.abs(loop.temperature - np.array(temperatures)))
        for iteration in range(2):
            if iteration < len(indices):
                # pick guess
                guess = good_guesses[indices[iteration]]
                # do fit
                fit_label = label + "_" + str(iteration)
                result = _do_fit(loop, guess, fit_type, fit_kwargs, sigma, fit_label)
                log.info(FIT_MESSAGE.format(loop.name, fit_label, _red_chi(fit_type, result)))
        if callback is not None:
            callback()
    return _prepare_output(loops, fit_type, return_dict=return_dict)


def linear_fit(data, fit_type="lmfit", label="linear_fit", sigma=True, parameter=None, parallel=False,
               return_dict=False, callback=None, **fit_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity turned
    off.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The label to store the fit results under. The default is
            "nonlinear_fit".
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        parameter: string (optional)
            The nonlinear parameter name to use.
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the lmfit_results dictionary if True. The default is
            False.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the fitting function. E.g.
            loop.lmfit() or loop.loopfit().
    Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    return nonlinear_fit(data, fit_type=fit_type, label=label, sigma=sigma, parameter=parameter, value=0, vary=False,
                         parallel=parallel, return_dict=return_dict, callback=callback, **fit_kwargs)


def nonlinear_fit(data, fit_type="lmfit", label="nonlinear_fit", sigma=True, parameter=None, value=None, vary=True,
                  parallel=False, return_dict=False, callback=None, **fit_kwargs):
    """
    Fit the loop using a previous good fit, but with the nonlinearity.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The label to store the fit results under. The default is
            "nonlinear_fit".
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        parameter: string (optional)
            The nonlinear parameter name and value to use. If None, the default
            for the fit_type is used.
        value: float (optional)
            The nonlinear parameter name and value to use. If None, the default
            for the fit_type is used.
        vary: boolean (optional)
            Determines if the nonlinearity is varied in the fit. The default is
            True.
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the fit results dictionary if True. The default is
            False.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the fitting function. E.g.
            loop.lmfit() or loop.loopfit().
    Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    loops = _get_loops(data)
    if parallel:
        loops = _parallel(nonlinear_fit, loops, parallel, fit_type, label=label, sigma=sigma, parameter=parameter,
                          value=value, vary=vary, **fit_kwargs)
        return _prepare_output(loops, fit_type, return_dict=return_dict)
    for loop in loops:
        # make guess
        results_dict = getattr(loop, fit_type + "_results")
        if "best" in results_dict.keys():
            # only fit if previous fit has been done
            if fit_type == "lmfit":
                guess = loop.lmfit_results["best"]["result"].params.copy()
                if parameter is None:
                    parameter = "a_sqrt"
                if value is None:
                    value = 0.05
            elif fit_type == "loopfit":
                guess = loop.loopfit_results["best"].copy()
                if parameter is None:
                    parameter = "a"
                if value is None:
                    value = 0.0025
            else:
                raise ValueError("'fit_type' must be either 'loopfit' or 'lmfit'")
            _set(fit_type, parameter, guess, value, fit_kwargs, vary=vary)
            # do fit
            result = _do_fit(loop, guess, fit_type, fit_kwargs, sigma, label)
            log.info(FIT_MESSAGE.format(loop.name, label, _red_chi(fit_type, result)))
        else:
            raise AttributeError("loop does not have a previous fit on which to base the nonlinear fit.")
        if callback is not None:
            callback()
    return _prepare_output(loops, fit_type, return_dict=return_dict)


def multiple_fit(data, fit_type="lmfit", extra_fits=None, sigma=True, iterations=2, parallel=False,
                 return_dict=False, callback=None, fit_kwargs=None, **basic_fit_kwargs):
    """
    Fit the loops using multiple methods.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The loop or loops to fit. If Resonator or Sweep objects are given
            all of the contained loops are fit.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        extra_fits: tuple of functions (optional)
            Extra functions to use to try to fit the loops. They must have
            the arguments of basic_fit(). The default is None and we use
            (temperature_fit, power_fit, nonlinear_fit, linear_fit). The loops
            must be associated with resonator objects for the temperature_fit
            and power_fit to work.
        sigma: boolean, complex number, or numpy.ndarray
            If True, the standard deviation of the S21 data is computed from
            the loop. It can also be specified with a complex number or an
            array of complex numbers. The value is then passed to the model
            residual with the sigma keyword argument. If the model does not
            have this key word, this will not work and sigma should be set to
            False.
        iterations: integer (optional)
            Number of times to run the extra_fits. The default is 2. This is
            useful for when the extra_fits use fit information from other loops
            in the resonator.
        parallel: boolean or integer (optional)
            Compute the fit for each loop in parallel. The default is False,
            and the computation is done in serial. If True, a Pool object is
            created with multiprocessing.cpu_count() // 2 CPUs. If an integer,
            that many CPUs will be used. This method will only work on systems
            that can use os.fork().
        return_dict: boolean (optional)
            Return only the lmfit_results dictionary if True. The default is
            False.
        callback: function (optional)
            A function to be called after every loop fit. The default is None
            and no function call is done.
        fit_kwargs: dictionary or iterable of dictionaries (optional)
            Extra keyword arguments to send to the extra_fits. The default is
            None and no extra keywords are used. If a single dictionary is
            given, it will be used for all of the extra fits.
        basic_fit_kwargs: optional keyword arguments
            Additional keyword arguments to pass to the basic_fit function
            before the extra fits are used.
    Returns:
        loops: a list of mkidcalculator.Loop objects or a list of dictionaries
            The loop objects that were fit. If return_dict is True, the
            loop.lmfit_results or loop.loopfit_results dictionaries are
            returned instead.
    """
    if extra_fits is None:
        extra_fits = (temperature_fit, power_fit, nonlinear_fit, linear_fit)
    loops = _get_loops(data)
    # fit the resonator loops with the basic fit
    log.info("starting {}".format(basic_fit))
    kwargs = {"fit_type": fit_type, "sigma": sigma, "parallel": parallel, "callback": callback}
    kwargs.update(basic_fit_kwargs)
    basic_fit(loops, **kwargs)
    # setup extra fit kwargs
    if fit_kwargs is None:
        fit_kwargs = [{}] * len(extra_fits)
    if isinstance(fit_kwargs, dict):
        fit_kwargs = [fit_kwargs] * len(extra_fits)
    # do the extra fits loop <iterations> times
    for iteration in range(iterations):
        log.info("starting iteration: {}".format(iteration))
        # do the extra fits
        for extra_index, fit in enumerate(extra_fits):
            log.info("starting {}".format(fit))
            kwargs = {"label": fit.__name__ + str(iteration), "fit_type": fit_type, "sigma": sigma,
                      "parallel": parallel, "callback": callback}
            kwargs.update(fit_kwargs[extra_index])
            fit(loops, **kwargs)
    return _prepare_output(loops, fit_type, return_dict=return_dict)


def loop_fit_data(data, parameters=("chi2",), fit_type="lmfit", label='best', bounds=None, errorbars=None, success=None,
                  power=None, field=None, temperature=None):
    """
    Collect fit information from Loop fits into arrays.
    Args:
        data: Loop, Resonator, Sweep, or collection of those objects
            The fitted loop or loops to extract information from. If Resonator
            or Sweep objects are given all of the contained loops are used.
        parameters: tuple of strings
            The fit parameters to report. "chi2" can be used to retrieve
            the reduced chi squared values. The default is to just return chi2.
            "_sigma" can be appended to any parameter name to retrieve the
            standard error for the estimate. "power", "field", and
            "temperature" can be used to return those values corresponding
            to the other returned fit parameters.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit" and
            "loopfit". The default is "lmfit".
        label: string (optional)
            The fit label to use.
        bounds: tuple of numbers or tuples
            The bounds for the parameters. It must be a tuple of the same
            length as the parameters keyword argument. Each element is either
            an upper bound on the parameter or a tuple, e.g. (lower bound,
            upper bound). None can be used as a placeholder to skip a
            parameter. The default is None and no bounds are used.
        errorbars: boolean
            If errorbars is True, only data from loop fits that could compute
            errorbars on the fit parameters is included. If errorbars is False,
            only data from loop fits that could not compute errorbars on the
            fit parameters is included. The default is None, and no filtering
            on the errorbars is done. This keyword has no effect if the
            fit_type is "loopfit" since no error bars are computed.
        success: boolean
            If success is True, only data from successful loop fits is
            included. If False, only data from failed loop fits is
            included. The default is None, and no filtering on fit success is
            done. Note: fit success is typically a bad indicator on fit
            quality. It only ever fails when something really bad happens.
        power: tuple of two number tuples or numbers
            Inclusive range or ranges of powers to return. A single number
            will cause only that value to be returned. The default is to
            include all of the powers.
        field: tuple of two number tuples or numbers
            Inclusive range or ranges of fields to return. A single number
            will cause only that value to be returned. The default is to
            include all of the fields.
        temperature: tuple of two number tuples or numbers
            Inclusive range or ranges of temperatures to return. A single
            number will cause only that value to be returned. The default is
            to include all of the temperatures.
    Returns:
        outputs: tuple of numpy.ndarray objects
            The outputs in the same order as parameters.
    """
    loops = _get_loops(data)
    return _loop_fit_data(loops, parameters=parameters, fit_type=fit_type, label=label, bounds=bounds,
                          errorbars=errorbars, success=success, power=power, field=field, temperature=temperature)
