import numpy as np


def swenson(y0, a, increasing=True, numeric=False):
    """
    Returns the nonlinear y parameter from Swenson et al.
    (doi: 10.1063/1.4794808) using the formula from McCarrick et al. when
    applicable (doi: 10.1063/1.4903855).
    Args:
        y0: numpy.ndarray
            Generator detuning in units of linewidths for a = 0.
        a: float
            The nonlinearity parameter.
        increasing: boolean
            The direction of the frequency sweep.
        numeric: boolean
            Use numpy.roots iteratively to solve for y when the McCarrick
            formula doesn't work. If False, the roots are computed using
            real_cubic_roots(). The default is False.
    """
    y = np.empty(y0.shape)
    if increasing:
        # try to compute analytically using simple formula (faster)
        a2 = ((y0 / 3)**3 + y0 / 12 + a / 8)**2 - ((y0 / 3)**2 - 1 / 12)**3
        k2 = np.empty(a2.shape)
        k2[a2 >= 0] = np.sqrt(a2[a2 >= 0])
        k2[a2 < 0] = 0
        a1 = a / 8 + y0 / 12 + k2 + (y0 / 3)**3
        analytic = np.logical_and(a2 >= 0, a1 != 0)  # formula breaks down here
        k1 = np.sign(a1[analytic]) * np.abs(a1[analytic])**(1 / 3)  # need the real branch if a1 < 0
        y0_analytic = y0[analytic]
        y[analytic] = y0_analytic / 3 + ((y0_analytic / 3)**2 - 1 / 12) / k1 + k1
        # use numeric calculation if required (slower)
        logic = np.logical_not(analytic)
        if numeric:
            y_numeric = np.empty_like(y0[logic])
            for ii, y0_ii in enumerate(y0[logic]):
                roots = np.roots([4, -4 * y0_ii, 1, -(y0_ii + a)])
                y_numeric[ii] = np.min(roots[np.isreal(roots)].real)
            y[logic] = y_numeric
        else:  # slightly faster
            y[logic] = real_cubic_roots(np.full_like(y0[logic], 4), -4 * y0[logic],
                                        np.ones_like(y0[logic]), -y0[logic] - a, select='min')
    else:
        # no known simple analytic formulas for the other direction (yet)
        if numeric:
            for ii, y0_ii in enumerate(y0):
                roots = np.roots([4, -4 * y0_ii, 1, -(y0_ii + a)])
                y[ii] = np.max(roots[np.isreal(roots)].real)
        else:  # slightly faster
            y = real_cubic_roots(np.full_like(y0, 4), -4 * y0, np.ones_like(y0), -y0 - a, select='max')
    return y


def real_cubic_roots(c3, c2, c1, c0, select='min'):
    """
    Returns the real roots of
    c3 * x^3 + c2 * x^2 + c1 * x + c0 = 0.
    Follows the formula laid out in
    https://www.e-education.psu.edu/png520/m11_p6.html (Accessed April 6, 2020)
    Args:
        c3: numpy.ndarray or float
            The third order coefficient.
        c2: numpy.ndarray or float
            The second order coefficient.
        c1: numpy.ndarray or float
            The first order coefficient.
        c0: numpy.ndarray or float
            The constant coefficient.
        select: string
            The default is 'min' and the smallest real root is returned. 'max'
            and 'middle' return the largest and intermediate real roots. The
            'middle' option will return the largest root if 'c3' = 0.
    Returns:
        roots: numpy.ndarray
            A numpy array of the real roots.
    """
    c3 = np.array(c3, ndmin=1, dtype=float, copy=False)
    c2 = np.array(c2, ndmin=1, dtype=float, copy=False)
    c1 = np.array(c1, ndmin=1, dtype=float, copy=False)
    c0 = np.array(c0, ndmin=1, dtype=float, copy=False)
    roots = np.empty_like(c3)

    quad = (c3 == 0)
    if select.lower().startswith(('max', 'middle')):
        roots[quad] = (-c1[quad] + np.sqrt(c1[quad]**2 - 4 * c2[quad] * c0[quad])) / (2 * c2[quad])
    elif select.lower().startswith('min'):
        roots[quad] = (-c1[quad] - np.sqrt(c1[quad]**2 - 4 * c2[quad] * c0[quad])) / (2 * c2[quad])

    cube = (c3 != 0)
    roots_cube = np.empty_like(c3[cube])
    a = c2[cube] / c3[cube]
    b = c1[cube] / c3[cube]
    c = c0[cube] / c3[cube]
    q = np.asarray((a**2 - 3 * b) / 9, dtype=complex)  # allow sqrt(q) to be complex
    r = np.asarray((2 * a**3 - 9 * a * b + 27 * c) / 54, dtype=complex)  # allow sqrt(r) to be complex

    less = (r.real**2 <= q.real**3)
    three_roots = np.array([-2 * np.sqrt(q[less]) * np.cos(np.arccos(r[less] / np.sqrt(q[less]**3)) / 3) - a[less] / 3,
                            -2 * np.sqrt(q[less]) * np.cos((np.arccos(r[less] / np.sqrt(q[less]**3)) + 2 * np.pi) / 3)
                            - a[less] / 3,
                            -2 * np.sqrt(q[less]) * np.cos((np.arccos(r[less] / np.sqrt(q[less]**3)) - 2 * np.pi) / 3)
                            - a[less] / 3]).real
    if select.lower().startswith('max'):
        roots_cube[less] = np.max(three_roots, axis=0)
    elif select.lower().startswith('middle'):
        roots_cube[less] = np.median(three_roots, axis=0)
    elif select.lower().startswith('min'):
        roots_cube[less] = np.min(three_roots, axis=0)

    greater = (r.real**2 > q.real**3)
    s = -np.sign(r[greater].real) * np.cbrt(np.abs(r[greater]).real + np.sqrt(r[greater]**2 - q[greater]**3).real)
    t = np.empty_like(s)
    t[s == 0] = 0
    t[s != 0] = q[greater][s != 0].real / s[s != 0]
    roots_cube[greater] = s + t - a[greater] / 3

    roots[cube] = roots_cube

    return roots
