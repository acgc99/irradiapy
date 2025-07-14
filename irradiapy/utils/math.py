"""This module contains math utilities for the irradiapy package."""

# pylint: disable=unbalanced-tuple-unpacking

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


def repeated_prime_factors(n: int) -> list[int]:
    """Return the prime factors of n (with repetition).

    Parameters
    ----------
    n : int
        The number to factorize.

    Returns
    -------
    list[int]
        List of prime factors of n, including repetitions.
    """
    facs = []
    # Factor out 2s
    while n % 2 == 0:
        facs.append(2)
        n //= 2
    # Factor odd primes up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            facs.append(f)
            n //= f
        f += 2
    if n > 1:
        facs.append(n)
    return facs


# region Lorentzian


def lorentzian(
    xs: np.ndarray,
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> float | np.ndarray:
    """Evaluate a Lorentzian function.

    Parameters
    ----------
    xs : np.ndarray
        Where to evaluate the function.
    x_peak : float
        Position with maximum value.
    linewidth : float
        Linewidth.
    amplitude : float
        Maximum amplitude.
    asymmetry : float
        Asymmetry.

    Returns
    -------
    float | np.ndarray
        Evaluated Lorentzian function.

    References
    ----------
    See https://doi.org/10.1016/j.nimb.2021.05.014
    """
    delta_x = xs - x_peak
    linewidth_sq = linewidth**2
    exp_term = np.exp(asymmetry * delta_x)
    alpha = (1.0 + exp_term) ** 2
    alpha_quarter = alpha / 4.0
    exponent = -alpha_quarter * delta_x**2 / (2.0 * linewidth_sq)
    return amplitude * alpha_quarter * np.exp(exponent)


def fit_lorentzian(
    xs: np.ndarray,
    ys: np.ndarray,
    p0: None | np.ndarray = None,
    asymmetry: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Fit data to a Lorentzian function.

    Parameters
    ----------
    xs : np.ndarray
        X values where the function is evaluated.
    ys : np.ndarray
        Y values at the given xs.
    p0 : np.ndarray, optional
        Initial guess of fit parameters. If None, a guess is generated. Default is None.
    asymmetry : float, optional
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).
        Default is 1.0.

    Returns
    -------
    popt : np.ndarray
        Optimal values for the parameters.
    pcov : np.ndarray
        Covariance of popt.
    fit_function : Callable[[np.ndarray], np.ndarray]
        Function that evaluates the fitted Lorentzian.
    """
    if p0 is None:
        peak_index = np.argmax(ys)
        x_start = xs[0]
        x_end = xs[-1]
        sigma_guess = 0.5 * (x_end - x_start)
        p0 = np.array(
            [
                xs[peak_index],
                sigma_guess,
                ys[peak_index],
                0.0,
            ]
        )
    x_start = xs[0]
    x_end = xs[-1]
    x_sum = x_start + x_end
    popt, pcov = curve_fit(
        lorentzian,
        xs,
        ys,
        p0=p0,
        bounds=(
            [x_start, 0.0, ys.min(), -asymmetry],
            [x_end, x_sum, ys.max(), asymmetry],
        ),
    )

    def fit_function(xs_fit: np.ndarray) -> np.ndarray:
        return lorentzian(xs_fit, *popt)

    return popt, pcov, fit_function


# region Gaussian


def gaussian(
    xs: np.ndarray,
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> float | np.ndarray:
    """Evaluate a Gaussian function.

    Parameters
    ----------
    xs : np.ndarray
        Where to evaluate the function.
    x_peak : float
        Position with maximum value.
    linewidth : float
        Linewidth.
    amplitude : float
        Maximum amplitude.
    asymmetry : float
        Asymmetry.

    Returns
    -------
    float | np.ndarray
        Evaluated Gaussian function.

    References
    ----------
    See https://doi.org/10.1016/j.nimb.2021.05.014
    """
    delta_x = xs - x_peak
    linewidth_sq = linewidth**2
    exp_term = np.exp(asymmetry * delta_x)
    alpha = (1.0 + exp_term) ** 2
    alpha_quarter = alpha / 4.0
    exponent = -alpha_quarter * delta_x**2 / (2.0 * linewidth_sq)
    return amplitude * alpha_quarter * np.exp(exponent)


def fit_gaussian(
    xs: np.ndarray,
    ys: np.ndarray,
    p0: None | np.ndarray = None,
    asymmetry: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Fit data to a Gaussian function.

    Parameters
    ----------
    xs : np.ndarray
        X values where the function is evaluated.
    ys : np.ndarray
        Y values at the given xs.
    p0 : np.ndarray, optional
        Initial guess of fit parameters. If None, a guess is generated. Default is None.
    asymmetry : float, optional
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).
        Default is 1.0.

    Returns
    -------
    popt : np.ndarray
        Optimal values for the parameters.
    pcov : np.ndarray
        Covariance of popt.
    fit_function : Callable[[np.ndarray], np.ndarray]
        Function that evaluates the fitted Gaussian.
    """
    if p0 is None:
        peak_index = np.argmax(ys)
        x_start = xs[0]
        x_end = xs[-1]
        sigma_guess = 0.5 * (x_end - x_start)
        p0 = np.array(
            [
                xs[peak_index],
                sigma_guess,
                ys[peak_index],
                0.0,
            ]
        )
    x_start = xs[0]
    x_end = xs[-1]
    x_sum = x_start + x_end
    popt, pcov = curve_fit(
        gaussian,
        xs,
        ys,
        p0=p0,
        bounds=(
            [x_start, 0.0, ys.min(), -asymmetry],
            [x_end, x_sum, ys.max(), asymmetry],
        ),
    )

    def fit_function(xs_fit: np.ndarray) -> np.ndarray:
        return gaussian(xs_fit, *popt)

    return popt, pcov, fit_function


# region Power law


def scaling_law(x: np.ndarray, a: float, s: float) -> np.ndarray:
    """Evaluate the scaling law function a / x**s.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : float
        Prefactor.
    s : float
        Exponent.

    Returns
    -------
    np.ndarray
        Evaluated scaling law.
    """
    return a / x**s


def fit_scaling_law(
    centers: np.ndarray, counts: np.ndarray
) -> tuple[float, float, Callable[[np.ndarray], np.ndarray]]:
    """Fit a scaling law to the given histogram data.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the bins.
    counts : np.ndarray
        The values of the histogram.

    Returns
    -------
    tuple
        A tuple containing: the prefactor of the scaling law, the exponent of the scaling law,
        and the scaling law function.
    """
    popt, _ = curve_fit(lambda x, a, b: a + b * x, np.log10(centers), np.log10(counts))
    a, s = popt
    a, s = 10.0**a, -s

    def fit_function(x: np.ndarray) -> np.ndarray:
        return scaling_law(x, a, s)

    return a, s, fit_function
