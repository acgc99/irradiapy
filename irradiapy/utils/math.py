"""This module contains math utilities for the irradiapy package."""

# pylint: disable=unbalanced-tuple-unpacking

from typing import Callable

import numpy as np
from numpy import typing as npt
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
    xs: npt.NDArray[np.float64],
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> float | npt.NDArray[np.float64]:
    """Evaluate a Lorentzian function.

    Parameters
    ----------
    xs : npt.NDArray[np.float64]
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
    float | npt.NDArray[np.float64]
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
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    p0: None | npt.NDArray[np.float64] = None,
    asymmetry: float = 1.0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
]:
    """Fit data to a Lorentzian function.

    Parameters
    ----------
    xs : npt.NDArray[np.float64]
        X values where the function is evaluated.
    ys : npt.NDArray[np.float64]
        Y values at the given xs.
    p0 : npt.NDArray[np.float64], optional (default=None)
        Initial guess of fit parameters. If None, a guess is generated.
    asymmetry : float, optional (default=1.0)
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).

    Returns
    -------
    popt : npt.NDArray[np.float64]
        Optimal values for the parameters.
    pcov : npt.NDArray[np.float64]
        Covariance of popt.
    fit_function : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
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

    def fit_function(xs_fit: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return lorentzian(xs_fit, *popt)

    return popt, pcov, fit_function


# region Gaussian


def gaussian(
    xs: npt.NDArray[np.float64],
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> float | npt.NDArray[np.float64]:
    """Evaluate a Gaussian function.

    Parameters
    ----------
    xs : npt.NDArray[np.float64]
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
    float | npt.NDArray[np.float64]
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
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    p0: None | npt.NDArray[np.float64] = None,
    asymmetry: float = 1.0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
]:
    """Fit data to a Gaussian function.

    Parameters
    ----------
    xs : npt.NDArray[np.float64]
        X values where the function is evaluated.
    ys : npt.NDArray[np.float64]
        Y values at the given xs.
    p0 : npt.NDArray[np.float64], optional (default=None)
        Initial guess of fit parameters. If None, a guess is generated.
    asymmetry : float, optional (default=1.0)
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).

    Returns
    -------
    popt : npt.NDArray[np.float64]
        Optimal values for the parameters.
    pcov : npt.NDArray[np.float64]
        Covariance of popt.
    fit_function : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
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

    def fit_function(xs_fit: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return gaussian(xs_fit, *popt)

    return popt, pcov, fit_function


# region Power law


def scaling_law(
    x: npt.NDArray[np.float64], a: float, s: float
) -> npt.NDArray[np.float64]:
    """Evaluate the scaling law function a / x**s.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Input values.
    a : float
        Prefactor.
    s : float
        Exponent.

    Returns
    -------
    npt.NDArray[np.float64]
        Evaluated scaling law.
    """
    return a / x**s


def fit_scaling_law(
    centers: npt.NDArray[np.float64], counts: npt.NDArray[np.float64]
) -> tuple[float, float, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Fit a scaling law to the given histogram data.

    Parameters
    ----------
    centers : npt.NDArray[np.float64]
        The centers of the bins.
    counts : npt.NDArray[np.float64]
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

    def fit_function(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return scaling_law(x, a, s)

    return a, s, fit_function
