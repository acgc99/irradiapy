"""This module contains math utilities for the irradiapy package."""

# pylint: disable=unbalanced-tuple-unpacking

from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import curve_fit


def lorentzian(
    xs: np.ndarray,
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> Union[float, np.ndarray]:
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
    Union[float, np.ndarray]
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
    p0: Optional[np.ndarray] = None,
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
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry). Default is 1.0.

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


def gaussian(
    xs: np.ndarray,
    x_peak: float,
    linewidth: float,
    amplitude: float,
    asymmetry: float,
) -> Union[float, np.ndarray]:
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
    Union[float, np.ndarray]
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
    p0: Optional[np.ndarray] = None,
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
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry). Default is 1.0.

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
