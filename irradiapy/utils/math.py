"""This module contains math utilities for the irradiapy package."""

# pylint: disable=unbalanced-tuple-unpacking

from collections import defaultdict
from typing import Any, Callable

import numpy as np
from numpy import typing as npt
from numpy.lib.recfunctions import structured_to_unstructured as str2unstr
from scipy.optimize import curve_fit

# region Math


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


def power_law(
    x: npt.NDArray[np.float64], a: float, k: float
) -> npt.NDArray[np.float64]:
    """Evaluate a power law: y = a * x**k

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Input values.
    a : float
        Prefactor.
    k : float
        Exponent.

    Returns
    -------
    npt.NDArray[np.float64]
        Evaluated power law.
    """
    return a * x**k


def fit_power_law(
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    yerrs: None | npt.NDArray[np.float64] = None,
) -> tuple[float, float, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Fit a power law to the given histogram data: y = a * x**k.

    Note
    ----
    The fit is done in log-log space, so the input data should be positive.

    Parameters
    ----------
    xs : npt.NDArray[np.float64]
        X values where the function is evaluated.
    ys : npt.NDArray[np.float64]
        Y values at the given xs.
    yerrs : npt.NDArray[np.float64], optional
        Y errors.

    Returns
    -------
    tuple[float, float, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]
        Tuple containing:
        - (a, k): Fitted parameters of the power law.
        - (error_a, error_k): Errors of the fitted parameters.
        - fit_function: Function that evaluates the fitted power law.
    """
    # Power law: y = a * x**k
    # Fitted in log-log space: log(y) = log(a) + k * log(x)
    popt, popv = curve_fit(
        lambda x, a, b: a + b * x, np.log10(xs), np.log10(ys), sigma=yerrs
    )
    a = 10.0 ** popt[0]
    k = popt[1]
    errors = np.sqrt(np.diag(popv))
    error_log_a = errors[0]
    error_a = a * error_log_a * np.log(10)
    error_k = errors[1]

    def fit_function(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return power_law(x, a, k)

    return (a, k), (error_a, error_k), fit_function


# region Atoms quick calculations


def apply_boundary_conditions(
    data_atoms: defaultdict[str, Any],
    x: bool,
    y: bool,
    z: bool,
) -> defaultdict[str, Any]:
    """Apply boundary conditions to atoms.

    Parameters
    ----------
    data_atoms : defaultdict[str, Any]
        Atoms data containing atom positions and boundaries.
    x : bool
        Whether to apply periodic boundary conditions in the x direction.
    y : bool
        Whether to apply periodic boundary conditions in the y direction.
    z : bool
        Whether to apply periodic boundary conditions in the z direction.

    Returns
    -------
    defaultdict[str, Any]
        Updated atoms data with applied boundary conditions.
    """
    xlo, xhi = data_atoms["xlo"], data_atoms["xhi"]
    ylo, yhi = data_atoms["ylo"], data_atoms["yhi"]
    zlo, zhi = data_atoms["zlo"], data_atoms["zhi"]
    dx = xhi - xlo
    dy = yhi - ylo
    dz = zhi - zlo
    atoms = data_atoms["atoms"]
    if x:
        atoms["x"] = ((atoms["x"] - xlo) % dx) + xlo
        data_atoms["boundary"][0] = "pp"
    else:
        atoms["x"] = np.clip(atoms["x"], xlo, xhi)
        data_atoms["boundary"][0] = "ff"
    if y:
        atoms["y"] = ((atoms["y"] - ylo) % dy) + ylo
        data_atoms["boundary"][1] = "pp"
    else:
        atoms["y"] = np.clip(atoms["y"], ylo, yhi)
        data_atoms["boundary"][1] = "ff"
    if z:
        atoms["z"] = ((atoms["z"] - zlo) % dz) + zlo
        data_atoms["boundary"][2] = "pp"
    else:
        atoms["z"] = np.clip(atoms["z"], zlo, zhi)
        data_atoms["boundary"][2] = "ff"
    data_atoms["atoms"] = atoms
    return data_atoms


def recombine_in_radius(
    data_defects: defaultdict[str, Any], radius: float
) -> defaultdict[str, Any]:
    """Recombine defects (interstitials and vacancies) within a given radius.

    Takes into account periodic boundary conditions.

    Parameters
    ----------
    data_defects : defaultdict[str, Any]
        Defects data containing defect positions and boundaries.
    radius : float
        Radius within which to recombine defects.

    Returns
    -------
    defaultdict[str, Any]
        Updated defects data with recombined defects.
    """

    defects = data_defects["atoms"]
    boundary = data_defects["boundary"]
    xlo, xhi = data_defects["xlo"], data_defects["xhi"]
    ylo, yhi = data_defects["ylo"], data_defects["yhi"]
    zlo, zhi = data_defects["zlo"], data_defects["zhi"]

    cond = defects["type"] == 0
    vacs = defects[cond]
    sias = defects[~cond]

    box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])
    box_size = box[:, 1] - box[:, 0]

    vac_pos = str2unstr(vacs[["x", "y", "z"]])
    sia_pos = str2unstr(sias[["x", "y", "z"]])
    # Mask to keep track of recombined interstitials
    sia_used = np.zeros(len(sias), dtype=bool)
    # List to keep indices of vacancies and interstitials to remove
    vac_to_remove = []
    sia_to_remove = []

    radius2 = radius**2
    # For each vacancy, find closest interstitial within radius
    for i, vpos in enumerate(vac_pos):
        # Compute vector distances to all interstitials
        delta = sia_pos - vpos
        # Apply minimum image convention for periodic boundary conditions
        for d in range(3):
            if boundary[d] == "pp":
                delta[:, d] -= box_size[d] * np.round(delta[:, d] / box_size[d])
        dist2 = np.sum(np.square(delta), axis=1)
        # Mask out already recombined interstitials
        dist2[sia_used] = np.inf
        # Find closest interstitial within squared radius
        min_idx = np.argmin(dist2)
        if dist2[min_idx] <= radius2:
            vac_to_remove.append(i)
            sia_to_remove.append(min_idx)
            sia_used[min_idx] = True

    # Remove recombined vacancies and interstitials
    vac_mask = np.ones(len(vacs), dtype=bool)
    vac_mask[vac_to_remove] = False
    sia_mask = np.ones(len(sias), dtype=bool)
    sia_mask[sia_to_remove] = False
    defects = np.concatenate([vacs[vac_mask], sias[sia_mask]])

    data_defects["atoms"] = defects
    data_defects["natoms"] = len(defects)

    return data_defects
