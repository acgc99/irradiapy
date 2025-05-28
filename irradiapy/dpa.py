"""Contains definitions and utilities related to dpa."""

from enum import Enum, auto
from typing import Union

import numpy as np

from irradiapy import materials


class DpaMode(Enum):
    """Enumeration of dpa calculation modes.

    References
    ----------
    NRT : https://doi.org/10.1016/0029-5493(75)90035-7
    ARC : https://doi.org/10.1038/s41467-018-03415-5
    FERARC : https://doi.org/10.1103/PhysRevMaterials.5.073602
    """

    NRT = auto()
    ARC = auto()
    FERARC = auto()


def compute_damage_energy(
    epka: Union[float, np.ndarray],
    mat_pka: materials.Material,
    mat_target: materials.Material,
    force_lss: bool = False,
) -> Union[float, np.ndarray]:
    """Compute damage energy from the given PKA energy.

    Uses SRIM stopping powers if available and `force_lss` is False.
    Otherwise uses the Lindhard formula.

    Parameters
    ----------
    epka : float or np.ndarray
        PKA energy in electron volts.
    mat_pka : materials.Material
        Material of the primary knock-on atom.
    mat_target : materials.Material
        Target material in which the knock-on occurs.
    force_lss : bool, optional
        If True, force the use of the Lindhard formula. Default is False.

    Returns
    -------
    float or np.ndarray
        Damage energy in electron volts.
    """
    fe = materials.get_material_by_name("Fe")
    w = materials.get_material_by_name("W")
    if not force_lss:
        if mat_pka == fe and mat_target == fe:
            #  SRIM Quick-Calculation, D1
            return 699e-3 * epka - 460e-9 * np.square(epka)
        if mat_pka == w and mat_target == w:
            # SRIM Quick-Calculation, D1
            return 752e-3 * epka - 216e-9 * np.square(epka)
    # Otherwise use Lindhard formula
    a0 = 0.529177e-10  # m, Bohr radius
    e2 = 1.4e-9  # eV2 m s, squared unit charge for Lindhard expression
    a = (
        (9.0 * np.pi**2 / 128.0) ** (1.0 / 3.0)
        * a0
        / (
            mat_pka.atomic_number ** (2.0 / 3.0)
            + mat_target.atomic_number ** (2.0 / 3.0)
        )
        ** 0.5
    )
    redu = (
        (mat_target.mass_number * epka)
        / (mat_pka.mass_number + mat_target.mass_number)
        * a
        / (mat_pka.atomic_number * mat_target.atomic_number * e2)
    )
    k = (
        0.1337
        * mat_pka.atomic_number ** (1.0 / 6.0)
        * (mat_pka.atomic_number / mat_pka.mass_number) ** 0.5
    )
    g = 3.4008 * redu ** (1.0 / 6.0) + 0.40244 * redu ** (3.0 / 4.0) + redu
    return epka / (1.0 + k * g)


def calc_nrt_dpa(
    tdam: Union[int, float, np.ndarray],
    mat: materials.Material,
) -> Union[int, np.ndarray]:
    """Calculate the NRT-dpa for the given damage energy.

    Parameters
    ----------
    tdam : int, float, or numpy.ndarray
        Damage energy in electron volts.
    mat : materials.Material
        Material.

    Returns
    -------
    int or numpy.ndarray
        Number of Frenkel pairs predicted by NRT-dpa.
    """
    min_threshold = mat.ed_avr
    max_threshold = 2.5 * mat.ed_avr

    def scaling_func(x):
        return 0.4 * x / mat.ed_avr

    if isinstance(tdam, (float, int)):
        if tdam < min_threshold:
            return 0.0
        elif tdam > max_threshold:
            return scaling_func(tdam)
        else:
            return 1.0
    elif isinstance(tdam, np.ndarray) and np.issubdtype(tdam.dtype, np.number):
        return _apply_dpa_thresholds(tdam, min_threshold, max_threshold, scaling_func)
    else:
        raise TypeError("tdam must be a number or numpy array of numbers")


def calc_arc_dpa(
    tdam: Union[int, float, np.ndarray],
    mat: materials.Material,
) -> Union[int, np.ndarray]:
    """Calculate the arc-dpa for the given damage energy in eV.

    Parameters
    ----------
    tdam : int, float, or numpy.ndarray
        Damage energy in electron volts.
    mat : materials.Material
        Material in which damage is calculated.

    Returns
    -------
    int or numpy.ndarray
        Number of Frenkel pairs predicted by arc-dpa.
    """
    min_threshold = mat.ed_avr
    max_threshold = 2.5 * mat.ed_avr

    def scaling_func(x):
        return 0.4 * x / mat.ed_avr

    def efficiency_func(x):
        return (1.0 - mat.c_arc) / (max_threshold**mat.b_arc) * x**mat.b_arc + mat.c_arc

    if isinstance(tdam, (float, int)):
        if tdam < min_threshold:
            return 0.0
        elif tdam > max_threshold:
            eff = efficiency_func(tdam)
            return scaling_func(tdam) * eff
        else:
            return 1.0
    elif isinstance(tdam, np.ndarray):
        return _apply_dpa_thresholds(
            tdam, min_threshold, max_threshold, scaling_func, efficiency_func
        )
    else:
        raise TypeError("tdam must be a number or numpy array of numbers")


def calc_fer_arc_dpa(
    tdam: Union[int, float, np.ndarray],
    mat: materials.Material,
) -> Union[int, np.ndarray]:
    """Calculate the fer-arc-dpa for the given damage energy.

    Parameters
    ----------
    tdam : int, float, or numpy.ndarray
        Damage energy in electron volts.
    mat : materials.Material
        Material.

    Returns
    -------
    int or numpy.ndarray
        Number of Frenkel pairs predicted by modified arc-dpa.
    """
    min_threshold = mat.ed_min
    max_threshold = 2.5 * mat.ed_avr

    def scaling_func(x):
        return 0.4 * x / mat.ed_avr

    def efficiency_func(x):
        return (1.0 - mat.c_arc) / (max_threshold**mat.b_arc) * x**mat.b_arc + mat.c_arc

    if isinstance(tdam, (float, int)):
        if tdam < min_threshold:
            return 0.0
        elif tdam > max_threshold:
            eff = efficiency_func(tdam)
            return scaling_func(tdam) * eff
        else:
            return scaling_func(tdam)
    elif isinstance(tdam, np.ndarray):
        return _apply_dpa_thresholds(
            tdam,
            min_threshold,
            max_threshold,
            scaling_func,
            efficiency_func,
            scaling_func,
        )
    else:
        raise TypeError("tdam must be a number or numpy array of numbers")


def _apply_dpa_thresholds(
    tdam: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    scaling_func: callable,
    efficiency_func: callable = None,
    middle_func: callable = None,
):
    """Apply dpa thresholds and scaling/efficiency functions.

    Parameters
    ----------
    tdam : np.ndarray
        Damage energy array.
    min_threshold : float
        Minimum threshold for dpa.
    max_threshold : float
        Maximum threshold for dpa.
    scaling_func : callable
        Function to scale damage energy.
    efficiency_func : callable, optional
        Efficiency function for high energies. Default is None.
    middle_func : callable, optional
        Function for values between thresholds. Default is None.

    Returns
    -------
    np.ndarray
        Array of dpa values.
    """
    result = np.ones_like(tdam, dtype=np.float64)
    below_mask = tdam < min_threshold
    above_mask = tdam > max_threshold
    result[below_mask] = 0
    if middle_func is not None:
        middle_mask = (~below_mask) & (~above_mask)
        result[middle_mask] = middle_func(tdam[middle_mask])
    # else: keep as 1
    if efficiency_func:
        result[above_mask] = scaling_func(tdam[above_mask]) * efficiency_func(
            tdam[above_mask]
        )
    else:
        result[above_mask] = scaling_func(tdam[above_mask])
    return result
