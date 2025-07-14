"""This module contains the `Material` class."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import numpy as np

from irradiapy.srimpy.target import element as srim_element


@dataclass
class Material:
    """Class for storing parameters of a material.

    Parameters
    ----------
    atomic_number : int
        Atomic number.
    mass_number : float
        Mass number (atomic mass units).
    a0 : float, optional
        Lattice parameter (Å). Default is None.
    cutoff_sia : float, optional
        Cutoff distance for interstitial clusters detection (Å). Default is None.
    cutoff_vac : float, optional
        Cutoff distance for vacancy clusters detection (Å). Default is None.
    dist_fp : float, optional
        Frenkel pair distance (Å). Default is None.
    density : float, optional
        Atomic density (atoms/Å³). Default is None.
    ed_min : float, optional
        Minimum displacement energy (eV). Default is None.
    ed_avr : float, optional
        Average displacement energy (eV). Default is None.
    b_arc : float, optional
        'b' parameter of the arc-dpa fit. Default is None.
    c_arc : float, optional
        'c' parameter of the arc-dpa fit. Default is None.
    srim_element : srim_element.Element, optional
        SRIM element object. Default is None. Check this reference for recommended values:
        https://doi.org/10.1016/j.nimb.2021.06.018
    """

    atomic_number: int
    mass_number: float
    a0: Optional[float] = None
    cutoff_sia: Optional[float] = None
    cutoff_vac: Optional[float] = None
    dist_fp: Optional[float] = None
    density: Optional[float] = None
    ed_min: Optional[float] = None
    ed_avr: Optional[float] = None
    b_arc: Optional[float] = None
    c_arc: Optional[float] = None
    srim_element: Optional["srim_element.Element"] = None

    # region Damage energy

    class TdamMode(Enum):
        """Enumeration of damage energy calculation modes."""

        SRIM = auto()
        LINDHARD = auto()

    def epka_to_tdam(
        self, mat_pka: "Material", epka: float, mode: TdamMode = TdamMode.SRIM
    ) -> float:
        """Convert PKA energy to damage energy.

        Parameters
        ----------
        mat_pka : Material
            Material of the PKA.
        epka : float
            PKA energy, in eV.
        mode : Material.TdamMode
            Mode for damage energy calculation. Can be either `Material.TdamMode.SRIM` or
            `Material.TdamMode.LINDHARD`.

        Returns
        -------
        float
            Damage energy (eV).
        """
        if mode == Material.TdamMode.SRIM:
            if self.atomic_number == mat_pka.atomic_number == 26:  # Fe
                #  SRIM Quick-Calculation, D1
                return 699e-3 * epka - 460e-9 * np.square(epka)
            if self.atomic_number == mat_pka.atomic_number == 74:  # W
                # SRIM Quick-Calculation, D1
                return 752e-3 * epka - 216e-9 * np.square(epka)
        else:  # mode == Material.TdamMode.LINDHARD:
            return self.epka_to_tdam_lindhard(mat_pka, epka)
        # else:
        #     raise ValueError("Invalid damage energy calculation mode.")

    def epka_to_tdam_lindhard(self, mat_pka: "Material", epka: float) -> float:
        """Convert PKA energy to damage energy using the Lindhard equation.

        Parameters
        ----------
        mat_pka : Material
            Material of the PKA.
        epka : float
            PKA energy, in eV.

        Returns
        -------
        float
            Damage energy, in eV.
        """
        a0 = 0.529177e-10  # m, Bohr radius
        e2 = 1.4e-9  # eV2 m s, squared unit charge for Lindhard expression
        a = (
            (9.0 * np.pi**2 / 128.0) ** (1.0 / 3.0)
            * a0
            / (mat_pka.atomic_number ** (2.0 / 3.0) + self.atomic_number ** (2.0 / 3.0))
            ** 0.5
        )
        redu = (
            (self.mass_number * epka)
            / (mat_pka.mass_number + self.mass_number)
            * a
            / (mat_pka.atomic_number * self.atomic_number * e2)
        )
        k = (
            0.1337
            * mat_pka.atomic_number ** (1.0 / 6.0)
            * (mat_pka.atomic_number / mat_pka.mass_number) ** 0.5
        )
        g = 3.4008 * redu ** (1.0 / 6.0) + 0.40244 * redu ** (3.0 / 4.0) + redu
        return epka / (1.0 + k * g)

    # region dpa calculation

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

    def tdam_to_dpa(
        self,
        tdam: Union[int, float, np.ndarray],
        mode: DpaMode = DpaMode.FERARC,
    ) -> Union[int, np.ndarray]:
        """Convert damage energy to dpa.

        Parameters
        ----------
        tdam : int, float, or numpy.ndarray
            Damage energy, in eV.
        mode : Material.DpaMode, optional
            Mode for dpa calculation. Can be one of `Material.DpaMode.NRT`, `Material.DpaMode.ARC`,
            or `Material.DpaMode.FERARC`.
        Returns
        -------
        int or numpy.ndarray
            Number of Frenkel pairs predicted by the specified dpa mode.
        """
        if mode == Material.DpaMode.FERARC:
            return self.calc_fer_arc_dpa(tdam)
        elif mode == Material.DpaMode.ARC:
            return self.calc_arc_dpa(tdam)
        else:  # if mode == Material.DpaMode.NRT:
            return self.calc_nrt_dpa(tdam)
        # else:
        #     raise ValueError("Invalid dpa calculation mode.")

    def calc_nrt_dpa(
        self,
        tdam: Union[int, float, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """Calculate the NRT-dpa for the given damage energy.

        Parameters
        ----------
        tdam : int, float, or numpy.ndarray
            Damage energy in electron volts.

        Returns
        -------
        int or numpy.ndarray
            Number of Frenkel pairs predicted by NRT-dpa.
        """
        min_threshold = self.ed_avr
        max_threshold = 2.5 * self.ed_avr

        def scaling_func(x):
            return 0.4 * x / self.ed_avr

        if isinstance(tdam, (float, int)):
            if tdam < min_threshold:
                return 0.0
            elif tdam > max_threshold:
                return scaling_func(tdam)
            else:
                return 1.0
        elif isinstance(tdam, np.ndarray) and np.issubdtype(tdam.dtype, np.number):
            return self.__apply_dpa_thresholds(
                tdam, min_threshold, max_threshold, scaling_func
            )
        else:
            raise TypeError("tdam must be a number or numpy array of numbers")

    def calc_arc_dpa(
        self,
        tdam: Union[int, float, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """Calculate the arc-dpa for the given damage energy in eV.

        Parameters
        ----------
        tdam : int, float, or numpy.ndarray
            Damage energy in electron volts.

        Returns
        -------
        int or numpy.ndarray
            Number of Frenkel pairs predicted by arc-dpa.
        """
        min_threshold = self.ed_avr
        max_threshold = 2.5 * self.ed_avr

        def scaling_func(x):
            return 0.4 * x / self.ed_avr

        def efficiency_func(x):
            return (1.0 - self.c_arc) / (
                max_threshold**self.b_arc
            ) * x**self.b_arc + self.c_arc

        if isinstance(tdam, (float, int)):
            if tdam < min_threshold:
                return 0.0
            elif tdam > max_threshold:
                eff = efficiency_func(tdam)
                return scaling_func(tdam) * eff
            else:
                return 1.0
        elif isinstance(tdam, np.ndarray):
            return self.__apply_dpa_thresholds(
                tdam, min_threshold, max_threshold, scaling_func, efficiency_func
            )
        else:
            raise TypeError("tdam must be a number or numpy array of numbers")

    def calc_fer_arc_dpa(
        self,
        tdam: Union[int, float, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """Calculate the fer-arc-dpa for the given damage energy.

        Parameters
        ----------
        tdam : int, float, or numpy.ndarray
            Damage energy, in eV.

        Returns
        -------
        int or numpy.ndarray
            Number of Frenkel pairs predicted by modified arc-dpa.
        """
        min_threshold = self.ed_min
        max_threshold = 2.5 * self.ed_avr

        def scaling_func(x):
            return 0.4 * x / self.ed_avr

        def efficiency_func(x):
            return (1.0 - self.c_arc) / (
                max_threshold**self.b_arc
            ) * x**self.b_arc + self.c_arc

        if isinstance(tdam, (float, int)):
            if tdam < min_threshold:
                return 0.0
            elif tdam > max_threshold:
                eff = efficiency_func(tdam)
                return scaling_func(tdam) * eff
            else:
                return scaling_func(tdam)
        elif isinstance(tdam, np.ndarray):
            return self.__apply_dpa_thresholds(
                tdam,
                min_threshold,
                max_threshold,
                scaling_func,
                efficiency_func,
                scaling_func,
            )
        else:
            raise TypeError("tdam must be a number or numpy array of numbers")

    @staticmethod
    def __apply_dpa_thresholds(
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
