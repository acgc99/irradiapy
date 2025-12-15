"""This module contains the `Component` class."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import numpy.typing as npt

from irradiapy.materials.element import Element

if TYPE_CHECKING:
    from irradiapy.recoilsdb import RecoilsDB


@dataclass
class Component:
    """Class for storing parameters of a component.

    We identified clusters using a union-find based procedure [51].
    Two SIAs (vacancies) are assigned to the same cluster if their separation
    is shorter than the midpoint between the third (second) and fourth (third)
    nearest-neighbour distances

    Parameters
    ----------
    """

    elements: tuple[Element, ...]
    stoichs: tuple[float, ...]
    name: str
    phase: "Phases"
    density: float  # g/cm3

    # Position
    x0: None | float = None  # Angstrom
    y0: None | float = None  # Angstrom
    z0: None | float = None  # Angstrom
    # Extension
    width: None | float = None  # Angstrom
    height: None | float = None  # Angstrom
    length: None | float = None  # Angstrom

    # Lattice parameters
    ax: None | float = None  # Angstrom
    ay: None | float = None  # Angstrom
    az: None | float = None  # Angstrom
    c: None | float = None  # for hcp
    structure: None | str = None  # bcc, fcc, hcp, amorphous

    # Defect parameters
    cutoff_sia: None | float = None  # Angstrom
    cutoff_vac: None | float = None  # Angstrom

    # dpa parameters
    ed_min: None | float = None  # displacement energy, eV
    ed_avr: None | float = None  # average displacement energy, eV
    b_arc: None | float = None
    c_arc: None | float = None
    calculate_energies: bool = False

    # SRIM values
    srim_el: None | float = None  # SRIM lattice binding energy, eV
    srim_es: None | float = None  # SRIM surface binding energy, eV
    srim_phase: None | int = field(init=False)  # SRIM phase (solid = 0; gas = 1)
    srim_bragg: int = 1  # Stopping corrections for special bonding in compound targets.

    nelements: int = field(init=False)

    def __post_init__(self) -> None:
        self.nelements = len(self.elements)

        if not isinstance(self.phase, Component.Phases):
            raise ValueError("phase must be an instance of Component.Phases Enum.")
        self.srim_phase = self.phase.value - 1

        if sum(self.stoichs) != 1.0:
            raise ValueError("Sum of stoichiometric coefficients must be 1.0.")

        if self.cutoff_sia is None and self.ax is not None:
            self.cutoff_sia = ((np.sqrt(2.0) + np.sqrt(11.0) / 2.0) * self.ax / 2.0,)
        if self.cutoff_vac is None and self.ax is not None:
            self.cutoff_vac = ((1.0 + np.sqrt(2.0)) * self.ax / 2.0,)

        if self.calculate_energies:
            if self.ed_avr is None:
                self.ed_avr = self.__calculate_inverse_weighted_average(
                    self.elements,
                    self.stoichs,
                    "ed_avr",
                )
            if self.ed_min is None:
                self.ed_min = self.__calculate_inverse_weighted_average(
                    self.elements,
                    self.stoichs,
                    "ed_min",
                )
            if self.srim_el is None:
                self.srim_el = self.__calculate_inverse_weighted_average(
                    self.elements,
                    self.stoichs,
                    "srim_el",
                )
            if self.srim_es is None:
                self.srim_es = self.__calculate_inverse_weighted_average(
                    self.elements,
                    self.stoichs,
                    "srim_es",
                )

    def save(self, recoilsdb: RecoilsDB) -> None:
        """Save the component to a SQLite database."""
        cur = recoilsdb.cursor()
        cur.execute(
            (
                "INSERT INTO components ("
                "name, phase, density, "
                "x0, y0, z0, width, height, length, "
                "ax, ay, az, c, structure, "
                "ed_min, ed_avr, b_arc, c_arc, "
                "srim_el, srim_es, srim_phase, srim_bragg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?)"
            ),
            (
                self.name,
                self.phase.name,
                self.density,
                self.x0,
                self.y0,
                self.z0,
                self.width,
                self.height,
                self.length,
                self.ax,
                self.ay,
                self.az,
                self.c,
                self.structure,
                self.ed_min,
                self.ed_avr,
                self.b_arc,
                self.c_arc,
                self.srim_el,
                self.srim_es,
                self.srim_phase,
                self.srim_bragg,
            ),
        )
        component_id = cur.lastrowid
        cur.close()
        for element, stoich in zip(self.elements, self.stoichs):
            cur = recoilsdb.cursor()
            cur.execute(
                (
                    "INSERT INTO elements2 (component_id, atomic_number, "
                    "mass_number, symbol, stoich, ed_min, ed_avr, b_arc, c_arc, srim_el, srim_es) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    component_id,
                    element.atomic_number,
                    element.mass_number,
                    element.symbol,
                    stoich,
                    element.ed_min,
                    element.ed_avr,
                    element.b_arc,
                    element.c_arc,
                    element.srim_el,
                    element.srim_es,
                ),
            )
            cur.close()
        recoilsdb.commit()
        return component_id

    @staticmethod
    def __calculate_inverse_weighted_average(
        elements: tuple[Element, ...],
        stoichs: tuple[float, ...],
        attribute: str,
    ) -> float:
        """Calculate the inverse weighted average of a given attribute.

        Parameters
        ----------
        elements : tuple[Element, ...]
            Tuple of elements.
        stoichs : tuple[float, ...]
            Tuple of stoichiometric coefficients.
        attribute : str
            Attribute name to calculate the average for.

        Returns
        -------
        float
            Inverse weighted average of the specified attribute.
        """
        return 1.0 / sum(
            stoich / getattr(element, attribute)
            for element, stoich in zip(elements, stoichs)
            if getattr(element, attribute) is not None
        )

    class Phases(Enum):
        """Enumeration of material phases."""

        SOLID = auto()
        GAS = auto()
        LIQUID = auto()

    # region Recoil to damage energy

    class DamageEnergyMode(Enum):
        """Enumeration of damage energy calculation modes."""

        LINDHARD = auto()
        SRIM = auto()

    def recoil_energy_to_damage_energy(
        self,
        recoil_energy: float,
        recoil: Element,
        mode: DamageEnergyMode,
    ) -> float:
        """Convert PKA energy to damage energy.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy, in eV.
        recoil : Element
            Element of the recoil atom.
        mode : DamageEnergyMode
            Damage energy calculation mode.

        Returns
        -------
        float
            Damage energy (eV).
        """
        if mode == Component.DamageEnergyMode.SRIM:
            return self.__recoil_energy_to_damage_energy_srim(
                recoil_energy, recoil, self
            )
        if mode == Component.DamageEnergyMode.LINDHARD:
            return self.__recoil_energy_to_damage_energy_lindhard_component(
                recoil_energy, recoil, self
            )
        raise ValueError("Invalid damage energy calculation mode.")

    @staticmethod
    def __recoil_energy_to_damage_energy_srim(
        recoil_energy: float,
        recoil: Element,
        component: "Component",
    ) -> float:
        """Convert recoil energy to damage energy using SRIM equations.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy, in eV.
        recoil : Element
            Element of the recoil atom.
        component : Component
            Target component.

        Returns
        -------
        float
            Damage energy, in eV.
        """
        if component.name.startswith("Iron") and recoil.symbol == "Fe":
            # SRIM Quick-Calculation, D1
            return 699e-3 * recoil_energy - 460e-9 * np.square(recoil_energy)
        if component.name.startswith("Tungsten") and recoil.symbol == "W":
            # SRIM Quick-Calculation, D1
            return 752e-3 * recoil_energy - 216e-9 * np.square(recoil_energy)
        raise ValueError(
            (
                "This combination of ion-target is not supported for SRIM damage energy "
                "calculation."
            )
        )

    @staticmethod
    def __recoil_energy_to_damage_energy_lindhard(
        recoil_energy: float,
        recoil: Element,
        element: Element,
    ) -> float:
        """Convert recoil energy to damage energy using the Lindhard equation.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy, in eV.
        recoil : Element
            Element of the recoil atom.
        element : Element
            Element of the target.

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
            / (
                recoil.atomic_number ** (2.0 / 3.0)
                + element.atomic_number ** (2.0 / 3.0)
            )
            ** 0.5
        )
        redu = (
            (element.mass_number * recoil_energy)
            / (recoil.mass_number + element.mass_number)
            * a
            / (recoil.atomic_number * element.atomic_number * e2)
        )
        k = (
            0.1337
            * recoil.atomic_number ** (1.0 / 6.0)
            * (recoil.atomic_number / recoil.mass_number) ** 0.5
        )
        g = 3.4008 * redu ** (1.0 / 6.0) + 0.40244 * redu ** (3.0 / 4.0) + redu
        return recoil_energy / (1.0 + k * g)

    @staticmethod
    def __recoil_energy_to_damage_energy_lindhard_component(
        recoil_energy: float,
        recoil: Element,
        component: "Component",
    ) -> float:
        """Convert recoil energy to damage energy using the Lindhard equation.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy, in eV.
        recoil : Element
            Element of the recoil atom.
        component : Component
            Target component.

        Returns
        -------
        float
            Damage energy, in eV.
        """
        damage_energy = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            damage_energy += (
                stoich
                * Component.__recoil_energy_to_damage_energy_lindhard(
                    recoil_energy,
                    recoil,
                    element,
                )
            )
        return damage_energy

    # endregion

    # region dpa models

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

    def damage_energy_to_dpa(
        self,
        damage_energy: float | npt.NDArray[np.float64],
        mode: DpaMode,
    ) -> int | npt.NDArray[np.float64]:
        """Convert damage energy to dpa.

        Tries to use the component parameters first; if not available, uses the element
        parameters (weighted average).

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy, in eV.
        mode : DpaMode
            Dpa calculation mode.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by the specified dpa mode.
        """
        print(self.ed_min, self.ed_avr, self.b_arc, self.c_arc)
        if mode == Component.DpaMode.FERARC:
            if (
                self.ed_min is None
                or self.ed_avr is None
                or self.b_arc is None
                or self.c_arc is None
            ):
                return self.__calc_fer_arc_dpa_elements(damage_energy, self)
            return self.__calc_fer_arc_dpa(damage_energy, self)
        if mode == Component.DpaMode.ARC:
            if self.ed_avr is None or self.b_arc is None or self.c_arc is None:
                self.__calc_arc_dpa_elements(damage_energy, self)
            return self.__calc_arc_dpa(damage_energy, self)
        if mode == Component.DpaMode.NRT:
            if self.ed_avr is None:
                return self.__calc_nrt_dpa_elements(damage_energy, self)
            return self.__calc_nrt_dpa(damage_energy, self)
        raise ValueError("Invalid dpa calculation mode.")

    @staticmethod
    def __calc_nrt_dpa(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the NRT-dpa for the given damage energy.

        Parameters
        ----------
        damage_energy : int | float | numpy.ndarray
            Damage energy in electron volts.
        target : Element | Component
            Target element or component.

        Returns
        -------
        int | numpy.ndarray
            Number of Frenkel pairs predicted by NRT-dpa.
        """
        min_threshold = target.ed_avr
        max_threshold = 2.5 * target.ed_avr

        def scaling_func(x):
            return 0.4 * x / target.ed_avr

        if isinstance(damage_energy, (float, int)):
            if damage_energy < min_threshold:
                return 0.0
            if damage_energy > max_threshold:
                return scaling_func(damage_energy)
            return 1.0
        if isinstance(damage_energy, np.ndarray) and np.issubdtype(
            damage_energy.dtype, np.number
        ):
            return Component.__apply_dpa_thresholds(
                damage_energy, min_threshold, max_threshold, scaling_func
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_nrt_dpa_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the NRT-dpa for the given damage energy using component elements.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by NRT-dpa.
        """
        nrt_dpa = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            nrt_dpa += stoich * Component.__calc_nrt_dpa(damage_energy, element)
        return nrt_dpa

    @staticmethod
    def __calc_arc_dpa(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the arc-dpa for the given damage energy.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        target : Element | Component
            Target element or component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by arc-dpa.
        """
        min_threshold = target.ed_avr
        max_threshold = 2.5 * target.ed_avr

        def scaling_func(x):
            return 0.4 * x / target.ed_avr

        def efficiency_func(x):
            return (1.0 - target.c_arc) / (
                max_threshold**target.b_arc
            ) * x**target.b_arc + target.c_arc

        if isinstance(damage_energy, (float, int)):
            if damage_energy < min_threshold:
                return 0.0
            if damage_energy > max_threshold:
                eff = efficiency_func(damage_energy)
                return scaling_func(damage_energy) * eff
            return 1.0
        if isinstance(damage_energy, np.ndarray):
            return Component.__apply_dpa_thresholds(
                damage_energy,
                min_threshold,
                max_threshold,
                scaling_func,
                efficiency_func,
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_arc_dpa_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the arc-dpa for the given damage energy using component elements.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by arc-dpa.
        """
        arc_dpa = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            arc_dpa += stoich * Component.__calc_arc_dpa(damage_energy, element)
        return arc_dpa

    @staticmethod
    def __calc_fer_arc_dpa(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fer-arc-dpa for the given damage energy.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy, in eV.
        target : Element | Component
            Target element or component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by modified arc-dpa.
        """
        min_threshold = target.ed_min
        max_threshold = 2.5 * target.ed_avr

        def scaling_func(x):
            return 0.4 * x / target.ed_avr

        def efficiency_func(x):
            return (1.0 - target.c_arc) / (
                max_threshold**target.b_arc
            ) * x**target.b_arc + target.c_arc

        if isinstance(damage_energy, (float, int)):
            if damage_energy < min_threshold:
                return 0.0
            if damage_energy > max_threshold:
                eff = efficiency_func(damage_energy)
                return scaling_func(damage_energy) * eff
            return scaling_func(damage_energy)
        if isinstance(damage_energy, np.ndarray):
            return Component.__apply_dpa_thresholds(
                damage_energy,
                min_threshold,
                max_threshold,
                scaling_func,
                efficiency_func,
                scaling_func,
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_fer_arc_dpa_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fer-arc-dpa for the given damage energy using component elements.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by modified arc-dpa.
        """
        fer_arc_dpa = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            fer_arc_dpa += stoich * Component.__calc_fer_arc_dpa(damage_energy, element)
        return fer_arc_dpa

    @staticmethod
    def __apply_dpa_thresholds(
        damage_energy: npt.NDArray[np.float64],
        min_threshold: float,
        max_threshold: float,
        scaling_func: Callable[[float], float],
        efficiency_func: Callable[[float], float] = None,
        middle_func: Callable[[float], float] = None,
    ) -> npt.NDArray[np.float64]:
        """Apply dpa thresholds and scaling/efficiency functions.

        Parameters
        ----------
        damage_energy : npt.NDArray[np.float64]
            Damage energy array.
        min_threshold : float
            Minimum threshold for dpa.
        max_threshold : float
            Maximum threshold for dpa.
        scaling_func : Callable[[float], float]
            Function to scale damage energy.
        efficiency_func : Callable[[float], float], optional (default=None)
            Efficiency function for high energies.
        middle_func : Callable[[float], float], optional (default=None)
            Function for values between thresholds.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of dpa values.
        """
        result = np.ones_like(damage_energy, dtype=np.float64)
        below_mask = damage_energy < min_threshold
        above_mask = damage_energy > max_threshold
        result[below_mask] = 0
        if middle_func is not None:
            middle_mask = (~below_mask) & (~above_mask)
            result[middle_mask] = middle_func(damage_energy[middle_mask])
        # else: keep as 1
        if efficiency_func:
            result[above_mask] = scaling_func(
                damage_energy[above_mask]
            ) * efficiency_func(damage_energy[above_mask])
        else:
            result[above_mask] = scaling_func(damage_energy[above_mask])
        return result

    # endregion
