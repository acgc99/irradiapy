"""This module contains the `Component` class."""

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from irradiapy.enums import DamageEnergyMode, DisplacementMode, Phases
from irradiapy.materials.element import Element


@dataclass
class Component:
    """Class for storing parameters of a component.

    Parameters
    ----------
    elements : tuple[Element, ...]
        Tuple of elements in the component.
    stoichs : tuple[float, ...]
        Tuple of stoichiometric coefficients for each element.
    name : str
        Name of the component.
    phase : Phases
        Phase of the component (solid, liquid, gas).
    density : float
        Density of the component in g/cm3.
    x0 : float, optional
        Initial x position of the component in angstroms.
    y0 : float, optional
        Initial y position of the component in angstroms.
    z0 : float, optional
        Initial z position of the component in angstroms.
    width : float, optional
        Width of the component in angstroms.
    height : float, optional
        Height of the component in angstroms.
    length : float, optional
        Length of the component in angstroms.
    ax : float, optional
        Lattice parameter in x-axis in angstroms.
    ay : float, optional
        Lattice parameter in y-axis in angstroms.
    az : float, optional
        Lattice parameter in z-axis in angstroms.
    c : float, optional
        c parameter for hcp structure in angstroms.
    structure : str, optional
        Crystal structure (bcc, fcc, hcp, amorphous).
    ed_min : float, optional
        Minimum displacement energy in eV.
    ed_avr : float, optional
        Average displacement energy in eV.
    b_arc : float, optional
        b parameter for arc-dpa model.
    c_arc : float, optional
        c parameter for arc-dpa model.
    calculate_energies : bool, optional
        Whether to calculate missing energy parameters from elements.
    srim_el : float, optional
        SRIM lattice binding energy in eV.
    srim_es : float, optional
        SRIM surface binding energy in eV.
    """

    elements: tuple[Element, ...]
    stoichs: tuple[float, ...]
    name: str
    phase: "Phases"
    density: float  # g/cm3
    atomic_density: float = field(init=False)  # atoms/angstrom^3

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

    # Displacement parameters
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
        self.atomic_density = self.__calculate_atomic_density()

        if not isinstance(self.phase, Phases):
            raise ValueError("phase must be an instance of Phases Enum.")
        self.srim_phase = self.phase.to_int() - 1

        if sum(self.stoichs) != 1.0:
            raise ValueError("Sum of stoichiometric coefficients must be 1.0.")

        if self.structure in ["bcc", "fcc"] and self.ax is not None:
            if self.ay is None:
                self.ay = self.ax
            if self.az is None:
                self.az = self.ax

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

    def __calculate_atomic_density(self) -> float:
        """Calculate the density in units of atoms / angstrom^3 from density in g / cm3."""
        g_cm3_to_amu_a3 = 0.602214129
        atomic_mass = sum(
            element.atomic_weight * stoich
            for element, stoich in zip(self.elements, self.stoichs)
        )  # amu / atom
        atomic_density = self.density * g_cm3_to_amu_a3 / atomic_mass
        return atomic_density

    # region Recoil to damage energy

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
        if mode == DamageEnergyMode.SRIM:
            return self.__recoil_energy_to_damage_energy_srim(
                recoil_energy, recoil, self
            )
        if mode == DamageEnergyMode.LINDHARD:
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
            (element.atomic_weight * recoil_energy)
            / (recoil.atomic_weight + element.atomic_weight)
            * a
            / (recoil.atomic_number * element.atomic_number * e2)
        )
        k = (
            0.1337
            * recoil.atomic_number ** (1.0 / 6.0)
            * (recoil.atomic_number / recoil.atomic_weight) ** 0.5
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

    # region Displacement models

    def damage_energy_to_displacements(
        self,
        damage_energy: float | npt.NDArray[np.float64],
        mode: DisplacementMode,
    ) -> int | npt.NDArray[np.float64]:
        """Convert damage energy to displaced atoms.

        Tries to use the component parameters first; if not available, uses the element
        parameters (weighted average).

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy, in eV.
        mode : DisplacementMode
            Displaced atoms calculation mode.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by the specified displacement mode.
        """
        if mode == DisplacementMode.FERARC:
            if (
                self.ed_min is None
                or self.ed_avr is None
                or self.b_arc is None
                or self.c_arc is None
            ):
                return self.__calc_fer_arc_displacements_elements(damage_energy, self)
            return self.__calc_fer_arc_displacements(damage_energy, self)
        if mode == DisplacementMode.ARC:
            if self.ed_avr is None or self.b_arc is None or self.c_arc is None:
                return self.__calc_arc_displacements_elements(damage_energy, self)
            return self.__calc_arc_displacements(damage_energy, self)
        if mode == DisplacementMode.NRT:
            if self.ed_avr is None:
                return self.__calc_nrt_displacements_elements(damage_energy, self)
            return self.__calc_nrt_displacements(damage_energy, self)
        raise ValueError("Invalid displacement calculation mode.")

    @staticmethod
    def __calc_nrt_displacements(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the NRT-displacements for the given damage energy.

        Parameters
        ----------
        damage_energy : int | float | numpy.ndarray
            Damage energy in electron volts.
        target : Element | Component
            Target element or component.

        Returns
        -------
        int | numpy.ndarray
            Number of Frenkel pairs predicted by NRT-displacements.
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
            return Component.__apply_displacement_thresholds(
                damage_energy, min_threshold, max_threshold, scaling_func
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_nrt_displacements_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the NRT-displacements for the given damage energy using component elements.
        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by NRT-displacements.
        """
        nrt_displacements = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            nrt_displacements += stoich * Component.__calc_nrt_displacements(
                damage_energy, element
            )
        return nrt_displacements

    @staticmethod
    def __calc_arc_displacements(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the arc-displacements for the given damage energy.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        target : Element | Component
            Target element or component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by arc-displacements.
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
            return Component.__apply_displacement_thresholds(
                damage_energy,
                min_threshold,
                max_threshold,
                scaling_func,
                efficiency_func,
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_arc_displacements_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the arc-displacements for the given damage energy using component elements.

        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by arc-displacements.
        """
        arc_displacements = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            arc_displacements += stoich * Component.__calc_arc_displacements(
                damage_energy, element
            )
        return arc_displacements

    @staticmethod
    def __calc_fer_arc_displacements(
        damage_energy: float | npt.NDArray[np.float64],
        target: Union[Element, "Component"],
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fer-arc-displacements for the given damage energy.
        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy, in eV.
        target : Element | Component
            Target element or component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by modified arc-displacements.
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
            return Component.__apply_displacement_thresholds(
                damage_energy,
                min_threshold,
                max_threshold,
                scaling_func,
                efficiency_func,
                scaling_func,
            )
        raise TypeError("damage_energy must be a number or numpy array of numbers")

    @staticmethod
    def __calc_fer_arc_displacements_elements(
        damage_energy: float | npt.NDArray[np.float64],
        component: "Component",
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fer-arc-displacements for the given damage energy using component elements.
        Parameters
        ----------
        damage_energy : float | npt.NDArray[np.float64]
            Damage energy in electron volts.
        component : Component
            Target component.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Number of Frenkel pairs predicted by fer-arc-displacements.
        """
        fer_arc_displacements = 0.0
        for element, stoich in zip(component.elements, component.stoichs):
            fer_arc_displacements += stoich * Component.__calc_fer_arc_displacements(
                damage_energy, element
            )
        return fer_arc_displacements

    @staticmethod
    def __apply_displacement_thresholds(
        damage_energy: npt.NDArray[np.float64],
        min_threshold: float,
        max_threshold: float,
        scaling_func: Callable[[float], float],
        efficiency_func: Callable[[float], float] = None,
        middle_func: Callable[[float], float] = None,
    ) -> npt.NDArray[np.float64]:
        """Apply displacement thresholds and scaling/efficiency functions.

        Parameters
        ----------
        damage_energy : npt.NDArray[np.float64]
            Damage energy array.
        min_threshold : float
            Minimum threshold for displacements.
        max_threshold : float
            Maximum threshold for displacements.
        scaling_func : Callable[[float], float]
            Function to scale damage energy.
        efficiency_func : Callable[[float], float], optional (default=None)
            Efficiency function for high energies.
        middle_func : Callable[[float], float], optional (default=None)
            Function for values between thresholds.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of displacement values.
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
