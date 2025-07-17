"""This module contains the `Element` class."""

from dataclasses import dataclass


@dataclass
class Element:
    """Class for defining an element in SRIM simulations.

    Attributes
    ----------
    symbol : str
        The symbol of the element.
    atomic_number : int
        The atomic number of the element.
    atomic_mass : float
        The atomic mass of the element in atomic mass units.
    e_d : float
        The displacement energy of the element in eV.
    e_l : float
        The lattice binding energy of the element in eV.
    e_s : float
        The surface binding energy of the element in eV.
    density : float, optional
        The density of the element in g/cm^3. Only to be used by predefined
        materials to simplify the definition of the layers in the target by the
        user.
    """

    symbol: str
    atomic_number: int
    atomic_mass: float
    e_d: float
    e_l: float
    e_s: float
    density: None | float = None
