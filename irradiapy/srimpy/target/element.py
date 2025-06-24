"""This module contains the `Element` class."""

from dataclasses import dataclass
from typing import Optional


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
    density: Optional[float] = None


# Some predefined elements
# Check this reference for recommended values:
# https://doi.org/10.1016/j.nimb.2021.06.018
#: Element: Predefined chromium.
Cr = Element("Cr", 24, 51.9961, 40.0, 7.8, 13.2, 7.19)
#: Element: Predefined iron.
Fe = Element("Fe", 26, 55.85, 40.0, 5.8, 4.34, 7.8658)
#: Element: Predefined silver.
Ag = Element("Ag", 47, 107.8682, 39.0, 4.0, 2.97, 10.49)
#: Element: Predefined tungsten.
W = Element("W", 74, 183.84, 70.0, 13.2, 8.68, 19.3)
#: Element: Predefined copper.
Cu = Element("Cu", 29, 63.546, 33.0, 4.4, 3.52, 8.92)
