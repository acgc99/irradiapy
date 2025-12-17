"""This module contains the `Element` class."""

from dataclasses import dataclass


@dataclass
class Element:
    """Class for storing parameters of an element.

    Parameters
    ----------
    atomic_number : int
        Atomic number.
    atomic_weight: float
        Atomic weight (atomic mass units).
    symbol : str
        Chemical symbol.
    """

    atomic_number: int
    atomic_weight: float
    symbol: str

    # Displacement parameters
    ed_min: None | float = None  # displacement energy, eV
    ed_avr: None | float = None  # average displacement energy, eV
    b_arc: None | float = None
    c_arc: None | float = None

    # SRIM values
    srim_el: None | float = None  # SRIM lattice binding energy, eV
    srim_es: None | float = None  # SRIM surface binding energy, eV
