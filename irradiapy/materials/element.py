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
    ed_min: float | None = None  # displacement energy, eV
    ed_avr: float | None = None  # average displacement energy, eV
    b_arc: float | None = None
    c_arc: float | None = None

    # SRIM values
    srim_el: float | None = None  # SRIM lattice binding energy, eV
    srim_es: float | None = None  # SRIM surface binding energy, eV
