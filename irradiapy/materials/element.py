"""This module contains the `Element` class."""

from dataclasses import dataclass


@dataclass
class Element:
    """Class for storing parameters of an element.

    Parameters
    ----------
    atomic_number : int
        Atomic number.
    mass_number : float
        Mass number (atomic mass units).
    symbol : str
        Chemical symbol.
    """

    atomic_number: int
    mass_number: float
    symbol: str

    # dpa parameters
    ed_min: None | float = None  # displacement energy, eV
    ed_avr: None | float = None  # average displacement energy, eV
    b_arc: None | float = None
    c_arc: None | float = None

    # SRIM values
    el_srim: None | float = None  # SRIM lattice binding energy, eV
    es_srim: None | float = None  # SRIM surface binding energy, eV


H = Element(
    atomic_number=1,
    mass_number=1.00784,
    symbol="H",
)
He = Element(
    atomic_number=2,
    mass_number=4.002602,
    symbol="He",
)
O = Element(
    atomic_number=8,
    mass_number=15.999,
    symbol="O",
)
Cr = Element(
    atomic_number=24,
    mass_number=51.9961,
    symbol="Cr",
    ed_avr=40.0,
    el_srim=7.8,
    es_srim=13.2,
)
Mn = Element(
    atomic_number=25,
    mass_number=54.938044,
    symbol="Mn",
)
Fe = Element(
    atomic_number=26,
    mass_number=55.845,
    symbol="Fe",
    ed_min=20.0,
    ed_avr=40.0,
    b_arc=-0.568,
    c_arc=0.286,
)
Cu = Element(
    atomic_number=29,
    mass_number=63.546,
    symbol="Cu",
    ed_min=25.0,
    ed_avr=33.0,
    b_arc=-0.68,
    c_arc=0.16,
)
Ag = Element(
    atomic_number=47,
    mass_number=107.87,
    symbol="Ag",
    ed_avr=39.0,
    el_srim=4.0,
    es_srim=2.97,
)
Hf = Element(
    atomic_number=72,
    mass_number=178.49,
    symbol="Hf",
)
Ta = Element(
    atomic_number=73,
    mass_number=180.94788,
    symbol="Ta",
)
W = Element(
    atomic_number=74,
    mass_number=183.84,
    symbol="W",
    ed_min=42.0,
    ed_avr=70.0,
    el_srim=13.2,
    es_srim=8.68,
    b_arc=-0.56,
    c_arc=0.12,
)

ELEMENTS = [H, He, O, Cr, Mn, Fe, Cu, Ag, Hf, Ta, W]
ELEMENT_BY_SYMBOL = {element.symbol: element for element in ELEMENTS}
ELEMENT_BY_ATOMIC_NUMBER = {
    element.atomic_number: element for element in ELEMENT_BY_SYMBOL.values()
}
MASS_NUMBER_BY_SYMBOL = {
    symbol: element.mass_number for symbol, element in ELEMENT_BY_SYMBOL.items()
}
MASS_NUMBER_BY_ATOMIC_NUMBER = {
    atomic_number: element.mass_number
    for atomic_number, element in ELEMENT_BY_ATOMIC_NUMBER.items()
}
ATOMIC_NUMBER_BY_SYMBOL = {
    symbol: element.atomic_number for symbol, element in ELEMENT_BY_SYMBOL.items()
}
