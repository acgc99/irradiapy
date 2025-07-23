"""Cr material."""

from irradiapy.materials.material import Material
from irradiapy.srim.target import element as srim_element

Cr = Material(
    atomic_number=24,
    mass_number=51.996,
    symbol="Cr",
    srim_element=srim_element.Element(
        symbol="Cr",
        atomic_number=24,
        atomic_mass=51.9961,
        e_d=40.0,
        e_l=7.8,
        e_s=13.2,
        density=7.19,
    ),
)
