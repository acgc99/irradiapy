"""Ag material."""

from irradiapy.materials.material import Material
from irradiapy.srimpy.target import element as srim_element

Ag = Material(
    atomic_number=47,
    mass_number=107.87,
    srim_element=srim_element.Element(
        symbol="Ag",
        atomic_number=47,
        atomic_mass=107.8682,
        e_d=39.0,
        e_l=4.0,
        e_s=2.97,
        density=10.49,
    ),
)
