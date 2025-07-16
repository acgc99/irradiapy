"""W material."""

from math import sqrt

from irradiapy.materials.material import Material
from irradiapy.srimpy.target import element as srim_element

W = Material(
    atomic_number=74,
    mass_number=183.84,
    a0=3.1652,
    cutoff_sia=3.1652 * sqrt(2.0),
    cutoff_vac=3.1652,
    dist_fp=4.0 * 3.1652,
    density=6.3e-2,
    ed_min=42,
    ed_avr=70,
    b_arc=-0.56,
    c_arc=0.12,
    srim_element=srim_element.Element(
        symbol="W",
        atomic_number=74,
        atomic_mass=183.84,
        e_d=70.0,
        e_l=13.2,
        e_s=8.68,
        density=19.3,
    ),
)
