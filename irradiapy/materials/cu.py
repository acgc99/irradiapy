"""Cu material."""

from math import sqrt

from irradiapy.materials.material import Material
from irradiapy.srimpy.target import element as srim_element

Cu = Material(
    atomic_number=29,
    mass_number=63.546,
    a0=3.61491,
    cutoff_sia=3.61491 * sqrt(2.0),
    cutoff_vac=3.61491,
    dist_fp=4.0 * 3.61491,
    density=8.46e-2,
    ed_min=25,
    ed_avr=33,
    b_arc=-0.68,
    c_arc=0.16,
    srim_element=srim_element.Element(
        symbol="Cu",
        atomic_number=29,
        atomic_mass=63.546,
        e_d=33.0,
        e_l=4.4,
        e_s=3.52,
        density=8.92,
    ),
)
