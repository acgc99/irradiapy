"""Fe material."""

from math import sqrt

from irradiapy.materials.material import Material
from irradiapy.srimpy.target import element as srim_element

Fe = Material(
    atomic_number=26,
    mass_number=55.845,
    a0=2.87,
    cutoff_sia=2.87 * sqrt(2.0),
    cutoff_vac=2.87,
    dist_fp=4.0 * 2.87,
    density=8.5e-2,
    ed_min=20,
    ed_avr=40,
    b_arc=-0.568,
    c_arc=0.286,
    srim_element=srim_element.Element(
        symbol="Fe",
        atomic_number=26,
        atomic_mass=55.85,
        e_d=40.0,
        e_l=5.8,
        e_s=4.34,
        density=7.8658,
    ),
)
