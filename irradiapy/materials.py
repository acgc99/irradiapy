"""This module contains the `Material` class and some predefined materials."""

from dataclasses import dataclass
from math import sqrt
from typing import Optional


@dataclass
class Material:
    """Class for storing parameters of a material.

    Parameters
    ----------
    atomic_number : int
        Atomic number.
    mass_number : float
        Mass number (atomic mass units).
    a0 : float, optional
        Lattice parameter (Å). Default is None.
    cutoff_sia : float, optional
        Cutoff distance for interstitial clusters detection (Å). Default is None.
    cutoff_vac : float, optional
        Cutoff distance for vacancy clusters detection (Å). Default is None.
    dist_fp : float, optional
        Frenkel pair distance (Å). Default is None.
    density : float, optional
        Atomic density (atoms/Å³). Default is None.
    ed_min : float, optional
        Minimum displacement energy (eV). Default is None.
    ed_avr : float, optional
        Average displacement energy (eV). Default is None.
    b_arc : float, optional
        'b' parameter of the arc-dpa fit. Default is None.
    c_arc : float, optional
        'c' parameter of the arc-dpa fit. Default is None.
    """

    atomic_number: int
    mass_number: float
    a0: Optional[float] = None
    cutoff_sia: Optional[float] = None
    cutoff_vac: Optional[float] = None
    dist_fp: Optional[float] = None
    density: Optional[float] = None
    ed_min: Optional[float] = None
    ed_avr: Optional[float] = None
    b_arc: Optional[float] = None
    c_arc: Optional[float] = None


def get_material_by_name(name: str) -> Material:
    """Retrieve a predefined Material by its name.

    Parameters
    ----------
    name : str
        The name of the predefined material.

    Returns
    -------
    Material
        The requested Material instance.

    Raises
    ------
    AttributeError
        If the material is not defined.
    """
    material = PREDEFINED_MATERIALS.get(name)
    if material is None:
        err_msg = f"Material with name '{name}' is not defined."
        raise AttributeError(err_msg)
    return material


def get_material_by_atomic_number(atomic_number: int) -> Material:
    """Retrieve a predefined Material by its atomic number.

    Parameters
    ----------
    atomic_number : int
        The atomic number of the material.

    Returns
    -------
    Material
        The requested Material instance.

    Raises
    ------
    AttributeError
        If the material is not defined.
    """
    material = MATERIALS_BY_ATOMIC_NUMBER.get(atomic_number)
    if material is None:
        err_msg = f"Material with atomic number {atomic_number} is not defined."
        raise AttributeError(err_msg)
    return material


def get_atomic_number_by_name(name: str) -> int:
    """Retrieve the atomic number of a material by its name.

    Parameters
    ----------
    name : str
        The name of the material.

    Returns
    -------
    int
        The atomic number of the material.

    Raises
    ------
    AttributeError
        If the material is not defined.
    """
    material = PREDEFINED_MATERIALS.get(name)
    if material is None:
        err_msg = f"Material with name '{name}' is not defined."
        raise AttributeError(err_msg)
    return material.atomic_number


def get_mass_number_by_name(name: str) -> float:
    """Retrieve the mass number of a material by its name.

    Parameters
    ----------
    name : str
        The name of the material.

    Returns
    -------
    float
        The mass number of the material.

    Raises
    ------
    AttributeError
        If the material is not defined.
    """
    material = PREDEFINED_MATERIALS.get(name)
    if material is None:
        err_msg = f"Material with name '{name}' is not defined."
        raise AttributeError(err_msg)
    return material.mass_number


def get_mass_number_by_atomic_number(atomic_number: int) -> float:
    """Retrieve the mass number of a material by its atomic number.

    Parameters
    ----------
    atomic_number : int
        The atomic number of the material.

    Returns
    -------
    float
        The mass number of the material.

    Raises
    ------
    AttributeError
        If the material is not defined.
    """
    material = MATERIALS_BY_ATOMIC_NUMBER.get(atomic_number)
    if material is None:
        err_msg = f"Material with atomic number {atomic_number} is not defined."
        raise AttributeError(err_msg)
    return material.mass_number


#: Material: Predefined materials.
PREDEFINED_MATERIALS = {
    "H": Material(atomic_number=1, mass_number=1.008),
    "He": Material(atomic_number=2, mass_number=4.002602),
    "Li": Material(atomic_number=3, mass_number=6.94),
    "Be": Material(atomic_number=4, mass_number=9.0122),
    "B": Material(atomic_number=5, mass_number=10.81),
    "C": Material(atomic_number=6, mass_number=12.011),
    "N": Material(atomic_number=7, mass_number=14.007),
    "O": Material(atomic_number=8, mass_number=15.999),
    "F": Material(atomic_number=9, mass_number=18.998),
    "Ne": Material(atomic_number=10, mass_number=20.180),
    "Na": Material(atomic_number=11, mass_number=22.990),
    "Mg": Material(atomic_number=12, mass_number=24.305),
    "Al": Material(atomic_number=13, mass_number=26.982),
    "Si": Material(atomic_number=14, mass_number=28.085),
    "P": Material(atomic_number=15, mass_number=30.974),
    "S": Material(atomic_number=16, mass_number=32.06),
    "Cl": Material(atomic_number=17, mass_number=35.45),
    "Ar": Material(atomic_number=18, mass_number=39.948),
    "K": Material(atomic_number=19, mass_number=39.098),
    "Ca": Material(atomic_number=20, mass_number=40.078),
    "Sc": Material(atomic_number=21, mass_number=44.956),
    "Ti": Material(atomic_number=22, mass_number=47.867),
    "V": Material(atomic_number=23, mass_number=50.942),
    "Cr": Material(atomic_number=24, mass_number=51.996),
    "Mn": Material(atomic_number=25, mass_number=54.938),
    "Fe": Material(
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
    ),
    "Co": Material(atomic_number=27, mass_number=58.933),
    "Ni": Material(atomic_number=28, mass_number=58.693),
    "Cu": Material(
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
    ),
    "Zn": Material(atomic_number=30, mass_number=65.38),
    "Ga": Material(atomic_number=31, mass_number=69.723),
    "Ge": Material(atomic_number=32, mass_number=72.63),
    "As": Material(atomic_number=33, mass_number=74.921595),
    "Se": Material(atomic_number=34, mass_number=78.971),
    "Br": Material(atomic_number=35, mass_number=79.904),
    "Kr": Material(atomic_number=36, mass_number=83.798),
    "Rb": Material(atomic_number=37, mass_number=85.468),
    "Sr": Material(atomic_number=38, mass_number=87.62),
    "Y": Material(atomic_number=39, mass_number=88.90584),
    "Zr": Material(atomic_number=40, mass_number=91.224),
    "Nb": Material(atomic_number=41, mass_number=92.906),
    "Mo": Material(atomic_number=42, mass_number=95.95),
    "Tc": Material(atomic_number=43, mass_number=98),
    "Ru": Material(atomic_number=44, mass_number=101.07),
    "Rh": Material(atomic_number=45, mass_number=102.91),
    "Pd": Material(atomic_number=46, mass_number=106.42),
    "Ag": Material(atomic_number=47, mass_number=107.87),
    "Cd": Material(atomic_number=48, mass_number=112.41),
    "In": Material(atomic_number=49, mass_number=114.82),
    "Sn": Material(atomic_number=50, mass_number=118.71),
    "Sb": Material(atomic_number=51, mass_number=121.760),
    "Te": Material(atomic_number=52, mass_number=127.60),
    "I": Material(atomic_number=53, mass_number=126.90447),
    "Xe": Material(atomic_number=54, mass_number=131.29),
    "Cs": Material(atomic_number=55, mass_number=132.91),
    "Ba": Material(atomic_number=56, mass_number=137.33),
    "La": Material(atomic_number=57, mass_number=138.91),
    "Ce": Material(atomic_number=58, mass_number=140.12),
    "Pr": Material(atomic_number=59, mass_number=140.91),
    "Nd": Material(atomic_number=60, mass_number=144.242),
    "Pm": Material(atomic_number=61, mass_number=145),
    "Sm": Material(atomic_number=62, mass_number=150.36),
    "Eu": Material(atomic_number=63, mass_number=151.964),
    "Gd": Material(atomic_number=64, mass_number=157.25),
    "Tb": Material(atomic_number=65, mass_number=158.92535),
    "Dy": Material(atomic_number=66, mass_number=162.500),
    "Ho": Material(atomic_number=67, mass_number=164.93033),
    "Er": Material(atomic_number=68, mass_number=167.259),
    "Tm": Material(atomic_number=69, mass_number=168.93422),
    "Yb": Material(atomic_number=70, mass_number=173.04),
    "Lu": Material(atomic_number=71, mass_number=174.9668),
    "Hf": Material(atomic_number=72, mass_number=178.49),
    "Ta": Material(atomic_number=73, mass_number=180.94788),
    "W": Material(
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
    ),
    "Re": Material(atomic_number=75, mass_number=186.21),
    "Os": Material(atomic_number=76, mass_number=190.23),
    "Ir": Material(atomic_number=77, mass_number=192.22),
    "Pt": Material(atomic_number=78, mass_number=195.084),
    "Au": Material(atomic_number=79, mass_number=196.97),
    "Hg": Material(atomic_number=80, mass_number=200.592),
    "Tl": Material(atomic_number=81, mass_number=204.38),
    "Pb": Material(atomic_number=82, mass_number=207.2),
    "Bi": Material(atomic_number=83, mass_number=208.98040),
    "Po": Material(atomic_number=84, mass_number=209),
    "At": Material(atomic_number=85, mass_number=210),
    "Rn": Material(atomic_number=86, mass_number=222),
    "Fr": Material(atomic_number=87, mass_number=223),
    "Ra": Material(atomic_number=88, mass_number=226),
    "Ac": Material(atomic_number=89, mass_number=227),
    "Th": Material(atomic_number=90, mass_number=232.03805),
    "Pa": Material(atomic_number=91, mass_number=231.03588),
    "U": Material(atomic_number=92, mass_number=238.02891),
    "Np": Material(atomic_number=93, mass_number=237),
    "Pu": Material(atomic_number=94, mass_number=244),
    "Am": Material(atomic_number=95, mass_number=243),
    "Cm": Material(atomic_number=96, mass_number=247),
    "Bk": Material(atomic_number=97, mass_number=247),
    "Cf": Material(atomic_number=98, mass_number=251),
    "Es": Material(atomic_number=99, mass_number=252),
    "Fm": Material(atomic_number=100, mass_number=257),
    "Md": Material(atomic_number=101, mass_number=258),
    "No": Material(atomic_number=102, mass_number=259),
    "Lr": Material(atomic_number=103, mass_number=262),
    "Rf": Material(atomic_number=104, mass_number=267),
    "Db": Material(atomic_number=105, mass_number=270),
    "Sg": Material(atomic_number=106, mass_number=271),
    "Bh": Material(atomic_number=107, mass_number=270),
    "Hs": Material(atomic_number=108, mass_number=277),
    "Mt": Material(atomic_number=109, mass_number=278),
    "Ds": Material(atomic_number=110, mass_number=281),
    "Rg": Material(atomic_number=111, mass_number=282),
    "Cn": Material(atomic_number=112, mass_number=285),
    "Nh": Material(atomic_number=113, mass_number=286),
    "Fl": Material(atomic_number=114, mass_number=289),
    "Mc": Material(atomic_number=115, mass_number=290),
    "Lv": Material(atomic_number=116, mass_number=293),
    "Ts": Material(atomic_number=117, mass_number=294),
    "Og": Material(atomic_number=118, mass_number=294),
}
MATERIALS_BY_ATOMIC_NUMBER = {
    mat.atomic_number: mat for mat in PREDEFINED_MATERIALS.values()
}
