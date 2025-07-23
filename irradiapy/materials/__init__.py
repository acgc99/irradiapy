"""Subpackage for materials."""

from irradiapy.materials.ag import Ag
from irradiapy.materials.cr import Cr
from irradiapy.materials.cu import Cu
from irradiapy.materials.fe import Fe
from irradiapy.materials.material import Material
from irradiapy.materials.o import O
from irradiapy.materials.w import W

MATERIALS_BY_SYMBOL = {
    "Ag": Ag,
    "Cr": Cr,
    "Cu": Cu,
    "Fe": Fe,
    "O": O,
    "W": W,
}
MATERIALS_BY_ATOMIC_NUMBER = {
    8: O,
    24: Cr,
    26: Fe,
    29: Cu,
    47: Ag,
    74: W,
}

ATOMIC_NUMBER_BY_SYMBOL = {
    symbol: material.atomic_number for symbol, material in MATERIALS_BY_SYMBOL.items()
}
MASS_NUMBER_BY_ATOMIC_NUMBER = {
    atomic_number: material.mass_number
    for atomic_number, material in MATERIALS_BY_ATOMIC_NUMBER.items()
}
