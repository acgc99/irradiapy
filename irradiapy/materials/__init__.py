"""Subpackage for materials."""

from irradiapy.materials.ag import Ag
from irradiapy.materials.cr import Cr
from irradiapy.materials.cu import Cu
from irradiapy.materials.fe import Fe
from irradiapy.materials.h import H
from irradiapy.materials.he import He
from irradiapy.materials.hf import Hf
from irradiapy.materials.material import Material
from irradiapy.materials.mn import Mn
from irradiapy.materials.o import O
from irradiapy.materials.ta import Ta
from irradiapy.materials.w import W

MATERIALS_BY_SYMBOL = {
    "H": H,
    "He": He,
    "O": O,
    "Cr": Cr,
    "Mn": Mn,
    "Fe": Fe,
    "Cu": Cu,
    "Ag": Ag,
    "Hf": Hf,
    "Ta": Ta,
    "W": W,
}
MATERIALS_BY_ATOMIC_NUMBER = {
    1: H,
    2: He,
    8: O,
    24: Cr,
    25: Mn,
    26: Fe,
    29: Cu,
    47: Ag,
    72: Hf,
    73: Ta,
    74: W,
}

ATOMIC_NUMBER_BY_SYMBOL = {
    symbol: material.atomic_number for symbol, material in MATERIALS_BY_SYMBOL.items()
}
MASS_NUMBER_BY_ATOMIC_NUMBER = {
    atomic_number: material.mass_number
    for atomic_number, material in MATERIALS_BY_ATOMIC_NUMBER.items()
}
