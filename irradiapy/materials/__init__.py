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
