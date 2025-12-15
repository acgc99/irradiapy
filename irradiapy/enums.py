"""Module containing enumerations used globally."""

from enum import Enum, auto


class DamageEnergyMode(Enum):
    """Enumeration of damage energy calculation modes."""

    LINDHARD = auto()
    SRIM = auto()


class DpaMode(Enum):
    """Enumeration of dpa calculation modes.

    References
    ----------
    NRT : https://doi.org/10.1016/0029-5493(75)90035-7
    ARC : https://doi.org/10.1038/s41467-018-03415-5
    FERARC : https://doi.org/10.1103/PhysRevMaterials.5.073602
    """

    NRT = auto()
    ARC = auto()
    FERARC = auto()


class Phases(Enum):
    """Enumeration of material phases."""

    SOLID = auto()
    GAS = auto()
    LIQUID = auto()
