"""Module containing enumerations used globally."""

from enum import Enum, auto


class CustomEnum(Enum):
    """Base class for custom enumerations."""

    def to_int(self) -> int:
        """Get the integer value of the enum member.

        Returns
        -------
        int
            The integer value of the enum member.
        """
        return self.value

    @classmethod
    def from_int(cls, value: int) -> "CustomEnum":
        """Get the enum member from its integer value.

        Parameters
        ----------
        value : int
            The integer value of the enum member.

        Returns
        -------
        CustomEnum
            The corresponding enum member.

        Raises
        ------
        ValueError
            If the integer value does not correspond to any enum member.
        """
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid integer value for {cls.__name__}: {value}")


class DamageEnergyMode(CustomEnum):
    """Enumeration of damage energy calculation modes."""

    LINDHARD = auto()
    SRIM = auto()


class DisplacementMode(CustomEnum):
    """Enumeration of displaced atoms calculation modes.

    References
    ----------
    NRT : https://doi.org/10.1016/0029-5493(75)90035-7
    ARC : https://doi.org/10.1038/s41467-018-03415-5
    FERARC : https://doi.org/10.1103/PhysRevMaterials.5.073602
    """

    NRT = auto()
    ARC = auto()
    FERARC = auto()


class Phases(CustomEnum):
    """Enumeration of material phases."""

    SOLID = auto()
    GAS = auto()
    LIQUID = auto()
