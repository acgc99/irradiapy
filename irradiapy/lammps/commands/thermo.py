"""This module contains the Thermo class for LAMMPS thermo commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Thermo(Command):
    """Class representing a LAMMPS thermo command.

    Reference
    ---------
    https://docs.lammps.org/thermo.html
    """

    n: str = "0"

    def __post_init__(self) -> None:
        self.n = str(self.n)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"thermo {self.n}"
