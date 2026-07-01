"""This module contains the Units class for LAMMPS units commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Units(Command):
    """Class representing a LAMMPS units command.

    Reference
    ---------
    https://docs.lammps.org/units.html
    """

    style: str = "lj"

    def __post_init__(self) -> None:
        self.style = str(self.style)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"units {self.style}"
