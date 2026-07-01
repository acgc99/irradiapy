"""This module contains the Boundary class for LAMMPS boundary commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Boundary(Command):
    """Class representing a LAMMPS boundary command.

    Reference
    ---------
    https://docs.lammps.org/boundary.html
    """

    x: str = "p"
    y: str = "p"
    z: str = "p"

    def __post_init__(self) -> None:
        self.x = str(self.x)
        self.y = str(self.y)
        self.z = str(self.z)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"boundary {self.x} {self.y} {self.z}"
