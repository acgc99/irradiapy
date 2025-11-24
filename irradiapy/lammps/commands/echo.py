"""This module contains the Echo class for LAMMPS echo commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Echo(Command):
    """Class representing a LAMMPS echo command.

    Reference
    ---------
    https://docs.lammps.org/echo.html
    """

    style: str

    def __post_init__(self) -> None:
        self.style = str(self.style)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"echo {self.style}"
