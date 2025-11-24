"""This module contains the ReadRestart class for LAMMPS write_data commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class ReadRestart(Command):
    """Class representing a LAMMPS read_restart command.

    Reference
    ---------
    https://docs.lammps.org/read_restart.html
    """

    file: str

    def __post_init__(self) -> None:
        self.file = str(self.file)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"read_restart {self.file}"
