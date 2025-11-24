"""This module contains the Log class for LAMMPS log commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Log(Command):
    """Class representing a LAMMPS log command.

    Reference
    ---------
    https://docs.lammps.org/log.html
    """

    file: str
    keyword: str = ""

    def __post_init__(self) -> None:
        self.file = str(self.file)
        self.keyword = str(self.keyword) if self.keyword else ""

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"log {self.file} {' ' + self.keyword if self.keyword else ''}"
