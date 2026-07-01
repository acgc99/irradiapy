"""This module contains the Dump class for LAMMPS dump commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command
from irradiapy.lammps.commands.undump import Undump


@dataclass(kw_only=True)
class Dump(Command):
    """Class representing a LAMMPS dump command.

    Reference
    ---------
    https://docs.lammps.org/dump.html
    """

    id: str
    group_id: str
    style: str
    n: str
    file: str
    attributes: list[str]

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.group_id = str(self.group_id)
        self.style = str(self.style)
        self.n = str(self.n)
        self.file = str(self.file)
        self.attributes = [str(attr) for attr in self.attributes]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        attributes_str = " ".join(self.attributes)
        return f"dump {self.id} {self.group_id} {self.style} {self.n} {self.file} {attributes_str}"

    def delete(self) -> str:
        """Returns the corresponding Undump instance."""
        return Undump(dump_id=self.id)
