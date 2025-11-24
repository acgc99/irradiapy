"""This module contains the Undump class for LAMMPS undump commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Undump(Command):
    """Class representing a LAMMPS undump command.

    Reference
    ---------
    https://docs.lammps.org/undump.html
    """

    dump_id: str

    def __post_init__(self) -> None:
        self.dump_id = str(self.dump_id)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"undump {self.dump_id}"
