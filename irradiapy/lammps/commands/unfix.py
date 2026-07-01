"""This module contains the Unfix class for LAMMPS unfix commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Unfix(Command):
    """Class representing a LAMMPS unfix command.

    Reference
    ---------
    https://docs.lammps.org/unfix.html
    """

    fix_id: str

    def __post_init__(self) -> None:
        self.fix_id = str(self.fix_id)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"unfix {self.fix_id}"
