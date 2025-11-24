"""This module contains the Uncompute class for LAMMPS uncompute commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Uncompute(Command):
    """Class representing a LAMMPS uncompute command.

    Reference
    ---------
    https://docs.lammps.org/uncompute.html
    """

    compute_id: str

    def __post_init__(self) -> None:
        self.compute_id = str(self.compute_id)

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return f"uncompute {self.compute_id}"
