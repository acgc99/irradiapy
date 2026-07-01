"""This module contains the DumpModify class for LAMMPS dump_modify commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class DumpModify(Command):
    """Class representing a LAMMPS dump_modify command.

    Reference
    ---------
    https://docs.lammps.org/dump_modify.html
    """

    id: str
    kw_vals: dict[str, str]

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
        return f"dump_modify {self.id} {kw_vals_str}"
