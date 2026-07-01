"""This module contains the FixModify class for LAMMPS fix_modify commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class FixModify(Command):
    """Class representing a LAMMPS fix_modify command.

    Reference
    ---------
    https://docs.lammps.org/fix_modify.html
    """

    fix_id: str
    kw_vals: dict[str, str]

    def __post_init__(self) -> None:
        self.fix_id = str(self.fix_id)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
        return f"fix_modify {self.fix_id} {kw_vals_str}"
