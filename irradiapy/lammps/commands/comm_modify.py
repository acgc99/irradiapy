"""This module contains the CommModify class for LAMMPS comm_modify commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class CommModify(Command):
    """Class representing a LAMMPS comm_modify command.

    Reference
    ---------
    https://docs.lammps.org/comm_modify.html
    """

    kw_vals: dict[str, str]

    def __post_init__(self) -> None:
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
        return f"comm_modify {kw_vals_str}"
