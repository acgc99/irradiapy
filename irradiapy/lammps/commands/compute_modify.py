"""This module contains the ComputeModify class for LAMMPS compute_modify commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class ComputeModify(Command):
    """Class representing a LAMMPS compute_modify command.

    Reference
    ---------
    https://docs.lammps.org/compute_modify.html
    """

    compute_id: str
    kw_vals: dict[str, str]

    def __post_init__(self) -> None:
        self.compute_id = str(self.compute_id)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
        return f"compute_modify {self.compute_id} {kw_vals_str}"
