"""This module contains the ThermoModify class for LAMMPS thermo_modify commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class ThermoModify(Command):
    """Class representing a LAMMPS thermo_modify command.

    Reference
    ---------
    https://docs.lammps.org/thermo_modify.html
    """

    kw_vals: dict[str, str]

    def __post_init__(self) -> None:
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
        return f"thermo_modify {kw_vals_str}"
