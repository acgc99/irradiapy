"""This module contains the WriteData class for LAMMPS write_data commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class WriteData(Command):
    """Class representing a LAMMPS write_data command.

    Reference
    ---------
    https://docs.lammps.org/write_data.html
    """

    file: str
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.file = str(self.file)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"write_data {self.file}{kw_vals_str}"
