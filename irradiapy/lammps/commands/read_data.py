"""This module contains the ReadData class for LAMMPS read_data commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class ReadData(Command):
    """Class representing a LAMMPS read_data command.

    Reference
    ---------
    https://docs.lammps.org/read_data.html
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
        return f"read_data {self.file}{kw_vals_str}"
