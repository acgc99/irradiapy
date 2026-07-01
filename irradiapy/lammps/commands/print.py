"""This module contains the Print class for LAMMPS print commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Print(Command):
    """Class representing a LAMMPS print command.

    Reference
    ---------
    https://docs.lammps.org/print.html
    """

    string: str
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.string = str(self.string)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"print {self.string}{kw_vals_str}"
