"""This module contains the Lattice class for LAMMPS lattice commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Lattice(Command):
    """Class representing a LAMMPS lattice command.

    Reference
    ---------
    https://docs.lammps.org/lattice.html
    """

    style: str = "none"
    scale: str = 1.0
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.style = str(self.style)
        self.scale = str(self.scale)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"lattice {self.style} {self.scale}{kw_vals_str}"
