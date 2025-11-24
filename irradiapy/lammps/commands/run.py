"""This module contains the Run class for LAMMPS run commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Run(Command):
    """Class representing a LAMMPS run command.

    Reference
    ---------
    https://docs.lammps.org/run.html
    """

    n: str
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n = str(self.n)
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"run {self.n}{kw_vals_str}"
