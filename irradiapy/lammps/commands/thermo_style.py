"""This module contains the ThermoStyle class for LAMMPS thermo_style commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class ThermoStyle(Command):
    """Class representing a LAMMPS thermo_style command.

    Reference
    ---------
    https://docs.lammps.org/thermo_style.html
    """

    style: str = "one"
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        return f"thermo_style {self.style}{args_str}"
