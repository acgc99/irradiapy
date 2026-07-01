"""This module contains the AtomStyle class for LAMMPS atom_style commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class AtomStyle(Command):
    """Class representing a LAMMPS atom_style command.

    Reference
    ---------
    https://docs.lammps.org/atom_style.html
    """

    style: str
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        return f"atom_style {self.style}{args_str}"
