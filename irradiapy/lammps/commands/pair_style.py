"""This module contains the PairStyle class for LAMMPS pair_style commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class PairStyle(Command):
    """Class representing a LAMMPS pair_style command.

    Reference
    ---------
    https://docs.lammps.org/pair_style.html
    """

    style: str
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " ".join(self.args)
        return f"pair_style {self.style} {args_str}"
