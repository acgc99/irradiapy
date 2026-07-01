"""This module contains the PairCoeff class for LAMMPS pair_coeff commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class PairCoeff(Command):
    """Class representing a LAMMPS pair_coeff command.

    Reference
    ---------
    https://docs.lammps.org/pair_coeff.html
    """

    i: str
    j: str
    args: list[str]

    def __post_init__(self) -> None:
        self.i = str(self.i)
        self.j = str(self.j)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " ".join(self.args)
        return f"pair_coeff {self.i} {self.j} {args_str}"
