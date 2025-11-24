"""This module contains the Compute class for LAMMPS compute commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command
from irradiapy.lammps.commands.uncompute import Uncompute


@dataclass(kw_only=True)
class Compute(Command):
    """Class representing a LAMMPS compute command.

    Reference
    ---------
    https://docs.lammps.org/compute.html
    """

    id: str
    group_id: str
    style: str
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.group_id = str(self.group_id)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        return f"compute {self.id} {self.group_id} {self.style}{args_str}"

    def delete(self) -> str:
        """Returns the corresponding Uncompute instance."""
        return Uncompute(compute_id=self.id)
