"""This module contains the Variable class for LAMMPS variable commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Variable(Command):
    """Class representing a LAMMPS variable command.

    Reference
    ---------
    https://docs.lammps.org/variable.html
    """

    name: str
    style: str
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        return f"variable {self.name} {self.style}{args_str}"

    def delete(self) -> str:
        """Returns the corresponding Variable instance to delete this variable."""
        return Variable(name=self.name, style="delete")
