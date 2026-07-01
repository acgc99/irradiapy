"""This module contains the Fix class for LAMMPS fix commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command
from irradiapy.lammps.commands.unfix import Unfix


@dataclass(kw_only=True)
class Fix(Command):
    """Class representing a LAMMPS fix command.

    Reference
    ---------
    https://docs.lammps.org/fix.html
    """

    id: str
    group_id: str
    style: str
    args: list[str] = field(default_factory=list)
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.group_id = str(self.group_id)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join([str(arg) for arg in self.args]) if self.args else ""
        kw_vals_str = (
            " "
            + " ".join(f"{key} {value}" for key, value in (self.kw_vals or {}).items())
            if self.kw_vals
            else ""
        )
        return f"fix {self.id} {self.group_id} {self.style}{args_str}{kw_vals_str}"

    def delete(self) -> str:
        """Returns the corresponding Unfix instance."""
        return Unfix(fix_id=self.id)
