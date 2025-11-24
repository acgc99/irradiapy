"""This module contains the CreateAtoms class for LAMMPS create_atoms commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class CreateAtoms(Command):
    """Class representing a LAMMPS create_atoms command.

    Reference
    ---------
    https://docs.lammps.org/create_atoms.html
    """

    type: str
    style: str
    args: list[str] = field(default_factory=list)
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = str(self.type)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"create_atoms {self.type} {self.style}{args_str}{kw_vals_str}"
