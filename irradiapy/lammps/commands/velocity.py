"""This module contains the Velocity class for LAMMPS velocity commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Velocity(Command):
    """Class representing a LAMMPS velocity command.

    Reference
    ---------
    https://docs.lammps.org/velocity.html
    """

    group_id: str
    style: str
    args: list[str]
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_id = str(self.group_id)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " ".join(self.args)
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"velocity {self.group_id} {self.style} {args_str}{kw_vals_str}"
