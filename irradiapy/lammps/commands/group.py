"""This module contains the Group class for LAMMPS group commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Group(Command):
    """Class representing a LAMMPS group command.

    Reference
    ---------
    https://docs.lammps.org/group.html
    """

    id: str
    style: str
    args: list[str] = field(default_factory=list)
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.style = str(self.style)
        self.args = [str(arg) for arg in self.args]
        self.kw_vals = {str(k): str(v) for k, v in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " " + " ".join(self.args) if self.args else ""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        return f"group {self.id} {self.style}{args_str}{kw_vals_str}"

    def delete(self) -> str:
        """Returns the corresponding Group instance to delete this group."""
        return Group(id=self.id, style="delete")
