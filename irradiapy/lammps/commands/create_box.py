"""This module contains the CreateBox class for LAMMPS create_box commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class CreateBox(Command):
    """Class representing a LAMMPS create_box command.

    Reference
    ---------
    https://docs.lammps.org/create_box.html
    """

    n: str
    region_id: str
    args: list[str] = field(default_factory=list)
    kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n = str(self.n)
        self.region_id = str(self.region_id)
        self.args = [str(arg) for arg in self.args]
        self.kw_vals = {str(key): str(value) for key, value in self.kw_vals.items()}

    def command(self) -> str:
        """Generate the LAMMPS command."""
        kw_vals_str = (
            " " + " ".join(f"{key} {value}" for key, value in self.kw_vals.items())
            if self.kw_vals
            else ""
        )
        if self.region_id == "NULL":
            args_str = " " + " ".join(self.args) if self.args else ""
            return f"create_box {self.n} {self.region_id}{args_str}{kw_vals_str}"
        else:
            return f"create_box {self.n} {self.region_id}{kw_vals_str}"
