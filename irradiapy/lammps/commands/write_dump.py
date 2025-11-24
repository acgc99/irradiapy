"""This module contains the WriteDump class for LAMMPS write_dump commands."""

from dataclasses import dataclass, field

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class WriteDump(Command):
    """Class representing a LAMMPS write_dump command.

    Reference
    ---------
    https://docs.lammps.org/write_dump.html
    """

    group_id: str
    style: str
    file: str
    dump_args: list[str] = field(default_factory=list)
    dump_modify_kw_vals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_id = str(self.group_id)
        self.style = str(self.style)
        self.dump_args = [str(arg) for arg in self.dump_args]
        self.dump_modify_kw_vals = {
            str(key): str(value) for key, value in self.dump_modify_kw_vals.items()
        }

    def command(self) -> str:
        """Generate the LAMMPS command."""
        dump_args_str = " " + " ".join(self.dump_args) if self.dump_args else ""
        dump_modify_kw_vals_str = (
            " modify "
            + " ".join(
                f"{key} {value}" for key, value in self.dump_modify_kw_vals.items()
            )
            if self.dump_modify_kw_vals
            else ""
        )
        return f"write_dump {self.group_id} {self.style} {self.file}{dump_args_str}{dump_modify_kw_vals_str}"
