"""This module contains the Timer class for LAMMPS timer commands."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class Timer(Command):
    """Class representing a LAMMPS timer command.

    Reference
    ---------
    https://docs.lammps.org/timer.html
    """

    args: list[str]

    def __post_init__(self) -> None:
        self.args = [str(arg) for arg in self.args]

    def command(self) -> str:
        """Generate the LAMMPS command."""
        args_str = " ".join([str(arg) for arg in self.args])
        return f"timer {args_str}"
