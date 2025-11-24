"""This module contains the base class for LAMMPS commands."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class Command:
    """Base class for LAMMPS commands classes and for typed annotations."""

    def command(self) -> str:
        """Generate the LAMMPS command."""
        return ""
