"""This module contains the MBoostIon class to mark when an ion must be boosted."""

from dataclasses import dataclass

from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class MBoostIon(Command):
    """Class to mark when an ion must be boosted."""
