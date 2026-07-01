"""This module contains the `SRIMFile` class."""

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


@dataclass
class SRIMFile:
    """Base class for SRIM files."""

    srim: "SRIMDB"

    def __post_init__(self) -> None:
        # Required for Collision, Vacancy, Range, and E2Recoil classes
        pass

    def cursor(self) -> sqlite3.Cursor:
        """Returns a cursor object using the current SRIM database connection.

        Returns
        -------
        sqlite3.Cursor
            A cursor object for the SRIM database connection.
        """
        return self.srim.cursor()

    def commit(self) -> None:
        """Commits the current transaction."""
        self.srim.commit()
