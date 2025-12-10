"""This module contains the `RecoilsDB` class."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Generator


@dataclass
class RecoilsDB(sqlite3.Connection):
    """SQLite3 database for SPECTRA-PKA to SRIM events, recoils and ions/vacancies.

    Parameters
    ----------
    path : Path
        Path to the recoils database.
    """

    path: Path

    def __post_init__(self) -> None:
        super().__init__(self.path)
        self.create_tables()

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        self.close()
        return False

    def optimize(self) -> None:
        """Optimize the SQLite database.

        This method performs two operations to optimize the database:
        1. Executes the "PRAGMA optimize" command to analyze and optimize the database.
        2. Executes the "VACUUM" command to rebuild the database file,
        repacking it into a minimal amount of disk space.
        """
        cur = self.cursor()
        cur.execute("PRAGMA optimize")
        cur.execute("VACUUM")
        cur.close()

    def table_exists(self, table_name: str) -> bool:
        """Checks if the given table exists in the database.

        Parameters
        ----------
        table_name : str
            Table's name to check.

        Returns
        -------
        bool
            Whether the table already exists or not.
        """
        cur = self.cursor()
        cur.execute(
            (
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                f"AND name='{table_name}'"
            )
        )
        result = cur.fetchone()[0]
        cur.close()
        return bool(result)

    def process_config_events(
        self, path_spectrapka_events: Path, exclude_recoils: list[str] | None = None
    ) -> None:
        """Transform SPECTRA-PKA `config_event.pka` file into a SQLite3 table.

        Parameters
        ----------
        path_spectrapka_events : Path
            Path to the SPECTRA-PKA `config_event.pka` file.
        exclude_recoils : list[str] | None (default=None)
            List of symbols of recoils atoms to exclude from processing.
        """
        if exclude_recoils is None:
            exclude_recoils = []
        cur = self.cursor()

        cur.execute("DROP TABLE IF EXISTS events")
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS events ("
                "atom INTEGER, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, element TEXT, "
                "mass REAL, timestep INTEGER, recoil_energy REAL, time REAL, event INTEGER)"
            )
        )
        with open(path_spectrapka_events, "r", encoding="utf-8") as file:
            file.readline()
            for line in file:
                row = line.split()[:-3]  # exclude last columns (not documented)
                if row[7] in exclude_recoils:
                    continue
                cur.execute(
                    (
                        "INSERT INTO events(atom, x, y, z, vx, vy, vz, element, mass, "
                        "timestep, recoil_energy, time, event) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    row,
                )
        self.commit()
        cur.close()

    def create_tables(self) -> None:
        """Create recoils and ions_vacs tables in the database."""
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS recoils ("
                "event INTEGER, atom_numb INTEGER, recoil_energy REAL, depth REAL, "
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS ions_vacs ("
                "event INTEGER, atom_numb INTEGER, depth REAL, "
                "y REAL, z REAL)"
            )
        )
        cur.close()
        self.commit()

    def insert_recoil(
        self,
        event: int,
        atomic_number: int,
        recoil_energy: float,
        depth: float,
        y: float,
        z: float,
        cosx: float,
        cosy: float,
        cosz: float,
    ) -> None:
        """Insert a recoil into the recoils database.

        Parameters
        ----------
        event : int
            Original SPECTRA event index (1-based).
        atomic_number : int
            Atomic number of the recoil.
        recoil_energy : float
            Recoil energy in eV.
        depth : float
            Depth, x-position, in Angstrom.
        y : float
            y-position in Angstrom.
        z : float
            z-position in Angstrom.
        cosx : float
            x-cosine velocity direction.
        cosy : float
            y-cosine velocity direction.
        cosz : float
            z-cosine velocity direction.
        """
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO recoils(event, atom_numb, recoil_energy, depth, y, z, "
                "cosx, cosy, cosz) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                event,
                atomic_number,
                recoil_energy,
                depth,
                y,
                z,
                cosx,
                cosy,
                cosz,
            ),
        )
        cur.close()

    def insert_ion_vac(
        self,
        event: int,
        atom_numb: int,
        depth: float,
        y: float,
        z: float,
    ) -> None:
        """Insert an ion or vacancy into the ions_vacs database.

        Parameters
        ----------
        event : int
            Original SPECTRA event index (1-based).
        atom_numb : int
            Atomic number of the ion (0 for vacancy).
        depth : float
            Depth, x-position, in Angstrom.
        y : float
            y-position in Angstrom.
        z : float
            z-position in Angstrom.
        """
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO ions_vacs(event, atom_numb, depth, y, z) "
                "VALUES(?, ?, ?, ?, ?)"
            ),
            (event, atom_numb, depth, y, z),
        )
        cur.close()

    def read(
        self, table: str, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads table data from the database as a generator.

        Parameters
        ----------
        table : str
            Table to read from.
        what : str
            Columns to select.
        condition : str
            Condition to filter data.

        Yields
        ------
        Generator[tuple, None, None]
            Data from the database.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM {table} {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
