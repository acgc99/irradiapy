"""SPECTRA-PKA subpackage core module."""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Generator

import numpy as np

from irradiapy import dtypes


def spectrapka_to_sqlite3(path_db: Path, path_spectrapka_pkas: Path) -> None:
    """Transform SPECTRA-PKA `config_event.pka` file into a SQLite3 table.

    Parameters
    ----------
    path_db : Path
        The path to the SQLite3 database file.
    path_spectrapka_pkas : Path
        The path to the `config_event.pka` file from SPECTRA-PKA.
    """
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS spectrapka_pkas")
    cur.execute(
        (
            "CREATE TABLE IF NOT EXISTS spectrapka_pkas ("
            "atom INTEGER, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, element TEXT, "
            "mass REAL, timestep INTEGER, recoil_energy REAL, time REAL, event INTEGER)"
        )
    )
    with open(path_spectrapka_pkas, "r", encoding="utf-8") as file:
        file.readline()
        for line in file:
            row = line.split()[:-3]  # exclude last columns (not documented)
            cur.execute(
                (
                    "INSERT INTO spectrapka_pkas(atom, x, y, z, vx, vy, vz, element, mass, "
                    "timestep, recoil_energy, time, event) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?*1e6, ?, ?)"
                ),
                row,
            )
    conn.commit()
    cur.close()
    conn.close()


def get_bca_pkas(
    path_db: Path,
    criterion: Callable[[dict], bool],
    negate: bool = False,
    conn: sqlite3.Connection = None,
    what: str = "*",
    condition: str = "",
) -> Generator[tuple, None, None]:
    """Yield PKAs where criterion returns False to be transferred to a BCA code.

    Parameters
    ----------
    path_db : Path
        The path to the database file.
    criterion : Callable[[dict], bool]
        A function that takes a row of the database and returns False if the PKA must be ran in a
        BCA code.
    negate : bool, optional (default=False)
        If True, the function will yield the PKAs that meet the criterion instead of those that
        do not.
    conn : sqlite3.Connection, optional (default=None)
        An existing SQLite3 connection to use. If `None`, a new connection will be created.
    what : str, optional (default="*")
        Columns to select from the `spectrapka_pkas` table.
    condition : str, optional (default="")
        SQL condition to filter the rows.

    Yields
    ------
    Generator[tuple, None, None]
        Rows from the `spectrapka_pkas` table that meet the criterion.
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(path_db)
        close = True
    cur = conn.cursor()
    cur.execute(f"SELECT {what} FROM spectrapka_pkas {condition}")
    names = dtypes.spectra_event.names
    while True:
        row = cur.fetchone()
        if row is None:
            break
        cond = criterion(**dict(zip(names, row)))
        if not negate:
            cond = not cond
        if cond:
            yield np.array(row, dtype=dtypes.spectra_event)
    cur.close()
    if close:
        conn.close()


def get_npkas(path_db: Path) -> int:
    """Get the number of PKAs in the SPECTRA-PKA database.

    Parameters
    ----------
    path_db : Path
        The path to the database file.

    Returns
    -------
    int
        The number of PKAs in the `spectrapka_pkas` table.
    """
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM spectrapka_pkas")
    nevents = cur.fetchone()[0]
    cur.close()
    conn.close()
    return nevents
