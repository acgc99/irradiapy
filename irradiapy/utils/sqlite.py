"""sqlite3 utilities."""

import io
import sqlite3
from pathlib import Path

import numpy as np
from numpy import typing as npt


def insert_array(path_db: Path, name: str, **kargs) -> None:
    """Insert or update an array in the arrays table.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    name : str
        Name of the array to insert or update.
    kargs : dict
        Keyword arguments representing the array data.
    """
    buffer = io.BytesIO()
    np.savez(buffer, **kargs)
    blob = buffer.getvalue()
    buffer.close()
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS arrays (name TEXT UNIQUE, data BLOB)")
    cur.execute(
        "INSERT OR REPLACE INTO arrays (name, data) VALUES (?, ?)",
        (name, blob),
    )
    conn.commit()
    cur.close()
    conn.close()


def read_array(path_db: Path, name: str) -> dict[str, npt.NDArray]:
    """Read an array from the arrays table.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    name : str
        Name of the array to read.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary containing the array data.
    """
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()
    cur.execute("SELECT data FROM arrays WHERE name = ?", (name,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        raise ValueError(f"Array '{name}' not found in the database.")

    buffer = io.BytesIO(row[0])
    data = np.load(buffer, allow_pickle=False)
    data = dict(data.items())
    buffer.close()

    return data


def delete_array(path_db: Path, name: str) -> None:
    """Delete an array from the arrays table.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    name : str
        Name of the array to delete.
    """
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()
    cur.execute("DELETE FROM arrays WHERE name = ?", (name,))
    conn.commit()
    cur.close()
    conn.close()


def delete_all_arrays(path_db: Path) -> None:
    """Drop the arrays table from the database.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    """
    conn = sqlite3.connect(path_db)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS arrays")
    conn.commit()
    cur.close()
    conn.close()
