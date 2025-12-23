"""This module contains the `Database` class."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Generator

import numpy as np
import numpy.typing as npt


@dataclass
class Database(sqlite3.Connection):
    """A SQLite database with some methods. Not intended to be used directly, instead use it to
    inherit other database classes.

    Parameters
    ----------
    path : Path
        Path to the SQLite database file.
    """

    path: Path

    def __post_init__(self) -> None:
        super().__init__(self.path)

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

    def table_has_column(self, table_name: str, column_name: str) -> bool:
        """Checks if the given table has the specified column.

        Parameters
        ----------
        table_name : str
            Table's name to check.
        column_name : str
            Column's name to check.

        Returns
        -------
        bool
            Whether the column exists in the table or not.
        """
        cur = self.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cur.fetchall()]
        cur.close()
        return column_name in columns

    def read(
        self,
        table: str,
        what: str = "*",
        conditions: str = "",
    ) -> Generator[tuple, None, None]:
        """Reads table data from the database as a generator.

        Parameters
        ----------
        table : str
            Table to read from.
        what : str
            Columns to select.
        conditions : str
            Conditions to filter data.

        Yields
        ------
        Generator[tuple, None, None]
            Data from the database.
        """
        cur = self.cursor()
        try:
            cur.execute(f"SELECT {what} FROM {table} {conditions}")
            yield from cur
        finally:
            cur.close()

    def read_chunk(
        self,
        table: str,
        what: str = "*",
        condition: str = "",
        chunksize: int = 10_000,
    ) -> Generator[tuple, None, None]:
        """Reads table data from the database as a generator in chunks. It might be faster than
        ``read`` for huge tables.

        Parameters
        ----------
        table : str
            Table to read from.
        what : str
            Columns to select.
        condition : str
            Conditions to filter data.
        chunksize : int, optional (default=10_000)
            Number of rows to read per chunk.

        Yields
        ------
        Generator[tuple, None, None]
            Data from the database.
        """
        cur = self.cursor()
        try:
            cur.execute(f"SELECT {what} FROM {table} {condition}")
            while True:
                rows = cur.fetchmany(chunksize)
                if not rows:
                    break
                yield from rows
        finally:
            cur.close()

    def read_numpy(
        self,
        table: str,
        what: str,
        conditions: str = "",
    ) -> npt.NDArray:
        """Reads table data from the database as a NumPy structured array.

        Parameters
        ----------
        table : str
            Table to read from.
        what : str
            Columns to select.
        conditions : str
            Conditions to filter data.

        Returns
        -------
        npt.NDArray
            Data from the database as a NumPy structured array.
        """
        cur = self.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        columns_info = cur.fetchall()
        cur.close()

        columns_dtype = {}
        for column in columns_info:
            column_name = column[1]
            column_type = column[2].upper()
            if "INT" in column_type:
                columns_dtype[column_name] = "i8"
            elif (
                "REAL" in column_type
                or "FLOAT" in column_type
                or "DOUBLE" in column_type
            ):
                columns_dtype[column_name] = "f8"
            else:
                # Default to string type
                columns_dtype[column_name] = "S256"

        if what.strip() == "*":
            column_names = [column[1] for column in columns_info]
        else:
            column_names = []
            for name in what.split(","):
                name = name.strip()
                parts = name.split()
                # column AS alias
                if len(parts) >= 3 and parts[-2].lower() == "as":
                    name = parts[-1]
                # column alias (implicit alias)
                elif len(parts) >= 2:
                    name = parts[-1]
                # column
                else:
                    name = parts[0]
                column_names.append(name)

        dtype_fields = [
            (column_name, columns_dtype.get(column_name, "S256"))
            for column_name in column_names
        ]
        dtype = np.dtype(dtype_fields)

        array = np.fromiter(
            self.read(table=table, what=what, conditions=conditions),
            dtype=dtype,
        )
        return array
