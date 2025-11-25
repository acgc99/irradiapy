"""This module contains the `SubRange3D` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class SubRange3D(SRIMFile):
    """Class to handle intermediate `RANGE_3D.txt` files."""

    def process_file(self, range3d_path: Path) -> None:
        """Processes `RANGE_3D.txt` file.

        Parameters
        ----------
        range3d_path : Path
            `RANGE_3D.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TEMPORARY TABLE IF NOT EXISTS subrange3d"
                "(ion_numb INTEGER, depth REAL, y REAL, z REAL)"
            )
        )
        with open(range3d_path, "r", encoding="latin1") as file:
            for line in file:
                if line.startswith("Number"):
                    break
            next(file)
            for line in file:
                data = list(map(float, line[:-1].replace(",", ".").split()))
                ion_numb = data[0]
                depth = data[1]
                y = data[2]
                z = data[3]
                cur.execute(
                    "INSERT INTO subrange3d(ion_numb, depth, y, z) VALUES(?, ?, ?, ?)",
                    [ion_numb, depth, y, z],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads range3d data from the database as a generator.

        Parameters
        ----------
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
        cur.execute(f"SELECT {what} FROM subrange3d {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def empty(self) -> None:
        """Empties the subrange3d table."""
        cur = self.cursor()
        cur.execute("DELETE FROM subrange3d")
        cur.close()
