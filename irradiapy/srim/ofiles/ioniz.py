"""This module contains the `Ioniz` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Ioniz(SRIMFile):
    """Class to handle `IONIZ.txt` file."""

    def process_file(self, ioniz_path: Path) -> None:
        """Processes `IONIZ.txt` file.

        Parameters
        ----------
        ioniz_path : Path
            `IONIZ.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            "CREATE TABLE ioniz (depth REAL, ioniz_ions REAL, ioniz_recoils REAL)"
        )
        with open(ioniz_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  TARGET"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                ioniz_ions = data[1]
                ioniz_recoils = data[2]
                cur.execute(
                    (
                        "INSERT INTO ioniz(depth, ioniz_ions, ioniz_recoils)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, ioniz_ions, ioniz_recoils],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads ioniz data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM ioniz {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the ioniz table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        ioniz1 = np.array(list(self.read()))
        ioniz2 = np.array(list(srimdb2.ioniz.read()))
        ioniz1[:, 1:] *= nions1
        ioniz2[:, 1:] *= nions2
        ioniz1[:, 1:] += ioniz2[:, 1:]
        ioniz1[:, 1:] /= nions
        cur = self.cursor()
        cur.execute("DELETE FROM ioniz")
        cur.executemany(
            "INSERT INTO ioniz(depth, ioniz_ions, ioniz_recoils) VALUES(?, ?, ?)",
            ioniz1,
        )
        cur.close()
        self.srim.commit()
