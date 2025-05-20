"""This module contains the `Novac` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srimpy.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srimpy.srimdb import SRIMDB


class Novac(SRIMFile):
    """Class for processing the `NOVAC.txt` file."""

    def process_file(self, novac_path: Path) -> None:
        """Processes `NOVAC.txt` file.

        Parameters
        ----------
        novac_path : Path
            `NOVAC.txt` path.
        """
        cur = self.cursor()
        cur.execute("CREATE TABLE novac(depth REAL, number REAL)")
        with open(novac_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  DEPTH"):
                    break
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                energy_ions = data[1]
                cur.execute(
                    "INSERT INTO novac(depth, number) VALUES(?, ?)",
                    [depth, energy_ions],
                )
        cur.close()
        self.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads the novac table.

        Parameters
        ----------
        what : str
            Columns to read.
        condition : str
            Condition to filter the data.

        Yields
        ------
        tuple
            Data from the novac table.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM novac {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the novac table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        novac1 = np.array(list(self.read()))
        novac2 = np.array(list(srimdb2.novac.read()))
        novac1[:, 1:] *= nions1
        novac2[:, 1:] *= nions2
        novac1[:, 1:] = novac1[:, 1:] + novac2[:, 1:]
        novac1[:, 1:] /= nions1 + nions2
        cur = self.cursor()
        cur.execute("DELETE FROM novac")
        cur.executemany("INSERT INTO novac(depth, number) VALUES(?, ?)", novac1)
        cur.close()
        self.commit()
