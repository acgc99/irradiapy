"""This module contains the `Phonon` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srim.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Phonon(SRIMFile):
    """Class to handle `PHONON.txt` file."""

    def process_file(self, phonon_path: Path) -> None:
        """Processes `PHONON.txt` file.

        Parameters
        ----------
        phonon_path : Path
            `PHONON.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE phonon"
                "(depth REAL, phonons_ions REAL, phonons_recoils REAL)"
            )
        )
        with open(phonon_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  DEPTH"):
                    break
            next(file)
            next(file)
            for _ in range(100):
                line = next(file).replace(",", ".")
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                phonons_ions = data[1]
                phonons_recoils = data[2]
                cur.execute(
                    (
                        "INSERT INTO phonon(depth, phonons_ions, phonons_recoils)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, phonons_ions, phonons_recoils],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads phonon data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM phonon {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the phonon table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        phonon1 = np.array(list(self.read()))
        phonon2 = np.array(list(srimdb2.phonon.read()))
        phonon1[:, 1:] *= nions1
        phonon2[:, 1:] *= nions2
        phonon1[:, 1:] += phonon2[:, 1:]
        phonon1[:, 1:] /= nions
        cur = self.cursor()
        cur.execute("DELETE FROM phonon")
        cur.executemany(
            (
                "INSERT INTO phonon(depth, phonons_ions, phonons_recoils)"
                "VALUES(?, ?, ?)"
            ),
            phonon1,
        )
        cur.close()
        self.srim.commit()
