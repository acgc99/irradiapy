"""This module contains the `Sputter` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Sputter(SRIMFile):
    """Class to handle `SPUTTER.txt` file."""

    def process_file(self, sputter_path: Path) -> None:
        """Processes `SPUTTER.txt` file.

        Parameters
        ----------
        sputter_path : Path
            `SPUTTER.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE sputter"
                "(ion_numb INTEGER, atom_numb INTEGER, energy REAL, depth REAL,"
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        with open(sputter_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(" Numb"):
                    break
            cur = self.cursor()
            for line in file:
                line = line[1:-2]
                data = list(map(float, line[:-1].split()))
                ion_numb = data[0]
                atom_numb = data[1]
                energy = data[2]
                depth = data[3]
                y = data[4]
                z = data[5]
                cosx = data[6]
                cosy = data[7]
                cosz = data[8]
                cur.execute(
                    (
                        "INSERT INTO sputter"
                        "(ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz)"
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads sputter data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM sputter {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def get_natoms(self) -> int:
        """Returns the number of atoms in the database.

        Returns
        -------
        int
            Number of atoms.
        """
        cur = self.cursor()
        cur.execute("SELECT COUNT(1) FROM sputter")
        natoms = cur.fetchone()[0]
        cur.close()
        return natoms

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the sputter table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions = self.srim.nions
        cur = self.cursor()
        cur.execute(f"ATTACH DATABASE '{srimdb2.db_path}' AS srimdb2")
        cur.execute(
            (
                "INSERT INTO sputter"
                "(ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz)"
                "SELECT ion_numb + ?, atom_numb, energy, depth, y, z, cosx, "
                "cosy, cosz FROM srimdb2.sputter"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()
