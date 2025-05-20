"""This module contains the `Trimdat` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from irradiapy.srimpy.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srimpy.srimdb import SRIMDB


class Trimdat(SRIMFile):
    """Class to handle `TRIM.DAT` file."""

    def process_file(self, trimdat_path: Path) -> None:
        """Processes `TRIM.DAT` file.

        Parameters
        ----------
        trimdat_path : Path
            `TRIM.DAT` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE trimdat"
                "(ion_numb INTEGER, atom_numb INTEGER, energy REAL, depth REAL,"
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        with open(trimdat_path, "r", encoding="utf-8") as file:
            for _ in range(10):
                file.readline()
            cur = self.cursor()
            for line in file:
                data = line[:-1].split()
                ion_numb = int(data[0])
                atom_numb = int(data[1])
                energy = float(data[2])
                depth = float(data[3])
                y = float(data[4])
                z = float(data[5])
                cosx = float(data[6])
                cosy = float(data[7])
                cosz = float(data[8])
                cur.execute(
                    (
                        "INSERT INTO trimdat"
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
        """Reads trimdat data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM trimdat {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the trimdat table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge with.
        """
        nions = self.srim.nions
        cur = self.cursor()
        cur.execute(f"ATTACH DATABASE '{srimdb2.db_path}' AS srimdb2")
        cur.execute(
            (
                "INSERT INTO trimdat"
                "(ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz)"
                "SELECT ion_numb + ?, atom_numb, energy, depth, y, z, cosx, cosy, cosz "
                "FROM srimdb2.trimdat"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()
