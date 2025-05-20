"""This module contains the `Range` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srimpy.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srimpy.srimdb import SRIMDB


class Range(SRIMFile):
    """Class to handle `RANGE.txt` file."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `Range` object.

        Parameters
        ----------
        srimdb : SRIMDB
            `SRIMDB` object.
        """
        super().__init__(srimdb)
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
            self.merge = self.__merge_qc
        else:
            self.process_file = self.__process_file_fc
            self.merge = self.__merge_fc

    def __process_file_qc(self, range_path: Path) -> None:
        """Processes `RANGE.txt` file in Quick-Calculation mode.

        Parameters
        ----------
        range_path : Path
            `RANGE.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE range"
                "(depth REAL,"
                "ions REAL,"
                "recoil_distribution REAL)"
            )
        )
        with open(range_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   DEPTH"):
                    break
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                ions = data[1]
                recoil_distribution = data[2]
                cur.execute(
                    (
                        "INSERT INTO range(depth, ions, recoil_distribution)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, ions, recoil_distribution],
                )
        cur.close()
        self.srim.commit()

    def __process_file_fc(self, range_path: Path) -> None:
        """Processes `RANGE.txt` file in Full-Calculation mode.

        Parameters
        ----------
        range_path : Path
            `RANGE.txt` path.
        """
        target_atoms_1 = ", ".join(
            f"tgt_atoms_{i}_{j} REAL"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        target_atoms_2 = ", ".join(
            f"tgt_atoms_{i}_{j}"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        target_atoms_3 = ", ".join(
            ["?" for _ in range(len(target_atoms_1.split(", ")))]
        )
        cur = self.cursor()
        cur.execute(
            ("CREATE TABLE range" "(depth REAL," "ions REAL," f"{target_atoms_1})")
        )
        with open(range_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   DEPTH"):
                    break
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                ions = data[1]
                tgt_atoms = data[2:]
                cur.execute(
                    (
                        f"INSERT INTO range(depth, ions, {target_atoms_2})"
                        f"VALUES(?, ?, {target_atoms_3})"
                    ),
                    [depth, ions, *tgt_atoms],
                )
        cur.close()
        self.srim.commit()

    def __merge_qc(self, srimdb2: "SRIMDB") -> None:
        """Merges the range table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        range1 = np.array(list(self.read()))
        range2 = np.array(list(srimdb2.range.read()))

        range1[:, 1:] *= nions1
        range2[:, 1:] *= nions2
        range1[:, 1:] += range2[:, 1:]
        range1[:, 1:] /= nions

        cur = self.cursor()
        cur.execute("DELETE FROM range")
        cur.executemany(
            "INSERT INTO range(depth, ions, recoil_distribution) VALUES(?, ?, ?)",
            range1,
        )
        cur.close()
        self.srim.commit()

    def __merge_fc(self, srimdb2: "SRIMDB") -> None:
        """Merges the range table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        range1 = np.array(list(self.read()))
        range2 = np.array(list(srimdb2.range.read()))
        range1[:, 1:] *= nions1
        range2[:, 1:] *= nions2
        range1[:, 1:] += range2[:, 1:]
        range1[:, 1:] /= nions
        target_atoms_1 = ", ".join(
            f"tgt_atoms_{i}_{j} REAL"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        target_atoms_2 = ", ".join(
            f"tgt_atoms_{i}_{j}"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        target_atoms_3 = ", ".join(
            ["?" for _ in range(len(target_atoms_1.split(", ")))]
        )
        cur = self.cursor()
        cur.execute("DELETE FROM range")
        cur.executemany(
            (
                f"INSERT INTO range(depth, ions, {target_atoms_2})"
                f"VALUES(?, ?, {target_atoms_3})"
            ),
            range1,
        )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads range data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM range {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
