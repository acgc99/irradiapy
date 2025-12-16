"""This module contains the `E2Recoil` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class E2Recoil(SRIMFile):
    """Class to handle `E2RECOIL.txt` file."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `E2Recoil` object.

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

    def __process_file_qc(self, e2recoil_path: Path) -> None:
        """Processes `E2RECOIL.txt` file in Quick-Calculation mode.

        Parameters
        ----------
        e2recoil_path : Path
            `E2RECOIL.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE e2recoil"
                "(depth REAL, energy_ions REAL, energy_absorbed REAL)"
            )
        )
        with open(e2recoil_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   DEPTH"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file).replace(",", ".")
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                energy_ions = data[1]
                energy_absorbed = data[2]
                cur.execute(
                    "INSERT INTO e2recoil(depth, energy_ions, energy_absorbed) VALUES(?, ?, ?)",
                    [depth, energy_ions, energy_absorbed],
                )
        cur.close()
        self.srim.commit()

    def __process_file_fc(self, e2recoil_path: Path) -> None:
        """Processes `E2RECOIL.txt` file in Full-Calculation mode.

        Parameters
        ----------
        e2recoil_path : Path
            `E2RECOIL.txt` path.
        """
        energy_absorbed_1 = ", ".join(
            f"energy_absorbed_{i}_{j} REAL"
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
        )
        energy_absorbed_2 = ", ".join(
            f"energy_absorbed_{i}_{j}"
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
        )
        energy_absorbed_3 = ", ".join(
            ["?" for _ in range(len(energy_absorbed_1.split(", ")))]
        )
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE e2recoil"
                f"(depth REAL, energy_ions REAL, {energy_absorbed_1})"
            )
        )
        with open(e2recoil_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   DEPTH"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                energy_ions = data[1]
                energy_absorbed = data[2:]
                cur.execute(
                    (
                        f"INSERT INTO e2recoil(depth, energy_ions, {energy_absorbed_2})"
                        f"VALUES(?, ?, {energy_absorbed_3})"
                    ),
                    [depth, energy_ions, *energy_absorbed],
                )
        cur.close()
        self.srim.commit()

    def __merge_qc(self, srimdb2: "SRIMDB") -> None:
        """Merges the e2recoil table with another database for Quick-Calculation mode.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        e2recoil1 = np.array(list(self.read()))
        e2recoil2 = np.array(list(srimdb2.e2recoil.read()))
        e2recoil1[:, 1:] *= nions1
        e2recoil2[:, 1:] *= nions2
        e2recoil1[:, 1:] += e2recoil2[:, 1:]
        e2recoil1[:, 1:] /= nions
        cur = self.cursor()
        cur.execute("DELETE FROM e2recoil")
        cur.executemany(
            "INSERT INTO e2recoil(depth, energy_ions, energy_absorbed) VALUES(?, ?, ?)",
            e2recoil1,
        )
        cur.close()
        self.srim.commit()

    def __merge_fc(self, srimdb2: "SRIMDB") -> None:
        """Merges the e2recoil table with another database for Full-Calculation mode.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        e2recoil1 = np.array(list(self.read()))
        e2recoil2 = np.array(list(srimdb2.e2recoil.read()))
        e2recoil1[:, 1:] *= nions1
        e2recoil2[:, 1:] *= nions2
        e2recoil1[:, 1:] += e2recoil2[:, 1:]
        e2recoil1[:, 1:] /= nions
        energy_absorbed_1 = ", ".join(
            f"energy_absorbed_{i}_{j} REAL"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        energy_absorbed_2 = ", ".join(
            f"energy_absorbed_{i}_{j}"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        energy_absorbed_3 = ", ".join(
            ["?" for _ in range(len(energy_absorbed_1.split(", ")))]
        )
        cur = self.cursor()
        cur.execute("DELETE FROM e2recoil")
        cur.executemany(
            (
                f"INSERT INTO e2recoil(depth, energy_ions, {energy_absorbed_2})"
                f"VALUES(?, ?, {energy_absorbed_3})"
            ),
            e2recoil1,
        )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads e2recoil data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM e2recoil {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
