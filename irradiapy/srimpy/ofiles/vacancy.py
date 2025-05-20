"""This module contains the `Vacancy` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srimpy.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srimpy.srimdb import SRIMDB


class Vacancy(SRIMFile):
    """Class to handle `VACANCY.txt` file."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `Vacancy` object.

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

    def __process_file_qc(self, vacancy_path: Path) -> None:
        """Processes `VACANCY.txt` file in Quick-Calculation mode.

        Parameters
        ----------
        vacancy_path : Path
            `VACANCY.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE vacancy"
                "(depth REAL, vacancies_ions REAL, vacancies_recoils REAL)"
            )
        )
        with open(vacancy_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   TARGET"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                vacancies_ions = data[1]
                vacancies_recoils = data[2]
                cur.execute(
                    (
                        "INSERT INTO vacancy"
                        "(depth, vacancies_ions, vacancies_recoils)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, vacancies_ions, vacancies_recoils],
                )
        cur.close()
        self.srim.commit()

    def __process_file_fc(self, vacancy_path: Path) -> None:
        """Processes `VACANCY.txt` file in Full-Calculation mode.

        Parameters
        ----------
        vacancy_path : Path
            `VACANCY.txt` path.
        """
        vacancies_1 = ", ".join(
            f"vacancies_{i}_{j} REAL"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        vacancies_2 = ", ".join(
            f"vacancies_{i}_{j}"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        vacancies_3 = ", ".join(["?" for _ in range(len(vacancies_1.split(", ")))])
        cur = self.cursor()
        cur.execute(
            ("CREATE TABLE vacancy" f"(depth REAL, knock_ons REAL, {vacancies_1})")
        )
        with open(vacancy_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("   TARGET"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                knock_ons = data[1]
                vacancies = data[2:]
                cur.execute(
                    (
                        "INSERT INTO vacancy"
                        f"(depth, knock_ons, {vacancies_2})"
                        f"VALUES(?, ?, {vacancies_3})"
                    ),
                    [depth, knock_ons, *vacancies],
                )
        cur.close()
        self.srim.commit()

    def __merge_qc(self, srimdb2: "SRIMDB") -> None:
        """Merges the e2recoil table with another database for Quick-Calculation mode.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge with.
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        vacancy1 = np.array(list(self.read()))
        vacancy2 = np.array(list(srimdb2.vacancy.read()))
        vacancy1[:, 1:] *= nions1
        vacancy2[:, 1:] *= nions2
        vacancy1[:, 1:] += vacancy2[:, 1:]
        vacancy1[:, 1:] /= nions
        cur = self.cursor()
        cur.execute("DELETE FROM vacancy")
        cur.executemany(
            (
                "INSERT INTO vacancy(depth, vacancies_ions, vacancies_recoils)"
                "VALUES(?, ?, ?)"
            ),
            vacancy1,
        )
        cur.close()
        self.srim.commit()

    def __merge_fc(self, srimdb2: "SRIMDB") -> None:
        """Merges the vacancy table with another database for Full-Calculation mode.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge with
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        vacancy1 = np.array(list(self.read()))
        vacancy2 = np.array(list(srimdb2.vacancy.read()))
        vacancy1[:, 1:] *= nions1
        vacancy2[:, 1:] *= nions2
        vacancy1[:, 1:] += vacancy2[:, 1:]
        vacancy1[:, 1:] /= nions
        vacancies_1 = ", ".join(
            f"vacancies_{i}_{j} REAL"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        vacancies_2 = ", ".join(
            f"vacancies_{i}_{j}"
            for i, layer in enumerate(self.srim.target.layers)
            for j in range(len(layer.elements))
        )
        vacancies_3 = ", ".join(["?" for _ in range(len(vacancies_1.split(", ")))])
        cur = self.cursor()
        cur.execute("DELETE FROM vacancy")
        cur.executemany(
            (
                f"INSERT INTO vacancy(depth, knock_ons, {vacancies_2})"
                f"VALUES(?, ?, {vacancies_3})"
            ),
            vacancy1,
        )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads vacancy data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM vacancy {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
