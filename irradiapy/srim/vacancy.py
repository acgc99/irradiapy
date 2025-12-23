"""This module contains the `Vacancy` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Vacancy(SRIMFile):
    """Class to handle `VACANCY.txt` file."""

    def __post_init__(self):
        super().__post_init__()
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
        else:
            self.process_file = self.__process_file_fc

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
                line = next(file).replace(",", ".")
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
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
        )
        vacancies_2 = ", ".join(
            f"vacancies_{i}_{j}"
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
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
