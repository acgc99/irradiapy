"""This module contains the `E2Recoil` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class E2Recoil(SRIMFile):
    """Class to handle `E2RECOIL.txt` file."""

    def __post_init__(self):
        super().__post_init__()
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
        else:
            self.process_file = self.__process_file_fc

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
