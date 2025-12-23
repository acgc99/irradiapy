"""This module contains the `Range` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Range(SRIMFile):
    """Class to handle `RANGE.txt` file."""

    def __post_init__(self):
        super().__post_init__()
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
        else:
            self.process_file = self.__process_file_fc

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
                line = next(file).replace(",", ".")
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
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
        )
        target_atoms_2 = ", ".join(
            f"tgt_atoms_{i}_{j}"
            for i, component in enumerate(self.srim.target)
            for j in range(len(component.elements))
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
