"""This module contains the `Subcollision` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from irradiapy.srimpy.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srimpy.srimdb import SRIMDB


class Subcollision(SRIMFile):
    """Class for processing subcollision data."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `Subcollision` object.

        Parameters
        ----------
        srimdb : SRIMDB
            `SRIMDB` object.
        """
        super().__init__(srimdb)
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
        else:
            self.process_file = self.__process_file_fc

    def __process_file_qc(self, collision_path: Path) -> None:
        """Processes `COLLISON.txt` file in Quick-Calculation mode.

        Parameters
        ----------
        collision_path : Path
            `COLLISON.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TEMPORARY TABLE IF NOT EXISTS subcollision"
                # (f"CREATE TABLE IF NOT EXISTS subcollision"
                "(ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL,"
                "se REAL, atom_hit TEXT, recoil_energy REAL,"
                "target_disp REAL)"
            )
        )
        with open(collision_path, "r", encoding="latin1") as file:
            for line in file:
                # Skip this line
                # ³=> Recoils Calculated with Kinchin-Pease Theory (Only Vacancies Calc) <=³
                if line[0] == "³":
                    break
            for line in file:
                if line[0] == "³":
                    line = line[1:-2]
                    data = line.split("³")
                    ion_numb = int(data[0])
                    energy = float(data[1])
                    depth = float(data[2])
                    y = float(data[3])
                    z = float(data[4])
                    se = float(data[5])
                    atom_hit = data[6].strip()
                    recoil_energies = float(data[7])
                    target_disp = float(data[8])
                    cur.execute(
                        (
                            "INSERT INTO subcollision"
                            "(ion_numb, energy, depth, y, z, se, atom_hit, recoil_energy,"
                            "target_disp)"
                            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        ),
                        [
                            ion_numb,
                            energy,
                            depth,
                            y,
                            z,
                            se,
                            atom_hit,
                            recoil_energies,
                            target_disp,
                        ],
                    )
        cur.close()

    def __process_file_fc(self, collision_path: Path) -> None:
        """Processes `COLLISON.txt` file in Full-Calculation mode.

        Parameters
        ----------
        collision_path : Path
            `COLLISON.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TEMPORARY TABLE IF NOT EXISTS subcollision"
                # (f"CREATE TABLE IF NOT EXISTS subcollision"
                "(ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL,"
                "se REAL, atom_hit TEXT, recoil_energy REAL,"
                "target_disp INTEGER, target_vac INTEGER,"
                "target_replac INTEGER, target_inter INTEGER)"
            )
        )
        with open(collision_path, "r", encoding="latin1") as file:
            for line in file:
                if line[0] == "³":
                    line = line[1:-2]
                    data = line.split("³")
                    ion_numb = int(data[0])
                    energy = float(data[1])
                    depth = float(data[2])
                    y = float(data[3])
                    z = float(data[4])
                    se = float(data[5])
                    atom_hit = data[6].strip()
                    recoil_energies = float(data[7])
                    target_disp = int(data[8])
                    target_vac = int(data[9])
                    target_replac = int(data[10])
                    target_inter = int(data[11])
                    cur.execute(
                        (
                            "INSERT INTO subcollision"
                            "(ion_numb, energy, depth, y, z, se, atom_hit, recoil_energy,"
                            "target_disp, target_vac, target_replac, target_inter)"
                            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        ),
                        [
                            ion_numb,
                            energy,
                            depth,
                            y,
                            z,
                            se,
                            atom_hit,
                            recoil_energies,
                            target_disp,
                            target_vac,
                            target_replac,
                            target_inter,
                        ],
                    )
        cur.close()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads data from the subcollision table.

        Parameters
        ----------
        what : str, optional (default="*")
            Columns to read.
        condition : str, optional (default="")
            Condition to filter data.

        Yields
        ------
        tuple
            Data from the subcollision table.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM subcollision {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def empty(self) -> None:
        """Empties the subcollision table."""
        cur = self.cursor()
        cur.execute("DELETE FROM subcollision")
        cur.close()
