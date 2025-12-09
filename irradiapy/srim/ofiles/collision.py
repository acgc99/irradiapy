"""This module contains the `Collision` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Collision(SRIMFile):
    """Class for processing collision data."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `Collision` object.

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
                "CREATE TABLE IF NOT EXISTS collision"
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
                    line = line[1:-2].replace(",", ".")
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
                            "INSERT INTO collision"
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
                "CREATE TABLE IF NOT EXISTS collision"
                # (f"CREATE TABLE IF NOT EXISTS collision"
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
                            "INSERT INTO collision"
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

    def __merge_qc(self, srimdb2: "SRIMDB") -> None:
        """Merges the collision table with another database for Quick-Calculation mode.

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
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp, target_vac, target_replac,"
                "target_inter) SELECT ion_numb + ?, energy, depth, y, z, cosx, cosy,"
                "cosz, se, atom_hit, recoil_energy, target_disp "
                "FROM srimdb2.collision"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()

    def __merge_fc(self, srimdb2: "SRIMDB") -> None:
        """Merges the collision table with another database for Full-Calculation mode.

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
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp, target_vac, target_replac,"
                "target_inter) SELECT ion_numb + ?, energy, depth, y, z, cosx, cosy,"
                "cosz, se, atom_hit, recoil_energy, target_disp, target_vac,"
                "target_replac, target_inter FROM srimdb2.collision"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads data from the collision table.

        Parameters
        ----------
        what : str, optional (default="*")
            Columns to read.
        condition : str, optional (default="")
            Condition to filter data.

        Yields
        ------
        tuple
            Data from the collision table.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM collision {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
