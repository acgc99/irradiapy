"""This module contains the `Collision` class."""

from __future__ import annotations

import warnings
from math import sqrt
from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile

warnings.filterwarnings(
    "once",
    message=r"Two recoils at the same position.*",
    category=RuntimeWarning,
    module="irradiapy.srim.ofiles.collision",
)


class Collision(SRIMFile):
    """Class for processing collision data.

    Notes
    -----
    SRIM's `COLLISON.txt` does not include recoil direction cosines. We compute them at
    insertion time using the initial ion position/direction stored in the `trimdat`
    table, and the sequential collision positions for each ion. This avoids a costly
    post-processing pass that updates every collision row.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.srim.calculation == "quick":
            self.process_file = self.__process_file_qc
        else:
            self.process_file = self.__process_file_fc

    @staticmethod
    def __dir_from_positions(
        pos0: tuple[float, float, float],
        pos: tuple[float, float, float],
        prev: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute direction cosines from two positions.

        If the positions are identical, reuse `prev`.
        """
        dx = pos[0] - pos0[0]
        dy = pos[1] - pos0[1]
        dz = pos[2] - pos0[2]
        norm = sqrt(dx * dx + dy * dy + dz * dz)
        if norm == 0.0:
            warnings.warn(
                (
                    "Two recoils at the same position, assuming same direction. This is "
                    "because they are close and when saved into COLLISON.txt, "
                    "positions are rounded and they coincide."
                ),
                RuntimeWarning,
            )
            return prev
        return (dx / norm, dy / norm, dz / norm)

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
                "CREATE TABLE IF NOT EXISTS collision("
                "ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL, "
                "cosx REAL, cosy REAL, cosz REAL, "
                "se REAL, atom_hit TEXT, recoil_energy REAL, "
                "target_disp REAL)"
            )
        )

        current_ion: int | None = None
        trimdat_reader = self.srim.read(
            table="trimdat",
            what="depth, y, z, cosx, cosy, cosz",
        )
        with open(collision_path, "r", encoding="latin1") as file:
            # Skip header block
            for line in file:
                # ³=> Recoils Calculated with Kinchin-Pease Theory (Only Vacancies Calc) <=³
                if line and line[0] == "³":
                    break

            for line in file:
                if not line or line[0] != "³":
                    continue

                line = line[1:-2].replace(",", ".")
                data = line.split("³")

                ion_numb = int(data[0])
                energy = float(data[1])
                depth = float(data[2])
                y = float(data[3])
                z = float(data[4])
                se = float(data[5])
                atom_hit = data[6].strip()
                recoil_energy = float(data[7])
                target_disp = float(data[8])

                # If ion number has changed, get initial position/direction from trimdat
                if ion_numb != current_ion:
                    # If trimdat is added before collision, this should never be None
                    row = next(trimdat_reader)
                    pos0 = (float(row[0]), float(row[1]), float(row[2]))
                    cos_prev = (float(row[3]), float(row[4]), float(row[5]))
                    current_ion = ion_numb

                pos = (depth, y, z)
                cosx, cosy, cosz = self.__dir_from_positions(pos0, pos, cos_prev)
                cos_prev = (cosx, cosy, cosz)
                pos0 = pos

                cur.execute(
                    (
                        "INSERT INTO collision("
                        "ion_numb, energy, depth, y, z, cosx, cosy, cosz, "
                        "se, atom_hit, recoil_energy, target_disp)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        ion_numb,
                        energy,
                        depth,
                        y,
                        z,
                        cosx,
                        cosy,
                        cosz,
                        se,
                        atom_hit,
                        recoil_energy,
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
                "CREATE TABLE IF NOT EXISTS collision("
                "ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL, "
                "cosx REAL, cosy REAL, cosz REAL, "
                "se REAL, atom_hit TEXT, recoil_energy REAL, "
                "target_disp INTEGER, target_vac INTEGER, "
                "target_replac INTEGER, target_inter INTEGER)"
            )
        )

        current_ion: int | None = None
        trimdat_reader = self.srim.read(
            table="trimdat",
            what="depth, y, z, cosx, cosy, cosz",
        )
        with open(collision_path, "r", encoding="latin1") as file:
            for line in file:
                if not line or line[0] != "³":
                    continue

                line = line[1:-2].replace(",", ".")
                data = line.split("³")

                ion_numb = int(data[0])
                energy = float(data[1])
                depth = float(data[2])
                y = float(data[3])
                z = float(data[4])
                se = float(data[5])
                atom_hit = data[6].strip()
                recoil_energy = float(data[7])
                target_disp = int(data[8])
                target_vac = int(data[9])
                target_replac = int(data[10])
                target_inter = int(data[11])

                # If ion number has changed, get initial position/direction from trimdat
                if ion_numb != current_ion:
                    # If trimdat is added before collision, this should never be None
                    row = next(trimdat_reader)
                    pos0 = (float(row[0]), float(row[1]), float(row[2]))
                    cos_prev = (float(row[3]), float(row[4]), float(row[5]))
                    current_ion = ion_numb

                pos = (depth, y, z)
                cosx, cosy, cosz = self.__dir_from_positions(pos0, pos, cos_prev)
                cos_prev = (cosx, cosy, cosz)
                pos0 = pos

                cur.execute(
                    (
                        "INSERT INTO collision("
                        "ion_numb, energy, depth, y, z, cosx, cosy, cosz, "
                        "se, atom_hit, recoil_energy, "
                        "target_disp, target_vac, target_replac, target_inter)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        ion_numb,
                        energy,
                        depth,
                        y,
                        z,
                        cosx,
                        cosy,
                        cosz,
                        se,
                        atom_hit,
                        recoil_energy,
                        target_disp,
                        target_vac,
                        target_replac,
                        target_inter,
                    ],
                )

        cur.close()
