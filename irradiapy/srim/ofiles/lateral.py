"""This module contains the `Ioniz` class."""

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Lateral(SRIMFile):
    """Class to handle `LATERAL.txt` file."""

    def process_file(self, lateral_path: Path) -> None:
        """Processes `LATERAL.txt` file.

        Parameters
        ----------
        lateral_path : Path
            `LATERAL.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE lateral"
                "(depth REAL, lateral_proj_range REAL,"
                "projected_straggling REAL, lateral_radial REAL,"
                "radial_straggling REAL)"
            )
        )
        with open(lateral_path, "r", encoding="latin1") as file:
            for line in file:
                if line.startswith("  TARGET"):
                    break
            next(file)
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                lateral_proj_range = data[1]
                projected_straggling = data[2]
                lateral_radial = data[3]
                radial_straggling = data[4]
                cur.execute(
                    (
                        "INSERT INTO lateral"
                        "(depth, lateral_proj_range, projected_straggling,"
                        "lateral_radial, radial_straggling)"
                        "VALUES(?, ?, ?, ?, ?)"
                    ),
                    [
                        depth,
                        lateral_proj_range,
                        projected_straggling,
                        lateral_radial,
                        radial_straggling,
                    ],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads lateral data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM lateral {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the lateral table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.

        Note
        ----
        Not sure if calculations are right, see Eqs. 9-1, 9-2 and 9-3,
        """
        nions1 = self.srim.nions
        nions2 = srimdb2.nions
        nions = nions1 + nions2
        lateral1 = np.array(list(self.read()))
        lateral2 = np.array(list(srimdb2.lateral.read()))
        # Lateral projected range
        lateral1[:, 1] *= nions1
        lateral2[:, 1] *= nions2
        lateral1[:, 1] += lateral2[:, 1]
        lateral1[:, 1] /= nions
        # Projected straggling
        lateral1[:, 2] **= 2.0
        lateral2[:, 2] **= 2.0
        lateral1[:, 2] *= nions1
        lateral2[:, 2] *= nions2
        lateral1[:, 2] += lateral2[:, 2]
        lateral1[:, 2] /= nions
        lateral1[:, 2] **= 0.5
        # Lateral radial
        lateral1[:, 3] *= nions1
        lateral2[:, 3] *= nions2
        lateral1[:, 3] += lateral2[:, 3]
        lateral1[:, 3] /= nions
        # Radial straggling
        lateral1[:, 4] **= 2.0
        lateral2[:, 4] **= 2.0
        lateral1[:, 4] *= nions1
        lateral2[:, 4] *= nions2
        lateral1[:, 4] += lateral2[:, 4]
        lateral1[:, 4] /= nions
        lateral1[:, 4] **= 0.5
        cur = self.cursor()
        cur.execute("DELETE FROM lateral")
        cur.executemany(
            "INSERT INTO lateral"
            "(depth, lateral_proj_range, projected_straggling,"
            "lateral_radial, radial_straggling)"
            "VALUES(?, ?, ?, ?, ?)",
            lateral1,
        )
        cur.close()
        self.srim.commit()
