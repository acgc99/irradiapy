"""This module contains the `Ioniz` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


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
                line = next(file).replace(",", ".")
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
