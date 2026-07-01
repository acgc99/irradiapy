"""This module contains the `Range3D` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Range3D(SRIMFile):
    """Class to handle `RANGE_3D.txt` file."""

    def process_file(self, range3d_path: Path) -> None:
        """Processes `RANGE_3D.txt` file.

        Parameters
        ----------
        range3d_path : Path
            `RANGE_3D.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            "CREATE TABLE range3d(ion_numb INTEGER, depth REAL, y REAL, z REAL)"
        )
        with open(range3d_path, "r", encoding="latin1") as file:
            for line in file:
                if line.startswith("Number"):
                    break
            next(file)
            for line in file:
                data = list(map(float, line[:-1].replace(",", ".").split()))
                ion_numb = data[0]
                depth = data[1]
                y = data[2]
                z = data[3]
                cur.execute(
                    "INSERT INTO range3d(ion_numb, depth, y, z) VALUES(?, ?, ?, ?)",
                    [ion_numb, depth, y, z],
                )
        cur.close()
        self.srim.commit()
