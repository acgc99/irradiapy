"""This module contains the `Ioniz` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Ioniz(SRIMFile):
    """Class to handle `IONIZ.txt` file."""

    def process_file(self, ioniz_path: Path) -> None:
        """Processes `IONIZ.txt` file.

        Parameters
        ----------
        ioniz_path : Path
            `IONIZ.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            "CREATE TABLE ioniz (depth REAL, ioniz_ions REAL, ioniz_recoils REAL)"
        )
        with open(ioniz_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  TARGET"):
                    break
            next(file)
            next(file)
            next(file)
            for _ in range(100):
                line = next(file).replace(",", ".")
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                ioniz_ions = data[1]
                ioniz_recoils = data[2]
                cur.execute(
                    (
                        "INSERT INTO ioniz(depth, ioniz_ions, ioniz_recoils)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, ioniz_ions, ioniz_recoils],
                )
        cur.close()
        self.srim.commit()
