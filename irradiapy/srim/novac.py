"""This module contains the `Novac` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Novac(SRIMFile):
    """Class for processing the `NOVAC.txt` file."""

    def process_file(self, novac_path: Path) -> None:
        """Processes `NOVAC.txt` file.

        Parameters
        ----------
        novac_path : Path
            `NOVAC.txt` path.
        """
        cur = self.cursor()
        cur.execute("CREATE TABLE novac(depth REAL, number REAL)")
        with open(novac_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  DEPTH"):
                    break
            next(file)
            for _ in range(100):
                line = next(file)
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                energy_ions = data[1]
                cur.execute(
                    "INSERT INTO novac(depth, number) VALUES(?, ?)",
                    [depth, energy_ions],
                )
        cur.close()
        self.commit()
