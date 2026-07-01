"""This module contains the `Backscat` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Backscat(SRIMFile):
    """Class to handle `BACKSCAT.txt` file."""

    def process_file(self, backscat_path: Path) -> None:
        """Processes `BACKSCAT.txt` file.

        Parameters
        ----------
        backscat_path : Path
            `BACKSCAT.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE backscat"
                "(ion_numb INTEGER, atom_numb INTEGER, energy REAL,"
                "depth REAL, y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        with open(backscat_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(" Numb"):
                    break
            cur = self.cursor()
            for line in file:
                line = line[1:-2]
                data = line[:-1].split()
                ion_numb, atom_numb, energy = int(data[0]), int(data[1]), float(data[2])
                depth, y, z = -float(data[4]), float(data[5]), float(data[6])
                cosx, cosy, cosz = float(data[7]), float(data[8]), float(data[9])
                cur.execute(
                    (
                        "INSERT INTO backscat"
                        "(ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz)"
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz],
                )
        cur.close()
        self.srim.commit()

    def get_nions(self) -> int:
        """Returns the number of ions in the database.

        Returns
        -------
        int
            Number of ions.
        """
        cur = self.cursor()
        cur.execute("SELECT COUNT(1) FROM backscat")
        nions = cur.fetchone()[0]
        cur.close()
        return nions
