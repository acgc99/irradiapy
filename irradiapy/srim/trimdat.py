"""This module contains the `Trimdat` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Trimdat(SRIMFile):
    """Class to handle `TRIM.DAT` file."""

    def process_file(self, trimdat_path: Path) -> None:
        """Processes `TRIM.DAT` file.

        Parameters
        ----------
        trimdat_path : Path
            `TRIM.DAT` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE trimdat"
                "(ion_numb INTEGER, atom_numb INTEGER, energy REAL, depth REAL,"
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        with open(trimdat_path, "r", encoding="utf-8") as file:
            for _ in range(10):
                file.readline()
            cur = self.cursor()
            for line in file:
                data = line[:-1].split()
                ion_numb = int(data[0])
                atom_numb = int(data[1])
                energy = float(data[2])
                depth = float(data[3])
                y = float(data[4])
                z = float(data[5])
                cosx = float(data[6])
                cosy = float(data[7])
                cosz = float(data[8])
                cur.execute(
                    (
                        "INSERT INTO trimdat"
                        "(ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz)"
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [ion_numb, atom_numb, energy, depth, y, z, cosx, cosy, cosz],
                )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trimdat_ion ON trimdat(ion_numb)")
        cur.close()
        self.srim.commit()
