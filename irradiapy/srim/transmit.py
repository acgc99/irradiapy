"""This module contains the `Transmit` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Transmit(SRIMFile):
    """Class to handle `TRANSMIT.txt` file."""

    def process_file(self, transmit_path: Path) -> None:
        """Processes `TRANSMIT.txt` file.

        Parameters
        ----------
        transmit_path : Path
            `TRANSMIT.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE transmit"
                "(ion_numb INTEGER, atom_numb INTEGER, energy REAL, depth REAL,"
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        with open(transmit_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(" Numb"):
                    break
            cur = self.cursor()
            for line in file:
                line = line[1:-2]
                data = list(map(float, line[:-1].split()))
                ion_numb = data[0]
                atom_numb = data[1]
                energy = data[2]
                depth = data[3]
                y = data[4]
                z = data[5]
                cosx = data[6]
                cosy = data[7]
                cosz = data[8]
                cur.execute(
                    (
                        "INSERT INTO transmit"
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
        cur.execute("SELECT COUNT(1) FROM transmit")
        nions = cur.fetchone()[0]
        cur.close()
        return nions
