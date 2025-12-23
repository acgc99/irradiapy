"""This module contains the `Phonon` class."""

from pathlib import Path

from irradiapy.srim.srimfile import SRIMFile


class Phonon(SRIMFile):
    """Class to handle `PHONON.txt` file."""

    def process_file(self, phonon_path: Path) -> None:
        """Processes `PHONON.txt` file.

        Parameters
        ----------
        phonon_path : Path
            `PHONON.txt` path.
        """
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE phonon"
                "(depth REAL, phonons_ions REAL, phonons_recoils REAL)"
            )
        )
        with open(phonon_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("  DEPTH"):
                    break
            next(file)
            next(file)
            for _ in range(100):
                line = next(file).replace(",", ".")
                data = list(map(float, line[:-1].split()))
                depth = data[0]
                phonons_ions = data[1]
                phonons_recoils = data[2]
                cur.execute(
                    (
                        "INSERT INTO phonon(depth, phonons_ions, phonons_recoils)"
                        "VALUES(?, ?, ?)"
                    ),
                    [depth, phonons_ions, phonons_recoils],
                )
        cur.close()
        self.srim.commit()
