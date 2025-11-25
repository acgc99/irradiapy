"""This module contains the `IonsFPs` class."""

from typing import TYPE_CHECKING, Generator

import numpy.typing as npt

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class IonsFPs(SRIMFile):
    """Class to handle vacancies and interstitials, Frenkel pair (FP),
    produced by SRIM ions creation."""

    def add_data(
        self,
        trimdat: npt.NDArray,
        nsubcollisions: int,
        exclude_vacancies_ion: list[int],
    ) -> None:
        """Adds input.

        Parameters
        ----------
        trimdat : npt.NDArray
            TRIM.DAT intermediate data.
        nsubcollisions : int
            Number of sub-collisions per ion.
        exclude_vacancies_ion : list[int], optional
            If an ion has an atomic number in this list, it will not be considered that it adds
            a vacancy to the target. For example, if you want to simulate a bulk PKA, Fe in Fe,
            leave this list empty and a vacancy will be placed where the PKA starts; however, if you
            want to simulate H in Fe, no vacancies should be created, then this list must be [1].
        """
        ionsfps = []
        for i in range(self.srim.nions):
            nion = i + 1
            jmin = sum(nsubcollisions[:i])
            jmax = jmin + nsubcollisions[i]
            for j in range(jmin, jmax):
                nsubion = j + 1
                subrange3d = list(
                    self.srim.subrange3d.read(
                        what="ion_numb, depth, y, z",
                        condition=f"WHERE ion_numb == {nsubion}",
                    )
                )
                ionfps = []
                if trimdat[j][1] not in exclude_vacancies_ion:
                    vac = [nion, 0, *trimdat[j][3]]
                    ionfps.append(vac)
                if subrange3d:  # safeguard for transmitted/backscattered ions
                    sia = [nion, trimdat[j][1], *subrange3d[0][1:]]
                    ionfps.append(sia)
                ionsfps.append(tuple(ionfps))

        # This works only if no transmitted or backscattered ions are present
        # ionsfps = []
        # for i in range(self.srim.nions):
        #     nion = i + 1
        #     jmin = sum(nsubcollisions[:i])
        #     jmax = jmin + nsubcollisions[i]
        #     for j in range(jmin, jmax):
        #         print(f"{nion=}, {j=}")
        #         vac = [nion, 0, *trimdat[j][3]]
        #         sia = [nion, trimdat[j][1], *subrange3d[j][1:]]
        #         ionsfps.append((vac, sia))

        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS ionsfps"
                "(ion_numb INTEGER, atom_numb INTEGER, depth REAL, y REAL, z REAL)"
            )
        )
        for ionfp in ionsfps:
            for defect in ionfp:
                ion_numb = int(defect[0])
                atomic_number = int(defect[1])
                depth = float(defect[2])
                y = float(defect[3])
                z = float(defect[4])
                cur.execute(
                    (
                        "INSERT INTO ionsfps(ion_numb, atom_numb, depth, y, z)"
                        "VALUES(?, ?, ?, ?, ?)"
                    ),
                    [ion_numb, atomic_number, depth, y, z],
                )
        cur.close()
        self.srim.commit()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads ionsfps data from the database as a generator.

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
        cur.execute(f"SELECT {what} FROM ionsfps {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def merge(self, srimdb2: "SRIMDB") -> None:
        """Merges the ionsfps table with another database.

        Parameters
        ----------
        srimdb2 : SRIMDB
            SRIM database to merge.
        """
        nions = self.srim.nions
        cur = self.cursor()
        cur.execute(f"ATTACH DATABASE '{srimdb2.db_path}' AS srimdb2")
        cur.execute(
            (
                "INSERT INTO ionsfps(ion_numb, type, depth, y, z)"
                "SELECT ion_numb + ?, 0, depth, y, z FROM srimdb2.ionsfps"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()
