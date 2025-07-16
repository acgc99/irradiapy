"""This module contains the `Collision` class."""

import sqlite3
from typing import TYPE_CHECKING, Generator

from irradiapy.srim.ofiles.srimfile import SRIMFile

if TYPE_CHECKING:
    from irradiapy.srim.srimdb import SRIMDB


class Collision(SRIMFile):
    """Class for processing the `COLLISON.txt` file."""

    def __init__(self, srimdb: "SRIMDB") -> None:
        """Initializes the `Collision` object.

        Parameters
        ----------
        srimdb : SRIMDB
            `SRIMDB` object.
        """
        super().__init__(srimdb)
        if self.srim.calculation == "quick":
            self.create_table = self.__create_table_qc
            self.insert = self.__insert_qc
            self.merge = self.__merge_qc
        else:
            self.create_table = self.__create_table_fc
            self.insert = self.__insert_fc
            self.merge = self.__merge_fc

    def __create_table_qc(self) -> None:
        """Creates the collision table for Quick-Calculation mode."""
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE collision"
                "(ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL,"
                "cosx REAL, cosy REAL, cosz REAL, se REAL, atom_hit TEXT,"
                "recoil_energy REAL, target_disp REAL)"
            )
        )
        cur.close()
        self.srim.commit()

    def __create_table_fc(self) -> None:
        """Creates the collision table for Full-Calculation mode."""
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE collision"
                "(ion_numb INTEGER, energy REAL, depth REAL, y REAL, z REAL,"
                "cosx REAL, cosy REAL, cosz REAL, se REAL, atom_hit TEXT,"
                "recoil_energy REAL, target_disp INTEGER, target_vac INTEGER,"
                "target_replac INTEGER, target_inter INTEGER)"
            )
        )
        cur.close()
        self.srim.commit()

    def __insert_qc(
        self,
        cur: sqlite3.Cursor,
        nion: int,
        energy: float,
        depth: float,
        y: float,
        z: float,
        cosx: float,
        cosy: float,
        cosz: float,
        se: float,
        atom_hit: str,
        recoil_energy: float,
        target_disp: float,
    ) -> None:
        """Inserts data into the collision table for Quick-Calculation mode.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor.
        nion : int
            Ion number.
        energy : float
            Energy.
        depth : float
            Depth.
        y : float
            Y coordinate.
        z : float
            Z coordinate.
        cosx : float
            X direction.
        cosy : float
            Y direction.
        cosz : float
            Z direction.
        se : float
            Stopping power.
        atom_hit : str
            Atom hit.
        recoil_energy : float
            Recoil energy.
        target_disp : int
            Target displacement.
        """
        # try:
        cur.execute(
            (
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp)"
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                nion,
                energy,
                depth,
                y,
                z,
                cosx,
                cosy,
                cosz,
                se,
                atom_hit,
                recoil_energy,
                target_disp,
            ],
        )

    def __insert_fc(
        self,
        cur: sqlite3.Cursor,
        nion: int,
        energy: float,
        depth: float,
        y: float,
        z: float,
        cosx: float,
        cosy: float,
        cosz: float,
        se: float,
        atom_hit: str,
        recoil_energy: float,
        target_disp: int,
        target_vac: int,
        target_replac: int,
        target_inter: int,
    ) -> None:
        """Inserts data into the collision table for Full-Calculation mode.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor.
        nion : int
            Ion number.
        energy : float
            Energy.
        depth : float
            Depth.
        y : float
            Y coordinate.
        z : float
            Z coordinate.
        cosx : float
            X direction.
        cosy : float
            Y direction.
        cosz : float
            Z direction.
        se : float
            Stopping power.
        atom_hit : str
            Atom hit.
        recoil_energy : float
            Recoil energy.
        target_disp : int
            Target displacement.
        target_vac : int
            Target vacancy.
        target_replac : int
            Target replacement.
        target_inter : int
            Target interstitial.
        """
        # try:
        cur.execute(
            (
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp, target_vac, target_replac,"
                "target_inter)"
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                nion,
                energy,
                depth,
                y,
                z,
                cosx,
                cosy,
                cosz,
                se,
                atom_hit,
                recoil_energy,
                target_disp,
                target_vac,
                target_replac,
                target_inter,
            ],
        )

    def __merge_qc(self, srimdb2: "SRIMDB") -> None:
        """Merges the collision table with another database for Quick-Calculation mode.

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
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp, target_vac, target_replac,"
                "target_inter) SELECT ion_numb + ?, energy, depth, y, z, cosx, cosy,"
                "cosz, se, atom_hit, recoil_energy, target_disp "
                "FROM srimdb2.collision"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()

    def __merge_fc(self, srimdb2: "SRIMDB") -> None:
        """Merges the collision table with another database for Full-Calculation mode.

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
                "INSERT INTO collision"
                "(ion_numb, energy, depth, y, z, cosx, cosy, cosz, se, atom_hit,"
                "recoil_energy, target_disp, target_vac, target_replac,"
                "target_inter) SELECT ion_numb + ?, energy, depth, y, z, cosx, cosy,"
                "cosz, se, atom_hit, recoil_energy, target_disp, target_vac,"
                "target_replac, target_inter FROM srimdb2.collision"
            ),
            (nions,),
        )
        self.srim.commit()
        cur.execute("DETACH DATABASE srimdb2")
        cur.close()

    def read(
        self, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads data from the collision table.

        Parameters
        ----------
        what : str, optional (default="*")
            Columns to read.
        condition : str, optional (default="")
            Condition to filter data.

        Yields
        ------
        tuple
            Data from the collision table.
        """
        cur = self.cursor()
        cur.execute(f"SELECT {what} FROM collision {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()
