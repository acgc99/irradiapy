"""This module contains the `RecoilsDB` class."""

import sqlite3
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import TracebackType
from typing import Generator

from irradiapy.materials import ELEMENT_BY_ATOMIC_NUMBER
from irradiapy.materials.component import Component as Layer
from irradiapy.materials.element import Element
from irradiapy.materials.srim_target import SRIMTarget


@dataclass
class RecoilsDB(sqlite3.Connection):
    """SQLite3 database for SPECTRA-PKA to SRIM events, recoils and ions/vacancies.

    Parameters
    ----------
    path : Path
        Path to the recoils database.
    """

    path: Path
    srim_target: SRIMTarget = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(self.path)
        self.create_tables()
        if self.table_exists("layers") and self.table_exists("elements"):
            self.load_srim_target()

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        self.close()
        return False

    def optimize(self) -> None:
        """Optimize the SQLite database.

        This method performs two operations to optimize the database:
        1. Executes the "PRAGMA optimize" command to analyze and optimize the database.
        2. Executes the "VACUUM" command to rebuild the database file,
        repacking it into a minimal amount of disk space.
        """
        cur = self.cursor()
        cur.execute("PRAGMA optimize")
        cur.execute("VACUUM")
        cur.close()

    def table_exists(self, table_name: str) -> bool:
        """Checks if the given table exists in the database.

        Parameters
        ----------
        table_name : str
            Table's name to check.

        Returns
        -------
        bool
            Whether the table already exists or not.
        """
        cur = self.cursor()
        cur.execute(
            (
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                f"AND name='{table_name}'"
            )
        )
        result = cur.fetchone()[0]
        cur.close()
        return bool(result)

    def process_config_events(
        self, path_spectrapka_events: Path, exclude_recoils: list[str] | None = None
    ) -> None:
        """Transform SPECTRA-PKA `config_event.pka` file into a SQLite3 table.

        Parameters
        ----------
        path_spectrapka_events : Path
            Path to the SPECTRA-PKA `config_event.pka` file.
        exclude_recoils : list[str] | None (default=None)
            List of symbols of recoils atoms to exclude from processing.
        """
        if exclude_recoils is None:
            exclude_recoils = []
        cur = self.cursor()

        cur.execute("DROP TABLE IF EXISTS spectrapkas")
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS spectrapkas ("
                "atom INTEGER, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, element TEXT, "
                "mass REAL, timestep INTEGER, recoil_energy REAL, time REAL, event INTEGER)"
            )
        )
        with open(path_spectrapka_events, "r", encoding="utf-8") as file:
            file.readline()
            for line in file:
                row = line.split()[:-3]  # exclude last columns (not documented)
                if row[7] in exclude_recoils:
                    continue
                cur.execute(
                    (
                        "INSERT INTO spectrapkas(atom, x, y, z, vx, vy, vz, element, mass, "
                        "timestep, recoil_energy, time, event) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    row,
                )
        self.commit()
        cur.close()

    def create_tables(self) -> None:
        """Create recoils and ions_vacs tables in the database."""
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS recoils ("
                "event INTEGER, atom_numb INTEGER, recoil_energy REAL, depth REAL, "
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS ions_vacs ("
                "event INTEGER, atom_numb INTEGER, depth REAL, "
                "y REAL, z REAL)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS components ("
                "component_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, "
                "phase INTEGER, density REAL, x0 REAL, y0 REAL, z0 REAL, "
                "width REAL, height REAL, length REAL, "
                "ax REAL, ay REAL, az REAL, c REAL, structure TEXT, "
                "ed_min REAL, ed_avr REAL, b_arc REAL, c_arc REAL, calculate_energies BOOLEAN, "
                "srim_el REAL, srim_es REAL, srim_phase INTEGER, srim_bragg INTEGER)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS elements2 ("
                "component_id INTEGER, atomic_number INTEGER, mass_number REAL, "
                "symbol TEXT, stoich REAL, ed_min REAL, ed_avr REAL, b_arc REAL, c_arc REAL, "
                "srim_el REAL, srim_es REAL)"
            )
        )
        cur.close()
        self.commit()

    def insert_recoil(
        self,
        event: int,
        atomic_number: int,
        recoil_energy: float,
        depth: float,
        y: float,
        z: float,
        cosx: float,
        cosy: float,
        cosz: float,
    ) -> None:
        """Insert a recoil into the recoils database.

        Parameters
        ----------
        event : int
            Original SPECTRA event index (1-based).
        atomic_number : int
            Atomic number of the recoil.
        recoil_energy : float
            Recoil energy in eV.
        depth : float
            Depth, x-position, in Angstrom.
        y : float
            y-position in Angstrom.
        z : float
            z-position in Angstrom.
        cosx : float
            x-cosine velocity direction.
        cosy : float
            y-cosine velocity direction.
        cosz : float
            z-cosine velocity direction.
        """
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO recoils(event, atom_numb, recoil_energy, depth, y, z, "
                "cosx, cosy, cosz) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                event,
                atomic_number,
                recoil_energy,
                depth,
                y,
                z,
                cosx,
                cosy,
                cosz,
            ),
        )
        cur.close()

    def insert_ion_vac(
        self,
        event: int,
        atom_numb: int,
        depth: float,
        y: float,
        z: float,
    ) -> None:
        """Insert an ion or vacancy into the ions_vacs database.

        Parameters
        ----------
        event : int
            Original SPECTRA event index (1-based).
        atom_numb : int
            Atomic number of the ion (0 for vacancy).
        depth : float
            Depth, x-position, in Angstrom.
        y : float
            y-position in Angstrom.
        z : float
            z-position in Angstrom.
        """
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO ions_vacs(event, atom_numb, depth, y, z) "
                "VALUES(?, ?, ?, ?, ?)"
            ),
            (event, atom_numb, depth, y, z),
        )
        cur.close()

    def read(
        self, table: str, what: str = "*", condition: str = ""
    ) -> Generator[tuple, None, None]:
        """Reads table data from the database as a generator.

        Parameters
        ----------
        table : str
            Table to read from.
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
        cur.execute(f"SELECT {what} FROM {table} {condition}")
        while True:
            data = cur.fetchone()
            if data:
                yield data
            else:
                break
        cur.close()

    def save_srim_target(self, srim_target: SRIMTarget) -> None:
        """Saves the SRIM target into the database."""
        cur = self.cursor()
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS layers("
                "layer_numb INTEGER, width REAL,"
                "phase INTEGER, density REAL, bragg INTEGER)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS elements("
                "layer_numb INTEGER, stoich REAL, symbol TEXT,"
                "atomic_number INTEGER, atomic_mass REAL,"
                "e_d REAL, e_l REAL, e_s REAL)"
            )
        )
        for i, layer in enumerate(srim_target.layers):
            cur.execute(
                (
                    "INSERT INTO layers"
                    "(layer_numb, width, phase, density, bragg)"
                    "VALUES(?, ?, ?, ?, ?)"
                ),
                [i, layer.width, layer.srim_phase, layer.density, layer.srim_bragg],
            )
            for j, element in enumerate(layer.elements):
                cur.execute(
                    (
                        "INSERT INTO elements"
                        "(layer_numb, stoich, symbol, atomic_number, atomic_mass, e_d, e_l, e_s)"
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    [
                        i,
                        layer.stoichs[j],
                        element.symbol,
                        element.atomic_number,
                        element.mass_number,
                        element.ed_avr,
                        element.srim_el,
                        element.srim_es,
                    ],
                )
        cur.close()
        self.srim_target = srim_target

    def load_srim_target(self) -> SRIMTarget:
        """Loads the SRIM target from the database."""
        cur = self.cursor()
        cur.execute("SELECT * FROM layers")
        db_layers = list(cur.fetchall())
        layers = []
        for layer_numb, width, srim_phase, density, srim_bragg in db_layers:
            phase = Layer.Phases.SOLID if srim_phase == 1 else Layer.Phases.GAS
            cur.execute(("SELECT * FROM elements2 " f"WHERE layer_numb = {layer_numb}"))
            db_elements = list(cur.fetchall())
            elements = []
            for (
                _layer_numb,
                _stoich,
                symbol,
                atomic_number,
                mass_number,
                ed_avr,
                srim_el,
                srim_es,
            ) in db_elements:
                element = Element(
                    atomic_number=atomic_number,
                    mass_number=mass_number,
                    symbol=symbol,
                    ed_min=ELEMENT_BY_ATOMIC_NUMBER[atomic_number].ed_min,
                    b_arc=ELEMENT_BY_ATOMIC_NUMBER[atomic_number].b_arc,
                    c_arc=ELEMENT_BY_ATOMIC_NUMBER[atomic_number].c_arc,
                    ed_avr=ed_avr,
                    srim_el=srim_el,
                    srim_es=srim_es,
                )
                elements.append(element)

            stoichs = [db_element[1] for db_element in db_elements]
            layer = Layer(
                elements=elements,
                stoichs=stoichs,
                name=f"layer{layer_numb}",
                width=width,
                density=density,
                phase=phase,
                srim_bragg=srim_bragg,
            )
            layers.append(layer)

        self.srim_target = SRIMTarget(layers=layers)
        return self.srim_target

    def get_nevents(self) -> int:
        """Get the number of events in the recoils database.

        Returns
        -------
        int
            Number of events.
        """
        cur = self.cursor()
        nevents = cur.execute("SELECT MAX(event) FROM recoils").fetchone()[0]
        cur.close()
        return nevents if nevents is not None else 0
