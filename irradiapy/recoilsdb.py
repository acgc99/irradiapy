"""This module contains the `RecoilsDB` class."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Generator

from irradiapy.enums import Phases
from irradiapy.materials.component import Component
from irradiapy.materials.element import Element


@dataclass
class RecoilsDB(sqlite3.Connection):
    """SQLite3 database for SPECTRA-PKA to SRIM events, recoils and ions/vacancies.

    Parameters
    ----------
    path : Path
        Path to the recoils database.
    """

    path: Path
    target: list[Component] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(self.path)
        self.create_tables()
        if self.table_exists("components") and self.table_exists("elements"):
            self.load_target()

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
                "CREATE TABLE IF NOT EXISTS elements ("
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

    def __save_component(self, component: Component) -> None:
        """Save component into the recoils database."""
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO components ("
                "name, phase, density, "
                "x0, y0, z0, width, height, length, "
                "ax, ay, az, c, structure, "
                "ed_min, ed_avr, b_arc, c_arc, "
                "srim_el, srim_es, srim_phase, srim_bragg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?)"
            ),
            (
                component.name,
                component.phase.name,
                component.density,
                component.x0,
                component.y0,
                component.z0,
                component.width,
                component.height,
                component.length,
                component.ax,
                component.ay,
                component.az,
                component.c,
                component.structure,
                component.ed_min,
                component.ed_avr,
                component.b_arc,
                component.c_arc,
                component.srim_el,
                component.srim_es,
                component.srim_phase,
                component.srim_bragg,
            ),
        )
        # Save elements
        component_id = cur.lastrowid
        for element, stoich in zip(component.elements, component.stoichs):
            cur.execute(
                (
                    "INSERT INTO elements (component_id, atomic_number, "
                    "mass_number, symbol, stoich, ed_min, ed_avr, b_arc, c_arc, srim_el, srim_es) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    component_id,
                    element.atomic_number,
                    element.mass_number,
                    element.symbol,
                    stoich,
                    element.ed_min,
                    element.ed_avr,
                    element.b_arc,
                    element.c_arc,
                    element.srim_el,
                    element.srim_es,
                ),
            )
        cur.close()
        self.commit()
        return component_id

    def save_target(self, target: list[Component]) -> None:
        """Saves the target components into the database.

        Parameters
        ----------
        target : list[Component]
            List of components representing the target material.
        """
        for component in target:
            self.__save_component(component)

    def load_target(self) -> list[Component]:
        """Loads the target from the database."""
        cur = self.cursor()
        cur.execute(
            "SELECT component_id, width, srim_phase, density, srim_bragg FROM components"
        )
        components = list(cur.fetchall())
        target = []
        for component_id, width, srim_phase, density, srim_bragg in components:
            phase = Phases.SOLID if srim_phase == 1 else Phases.GAS
            cur.execute(
                (
                    "SELECT component_id, atomic_number, "
                    "mass_number, symbol, stoich, ed_min, ed_avr, b_arc, c_arc, srim_el, srim_es "
                    "FROM elements WHERE component_id = ?"
                ),
                (component_id,),
            )
            db_elements = list(cur.fetchall())
            elements = []
            for (
                _component_id,
                atomic_number,
                mass_number,
                symbol,
                _stoich,
                ed_min,
                ed_avr,
                b_arc,
                c_arc,
                srim_el,
                srim_es,
            ) in db_elements:
                element = Element(
                    atomic_number=atomic_number,
                    mass_number=mass_number,
                    symbol=symbol,
                    ed_min=ed_min,
                    b_arc=b_arc,
                    c_arc=c_arc,
                    ed_avr=ed_avr,
                    srim_el=srim_el,
                    srim_es=srim_es,
                )
                elements.append(element)

            stoichs = [db_element[4] for db_element in db_elements]
            component = Component(
                elements=elements,
                stoichs=stoichs,
                name=f"layer{component_id}",
                width=width,
                density=density,
                phase=phase,
                srim_bragg=srim_bragg,
            )
            target.append(component)

        cur.close()
        self.target = target
        return target

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
