"""This module contains the `RecoilsDB` class."""

from dataclasses import dataclass, field
from pathlib import Path

from irradiapy.database import Database
from irradiapy.enums import Phases
from irradiapy.materials.component import Component
from irradiapy.materials.element import Element


@dataclass
class RecoilsDB(Database):
    """SQLite3 database for SPECTRA-PKA to SRIM events, recoils and ions/vacancies."""

    target: list[Component] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.create_tables()
        if self.table_exists("components") and self.table_exists("elements"):
            self.load_target()

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
                "event INTEGER, atom_numb INTEGER, recoil_energy REAL, x REAL, "
                "y REAL, z REAL, cosx REAL, cosy REAL, cosz REAL)"
            )
        )
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS ions_vacs ("
                "event INTEGER, atom_numb INTEGER, x REAL, "
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
                "component_id INTEGER, atomic_number INTEGER, atomic_weight REAL, "
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
        x: float,
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
        x : float
            x-position, in Angstrom.
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
                "INSERT INTO recoils(event, atom_numb, recoil_energy, x, y, z, "
                "cosx, cosy, cosz) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                event,
                atomic_number,
                recoil_energy,
                x,
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
        x: float,
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
        x : float
            x-position, in Angstrom.
        y : float
            y-position in Angstrom.
        z : float
            z-position in Angstrom.
        """
        cur = self.cursor()
        cur.execute(
            (
                "INSERT INTO ions_vacs(event, atom_numb, x, y, z) "
                "VALUES(?, ?, ?, ?, ?)"
            ),
            (event, atom_numb, x, y, z),
        )
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
                component.phase.to_int(),
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
                    "atomic_weight, symbol, stoich, ed_min, ed_avr, b_arc, c_arc, srim_el, srim_es) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    component_id,
                    element.atomic_number,
                    element.atomic_weight,
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
            "SELECT component_id, name, phase, density, width, height, length, ax, structure, srim_bragg FROM components"
        )
        components = list(cur.fetchall())
        target = []
        for (
            component_id,
            name,
            phase,
            density,
            width,
            height,
            length,
            ax,
            structure,
            srim_bragg,
        ) in components:
            cur.execute(
                (
                    "SELECT component_id, atomic_number, "
                    "atomic_weight, symbol, stoich, ed_min, ed_avr, b_arc, c_arc, srim_el, srim_es "
                    "FROM elements WHERE component_id = ?"
                ),
                (component_id,),
            )
            db_elements = list(cur.fetchall())
            elements = []
            for (
                _component_id,
                atomic_number,
                atomic_weight,
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
                    atomic_weight=atomic_weight,
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
                name=name,
                phase=Phases.from_int(phase),
                density=density,
                width=width,
                height=height,
                length=length,
                ax=ax,
                structure=structure,
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

    def get_nrecoils(self) -> int:
        """Get the number of recoils in the recoils database.

        Returns
        -------
        int
            Number of recoils.
        """
        cur = self.cursor()
        nrecoils = cur.execute("SELECT COUNT(*) FROM recoils").fetchone()[0]
        cur.close()
        return nrecoils if nrecoils is not None else 0
