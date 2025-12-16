"""This module contains the `SRIMDB` class."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from irradiapy import config
from irradiapy.enums import Phases
from irradiapy.materials.component import Component
from irradiapy.materials.element import Element
from irradiapy.srim.ofiles.backscat import Backscat
from irradiapy.srim.ofiles.collision import Collision
from irradiapy.srim.ofiles.e2recoil import E2Recoil
from irradiapy.srim.ofiles.ioniz import Ioniz
from irradiapy.srim.ofiles.lateral import Lateral
from irradiapy.srim.ofiles.novac import Novac
from irradiapy.srim.ofiles.phonon import Phonon
from irradiapy.srim.ofiles.range import Range
from irradiapy.srim.ofiles.range3d import Range3D
from irradiapy.srim.ofiles.sputter import Sputter
from irradiapy.srim.ofiles.transmit import Transmit
from irradiapy.srim.ofiles.trimdat import Trimdat
from irradiapy.srim.ofiles.vacancy import Vacancy


@dataclass(kw_only=True)
class SRIMDB(sqlite3.Connection):
    """Base class for storing SRIM output data in a SQLite database.

    Attributes
    ----------
    path : Path
        Output database path.
    target : list[Component], optional (default=None)
        SRIM target. Do not provide this argument for read only.
    calculation : str, optional (default=None)
        SRIM calculation. Do not provide this argument for read only.
        Accepted values are: "quick", "full" and "mono".
    seed : int (default=0)
        Seed for SRIM randomness.
    dir_srim : Path (default=config.DIR_SRIM)
        Directory where SRIM is installed.
    check_interval : float
        Interval to check for SRIM window/popups.
    srim_path : Path
        Where all SRIM output files are.
        If given, it will automatically add all those files into the database.
    con : sqlite3.Connection
        Database connection.
    backscat : Backscat
        Class storing `BACKSCAT.txt` data.
    e2recoil : E2Recoil
        Class storing `E2RECOIL.txt` data.
    ioniz : Ioniz
        Class storing `IONIZ.txt` data.
    lateral : Lateral
        Class storing `LATERAL.txt` data.
    phonon : Phonon
        Class storing `PHONON.txt` data.
    range : Range
        Class storing `RANGE.txt` data.
    range3d : Range3D
        Class storing `RANGE_3D.txt` data.
    sputter : Sputter
        Class storing `SPUTTER.txt` data.
    transmit : Transmit
        Class storing `TRANSMIT.txt` data.
    trimdat : Trimdat
        Class storing `TRIM.DAT` data.
    vacancy : Vacancy
        Class storing `VACANCY.txt` data.
    """

    path: Path
    target: None | list[Component] = None
    calculation: None | str = None

    seed: int = 0
    dir_srim: Path = field(default_factory=lambda: config.DIR_SRIM)

    plot_type: int = 5
    xmin: float = 0.0
    xmax: float = 0.0
    do_ranges: int = 1
    do_backscatt: int = 1
    do_transmit: int = 1
    do_sputtered: int = 1
    do_collisions: int = 1
    exyz: float = 0.0
    autosave: int = 0

    reminders: int = field(default=0, init=False)
    bragg: int = field(default=1, init=False)

    backscat: Backscat = field(init=False)
    e2recoil: E2Recoil = field(init=False)
    ioniz: Ioniz = field(init=False)
    lateral: Lateral = field(init=False)
    phonon: Phonon = field(init=False)
    range: Range = field(init=False)
    range3d: Range3D = field(init=False)
    sputter: Sputter = field(init=False)
    transmit: Transmit = field(init=False)
    trimdat: Trimdat = field(init=False)
    vacancy: Vacancy = field(init=False)
    collision: Collision = field(init=False)

    nions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initializes the `SRIMDB` object.

        Parameters
        ----------
        path : Path
            Output database path.
        target : Target, optional (default=None)
            SRIM target. Do not provide this argument for read only.
        calculation : Calculation, optional (default=None)
            SRIM calculation. Do not provide this argument for read only.
        """
        super().__init__(self.path)
        self.backscat = Backscat(self)
        self.e2recoil = E2Recoil(self)
        self.ioniz = Ioniz(self)
        self.lateral = Lateral(self)
        self.phonon = Phonon(self)
        self.range = Range(self)
        self.range3d = Range3D(self)
        self.sputter = Sputter(self)
        self.transmit = Transmit(self)
        self.trimdat = Trimdat(self)
        self.vacancy = Vacancy(self)
        self.collision = Collision(self)

        if self.calculation not in ["quick", "full", "mono", None]:
            raise ValueError("Invalid calculation mode.")

        self.nions = self.get_nions()
        if self.target and self.calculation and not self.table_exists("calculation"):
            self.__save_target_calculation()
        elif (
            not self.target
            and not self.calculation
            and self.table_exists("calculation")
        ):
            self.load_target_calculation()
        elif (self.target and not self.calculation) or (
            not self.target and self.calculation
        ):
            raise ValueError(
                "Both `srim_target` and `calculation` must be provided or None."
            )

        if self.calculation != "quick":
            self.novac = Novac(self)

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        self.close()
        return False

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

    def __save_target_calculation(self) -> None:
        """Saves the target and calculation parameters into the database."""
        cur = self.cursor()
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
        for component in self.target:
            self.__save_component(component)

        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS calculation("
                "mode INTEGER, seed INTEGER, reminders INTEGER, plot_type INTEGER,"
                "xmin REAL, xmax REAL, ranges INTEGER, backscatt INTEGER,"
                "transmit INTEGER, sputtered INTEGER, collisions INTEGER, exyz REAL,"
                "bragg INTEGER, autosave INTEGER)"
            )
        )
        if self.calculation == "quick":
            calculation = 4
        elif self.calculation == "full":
            calculation = 5
        else:
            calculation = 7
        cur.execute(
            (
                "INSERT INTO calculation"
                "(mode, seed, reminders, plot_type, xmin, xmax, ranges, backscatt,"
                "transmit, sputtered, collisions, exyz, bragg, autosave)"
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            [
                calculation,
                self.seed,
                self.reminders,
                self.plot_type,
                self.xmin,
                self.xmax,
                self.do_ranges,
                self.do_backscatt,
                self.do_transmit,
                self.do_sputtered,
                self.do_collisions,
                self.exyz,
                self.bragg,
                self.autosave,
            ],
        )
        cur.close()
        self.commit()

    def load_target_calculation(self) -> list[Component]:
        """Loads the target and calculation parameters from the database."""
        cur = self.cursor()
        cur.execute(
            "SELECT component_id, width, phase, density, srim_bragg FROM components"
        )
        components = list(cur.fetchall())
        target = []
        for component_id, width, phase, density, srim_bragg in components:
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
                phase=Phases.from_int(phase),
                srim_bragg=srim_bragg,
            )
            target.append(component)
        self.target = target

        cur.execute("SELECT * FROM calculation")
        db_calculation = cur.fetchone()
        cur.close()
        self.seed = db_calculation[1]
        self.reminders = db_calculation[2]
        self.plot_type = db_calculation[3]
        self.xmin = db_calculation[4]
        self.xmax = db_calculation[5]
        self.do_ranges = db_calculation[6]
        self.do_backscatt = db_calculation[7]
        self.do_transmit = db_calculation[8]
        self.do_sputtered = db_calculation[9]
        self.do_collisions = db_calculation[10]
        self.exyz = db_calculation[11]
        self.bragg = db_calculation[12]
        self.autosave = db_calculation[13]
        mode = db_calculation[0]
        if mode == 4:
            self.calculation = "quick"
        elif mode == 5:
            self.calculation = "full"
        else:
            self.calculation = "mono"
        cur.close()
        return target

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

    def get_nions(self) -> int:
        """Gets the number of ions in the simulation based on trimdat table."""
        if self.table_exists("trimdat"):
            cur = self.cursor()
            cur.execute("SELECT COUNT(1) FROM trimdat")
            nions = cur.fetchone()[0]
            cur.close()
            return nions
        return 0

    def merge(
        self,
        srimdb: "SRIMDB",
        backscat: bool = True,
        e2recoil: bool = True,
        ioniz: bool = True,
        lateral: bool = True,
        phonon: bool = True,
        range3d: bool = True,
        range_: bool = True,
        sputter: bool = True,
        transmit: bool = True,
        vacancy: bool = True,
        collision: bool = True,
        trimdat: bool = True,
        novac: bool = True,
    ) -> None:
        """Merges two databases.

        Parameters
        ----------
        srimdb : SRIMDB
            SRIM database to merge.
        backscat : bool, optional
            Merge backscattering data.
        e2recoil : bool, optional
            Merge energy to recoil data.
        ioniz : bool, optional
            Merge ionization data.
        lateral : bool, optional
            Merge lateral data.
        phonon : bool, optional
            Merge phonon data.
        range3d : bool, optional
            Merge 3D range data.
        range_ : bool, optional
            Merge range data.
        sputter : bool, optional
            Merge sputtering data.
        transmit : bool, optional
            Merge transmission data.
        vacancy : bool, optional
            Merge vacancy data.
        collision : bool, optional
            Merge collision data.
        trimdat : bool, optional
            Merge TRIMDAT data.
        novac : bool, optional
            Merge NOVAC data.
        """
        if backscat:
            self.backscat.merge(srimdb)
        if e2recoil:
            self.e2recoil.merge(srimdb)
        if ioniz:
            self.ioniz.merge(srimdb)
        if lateral:
            self.lateral.merge(srimdb)
        if phonon:
            self.phonon.merge(srimdb)
        if range3d:
            self.range3d.merge(srimdb)
        if range_:
            self.range.merge(srimdb)
        if sputter:
            self.sputter.merge(srimdb)
        if transmit:
            self.transmit.merge(srimdb)
        if vacancy:
            self.vacancy.merge(srimdb)
        if collision:
            self.collision.merge(srimdb)
        if self.calculation in ["full", "mono"] and novac:
            self.novac.merge(srimdb)
        if trimdat:
            self.trimdat.merge(srimdb)
        self.optimize()

    def append_output(self) -> None:
        """Appends SRIM output files into the database."""
        self.trimdat.process_file(self.dir_srim / "TRIM.DAT")  # This must be first
        self.backscat.process_file(self.dir_srim / "SRIM Outputs/BACKSCAT.txt")
        self.collision.process_file(self.dir_srim / "SRIM Outputs/COLLISON.txt")
        self.e2recoil.process_file(self.dir_srim / "E2RECOIL.txt")
        self.ioniz.process_file(self.dir_srim / "IONIZ.txt")
        self.lateral.process_file(self.dir_srim / "LATERAL.txt")
        self.phonon.process_file(self.dir_srim / "PHONON.txt")
        self.range3d.process_file(self.dir_srim / "SRIM Outputs/RANGE_3D.txt")
        self.range.process_file(self.dir_srim / "RANGE.txt")
        self.sputter.process_file(self.dir_srim / "SRIM Outputs/SPUTTER.txt")
        self.transmit.process_file(self.dir_srim / "SRIM Outputs/TRANSMIT.txt")
        self.nions = self.get_nions()
        self.vacancy.process_file(self.dir_srim / "VACANCY.txt")
        if self.calculation in ["full", "mono"]:
            self.novac.process_file(self.dir_srim / "NOVAC.txt")
