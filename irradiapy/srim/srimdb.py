"""This module contains the `SRIMDB` class."""

# pylint: disable=too-many-lines,protected-access
import os
import platform
import sqlite3
import subprocess
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Callable

import numpy as np
from numpy import typing as npt

from irradiapy import config, dtypes, materials
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
from irradiapy.srim.ofiles.subcollision import Subcollision
from irradiapy.srim.ofiles.transmit import Transmit
from irradiapy.srim.ofiles.trimdat import Trimdat
from irradiapy.srim.ofiles.vacancy import Vacancy
from irradiapy.srim.target.element import Element
from irradiapy.srim.target.layer import Layer
from irradiapy.srim.target.target import Target

platform = platform.system()
if platform == "Windows":
    import pygetwindow
    import pywinauto
else:
    warnings.warn(
        "SRIM subpackage only works for Windows. " f"'{platform}' not supported."
    )


@dataclass(kw_only=True)
class SRIMDB(sqlite3.Connection):
    """Base class for running SRIM calculations and storing the output data in a SQLite database.

    Attributes
    ----------
    path_db : Path
        Output database path.
    target : Target, optional
        SRIM target.
    calculation : str, optional
        SRIM calculation.
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

    path_db: Path
    calculation: None | str = None
    target: None | Target = None
    check_interval: float = 0.2
    seed: int = 0
    reminders: int = 0
    plot_type: int = 5
    xmin: float = 0.0
    xmax: float = 0.0
    do_ranges: int = 1
    do_backscatt: int = 1
    do_transmit: int = 1
    do_sputtered: int = 1
    do_collisions: int = 1
    exyz: float = 0.0
    bragg: int = 1
    autosave: int = 0
    dir_srim: Path = field(default_factory=lambda: config.DIR_SRIM)

    def __post_init__(self) -> None:
        """Initializes the `SRIMDB` object.

        Parameters
        ----------
        path_db : Path
            Output database path.
        target : Target, optional (default=None)
            SRIM target. Do not provide this argument for read only.
        calculation : Calculation, optional (default=None)
            SRIM calculation. Do not provide this argument for read only.
        check_interval : float, optional (default=0.2)
            Interval to check for SRIM window/popups.
        """
        super().__init__(self.path_db)
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
        self.subcollision = Subcollision(self)

        if self.calculation not in ["quick", "full", "mono", None]:
            raise ValueError("Invalid calculation mode.")

        self.nions = self.get_nions()
        if self.target and self.calculation and not self.table_exists("calculation"):
            self.save_target_calculation()
        elif (
            not self.target
            and not self.calculation
            and self.table_exists("calculation")
        ):
            self.target = self.load_target_calculation()
        elif (self.target and not self.calculation) or (
            not self.target and self.calculation
        ):
            raise ValueError(
                "Both `target` and `calculation` must be provided or None."
            )

        if self.calculation == "quick":
            self._filter_subcollisions_logic = self.__filter_subcollisions_logic_qc
        else:
            self.novac = Novac(self)
            self._filter_subcollisions_logic = self.__filter_subcollisions_logic_fc

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        self.close()
        return False

    def save_target_calculation(self) -> None:
        """Saves the target and calculation parameters into the database."""
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
        for i, layer in enumerate(self.target.layers):
            cur.execute(
                (
                    "INSERT INTO layers"
                    "(layer_numb, width, phase, density, bragg)"
                    "VALUES(?, ?, ?, ?, ?)"
                ),
                [i, layer.width, layer.phase, layer.density, layer.bragg],
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
                        element.atomic_mass,
                        element.e_d,
                        element.e_l,
                        element.e_s,
                    ],
                )
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

    def load_target_calculation(self) -> Target:
        """Loads the target and calculation parameters from the database."""
        cur = self.cursor()
        cur.execute("SELECT * FROM layers")
        db_layers = list(cur.fetchall())
        layers = []
        for db_layer in db_layers:
            cur.execute(("SELECT * FROM elements " f"WHERE layer_numb = {db_layer[0]}"))
            db_elements = list(cur.fetchall())
            elements = []
            for db_element in db_elements:
                elements.append(
                    Element(
                        db_element[2],
                        db_element[3],
                        db_element[4],
                        db_element[5],
                        db_element[6],
                        db_element[7],
                    )
                )
            stoichs = [db_element[1] for db_element in db_elements]
            layers.append(
                Layer(
                    db_layer[1],
                    db_layer[2],
                    db_layer[3],
                    elements,
                    stoichs,
                    db_layer[4],
                )
            )
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
        return Target(layers)

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
        """Checks if the given table already exists in the database.

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
        """Gets the number of ions in the simulation."""
        if self.table_exists("trimdat"):
            cur = self.cursor()
            cur.execute("SELECT COUNT(1) FROM trimdat")
            nions = cur.fetchone()[0]
            cur.close()
            return nions
        return 0

    def append_backscat(self, backscat_path: Path) -> None:
        """Appends `BACKSCAT.txt` into the database.

        Parameters
        ----------
        backscat_path : Path
            `BACKSCAT.txt` path.
        """
        self.backscat.process_file(backscat_path)

    def append_e2recoil(self, e2recoil_path: Path) -> None:
        """Appends `E2RECOIL.txt` into the database.

        Parameters
        ----------
        e2recoil_path : Path
            `E2RECOIL.txt` path.
        """
        self.e2recoil.process_file(e2recoil_path)

    def append_ioniz(self, ioniz_path: Path) -> None:
        """Appends `IONIZ.txt` into the database.

        Parameters
        ----------
        ioniz_path : Path
            `IONIZ.txt` path.
        """
        self.ioniz.process_file(ioniz_path)

    def append_lateral(self, lateral_path: Path) -> None:
        """Appends `LATERAL.txt` into the database.

        Parameters
        ----------
        lateral_path : Path
            `LATERAL.txt` path.
        """
        self.lateral.process_file(lateral_path)

    def append_phonon(self, phonon_path: Path) -> None:
        """Appends `PHONON.txt` into the database.

        Parameters
        ----------
        phonon_path : Path
            `PHONON.txt` path.
        """
        self.phonon.process_file(phonon_path)

    def append_range(self, range_path: Path) -> None:
        """Appends `RANGE.txt` into the database.

        Parameters
        ----------
        range_path : Path
            `RANGE.txt` path.
        """
        self.range.process_file(range_path)

    def append_range3d(self, range3d_path: Path) -> None:
        """Appends `RANGE_3D.txt` into the database.

        Parameters
        ----------
        range3d_path : Path
            `RANGE_3D.txt` path.
        """
        self.range3d.process_file(range3d_path)

    def append_sputter(self, sputter_path: Path) -> None:
        """Appends `SPUTTER.txt` into the database.

        Parameters
        ----------
        sputter_path : Path
            `SPUTTER.txt` path.
        """
        self.sputter.process_file(sputter_path)

    def append_transmit(self, transmit_path: Path) -> None:
        """Appends `TRANSMIT.txt` into the database.

        Parameters
        ----------
        transmit_path : Path
            `TRANSMIT.txt` path.
        """
        self.transmit.process_file(transmit_path)

    def append_trimdat(self, trimdat_path: Path) -> None:
        """Appends `TRIM.DAT` into the database.

        Parameters
        ----------
        trimdat_path : Path
            `TRIM.DAT` path.
        """
        self.trimdat.process_file(trimdat_path)
        self.nions = self.get_nions()

    def append_vacancy(self, vacancy_path: Path) -> None:
        """Appends `VACANCY.txt` into the database.

        Parameters
        ----------
        vacancy_path : Path
            `VACANCY.txt` path.
        """
        self.vacancy.process_file(vacancy_path)

    def append_subcollision(self, collision_path: Path) -> None:
        """Appends currect iteration `COLLISON.txt` into the database.

        Parameters
        ----------
        collision_path : Path
            `COLLISON.txt` path.
        """
        self.subcollision.process_file(collision_path)

    def append_novac(self, novac_path: Path) -> None:
        """Appends `NOVAC.txt` into the database.

        Parameters
        ----------
        novac_path : Path
            `NOVAC.txt` path.
        """
        self.novac.process_file(novac_path)

    def merge(
        self,
        srimdb2: "SRIMDB",
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
        srimdb2 : SRIMDBIter
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
            self.backscat.merge(srimdb2)
        if e2recoil:
            self.e2recoil.merge(srimdb2)
        if ioniz:
            self.ioniz.merge(srimdb2)
        if lateral:
            self.lateral.merge(srimdb2)
        if phonon:
            self.phonon.merge(srimdb2)
        if range3d:
            self.range3d.merge(srimdb2)
        if range_:
            self.range.merge(srimdb2)
        if sputter:
            self.sputter.merge(srimdb2)
        if transmit:
            self.transmit.merge(srimdb2)
        if vacancy:
            self.vacancy.merge(srimdb2)
        if collision:
            self.collision.merge(srimdb2)
        if self.calculation in ["full", "mono"] and novac:
            self.novac.merge(srimdb2)
        if trimdat:
            self.trimdat.merge(srimdb2)
        self.optimize()

    def generate_trimin(
        self,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        target: Target,
    ) -> None:
        """Generates `TRIM.IN` file."""
        nions = len(atomic_numbers)
        atomic_mass = materials.MATERIALS_BY_ATOMIC_NUMBER[
            atomic_numbers[0]
        ].mass_number
        energy = np.ceil(energies.max()) / 1e3
        if self.calculation == "quick":
            calculation = 4
        elif self.calculation == "full":
            calculation = 5
        else:
            calculation = 7
        trimin_path = self.dir_srim / "TRIM.IN"
        with open(trimin_path, "w", encoding="utf-8") as file:
            file.write("TRIM.IN file generated by irradiapy.\n")
            file.write(
                (
                    "ion Z, A, energy, angle, number of ions, Bragg correction, autosave\n"
                    f"{atomic_numbers[0]} {atomic_mass} {energy} 0 "
                    f"{nions} {self.bragg} {self.autosave}\n"
                )
            )
            file.write(
                (
                    "calculation, seed, reminders\n"
                    f"{calculation} {self.seed} {self.reminders}\n"
                )
            )
            file.write(
                (
                    "Diskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, "
                    "Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file\n"
                    f"{self.do_ranges} {self.do_backscatt} {self.do_transmit} {self.do_sputtered} "
                    f"{self.do_collisions} {self.exyz}\n"
                )
            )
            file.write(target.trimin_description() + "\n")
            file.write(
                (
                    "PlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]\n"
                    f"{self.plot_type} {self.xmin} {self.xmax}\n"
                )
            )
            file.write(target.trimin_target_elements())
            file.write(target.trimin_target_layers() + "\n")
            file.write(target.trimin_phases() + "\n")
            file.write(target.trimin_bragg() + "\n")
            file.write(target.trimin_displacement() + "\n")
            file.write(target.trimin_lattice() + "\n")
            file.write(target.trimin_surface() + "\n")
            file.write("Stopping Power Version\n0\n")

    def generate_trimdat(
        self,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: None | npt.NDArray[np.float64] = None,
        ys: None | npt.NDArray[np.float64] = None,
        zs: None | npt.NDArray[np.float64] = None,
        cosxs: None | npt.NDArray[np.float64] = None,
        cosys: None | npt.NDArray[np.float64] = None,
        coszs: None | npt.NDArray[np.float64] = None,
    ) -> npt.NDArray[np.float64]:
        """Generates `TRIM.DAT` file.

        Parameters
        ----------
        atomic_numbers : npt.NDArray[np.int64]
            Atomic numbers.
        energies : npt.NDArray[np.float64]
            Energies.
        depths : npt.NDArray[np.float64], optional
            Depths.
        ys : npt.NDArray[np.float64], optional
            Y positions.
        zs : npt.NDArray[np.float64], optional
            Z positions.
        cosxs : npt.NDArray[np.float64], optional
            X directions.
        cosys : npt.NDArray[np.float64], optional
            Y directions.
        coszs : npt.NDArray[np.float64], optional
            Z directions.

        Returns
        -------
        npt.NDArray[np.float64]
            `TRIM.DAT` data.
        """
        trimdat_path = self.dir_srim / "TRIM.DAT"
        nions = atomic_numbers.size
        if depths is None:
            depths = np.zeros(nions)
        if ys is None:
            ys = np.zeros(nions)
        if zs is None:
            zs = np.zeros(nions)
        if cosxs is None:
            cosxs = np.ones(nions)
        if cosys is None:
            cosys = np.zeros(nions)
        if coszs is None:
            coszs = np.zeros(nions)
        names = np.array([f"{i:06d}" for i in range(1, nions + 1)], dtype=str)
        with open(trimdat_path, "w", encoding="utf-8") as file:
            file.write("<TRIM>\n" * 9)
            file.write("Name atomic_number E x y z cosx cosy cosz\n")
            for i in range(nions):
                file.write(
                    (
                        f"{names[i]} {atomic_numbers[i]} {energies[i]} {depths[i]} {ys[i]} "
                        f"{zs[i]} {cosxs[i]} {cosys[i]} {coszs[i]}\n"
                    )
                )
        data = np.empty(nions, dtype=dtypes.trimdat)
        for i in range(nions):
            data[i]["name"] = names[i]
            data[i]["atomic_number"] = atomic_numbers[i]
            data[i]["energy"] = energies[i]
            data[i]["pos"] = np.array([depths[i], ys[i], zs[i]])
            data[i]["dir"] = np.array([cosxs[i], cosys[i], coszs[i]])
        return data

    def __generate_trimauto(self) -> None:
        """Generates `TRIMAUTO` file."""
        with open(self.dir_srim / "TRIMAUTO", "w", encoding="utf-8") as file:
            file.write("1\n\nCheck TRIMAUTO.txt for details.\n")

    def minimize_and_handle_popup(self):
        """Minimizes the SRIM window and handles the end of calculation popup."""
        window_title = "SRIM-2013.00"
        popup_title = "End of TRIM.DAT calculation"
        if platform == "Windows":
            # Minimize window
            while True:
                windows = pygetwindow.getWindowsWithTitle(window_title)
                if windows:
                    window = windows[0]
                    app = pywinauto.Application().connect(handle=window._hWnd)
                    app.window(handle=window._hWnd).minimize()
                    break
                time.sleep(self.check_interval)
            # Dismiss popup (quick-calculation does not have this popup)
            if self.calculation != "quick":
                while True:
                    popups = pygetwindow.getWindowsWithTitle(popup_title)
                    if popups:
                        popup = popups[0]
                        app = pywinauto.Application().connect(handle=popup._hWnd)
                        app.window(handle=popup._hWnd).send_keystrokes("{ENTER}")
                        break
                    time.sleep(self.check_interval)
        elif platform == "Linux":
            # Minimize window
            while True:
                windows = pygetwindow.getWindowsWithTitle(window_title)
                if windows:
                    window = windows[0]
                    window_id = window._hWnd  # pylint: disable=protected-access
                    subprocess.run(
                        ["xdotool", "windowminimize", str(window_id)], check=True
                    )
                    break
                time.sleep(self.check_interval)
            # Dismiss popup (quick-calculation does not have this popup)
            if self.calculation != "quick":
                while True:
                    popups = pygetwindow.getWindowsWithTitle(popup_title)
                    if popups:
                        popup = popups[0]
                        popup_id = popup._hWnd  # pylint: disable=protected-access
                        subprocess.run(
                            ["xdotool", "windowactivate", str(popup_id)], check=True
                        )
                        subprocess.run(["xdotool", "key", "Return"], check=True)
                        break
                    time.sleep(self.check_interval)

    def run(
        self,
        criterion: Callable,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        remove_offsets: bool,
        depths: None | npt.NDArray[np.float64] = None,
        ys: None | npt.NDArray[np.float64] = None,
        zs: None | npt.NDArray[np.float64] = None,
        cosxs: None | npt.NDArray[np.float64] = None,
        cosys: None | npt.NDArray[np.float64] = None,
        coszs: None | npt.NDArray[np.float64] = None,
        iter_max: None | int = None,
        ignore_32bit_warning: bool = True,
    ) -> None:
        """Runs the SRIM simulation.

        Parameters
        ----------
        criterion : Callable
            Criterion to repeat calculation, must return False to repeat calculation.
            Its signature is:
            `criterion(nion, energy, depth, y, z, se, atom_hit,
            pka_e, target_disp)`.
            Recommended to be defined as `def criterion(**kwargs: dict) -> bool:`.
        atomic_numbers : npt.NDArray[np.int64]
            Ion atomic numbers.
        energies : npt.NDArray[np.float64]
            Ion energies.
        remove_offsets : bool
            Whether to remove initial depth offsets or not.
        depths : npt.NDArray[np.float64], optional
            Ion initial depths.
        ys : npt.NDArray[np.float64], optional
            Ion initial y positions.
        zs : npt.NDArray[np.float64], optional
            Ion initial z positions.
        cosxs : npt.NDArray[np.float64], optional
            Ion initial x directions.
        cosys : npt.NDArray[np.float64], optional
            Ion initial y directions.
        coszs : npt.NDArray[np.float64], optional
            Ion initial z directions.
        iter_max : int, optional (default=None)
            Maximum number of iterations.
        """
        if ignore_32bit_warning:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="32-bit application should be automated using 32-bit Python",
            )

        if self.table_exists("collision"):
            raise RuntimeError(
                (
                    f"The database {self.path_db} is already populated "
                    "with data from another simulation, use another one."
                )
            )
        self.collision.create_table()
        cur = self.cursor()

        # First iteration
        nions = len(atomic_numbers)
        nsubcollisions = np.ones(nions, dtype=np.int64)
        nsubcollisions0 = nsubcollisions.copy()
        trimdat = self._run_iter(
            self.target,
            atomic_numbers,
            energies,
            depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
        )
        self.__append_output()
        (
            nsubcollisions,
            atomic_numbers,
            energies,
            depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
        ) = self._filter_subcollisions(cur, nions, trimdat, nsubcollisions0, criterion)
        nsubcollisions_total = nsubcollisions.sum()

        niter = 1
        while nsubcollisions_total and niter != iter_max:
            niter += 1
            nsubcollisions0 = nsubcollisions.copy()
            trimdat = self._run_iter(
                self.target,
                atomic_numbers,
                energies,
                depths,
                ys,
                zs,
                cosxs,
                cosys,
                coszs,
            )
            (
                nsubcollisions,
                atomic_numbers,
                energies,
                depths,
                ys,
                zs,
                cosxs,
                cosys,
                coszs,
            ) = self._filter_subcollisions(
                cur, nions, trimdat, nsubcollisions0, criterion
            )
            nsubcollisions_total = nsubcollisions.sum()

        # Reordering database might reduce I/O operations filtered by ion number
        # PKAs are not sequentally ordered
        cur.execute(
            "CREATE TABLE collision0 AS SELECT * FROM collision ORDER BY ion_numb"
        )
        cur.execute("DROP TABLE collision")
        cur.execute("ALTER TABLE collision0 RENAME TO collision")
        cur.execute(
            "CREATE INDEX ionRecoilEenrgyIdx ON collision(ion_numb, recoil_energy)"
        )
        self.commit()
        cur.close()
        if remove_offsets:
            self.__remove_offsets()
        self.optimize()

        if ignore_32bit_warning:
            warnings.filterwarnings(
                "default",
                category=UserWarning,
                message="32-bit application should be automated using 32-bit Python",
            )

    def _run_iter(
        self,
        target: Target,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: None | npt.NDArray[np.float64] = None,
        ys: None | npt.NDArray[np.float64] = None,
        zs: None | npt.NDArray[np.float64] = None,
        cosxs: None | npt.NDArray[np.float64] = None,
        cosys: None | npt.NDArray[np.float64] = None,
        coszs: None | npt.NDArray[np.float64] = None,
    ) -> dtypes.Trimdat:
        """Runs SRIM for a single iteration.

        Parameters
        ----------
        target : Target
            Target material.
        calculation : Calculation
            Calculation parameters.
        atomic_numbers : npt.NDArray[np.int64]
            Ion atomic numbers.
        energies : npt.NDArray[np.float64]
            Ion energies.
        depths : npt.NDArray[np.float64], optional
            Ion initial depths.
        ys : npt.NDArray[np.float64], optional
            Ion initial y positions.
        zs : npt.NDArray[np.float64], optional
            Ion initial z positions.
        cosxs : npt.NDArray[np.float64], optional
            Ion initial x directions.
        cosys : npt.NDArray[np.float64], optional
            Ion initial y directions.
        coszs : npt.NDArray[np.float64], optional
            Ion initial z directions.

        Returns
        -------
        dtypes.Trimdat
            `TRIM.DAT` data.
        """
        self.__generate_trimauto()
        self.generate_trimin(atomic_numbers, energies, target)
        trimdat = self.generate_trimdat(
            atomic_numbers,
            energies,
            depths=depths,
            ys=ys,
            zs=zs,
            cosxs=cosxs,
            cosys=cosys,
            coszs=coszs,
        )
        # Run SRIM
        print(f"Running {len(atomic_numbers)} SRIM ions")
        try:
            window_thread = threading.Thread(target=self.minimize_and_handle_popup)
            window_thread.start()
            curr_dir = os.getcwd()
            os.chdir(self.dir_srim)
            subprocess.check_call([Path("./TRIM.exe")])
            os.chdir(curr_dir)
            window_thread.join()
        except subprocess.CalledProcessError as e:
            print(traceback.format_exc())
            print(f"An error occurred while running the subprocess: {e}")
        return trimdat

    def __append_output(self) -> None:
        """Appends SRIM output files into the database.

        Note
        ----
        Although reading and saving information from the TRIM.DAT and TRIM.IN files
        instead of directly using the Python objects provided as input to SRIMDB.run
        may seem redundant, this approach is maintained to preserve the ability to read
        SRIM configuration files. The resulting performance loss is relatively minor.
        """
        self.append_backscat(self.dir_srim / "SRIM Outputs/BACKSCAT.txt")
        self.append_e2recoil(self.dir_srim / "E2RECOIL.txt")
        self.append_ioniz(self.dir_srim / "IONIZ.txt")
        self.append_lateral(self.dir_srim / "LATERAL.txt")
        self.append_phonon(self.dir_srim / "PHONON.txt")
        self.append_range3d(self.dir_srim / "SRIM Outputs/RANGE_3D.txt")
        self.append_range(self.dir_srim / "RANGE.txt")
        self.append_sputter(self.dir_srim / "SRIM Outputs/SPUTTER.txt")
        self.append_transmit(self.dir_srim / "SRIM Outputs/TRANSMIT.txt")
        self.append_trimdat(self.dir_srim / "TRIM.DAT")
        self.append_vacancy(self.dir_srim / "VACANCY.txt")
        if self.calculation in ["full", "mono"]:
            self.append_novac(self.dir_srim / "NOVAC.txt")

    def _get_dir(
        self, pos0: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Gets the direction from two positions.

        Parameters
        pos0 : npt.NDArray[np.float64]
            Initial position.
        pos : npt.NDArray[np.float64]
            Final position.
        """
        diff = pos - pos0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            direction = diff / np.linalg.norm(diff)
        return direction

    def __filter_subcollisions_logic_qc(self, subcollision_data: tuple) -> dict:
        """Convert subcollision data into dictionary for
        later handling for Quick-Calculation mode.

        Parameters
        ----------
        subcollision_data : tuple
            Subcollision data.
        """
        _, energy, depth, y, z, se, atom_hit, recoil_energy, target_disp = (
            subcollision_data
        )
        return {
            "energy": energy,
            "depth": depth,
            "y": y,
            "z": z,
            "se": se,
            "atom_hit": atom_hit,
            "recoil_energy": recoil_energy,
            "target_disp": target_disp,
        }

    def __filter_subcollisions_logic_fc(self, subcollision_data: tuple) -> dict:
        """Convert subcollision data into dictionary for
        later handling for Full-Calculation mode.

        Parameters
        ----------
        subcollision_data : tuple
            Subcollision data.
        """
        (
            _,
            energy,
            depth,
            y,
            z,
            se,
            atom_hit,
            recoil_energy,
            target_disp,
            target_vac,
            target_replac,
            target_inter,
        ) = subcollision_data
        return {
            "energy": energy,
            "depth": depth,
            "y": y,
            "z": z,
            "se": se,
            "atom_hit": atom_hit,
            "recoil_energy": recoil_energy,
            "target_disp": target_disp,
            "target_vac": target_vac,
            "target_replac": target_replac,
            "target_inter": target_inter,
        }

    def _filter_subcollisions(
        self,
        cur: sqlite3.Cursor,
        nions: int | np.int64,
        trimdat: dtypes.Trimdat,
        nsubcollisions0: npt.NDArray[np.int64],
        criterion: Callable,
    ) -> tuple:
        """Filters subcollisions.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor.
        nions : int | np.int64
            Number of ions.
        trimdat : dtypes.Trimdat
            TRIMDAT data.
        nsubcollisions0 : npt.NDArray[np.int64]
            Initial number of subcollisions.
        criterion : Callable
            Criterion to repeat calculation, must return False to repeat calculation.
        """
        self.append_subcollision(self.dir_srim / "SRIM Outputs/COLLISON.txt")
        atomic_numbers, recoil_energies, depths, ys, zs, cosxs, cosys, coszs = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        nsubcollisions = np.zeros(nions, dtype=np.int64)
        for nion in range(nions):
            nsubcollision_min = 0 if nion == 0 else nsubcollision_max
            nsubcollision_max = nsubcollision_min + nsubcollisions0[nion]
            for nsubcollision in range(nsubcollision_min, nsubcollision_max):
                pos0 = trimdat[nsubcollision]["pos"]
                cosx0, cosy0, cosz0 = trimdat[nsubcollision]["dir"]
                for subcollision in self.subcollision.read(
                    condition=f"WHERE ion_numb = {nsubcollision + 1}"
                ):
                    subcollision = self._filter_subcollisions_logic(subcollision)
                    pos = np.array(
                        [subcollision["depth"], subcollision["y"], subcollision["z"]]
                    )
                    cosx, cosy, cosz = self._get_dir(pos0, pos)
                    # There are some rare cases, specially at high energies,
                    # when there are two PKA at the same position. I think this
                    # is because they are really close and when saved into
                    # COLLISON.txt, positions are rounded and they coincide.
                    # In such cases (if statement triggered), we assume that the
                    # second PKA has the same direction as the first one.
                    if np.isnan(cosx) or np.isnan(cosy) or np.isnan(cosz):
                        warnings.warn(
                            (
                                "Two PKAs at the same position, assuming same direction. This is "
                                "likely because they are really close and when saved into "
                                "COLLISON.txt, positions are rounded and they coincide."
                            ),
                            RuntimeWarning,
                        )
                        cosx, cosy, cosz = cosx0, cosy0, cosz0
                    else:
                        cosx0, cosy0, cosz0 = cosx, cosy, cosz
                    subcollision.update({"cosx": cosx, "cosy": cosy, "cosz": cosz})
                    # Check if SRIM has to be run again
                    ok = criterion(nion=nion + 1, **subcollision)
                    # If not, save into the database
                    if ok:
                        self.collision.insert(cur, nion + 1, **subcollision)
                    # Else, save it for later
                    else:
                        atomic_numbers.append(
                            materials.MATERIALS_BY_SYMBOL[
                                subcollision["atom_hit"]
                            ].atomic_number
                        )
                        recoil_energies.append(subcollision["recoil_energy"])
                        depths.append(subcollision["depth"])
                        ys.append(subcollision["y"])
                        zs.append(subcollision["z"])
                        cosxs.append(subcollision["cosx"])
                        cosys.append(subcollision["cosy"])
                        coszs.append(subcollision["cosz"])
                        nsubcollisions[nion] += 1
                    pos0 = pos
                    # print(pos0, [cosx, cosy, cosz])
        self.subcollision.empty()
        atomic_numbers = np.array(atomic_numbers, dtype=np.int64)
        recoil_energies, depths, ys, zs = (
            np.array(recoil_energies, dtype=np.float64),
            np.array(depths, dtype=np.float64),
            np.array(ys, dtype=np.float64),
            np.array(zs, dtype=np.float64),
        )
        cosxs, cosys, coszs = np.array(cosxs), np.array(cosys), np.array(coszs)
        return (
            nsubcollisions,
            atomic_numbers,
            recoil_energies,
            depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
        )

    def __remove_mean_depth_offsets(
        self, cur: sqlite3.Cursor, table_name: str, depth_mean: tuple[float]
    ) -> None:
        """Removes ion mean depth offsets from the given table.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor.
        table_name : str
            Table name.
        depth_mean : tuple[float]
            Mean depth.
        """
        cur.execute(f"UPDATE {table_name} SET depth = depth - ?", depth_mean)

    def __remove_individual_offsets(self, cur: sqlite3.Cursor, table_name: str) -> None:
        """Removes individual ion offsets from the given table.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor.
        table_name : str
            Table name.
        """
        cur.execute(
            f"""
            UPDATE {table_name}
            SET depth = {table_name}.depth - trimdat.depth,
                y = {table_name}.y - trimdat.y,
                z = {table_name}.z - trimdat.z
            FROM trimdat
            WHERE {table_name}.ion_numb = trimdat.ion_numb
        """
        )

    def __remove_offsets(self) -> None:
        """Removes ion offsets."""
        cur = self.cursor()
        cur.execute("SELECT AVG(depth) FROM trimdat")
        depth_mean = cur.fetchone()
        self.__remove_mean_depth_offsets(cur, "e2recoil", depth_mean)
        self.__remove_mean_depth_offsets(cur, "ioniz", depth_mean)
        self.__remove_mean_depth_offsets(cur, "lateral", depth_mean)
        self.__remove_mean_depth_offsets(cur, "phonon", depth_mean)
        self.__remove_mean_depth_offsets(cur, "range", depth_mean)
        self.__remove_mean_depth_offsets(cur, "vacancy", depth_mean)
        if self.calculation in ["full", "mono"]:
            self.__remove_mean_depth_offsets(cur, "novac", depth_mean)
        self.__remove_individual_offsets(cur, "backscat")
        self.__remove_individual_offsets(cur, "collision")
        self.__remove_individual_offsets(cur, "range3d")
        self.__remove_individual_offsets(cur, "sputter")
        self.__remove_individual_offsets(cur, "transmit")
        self.__remove_individual_offsets(cur, "transmit")
        cur.close()
        self.commit()
