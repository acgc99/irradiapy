"""This module contains the `Py2SRIM` class."""

# pylint: disable=too-many-lines

import os
import platform
import subprocess
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import shutil

import numpy as np
from numpy import typing as npt

from irradiapy import config, dtypes, materials
from irradiapy.debris_database import DebrisDatabase
from irradiapy.materials.component import Component
from irradiapy.materials.element import Element
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.srim.srimdb import SRIMDB

OS_NAME = platform.system()
if OS_NAME == "Windows":
    import pygetwindow
    import pywinauto
elif OS_NAME == "Linux":
    missing_tools = [tool for tool in ("wine", "xdotool") if shutil.which(tool) is None]
    if missing_tools:
        warnings.warn(
            "SRIM through Wine requires the following missing executable(s): "
            + ", ".join(missing_tools)
        )
    if not os.environ.get("DISPLAY"):
        warnings.warn(
            "SRIM automation requires an X11-compatible DISPLAY. "
            "It works in a full X11 session, and may also work in a Wayland "
            "session when SRIM/Wine runs through XWayland."
        )
else:
    warnings.warn(
        "SRIM subpackage only supports Windows or Linux/Wine. "
        f"{OS_NAME!r} not supported."
    )


@dataclass
class Py2SRIM:
    """Base class for running SRIM calculations from Python.

    Attributes
    ----------
    seed : int (default=0)
        Seed for SRIM randomness.
    root_dir: Path
        Root directory where all calculations will be stored.
    target : list[Component]
        List of target components.
    calculation : str
        SRIM calculation.
    srim_dir : Path (default=config.get_srim_dir())
        Directory where SRIM is installed.
    wineprefix : Path (default=Path(os.environ.get("WINEPREFIX", Path.home() / ".wine-srim2013")))
        Wine prefix to use for running SRIM on Linux. By default, it uses the `WINEPREFIX`
        environment variable if set, or `~/.wine-srim2013` otherwise.
    wine_cmd : str (default=os.environ.get("WINE", "wine"))
        Command to run Wine on Linux. By default, it uses the `WINE` environment variable if set,
        or `wine` otherwise.
    recoilsdb : RecoilsDB
        Database to store all recoils collected from SPECTRA-PKA and SRIM calculations.
    check_interval : float (default=0.2)
        Interval to check for SRIM window/popups.
    minimize_window : bool (default=False)
        Whether to minimize the SRIM/TRIM window while calculations run.
    plot_type : int (default=5)
        Plot type during SRIM calculations. 5 for no plots (faster calculations).
    xmin : float (default=0.0)
        Minimum x for plots and depth-dependent means during SRIM calculations. Particularly
        important for large targets, since SRIM divides it in 100 segments.
    xmax : float (default=0.0)
        Maximum x for plots and depth-dependent means during SRIM calculations. Particularly
        important for large targets, since SRIM divides it in 100 segments.
        0.0 for full target.
    do_ranges : int (default=1)
        Whether to save `RANGE.txt` file.
        Disabling this might cause errors afterwards because of missing tables.
    do_backscatt : int (default=1)
        Whether to save `BACKSCAT.txt` file.
        Disabling this might cause errors afterwards because of missing tables.
    do_transmit : int (default=1)
        Whether to save `TRANSMIT.txt` file.
        Disabling this might cause errors afterwards because of missing tables.
    do_sputtered : int (default=1)
        Whether to save `SPUTTER.txt` file.
        Disabling this might cause errors afterwards because of missing tables.
    do_collisions : int (default=1)
        Whether to save `COLLISON.txt` file.
        Disabling this might cause errors afterwards because of missing tables.
    exyz : float (default=0.0)
        Whether to save ions position every time they loose `exyz` energy in the `EXYZ.txt` file.
    autosave : int (default=0)
        Autosave every this number of ions. 0 to disable.
    """

    seed: int = 0

    root_dir: Path = field(init=False)
    target: list[Component] = field(init=False)
    calculation: str = field(init=False)
    srim_dir: Path = field(default_factory=config.get_srim_dir)
    wineprefix: Path = field(
        default_factory=lambda: Path(
            os.environ.get("WINEPREFIX", Path.home() / ".wine-srim2013")
        )
    )
    wine_cmd: str = field(default_factory=lambda: os.environ.get("WINE", "wine"))
    recoilsdb: RecoilsDB = field(init=False)

    check_interval: float = 0.2
    minimize_window: bool = False
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

    # region SRIM files

    def __generate_trimin(
        self,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
    ) -> None:
        """Generates `TRIM.IN` file."""
        nions = len(atomic_numbers)
        atomic_mass = materials.ATOMIC_WEIGHT_BY_ATOMIC_NUMBER[atomic_numbers[0]]
        energy = np.ceil(energies.max()) / 1e3
        ncomponents = len(self.target)
        nelements = sum(len(component.elements) for component in self.target)
        if self.calculation == "quick":
            calculation = 4
        elif self.calculation == "full":
            calculation = 5
        else:
            calculation = 7

        trimin_path = self.srim_dir / "TRIM.IN"
        with open(trimin_path, "w", encoding="ascii", newline="\r\n") as file:
            file.write("TRIM.IN file generated by irradiapy.\n")
            # Ion parameters
            file.write(
                (
                    "ion Z, A, energy, angle, number of ions, Bragg correction, autosave\n"
                    f"{atomic_numbers[0]} {atomic_mass} {energy} 0 "
                    f"{nions} {self.bragg} {self.autosave}\n"
                )
            )
            # Calculation type and seed
            file.write(
                (
                    "calculation, seed, reminders\n"
                    f"{calculation} {self.seed} {self.reminders}\n"
                )
            )
            # Output files
            file.write(
                (
                    "Diskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, "
                    "Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file\n"
                    f"{self.do_ranges} {self.do_backscatt} {self.do_transmit} {self.do_sputtered} "
                    f"{self.do_collisions} {self.exyz}\n"
                )
            )
            # Target material
            file.write(
                (
                    "target material, number of elements, layers\n"
                    f'"irradiapy" {nelements} {ncomponents}\n'
                )
            )
            # Plot settings
            file.write(
                (
                    "PlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]\n"
                    f"{self.plot_type} {self.xmin} {self.xmax}\n"
                )
            )
            # Target elements
            string = "target element, Z, mass\n"
            for i, component in enumerate(self.target):
                for j, element in enumerate(component.elements):
                    string += (
                        f"Atom {i*len(component.elements)+j+1} = {element.symbol} "
                        f"=      {element.atomic_number} {element.atomic_weight}\n"
                    )
            file.write(string)
            # Target layers
            string = "layer name, width density elements\nnumb. desc. (ang) (g/cm3) stoich.\n"
            layers_info = []
            for i, component in enumerate(self.target):
                prev_stoichs = [0.0] * sum(len(l.elements) for l in self.target[:i])
                next_stoichs = [0.0] * sum(
                    len(l.elements) for l in self.target[i + 1 :]
                )
                layer_info = f'{i} "layer{i}" {component.width} {component.density} '
                layer_info += " ".join(
                    map(str, prev_stoichs + list(component.stoichs) + next_stoichs)
                )
                layers_info.append(layer_info)
            file.write(string + "\n".join(layers_info) + "\n")
            # Phases
            string = "phases\n"
            string += " ".join(map(str, (component.phase for component in self.target)))
            file.write(string + "\n")
            # Bragg
            string = "target compound corrections (bragg)\n"
            string += " ".join(
                map(str, (component.srim_bragg for component in self.target))
            )
            file.write(string + "\n")
            # Displacement energies
            string = "displacement energies (eV)\n"
            string += " ".join(
                map(
                    str,
                    (
                        element.ed_avr
                        for component in self.target
                        for element in component.elements
                    ),
                )
            )
            file.write(string + "\n")
            # Lattice energies
            string = "lattice binding energies (eV)\n"
            string += " ".join(
                map(
                    str,
                    (
                        element.srim_el
                        for component in self.target
                        for element in component.elements
                    ),
                )
            )
            file.write(string + "\n")
            # Binding energies
            string = "surface binding energies (eV)\n"
            string += " ".join(
                map(
                    str,
                    (
                        element.srim_es
                        for component in self.target
                        for element in component.elements
                    ),
                )
            )
            file.write(string + "\n")
            # Stopping power version
            file.write("Stopping Power Version\n0\n")

    def __generate_trimdat(
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
        depths : npt.NDArray[np.float64] | None  (default=None)
            Depths.
        ys : npt.NDArray[np.float64] | None (default=None)
            Y positions.
        zs : npt.NDArray[np.float64] | None (default=None)
            Z positions.
        cosxs : npt.NDArray[np.float64] | None (default=None)
            X directions.
        cosys : npt.NDArray[np.float64] | None (default=None)
            Y directions.
        coszs : npt.NDArray[np.float64] | None (default=None)
            Z directions.

        Returns
        -------
        npt.NDArray[np.float64]
            `TRIM.DAT` data.
        """
        trimdat_path = self.srim_dir / "TRIM.DAT"
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
        with open(trimdat_path, "w", encoding="ascii", newline="\r\n") as file:
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
        with open(
            self.srim_dir / "TRIMAUTO", "w", encoding="ascii", newline="\r\n"
        ) as file:
            file.write("1\n\nCheck TRIMAUTO.txt for details.\n")

    # endregion

    # region Checks

    def __check_transmit(self, srimdb: SRIMDB) -> None:
        """Checks if there are transmitted ions in the database."""
        transmit_rows = list(srimdb.read("transmit", "atom_numb, energy"))
        if transmit_rows:
            msg = ", ".join(f"({row[0]}, {row[1]:.2g})" for row in transmit_rows)
            raise RuntimeError(
                "SRIM ions ended up transmitted. Consider increasing the "
                "effective target width. (Z, E (eV)) = " + msg
            )

    def __check_backscat(self, srimdb: SRIMDB) -> None:
        """Checks if there are backscattered ions in the database."""
        backscat_rows = list(srimdb.read("backscat", "atom_numb, energy"))
        if backscat_rows:
            msg = ", ".join(f"({row[0]}, {row[1]:.2g})" for row in backscat_rows)
            raise RuntimeError(
                "SRIM ions ended up backscattered. Consider increasing the "
                "effective target width. (Z, E (eV)) = " + msg
            )

    # endregion

    # region Run

    @staticmethod
    def __should_run_srim_for_recoil(
        recoil: Element,
        component: Component,
        recoil_energy: float,
        max_recoil_energy: float,
        invalid_recoil_energy: float,
        debris_database: DebrisDatabase,
    ) -> bool:
        """Return whether a recoil should be sent to SRIM."""
        # Any recoil above max_recoil_energy is sent to SRIM.
        if recoil_energy > max_recoil_energy:
            return True
        matches = debris_database.has_matches(
            recoil=recoil,
            component=component,
        )
        # If it matches the database, no need to run SRIM.
        # If the recoil matches the database but its energy is lower than
        # the lowest one available in the database, matches is still true,
        # FP will be used.
        if matches:
            return False
        # Any unmatched recoil below invalid_recoil_energy is not sent to SRIM,
        # FPs will be placed instead.
        if recoil_energy < invalid_recoil_energy:
            return False
        # Send to SRIM unmatched recoils between invalid_recoil_energy and max_recoil_energy.
        return True

    def __create_srimdb(
        self,
        path: Path,
        target: list[Component],
        calculation: str,
    ) -> SRIMDB:
        """Creates a SRIMDB instance."""
        srimdb = SRIMDB(
            path=path,
            target=target,
            calculation=calculation,
            seed=self.seed,
            plot_type=self.plot_type,
            xmin=self.xmin,
            xmax=self.xmax,
            do_ranges=self.do_ranges,
            do_backscatt=self.do_backscatt,
            do_transmit=self.do_transmit,
            do_sputtered=self.do_sputtered,
            do_collisions=self.do_collisions,
            exyz=self.exyz,
            autosave=self.autosave,
        )
        return srimdb

    def __srim_env(self) -> dict[str, str]:
        """Return the environment used to run SRIM/TRIM through Wine."""
        env = os.environ.copy()
        if OS_NAME == "Linux":
            env["WINEPREFIX"] = str(
                Path(env.get("WINEPREFIX", Path.home() / ".wine-srim2013"))
            )
            env.pop("WINEARCH", None)
            # Avoid comma decimal formatting leaking into Wine/SRIM.
            env.setdefault("LC_NUMERIC", "C")
        return env

    def __trim_command(self) -> list[str]:
        """Return the platform-specific TRIM launch command."""
        if OS_NAME == "Windows":
            return [str(self.srim_dir / "TRIM.exe")]
        if OS_NAME == "Linux":
            wine = os.environ.get("WINE", "wine")
            return [wine, "TRIM.exe"]
        raise RuntimeError(f"Unsupported operating system for SRIM: {OS_NAME!r}")

    def __xdotool_search(self, title: str) -> str | None:
        """Return the first X11 window id matching a title, or None."""
        result = subprocess.run(
            ["xdotool", "search", "--name", title],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            env=self.__srim_env(),
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            window_id = line.strip()
            if window_id:
                return window_id
        return None

    def __wineserver_wait(self) -> None:
        """Wait until Wine has no pending work for this prefix."""
        if OS_NAME != "Linux":
            return
        wineserver = shutil.which("wineserver")
        if wineserver is None:
            return
        subprocess.run(
            [wineserver, "-w"],
            env=self.__srim_env(),
            check=False,
        )

    def __minimize_and_handle_popup(
        self,
        stop_event: threading.Event,
        minimize_window: bool,
    ) -> None:
        """Optionally minimize SRIM/TRIM and dismiss the final TRIM popup."""
        # pylint: disable=protected-access
        window_title = "SRIM-2013.00"
        popup_title = "End of TRIM.DAT calculation"

        if OS_NAME == "Windows":
            if minimize_window:
                while not stop_event.is_set():
                    windows = pygetwindow.getWindowsWithTitle(window_title)
                    if windows:
                        window = windows[0]
                        app = pywinauto.Application().connect(handle=window._hWnd)
                        app.window(handle=window._hWnd).minimize()
                        break
                    time.sleep(self.check_interval)
            if self.calculation != "quick":
                while not stop_event.is_set():
                    popups = pygetwindow.getWindowsWithTitle(popup_title)
                    if popups:
                        popup = popups[0]
                        app = pywinauto.Application().connect(handle=popup._hWnd)
                        app.window(handle=popup._hWnd).send_keystrokes("{ENTER}")
                        break
                    time.sleep(self.check_interval)
            return

        if OS_NAME == "Linux":
            if minimize_window:
                while not stop_event.is_set():
                    window_id = self.__xdotool_search(window_title)
                    if window_id is not None:
                        subprocess.run(
                            ["xdotool", "windowminimize", window_id],
                            check=False,
                            env=self.__srim_env(),
                        )
                        break
                    time.sleep(self.check_interval)
            if self.calculation != "quick":
                while not stop_event.is_set():
                    popup_id = self.__xdotool_search(popup_title)
                    if popup_id is not None:
                        subprocess.run(
                            ["xdotool", "windowactivate", popup_id],
                            check=False,
                            env=self.__srim_env(),
                        )
                        subprocess.run(
                            ["xdotool", "key", "Return"],
                            check=False,
                            env=self.__srim_env(),
                        )
                        break
                    time.sleep(self.check_interval)
            return

        raise RuntimeError(
            f"Unsupported operating system for SRIM automation: {OS_NAME!r}"
        )

    def __component_from_depth(self, depth: float) -> Component:
        """Return the target component at given depth."""
        current_depth = 0.0
        for component in self.target:
            if component.width is None:
                raise ValueError("All target components must have a width.")
            next_depth = current_depth + component.width
            if current_depth <= depth <= next_depth:
                return component
            current_depth = next_depth
        raise ValueError(f"depth={depth} is out of bounds of the target.")

    def __srim_mask(
        self,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64],
        max_recoil_energy: float,
        invalid_recoil_energy: float,
        debris_database: DebrisDatabase,
    ) -> npt.NDArray[np.bool_]:
        """Return the mask of recoils that should be sent to SRIM."""
        mask = np.zeros(atomic_numbers.shape, dtype=bool)
        for i, atomic_number in enumerate(atomic_numbers):
            recoil = materials.ELEMENT_BY_ATOMIC_NUMBER[int(atomic_number)]
            component = self.__component_from_depth(depths[i])
            mask[i] = self.__should_run_srim_for_recoil(
                recoil=recoil,
                component=component,
                recoil_energy=energies[i],
                max_recoil_energy=max_recoil_energy,
                invalid_recoil_energy=invalid_recoil_energy,
                debris_database=debris_database,
            )
        return mask

    @staticmethod
    def __group_ions_by_atomic_number(
        mask: npt.NDArray[np.bool_],
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        **extra_fields: npt.NDArray[np.float64],
    ) -> dict[int, dict[str, npt.NDArray[np.float64]]]:
        """Group selected ions by atomic number.

        Parameters
        ----------
        mask : npt.NDArray[np.bool_]
            Boolean mask indicating which ions to include.
        atomic_numbers : npt.NDArray[np.int64]
            Atomic number of each ion.
        energies : npt.NDArray[np.float64]
            Energy (eV) of each ion. Used for thresholding.
        **extra_fields : npt.NDArray[np.float64]
            Any extra per-ion arrays to propagate (depths, ys, zs, cosxs, ...).

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            A dictionary keyed by atomic number. Each value is another dict with:
                'atomic_numbers' : np.ndarray
                'energies'       : np.ndarray
                '<field name>'   : np.ndarray  # for each entry in extra_fields
                'nions'          : int
        """
        ions: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        unique_atomic_numbers = np.unique(atomic_numbers[mask])

        for atomic_number in unique_atomic_numbers:
            atomic_mask = (atomic_numbers == atomic_number) & mask
            batch: dict[str, npt.NDArray[np.float64]] = {
                "atomic_numbers": atomic_numbers[atomic_mask],
                "energies": energies[atomic_mask],
            }
            for name, arr in extra_fields.items():
                batch[name] = arr[atomic_mask]
            batch["nions"] = batch["atomic_numbers"].size
            ions[int(atomic_number)] = batch

        return ions

    def __tree2path_db(
        self,
        tree: tuple[int, ...],
        create_parent_dir: bool,
    ) -> Path:
        """Converts a tree tuple into a database path.

        Parameters
        ----------
        tree : tuple[int, ...]
            Tree tuple.
        create_parent_dir : bool
            Whether to create the parent directory.

        Returns
        -------
        Path
            Database path.
        """
        branch_dir = self.root_dir.joinpath(
            *(str(atomic_number) for atomic_number in tree)
        )
        if create_parent_dir:
            branch_dir.mkdir(parents=True, exist_ok=True)
        path = branch_dir / "srim.db"
        return path

    def __run(
        self,
        srimdb: SRIMDB,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64] | None,
        ys: npt.NDArray[np.float64] | None,
        zs: npt.NDArray[np.float64] | None,
        cosxs: npt.NDArray[np.float64] | None,
        cosys: npt.NDArray[np.float64] | None,
        coszs: npt.NDArray[np.float64] | None,
        fail_on_backscat: bool,
        fail_on_transmit: bool,
        minimize_window: bool,
        ignore_32bit_warning: bool = True,
    ) -> None:
        """Run the SRIM simulation.

        Parameters
        ----------
        srimdb : SRIMDB
            SRIM database.
        atomic_numbers : npt.NDArray[np.int64]
            Ion atomic numbers.
        energies : npt.NDArray[np.float64]
            Ion energies.
        depths : npt.NDArray[np.float64] | None
            Ion initial depths.
        ys : npt.NDArray[np.float64] | None
            Ion initial y positions.
        zs : npt.NDArray[np.float64] | None
            Ion initial z positions.
        cosxs : npt.NDArray[np.float64] | None
            Ion initial x directions.
        cosys : npt.NDArray[np.float64] | None
            Ion initial y directions.
        coszs : npt.NDArray[np.float64] | None
            Ion initial z directions.
        fail_on_backscat : bool
            Whether to fail if there are backscattered ions.
        fail_on_transmit : bool
            Whether to fail if there are transmitted ions.
        minimize_window : bool
            Whether to minimize the SRIM window while SRIM simulations run.
        """
        if ignore_32bit_warning:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="32-bit application should be automated using 32-bit Python",
            )

        if srimdb.table_exists("trimdat"):
            raise RuntimeError(
                (
                    f"The database {srimdb.path} is already populated "
                    "with data from another simulation, use another one."
                )
            )

        # Generate input files
        self.__generate_trimauto()
        self.__generate_trimin(atomic_numbers, energies)
        self.__generate_trimdat(
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
        stop_event = threading.Event()
        window_thread = threading.Thread(
            target=self.__minimize_and_handle_popup,
            args=(stop_event, minimize_window),
            daemon=True,
        )
        try:
            window_thread.start()
            subprocess.run(
                self.__trim_command(),
                cwd=self.srim_dir,
                env=self.__srim_env(),
                check=True,
            )
            self.__wineserver_wait()
        finally:
            stop_event.set()
            window_thread.join(timeout=5.0)

        # Append output files
        srimdb.append_output()
        srimdb.commit()
        if fail_on_transmit:
            self.__check_transmit(srimdb)
        if fail_on_backscat:
            self.__check_backscat(srimdb)
        srimdb.optimize()

    def __collect_recoils(
        self,
        max_recoil_energy: float,
        invalid_recoil_energy: float,
        debris_database: DebrisDatabase,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        zs: npt.NDArray[np.float64],
        cosxs: npt.NDArray[np.float64],
        cosys: npt.NDArray[np.float64],
        coszs: npt.NDArray[np.float64],
    ) -> None:
        """Collect all recoils from all SRIM databases into a single database taking into account
        their hierarchy.

        Parameters
        ----------
        max_recoil_energy : float
            Recoils above this energies will be sent to SRIM, in eV.
        invalid_recoil_energy : float
            Unmatched recoils at or above this energy will be sent to SRIM, in eV.
        atomic_numbers : npt.NDArray[np.int64]
            Ion atomic numbers.
        energies : npt.NDArray[np.float64]
            Ion energies.
        depths : npt.NDArray[np.float64]
            Ion initial depths.
        ys : npt.NDArray[np.float64]
            Ion initial y positions.
        zs : npt.NDArray[np.float64]
            Ion initial z positions.
        cosxs : npt.NDArray[np.float64]
            Ion initial x directions.
        cosys : npt.NDArray[np.float64]
            Ion initial y directions.
        coszs : npt.NDArray[np.float64]
            Ion initial z directions.
        """
        # For each SRIM database, keep track of how many ions we have already consumed.
        # Key is the full path to the srim.db file.
        srim_ion_counters: dict[Path, int] = defaultdict(int)

        def collect_recoils_from_srim(
            tree: tuple[int, ...],
            ion_numb: int,
            event: int,
        ) -> None:
            """Collect recoils from a single SRIM ion and recurse on high-energy recoils.

            Parameters
            ----------
            tree : tuple[int, ...]
                Sequence of atomic numbers describing the SRIM path, e.g. (26,), (26, 76), ...
                The corresponding SRIM DB is at:
                    root_dir / '26' / '76' / ... / 'srim.db'
            ion_numb : int
                Ion number inside this SRIM DB (ion_numb column).
            event : int
                Event index, 1-based. Here it is the original ion index + 1 from Py2SRIM.run.
            """
            # Path to current SRIM DB for this tree
            branch_path = self.__tree2path_db(tree, create_parent_dir=False)

            # If this SRIM DB does not exist (e.g., max_srim_iters prevented creation), stop here.
            if not branch_path.exists():
                return

            srimdb_branch = self.__create_srimdb(
                path=branch_path,
                target=None,
                calculation=None,
            )
            collisions = list(
                srimdb_branch.read(
                    table="collision",
                    what="depth, y, z, cosx, cosy, cosz, recoil_energy, atom_hit",
                    conditions=f"WHERE ion_numb={ion_numb}",
                )
            )
            srimdb_branch.close()

            for (
                depth,
                y,
                z,
                cosx,
                cosy,
                cosz,
                recoil_energy,
                atom_hit,
            ) in collisions:
                atomic_number = int(materials.ATOMIC_NUMBER_BY_SYMBOL[atom_hit])

                component = self.__component_from_depth(float(depth))
                should_run = self.__should_run_srim_for_recoil(
                    recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atomic_number],
                    component=component,
                    recoil_energy=float(recoil_energy),
                    max_recoil_energy=max_recoil_energy,
                    invalid_recoil_energy=invalid_recoil_energy,
                    debris_database=debris_database,
                )
                if not should_run:
                    # This recoil is terminal; store it.
                    self.recoilsdb.insert_recoil(
                        event=event,
                        atomic_number=atomic_number,
                        recoil_energy=recoil_energy,
                        x=depth,
                        y=y,
                        z=z,
                        cosx=cosx,
                        cosy=cosy,
                        cosz=cosz,
                    )
                    continue

                # Above threshold: try to follow to the next SRIM level.
                leaf_tree = (*tree, atomic_number)
                leaf_db_path = self.__tree2path_db(
                    leaf_tree,
                    create_parent_dir=False,
                )

                # If the child SRIM DB does not exist (e.g., limited by max_srim_iters),
                # treat as terminal.
                if not leaf_db_path.exists():
                    self.recoilsdb.insert_recoil(
                        event=event,
                        atomic_number=atomic_number,
                        recoil_energy=recoil_energy,
                        x=depth,
                        y=y,
                        z=z,
                        cosx=cosx,
                        cosy=cosy,
                        cosz=cosz,
                    )
                    continue

                # Map this collision to the correct ion in the child SRIM DB.
                child_ion_numb = srim_ion_counters[leaf_db_path] + 1
                srim_ion_counters[leaf_db_path] = child_ion_numb
                collect_recoils_from_srim(
                    tree=leaf_tree,
                    ion_numb=child_ion_numb,
                    event=event,
                )

        # Start from the primary ions given to Py2SRIM.run
        nions = atomic_numbers.size

        for i in range(nions):
            event = int(i + 1)
            ion_atomic_number = int(atomic_numbers[i])
            ion_energy = float(energies[i])
            depth = float(depths[i])
            y = float(ys[i])
            z = float(zs[i])
            cosx = float(cosxs[i])
            cosy = float(cosys[i])
            cosz = float(coszs[i])

            component = self.__component_from_depth(depth)
            should_run = self.__should_run_srim_for_recoil(
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[ion_atomic_number],
                component=component,
                recoil_energy=ion_energy,
                max_recoil_energy=max_recoil_energy,
                invalid_recoil_energy=invalid_recoil_energy,
                debris_database=debris_database,
            )

            # Terminal primary ion -> store into final DB as terminal recoil.
            if not should_run:
                self.recoilsdb.insert_recoil(
                    event=event,
                    atomic_number=ion_atomic_number,
                    recoil_energy=ion_energy,
                    x=depth,
                    y=y,
                    z=z,
                    cosx=cosx,
                    cosy=cosy,
                    cosz=cosz,
                )
                continue

            # High-energy primary ion: follow SRIM cascades starting at atomic_number/srim.db.
            branch_path = self.__tree2path_db(
                (ion_atomic_number,),
                create_parent_dir=False,
            )

            # No SRIM data for this ion: treat as terminal.
            # Likely due to max_srim_iters or an aborted run.
            if not branch_path.exists():
                self.recoilsdb.insert_recoil(
                    event=event,
                    atomic_number=ion_atomic_number,
                    recoil_energy=ion_energy,
                    x=depth,
                    y=y,
                    z=z,
                    cosx=cosx,
                    cosy=cosy,
                    cosz=cosz,
                )
                continue

            # Ion index in this first-level SRIM DB.
            ion_numb = srim_ion_counters[branch_path] + 1
            srim_ion_counters[branch_path] = ion_numb

            collect_recoils_from_srim(
                tree=(ion_atomic_number,),
                ion_numb=ion_numb,
                event=event,
            )

    def __collect_ions_vacs(
        self,
        max_recoil_energy: float,
        invalid_recoil_energy: float,
        debris_database: DebrisDatabase,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64],
    ) -> None:
        """Collect all ions and vacancies from all SRIM databases into a single database
        taking into account their hierarchy.
        """
        # For each SRIM database, keep track of how many ions we have already consumed.
        # Key is the full path to the srim.db file.
        srim_ion_counters: dict[Path, int] = defaultdict(int)

        def collect_ions_vacs_from_srim(
            tree: tuple[int, ...],
            ion_numb: int,
            event: int,
        ) -> None:
            """Collect ions and vacancies from a single SRIM ion and recurse on high-energy
            recoils.

            Parameters
            ----------
            tree : tuple[int, ...]
                Sequence of atomic numbers describing the SRIM path, e.g. (26,), (26, 76), ...
                The corresponding SRIM DB is at:
                    root_dir / '26' / '76' / ... / 'srim.db'
            ion_numb : int
                Ion number inside this SRIM DB (ion_numb column).
            event : int
                Event index, 1-based (ion index from Py2SRIM.run).
            """
            # Path to current SRIM DB for this tree
            branch_path = self.__tree2path_db(tree, create_parent_dir=False)

            # If this SRIM DB does not exist (e.g., max_srim_iters prevented creation), stop.
            if not branch_path.exists():
                return

            srimdb_branch = self.__create_srimdb(
                path=branch_path,
                target=None,
                calculation=None,
            )

            # Initial ion position: potential vacancy at the initial position
            # Indicate with negative atomic number to distinguish from ion
            trimdat = list(
                srimdb_branch.read(
                    table="trimdat",
                    what="depth, y, z, atom_numb",
                    conditions=f"WHERE ion_numb={ion_numb}",
                )
            )
            self.recoilsdb.insert_ion_vac(
                event=event,
                atom_numb=-int(trimdat[0][3]),
                x=trimdat[0][0],
                y=trimdat[0][1],
                z=trimdat[0][2],
            )
            # Final ion position is saved if not transmitted/backscattered
            range3d = list(
                srimdb_branch.read(
                    table="range3d",
                    what="depth, y, z",
                    conditions=f"WHERE ion_numb={ion_numb}",
                )
            )
            if range3d:
                # If not empty, ion stopped inside target
                self.recoilsdb.insert_ion_vac(
                    event=event,
                    atom_numb=int(trimdat[0][3]),
                    x=range3d[0][0],
                    y=range3d[0][1],
                    z=range3d[0][2],
                )

            # Recoils for this ion_numb to decide further SRIM levels
            collision_rows = list(
                srimdb_branch.read(
                    table="collision",
                    what="depth, recoil_energy, atom_hit",
                    conditions=f"WHERE ion_numb={ion_numb}",
                )
            )
            srimdb_branch.close()

            # Recurse on high-energy recoils
            for depth, recoil_energy, atom_hit in collision_rows:
                recoil_energy = float(recoil_energy)
                atomic_number = int(materials.ATOMIC_NUMBER_BY_SYMBOL[atom_hit])
                component = self.__component_from_depth(float(depth))
                should_run = self.__should_run_srim_for_recoil(
                    recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atomic_number],
                    component=component,
                    recoil_energy=recoil_energy,
                    max_recoil_energy=max_recoil_energy,
                    invalid_recoil_energy=invalid_recoil_energy,
                    debris_database=debris_database,
                )

                if not should_run:
                    # Terminal for ions/vacs purposes
                    continue

                leaf_tree = (*tree, atomic_number)
                leaf_db_path = self.__tree2path_db(
                    leaf_tree,
                    create_parent_dir=False,
                )

                # If the child SRIM DB does not exist (e.g., limited by max_srim_iters),
                # treat as terminal.
                if not leaf_db_path.exists():
                    continue

                # Map this recoil to the correct ion in the child SRIM DB.
                child_ion_numb = srim_ion_counters[leaf_db_path] + 1
                srim_ion_counters[leaf_db_path] = child_ion_numb
                collect_ions_vacs_from_srim(
                    tree=leaf_tree,
                    ion_numb=child_ion_numb,
                    event=event,
                )

        # Start from the primary ions given to Py2SRIM.run
        nions = atomic_numbers.size

        for i in range(nions):
            event = int(i + 1)
            ion_atomic_number = int(atomic_numbers[i])
            ion_energy = float(energies[i])
            component = self.__component_from_depth(depths[i])
            should_run = self.__should_run_srim_for_recoil(
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[ion_atomic_number],
                component=component,
                recoil_energy=ion_energy,
                max_recoil_energy=max_recoil_energy,
                invalid_recoil_energy=invalid_recoil_energy,
                debris_database=debris_database,
            )

            # Terminal primary ions never enter a SRIM branch, so no ions/vacs are collected.
            if not should_run:
                continue

            # High-energy primary ion: follow SRIM cascades starting at Z/srim.db.
            branch_path = self.__tree2path_db(
                tree=(ion_atomic_number,),
                create_parent_dir=False,
            )

            # No SRIM data for this ion: treat as terminal.
            # Likely due to max_srim_iters or aborted runs.
            if not branch_path.exists():
                continue

            # Ion index in this first-level SRIM DB.
            ion_numb = srim_ion_counters[branch_path] + 1
            srim_ion_counters[branch_path] = ion_numb

            collect_ions_vacs_from_srim(
                tree=(ion_atomic_number,),
                ion_numb=ion_numb,
                event=event,
            )

    def run(
        self,
        root_dir: Path,
        target: list[Component],
        calculation: str,
        atomic_numbers: npt.NDArray[np.int64],
        energies: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        zs: npt.NDArray[np.float64],
        cosxs: npt.NDArray[np.float64],
        cosys: npt.NDArray[np.float64],
        coszs: npt.NDArray[np.float64],
        max_recoil_energy: float,
        max_srim_iters: int,
        fail_on_transmit: bool,
        fail_on_backscatt: bool,
        invalid_recoil_energy: float = 1e3,
        ignore_32bit_warning: bool = True,
        minimize_window: bool | None = None,
    ) -> RecoilsDB:
        """Run SRIM iteratively, creating a folder tree driven by a recoil-energy threshold.

        - group ions by atomic number, keeping only those with energy > ``max_recoil_energy``
        - create a directory tree under ``root_dir`` of the form:
              root_dir / atomic_number_1 / [atomic_number_2 / atomic_number_3 / ...]
          where each branch holds a ``srim.db`` file
        - run SRIM once per branch
        - recursively spawn new SRIM runs for recoils from the ``collision`` table whose
          energy is above ``max_recoil_energy``
        - stop at depth ``max_srim_iters`` (depth is len(tree), e.g. (26, 76, 26) -> 3)

        Parameters
        ----------
        root_dir: Path
            Root directory where all calculations will be stored.
        target : list[Component]
            SRIM target.
        calculation : str
            SRIM calculation.
        atomic_numbers : npt.NDArray[np.int64]
            Ion atomic numbers.
        energies : npt.NDArray[np.float64]
            Ion energies.
        depths : npt.NDArray[np.float64]
            Ion initial depths.
        ys : npt.NDArray[np.float64]
            Ion initial y positions.
        zs : npt.NDArray[np.float64]
            Ion initial z positions.
        cosxs : npt.NDArray[np.float64]
            Ion initial x directions.
        cosys : npt.NDArray[np.float64]
            Ion initial y directions.
        coszs : npt.NDArray[np.float64]
            Ion initial z directions.
        max_recoil_energy : float
            Recoils above this energy (eV) will spawn further SRIM branches.
        max_srim_iters : int
            Maximum number of SRIM iterations.
        fail_on_transmit : bool
            If True, raise if any ion is transmitted (TRANSMIT.txt non-empty).
        fail_on_backscatt : bool
            If True, raise if any ion is backscattered (BACKSCAT.txt non-empty).
        invalid_recoil_energy : float, optional (default=1e3)
            Unmatched recoils below this energy are terminal and become FP-only debris.
        ignore_32bit_warning : bool (default=True)
            Whether to ignore the 32-bit warning when using 32-bit SRIM with 64-bit Python.
        minimize_window : bool | None (default=None)
            Whether to minimize the SRIM/TRIM window while calculations run. If None, use
            the instance-level ``minimize_window`` setting. The default instance setting is False.

        Returns
        -------
        RecoilsDB
            Database with all recoils collected.
        """
        self.root_dir = root_dir
        self.target = target
        self.calculation = calculation
        if minimize_window is None:
            minimize_window = self.minimize_window
        if not isinstance(minimize_window, bool):
            raise TypeError("minimize_window must be a bool")
        self.minimize_window = minimize_window
        if max_srim_iters < 1:
            raise ValueError("max_srim_iters must be at least 1")
        if self.calculation not in {"quick", "full", "mono"}:
            raise ValueError("calculation must be 'quick', 'full' or 'mono'")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.recoilsdb = RecoilsDB(self.root_dir / "recoils.db")
        debris_database = config.get_debris_database()

        def run_branch(
            tree: tuple[int, ...],
            batch: dict[str, npt.NDArray[np.float64]],
        ) -> None:
            """Run SRIM for one branch in the tree and write root_dir/.../srim.db."""
            path = self.__tree2path_db(tree, True)
            srimdb = self.__create_srimdb(
                path=path,
                target=self.target,
                calculation=self.calculation,
            )
            self.__run(
                srimdb=srimdb,
                atomic_numbers=batch["atomic_numbers"],
                energies=batch["energies"],
                depths=batch["depths"],
                ys=batch["ys"],
                zs=batch["zs"],
                cosxs=batch["cosxs"],
                cosys=batch["cosys"],
                coszs=batch["coszs"],
                fail_on_backscat=fail_on_backscatt,
                fail_on_transmit=fail_on_transmit,
                minimize_window=minimize_window,
                ignore_32bit_warning=ignore_32bit_warning,
            )
            srimdb.close()

        def recurse(tree: tuple[int, ...]) -> None:
            """Process one srim.db and recursively spawn branches."""
            path = self.__tree2path_db(tree, False)
            # Database not existing: no recoils to process
            if not path.exists():
                return

            # Collect recoils from this branch for the next iteration
            srimdb_branch = self.__create_srimdb(
                path=path,
                target=None,
                calculation=None,
            )
            collisions = list(
                srimdb_branch.read(
                    table="collision",
                    what="depth, y, z, cosx, cosy, cosz, recoil_energy, atom_hit",
                    conditions="ORDER BY ion_numb",
                )
            )
            srimdb_branch.close()
            if not collisions:
                return

            branch_depths = np.array([row[0] for row in collisions], dtype=np.float64)
            branch_ys = np.array([row[1] for row in collisions], dtype=np.float64)
            branch_zs = np.array([row[2] for row in collisions], dtype=np.float64)
            branch_cosxs = np.array([row[3] for row in collisions], dtype=np.float64)
            branch_cosys = np.array([row[4] for row in collisions], dtype=np.float64)
            branch_coszs = np.array([row[5] for row in collisions], dtype=np.float64)
            branch_energies = np.array([row[6] for row in collisions], dtype=np.float64)
            branch_atomic_numbers = np.array(
                [materials.ATOMIC_NUMBER_BY_SYMBOL[row[7]] for row in collisions],
                dtype=np.int32,
            )
            srim_mask = self.__srim_mask(
                atomic_numbers=branch_atomic_numbers,
                energies=branch_energies,
                depths=branch_depths,
                max_recoil_energy=max_recoil_energy,
                invalid_recoil_energy=invalid_recoil_energy,
                debris_database=debris_database,
            )
            leaf_batches = self.__group_ions_by_atomic_number(
                mask=srim_mask,
                atomic_numbers=branch_atomic_numbers,
                energies=branch_energies,
                depths=branch_depths,
                ys=branch_ys,
                zs=branch_zs,
                cosxs=branch_cosxs,
                cosys=branch_cosys,
                coszs=branch_coszs,
            )
            # No recoils above threshold
            if not leaf_batches:
                return

            for atomic_number, batch in leaf_batches.items():
                leaf_tree = (*tree, int(atomic_number))
                depth_iter = len(leaf_tree)
                if depth_iter >= max_srim_iters:
                    tree_str = "/".join(str(z) for z in leaf_tree)
                    print(
                        f"Skipping SRIM for Z={atomic_number} (path {tree_str}): "
                        f"reached max_srim_iters={max_srim_iters}"
                    )
                    continue

                run_branch(leaf_tree, batch)
                recurse(leaf_tree)

        srim_mask = self.__srim_mask(
            atomic_numbers=atomic_numbers,
            energies=energies,
            depths=depths,
            max_recoil_energy=max_recoil_energy,
            invalid_recoil_energy=invalid_recoil_energy,
            debris_database=debris_database,
        )
        batches = self.__group_ions_by_atomic_number(
            mask=srim_mask,
            atomic_numbers=atomic_numbers,
            energies=energies,
            depths=depths,
            ys=ys,
            zs=zs,
            cosxs=cosxs,
            cosys=cosys,
            coszs=coszs,
        )
        if not batches:
            print("No ions above the recoil energy threshold; nothing to run.")
        else:
            for atomic_number, batch in batches.items():
                tree = (int(atomic_number),)
                run_branch(tree, batch)
                if max_srim_iters > 1:
                    recurse(tree)

        # Collect all recoils data into a single database
        self.__collect_recoils(
            max_recoil_energy=max_recoil_energy,
            invalid_recoil_energy=invalid_recoil_energy,
            debris_database=debris_database,
            atomic_numbers=atomic_numbers,
            energies=energies,
            depths=depths,
            ys=ys,
            zs=zs,
            cosxs=cosxs,
            cosys=cosys,
            coszs=coszs,
        )
        # Collect all ions and vacancies into a single database
        self.__collect_ions_vacs(
            max_recoil_energy=max_recoil_energy,
            invalid_recoil_energy=invalid_recoil_energy,
            debris_database=debris_database,
            atomic_numbers=atomic_numbers,
            energies=energies,
            depths=depths,
        )
        self.recoilsdb.save_target(self.target)
        self.recoilsdb.commit()

        return self.recoilsdb

    # endregion
