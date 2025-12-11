"""SPECTRA-PKA subpackage core module."""

# pylint: disable=too-many-lines

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irradiapy import materials, srim
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.srim.target import Target


@dataclass
class Spectra2SRIM:
    """Spectra2SRIM class to run SPECTRA-PKA to SRIM workflow.

    Attributes
    ----------
    seed : int (default=0)
        Seed for SRIM randomness.
    path_spectrapka_in : Path
        SPECTRA-PKA input file path, for target definition.
    path_spectrapka_events : Path
        SPECTRA-PKA config_events.pka file path, for recoils data.
    dir_root : Path
        Root output directory where all data will be stored.
    srim_width : float, optional (default=1e8)
        The SPECTRA-PKA box might be small for SRIM ions. To avoid backscattering and
        transmission, SPECTRA-PKA recoils are injected at the middle of a thick SRIM target
        of this width (in Angstrom). In postprocessing, the depth offset is corrected to get
        the position in the SPECTRA-PKA box. Note that due to SRIM limitations, recoils
        positions are rounded to a number of the form "xxxxx.E+xx",
        if this width is set too high, precision might be lost, but
        backscattering/transmission might happen if set too low.
        Unfortunately, there is not a strict rule to set this value.
    matdict : dict[str, Any]
        Material dictionary with SPECTRA-PKA material information.
    target : srim.target.Target
        SRIM target.
    recoils_db : RecoilsDB
        Database to store all recoils collected from SPECTRA-PKA and SRIM calculations.
    check_interval : float (default=0.2)
        Interval to check for SRIM window/popups.
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

    path_spectrapka_in: Path = field(init=False)
    path_spectrapka_events: Path = field(init=False)
    dir_root: Path = field(init=False)
    srim_width: float = field(default=1e8, init=False)
    matdict: dict[str, Any] = field(init=False)
    target: Target = field(init=False)
    recoils_db: RecoilsDB = field(init=False)

    check_interval: float = 0.2
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

    def __read_spectrapka_events_for_srim(self, condition: str = "") -> tuple:
        """Read data from the SPECTRA-PKA database for SRIM calculations.

        Parameters
        ----------
        condition : str
            Condition to filter data.

        Returns
        -------
        tuple
            nions, atomic_numbers, recoil_energies, depths, ys, zs, cosxs, cosys, coszs, times,
            events
        """
        data = list(
            self.recoils_db.read(
                table="events",
                what="x, y, z, vx, vy, vz, element, recoil_energy, time, event",
                condition=condition,
            )
        )
        nions = len(data)

        depths = np.array([row[0] for row in data], dtype=np.float64)
        ys = np.array([row[1] for row in data], dtype=np.float64)
        zs = np.array([row[2] for row in data], dtype=np.float64)

        vx = np.array([row[3] for row in data], dtype=np.float64)
        vy = np.array([row[4] for row in data], dtype=np.float64)
        vz = np.array([row[5] for row in data], dtype=np.float64)
        v = np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
        cosxs = vx / v
        cosys = vy / v
        coszs = vz / v

        atomic_numbers = np.array(
            [materials.ATOMIC_NUMBER_BY_SYMBOL[row[6]] for row in data], dtype=np.int32
        )
        recoil_energies = 1e6 * np.array(
            [row[7] for row in data], dtype=np.float64
        )  # MeV to eV
        times = np.array([row[8] for row in data], dtype=np.float64)
        events = np.array([row[9] for row in data], dtype=np.int32)

        return (
            nions,
            atomic_numbers,
            recoil_energies,
            depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
            times,
            events,
        )

    def __spectrapka_in_to_srim_target(self) -> None:
        """Process the SPECTRA-PKA input file to generate SRIM target.

        Notes
        -----
            Isotopes with the same element symbol are combined into a single element with the
            sum of their stoichiometries.
        """
        # Store material data
        self.matdict = {}
        cols = []
        reading_columns = False
        with open(self.path_spectrapka_in, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip().startswith("columns="):
                    reading_columns = True
                    continue
                if reading_columns and line.startswith('"'):
                    # remove quotes and split by space
                    cols.append(line.strip().strip('"').split())
                if reading_columns and not line.startswith('"'):
                    reading_columns = False
                    break
        self.matdict["stoichs"] = np.array([col[1] for col in cols], dtype=np.float64)
        self.matdict["symbols"] = np.array([col[2] for col in cols], dtype=str)
        # SRIM does not distinguish isotopes, merge them
        unique_symbols = np.unique(self.matdict["symbols"])
        unique_stoichs = [
            float(np.sum(self.matdict["stoichs"][self.matdict["symbols"] == el]))
            for el in unique_symbols
        ]
        self.matdict["stoichs"] = unique_stoichs
        self.matdict["symbols"] = unique_symbols.tolist()
        # Get lattice definition
        a0, lattice, nsize = None, None, None
        with open(self.path_spectrapka_in, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip().startswith("latt="):
                    a0 = float(line.split("=")[1].strip())
                elif line.strip().startswith("box_type="):
                    lattice = line.split("=")[1].strip()
                elif line.strip().startswith("box_nunits="):
                    nsize = int(line.split("=")[1].strip())
        self.matdict["a0"] = a0
        if lattice == "1":
            self.matdict["lattice"] = "bcc"
        elif lattice == "2":
            self.matdict["lattice"] = "fcc"
        elif lattice == "3":
            self.matdict["lattice"] = "hcp"
        else:
            raise ValueError(f"Unknown SPECTRA-PKA lattice type: {lattice}")
        self.matdict["nsize"] = nsize
        self.matdict["sizex"] = a0 * nsize
        self.matdict["sizey"] = (
            a0 * nsize * np.sqrt(3) if lattice == "3" else a0 * nsize
        )
        self.matdict["sizez"] = (
            a0 * nsize * np.sqrt(8 / 3) if lattice == "3" else a0 * nsize
        )

        # Create SRIM target
        elements = [
            materials.MATERIALS_BY_SYMBOL[sym].srim_element
            for sym in self.matdict["symbols"]
        ]
        density = np.mean(
            [
                materials.MATERIALS_BY_SYMBOL[sym].density
                for sym in self.matdict["symbols"]
            ]
        )
        layer = srim.target.Layer(
            width=self.srim_width,
            phase=0,
            density=density,
            elements=elements,
            stoichs=self.matdict["stoichs"],
        )
        self.target = srim.target.Target(layers=[layer])

    def run(
        self,
        path_spectrapka_in,
        path_spectrapka_events,
        dir_root,
        srim_width,
        calculation: str,
        max_recoil_energy: float,
        exclude_recoils: list[int] | None = None,
        max_srim_iters: int = 32,
    ) -> None:
        """Run the SPECTRA-PKA to SRIM workflow.

        Parameters
        ----------
        path_spectrapka_in : Path
            SPECTRA-PKA input file path, for target definition.
        path_spectrapka_events : Path
            SPECTRA-PKA config_events.pka file path, for recoils data.
        dir_root : Path
            Root output directory where all data will be stored.
        srim_width : float, optional (default=1e8)
            The SPECTRA-PKA box might be small for SRIM ions. To avoid backscattering and
            transmission, SPECTRA-PKA recoils are injected at the middle of a thick SRIM target
            of this width (in Angstrom). In postprocessing, the depth offset is corrected to get
            the position in the SPECTRA-PKA box. Note that due to SRIM limitations, recoils
            positions are rounded to a number of the form "xxxxx.E+xx",
            if this width is set too high, precision might be lost, but
            backscattering/transmission might happen if set too low.
            Unfortunately, there is not a strict rule to set this value.
        calculation : str
            SRIM calculation mode: "quick", "full" or "mono".
        max_recoil_energy : float
            Recoils above this energies, will be sent to SRIM, in eV.
        exclude_recoils : list[str] | None (default=None)
            List of symbols of recoils atoms to exclude from processing.
        max_srim_iters : int, optional (default=32)
            Maximum number of SRIM iterations.
        """
        self.path_spectrapka_in = path_spectrapka_in
        self.path_spectrapka_events = path_spectrapka_events
        self.dir_root = dir_root
        self.srim_width = srim_width

        self.dir_root.mkdir(parents=True, exist_ok=True)
        self.recoils_db = RecoilsDB(self.dir_root / "recoils.db")
        self.__spectrapka_in_to_srim_target()

        # Convert SPECTRA-PKA events to SQLite3 database
        self.recoils_db.process_config_events(
            self.path_spectrapka_events, exclude_recoils
        )
        # Read from SQLite3 database
        (
            nions,
            atomic_numbers,
            recoil_energies,
            _depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
            _times,
            _events,
        ) = self.__read_spectrapka_events_for_srim()
        # To avoid backscattering and transmission, inject all SPECTRA-PKA recoils at mid-target
        depths_srim = np.full(nions, self.srim_width / 2.0, dtype=np.float64)
        py2srim = srim.Py2SRIM()
        py2srim.run(
            dir_root=self.dir_root,
            target=self.target,
            calculation=calculation,
            atomic_numbers=atomic_numbers,
            energies=recoil_energies,
            depths=depths_srim,
            ys=ys,
            zs=zs,
            cosxs=cosxs,
            cosys=cosys,
            coszs=coszs,
            max_recoil_energy=max_recoil_energy,
            max_srim_iters=max_srim_iters,
            fail_on_backscatt=True,
            fail_on_transmit=True,
            ignore_32bit_warning=True,
        )
        # Undo artificial offsets during SRIM calculations to avoid backscattering/transmission.
        cur = self.recoils_db.cursor()
        cur.execute(
            """
            UPDATE recoils
            SET depth = depth + (
                SELECT x - ? FROM events WHERE events.event = recoils.event
            )
            """,
            (self.srim_width / 2.0,),
        )
        cur.execute(
            """
            UPDATE ions_vacs
            SET depth = depth + (
                SELECT x - ? FROM events WHERE events.event = ions_vacs.event
            )
            """,
            (self.srim_width / 2.0,),
        )
        self.recoils_db.commit()

        return self.recoils_db
