"""SPECTRA-PKA subpackage core module."""

# pylint: disable=too-many-lines

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from irradiapy import materials, srim
from irradiapy.spectrapka.recoilsdb import RecoilsDB


@dataclass
class Spectra2SRIM:
    """Spectra2SRIM class to run SPECTRA-PKA to SRIM workflow.

    Attributes
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
    matdict : dict[str, Any]
        Material dictionary with SPECTRA-PKA material information.
    target : srim.target.Target
        SRIM target.
    recoils_db : RecoilsDB
        Database to store all recoils collected from SPECTRA-PKA and SRIM calculations.
    """

    path_spectrapka_in: Path = field(init=False)
    path_spectrapka_events: Path = field(init=False)
    dir_root: Path = field(init=False)
    srim_width: float = field(default=1e8, init=False)
    matdict: dict[str, Any] = field(init=False)
    target: srim.target.Target = field(init=False)
    recoils_db: RecoilsDB = field(init=False)

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

    def __collect_recoils(self, max_recoil_energy: float) -> None:
        """Collect all recoils from all SRIM databases into a single database taking into account
        their hierarchy.

        Parameters
        ----------
        max_recoil_energy : float
            Recoils above this energies will be sent to SRIM, in eV.
        """
        # For each SRIM database, keep track of how many ions we have already consumed.
        # Key is the full path to the srim.db file.
        srim_ion_counters: dict[Path, int] = defaultdict(int)

        def collect_recoils_from_srim(
            tree: tuple[int, ...],
            ion_numb: int,
            event: int,
            depth_offset: float,
        ) -> None:
            """Collect recoils from a single SRIM ion and recurse on high-energy recoils.

            Parameters
            ----------
            tree : tuple[int, ...]
                Sequence of atomic numbers describing the SRIM path, e.g. (26,), (26, 76), ...
                The corresponding SRIM DB is at:
                    dir_out / '26' / '76' / ... / 'srim.db'
            ion_numb : int
                Ion number inside this SRIM DB (ion_numb column).
            event : int
                Original SPECTRA event index (1-based).
            """
            # Path to current SRIM DB for this tree
            dir_branch = self.dir_root.joinpath(*(str(branch) for branch in tree))
            path_branch_db = dir_branch / "srim.db"

            # If this SRIM DB does not exist (e.g., max_srim_iters prevented creation), stop here.
            if not path_branch_db.exists():
                return

            srimdb_branch = srim.SRIMDB(
                path_db=path_branch_db,
                calculation=None,
                target=None,
            )

            collisions = list(
                srimdb_branch.collision.read(
                    what="depth, y, z, cosx, cosy, cosz, recoil_energy, atom_hit",
                    condition=f"WHERE ion_numb={ion_numb}",
                )
            )
            srimdb_branch.close()

            for depth, y, z, cosx, cosy, cosz, recoil_energy, atom_hit in collisions:
                atomic_number = int(materials.ATOMIC_NUMBER_BY_SYMBOL[atom_hit])

                if recoil_energy < max_recoil_energy:
                    # This recoil is below threshold -> terminal; store it.
                    self.recoils_db.insert_recoil(
                        event=event,
                        atomic_number=atomic_number,
                        recoil_energy=recoil_energy,
                        depth=depth + depth_offset,
                        y=y,
                        z=z,
                        cosx=cosx,
                        cosy=cosy,
                        cosz=cosz,
                    )
                    continue

                # Above threshold: try to follow to the next SRIM level.
                leaf_tree = (*tree, atomic_number)
                dir_leaf = self.dir_root.joinpath(
                    *(str(subleaf) for subleaf in leaf_tree)
                )
                path_leaf_db = dir_leaf / "srim.db"

                # If the child SRIM DB does not exist (e.g., limited by max_srim_iters),
                # treat as terminal.
                if not path_leaf_db.exists():
                    self.recoils_db.insert_recoil(
                        event=event,
                        atomic_number=atomic_number,
                        recoil_energy=recoil_energy,
                        depth=depth + depth_offset,
                        y=y,
                        z=z,
                        cosx=cosx,
                        cosy=cosy,
                        cosz=cosz,
                    )
                    continue

                # Map this collision to the correct ion in the child SRIM DB.
                child_ion_numb = srim_ion_counters[path_leaf_db] + 1
                srim_ion_counters[path_leaf_db] = child_ion_numb
                collect_recoils_from_srim(
                    tree=leaf_tree,
                    ion_numb=child_ion_numb,
                    event=event,
                    depth_offset=depth_offset,
                )

        # Start from SPECTRA-PKA recoils
        (
            nions,
            atomic_numbers,
            recoil_energies,
            depths,
            ys,
            zs,
            cosxs,
            cosys,
            coszs,
            _,
            events,
        ) = self.__read_spectrapka_events_for_srim()

        for i in range(nions):
            event = int(events[i])
            atomic_number = int(atomic_numbers[i])
            recoil_energy = recoil_energies[i]
            depth = depths[i]
            y = ys[i]
            z = zs[i]
            cosx = cosxs[i]
            cosy = cosys[i]
            cosz = coszs[i]

            # Low-energy SPECTRA-PKA recoil -> store into final DB.
            if recoil_energy < max_recoil_energy:
                self.recoils_db.insert_recoil(
                    event=event,
                    atomic_number=atomic_number,
                    recoil_energy=recoil_energy,
                    depth=depth,
                    y=y,
                    z=z,
                    cosx=cosx,
                    cosy=cosy,
                    cosz=cosz,
                )
                continue

            # High-energy SPECTRA-PKA recoil: follow SRIM cascades starting at Z/srim.db.
            dir_branch = self.dir_root / str(atomic_number)
            path_branch_db = dir_branch / "srim.db"

            # No SRIM data for this recoil: treat as terminal.
            # Likely due to max SRIM iterations.
            if not path_branch_db.exists():
                self.recoils_db.insert_recoil(
                    event=event,
                    atomic_number=atomic_number,
                    recoil_energy=recoil_energy,
                    depth=depth,
                    y=y,
                    z=z,
                    cosx=cosx,
                    cosy=cosy,
                    cosz=cosz,
                )
                continue

            # Ion index in this first-level SRIM.
            ion_numb = srim_ion_counters[path_branch_db] + 1
            srim_ion_counters[path_branch_db] = ion_numb

            collect_recoils_from_srim(
                tree=(atomic_number,),
                ion_numb=ion_numb,
                event=event,
                depth_offset=-self.srim_width / 2 + depth,
            )

    def __collect_ions_vacs(self, max_recoil_energy: float) -> None:
        """Collect all ions and vacancies from all SRIM databases into a single database taking into
        account their hierarchy.

        TRIMDAT ->
        initial position of the ions, which are pushed, creating a vacancy.
        RANGE_3D ->
        final position of the ions after being pushed by SRIM (atomic number from TRIMDAT).

        Parameters
        ----------
        max_recoil_energy : float
            Recoils above this energies will be sent to SRIM, in eV.
        """
        # For each SRIM database, keep track of how many ions we have already consumed.
        # Key is the full path to the srim.db file.
        srim_ion_counters: dict[Path, int] = defaultdict(int)

        def collect_ions_vacs_from_srim(
            tree: tuple[int, ...],
            ion_numb: int,
            event: int,
            depth_offset: float,
        ) -> None:
            """Collect ions and vacs from a single SRIM ion and recurse on high-energy recoils.

            Parameters
            ----------
            tree : tuple[int, ...]
                Sequence of atomic numbers describing the SRIM path, e.g. (26,), (26, 76), ...
                The corresponding SRIM DB is at:
                    dir_out / '26' / '76' / ... / 'srim.db'
            ion_numb : int
                Ion number inside this SRIM DB (ion_numb column).
            event : int
                Original SPECTRA event index (1-based).
            """
            # Path to current SRIM DB for this tree
            dir_branch = self.dir_root.joinpath(*(str(branch) for branch in tree))
            path_branch_db = dir_branch / "srim.db"

            # If this SRIM DB does not exist (e.g., max_srim_iters prevented creation), stop here.
            if not path_branch_db.exists():
                return

            srimdb_branch = srim.SRIMDB(
                path_db=path_branch_db,
                calculation=None,
                target=None,
            )

            trimdat = list(
                srimdb_branch.trimdat.read(
                    what="depth, y, z, atom_numb",
                    condition=f"WHERE ion_numb={ion_numb}",
                )
            )
            range3d = list(
                srimdb_branch.range3d.read(
                    what="depth, y, z",
                    condition=f"WHERE ion_numb={ion_numb}",
                )
            )
            collisions = list(
                srimdb_branch.collision.read(
                    what="recoil_energy, atom_hit",
                    condition=f"WHERE ion_numb={ion_numb}",
                )
            )
            srimdb_branch.close()

            for depth, y, z, atom_numb in trimdat:
                self.recoils_db.insert_ion_vac(
                    event=event,
                    atom_numb=atom_numb,
                    depth=depth + depth_offset,
                    y=y,
                    z=z,
                )

            for depth, y, z in range3d:
                self.recoils_db.insert_ion_vac(
                    event=event,
                    atom_numb=0,
                    depth=depth + depth_offset,
                    y=y,
                    z=z,
                )

            for recoil_energy, atom_hit in collisions:
                atomic_number = int(materials.ATOMIC_NUMBER_BY_SYMBOL[atom_hit])

                if recoil_energy < max_recoil_energy:
                    # This recoil is below threshold -> terminal
                    continue

                leaf_tree = (*tree, atomic_number)
                dir_leaf = self.dir_root.joinpath(
                    *(str(subleaf) for subleaf in leaf_tree)
                )
                path_leaf_db = dir_leaf / "srim.db"

                # If the child SRIM DB does not exist (e.g., limited by max_srim_iters),
                # treat as terminal.
                if not path_leaf_db.exists():
                    continue

                # Map this recoil to the correct ion in the child SRIM DB.
                child_ion_numb = srim_ion_counters[path_leaf_db] + 1
                srim_ion_counters[path_leaf_db] = child_ion_numb
                collect_ions_vacs_from_srim(
                    tree=leaf_tree,
                    ion_numb=child_ion_numb,
                    event=event,
                    depth_offset=depth_offset,
                )

        # Start from SPECTRA-PKA recoils
        (
            nions,
            atomic_numbers,
            recoil_energies,
            depths,
            _,
            _,
            _,
            _,
            _,
            _,
            events,
        ) = self.__read_spectrapka_events_for_srim()

        for i in range(nions):
            event = int(events[i])
            atomic_number = int(atomic_numbers[i])
            depth = depths[i]
            recoil_energy = recoil_energies[i]

            # If this SPECTRA-PKA recoil is below the threshold, it was never transferred to SRIM,
            # so there are no ions/vacs to collect for it.
            if recoil_energy < max_recoil_energy:
                continue

            # High-energy SPECTRA-PKA recoil: follow SRIM cascades starting at Z/srim.db.
            dir_branch = self.dir_root / str(atomic_number)
            path_branch_db = dir_branch / "srim.db"

            # No SRIM data for this recoil: treat as terminal.
            # Likely due to max SRIM iterations.
            if not path_branch_db.exists():
                continue

            # Ion index in this first-level SRIM.
            ion_numb = srim_ion_counters[path_branch_db] + 1
            srim_ion_counters[path_branch_db] = ion_numb

            collect_ions_vacs_from_srim(
                tree=(atomic_number,),
                ion_numb=ion_numb,
                event=event,
                depth_offset=-self.srim_width / 2 + depth,
            )

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
            remove_offsets=False,
            ignore_32bit_warning=True,
        )
        # Collect all recoils data into a single database
        self.__collect_recoils(max_recoil_energy)
        # Collect all ions and vacancies into a single database
        self.__collect_ions_vacs(max_recoil_energy)
        self.recoils_db.commit()
