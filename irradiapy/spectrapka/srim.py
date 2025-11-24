"""Transfer and run SPECTRA-PKA PKAs to SRIM."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy import typing as npt

from irradiapy import dtypes, materials, srim, utils
from irradiapy.damagedb import DamageDB
from irradiapy.io import LAMMPSWriter
from irradiapy.spectrapka.core import get_bca_pkas, get_npkas, spectrapka_to_sqlite3


def process_spectrapka_in(path_spectrapka_in: Path) -> defaultdict[str, Any]:
    """Process the SPECTRA-PKA input file to extract information.

    Parameters
    ----------
    path_spectrapka_in : Path
        The path to the SPECTRA-PKA input file.

    Returns
    -------
    defaultdict[str, Any]
        A dictionary containing the processed data from the SPECTRA-PKA input file.

    Notes
    -----
        Isotopes with the same element symbol are combined into a single element with the
        sum of their stoichiometries.
    """

    data = defaultdict(None)

    # Get target
    cols = []
    reading_columns = False
    with open(path_spectrapka_in, "r", encoding="utf-8") as file:
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
    data["stoichs"] = np.array([col[1] for col in cols], dtype=np.float64)
    data["symbols"] = np.array([col[2] for col in cols], dtype=str)
    data["atomic_numbers"] = np.array([col[3] for col in cols], dtype=np.int64)

    # SRIM does not distinguish isotopes, so keep only the isotope with the highest
    # stoichiometry for each element
    unique_symbols = np.unique(data["symbols"])
    unique_stoichs = [
        np.sum(data["stoichs"][data["symbols"] == el]) for el in unique_symbols
    ]
    max_stoich_isotope = [
        np.argmax(data["stoichs"][data["symbols"] == el]) for el in unique_symbols
    ]
    unique_atomic_numbers = [
        data["atomic_numbers"][data["symbols"] == el][max_stoich_isotope[i]]
        for i, el in enumerate(unique_symbols)
    ]
    data["stoichs"] = np.array(unique_stoichs, dtype=np.float64)
    data["symbols"] = np.array(unique_symbols, dtype=str)
    data["atomic_numbers"] = np.array(unique_atomic_numbers, dtype=np.int64)

    # Get lattice definition
    a0, lattice, nsize = None, None, None
    with open(path_spectrapka_in, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().startswith("latt="):
                a0 = float(line.split("=")[1].strip())
            elif line.strip().startswith("box_type="):
                lattice = line.split("=")[1].strip()
            elif line.strip().startswith("box_nunits="):
                nsize = int(line.split("=")[1].strip())
    data["a0"] = a0
    data["lattice"] = lattice
    data["nsize"] = nsize

    return data


def __spectrapka_to_srim(path_db: Path, criterion: Callable) -> dict[str, np.ndarray]:
    """Convert SPECTRA-PKA PKAs to SRIM input format.

    Parameters
    ----------
    path_db : Path
        The path to the database file.
    criterion : Callable[[dict], bool]
        A function that takes a row of the database and returns False if the PKA must be ran in a
        BCA code.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the SRIM input.
    """
    srim_pkas = np.array(
        list(get_bca_pkas(path_db, criterion)), dtype=dtypes.spectra_event
    )
    trimdat = {}
    trimdat["nions"] = len(srim_pkas)
    trimdat["atomic_numbers"] = np.array(
        [
            materials.MATERIALS_BY_SYMBOL[element].atomic_number
            for element in srim_pkas["element"]
        ]
    )
    trimdat["recoil_energies"] = srim_pkas["recoil_energy"]  # Ion energy, eV
    trimdat["depths"] = srim_pkas["x"]  # Initial depth, angstroms
    trimdat["ys"] = srim_pkas["y"]  # Initial y, angstroms
    trimdat["zs"] = srim_pkas["z"]  # Initial z, angstroms
    vs = np.sqrt(
        np.square(srim_pkas["vx"])
        + np.square(srim_pkas["vy"])
        + np.square(srim_pkas["vz"])
    )
    trimdat["cosxs"] = srim_pkas["vx"] / vs  # Initial x direction
    trimdat["cosys"] = srim_pkas["vy"] / vs  # Initial y direction
    trimdat["coszs"] = srim_pkas["vz"] / vs  # Initial z direction
    return trimdat


def __combine_spectra_srim(
    path_db: Path,
    criterion: Callable[[dict], bool],
    depths: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    zs: npt.NDArray[np.float64],
    periodic: bool,
) -> None:
    """Combine SPECTRA-PKA and SRIM databases.

    Gets a list with the PKA numbers that have not been processed by SRIM (`spectra_pkas`) and
    the ones that have been processed (`srim_pkas`). For the total number of PKAs in the
    database, if a PKA is missing from `spectra_pkas`, it means that it has been processed
    by SRIM, so the data is read the SRIM output.

    Parameters
    ----------
    path_db : Path
        The path to the database file.
    criterion : Callable[[dict], bool]
        A function that takes a row of the database and returns False if the PKA must be ran in a
        BCA code.
    """
    srimdb = srim.SRIMDB(path_db=path_db)
    spectra_pkas = get_bca_pkas(
        path_db, criterion, negate=True, conn=srimdb
    )  # not used by SRIM
    srim_pkas = get_bca_pkas(
        path_db, criterion, negate=False, conn=srimdb
    )  # used by SRIM

    npkas = get_npkas(path_db)
    curr_spectra = 0  # SPECTRA-PKA event counter
    curr_srim = 0  # SRIM collision counter
    # If event is missing from spectra_pkas, pick next collision where ion_numb == curr_srim

    cur = srimdb.cursor()
    cur.execute("DROP TABLE IF EXISTS spectra_srim")
    cur.execute(
        (
            "CREATE TABLE IF NOT EXISTS spectra_srim (event INTEGER, timestep INTEGER, "
            "element TEXT, time REAL, recoil_energy REAL, depth REAL, y REAL, z REAL, cosx REAL, "
            "cosy REAL, cosz REAL)"
        )
    )
    for _ in range(npkas):
        curr_spectra += 1
        spectra = next(spectra_pkas, None)
        if spectra is None:
            srim_col = next(srim_pkas)
            event = int(srim_col["event"])
            timestep = int(srim_col["timestep"])
            time = float(srim_col["time"])

            curr_srim += 1
            collisions = srimdb.collision.read(
                what="recoil_energy, depth, y, z, cosx, cosy, cosz, atom_hit",
                condition=f"WHERE ion_numb = {curr_srim}",
            )
            for collision in collisions:
                recoil_energy, depth, y, z, cosx, cosy, cosz, atom_hit = collision
                if periodic:
                    depth += depths[curr_srim - 1]
                    y += ys[curr_srim - 1]
                    z += zs[curr_srim - 1]
                cur.execute(
                    (
                        "INSERT INTO spectra_srim(event, timestep, element, time, recoil_energy, "
                        "depth, y, z, cosx, cosy, cosz) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    (
                        event,
                        timestep,
                        atom_hit,
                        time,
                        recoil_energy,
                        depth,
                        y,
                        z,
                        cosx,
                        cosy,
                        cosz,
                    ),
                )
        else:
            event = int(spectra["event"])
            timestep = int(spectra["timestep"])
            element = str(spectra["element"])
            time = float(spectra["time"])
            recoil_energy = float(spectra["recoil_energy"])
            depth = float(spectra["x"])
            y = float(spectra["y"])
            z = float(spectra["z"])
            v = np.sqrt(
                np.square(spectra["vx"])
                + np.square(spectra["vy"])
                + np.square(spectra["vz"])
            )
            cosx = float(spectra["vx"] / v)
            cosy = float(spectra["vy"] / v)
            cosz = float(spectra["vz"] / v)
            cur.execute(
                (
                    "INSERT INTO spectra_srim(event, timestep, time, element, recoil_energy, "
                    "depth, y, z, cosx, cosy, cosz) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    event,
                    timestep,
                    time,
                    element,
                    recoil_energy,
                    depth,
                    y,
                    z,
                    cosx,
                    cosy,
                    cosz,
                ),
            )
    del spectra_pkas, srim_pkas  # otherwise the database is locked
    # Sort in event ascending order
    cur.execute(
        "CREATE TABLE spectra_srim0 AS SELECT * FROM spectra_srim ORDER BY event"
    )
    cur.execute("DROP TABLE spectra_srim")
    cur.execute("ALTER TABLE spectra_srim0 RENAME TO spectra_srim")
    cur.execute(
        "CREATE INDEX eventRecoilEnergyIdx ON spectra_srim(event, recoil_energy)"
    )

    srimdb.commit()
    cur.close()
    srimdb.close()


def run_spectrapka_srim(
    path_spectrapka_in: Path,
    path_spectrapka_pkas: Path,
    path_db: Path,
    periodic: bool,
    criterion: Callable[[dict], bool],
    srim_mode: str,
    srim_width: None | float = None,
    iter_max: None | int = None,
    seed: int = 0,
) -> None:
    """Run SRIM with SPECTRA-PKA PKAs.

    Parameters
    ----------
    path_spectrapka_in : Path
        The path to the SPECTRA-PKA input file.
    path_spectrapka_pkas : Path
        The path to the SPECTRA-PKA `config_event.pka` file.
    path_db : Path
        The path to the output database file.
    periodic : bool
        Whether the system has periodic boundary conditions. If `periodic = True`, then the width
        of the SRIM target must be given and must be large enough so that there is not
        backscattering or transmission of ions.
    srim_mode : str
        The SRIM calculation mode.
    srim_width : float, optional (default=None)
        The width of the SRIM target in angstroms. Required if `periodic = True`.
    iter_max : int, optional (default=None)
        The maximum number of iterations for the SRIM calculation.
    seed : int, optional (default=0)
        The random seed for the SRIM calculation.
    """

    # Process SPECTRA-PKA input file
    spectrapka_in = process_spectrapka_in(path_spectrapka_in)

    # Determine target width
    if periodic:
        if srim_width is None:
            raise ValueError("`srim_width` must be specified for periodic targets.")
    else:
        srim_width = spectrapka_in["nsize"] * spectrapka_in["a0"]

    # Convert SPECTRA-PKA events to SQLite3
    spectrapka_to_sqlite3(path_db, path_spectrapka_pkas)

    # Create SRIM target
    elements = [
        materials.MATERIALS_BY_SYMBOL[sym].srim_element
        for sym in spectrapka_in["symbols"]
    ]
    density = np.mean(
        [materials.MATERIALS_BY_SYMBOL[sym].density for sym in spectrapka_in["symbols"]]
    )
    layer = srim.target.Layer(
        width=srim_width, phase=0, density=density, elements=elements, stoichs=[1.0]
    )
    target = srim.target.Target(layers=[layer])

    # Get PKAs to be transferred to SRIM
    srim_pkas = __spectrapka_to_srim(path_db, criterion)
    # Copy orignal positions
    depths = srim_pkas["depths"].copy()
    ys = srim_pkas["ys"].copy()
    zs = srim_pkas["zs"].copy()
    # Start all PKAs half depth if periodic
    if periodic:
        srim_pkas["depths"] = np.full(
            len(srim_pkas["depths"]), srim_width / 2.0, dtype=np.float64
        )
        srim_pkas["ys"] = np.zeros(len(srim_pkas["ys"]), dtype=np.float64)
        srim_pkas["zs"] = np.zeros(len(srim_pkas["zs"]), dtype=np.float64)
    # Run SRIM
    srimdb = srim.SRIMDB(
        path_db=path_db, calculation=srim_mode, target=target, seed=seed
    )
    if len(srim_pkas["atomic_numbers"]) > 0:
        srimdb.run(
            criterion,
            srim_pkas["atomic_numbers"],
            srim_pkas["recoil_energies"],
            periodic,
            srim_pkas["depths"],
            srim_pkas["ys"],
            srim_pkas["zs"],
            srim_pkas["cosxs"],
            srim_pkas["cosys"],
            srim_pkas["coszs"],
            iter_max=iter_max,
        )

    # Combine SPECTRA-PKA and SRIM databases
    __combine_spectra_srim(
        path_db,
        criterion,
        depths,
        ys,
        zs,
        periodic,
    )


def generate_debris(
    path_spectrapka_in: Path,
    periodic: bool,
    path_db: Path,
    dir_mddb: Path,
    compute_tdam: bool,
    path_debris: Path,
    tdam_mode: materials.Material.TdamMode,
    dpa_mode: materials.Material.DpaMode,
    seed: int = 0,
) -> None:
    """Turns SPECTRA-PKA + SRIM's collisions into `.xyz` files for the given database of cascades'
    debris.

    Notes
    -----
        The box starts at (0, 0, 0) and ends at the length given by SPECTRA-PKA.

        Assumes a monolayer monoatomic target and same element for all ions.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    dir_mddb : Path
        Directory where the database of cascades' debris is stored.
    compute_tdam : bool
        Whether to transform the PKA energies into damage energies. It should be `True` for
        MD simulations without electronic stopping.
    path_debris : Path
        Directory where the ions debris will be stored as `.xyz` files.
    tdam_mode : materials.Material.TdamMode
        Mode to convert the PKA energy into damage energy.
    dpa_mode : materials.Material.DpaMode
        Formula to convert the residual energy into Frenkel pairs.
    seed : int, optional (default=0)
        Random seed for placing defects.
    """

    # Get box dimensions
    spectrapka_in = process_spectrapka_in(path_spectrapka_in)
    xlo, xhi = 0.0, spectrapka_in["nsize"] * spectrapka_in["a0"]
    ylo, yhi = 0.0, xhi
    zlo, zhi = 0.0, xhi

    # Extract data from the database
    srimdb = srim.SRIMDB(path_db=path_db)
    # Assume all PKAs are of the same element
    cur = srimdb.cursor()
    cur.execute("SELECT element FROM spectra_srim WHERE event = 1")
    element = cur.fetchone()[0]
    cur.close()
    mat_pka = materials.MATERIALS_BY_SYMBOL[element]
    mat_target = materials.MATERIALS_BY_ATOMIC_NUMBER[
        srimdb.target.layers[0].elements[0].atomic_number
    ]

    nions = get_npkas(srimdb.path_db)
    damagedb = DamageDB(
        dir_mddb=dir_mddb,
        compute_tdam=compute_tdam,
        mat_pka=mat_pka,
        mat_target=mat_target,
        dpa_mode=dpa_mode,
        tdam_mode=tdam_mode,
        seed=seed,
    )
    with LAMMPSWriter(path_debris) as writer:
        for nion in range(1, nions + 1):
            defects, time = __generate_pka_defects(nion, srimdb, damagedb)

            data_defects = defaultdict(None)
            data_defects["time"] = time
            data_defects["timestep"] = 0
            data_defects["natoms"] = defects.size
            data_defects["boundary"] = (
                ["pp", "pp", "pp"] if periodic else ["ff", "ff", "ff"]
            )
            data_defects["xlo"] = xlo
            data_defects["xhi"] = xhi
            data_defects["ylo"] = ylo
            data_defects["yhi"] = yhi
            data_defects["zlo"] = zlo
            data_defects["zhi"] = zhi
            data_defects["atoms"] = defects

            data_defects = utils.math.apply_boundary_conditions(
                data_defects, periodic, periodic, periodic
            )

            writer.write(data_defects)


def __generate_pka_defects(
    nion: int,
    srimdb: srim.SRIMDB,
    damagedb: DamageDB,
) -> np.ndarray:
    """Generates the defects for a specific PKA.

    Parameters
    ----------
    nion : int
        Ion number.
    srimdb : srim.SRIMDB
        SRIM database class.
    damagedb : DamageDB
        DamageDB class that will choose MD debris.

    Returns
    -------
    np.ndarray
        An array containing the defects generated by a single ion.
    """
    defects = np.empty(0, dtype=dtypes.defect)
    cur = srimdb.cursor()
    query = (
        "SELECT depth, y, z, cosx, cosy, cosz, recoil_energy, time FROM spectra_srim "
        f"WHERE event = {nion}"
    )
    time = 0.0
    for depth, y, z, cosx, cosy, cosz, pka_e, time in cur.execute(query):
        pka_pos = np.array([depth, y, z])
        pka_dir = np.array([cosx, cosy, cosz])
        defects_ = damagedb.get_pka_debris(
            pka_e=pka_e, pka_pos=pka_pos, pka_dir=pka_dir
        )
        defects = np.concatenate((defects, defects_))
    cur.close()
    return defects, time


def get_dpas(
    path_spectrapka_in: Path,
    path_db: Path,
    lattice: str,
) -> tuple[float, float, float]:
    """Turns SPECTRA-PKA + SRIM's collisions into the different dpa metrics.

    Parameters
    ----------
    path_spectrapka_in : Path
        The path to the SPECTRA-PKA input file.
    path_db : Path
        The path to the output database file.
    lattice : str
        The lattice type of the target material. Currently, only "bcc" is supported.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the total NRT, ARC, and FERARC dpa values in the simulation box.
    """
    if lattice not in ["bcc"]:
        raise ValueError(
            f"Unsupported lattice type: '{lattice}'. Only 'bcc' is supported."
        )
    natoms_cell = 1  # number of atoms per unit cell
    if lattice == "bcc":
        natoms_cell = 2

    # Get box dimensions
    spectrapka_in = process_spectrapka_in(path_spectrapka_in)
    nsize = spectrapka_in["nsize"]
    natoms = nsize**3 * natoms_cell
    # xlo, xhi = 0.0, spectrapka_in["nsize"] * spectrapka_in["a0"]
    # ylo, yhi = 0.0, xhi
    # zlo, zhi = 0.0, xhi

    # Extract data from the database
    srimdb = srim.SRIMDB(path_db=path_db)
    # Assume all PKAs are of the same element
    cur = srimdb.cursor()
    cur.execute("SELECT element FROM spectra_srim WHERE event = 1")
    element = cur.fetchone()[0]
    cur.close()
    mat_pka = materials.MATERIALS_BY_SYMBOL[element]
    mat_target = materials.MATERIALS_BY_ATOMIC_NUMBER[
        srimdb.target.layers[0].elements[0].atomic_number
    ]

    nrts = []
    arcs = []
    ferarcs = []
    cur = srimdb.cursor()
    query = "SELECT recoil_energy FROM spectra_srim "
    for (epka,) in cur.execute(query):
        tdam = mat_target.epka_to_tdam(mat_pka, epka)
        nrts.append(mat_target.calc_nrt_dpa(tdam))
        arcs.append(mat_target.calc_arc_dpa(tdam))
        ferarcs.append(mat_target.calc_fer_arc_dpa(tdam))
    nrts = np.array(nrts, dtype=np.float64)
    arcs = np.array(arcs, dtype=np.float64)
    ferarcs = np.array(ferarcs, dtype=np.float64)

    tot_nrt = np.sum(nrts)
    tot_arc = np.sum(arcs)
    tot_ferarc = np.sum(ferarcs)

    rel_nrt = tot_nrt / natoms
    rel_arc = tot_arc / natoms
    rel_ferarc = tot_ferarc / natoms

    return rel_nrt, rel_arc, rel_ferarc
