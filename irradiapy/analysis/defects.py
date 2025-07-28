"""Defects analysis module."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy import typing as npt

from irradiapy import dtypes, materials, utils
from irradiapy.analysis.defectsidentifier import DefectsIdentifier
from irradiapy.io import BZIP2LAMMPSReader, LAMMPSReader, LAMMPSWriter


def identify_defects(
    lattice: str,
    a0: float,
    data_atoms: defaultdict,
    a1: None | float = None,
    pos_pka: None | npt.NDArray[np.float64] = None,
    theta_pka: None | float = None,
    phi_pka: None | float = None,
    transform: None | bool = False,
    debug: bool = False,
) -> dtypes.Defect:
    """Identify defects in a given atomic structure.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'natoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    a1 : float, optional (default=None)
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : npt.NDArray[np.float64], optional (default=None)
        Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
        recentered and aligned.
    theta_pka : float, optional (default=None)
        Polar angle (in radians) for the PKA direction.
    phi_pka : float, optional (default=None)
        Azimuthal angle (in radians) for the PKA direction.
    transform : bool, optional (default=False)
        If True, defects are recentered and aligned with the PKA direction (if provided). If
        True but no PKA parameters are provided, defects are recentered based on their
        average position. Note that the box boundaries are not modified for visualization
        purposes, only the atomic positions are transformed.
    debug : bool, optional (default=False)
        If `True`, enables debug mode for additional output.

    Returns
    -------
    dtypes.Defect
        An array of identified defects in the structure.
    """
    defects_finder = DefectsIdentifier(lattice=lattice, a0=a0, debug=debug)
    defects = defects_finder.identify(
        data_atoms=data_atoms,
        a1=a1,
        pos_pka=pos_pka,
        theta_pka=theta_pka,
        phi_pka=phi_pka,
        transform=transform,
    )
    return defects


def identify_lammps_dump(
    lattice: str,
    a0: float,
    path_dump: Path,
    path_dump_defects: Path,
    a1: None | float = None,
    pos_pka: None | npt.NDArray[np.float64] = None,
    theta_pka: None | float = None,
    phi_pka: None | float = None,
    transform: None | bool = False,
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    """Identify defects in a LAMMPS dump file.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'natoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    path_dump : Path
        Path to the LAMMPS dump file to read. Can be compressed with `.bz2` or not.
    path_dump_defects : Path
        Path to the output file where identified defects will be written (in text format).
    a1 : float, optional
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : npt.NDArray[np.float64], optional
        Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
        recentered and aligned.
    theta_pka : float, optional
        Polar angle (in radians) for the PKA direction.
    phi_pka : float, optional
        Azimuthal angle (in radians) for the PKA direction.
    transform : bool, optional
        If True, defects are recentered and aligned with the PKA direction (if provided). If
        True but no PKA parameters are provided, defects are recentered based on their
        average position. Note that the box boundaries are not modified for visualization
        purposes, only the atomic positions are transformed.
    overwrite : bool, optional (default=False)
        If True, allows overwriting the output file if it already exists.
    debug : bool, optional (default=False)
        If `True`, enables debug mode for additional output.
    """
    if path_dump_defects.exists():
        if overwrite:
            path_dump_defects.unlink()
        else:
            raise FileExistsError(f"Defects file {path_dump_defects} already exists.")
    if debug:
        print(f"Identifying defects in {path_dump}")
    if path_dump.suffix == ".bz2":
        reader = BZIP2LAMMPSReader(path_dump)
    else:
        reader = LAMMPSReader(path_dump)
    writer = LAMMPSWriter(path_dump_defects, mode="a")
    defects_finder = DefectsIdentifier(lattice=lattice, a0=a0, debug=debug)
    for data_atoms in reader:
        if debug:
            print(f"Timestep {data_atoms['timestep']}")
        defects = defects_finder.identify(
            data_atoms,
            a1=a1,
            pos_pka=pos_pka,
            theta_pka=theta_pka,
            phi_pka=phi_pka,
            transform=transform,
        )
        writer.write(defects)


def plot_mddb_nd(
    target_dir: Path,
    mat_pka: materials.Material,
    mat_target: materials.Material,
    path_plot: Path,
    dpi: int = 300,
) -> None:
    """Plot the number of defects (vacancies) as a function of the PKA energy from a molecular
    dynamics database.

    Note
    ----
    The number of defects is estimated from the vacancy counts.

    Parameters
    ----------
    target_dir : Path
        Directory containing the MD database.
    mat_pka : materials.Material
        Material of the PKA.
    mat_target : materials.Material
        Target material.
    path_plot : Path, optional
        Path to save the plot. If `None`, the plot is shown but not saved.
    dpi : int, optional (default=300)
        Dots per inch.
    """

    # Extract data from xyz files
    nd_all = defaultdict(defaultdict)
    nfiles = defaultdict(int)
    for energy_dir in target_dir.glob("*"):
        if not energy_dir.is_dir():
            continue
        energy = int(energy_dir.name)
        nd_all[energy] = {
            "nd": [],
            "mean": 0,
            "std": 0,
            "ste": 0,
        }
        for path_defects in energy_dir.glob("*.xyz"):
            nfiles[energy] += 1
            data_defects = utils.io.get_last_reader(LAMMPSReader(path_defects))

            cond = data_defects["atoms"]["type"] == 0
            vacs = data_defects["atoms"][cond]
            nd_all[energy]["nd"].append(len(vacs))
        nd_all[energy]["mean"] = np.mean(nd_all[energy]["nd"])
        nd_all[energy]["std"] = np.std(nd_all[energy]["nd"])
        nd_all[energy]["ste"] = nd_all[energy]["std"] / np.sqrt(nfiles[energy])

    # Sort energies
    epkas = np.array(
        [int(path.name) for path in target_dir.glob("*") if path.is_dir()],
        dtype=np.int64,
    )
    epkas.sort()

    # Calculate NRT- and fer-arc-dpa
    range_epkas = np.linspace(epkas[0], epkas[-1], 1000)

    range_tdams = mat_target.epka_to_tdam(mat_pka, range_epkas)
    nrt_dpa = mat_target.calc_nrt_dpa(range_tdams)
    fer_arc_dpa = mat_target.calc_fer_arc_dpa(range_tdams)

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("$N_d$")

    axins = ax.inset_axes([0.05, 0.55, 0.4, 0.4])
    # Set the xlim and ylim for the inset in keV
    x1, x2 = 0, 21
    ydata = np.array(
        [nd_all[epka]["mean"] for epka in epkas],
        dtype=np.float64,
    )
    # Find the max y in the zoomed x region
    mask = epkas / 1e3 <= x2
    if np.any(mask):
        ymax = max(ydata[mask]) * 1.1
    else:
        ymax = max(ydata) * 1.1
    axins.set_xlim(x1, x2)
    axins.set_ylim(0.0, ymax)
    _, connectors = ax.indicate_inset_zoom(axins)
    for connector in connectors:  # otherwise the connectors are not visible
        connector.set_visible(True)

    ax.errorbar(
        epkas / 1e3,
        [nd_all[energy]["mean"] for energy in epkas],
        yerr=[nd_all[energy]["ste"] for energy in epkas],
        marker="o",
        linestyle="none",
        label="MD DB",
    )

    ax.plot(
        range_epkas / 1e3,
        nrt_dpa,
        label="NRT-dpa",
    )
    ax.plot(
        range_epkas / 1e3,
        fer_arc_dpa,
        label="fer-arc-dpa",
    )

    axins.errorbar(
        epkas / 1e3,
        [nd_all[energy]["mean"] for energy in epkas],
        yerr=[nd_all[energy]["ste"] for energy in epkas],
        marker="o",
        linestyle="none",
    )
    axins.plot(
        range_epkas / 1e3,
        nrt_dpa,
    )
    axins.plot(
        range_epkas / 1e3,
        fer_arc_dpa,
    )

    ax.legend(loc="lower right")
    fig.tight_layout()
    if path_plot is not None:
        fig.savefig(path_plot, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)
