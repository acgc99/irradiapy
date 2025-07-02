"""Utility functions related to dpa for SRIM data."""

import sqlite3
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from irradiapy import dpa, materials
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.math_utils import fit_lorentzian
from irradiapy.srimpy.srimdb import SRIMDB


def get_dpas(
    srimdb: SRIMDB,
    path_collisions: Path,
    fluence: float,
    path_db: Path,
    nbins: int = 100,
    depth_offset: Union[int, float, np.number] = 0.0,
) -> None:
    """Create a table with NRT, fer-arc and debris-dpa in a database.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    path_collisions : Path
        File containing all collision debris to analyze.
    fluence : float
        Fluence, in ions/A2.
    path_db : Path
        Path to the database file.
    nbins : int, optional
        Number of depth bins. Default is 100.
    """
    mat1 = materials.get_material_by_atomic_number(
        list(srimdb.trimdat.read(what="atom_numb", condition="WHERE ion_numb = 1"))[0][
            0
        ]
    )
    mat2 = materials.get_material_by_atomic_number(
        srimdb.target.layers[0].elements[0].atomic_number
    )
    nions = srimdb.nions
    # SRIM COLLISON.txt dpa
    depths, defects_nrt, defects_fer_arc = [], [], []
    for depth, pka_e in srimdb.collision.read(what="depth, recoil_energy"):
        depths.append(depth + depth_offset)
        tdam = dpa.compute_damage_energy(pka_e, mat1, mat2)
        defects_nrt.append(dpa.calc_nrt_dpa(tdam, mat2))
        defects_fer_arc.append(dpa.calc_fer_arc_dpa(tdam, mat2))
    depths = np.array(depths)
    defects_nrt = np.array(defects_nrt)
    defects_fer_arc = np.array(defects_fer_arc)
    # Debris dpa
    depth_debris = np.array([], dtype=float)
    nion = 0
    total_debris = 0
    for data_defects in LAMMPSReader(path_collisions):
        vacs = data_defects["atoms"][data_defects["atoms"]["type"] == 0]
        depth_debris = np.concatenate((depth_debris, vacs["x"]))
        nion += 1
        total_debris += vacs.size
        if nion % 100 == 0:
            print(f"{nion}/{nions}")
    # Depth binning
    depth_edges = np.histogram_bin_edges(depths, bins=nbins)
    depth_centers = (depth_edges[1:] - depth_edges[:-1]) / 2.0 + depth_edges[:-1]
    width = depth_edges[1] - depth_edges[0]
    depth_digitize = np.digitize(depths, depth_edges)
    # NRT-dpa
    hist_nrt = np.array(
        [np.sum(defects_nrt[depth_digitize == i]) for i in range(1, nbins + 1)]
    )
    dpa_nrt = hist_nrt / nions * fluence / mat2.density / width
    # fer-arc-dpa
    hist_fer_arc = np.array(
        [np.sum(defects_fer_arc[depth_digitize == i]) for i in range(1, nbins + 1)]
    )
    dpa_fer_arc = hist_fer_arc / nions * fluence / mat2.density / width
    # debris-dpa
    hist_debris, _ = np.histogram(depth_debris, bins=depth_edges)
    dpa_debris = hist_debris / nions * fluence / mat2.density / width
    # Save
    con = sqlite3.connect(path_db)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS dpa")
    cur.execute("CREATE TABLE dpa (depth REAL, nrt REAL, arc REAL, debris REAL)")
    cur.executemany(
        "INSERT INTO dpa (depth, nrt, arc, debris) VALUES (?, ?, ?, ?)",
        np.column_stack((depth_centers, dpa_nrt, dpa_fer_arc, dpa_debris)),
    )
    con.commit()
    cur.close()
    con.close()


def load_results(path_db: Path, what: str = "*", condition: str = "") -> tuple:
    """Load results from the table created by `get_dpas`.

    Parameters
    ----------
    path_db : Path
        Path to the database file.
    what : str, optional
        Columns to select from the database, by default "*".
    condition : str, optional
        SQL condition for the query, by default "".

    Returns
    -------
    tuple
        Tuple containing depth_centers, dpa_nrt, dpa_fer_arc, and dpa_debris arrays.
    """
    con = sqlite3.connect(path_db)
    cur = con.cursor()
    cur.execute(f"SELECT {what} FROM dpa {condition}")
    data = np.array(cur.fetchall())
    depth_centers = data[:, 0]
    dpa_nrt = data[:, 1]
    dpa_fer_arc = data[:, 2]
    dpa_debris = data[:, 3]
    cur.close()
    con.close()
    return depth_centers, dpa_nrt, dpa_fer_arc, dpa_debris


def plot_dpa(
    path_db: Path,
    # depth_offset: Union[int, float, np.number] = 0.0,
    path_plot: Optional[Path] = None,
    dpi: int = 300,
    path_fit: Optional[Path] = None,
    p0: Optional[float] = None,
    asymmetry: float = 1.0,
) -> None:
    """Plot the dpa analysis results from the database.

    Parameters
    ----------
    path_db : Path
        Path to the database file.
    path_plot : Path, optional
        Output path for the plot, by default None. If None, the plot is shown.
    depth_offset : Union[int, float, np.number], optional
        Depth offset to apply to be applied, by default 0.0.
    dpi : int, optional
        Dots per inch, by default 300.
    path_fit : Path, optional
        Output path for the fit parameters, by default None.
    p0 : float, optional
        Initial guess of fit parameters, by default None.
    asymmetry : float, optional
        Asymmetry fit parameter bound, by default 1.0.
    """
    depth_centers, dpa_nrt, dpa_fer_arc, dpa_debris = load_results(path_db)
    # depth_centers += depth_offset
    total_nrt = dpa_nrt.sum()
    total_debris = dpa_debris.sum()
    # Fit dpa_debris
    fit = False
    if path_fit:
        try:
            # p0 = [depth_centers[np.argmax(dpa_debris)], 1.0, 1.0, 1.0]
            popt, _, dpa_fit = fit_lorentzian(depth_centers, dpa_debris, p0, asymmetry)
            if path_fit:
                with open(path_fit, "w", encoding="utf-8") as file:
                    file.write("Debris dpa lorentzian fit: z0, sigma, A, a\n")
                    file.write(
                        (
                            "See Eq. (1) of Nuclear Instruments and Methods in Physics "
                            "Research B 500-501 (2021) 52-56\n"
                        )
                    )
                    file.write(", ".join(map(str, popt)) + "\n")
            fit = True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Fit failed: {exc}")
    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec()
    ax = fig.add_subplot(gs[0, 0])
    # Scatter
    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("dpa")
    ax.scatter(depth_centers, dpa_nrt, label="NRT-dpa")
    ax.scatter(depth_centers, dpa_fer_arc, label="fer-arc-dpa")
    ax.scatter(depth_centers, dpa_debris, label="debris-dpa")
    if fit:
        ax.plot(
            depth_centers,
            dpa_fit(depth_centers),
            label="debris-dpa fit",
        )
    efficiency = [
        Line2D(
            [0],
            [0],
            color="none",
            label=r"$\overline{\xi}$ = "
            + rf"{round(total_debris / total_nrt * 100)} %",
        )
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + efficiency)
    fig.tight_layout()
    if path_plot:
        plt.savefig(path_plot, dpi=dpi)
    else:
        plt.show()
    plt.close()
