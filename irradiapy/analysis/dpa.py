"""dpa analysis module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy import typing as npt

from irradiapy import materials, utils
from irradiapy.io import LAMMPSReader
from irradiapy.srimpy import SRIMDB

# region Histograms


# NOTE: The first argument type will determine the calculation method when other BCA methods are
# implemented.
# NOTE: SRIM support only for monoelemental targets.
def get_dpa_1d(
    srimdb: SRIMDB,
    path_debris: Path,
    path_db: Path,
    fluence: float,
    axis: str = "x",
    nbins: int = 100,
    depth_offset: float = 0.0,
) -> None:
    """Perform a dpa histogram as a function of depth along a specified axis.

    Note
    ----
    dpa is calculated from vacancies in the debris file.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    path_debris : Path
        File containing all collision debris to analyze.
    path_db : Path
        Path to the database file.
    fluence : float
        Fluence, in ions/A2.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`, or `"z"`.
    nbins : int, optional (default=100)
        Number of depth bins. Depth range determined from `path_debris` defect positions.
    depth_offset : float, optional (default=0.0)
        Offset to add to the depth values.
    """

    # Get materials
    atomic_number1 = list(
        srimdb.trimdat.read(what="atom_numb", condition="WHERE ion_numb = 1")
    )[0][0]
    mat_pka = materials.MATERIALS_BY_ATOMIC_NUMBER[atomic_number1]
    atomic_number2 = srimdb.target.layers[0].elements[0].atomic_number
    mat_target = materials.MATERIALS_BY_ATOMIC_NUMBER[atomic_number2]
    nions = srimdb.nions

    # SRIM COLLISON.txt dpa
    depths, defects_nrt, defects_fer_arc = [], [], []
    what = "depth, recoil_energy" if axis == "x" else f"{axis}, recoil_energy"
    for depth, pka_e in srimdb.collision.read(what=what):
        depths.append(depth)
        tdam = mat_target.epka_to_tdam(mat_pka, pka_e)
        defects_nrt.append(mat_target.calc_nrt_dpa(tdam))
        defects_fer_arc.append(mat_target.calc_fer_arc_dpa(tdam))
    depths = np.array(depths)
    defects_nrt = np.array(defects_nrt)
    defects_fer_arc = np.array(defects_fer_arc)

    # Debris dpa
    depth_debris = np.array([], dtype=np.float64)
    for data_defects in LAMMPSReader(path_debris):
        vacs = data_defects["atoms"][data_defects["atoms"]["type"] == 0]
        depth_debris = np.concatenate((depth_debris, vacs[axis]))

    # Depth binning (use debris to set bin edges)
    depth_edges = np.histogram_bin_edges(depth_debris, bins=nbins)
    depth_centers = (depth_edges[1:] - depth_edges[:-1]) / 2.0 + depth_edges[:-1]
    width = depth_edges[1] - depth_edges[0]
    srim_digitize = np.digitize(depths, depth_edges)

    # NRT-dpa
    hist_nrt = np.array(
        [np.sum(defects_nrt[srim_digitize == i]) for i in range(1, nbins + 1)]
    )
    dpa_nrt = hist_nrt / nions * fluence / mat_target.density / width
    # fer-arc-dpa
    hist_fer_arc = np.array(
        [np.sum(defects_fer_arc[srim_digitize == i]) for i in range(1, nbins + 1)]
    )
    dpa_fer_arc = hist_fer_arc / nions * fluence / mat_target.density / width
    # debris-dpa
    hist_debris, _ = np.histogram(depth_debris, bins=depth_edges)
    dpa_debris = hist_debris / nions * fluence / mat_target.density / width

    # Save to database
    utils.sqlite.insert_array(
        path_db,
        f"dpa_1D_{axis}",
        depth_centers=depth_centers + depth_offset,
        dpa_nrt=dpa_nrt,
        dpa_fer_arc=dpa_fer_arc,
        dpa_debris=dpa_debris,
    )


def read_dpa_1d(path_db: Path, axis: str = "x") -> dict[str, npt.NDArray[np.float64]]:
    """Read the 1D dpa histogram from the database.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`, or `"z"`.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        A dictionary containing the 1D dpa histogram data.
    """
    data = utils.sqlite.read_array(path_db, f"dpa_1D_{axis}")
    return data


# region Plots


def plot_dpa_1d(
    path_db: Path,
    path_plot: None | Path = None,
    path_fit: None | Path = None,
    axis: str = "x",
    depth_offset: float = 0.0,
    p0: None | float = None,
    asymmetry: float = 1.0,
    dpi: int = 300,
) -> None:
    """Plot the 1D dpa analysis results from the database.

    Parameters
    ----------
    path_db : Path
        Path to the database file.
    path_plot : Path, optional (default=None)
        Output path for the plot. If None, the plot is shown.
    path_fit : Path, optional (default=None)
        Output path for the fit parameters.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`, or `"z"`.
    depth_offset : float, optional (default=0.0)
        Offset to add to the depth values.
    p0 : float, optional (default=None)
        Initial guess of fit parameters.
    asymmetry : float, optional (default=1.0)
        Asymmetry fit parameter bound.
    dpi : int, optional (default=300)
        Dots per inch.
    """

    data = read_dpa_1d(path_db, axis=axis)
    depth_centers = data["depth_centers"] + depth_offset
    dpa_nrt = data["dpa_nrt"]
    dpa_fer_arc = data["dpa_fer_arc"]
    dpa_debris = data["dpa_debris"]

    total_nrt = dpa_nrt.sum()
    total_debris = dpa_debris.sum()

    # Fit dpa_debris
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
    efficiency = [
        Line2D(
            [0],
            [0],
            color="none",
            label=r"$\overline{\xi}$ = "
            + rf"{round(total_debris / total_nrt * 100)} %",
        )
    ]

    if path_fit:
        try:
            popt, _, dpa_fit = utils.math.fit_lorentzian(
                depth_centers, dpa_debris, p0, asymmetry
            )
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
            ax.plot(
                depth_centers,
                dpa_fit(depth_centers),
                label="debris-dpa fit",
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Fit failed: {exc}")

    ax.legend(handles=ax.get_legend_handles_labels()[0] + efficiency)
    fig.tight_layout()
    if path_plot:
        plt.savefig(path_plot, dpi=dpi)
    else:
        plt.show()
    plt.close()
