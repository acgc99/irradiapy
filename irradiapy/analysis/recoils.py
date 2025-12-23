"""Module for analyzing recoils and final ion positions."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LogNorm
from matplotlib.image import NonUniformImage

from irradiapy.analysis.analysisdb import AnalysisDB
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.utils.math import fit_gaussian, fit_power_law


def recoil_energies_hist(
    recoilsdb: RecoilsDB,
    analysisdb: AnalysisDB,
    nbins: int = 100,
    conditions: str = "",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate and store the recoil energy histogram and tries to fit it.

    Fit: power law function.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    analysisdb : AnalysisDB
        Database for storing results.
    nbins : int, optional (default=100)
        Number of recoil energy nbins.
    conditions : str, optional (default="")
        Conditions to filter recoils.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (energy_centers, hist)
    """
    nevents = recoilsdb.get_nevents()
    energies = recoilsdb.read_numpy(
        table="recoils",
        what="recoil_energy",
        conditions=conditions,
    )["recoil_energy"]
    hist, energy_edges = np.histogram(energies, bins=nbins)
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0
    hist = hist / nevents
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0
    analysisdb.save_recoil_energies_hist(energy_centers=energy_centers, hist=hist)

    try:
        params, err_params, _ = fit_power_law(energy_centers, hist)
        analysisdb.save_recoil_energies_hist_fit_params(*params)
        analysisdb.save_recoil_energies_hist_fit_errors(*err_params)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fit failed: {exc}")

    return energy_centers, hist


def recoil_energies_hist_plot(
    analysisdb: AnalysisDB,
    fit: bool = False,
    show: bool = False,
    plot_path: None | Path = None,
    dpi: int = 300,
) -> None:
    """Plot the recoil energy histogram.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database. Must contain recoil energies histogram data.
    fit : bool, optional (default=False)
        Whether to plot the fit function.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch for the plot.
    """
    energy_centers, hist = analysisdb.load_recoil_energies_hist()

    fig = plt.figure()
    gs = fig.add_gridspec()
    ax = fig.add_subplot(gs[0, 0])

    ax.set_xlabel(r"$E_{recoil}$ (eV)")
    ax.set_ylabel("Counts per event")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(energy_centers, hist, marker=".")

    if fit:
        try:
            fit_function = analysisdb.load_recoil_energies_hist_fit_function()
            ax.plot(
                energy_centers,
                fit_function(energy_centers),
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            )
        except sqlite3.OperationalError as exc:
            print(f"Could not plot fit function: {exc}")

    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def recoils_distances_hist_plot(
    recoilsdb: RecoilsDB,
    dist_nbins: int = 100,
    energy_nbins: int = 100,
    min_recoil_energy: float = 5e3,
    global_min_counts: int = 1,
    log: bool = False,
    show: bool = False,
    plot_path: None | Path = None,
    dpi: int = 300,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Plot the recoils pairwise distances vs sum of recoil energies histogram.

    For each event, calculates the pairwise distances between all recoils with
    recoil energy >= `min_recoil_energy`, and the sum of their recoil energies.
    The histogram shows the counts of pairs per event as a function of distance and
    sum of recoil energies.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    dist_nbins : int, optional (default=100)
        Number of nbins for the distance histogram.
    energy_nbins : int, optional (default=100)
        Number of nbins for the energy histogram.
    min_recoil_energy : float, optional (default=5e3)
        Minimum recoil energy to consider, in eV. If set too low, the amount of pairs is so large
        that the calculation may fail due to lack of memory.
    global_min_counts : int, optional (default=1)
        Minimum counts to show in the plot.
    log: bool, optional (default=False)
        Whether to use a logarithmic scale for the color map.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch for the plot.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (histogram, distance_edges, energy_edges). Rows correspond to distance nbins,
        columns to sum of recoil energy nbins.
    """
    nevents = recoilsdb.get_nevents()
    data = recoilsdb.read_numpy(
        table="recoils",
        what="event, x, y, z, recoil_energy",
        conditions=f"WHERE recoil_energy >= {min_recoil_energy} ORDER BY event",
    )
    events = data["event"]
    positions = np.column_stack((data["x"], data["y"], data["z"]))
    recoil_energies = data["recoil_energy"]

    # Indices where event changes
    cuts = np.flatnonzero(events[1:] != events[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, events.size]

    distances = []
    energies = []
    for start, end in zip(starts, ends):
        nrecoils = end - start
        if nrecoils < 2:
            continue
        event_positions = positions[start:end]
        event_recoil_energies = recoil_energies[start:end]
        ii, jj = np.triu_indices(nrecoils, k=1)
        distances.append(
            np.linalg.norm(event_positions[ii] - event_positions[jj], axis=1)
        )
        energies.append(event_recoil_energies[ii] + event_recoil_energies[jj])

    distances = np.concatenate(distances)
    energies = np.concatenate(energies) / 1e3

    dist_edges = np.histogram_bin_edges(distances, bins=dist_nbins)
    energy_edges = np.histogram_bin_edges(energies, bins=energy_nbins)
    hist, _, _ = np.histogram2d(distances, energies, bins=[dist_edges, energy_edges])
    hist /= nevents
    dist_centers = (dist_edges[1:] + dist_edges[:-1]) / 2.0
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0

    hist_plot = np.ma.masked_less(hist.T, global_min_counts / nevents)
    norm = LogNorm() if log else None

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
    cmap = plt.get_cmap("viridis")
    cmap.set_under(plt.rcParams["axes.facecolor"])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Pairwise distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Sum of $E_{recoil}$ (keV)")
    ax.set_ylim(energy_edges[[0, -1]])
    ax.set_xlim(dist_edges[[0, -1]])
    im = NonUniformImage(
        ax,
        cmap=cmap,
        extent=(*dist_edges[[0, -1]], *energy_edges[[0, -1]]),
        norm=norm,
    )
    im.set_clim(vmin=1 / nevents)
    im.set_data(dist_centers, energy_centers, hist_plot)
    ax.add_image(im)

    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax, label="Counts per event")

    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    return hist, dist_edges, energy_edges


def depth_recoil_energy_hist_plot(
    recoilsdb: RecoilsDB,
    axis: str,
    depth_nbins: int = 100,
    energy_nbins: int = 100,
    max_recoil_energy: float | None = None,
    log: bool = False,
    kev: bool = False,
    plot_path: None | Path = None,
    show: bool = False,
    dpi: int = 300,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Plot the recoil energy vs depth histogram.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    axis : str
        Axis along which to calculate depth ('x', 'y', or 'z').
    depth_nbins : int, optional (default=100)
        Number of nbins for depth histogram.
    energy_nbins : int, optional (default=100)
        Number of nbins for recoil energy histogram.
    max_recoil_energy : float | None, optional (default=None)
        Maximum recoil energy to consider, in eV. If `None`, all recoil energies are considered.
    log : bool, optional (default=False)
        Whether to use a logarithmic scale for the color map.
    kev : bool, optional (default=False)
        Whether to plot recoil energies in keV instead of eV.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    show : bool, optional (default=False)
        Whether to show the plot.
    dpi : int, optional (default=300)
        Dots per inch for the plot.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (depth_edges, energy_edges, hist).
        Rows correspond to depth nbins, columns to recoil energy nbins.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    nevents = recoilsdb.get_nevents()
    conditions = (
        f"WHERE recoil_energy <= {max_recoil_energy}"
        if max_recoil_energy is not None
        else ""
    )
    data = recoilsdb.read_numpy(
        table="recoils",
        what=f"{axis}, recoil_energy",
        conditions=conditions,
    )
    depths = data[axis]
    energies = data["recoil_energy"]
    if kev:
        energies /= 1e3

    depth_edges = np.histogram_bin_edges(depths, bins=depth_nbins)
    energy_edges = np.histogram_bin_edges(energies, bins=energy_nbins)
    hist, _, _ = np.histogram2d(
        depths,
        energies,
        bins=[depth_edges, energy_edges],
    )
    hist /= nevents
    depth_centers = (depth_edges[1:] + depth_edges[:-1]) / 2.0
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
    cmap = plt.get_cmap("viridis")
    cmap.set_under(plt.rcParams["axes.facecolor"])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_xlabel(r"$E_{recoil}$ (keV)" if kev else r"$E_{recoil}$ (eV)")
    ax.set_xlim(energy_edges[[0, -1]])
    ax.set_ylim(depth_edges[[0, -1]])
    im = NonUniformImage(
        ax,
        cmap=cmap,
        extent=(*energy_edges[[0, -1]], *depth_edges[[0, -1]]),
        norm=LogNorm() if log else None,
    )
    im.set_clim(vmin=1.0 / nevents)
    im.set_data(energy_centers, depth_centers, hist)
    ax.add_image(im)

    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax, label="Counts per event")

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    return depth_edges, energy_edges, hist


def depth_injected_ions_hist(
    recoilsdb: RecoilsDB,
    analysisdb: AnalysisDB,
    axis: str,
    nbins: int = 100,
    p0: None | float = None,
    asymmetry: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate and store the injected ions final depth histogram and tries to fit it.

    Fit: asymmetric gaussian function.
    Tries to fit to Eq. (1) of
    Nuclear Instruments and Methods in Physics Research B 500-501 (2021) 52-56.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    analysisdb : AnalysisDB
        Database for storing results.
    axis : str
        Axis along which to calculate depth ('x', 'y', or 'z').
    nbins : int, optional (default=100)
        Number of depth nbins.
    p0 : float | None, optional (default=None)
        Initial guess for Gaussian fit.
    asymmetry : float, optional (default=1.0)
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (depth_centers, hist)
    """

    if axis not in ("x", "y", "z"):
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    nevents = recoilsdb.get_nevents()
    depths = np.empty(nevents)
    for event in range(1, nevents + 1):
        ions_vacs = next(
            recoilsdb.read(
                table="ions_vacs",
                what=axis,
                conditions=f"WHERE event = {event} LIMIT 1 OFFSET 1",
            )
        )
        # This takes the second row, which is the rest position of
        # the primary recoil atom after all recoils
        depths[event - 1] = ions_vacs[0]

    hist, depth_edges = np.histogram(depths, bins=nbins)
    hist = hist / nevents
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0
    analysisdb.save_depth_ions_hist(axis=axis, depth_centers=depth_centers, hist=hist)

    try:
        params, err_params, _ = fit_gaussian(depth_centers, hist, p0, asymmetry)
        analysisdb.save_depth_ions_hist_fit_params(axis, *params)
        analysisdb.save_depth_ions_hist_fit_errors(axis, *err_params)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fit failed: {exc}")

    return depth_centers, hist


def depth_injected_ions_hist_plot(
    analysisdb: AnalysisDB,
    axis: str,
    fit: bool = False,
    show: bool = False,
    plot_path: None | Path = None,
    dpi: int = 300,
) -> None:
    """Plot the injected ions final depth histogram.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database. Must contain depth ions histogram data.
    axis : str
        Axis along which to plot depth ('x', 'y', or 'z').
    fit : bool, optional (default=False)
        Whether to plot the fit function.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch for the plot.
    """
    depth_centers, hist = analysisdb.load_depth_ions_hist(axis)

    fig = plt.figure()
    gs = fig.add_gridspec()
    ax = fig.add_subplot(gs[0, 0])

    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("Counts per event")
    ax.scatter(depth_centers, hist, marker=".")
    if fit:
        try:
            fit_function = analysisdb.load_depth_ions_hist_fit_function(axis)
            ax.plot(
                depth_centers,
                fit_function(depth_centers),
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            )
        except sqlite3.OperationalError as exc:
            print(f"Could not plot fit function: {exc}")

    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()
