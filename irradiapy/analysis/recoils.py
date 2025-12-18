"""Module for analyzing recoils."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LogNorm
from matplotlib.image import NonUniformImage

from irradiapy.recoilsdb import RecoilsDB
from irradiapy.utils.math import fit_gaussian, fit_power_law


def recoil_energies(
    recoilsdb: RecoilsDB,
    nbins: int = 100,
    condition: str = "",
    show: bool = False,
    plot_path: None | Path = None,
    dpi: int = 300,
    fit_path: None | Path = None,
) -> dict[str, any]:
    """Plot the recoil energy distribution and tries to fit it.

    The fit is done using a power law of the form: a * x ** k.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    nbins : int, optional (default=100)
        Number of recoil energy nbins. The fit will be done over non-empty nbins.
    condition : str, optional (default="")
        Condition to filter recoils.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch.
    fit_path : Path, optional (default=None)
        Output path for the fit parameters.

    Returns
    -------
    dict[str, any]
        ((a, k), (error_a, error_k), fit_function) or None if fit failed.
    """
    return_values = {}

    nevents = recoilsdb.get_nevents()
    energies = np.array(
        [
            recoil[0]
            for recoil in recoilsdb.read(
                table="recoils",
                what="recoil_energy",
                condition=condition,
            )
        ]
    )

    # Histogram
    hist, energy_edges = np.histogram(energies, bins=nbins)
    hist = hist / nevents
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0
    return_values["hist"] = hist
    return_values["energy_edges"] = energy_edges

    # Fit using fit_scaling_law
    if fit_path:
        try:
            params, err_params, fit_function = fit_power_law(energy_centers, hist)
            return_values["params"] = params
            return_values["err_params"] = err_params
            return_values["fit_function"] = fit_function

            file = open(fit_path, "w", encoding="utf-8")
            file.write("Recoil energy scaling law\n")
            file.write("E in eV\n")
            file.write("a * E ** k\n")
            file.write(f"a = {params[0]} ± {err_params[0]}\n")
            file.write(f"k = {params[1]} ± {err_params[1]}\n")
            file.close()

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Fit failed: {exc}")

    # Plot
    if show or plot_path:
        fig = plt.figure()
        gs = fig.add_gridspec()
        ax = fig.add_subplot(gs[0, 0])

        # Scatter
        ax.set_xlabel(r"$E_{recoil}$ (eV)")
        ax.set_ylabel("Counts per event")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(energy_centers, hist, marker=".")

        # Fit
        if "fit_function" in return_values:
            ax.plot(
                energy_centers,
                fit_function(energy_centers),
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            )

        # Finish
        fig.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    return return_values


def plot_recoil_energy_depth(
    recoilsdb: RecoilsDB,
    axis: str = "x",
    depth_nbins: int = 100,
    energy_nbins: int = 100,
    max_recoil_energy: float | None = None,
    plot_path: None | Path = None,
    show: bool = False,
    log: bool = False,
    kev: bool = False,
    dpi: int = 300,
) -> dict[str, npt.NDArray[np.float64]]:
    """Plots the depth-energy distribution of recoil energies.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    depth_nbins : int, optional (default=100)
        Number of nbins for depth histogram.
    energy_nbins : int, optional (default=100)
        Number of nbins for recoil energy histogram.
    condition : str, optional (default="")
        Condition to filter recoils.
    plot_path : Path, optional (default=None)
        Output path for the plot. If `None`, the plot is shown.
    log : bool, optional (default=False)
        Whether to use a logarithmic scale for the color map.
    dpi : int, optional (default=300)
        Dots per inch for the plot.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        (histogram, depth_edges, energy_edges). Rows correspond to depth nbins,
        columns to recoil energy nbins.
    """
    return_values = {}

    what = ""
    if axis == "x":
        what = "depth, recoil_energy"
    elif axis == "y":
        what = "y, recoil_energy"
    elif axis == "z":
        what = "z, recoil_energy"
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    nevents = recoilsdb.get_nevents()
    data = np.array(
        list(
            recoilsdb.read(
                table="recoils",
                what=what,
                condition=(
                    f"WHERE recoil_energy <= {max_recoil_energy}"
                    if max_recoil_energy is not None
                    else ""
                ),
            )
        ),
    )
    depths, energies = data[:, 0], data[:, 1]
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
    return_values["hist"] = hist
    return_values["depth_edges"] = depth_edges
    return_values["energy_edges"] = energy_edges

    # Plot
    if show or plot_path:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
        cmap = plt.get_cmap("viridis")
        cmap.set_under(plt.rcParams["axes.facecolor"])

        # Color map
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

        # Color bar
        cax = fig.add_subplot(gs[0, 1])
        fig.colorbar(im, cax, label="Counts per event")

        # Finish
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    return return_values


def plot_recoils_distances(
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
    """Plots a 2D histogram of pairwise distances and sum of recoil energies.

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
    distances = []
    energies = []
    # Get pairwise distances and energies for each ion
    for nevent in range(1, nevents + 1):
        data = np.array(
            list(
                recoilsdb.read(
                    table="recoils",
                    what="depth, y, z, recoil_energy",
                    condition=f"WHERE event = {nevent} AND recoil_energy >= {min_recoil_energy}",
                )
            )
        )
        if len(data):
            pos = data[:, :3]
            recoil_energy = data[:, 3]
            nrecoils = pos.shape[0]
            if nrecoils >= 2:
                ii, jj = np.triu_indices(nrecoils, k=1)
                distance = np.linalg.norm(pos[ii] - pos[jj], axis=1)
                energy = recoil_energy[ii] + recoil_energy[jj]
                distances.append(distance)
                energies.append(energy)

    distances = np.concatenate(distances)
    energies = np.concatenate(energies) / 1e3

    # Histogram
    dist_edges = np.histogram_bin_edges(distances, bins=dist_nbins)
    energy_edges = np.histogram_bin_edges(energies, bins=energy_nbins)
    hist, _, _ = np.histogram2d(distances, energies, bins=[dist_edges, energy_edges])
    hist /= nevents
    dist_centers = (dist_edges[1:] + dist_edges[:-1]) / 2.0
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2.0

    # Plot
    if show or plot_path:
        hist_plot = np.ma.masked_less(hist.T, global_min_counts / nevents)
        norm = LogNorm() if log else None

        fig = plt.figure()
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
        cmap = plt.get_cmap("viridis")
        cmap.set_under(plt.rcParams["axes.facecolor"])

        # Color map
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

        # Color bar
        cax = fig.add_subplot(gs[0, 1])
        fig.colorbar(im, cax, label="Counts per event")

        # Finish
        fig.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    return hist, dist_edges, energy_edges


def plot_injected_ions(
    recoilsdb: RecoilsDB,
    axis: str = "x",
    nbins: int = 100,
    p0: None | float = None,
    asymmetry: float = 1.0,
    show: bool = False,
    plot_path: None | Path = None,
    dpi: int = 300,
    path_fit: None | Path = None,
) -> dict[str, any]:
    """Plot injected ions final depth distribution.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    nbins : int, optional (default=100)
        Depth nbins.
    p0 : float, optional (default=None)
        Initial guess of fit parameters.
    asymmetry : float, optional (default=1.0)
        Asymmetry fit parameter bound.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch.
    path_fit : Path, optional (default=None)
        Output path for the fit parameters.

    Returns
    -------
    dict[str, any]
        Dictionary with depth edges, histogram, and fit parameters and function if fit
        was successful.
    """
    return_values = {}

    if axis == "x":
        what = "depth"
    elif axis == "y":
        what = "y"
    elif axis == "z":
        what = "z"
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    nevents = recoilsdb.get_nevents()
    depths = np.empty(nevents)
    for event in range(1, nevents + 1):
        ions_vacs = list(
            recoilsdb.read(
                table="ions_vacs",
                what=what,
                condition=f"WHERE event = {event}",
            )
        )
        # This takes the second row, which is the rest position of
        # the primary recoil atom after all recoils
        depths[event - 1] = ions_vacs[1][0]

    # Histogram
    hist, depth_edges = np.histogram(depths, bins=nbins)
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0
    return_values["hist"] = hist
    return_values["depth_edges"] = depth_edges

    # Fit
    if path_fit:
        try:
            params, err_params, fit_function = fit_gaussian(
                depth_centers, hist, p0, asymmetry
            )
            return_values["params"] = params
            return_values["err_params"] = err_params
            return_values["fit_function"] = fit_function

            file = open(path_fit, "w", encoding="utf-8")
            file.write(
                (
                    "Injected atoms gaussian fit. See Eq. (1) of "
                    "Nuclear Instruments and Methods in Physics Research B "
                    "500-501 (2021) 52-56\n"
                )
            )
            file.write("x in angstroms\n")
            file.write(
                (
                    "A * (1 + e ** (x - x0)) / 4 * "
                    "e ** (- (1 + e ** a (x - x0)) ** 2 / 4 * "
                    "(x - x0) ** 2 / (2 * sigma ** 2))\n"
                )
            )
            file.write(f"x0 = {params[0]} ± {err_params[0]}\n")
            file.write(f"sigma = {params[1]} ± {err_params[1]}\n")
            file.write(f"A = {params[2]} ± {err_params[2]}\n")
            file.write(f"a = {params[3]} ± {err_params[3]}\n")
            file.close()

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Fit failed: {exc}")

    # Plot
    if show or plot_path:
        fig = plt.figure()
        gs = fig.add_gridspec()
        ax = fig.add_subplot(gs[0, 0])

        # Scatter
        ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
        ax.set_ylabel("Counts per event")
        ax.scatter(depth_centers, hist, marker=".")
        if "fit_function" in return_values:
            ax.plot(
                depth_centers,
                fit_function(depth_centers),
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            )

        # Finish
        fig.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    return return_values
