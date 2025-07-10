"""Utility functions related to PKA for SRIM data."""

from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from irradiapy.srimpy.srimdb import SRIMDB
from irradiapy.utils.math import fit_scaling_law


def plot_pka_distribution(
    srimdb: SRIMDB,
    bins: int = 100,
    plot_path: Optional[Path] = None,
    dpi: int = 300,
    fit_path: Optional[Path] = None,
) -> Callable:
    """Plot the PKA energy distribution and tries to fit it.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    bins : int, optional
        Energy bins, by default 100. The fit will be done over non-empty bins.
    plot_path : Path, optional
        Output path for the plot, by default None. If None, the plot is shown.
    dpi : int, optional
        Dots per inch for the plot, by default 300.
    fit_path : Path, optional
        Output path for the fit parameters, by default None.

    Returns
    -------
    Callable
        Scaling law function.
    """
    # Read
    nions = srimdb.nions
    pka_es = np.array(list(srimdb.collision.read(what="recoil_energy")))
    # Histogram
    pka_hist, pka_edges = np.histogram(pka_es, bins=bins)
    pka_hist = pka_hist / nions
    pka_centers = pka_edges[:-1] + (pka_edges[1:] - pka_edges[:-1]) / 2.0
    # Fit using fit_scaling_law
    fit = False
    try:
        mask = pka_hist > 0
        a, s, curve = fit_scaling_law(pka_centers[mask] / 1e3, pka_hist[mask])
        fit = True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fit failed: {exc}")
    if fit and fit_path:
        with open(fit_path, "w", encoding="utf-8") as file:
            file.write(f"PKA energy scaling law (x in eV)\nA/x**S\nA S\n{a}, {s}\n")
    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec()
    # Scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"$E_{PKA}$ (keV)")
    ax.set_ylabel("Counts per ion")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(pka_centers / 1e3, pka_hist)
    if fit:
        ax.plot(
            pka_centers / 1e3,
            curve(pka_centers / 1e3),
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
    # Finish
    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    else:
        plt.show()
    plt.close()
    return curve


def plot_energy_depth(
    srimdb: SRIMDB,
    depth_bins: int = 100,
    pka_ebins: int = 100,
    pka_e_max: float = 200,
    plot_high_path: Optional[Path] = None,
    plot_low_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plots the depth-energy distribution of PKAs.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    depth_bins : int, optional
        Number of bins for depth histogram, by default 100.
    pka_ebins : int, optional
        Number of bins for PKA energy histogram, by default 100.
    pka_e_max : float, optional
        Maximum PKA energy, by default 200.
    plot_high_path : Path, optional
        Output path for the high energy plot, by default None. If None, the plot is shown.
    plot_low_path : Path, optional
        Output path for the low energy plot, by default None. If None, the plot is shown.
    dpi : int, optional
        Dots per inch for the plot, by default 300.
    """
    # Read
    nions = srimdb.nions
    data = np.array(list(srimdb.collision.read(what="depth, recoil_energy")))
    depths, pka_es = data[:, 0], data[:, 1]
    # Low energy, linear
    depth_edges = np.histogram_bin_edges(depths, bins=depth_bins)
    pka_e_edges = np.histogram_bin_edges(
        pka_es, bins=pka_ebins, range=(pka_es.min(), pka_e_max)
    )
    hist, _, _ = np.histogram2d(depths, pka_es, bins=[depth_edges, pka_e_edges])
    hist /= nions
    depth_centers = depth_edges[:-1] + (depth_edges[1:] - depth_edges[:-1]) / 2.0
    pka_e_centers = pka_e_edges[:-1] + (pka_e_edges[1:] - pka_e_edges[:-1]) / 2.0
    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
    cmap = plt.cm.get_cmap("viridis")
    cmap.set_under(plt.rcParams["axes.facecolor"])
    # Color map
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_xlabel(r"$E_{PKA}$ (eV) (low energies)")
    ax.set_xlim(pka_e_edges[[0, -1]])
    ax.set_ylim(depth_edges[[0, -1]])
    im = NonUniformImage(
        ax, cmap=cmap, extent=(*pka_e_edges[[0, -1]], *depth_edges[[0, -1]])
    )
    im.set_clim(vmin=1 / nions)
    im.set_data(pka_e_centers, depth_centers, hist)
    ax.add_image(im)
    # Color bar
    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax, label="Counts per ion")
    plt.tight_layout()
    if plot_low_path:
        plt.savefig(plot_low_path, dpi=dpi)
    else:
        plt.show()
    plt.close()
    # All energies, log
    depth_edges = np.histogram_bin_edges(depths, bins=depth_bins)
    pka_e_edges = np.histogram_bin_edges(pka_es, bins=pka_ebins)
    hist, _, _ = np.histogram2d(depths, pka_es, bins=[depth_edges, pka_e_edges])
    hist /= nions
    depth_centers = depth_edges[:-1] + (depth_edges[1:] - depth_edges[:-1]) / 2.0
    pka_e_centers = pka_e_edges[:-1] + (pka_e_edges[1:] - pka_e_edges[:-1]) / 2.0
    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
    # Color map
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_xlabel(r"$E_{PKA}$ (keV)")
    ax.set_xlim(pka_e_edges[[0, -1]] / 1e3)
    ax.set_ylim(depth_edges[[0, -1]])
    im = NonUniformImage(
        ax,
        cmap=cmap,
        norm="log",
        extent=(*pka_e_edges[[0, -1]] / 1e3, *depth_edges[[0, -1]]),
    )
    im.set_clim(vmin=1 / nions)
    im.set_data(pka_e_centers / 1e3, depth_centers, hist)
    ax.add_image(im)
    # Color bar
    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax, label="Counts per ion")
    # Finish
    plt.tight_layout()
    if plot_high_path:
        plt.savefig(plot_high_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_distances(
    srimdb: SRIMDB,
    pka_e_lim: float = 5e3,
    dist_bins: int = 100,
    energy_bins: int = 100,
    plot_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plots a 2D histogram of pairwise distances and sum of PKA energies for each ion.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    pka_elim : float, optional
        Minimum recoil energy to consider, by default 5e3.
    dist_bins : int, optional
        Number of bins for the distance histogram, by default 100.
    energy_bins : int, optional
        Number of bins for the energy histogram, by default 100.
    plot_path : Path, optional
        Output path for the plot, by default None. If None, the plot is shown.
    dpi : int, optional
        Dots per inch for the plot, by default 300.
    """
    nions = srimdb.nions
    distances = []
    energies = []
    # Get pairwise distances and energies for each ion
    for nion in range(1, nions + 1):
        data = np.array(
            list(
                srimdb.collision.read(
                    what="depth, y, z, recoil_energy",
                    condition=f"WHERE ion_numb = {nion} AND recoil_energy >= {pka_e_lim}",
                )
            )
        )
        if len(data):
            pos = data[:, :3]
            pka_e = data[:, 3]
            for i, j in combinations(range(len(pos)), 2):
                distance = np.linalg.norm(pos[i] - pos[j])
                energy = pka_e[i] + pka_e[j]
                distances.append(distance)
                energies.append(energy)
    distances = np.array(distances)
    energies = np.array(energies) / 1e3
    # Histogram
    dist_edges = np.histogram_bin_edges(distances, bins=dist_bins)
    energies_edges = np.histogram_bin_edges(energies, bins=energy_bins)
    hist, _, _ = np.histogram2d(distances, energies, bins=[dist_edges, energies_edges])
    hist /= nions
    dist_centers = dist_edges[:-1] + (dist_edges[1:] - dist_edges[:-1]) / 2.0
    energies_centers = (
        energies_edges[:-1] + (energies_edges[1:] - energies_edges[:-1]) / 2.0
    )
    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 0.05])
    cmap = plt.cm.get_cmap("viridis")
    cmap.set_under(plt.rcParams["axes.facecolor"])
    # Color map
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Pairwise distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Sum of $E_{PKA}$ (keV)")
    ax.set_ylim(energies_edges[[0, -1]])
    ax.set_xlim(dist_edges[[0, -1]])
    im = NonUniformImage(
        ax,
        cmap=cmap,
        extent=(*dist_edges[[0, -1]], *energies_edges[[0, -1]]),
    )
    im.set_clim(vmin=1 / nions)
    im.set_data(dist_centers, energies_centers, hist)
    ax.add_image(im)
    # Color bar
    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax, label="Counts per ion")
    # Finish
    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    else:
        plt.show()
    plt.close()
