"""Utility functions related to SRIM data analysis and debris production."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from irradiapy import dtypes, materials
from irradiapy.damagedb import DamageDB
from irradiapy.io.lammpswriter import LAMMPSWriter
from irradiapy.srimpy.srimdb import SRIMDB
from irradiapy.utils.math import fit_gaussian, fit_scaling_law

# region range3d


def plot_injected(
    srimdb: SRIMDB,
    bins: int = 100,
    plot_path: Optional[Path] = None,
    dpi: int = 300,
    path_fit: Optional[Path] = None,
    p0: Optional[float] = None,
    asymmetry: float = 1.0,
) -> None:
    """Plot injected ions final depth distribution.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    bins : int, optional
        Depth bins, by default 100.
    plot_path : Path, optional
        Output path for the plot, by default None. If None, the plot is shown.
    dpi : int, optional
        Dots per inch, by default 300.
    path_fit : Path, optional
        Output path for the fit parameters, by default None.
    p0 : float, optional
        Initial guess of fit parameters, by default None.
    asymmetry : float, optional
        Asymmetry fit parameter bound, by default 1.0.
    """
    # Read
    depths = np.array([ion[0] for ion in srimdb.range3d.read(what="depth")])
    # Histogram
    histogram, depth_edges = np.histogram(depths, bins=bins)
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0
    # Fit
    fit, injected_fit = False, None
    if path_fit:
        try:
            popt, _, injected_fit = fit_gaussian(
                depth_centers, histogram, p0, asymmetry
            )
            if path_fit:
                with open(path_fit, "w", encoding="utf-8") as file:
                    file.write("Injected atoms gaussian fit: z0, sigma, A, a\n")
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
    # Scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("Counts per ion")
    ax.scatter(depth_centers, histogram)
    if fit:
        ax.plot(
            depth_centers,
            injected_fit(depth_centers),
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
    # Finish
    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


# region collision


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


# region Debris


def generate_debris(
    srimdb: SRIMDB,
    dir_mddb: Path,
    compute_tdam: bool,
    path_collisions: Path,
    tdam_mode: "materials.Material.TdamMode",
    dpa_mode: "materials.Material.DpaMode",
    add_injected: bool,
    outsiders: bool,
    seed: Optional[int] = None,
    depth_offset: Optional[float] = 0.0,
    ylo: Optional[float] = None,
    yhi: Optional[float] = None,
    zlo: Optional[float] = None,
    zhi: Optional[float] = None,
) -> None:
    """Turns SRIM's collisions into `.xyz` files for the given database of cascades' debris.

    Warning
    -------
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
    path_collisions : Path
        Directory where the ions debris will be stored as `.xyz` files.
    tdam_mode : materials.Material.TdamMode
        Mode to convert the PKA energy into damage energy.
    dpa_mode : materials.Material.DpaMode
        Formula to convert the residual energy into Frenkel pairs.
    add_injected : bool
        Whether to add the injected interstitial.
    outsiders : bool
        Whether to include defects generated outside the material (PKAs close to surfaces).
        This can cause an inmbalance between the number of vacancies and interstitials.
    seed : int, optional
        Random seed for reproducibility.
    depth_offset : float, optional
        Offset to add to the depth of the defects, by default 0.0.
    ylo : float, optional
        Minimum y-coordinate for the defects, by default None
        (will be set to minus the target width).
    yhi : float, optional
        Maximum y-coordinate for the defects, by default None
        (will be set to the target width).
    zlo : float, optional
        Minimum z-coordinate for the defects, by default None
        (will be set to minus the target width).
    zhi : float, optional
        Maximum z-coordinate for the defects, by default None
        (will be set to the target width).
    """
    xlo, xhi = depth_offset, srimdb.target.layers[0].width + depth_offset
    if ylo is None:
        ylo = -xhi
    if yhi is None:
        yhi = xhi
    if zlo is None:
        zlo = -xhi
    if zhi is None:
        zhi = xhi
    pka_atomic_number = next(
        iter(srimdb.trimdat.read(what="atom_numb", condition="WHERE ion_numb = 1"))
    )[0]
    mat_pka = materials.MATERIALS_BY_ATOMIC_NUMBER[pka_atomic_number]
    mat_atomic_number = srimdb.target.layers[0].elements[0].atomic_number
    mat_target = materials.MATERIALS_BY_ATOMIC_NUMBER[mat_atomic_number]
    nions = srimdb.nions
    damagedb = DamageDB(
        dir_mddb=dir_mddb,
        compute_tdam=compute_tdam,
        mat_pka=mat_pka,
        mat_target=mat_target,
        tdam_mode=tdam_mode,
        dpa_mode=dpa_mode,
        seed=seed,
    )
    with LAMMPSWriter(path_collisions) as writer:
        for nion in range(1, nions + 1):
            defects = __generate_ion_defects(srimdb, nion, damagedb, add_injected)

            # Apply offsets and cuts
            defects["x"] += depth_offset
            if not outsiders:
                defects = defects[
                    (defects["x"] >= xlo)
                    & (defects["x"] <= xhi)
                    & (defects["y"] >= ylo)
                    & (defects["y"] <= yhi)
                    & (defects["z"] >= zlo)
                    & (defects["z"] <= zhi)
                ]

            data_defects = defaultdict(None)
            data_defects["time"] = 0.0
            data_defects["timestep"] = 0
            data_defects["natoms"] = defects.size
            data_defects["boundary"] = ["ff", "ff", "ff"]
            data_defects["xlo"] = xlo
            data_defects["xhi"] = xhi
            data_defects["ylo"] = ylo
            data_defects["yhi"] = yhi
            data_defects["zlo"] = zlo
            data_defects["zhi"] = zhi
            data_defects["atoms"] = defects
            writer.write(data_defects)


def __generate_ion_defects(
    srimdb: SRIMDB,
    nion: int,
    damagedb: DamageDB,
    add_injected: bool,
) -> np.ndarray:
    """Generates the defects for a specific ion in the SRIM simulation.

    Parameters
    ----------
    srimdb : SRIMDB
        SRIM database class.
    nion : int
        Ion number.
    damagedb : DamageDB
        DamageDB class that will choose MD debris.
    add_injected : bool
        Whether to add the injected interstitial.

    Returns
    -------
    np.ndarray
        An array containing the defects generated by a single ion.
    """
    defects = np.empty(0, dtype=dtypes.defect)
    for depth, y, z, cosx, cosy, cosz, pka_e in srimdb.collision.read(
        what="depth, y, z, cosx, cosy, cosz, recoil_energy",
        condition=f"WHERE ion_numb = {nion}",
    ):
        pka_pos = np.array([depth, y, z])
        pka_dir = np.array([cosx, cosy, cosz])
        defects_ = damagedb.get_pka_debris(
            pka_e=pka_e, pka_pos=pka_pos, pka_dir=pka_dir
        )
        defects = np.concatenate((defects, defects_))

    if add_injected:
        injected = list(
            srimdb.range3d.read(
                what="depth, y, z", condition=f"WHERE ion_numb = {nion} LIMIT 1"
            )
        )
        # No backscattered or transmitted ion
        if injected:
            atom_number = list(
                srimdb.trimdat.read(
                    what="atom_numb", condition=f"WHERE ion_numb = {nion}"
                )
            )[0][0]
            injected = np.array(
                [(atom_number, injected[0][0], injected[0][1], injected[0][2])],
                dtype=dtypes.defect,
            )
            defects = np.concatenate((defects, injected))
    return defects
