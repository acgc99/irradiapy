"""Module for analyzing dpa."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from irradiapy.analysis.analysisdb import AnalysisDB
from irradiapy.enums import DamageEnergyMode, DisplacementMode
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.materials import ELEMENT_BY_ATOMIC_NUMBER
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.utils.math import fit_lorentzian

if TYPE_CHECKING:
    from irradiapy.materials.component import Component


def depth_dpa_hist(
    recoilsdb: RecoilsDB,
    analysisdb: AnalysisDB,
    axis: str,
    damage_energy_mode: DamageEnergyMode,
    fluence: float,
    debris_path: Path | None = None,
    nbins: int = 100,
    p0: None | float = None,
    asymmetry: float = 1.0,
) -> None:
    """Get depth dpa histogram and save it into database.

    Fit: asymmetric lorentzian function.
    Tries to fit to Eq. (2) of
    Nuclear Instruments and Methods in Physics Research B 500-501 (2021) 52-56.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    analysisdb : AnalysisDB
        Database for storing results.
    axis : str
        Axis along which to calculate depth ('x', 'y', or 'z').
    damage_energy_mode : DamageEnergyMode
        Damage energy calculation mode.
    fluence : float
        Fluence in ions/Å².
    debris_path : Path | None, optional (default=None)
        Path to the debris LAMMPS file to compute the debris dpa.
    nbins : int, optional (default=100)
        Number of bins for the depth histogram.
    p0 : float | None, optional (default=None)
        Initial guess for Lorentzian fit.
    asymmetry : float, optional (default=1.0)
        Bound for the asymmetry fit parameter. Fit will be done in (-asymmetry, asymmetry).

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        Dictionary with depth centers and dpa histograms. Keys: 'depth_centers', 'nrt', 'arc',
        'ferarc', and optionally 'debris'.
    """
    target = recoilsdb.load_target()
    if axis == "x":
        width = sum(component.width for component in target)
        component_edges = np.cumsum([0.0] + [component.width for component in target])
    elif axis == "y":
        width = sum(component.height for component in target)
        component_edges = np.cumsum([0.0] + [component.height for component in target])
    elif axis == "z":
        width = sum(component.length for component in target)
        component_edges = np.cumsum([0.0] + [component.length for component in target])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    depth_edges = np.linspace(0.0, width, nbins + 1)
    depth_centers = (depth_edges[1:] + depth_edges[:-1]) / 2.0

    nevents = recoilsdb.get_nevents()
    nrecoils = recoilsdb.get_nrecoils()
    depths = np.empty(nrecoils)
    nrts = np.empty(nrecoils)
    arcs = np.empty(nrecoils)
    ferarcs = np.empty(nrecoils)
    i = 0
    recoils = recoilsdb.read("recoils", what=f"{axis}, atom_numb, recoil_energy")
    for depth, atom_numb, recoil_energy in recoils:
        # Use recoil depth to find the depth center, then use that to find the component
        # and use that center for the histogram
        bin_idx = np.searchsorted(depth_edges, depth, side="right") - 1
        depth_center = depth_centers[bin_idx]
        component_idx = np.searchsorted(component_edges, depth_center, side="right") - 1
        component: Component = target[component_idx]

        damage_energy = component.recoil_energy_to_damage_energy(
            recoil_energy,
            ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
            damage_energy_mode,
        )
        nrt = component.damage_energy_to_displacements(
            damage_energy,
            mode=DisplacementMode.NRT,
        )
        arc = component.damage_energy_to_displacements(
            damage_energy,
            mode=DisplacementMode.ARC,
        )
        ferarc = component.damage_energy_to_displacements(
            damage_energy,
            mode=DisplacementMode.FERARC,
        )
        depths[i] = depth_center
        nrts[i] = nrt
        arcs[i] = arc
        ferarcs[i] = ferarc
        i += 1

    natoms = np.zeros(nbins)
    dx = depth_edges[1] - depth_edges[0]
    for i in range(nbins):
        component_idx = (
            np.searchsorted(
                component_edges,
                (depth_edges[i] + depth_edges[i + 1]) / 2.0,
                side="right",
            )
            - 1
        )
        component = target[component_idx]
        natoms[i] = int(component.atomic_density * dx)

    hist_nrt, _ = np.histogram(depths, bins=depth_edges, weights=nrts)
    hist_nrt = hist_nrt * fluence / (natoms * nevents)
    hist_arc, _ = np.histogram(depths, bins=depth_edges, weights=arcs)
    hist_ferarc, _ = np.histogram(depths, bins=depth_edges, weights=ferarcs)
    hist_arc = hist_arc * fluence / (natoms * nevents)
    hist_ferarc = hist_ferarc * fluence / (natoms * nevents)
    hists = {
        "depth_centers": depth_centers,
        "nrt": hist_nrt,
        "arc": hist_arc,
        "ferarc": hist_ferarc,
    }

    hist_debris = None
    if debris_path is not None:
        depths = []
        reader = LAMMPSReader(debris_path)
        for defects in reader:
            depths.extend(defects["atoms"][axis][defects["atoms"]["type"] == 0])
        depths = np.array(depths)

        hist_debris, _ = np.histogram(depths, bins=depth_edges)
        hist_debris = hist_debris * fluence / (natoms * nevents)
        hists["debris"] = hist_debris

    for model, hist in hists.items():
        if model == "depth_centers":
            continue
        analysisdb.save_depth_dpa_hist(
            axis=axis, model=model, depth_centers=depth_centers, hist=hist
        )

    def fit_hist(model: str, hist: npt.NDArray[np.float64]) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        params, errors, _ = fit_lorentzian(depth_centers, hist, p0, asymmetry)
        analysisdb.save_depth_dpa_hist_fit_params(axis, model, *params)
        analysisdb.save_depth_dpa_hist_fit_errors(axis, model, *errors)
        return params, errors

    try:
        fit_hist(model="nrt", hist=hist_nrt)
        fit_hist(model="arc", hist=hist_arc)
        fit_hist(model="ferarc", hist=hist_ferarc)
        if hist_debris is not None:
            fit_hist(model="debris", hist=hist_debris)

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fit failed: {exc}")

    return hists


def depth_dpa_hist_plot(
    analysisdb: AnalysisDB,
    axis: str,
    fits: bool = False,
    plot_path: Path | None = None,
    show: bool = False,
    dpi: int = 300,
) -> None:
    """Plot depth dpa histogram.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database. Must contain depth dpa data.
    axis : str
        Axis along which to plot depth ('x', 'y', or 'z').
    fits : bool, optional (default=False)
        Whether to plot the fit functions.
    plot_path : Path | None, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional (default=False)
        Whether to show the plot.
    dpi : int, optional (default=300)
        DPI for saving the plot.
    """
    if fits:
        fit_nrt = analysisdb.load_depth_dpa_hist_fit_functions(axis, "nrt")
        fit_arc = analysisdb.load_depth_dpa_hist_fit_functions(axis, "arc")
        fit_ferarc = analysisdb.load_depth_dpa_hist_fit_functions(axis, "ferarc")
        fits = {
            "nrt": fit_nrt,
            "arc": fit_arc,
            "ferarc": fit_ferarc,
        }

    depth_centers, hist_nrt = analysisdb.load_depth_dpa_hist(axis=axis, model="nrt")
    _, hist_arc = analysisdb.load_depth_dpa_hist(axis=axis, model="arc")
    _, hist_ferarc = analysisdb.load_depth_dpa_hist(axis=axis, model="ferarc")
    hists = {
        "nrt": hist_nrt,
        "arc": hist_arc,
        "ferarc": hist_ferarc,
    }
    hist_debris = None
    if analysisdb.table_has_column(f"depth_dpa_{axis}", "debris"):
        _, hist_debris = analysisdb.load_depth_dpa_hist(axis=axis, model="debris")
        hists["debris"] = hist_debris
        if fits:
            fit_debris = analysisdb.load_depth_dpa_hist_fit_functions(axis, "debris")
            fits["debris"] = fit_debris
    models = list(hists.keys())

    fig = plt.figure()
    gs = fig.add_gridspec()
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("Displacements per atom")

    for model in models:
        scatter = ax.scatter(depth_centers, hists[model], label=model, marker=".")
        if fits is not False:
            try:
                ax.plot(
                    depth_centers, fits[model](depth_centers), c=scatter.get_facecolor()
                )
            except sqlite3.OperationalError as exc:
                print(f"Could not plot fit function for model {model}: {exc}")

    ax.legend()
    fig.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def dpa(recoilsdb: RecoilsDB, analysisdb: AnalysisDB) -> dict[str, float]:
    """Calculate dpa values from the recoils database.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    analysisdb : AnalysisDB
        Database. Must contain depth dpa data.

    Returns
    -------
    dict[str, float]
        Dictionary with dpa values: 'nrt', 'arc', 'ferarc', and 'debris'.
    """
    if analysisdb.table_exists("depth_dpa_x") is True:
        axis = "x"
    elif analysisdb.table_exists("depth_dpa_y") is True:
        axis = "y"
    elif analysisdb.table_exists("depth_dpa_z") is True:
        axis = "z"
    else:
        raise ValueError("No depth dpa data found in analysisdb.")

    depth_centers, hist_nrt = analysisdb.load_depth_dpa_hist(axis=axis, model="nrt")
    _, hist_arc = analysisdb.load_depth_dpa_hist(axis=axis, model="arc")
    _, hist_ferarc = analysisdb.load_depth_dpa_hist(axis=axis, model="ferarc")
    hist_debris = None
    if analysisdb.table_has_column(f"depth_dpa_{axis}", "debris"):
        _, hist_debris = analysisdb.load_depth_dpa_hist(axis=axis, model="debris")

    dx2 = (depth_centers[1] - depth_centers[0]) / 2.0
    depth_edges = np.concatenate(
        (
            [depth_centers[0] - dx2],
            (depth_centers[1:] + depth_centers[:-1]) / 2.0,
            [depth_centers[-1] + dx2],
        )
    )

    target = recoilsdb.load_target()
    component_edges = np.cumsum([0.0] + [component.width for component in target])

    nbins = len(depth_edges) - 1
    natoms = np.zeros(nbins, dtype=float)
    dx = depth_edges[1] - depth_edges[0]
    for i in range(nbins):
        component_idx = (
            np.searchsorted(
                component_edges,
                (depth_edges[i] + depth_edges[i + 1]) / 2.0,
                side="right",
            )
            - 1
        )
        component: Component = target[component_idx]
        natoms[i] = component.atomic_density * dx

    natoms_total = np.sum(natoms)
    nrt = float(np.sum(hist_nrt * natoms) / natoms_total)
    arc = float(np.sum(hist_arc * natoms) / natoms_total)
    ferarc = float(np.sum(hist_ferarc * natoms) / natoms_total)
    dpas = {"nrt": nrt, "arc": arc, "ferarc": ferarc}

    if hist_debris is not None:
        debris = float(np.sum(hist_debris * natoms) / natoms_total)
        dpas["debris"] = debris

    for model, dpa_val in dpas.items():
        analysisdb.save_dpa(model, dpa_val)

    return dpas
