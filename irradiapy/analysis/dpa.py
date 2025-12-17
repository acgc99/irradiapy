"""Module for analyzing dpa."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from irradiapy.enums import DamageEnergyMode, DisplacementMode
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.materials import ELEMENT_BY_ATOMIC_NUMBER
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.utils.math import fit_lorentzian

if TYPE_CHECKING:
    from irradiapy.materials.component import Component


def depth_dpa(
    recoilsdb: RecoilsDB,
    damage_energy_mode: DamageEnergyMode,
    fluence: float = 1.0,
    path_fit: None | Path = None,
    p0: None | float = None,
    asymmetry: float = 1.0,
    path_debris: Path | None = None,
    nbins: int = 100,
    show: bool = False,
    plot_path: Path | None = None,
    dpi: int = 300,
) -> dict[str, any]:
    """Get depth-resolved dpa distribution and plot it.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    damage_energy_mode : DamageEnergyMode
        Damage energy calculation mode.
    path_debris : Path | None, optional (default=None)
        Path to the debris LAMMPS file to compute the debris dpa, by default
    nbins : int, optional (default=100)
        Number of depth nbins.
    show : bool, optional (default=False)
        Whether to show the plot.
    plot_path : Path | None, optional (default=None)
        Output path for the plot.
    dpi : int, optional (default=300)
        Dots per inch.

    Returns
    -------
    defaultdict
        Dictionary with depth edges and histograms for NRT, ARC, FERARC, and debris
        (if debris path is provided). Also includes fit parameters and functions if fitting
        is requested.
    """
    return_values = {}

    target = recoilsdb.load_target()
    width = sum(component.width for component in target)
    component_edges = np.cumsum([0.0] + [component.width for component in target])
    depth_edges = np.linspace(0.0, width, nbins + 1)
    depth_centers = (depth_edges[1:] + depth_edges[:-1]) / 2.0

    nevents = recoilsdb.get_nevents()
    nrecoils = recoilsdb.get_nrecoils()
    recoils = recoilsdb.read("recoils", what="depth, atom_numb, recoil_energy")
    depths = np.empty(nrecoils)
    nrts = np.empty(nrecoils)
    arcs = np.empty(nrecoils)
    ferarcs = np.empty(nrecoils)
    i = 0
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
    hist_arc, _ = np.histogram(depths, bins=depth_edges, weights=arcs)
    hist_ferarc, _ = np.histogram(depths, bins=depth_edges, weights=ferarcs)
    hist_nrt = hist_nrt * fluence / (natoms * nevents)
    hist_arc = hist_arc * fluence / (natoms * nevents)
    hist_ferarc = hist_ferarc * fluence / (natoms * nevents)
    return_values["depth_edges"] = depth_edges
    return_values["hist_nrt"] = hist_nrt
    return_values["hist_arc"] = hist_arc
    return_values["hist_ferarc"] = hist_ferarc

    hist_debris = None
    if path_debris is not None:
        depths = []
        reader = LAMMPSReader(path_debris)
        for defects in reader:
            depths.extend(defects["atoms"]["x"][defects["atoms"]["type"] == 0])
        depths = np.array(depths)

        hist_debris, _ = np.histogram(depths, bins=depth_edges)
        hist_debris = hist_debris * fluence / (natoms * nevents)
        return_values["hist_debris"] = hist_debris

    # Fit
    if path_fit:

        def fit_hist(hist: npt.NDArray[np.float64], label: str) -> None:
            params, err_params, fit_function = fit_lorentzian(
                depth_centers, hist, p0, asymmetry
            )
            return_values[f"params_{label}"] = params
            return_values[f"errors_{label}"] = err_params
            return_values[f"fitfunc_{label}"] = fit_function

        def write_params(key: str, label: str) -> None:
            params = return_values[f"params_{key}"]
            err_params = return_values[f"errors_{key}"]
            file.write(f"{label}\n")
            file.write(f"x0 = {params[0]} ± {err_params[0]}\n")
            file.write(f"sigma = {params[1]} ± {err_params[1]}\n")
            file.write(f"A = {params[2]} ± {err_params[2]}\n")
            file.write(f"a = {params[3]} ± {err_params[3]}\n")

        try:
            fit_hist(hist_nrt, "nrt")
            fit_hist(hist_arc, "arc")
            fit_hist(hist_ferarc, "ferarc")
            if "hist_debris" in return_values:
                fit_hist(hist_debris, "debris")

            file = open(path_fit, "w", encoding="utf-8")
            file.write(
                (
                    "Displaced atoms fit fit. See Eq. (2) of "
                    "Nuclear Instruments and Methods in Physics Research B "
                    "500-501 (2021) 52-56\n"
                )
            )
            file.write("x in angstroms\n")
            file.write(
                (
                    "A * sigma ** 2 / "
                    "( sigma ** 2 + (1 + e ** (a * (x - x0)) ** 2) / 4 * (x - x0) ** 2 )\n"
                )
            )
            if "fitfunc_nrt" in return_values:
                write_params("nrt", "NRT")
            if "fitfunc_arc" in return_values:
                write_params("arc", "arc")
            if "fitfunc_ferarc" in return_values:
                write_params("ferarc", "fer-arc")
            if hist_debris is not None and "fitfunc_debris" in return_values:
                write_params("debris", "debris")
            file.close()

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Fit failed: {exc}")

    # Plot
    if show or plot_path:
        fig = plt.figure()
        gs = fig.add_gridspec()
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
        ax.set_ylabel("Displacements per atom")

        def plot_fit(key: str, scatter: plt.Line2D) -> None:
            if f"fitfunc_{key}" not in return_values:
                return
            ax.plot(
                depth_centers,
                return_values[f"fitfunc_{key}"](depth_centers),
                c=scatter.get_facecolor(),
            )

        if "hist_nrt" in return_values:
            s1 = ax.scatter(depth_centers, hist_nrt, label="NRT", marker=".")
            plot_fit("nrt", s1)
        if "hist_arc" in return_values:
            s2 = ax.scatter(depth_centers, hist_arc, label="arc", marker=".")
            plot_fit("arc", s2)
        if "hist_ferarc" in return_values:
            s3 = ax.scatter(depth_centers, hist_ferarc, label="fer-arc", marker=".")
            plot_fit("ferarc", s3)
        if "hist_debris" in return_values:
            s4 = ax.scatter(depth_centers, hist_debris, label="debris", marker=".")
            plot_fit("debris", s4)

        # Finish
        ax.legend()
        fig.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    return return_values


def dpa(
    recoilsdb: RecoilsDB,
    damage_energy_mode: DamageEnergyMode | None = None,
    fluence: float | None = None,
    path_debris: Path | None = None,
    depth_dpa_return: dict[str, any] | None = None,
) -> dict[str, float]:
    """Calculate dpa values from the recoils database.

    Parameters
    ----------
    recoilsdb : RecoilsDB
        Recoils database.
    damage_energy_mode : DamageEnergyMode, optional (default=None)
        Damage energy calculation mode. Required if histograms are not provided.
    fluence : float, optional (default=None)
        Fluence in ions/Å². Required if histograms are not provided.
    path_debris : Path | None, optional (default=None)
        Path to the debris LAMMPS file to compute the debris dpa.
    depth_dpa_return : defaultdict | None, optional (default=None)
        If provided (from depth_dpa()), uses the histograms and depth edges from this dictionary.

    Returns
    -------
    defaultdict
        Dictionary with dpa values: 'nrt', 'arc', 'ferarc', and 'debris' (if debris
        histogram is provided).
    """
    return_values = {}

    if not depth_dpa_return:
        depth_dpa_return = depth_dpa(
            recoilsdb,
            damage_energy_mode,
            path_debris=path_debris,
            fluence=fluence,
        )
    hist_nrt = depth_dpa_return["hist_nrt"]
    hist_arc = depth_dpa_return["hist_arc"]
    hist_ferarc = depth_dpa_return["hist_ferarc"]
    hist_debris = depth_dpa_return["hist_debris"]
    depth_edges = depth_dpa_return["depth_edges"]

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
    nrt = np.sum(hist_nrt * natoms) / natoms_total
    arc = np.sum(hist_arc * natoms) / natoms_total
    ferarc = np.sum(hist_ferarc * natoms) / natoms_total
    return_values["nrt"] = float(nrt)
    return_values["arc"] = float(arc)
    return_values["ferarc"] = float(ferarc)

    if hist_debris is not None:
        debris = np.sum(hist_debris * natoms) / natoms_total
        return_values["debris"] = float(debris)

    return return_values
