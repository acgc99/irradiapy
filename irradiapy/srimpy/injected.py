"""Utility functions related to injected ions for SRIM data."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from irradiapy.math_utils import fit_gaussian
from irradiapy.srimpy.srimdb import SRIMDB


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
