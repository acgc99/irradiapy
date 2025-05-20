"""Cluster analysis module."""

import sqlite3
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.image import NonUniformImage
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from irradiapy import dtypes
from irradiapy.io.xyzreader import XYZReader
from irradiapy.io.xyzwriter import XYZWriter


def clusterize(defects: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Finds defects clusters in the given defects.

    Notes
    -----
    Algorithm: union-find path compression. Output data structure (atom clusters):
    ``{1: [[accx0, ...], [accy0, ...], [accz0, ...]],
    2: [[accx0, ...], [accy0, ...], [accz0, ...]]}``

    Parameters
    ----------
    defects : dtypes.DEFECT_CHECK
        The defects.
    cutoff : float
        Cutoff distance for clustering.

    Returns
    -------
    tuple[dtypes.ACLUSTER_CHECK, dtypes.OCLUSTER_CHECK]
        Atomic clusters and object clusters.
    """
    ndefects = len(defects)
    defects_ = np.zeros(ndefects, dtype=dtypes.defect)
    defects_["type"] = defects["type"]
    if (
        "x" in defects.dtype.names
        or "y" in defects.dtype.names
        or "z" in defects.dtype.names
    ):
        defects_["pos"][:, 0] = defects["x"]
        defects_["pos"][:, 1] = defects["y"]
        defects_["pos"][:, 2] = defects["z"]
        defects = defects_

    cutoff2 = cutoff**2
    natoms = defects.size
    aclusters = np.empty(natoms, dtype=dtypes.acluster)
    aclusters["pos"] = defects["pos"]
    aclusters["type"] = defects["type"]
    aclusters["cluster"] = np.arange(1, natoms + 1)

    for i in range(natoms):
        curr_cluster = aclusters[i]["cluster"]
        dists = np.sum(
            np.square(aclusters[i]["pos"] - aclusters[i + 1 :]["pos"]), axis=1
        )
        neighbourhood = aclusters["cluster"][np.where(dists <= cutoff2)[0] + i + 1]
        if neighbourhood.size:
            for neighbour in neighbourhood:
                if neighbour != curr_cluster:
                    aclusters["cluster"][
                        aclusters["cluster"] == neighbour
                    ] = curr_cluster

    nclusters = np.unique(aclusters["cluster"])
    for i in range(nclusters.size):
        aclusters["cluster"][aclusters["cluster"] == nclusters[i]] = i + 1

    oclusters = atom_to_object(aclusters)
    return aclusters, oclusters


def atom_to_object(aclusters: np.ndarray) -> np.ndarray:
    """Transform atom clusters into object clusters.

    Parameters
    ----------
    aclusters : dtypes.ACLUSTER_CHECK
        Atomic clusters.

    Returns
    -------
    dtypes.OCLUSTER_CHECK
        Object clusters.
    """
    nclusters = np.unique(aclusters["cluster"])
    oclusters = np.empty(nclusters.size, dtype=dtypes.ocluster)
    for i in range(nclusters.size):
        acluster = aclusters[aclusters["cluster"] == nclusters[i]]
        oclusters[i]["pos"] = np.mean(acluster["pos"], axis=0)
        oclusters[i]["type"] = acluster[0]["type"]
        oclusters[i]["size"] = acluster.size
    return oclusters


def clusterize_file(
    path_collisions: Path,
    cutoff_sia: float,
    cutoff_vac: float,
    path_aclusters: Path,
    path_oclusters: Path,
    irradiated_particle: str = "Projectile",
) -> None:
    """Finds defect clusters in the given file.

    Parameters
    ----------
    path_collisions : Path
        Path of the file where defects are.
    cutoff_sia : float
        Cutoff distance for interstitials clustering.
    cutoff_vac : float
        Cutoff distance for vacancies clustering.
    path_aclusters : Path
        Where atomic clusters will be stored.
    path_oclusters : Path
        Where object clusters will be stored.
    irradiated_particle : str, optional
        Name of the irradiated particle, by default "Projectile".
    """
    reader = XYZReader(path_collisions)
    nsim = 0
    with XYZWriter(path_aclusters) as awriter, XYZWriter(path_oclusters) as owriter:
        for defects in reader:
            nsim += 1
            cond = defects["type"] == 0
            sia, vac = defects[~cond], defects[cond]
            iaclusters, ioclusters = clusterize(sia, cutoff_sia)
            vaclusters, voclusters = clusterize(vac, cutoff_vac)
            aclusters = np.concatenate((iaclusters, vaclusters))
            oclusters = np.concatenate((ioclusters, voclusters))
            awriter.save(aclusters, extra_comment=f"{irradiated_particle}={nsim}")
            owriter.save(oclusters, extra_comment=f"{irradiated_particle}={nsim}")


# region Analysis
def __get_size_depth_histogram(
    sizes: np.ndarray,
    depths: np.ndarray,
    depth_bins: int = 100,
    size_bins: Optional[np.integer] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate a 2D histogram of sizes and depths.

    Parameters
    ----------
    sizes : dtypes.FLOAT_CHECK
        An array of size values.
    depths : dtypes.FLOAT_CHECK
        An array of depth values.
    depth_bins : int, optional
        Number of bins for the depth axis, by default 100.
    size_bins : int, optional
        Number of bins for the size axis, by default None (one bin per size).

    Returns
    -------
    tuple[dtypes.FLOAT_CHECK, dtypes.FLOAT_CHECK, dtypes.FLOAT_CHECK, dtypes.FLOAT_CHECK,
    dtypes.FLOAT_CHECK]
        A tuple containing: the values of the histogram, the bin edges of the size histogram,
        the bin edges of the depth histogram, the centers of the size bins, the centers of the
        depth bins.
    """
    max_size = sizes.max()
    if size_bins is None:
        size_bins = max_size
    histogram, size_edges, depth_edges = np.histogram2d(
        x=sizes,
        y=depths,
        bins=(size_bins, depth_bins),
        range=((0.5, max_size + 0.5), (depths.min(), depths.max())),
    )
    size_centers = (size_edges[:-1] + size_edges[1:]) / 2.0
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0
    return histogram, size_edges, depth_edges, size_centers, depth_centers


def __get_size_histogram(
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a histogram of cluster sizes.

    Parameters
    ----------
    sizes : dtypes.FLOAT_CHECK
        An array of cluster sizes.

    Returns
    -------
    tuple[dtypes]
        A tuple containing: the histogram of cluster sizes, the edges of the histogram bins and
        the centers of the histogram bins.
    """
    max_size = sizes.max()
    histogram, size_edges = np.histogram(
        sizes, bins=max_size, range=(0.5, max_size + 0.5)
    )
    size_centers = (size_edges[:-1] + size_edges[1:]) / 2.0
    return histogram, size_edges, size_centers


def get_clusters(path_oclusters: Path, db_path: Path) -> None:
    """Processes cluster data from an XYZ file and stores the results in a SQLite database.

    Parameters
    ----------
    path_oclusters : Path
        Path to the input XYZ file containing cluster data.
    db_path : Path
        Path to the output SQLite database file.
    """
    isizes, vsizes = np.array([], dtype=int), np.array([], dtype=int)
    idepths, vdepths = np.array([], dtype=float), np.array([], dtype=float)

    reader = XYZReader(path_oclusters)
    for oclusters in reader:
        cond = oclusters["type"] == 0
        vacancies = oclusters[cond]
        if vacancies.size > 0:
            vsizes = np.concatenate((vsizes, vacancies["size"]))
            vdepths = np.concatenate((vdepths, vacancies["pos"][:, 0]))
        interstitials = oclusters[~cond]
        if interstitials.size > 0:
            isizes = np.concatenate((isizes, interstitials["size"]))
            idepths = np.concatenate((idepths, interstitials["pos"][:, 0]))

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS interstitials")
    cur.execute("CREATE TABLE interstitials (size INTEGER, depth REAL)")
    cur.executemany(
        "INSERT INTO interstitials (size, depth) VALUES (?, ?)",
        np.column_stack((isizes, idepths)),
    )
    cur.execute("DROP TABLE IF EXISTS vacancies")
    cur.execute("CREATE TABLE vacancies (size INTEGER, depth REAL)")
    cur.executemany(
        "INSERT INTO vacancies (size, depth) VALUES (?, ?)",
        np.column_stack((vsizes, vdepths)),
    )
    con.commit()
    cur.close()
    con.close()


def load_results(
    db_path: Path, what: str = "*", condition: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load results from an SQLite database generated with `get_clusters`.

    Parameters
    ----------
    db_path : Path
        The path to the SQLite database file.
    what : str, optional
        The columns to select from the tables, by default "*".
    condition : str, optional
        An optional SQL condition to filter the results, by default "".

    Returns
    -------
    tuple[dtypes.INT_CHECK, dtypes.FLOAT_CHECK, dtypes.INT_CHECK, dtypes.FLOAT_CHECK]
        A tuple containing: the sizes of the interstitials, the depths of the interstitials, the
        sizes of the vacancies, the depths of the vacancies
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(f"SELECT {what} FROM interstitials {condition}")
    data = np.array(cur.fetchall())
    isizes = data[:, 0].astype(int)
    idepths = data[:, 1]
    cur.execute(f"SELECT {what} FROM vacancies {condition}")
    data = np.array(cur.fetchall())
    vsizes = data[:, 0].astype(int)
    vdepths = data[:, 1]
    return isizes, idepths, vsizes, vdepths


def __scaling_law_fit(
    centers: np.ndarray, counts: np.ndarray
) -> tuple[float, float, callable]:
    """Fit a scaling law to the given histogram data.

    Parameters
    ----------
    centers : dtypes.FLOAT_CHECK
        The centers of the bins.
    counts : dtypes.FLOAT_CHECK
        The values of the histogram.

    Returns
    -------
    tuple
        A tuple containing: the prefactor of the scaling law, the exponent of the scaling law,
        and the scaling law function.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    popt, _ = curve_fit(lambda x, a, b: a + b * x, np.log10(centers), np.log10(counts))
    a, s = popt
    a, s = 10.0**a, -s

    def curve(x):
        return a / x**s

    return a, s, curve


def plot_results(
    db_path: Path,
    out_dir: Path,
    nions: int,
    depth_bins: int = 100,
    size_bins: Optional[int] = None,
) -> None:
    """Plots the results of size-depth histograms, clustering fractions,
    and scaling laws for interstitials and vacancies.

    Parameters
    ----------
    db_path : Path
        The path to the SQLite database file.
    out_dir : Path
        The output directory where the plots and results will be saved.
    nions : int
        The number of ions used for normalization.
    depth_bins : int, optional
        The number of bins for depth histograms, by default 100.
    size_bins : int, optional
        The number of bins for size histograms, by default None (one bin per size).
    """
    isizes, idepths, vsizes, vdepths = load_results(db_path)

    (
        sia_histogram,
        sia_size_edges,
        sia_depth_edges,
        sia_size_centers,
        sia_depth_centers,
    ) = __get_size_depth_histogram(
        isizes, idepths, depth_bins=depth_bins, size_bins=size_bins
    )
    (
        vac_histogram,
        vac_size_edges,
        vac_depth_edges,
        vac_size_centers,
        vac_depth_centers,
    ) = __get_size_depth_histogram(
        vsizes, vdepths, depth_bins=depth_bins, size_bins=size_bins
    )
    sia_histogram /= nions
    vac_histogram /= nions

    # Plot size-depth
    fig = plt.figure()
    # Interstitials
    iax = plt.subplot2grid((9, 2), (1, 0), rowspan=8)
    iax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
    iax.set_xlabel("Self-interstitial size")
    iax.set_xlim(sia_size_edges[[0, -1]])
    iax.set_ylim(sia_depth_edges[[0, -1]])
    iim = NonUniformImage(
        iax,
        interpolation="nearest",
        norm="log",
        extent=(*sia_size_edges[[0, -1]], *sia_depth_edges[[0, -1]]),
    )
    iim.set_data(sia_size_centers, sia_depth_centers, sia_histogram.T)
    iax.add_image(iim)
    # Vacancies
    vax = plt.subplot2grid((9, 2), (1, 1), rowspan=8)
    vax.yaxis.set_label_position("right")
    vax.yaxis.tick_right()
    # vax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
    vax.set_yticklabels([])
    vax.set_xlabel("Vacancy size")
    vax.set_xlim(vac_size_edges[[0, -1]])
    vax.set_ylim(vac_depth_edges[[0, -1]])
    vim = NonUniformImage(
        vax,
        interpolation="nearest",
        norm="log",
        extent=(*vac_size_edges[[0, -1]], *vac_depth_edges[[0, -1]]),
    )
    vim.set_data(vac_size_centers, vac_depth_centers, vac_histogram.T)
    vax.add_image(vim)
    # Color bar
    cax = plt.subplot2grid((9, 2), (0, 0), colspan=2)
    cbar = fig.colorbar(vim, cax=cax, orientation="horizontal")
    cbar.set_label("Counts per ion")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    # Final touches
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "clusters_depth.png", dpi=300)
    else:
        plt.show()
    plt.close()

    # Clustering fraction
    # One bin, one size
    if size_bins is not None:
        (
            sia_histogram,
            sia_size_edges,
            sia_depth_edges,
            sia_size_centers,
            sia_depth_centers,
        ) = __get_size_depth_histogram(
            isizes, idepths, depth_bins=depth_bins, size_bins=None
        )
        (
            vac_histogram,
            vac_size_edges,
            vac_depth_edges,
            vac_size_centers,
            vac_depth_centers,
        ) = __get_size_depth_histogram(
            vsizes, vdepths, depth_bins=depth_bins, size_bins=None
        )
    sia_histogram /= nions
    vac_histogram /= nions
    # Interstitials
    ihistogram1 = sia_histogram.copy()
    ihistogram1 *= sia_size_centers[:, np.newaxis]
    imonomoer = ihistogram1[0]
    itotal = ihistogram1.sum(axis=0)
    # Vacancies
    vhistogram1 = vac_histogram.copy()
    vhistogram1 *= vac_size_centers[:, np.newaxis]
    vmonomoer = vhistogram1[0]
    vtotal = vhistogram1.sum(axis=0)
    # Plot
    plt.scatter(sia_depth_centers, 1.0 - imonomoer / itotal, label="SIA")
    plt.scatter(vac_depth_centers, 1.0 - vmonomoer / vtotal, label="VAC")
    plt.xlabel(r"Depth ($\mathrm{\AA}$)")
    plt.ylabel("Clustering fraction")
    plt.legend()
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "clustering_fraction.png", dpi=300)
    else:
        plt.show()
    plt.close()

    # Plot interstitials scaling law
    try:
        icounts = sia_histogram.sum(axis=1)
        icounts_ = icounts[icounts > 0]
        sia_size_centers_ = sia_size_centers[icounts > 0]
        small_counts, big_counts = icounts_[:10], icounts_[10:]
        small_centers, big_centers = sia_size_centers_[:10], sia_size_centers_[10:]
        sia_small_a, sia_small_s, sia_small_curve = __scaling_law_fit(
            small_centers, small_counts
        )
        sia_big_a, sia_big_s, sia_big_curve = __scaling_law_fit(big_centers, big_counts)
        # Plot
        plt.scatter(
            small_centers,
            small_counts,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        plt.plot(
            small_centers,
            sia_small_curve(small_centers),
            label=r"Small self-interstitials",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        plt.scatter(
            big_centers,
            big_counts,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        )
        plt.plot(
            big_centers,
            sia_big_curve(big_centers),
            label=r"Big self-interstitials",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Size")
        plt.ylabel("Counts per ion")
        plt.legend()
        plt.tight_layout()
        if out_dir:
            plt.savefig(out_dir / "clusters_scaling_law_interstitials.png", dpi=300)
        else:
            plt.show()
        plt.close()
    except Exception:  # pylint: disable=broad-exception-caught
        print(
            "Error fitting interstitials scaling law. This functionality is a "
            "general recipe and may not work for all cases."
        )

    # Plot vacancies scaling law
    try:
        vcounts = vac_histogram.sum(axis=1)
        vcounts_ = vcounts[vcounts > 0]
        vac_size_centers_ = vac_size_centers[vcounts > 0]
        small_counts, big_counts = vcounts_[:10], vcounts_[10:]
        small_centers, big_centers = vac_size_centers_[:10], vac_size_centers_[10:]
        vac_small_a, vac_small_s, vac_small_curve = __scaling_law_fit(
            small_centers, small_counts
        )
        vac_big_a, vac_big_s, vac_big_curve = __scaling_law_fit(big_centers, big_counts)
        # Plot
        plt.scatter(
            small_centers,
            small_counts,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        plt.plot(
            small_centers,
            vac_small_curve(small_centers),
            label=r"Small vacancies",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        plt.scatter(
            big_centers,
            big_counts,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        )
        plt.plot(
            big_centers,
            vac_big_curve(big_centers),
            label=r"Big vacancies",
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Size")
        plt.ylabel("Counts per ion")
        plt.legend()
        plt.tight_layout()
        if out_dir:
            plt.savefig(out_dir / "clusters_scaling_law_vacancies.png", dpi=300)
            with open(
                out_dir / "clusters_scaling_law.txt", "w", encoding="utf-8"
            ) as file:
                file.write("Scaling law: A/x**S; A, S\n")
                file.write(f"Small SIA {sia_small_a}, {sia_small_s}\n")
                file.write(f"Big SIA {sia_big_a}, {sia_big_s}\n")
                file.write(f"Small VAC {vac_small_a}, {vac_small_s}\n")
                file.write(f"Big VAC {vac_big_a}, {vac_big_s}\n")
        else:
            plt.show()
        plt.close()
    except Exception:  # pylint: disable=broad-exception-caught
        print(
            "Error fitting vacancies scaling law. This functionality is a "
            "general recipe and may not work for all cases."
        )


def mddb_analysis(
    debris_dir: Path,
    cutoff_sia: float,
    cutoff_vac: float,
    bin_width: int = 10,
    lognorm: bool = True,
    plot_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Analyzes molecular dynamics debris data to generate histograms of cluster sizes
    for interstitials and vacancies.

    Parameters
    ----------
    debris_dir : Path
        Directory containing debris data files.
    cutoff_sia : float
        Cutoff distance for clustering self-interstitial atoms (SIAs).
    cutoff_vac : float
        Cutoff distance for clustering vacancies.
    bin_width : int, optional
        Bin size for histogram. Defaults to 10.
    lognorm : bool, optional
        Use log normalization for the color scale. Defaults to True.
    plot_path : Path, optional
        Path to save the plot. Defaults to None. If None, the plot is shown.
    dpi : int, optional
        DPI of the plot. Defaults to 300.
    """
    epkas, clusters, nfiles = [], {}, {}
    for file_path in debris_dir.iterdir():
        if file_path.is_dir():
            epka = int(file_path.name)
            epkas.append(epka)
            clusters[epka] = {"sia": [], "vac": []}
            nfiles[epka] = 0
            for file_path in file_path.iterdir():
                defects = list(XYZReader(file_path))[0]
                cond = defects["type"] == 0
                vacancies = defects[cond]
                interstitials = defects[np.invert(cond)]
                _, ioclusters = clusterize(interstitials, cutoff_sia)
                _, voclusters = clusterize(vacancies, cutoff_vac)
                clusters[epka]["sia"] += ioclusters["size"].tolist()
                clusters[epka]["vac"] += voclusters["size"].tolist()
                nfiles[epka] += 1
    epkas.sort()
    # Energy binning
    nepka_bins = len(epkas)
    # Determine maximum cluster size
    max_sia, max_vac = 0, 0
    for siasvacs in clusters.values():
        max_sia0 = max(siasvacs["sia"])
        max_vac0 = max(siasvacs["vac"])
        max_sia = max_sia0 if max_sia0 > max_sia else max_sia
        max_vac = max_vac0 if max_vac0 > max_vac else max_vac
    # Round to closest highest multiple of bin_width starting from 10,
    # where binning changes
    # if bin_width = 10, then 83 > 90
    # if bin_width = 10, then 57 > 60
    max_sia = 10 + int(np.ceil((max_sia - 10) / bin_width) * bin_width)
    max_vac = 10 + int(np.ceil((max_vac - 10) / bin_width) * bin_width)
    # Use the same maximum size for both types of clusters
    if max_vac > max_sia:
        max_sia = max_vac
    else:
        max_vac = max_sia
    # Histogram bin edges
    sia_edges = np.concatenate(
        (np.arange(0.5, 10.5), np.arange(10.5, max_sia + bin_width + 0.5, bin_width))
    )
    vac_edges = np.concatenate(
        (np.arange(0.5, 10.5), np.arange(10.5, max_vac + bin_width + 0.5, bin_width))
    )
    # Histogram
    ihist = []
    vhist = []
    for epka in epkas:
        siasvacs = clusters[epka]
        hist, _ = np.histogram(
            siasvacs["sia"],
            bins=sia_edges,
            # weights=siasvacs["sia"],
            density=False,
        )
        ihist.append(hist / nfiles[epka])
        hist, _ = np.histogram(
            siasvacs["vac"],
            bins=vac_edges,
            # weights=siasvacs["vac"],
            density=False,
        )
        vhist.append(hist / nfiles[epka])
    ihist = np.array(ihist)  # row: energy, column: size
    vhist = np.array(vhist)  # row: energy, column: size
    # Mask to remove zero counts.
    ihist = np.ma.masked_where(ihist == 0, ihist)
    vhist = np.ma.masked_where(vhist == 0, vhist)
    vmin = min(ihist.min(), vhist.min())
    vmax = max(ihist.max(), vhist.max())
    # Norm definition
    if lognorm:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    iax = fig.add_subplot(gs[0, 0])
    iim = iax.imshow(
        ihist.T,
        origin="lower",
        aspect="auto",
        norm=norm,
        cmap="viridis",
        extent=[0, nepka_bins, 0, len(sia_edges) - 1],
    )
    iax.set_title("Interstitials")
    iax.set_xlabel("Energy (keV)")
    iax.set_ylabel("Cluster size")

    vax = fig.add_subplot(gs[0, 1])
    vax.imshow(
        vhist.T,
        origin="lower",
        aspect="auto",
        norm=norm,
        cmap="viridis",
        extent=[0, nepka_bins, 0, len(vac_edges) - 1],
    )
    vax.set_title("Vacancies")
    vax.set_xlabel("Energy (keV)")
    vax.set_yticklabels([])
    vax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set correct energy labels and ticks
    for ax in [iax, vax]:
        ax.set_xticks(np.arange(nepka_bins) + 0.5)
        ax.set_xticklabels([f"{int(e/1e3)}" for e in epkas])
    # SIA size labels: 1, 2, ..., 10, 11-20, ...
    if bin_width == 1:
        labels = [str(i) for i in range(1, max_sia + 1)]
    else:
        labels = [str(i) for i in range(1, 11)] + [
            f"{i}-{i+bin_width-1}" for i in range(11, max_sia, bin_width)
        ]
    iax.yaxis.set_major_locator(MaxNLocator(integer=True))
    iax.set_yticks(np.arange(len(sia_edges) - 1) + 0.5)
    iax.set_yticklabels(labels)
    # VAC size labels: 1, 2, ..., 10, 11-20, ...
    if bin_width == 1:
        labels = [str(i) for i in range(1, max_vac + 1)]
    else:
        labels = [str(i) for i in range(1, 11)] + [
            f"{i}-{i+bin_width-1}" for i in range(11, max_vac, bin_width)
        ]
    vax.yaxis.set_major_locator(MaxNLocator(integer=True))
    vax.set_yticks(np.arange(len(vac_edges) - 1) + 0.5)
    # vax.set_yticklabels(labels)
    # vax.yaxis.tick_right()

    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(iim, cax=cax, label="Mean number of clusters")

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


# endregion


# region Counting
def write_count_0d(
    path_oclusters: Path,
    out_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Processes clusters from a given file, separates them into vacancies and interstitials,
    computes histograms of their sizes, and optionally writes the histograms to a CSV file.

    Parameters
    ----------
    path_oclusters : Path
        Path to the input XYZ file containing cluster data.
    out_dir : Path, optional
        Directory where the histograms will be saved, by default None.

    Returns
    -------
    tuple[dtypes.INT_CHECK, dtypes.INT_CHECK]
        A tuple containing: the sizes of the interstitials and the sizes of the vacancies.
    """
    isizes, vsizes = np.array([], dtype=int), np.array([], dtype=int)
    for oclusters in XYZReader(path_oclusters):
        cond = oclusters["type"] == 0
        vacancies = oclusters[cond]
        if vacancies.size > 0:
            vsizes0 = vacancies["size"]
            vsizes = np.concatenate((vsizes, vsizes0))
        cond = np.invert(cond)
        interstitials = oclusters[cond]
        if interstitials.size > 0:
            isizes0 = interstitials["size"]
            isizes = np.concatenate((isizes, isizes0))
    sia_histogram, _, _ = __get_size_histogram(isizes)
    vac_histogram, _, _ = __get_size_histogram(vsizes)
    if out_dir:
        out_path = out_dir / "oclusters0D.csv"
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(",".join(map(str, sia_histogram)))
            file.write("\n")
            file.write(",".join(map(str, vac_histogram)))
            file.write("\n")
    return isizes, vsizes


def read_count_0d(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads interstitial and vacancy counts from a CSV file.

    Parameters
    ----------
    out_dir : Path
        Directory where the histograms were saved.

    Returns
    -------
    tuple[dtypes.INT_CHECK, dtypes.INT_CHECK]
        A tuple containing: the sizes of the interstitials and the sizes of the vacancies.
    """
    with open(out_dir / "oclusters0D.csv", "r", encoding="utf-8") as file:
        interstitials = np.array(
            list(map(int, file.readline()[:-1].split(","))), dtype=int
        )
        vacancies = np.array(list(map(int, file.readline()[:-1].split(","))), dtype=int)
    return interstitials, vacancies


# endregion
