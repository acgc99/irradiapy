"""Cluster analysis module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.image import NonUniformImage
from numpy import typing as npt

from irradiapy import dtypes
from irradiapy.analysis.analysisdb import AnalysisDB
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.io.lammpswriter import LAMMPSWriter
from irradiapy.utils.math import fit_power_law


def clusterize_atoms(
    atoms: dtypes.Atom, cutoff: float
) -> tuple[dtypes.Acluster, dtypes.Ocluster]:
    """Identify atom clusters.

    Note
    ----
    Atom clusters are the individual atoms with their cluster number, while object clusters are a
    single point representing the average position of the atoms in the cluster and the type of the
    cluster (the cluster type is taken from the first atom in the cluster).

    Parameters
    ----------
    atoms : dtypes.Atom
        Array of atoms with fields "type", "x", "y", "z".
    cutoff : float
        Cutoff distance for clustering.

    Returns
    -------
    tuple[dtypes.Acluster, dtypes.Ocluster]
        Atomic and object clusters.
    """
    cutoff2 = cutoff**2.0
    natoms = atoms.size
    aclusters = np.empty(natoms, dtype=dtypes.acluster)
    aclusters["x"] = atoms["x"]
    aclusters["y"] = atoms["y"]
    aclusters["z"] = atoms["z"]
    aclusters["type"] = atoms["type"]
    # Each atom is its own cluster initially
    aclusters["cluster"] = np.arange(1, natoms + 1)

    # Cluster identification
    for i in range(natoms):
        curr_cluster = aclusters[i]["cluster"]
        dists = (
            np.square(aclusters[i]["x"] - aclusters["x"][i + 1 :])
            + np.square(aclusters[i]["y"] - aclusters["y"][i + 1 :])
            + np.square(aclusters[i]["z"] - aclusters["z"][i + 1 :])
        )
        neighbours = aclusters["cluster"][np.where(dists <= cutoff2)[0] + i + 1]
        if neighbours.size:
            for neighbour in neighbours:
                if neighbour != curr_cluster:
                    aclusters["cluster"][
                        aclusters["cluster"] == neighbour
                    ] = curr_cluster

    # Rearrange cluster numbers to be consecutive starting from 1
    nclusters = np.unique(aclusters["cluster"])
    for i in range(nclusters.size):
        aclusters["cluster"][aclusters["cluster"] == nclusters[i]] = i + 1

    # Calculate object clusters
    oclusters = atom_to_object(aclusters)

    return aclusters, oclusters


def clusterize_file(
    defects_path: Path,
    cutoff_sia: float,
    cutoff_vac: float,
    aclusters_path: Path | None = None,
    oclusters_path: Path | None = None,
) -> None:
    """Finds defect clusters in the given file. Type 0 are vacancies, others are interstitials.

    Parameters
    ----------
    defects_path : Path
        Path of the file where defects are.
    cutoff_sia : float
        Cutoff distance for interstitials clustering.
    cutoff_vac : float
        Cutoff distance for vacancies clustering.
    aclusters_path : Path | None (default=None)
        Where atomic clusters will be stored if provided.
    oclusters_path : Path | None (default=None)
        Where object clusters will be stored if provided.
    """
    reader = LAMMPSReader(defects_path)
    nsim = 0

    if aclusters_path:
        awriter = LAMMPSWriter(aclusters_path)
    if oclusters_path:
        owriter = LAMMPSWriter(oclusters_path)

    for data_defects in reader:
        nsim += 1
        cond = data_defects["atoms"]["type"] == 0
        sia, vac = data_defects["atoms"][~cond], data_defects["atoms"][cond]
        iaclusters, ioclusters = clusterize_atoms(sia, cutoff_sia)
        vaclusters, voclusters = clusterize_atoms(vac, cutoff_vac)
        aclusters = np.concatenate((iaclusters, vaclusters))
        oclusters = np.concatenate((ioclusters, voclusters))

        if aclusters_path:
            data_aclusters = {}
            data_aclusters["timestep"] = data_defects["timestep"]
            data_aclusters["time"] = data_defects["time"]
            data_aclusters["boundary"] = data_defects["boundary"]
            data_aclusters["xlo"] = data_defects["xlo"]
            data_aclusters["xhi"] = data_defects["xhi"]
            data_aclusters["ylo"] = data_defects["ylo"]
            data_aclusters["yhi"] = data_defects["yhi"]
            data_aclusters["zlo"] = data_defects["zlo"]
            data_aclusters["zhi"] = data_defects["zhi"]
            data_aclusters["atoms"] = aclusters
            awriter.write(data_aclusters)

        if oclusters_path:
            data_oclusters = {}
            data_oclusters["timestep"] = data_defects["timestep"]
            data_oclusters["time"] = data_defects["time"]
            data_oclusters["boundary"] = data_defects["boundary"]
            data_oclusters["xlo"] = data_defects["xlo"]
            data_oclusters["xhi"] = data_defects["xhi"]
            data_oclusters["ylo"] = data_defects["ylo"]
            data_oclusters["yhi"] = data_defects["yhi"]
            data_oclusters["zlo"] = data_defects["zlo"]
            data_oclusters["zhi"] = data_defects["zhi"]
            data_oclusters["atoms"] = oclusters
            owriter.write(data_oclusters)

    if aclusters_path:
        awriter.close()
    if oclusters_path:
        owriter.close()


def atom_to_object(aclusters: dtypes.Acluster) -> dtypes.Ocluster:
    """Transform atom clusters into object clusters.

    Parameters
    ----------
    aclusters : dtypes.Acluster
        Atomic clusters.

    Returns
    -------
    dtypes.Ocluster
        Object clusters.
    """
    nclusters = np.unique(aclusters["cluster"])
    oclusters = np.empty(nclusters.size, dtype=dtypes.ocluster)
    for i in range(nclusters.size):
        acluster = aclusters[aclusters["cluster"] == nclusters[i]]
        oclusters[i]["x"] = np.mean(acluster["x"])
        oclusters[i]["y"] = np.mean(acluster["y"])
        oclusters[i]["z"] = np.mean(acluster["z"])
        oclusters[i]["type"] = acluster[0]["type"]
        oclusters[i]["size"] = acluster.size
    return oclusters


# region Clustering fraction


def depth_clustering_fraction_hist(
    analysisdb: AnalysisDB,
    oclusters_path: Path,
    axis: str,
    min_size: int,
    nbins: int = 100,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the clustering fraction as a function of depth and stores it in the analysis
    database.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database for storing results.
    oclusters_path : Path
        Path to the object clusters file.
    axis : str
        Axis along which to compute depth ('x', 'y', or 'z').
    min_size : int
        Minimum cluster size to be considered clustered.
    nbins : int (default=100)
        Number of bins for the depth histogram.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        A tuple containing depth centers and clustering fractions for SIAs and vacancies.
    """
    if min_size <= 1:
        raise ValueError("min_size must be at least 2.")
    isizes, vsizes = [], []
    idepths, vdepths = [], []
    reader = LAMMPSReader(oclusters_path)
    for data in reader:
        oclusters = data["atoms"]
        cond = oclusters["type"] == 0
        sia_clusters = oclusters[~cond]
        vac_clusters = oclusters[cond]
        isizes.append(sia_clusters["size"])
        vsizes.append(vac_clusters["size"])
        idepths.append(sia_clusters[axis])
        vdepths.append(vac_clusters[axis])
    isizes = np.concatenate(isizes)
    vsizes = np.concatenate(vsizes)
    idepths = np.concatenate(idepths)
    vdepths = np.concatenate(vdepths)

    depth_edges = np.histogram_bin_edges(np.concatenate((idepths, vdepths)), bins=nbins)
    depth_centers = (depth_edges[1:] + depth_edges[:-1]) / 2.0

    def fraction_by_depth(depths, sizes):
        nbins = len(depth_edges) - 1
        b = np.digitize(depths, depth_edges) - 1
        valid = (b >= 0) & (b < nbins)
        total = np.bincount(b[valid], weights=sizes[valid], minlength=nbins)
        clustered = np.bincount(
            b[valid & (sizes > min_size)],
            weights=sizes[valid & (sizes > min_size)],
            minlength=nbins,
        )
        return np.divide(clustered, total, out=np.full(nbins, np.nan), where=total > 0)

    ifraction = fraction_by_depth(idepths, isizes)
    vfraction = fraction_by_depth(vdepths, vsizes)
    analysisdb.save_clustering_fraction_hist(
        axis=axis,
        min_size=min_size,
        depth_centers=depth_centers,
        ifraction=ifraction,
        vfraction=vfraction,
    )

    return depth_centers, ifraction, vfraction


def depth_clustering_fraction_hist_plot(
    analysisdb: AnalysisDB,
    axis: str,
    min_size: int,
    plot_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plots the clustering fraction as a function of depth from the analysis database.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database for storing results.
    axis : str
        Axis along which to compute depth ('x', 'y', or 'z').
    min_size : int
        Minimum cluster size to be considered clustered.
    plot_path : Path, optional (default=None)
        Output path for the plot.
    show : bool, optional (default=False)
        Whether to show the plot.
    """
    depth_centers, ifraction, vfraction = analysisdb.load_clustering_fraction_hist(
        axis=axis,
        min_size=min_size,
    )

    fig = plt.figure()
    gs = fig.add_gridspec()
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(depth_centers, ifraction, label="SIA", marker=".")
    ax.scatter(depth_centers, vfraction, label="Vacancy", marker=".")
    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("Clustering fraction")
    ax.legend()
    fig.tight_layout()
    if plot_path:
        fig.savefig(plot_path)
    if show:
        plt.show()
    plt.close()


# endregion

# region Size depth


def depth_cluster_sizes_plot(
    oclusters_path: Path,
    axis: str,
    global_min_counts: int,
    depth_nbins: int = 100,
    log: bool = True,
    dpi: int = 300,
    plot_sias_path: Path | None = None,
    plot_vacs_path: Path | None = None,
    show: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Plots the cluster size distribution as a function of depth.

    Parameters
    ----------
    oclusters_path : Path
        Path to the object clusters file.
    axis : str
        Axis along which to compute depth ('x', 'y', or 'z').
    global_min_counts : int
        Minimum cluster size counts to show a bin in the plot.
    depth_nbins : int (default=100)
        Number of bins for the depth histogram.
    log : bool (default=True)
        Whether to use a logarithmic scale for the color map.
    dpi : int (default=300)
        DPI for saving the plots.
    plot_sias_path : Path | None (default=None)
        Output path for the interstitials plot.
    plot_vacs_path : Path | None (default=None)
        Output path for the vacancies plot.
    show : bool (default=False)
        Whether to show the plots.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        A tuple containing depth edges, interstitials histogram, and vacancies histogram. Rows
        correspond to depth bins, columns to cluster sizes.
    """

    def size_histogram_by_depth(depths, sizes, depth_edges):
        if sizes.size == 0:
            return np.zeros((depth_nbins, 0), dtype=np.float64)

        bin_idx = np.searchsorted(depth_edges, depths, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < depth_nbins)
        bin_idx = bin_idx[valid]
        valid_sizes = sizes[valid] - 1

        max_size = int(valid_sizes.max()) + 1
        flat = bin_idx * max_size + valid_sizes  # unique id for (bin, size)
        hist = np.bincount(flat, minlength=depth_nbins * max_size).reshape(
            depth_nbins, max_size
        )
        return hist

    def plot(
        depth_edges: npt.NDArray[np.float64],
        histogram: npt.NDArray[np.float64],
        title: str,
        plot_path: Path | None,
    ) -> None:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        depth_min, depth_max = float(depth_edges[0]), float(depth_edges[-1])
        depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0

        min_size, max_size = 1, histogram.shape[1]
        size_centers = np.arange(min_size, max_size + 1)
        y0, y1 = min_size - 0.5, max_size + 0.5
        x0, x1 = depth_min, depth_max

        histogram = histogram / nevents
        vmin = global_min_counts / nevents
        histogram = np.ma.masked_less_equal(histogram, vmin)
        vmax = histogram.max()
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)

        ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
        ax.set_ylabel("Cluster size")
        ax.set_ylim(y0, y1)
        ax.set_xlim(x0, x1)
        # If not added, the top bins overlap with the plot frame. This seems to be a matplotlib bug.
        ax.spines["top"].set_position(("outward", 1))

        im = NonUniformImage(
            ax,
            interpolation="nearest",
            norm=norm,
            extent=(x0, x1, y0, y1),
        )
        im.set_data(depth_centers, size_centers, histogram.T)
        ax.add_image(im)

        cbar = fig.colorbar(ax.images[0], cax=cax, orientation="vertical")
        cbar.set_label("Counts per event")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")

        ax.set_title(title)
        fig.tight_layout()
        if plot_path is not None:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)

    isizes, vsizes = [], []
    idepths, vdepths = [], []
    nevents = 0
    reader = LAMMPSReader(oclusters_path)
    for data_oclusters in reader:
        cond_v = data_oclusters["atoms"]["type"] == 0
        vacancies = data_oclusters["atoms"][cond_v]
        interstitials = data_oclusters["atoms"][~cond_v]
        vsizes.append(vacancies["size"])
        isizes.append(interstitials["size"])
        vdepths.append(vacancies[axis])
        idepths.append(interstitials[axis])
        nevents += 1
    isizes = np.concatenate(isizes)
    vsizes = np.concatenate(vsizes)
    idepths = np.concatenate(idepths)
    vdepths = np.concatenate(vdepths)

    depth_edges = np.histogram_bin_edges(
        np.concatenate((idepths, vdepths)),
        bins=depth_nbins,
    )
    ihist = size_histogram_by_depth(idepths, isizes, depth_edges)
    vhist = size_histogram_by_depth(vdepths, vsizes, depth_edges)

    plot(depth_edges, ihist, "Interstitials", plot_sias_path)
    plot(depth_edges, vhist, "Vacancies", plot_vacs_path)

    return depth_edges, ihist, vhist


# endregions

# region Size scaling law


def cluster_size_scaling_law(
    analysisdb: AnalysisDB,
    oclusters_path: Path,
    min_size: int,
    nbins: int,
) -> tuple[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """Calculates the cluster size distribution and fits power laws to small and large sizes.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database for storing results.
    oclusters_path : Path
        Path to the object clusters file.
    min_size : int
        Minimum cluster size to separate small and large sizes (inclusive). Bins above this size
        are logarithmic. Recommended value is around 5.
    nbins : int
        Number of bins for the large sizes histogram. This is usually a low number, depending on the
        data (< 10).

    Returns
    -------
    tuple[
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ]
    """
    reader = LAMMPSReader(oclusters_path)
    nevents = 0
    vac_sizes = []
    sia_sizes = []
    for data in reader:
        oclusters = data["atoms"]
        cond = oclusters["type"] == 0
        vac_sizes.append(oclusters["size"][cond])
        sia_sizes.append(oclusters["size"][~cond])
        nevents += 1
    vac_sizes = np.concatenate(vac_sizes)
    sia_sizes = np.concatenate(sia_sizes)

    def histogram_sizes(sizes: npt.NDArray[np.float64], vacancies: bool) -> None:
        """Create histogram of cluster sizes with two regions: small sizes with bins of size 1,
        and large sizes with logarithmic bins."""

        # Separate sizes from [0, min_size] and (min_size, inf)
        sizes_small = sizes[sizes <= min_size]
        sizes_large = sizes[sizes > min_size]

        # Small sizes histogram: bins of size 1
        max_size_small = sizes_small.max() if sizes_small.size > 0 else min_size
        size_edges_small = np.arange(1.0, max_size_small + 2) - 0.5
        hist_small, _ = np.histogram(sizes_small, bins=size_edges_small)
        size_centers_small = (size_edges_small[:-1] + size_edges_small[1:]) / 2.0

        # Large sizes histogram: logarithmic bins
        if sizes_large.size > 0:
            size_min_large = size_edges_small[-1]
            size_max_large = sizes_large.max()
            size_edges_large = np.logspace(
                np.log10(size_min_large),
                np.log10(size_max_large),
                nbins,
            )
            hist_large, _ = np.histogram(sizes_large, bins=size_edges_large)
            size_centers_large = (size_edges_large[:-1] + size_edges_large[1:]) / 2.0
        else:
            raise ValueError("No cluster sizes found greater than min_size.")

        # Normalize
        hist_small = hist_small / nevents
        hist_large = hist_large / nevents
        hist_large = hist_large / np.diff(size_edges_large)  # non-uniform bins

        size_centers = np.concatenate((size_centers_small, size_centers_large))
        hist = np.concatenate((hist_small, hist_large))

        analysisdb.save_cluster_size_hist(
            vacancies=vacancies,
            size_centers=size_centers,
            hist=hist,
        )

        # Fit to power law
        try:
            small_params, small_err_params, _ = fit_power_law(
                size_centers_small,
                hist_small,
            )
            analysisdb.save_cluster_size_hist_fit_params(
                vacancies,
                True,
                *small_params,
            )
            analysisdb.save_cluster_size_hist_fit_errors(
                vacancies,
                True,
                *small_err_params,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Fit failed: {exc}") from exc
        try:
            large_params, large_err_params, _ = fit_power_law(
                size_centers_large,
                hist_large,
            )
            analysisdb.save_cluster_size_hist_fit_params(
                vacancies,
                False,
                *large_params,
            )
            analysisdb.save_cluster_size_hist_fit_errors(
                vacancies,
                False,
                *large_err_params,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Fit failed: {exc}") from exc

        return size_centers, hist

    vacs_size_centers, vacs_hist = histogram_sizes(vac_sizes, True)
    sias_size_centers, sias_hist = histogram_sizes(sia_sizes, False)
    return (vacs_size_centers, vacs_hist), (sias_size_centers, sias_hist)


def cluster_size_scaling_law_plot(
    analysisdb: AnalysisDB,
    show: bool = False,
    vacs_plot_path: Path | None = None,
    sias_plot_path: Path | None = None,
    dpi: int = 300,
) -> None:
    """Plots the cluster size distribution with power law fits for small and large sizes.

    Parameters
    ----------
    analysisdb : AnalysisDB
        Database for storing results.
    show : bool, optional (default=False)
        Whether to show the plots.
    vacs_plot_path : Path, optional (default=None)
        Output path for the vacancies plot.
    sias_plot_path : Path, optional (default=None)
        Output path for the interstitials plot.
    dpi : int, optional (default=300)
        Dots per inch for the plot.
    """
    for vacancies in [True, False]:
        size_centers, hist = analysisdb.load_cluster_size_hist(vacancies)
        small_fit = analysisdb.load_cluster_size_hist_fit_function(vacancies, True)
        large_fit = analysisdb.load_cluster_size_hist_fit_function(vacancies, False)
        # Determine the split point between small and large sizes:
        # the small size edges are N.5 or N.0
        size_centers_mod = size_centers % 0.5
        split_idx = np.where(size_centers_mod == 0.0)[0][-1] + 1
        size_centers_small = size_centers[:split_idx]
        hist_small = hist[:split_idx]
        size_centers_large = size_centers[split_idx:]
        hist_large = hist[split_idx:]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        small_scatter = ax.scatter(
            size_centers_small, hist_small, label="Small sizes", marker="."
        )
        large_scatter = ax.scatter(
            size_centers_large, hist_large, label="Large sizes", marker="."
        )
        ax.plot(
            size_centers_small,
            small_fit(size_centers_small),
            color=small_scatter.get_facecolor()[0],
        )
        ax.plot(
            size_centers_large,
            large_fit(size_centers_large),
            color=large_scatter.get_facecolor()[0],
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Counts per event")
        ax.legend()
        plt.tight_layout()
        if vacs_plot_path is not None and vacancies:
            plt.savefig(vacs_plot_path, dpi=dpi)
        if sias_plot_path is not None and not vacancies:
            plt.savefig(sias_plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()


# endregion
