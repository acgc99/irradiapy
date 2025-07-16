"""Cluster analysis module."""

from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.image import NonUniformImage
from numpy import typing as npt

from irradiapy import dtypes, utils
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.io.lammpswriter import LAMMPSWriter

# region Clustering


def clusterize(
    defects: dtypes.Atom, cutoff: float
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
    cutoff2 = cutoff**2
    natoms = defects.size
    aclusters = np.empty(natoms, dtype=dtypes.acluster)
    aclusters["x"] = defects["x"]
    aclusters["y"] = defects["y"]
    aclusters["z"] = defects["z"]
    aclusters["type"] = defects["type"]

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
    path_collisions: Path,
    cutoff_sia: float,
    cutoff_vac: float,
    path_aclusters: Path,
    path_oclusters: Path,
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
    """
    reader = LAMMPSReader(path_collisions)
    nsim = 0
    with (
        LAMMPSWriter(path_aclusters) as awriter,
        LAMMPSWriter(path_oclusters) as owriter,
    ):
        for data_defects in reader:
            nsim += 1
            cond = data_defects["atoms"]["type"] == 0
            sia, vac = data_defects["atoms"][~cond], data_defects["atoms"][cond]
            iaclusters, ioclusters = clusterize(sia, cutoff_sia)
            vaclusters, voclusters = clusterize(vac, cutoff_vac)
            aclusters = np.concatenate((iaclusters, vaclusters))
            oclusters = np.concatenate((ioclusters, voclusters))

            data_aclusters = defaultdict(None)
            data_aclusters["timestep"] = data_defects["timestep"]
            data_aclusters["time"] = data_defects["time"]
            data_aclusters["natoms"] = len(aclusters)
            data_aclusters["boundary"] = data_defects["boundary"]
            data_aclusters["xlo"] = data_defects["xlo"]
            data_aclusters["xhi"] = data_defects["xhi"]
            data_aclusters["ylo"] = data_defects["ylo"]
            data_aclusters["yhi"] = data_defects["yhi"]
            data_aclusters["zlo"] = data_defects["zlo"]
            data_aclusters["zhi"] = data_defects["zhi"]
            data_aclusters["atoms"] = aclusters
            awriter.write(data_aclusters)

            data_oclusters = defaultdict(None)
            data_oclusters["timestep"] = data_defects["timestep"]
            data_oclusters["time"] = data_defects["time"]
            data_oclusters["natoms"] = len(oclusters)
            data_oclusters["boundary"] = data_defects["boundary"]
            data_oclusters["xlo"] = data_defects["xlo"]
            data_oclusters["xhi"] = data_defects["xhi"]
            data_oclusters["ylo"] = data_defects["ylo"]
            data_oclusters["yhi"] = data_defects["yhi"]
            data_oclusters["zlo"] = data_defects["zlo"]
            data_oclusters["zhi"] = data_defects["zhi"]
            data_oclusters["atoms"] = oclusters
            owriter.write(data_oclusters)


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


# region Histograms


def get_clusters_0d(
    path_oclusters: Path,
    path_db: Path,
    scale: float = 1.0,
) -> None:
    """Perform a cluster size histogram for interstitials and vacancies.

    This is useful for cluster analysis and cluster dynamics codes.

    Parameters
    ----------
    path_oclusters : Path
        Path to the input file containing object clusters.
    path_db : Path
        Path to the output SQLite database file.
    scale : float, optional (default=1.0)
        All bin counts are multiplied by this factor. Useful for normalization.
    """

    # Extract data
    isizes, vsizes = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    reader = LAMMPSReader(path_oclusters)
    for data_oclusters in reader:
        cond = data_oclusters["atoms"]["type"] == 0
        vacancies = data_oclusters["atoms"][cond]
        if vacancies.size > 0:
            vsizes = np.concatenate((vsizes, vacancies["size"]))
        interstitials = data_oclusters["atoms"][~cond]
        if interstitials.size > 0:
            isizes = np.concatenate((isizes, interstitials["size"]))

    # Interstitials
    # Histogram sizes
    max_isize = isizes.max()
    # [1:] because size 0 does not count; max_isize + 1 because np.bincount counts from 0
    isize_histogram = np.bincount(isizes, minlength=max_isize + 1)[1:]

    # Vacancies
    # Histogram sizes
    max_vsize = vsizes.max()
    # [1:] because size 0 does not count; max_vsize + 1 because np.bincount counts from 0
    vsize_histogram = np.bincount(vsizes, minlength=max_vsize + 1)[1:]

    # Save to database
    utils.sqlite.insert_array(
        path_db,
        "clusters_0D",
        interstitials=isize_histogram * scale,
        vacancies=vsize_histogram * scale,
    )


def get_clusters_1d(
    path_oclusters: Path,
    path_db: Path,
    axis: str = "x",
    depth_bins: int = 100,
    depth_offset: float = 0.0,
    min_depth: None | float = None,
    max_depth: None | float = None,
    scale: float = 1.0,
) -> None:
    """Perform a cluster size histogram as a function of depth along a specified axis.

    This is useful for cluster analysis and cluster dynamics codes.

    Parameters
    ----------
    path_oclusters : Path
        Path to the input file containing object clusters.
    path_db : Path
        Path to the output SQLite database file.
    axis : str, optional (default="x")
        Axis along which to measure depth. It can be `"x"`, `"y"`, or `"z"`.
    depth_bins : int, optional (default=100)
        Number of bins for the depth histogram.
    depth_offset : float, optional (default=0.0)
        Offset to add to the depth values.
    min_depth : float, optional (default=None)
        Minimum depth to consider. If `None`, the minimum depth from the data is used.
    max_depth : float, optional (default=None)
        Maximum depth to consider. If `None`, the maximum depth from the data is used.
    scale : float, optional (default=1.0)
        All bin counts are multiplied by this factor. Useful for normalization.
    """

    # Extract data
    isizes, vsizes = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    idepths, vdepths = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    reader = LAMMPSReader(path_oclusters)
    for data_oclusters in reader:
        cond = data_oclusters["atoms"]["type"] == 0
        vacancies = data_oclusters["atoms"][cond]
        if vacancies.size > 0:
            vsizes = np.concatenate((vsizes, vacancies["size"]))
            vdepths = np.concatenate((vdepths, vacancies[axis]))
        interstitials = data_oclusters["atoms"][~cond]
        if interstitials.size > 0:
            isizes = np.concatenate((isizes, interstitials["size"]))
            idepths = np.concatenate((idepths, interstitials[axis]))
    vdepths += depth_offset
    idepths += depth_offset

    # Histogram depths
    if min_depth is None:
        min_depth = min(idepths.min(), vdepths.min())
    if max_depth is None:
        max_depth = max(idepths.max(), vdepths.max())
    depth_edges = np.histogram_bin_edges(
        idepths, bins=depth_bins, range=(min_depth, max_depth)
    )

    # Interstitials
    # For each depth bin, count the number of each cluster size
    max_isize = isizes.max()
    isize_histogram = np.zeros((depth_bins, max_isize), dtype=np.float64)
    for i in range(depth_bins):
        cond = (idepths >= depth_edges[i]) & (idepths < depth_edges[i + 1])
        # [1:] because size 0 does not count; max_isize + 1 because np.bincount counts from 0
        isize_histogram[i, :] = np.bincount(isizes[cond], minlength=max_isize + 1)[1:]

    # Vacancies
    # For each depth bin, count the number of each cluster size
    max_vsize = vsizes.max()
    vsize_histogram = np.zeros((depth_bins, max_vsize), dtype=np.float64)
    for i in range(depth_bins):
        cond = (vdepths >= depth_edges[i]) & (vdepths < depth_edges[i + 1])
        # [1:] because size 0 does not count; max_vsize + 1 because np.bincount counts from 0
        vsize_histogram[i, :] = np.bincount(vsizes[cond], minlength=max_vsize + 1)[1:]

    # Save to database
    utils.sqlite.insert_array(
        path_db,
        f"clusters_1D_{axis}",
        depth_edges=depth_edges,
        interstitials=isize_histogram * scale,
        vacancies=vsize_histogram * scale,
    )


def read_clusters_0d(path_db: Path) -> dict[str, npt.NDArray[np.float64]]:
    """Read the 0D cluster size histogram from the database.

    This is useful for cluster analysis and cluster dynamics codes.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        A dictionary containing:
        - "interstitials": The histogram of interstitial sizes.
        - "vacancies": The histogram of vacancy sizes.
    """
    data = utils.sqlite.read_array(path_db, "clusters_0D")
    return data


def read_clusters_1d(
    path_db: Path, axis: str = "x"
) -> dict[str, npt.NDArray[np.float64]]:
    """Read the 1D cluster size histogram from the database.

    This is useful for cluster analysis and cluster dynamics codes.

    Parameters
    ----------
    path_db : Path
        Path to the SQLite database file.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`,or `"z"`.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        A dictionary containing:
        - "depth_edges": The edges of the depth bins.
        - "interstitials": The histogram of interstitial sizes for each depth bin.
        - "vacancies": The histogram of vacancy sizes for each depth bin.
    """
    data = utils.sqlite.read_array(path_db, f"clusters_1D_{axis}")
    return data


# region Plots


def plot_size_1d(
    db_path: Path,
    path_sias: None | Path = None,
    path_vacs: None | Path = None,
    axis: str = "x",
    depth_offset: float = 0.0,
    transpose: bool = True,
    dpi: int = 300,
) -> None:
    """Plot the depth-size 1D histogram for interstitials and vacancies.

    Note
    ----
    The color bar's label shows "Mean counts", then it assumes that in
    `irradiapy.analysis.clusters.get_clusters_1d` a proper `scale` parameter was used.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file containing the cluster data generated by
        `irradiapy.analysis.clusters.get_clusters_1d`.
    path_sias : Path, optional (default=None)
        Path to save the interstitials plot. If `None`, the plot is shown instead of saved.
    path_vacs : Path, optional (default=None)
        Path to save the vacancies plot. If `None`, the plot is shown instead of saved.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`, or `"z"`.
    depth_offset : float, optional (default=0.0)
        Offset to add to the depth values.
    transpose : bool, optional (default=True)
        If `True`, the depth is on the x-axis and size on the y-axis. If `False`, the
        axes are swapped.
    dpi : int, optional (default=300)
        Dots per inch.
    """

    def plot(
        depth_edges: npt.NDArray[np.float64],
        histogram: npt.NDArray[np.float64],
        title: str,
        path_plot: Path,
        transpose: bool = True,
    ) -> None:

        min_depth, max_depth = depth_edges[0], depth_edges[-1]
        depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0
        min_size, max_size = 1, histogram.shape[1]
        size_centers = np.arange(min_size, max_size + 1)

        if transpose:
            fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(1, 2, width_ratios=[1, 0.05])

            ax = fig.add_subplot(gs[0, 0])
            ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
            ax.set_ylabel("Cluster size")
            ax.set_ylim(min_size, max_size)
            ax.set_xlim(min_depth, max_depth)

            im = NonUniformImage(
                ax,
                interpolation="nearest",
                norm="log",
                extent=(min_depth, max_depth, min_size, max_size),
            )
            im.set_data(depth_centers, size_centers, histogram.T)
            ax.add_image(im)

            cax = fig.add_subplot(gs[0, 1])
            cbar = fig.colorbar(ax.images[0], cax=cax, orientation="vertical")
            cbar.set_label("Counts per ion")
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.ax.yaxis.set_label_position("right")

        else:
            fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(1, 2, width_ratios=[1, 0.05])

            ax = fig.add_subplot(gs[0, 0])
            ax.set_ylabel(r"Depth ($\mathrm{\AA}$)")
            ax.set_xlabel("Cluster size")
            ax.set_xlim(min_size, max_size)
            ax.set_ylim(min_depth, max_depth)

            im = NonUniformImage(
                ax,
                interpolation="nearest",
                norm="log",
                extent=(min_size, max_size, min_depth, max_depth),
            )
            im.set_data(size_centers, depth_centers, histogram)
            ax.add_image(im)

            cax = fig.add_subplot(gs[0, 1])
            cbar = fig.colorbar(ax.images[0], cax=cax, orientation="vertical")
            cbar.set_label("Counts per ion")
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.ax.yaxis.set_label_position("right")

        fig.suptitle(title)
        fig.tight_layout()
        if path_plot:
            plt.savefig(path_plot, dpi=dpi)
        else:
            plt.show()
        plt.close(fig)

    data = read_clusters_1d(db_path, axis=axis)
    depth_edges = data["depth_edges"] + depth_offset
    interstitials = data["interstitials"]
    vacancies = data["vacancies"]
    plot(depth_edges, interstitials, "Interstitials", path_sias, transpose)
    plot(depth_edges, vacancies, "Vacancies", path_vacs, transpose)


def plot_clustering_fraction_1d(
    db_path: Path,
    path_plot: None | Path = None,
    axis: str = "x",
    depth_offset: float = 0.0,
    dpi: int = 300,
) -> None:
    """Plot the clustering fraction as a function of depth.

    Note
    ----
    The clustering fraction is defined as the ratio of the number of defects in clusters of size
    greater than 1 to the number of unclustered defects.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file containing the cluster data generated by
        `irradiapy.analysis.clusters.get_clusters_1d`.
    path_plot : Path, optional (default=None)
        Path to save the clustering fraction plot. If `None`, the plot is shown instead of saved.
    axis : str, optional (default="x")
        Axis along which the histogram was computed. It can be `"x"`, `"y"`, or `"z"`.
    depth_offset : float, optional (default=0.0)
        Offset to add to the depth values.
    dpi : int, optional (default=300)
        Dots per inch.
    """
    data = read_clusters_1d(db_path, axis=axis)
    depth_edges = data["depth_edges"] + depth_offset
    interstitials = data["interstitials"]
    vacancies = data["vacancies"]

    min_depth, max_depth = depth_edges[0], depth_edges[-1]
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2.0

    min_size, max_size = 1, interstitials.shape[1]
    size_centers = np.arange(min_size, max_size + 1)
    interstitials *= size_centers
    clustering_fraction_sia = np.sum(interstitials[:, 1:], axis=1) / interstitials[:, 0]

    min_size, max_size = 1, vacancies.shape[1]
    size_centers = np.arange(min_size, max_size + 1)
    vacancies *= size_centers
    clustering_fraction_vac = np.sum(vacancies[:, 1:], axis=1) / vacancies[:, 0]

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Depth ($\mathrm{\AA}$)")
    ax.set_ylabel("Clustering fraction")
    ax.plot(
        depth_centers,
        clustering_fraction_sia,
        marker="o",
        linestyle="none",
        label="Interstitials",
    )
    ax.plot(
        depth_centers,
        clustering_fraction_vac,
        marker="o",
        linestyle="none",
        label="Vacancies",
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(min_depth, max_depth)
    ax.legend()
    fig.suptitle("Clustering fraction")
    fig.tight_layout()
    if path_plot:
        plt.savefig(path_plot, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def plot_mddb_cluster_size(
    target_dir: Path,
    sia_cutoff: float,
    vac_cutoff: float,
    rel: bool,
    sia_bin_width: int = 10,
    vac_bin_width: int = 10,
    path_sias: None | Path = None,
    path_vacs: None | Path = None,
    max_sia: None | int = None,
    max_vac: None | int = None,
    vmin: float = 1e-5,
    dpi: int = 300,
) -> None:
    """Plot the cluster size distribution from a molecular dynamics database.

    Parameters
    ----------
    target_dir : Path
        Directory containing the MD database.
    sia_cutoff : float
        Cutoff distance for interstitial cluster identification.
    vac_cutoff : float
        Cutoff distance for vacancy cluster identification.
    rel : bool
        If `True`, each size bin is the number of clusters of that size divided by the total number
        of clusters for the given energy. If `False`, each size bin is the number of clusters of
        that size divided by the total number of simulations of the corresponding energy.
    sia_bin_width : int, optional (default=10)
        Bin size for interstitial histogram.
    vac_bin_width : int, optional (default=10)
        Bin size for vacancy histogram.
    path_sias : Path, optional (default=None)
        Path to save the interstitials size distribution plot. If `None`, the plot is shown instead
        of saved.
    path_vacs : Path, optional (default=None)
        Path to save the vacancies size distribution plot. If `None`, the plot is shown instead
        of saved.
    max_sia : int, optional (default=None)
        Sets an upper limit for interstitial cluster sizes. It has to be greater than the maximum
        interstitial cluster size in the data. If `None`, it is determined from the data. Useful to
        compare multiple databases with different maximum sizes.
    max_vac : int, optional (default=None)
        Sets an upper limit for vacancy cluster sizes. It has to be greater than the maximum
        vacancy cluster size in the data. If `None`, it is determined from the data. Useful to
        compare multiple databases with different maximum sizes.
    vmin : float, optional (default=1e-5)
        Minimum value for the color scale if `rel` is `True`.
    dpi : int, optional (default=300)
        Dots per inch.
    """

    # Extract data from xyz files
    sizes_all_raw = defaultdict(defaultdict)
    nfiles = defaultdict(int)
    for energy_dir in target_dir.glob("*"):
        if not energy_dir.is_dir():
            continue
        energy = int(energy_dir.name)
        sizes_all_raw[energy] = {
            "vacs": np.empty(0, dtype=np.int32),
            "sias": np.empty(0, dtype=np.int32),
        }
        for path_defects in energy_dir.glob("*.xyz"):
            nfiles[energy] += 1
            data_defects = utils.io.get_last_lammps_dump(path_defects)

            cond = data_defects["atoms"]["type"] == 0
            vacs = data_defects["atoms"][cond]
            sias = data_defects["atoms"][~cond]

            _, vac_oclusters = clusterize(vacs, vac_cutoff)
            _, sia_oclusters = clusterize(sias, sia_cutoff)

            sizes_all_raw[energy]["vacs"] = np.append(
                sizes_all_raw[energy]["vacs"],
                vac_oclusters["size"],
            )
            sizes_all_raw[energy]["sias"] = np.append(
                sizes_all_raw[energy]["sias"],
                sia_oclusters["size"],
            )

    # Maximum cluster sizes
    if max_sia is None or max_vac is None:
        max_sia = max_vac = 0
        for energy, siasvacs in sizes_all_raw.items():
            max_sia = max(max_sia, siasvacs["sias"].max())
            max_vac = max(max_vac, siasvacs["vacs"].max())

    # Bincount sizes (exclude size 0)
    sizes_all_bincount = defaultdict(defaultdict)
    for energy, sizes in sizes_all_raw.items():
        sizes_all_bincount[energy]["vacs"] = np.bincount(
            sizes["vacs"], minlength=max_vac
        )[1:]
        sizes_all_bincount[energy]["sias"] = np.bincount(
            sizes["sias"], minlength=max_sia
        )[1:]

    # Sort energies
    epkas = np.array(
        [int(path.name) for path in target_dir.glob("*") if path.is_dir()],
        dtype=np.int32,
    )
    epkas.sort()
    # Energy binning
    nepka_bins = len(epkas)

    # Round to closest highest multiple of bin_width starting from 10,
    # where binning changes
    # if bin_width = 10, then 83 > 90
    # if bin_width = 10, then 57 > 60
    max_sia = 10 + int(np.ceil((max_sia - 10) / sia_bin_width) * sia_bin_width)
    max_vac = 10 + int(np.ceil((max_vac - 10) / vac_bin_width) * vac_bin_width)

    # Histogram bin edges
    sia_edges = np.concatenate(
        (
            np.arange(0.5, 10.5),
            np.arange(10.5, max_sia + sia_bin_width + 0.5, sia_bin_width),
        )
    )
    vac_edges = np.concatenate(
        (
            np.arange(0.5, 10.5),
            np.arange(10.5, max_vac + vac_bin_width + 0.5, vac_bin_width),
        )
    )

    # row: energy, column: size
    ihist = np.empty((nepka_bins, len(sia_edges) - 1), dtype=np.float64)
    vhist = np.empty((nepka_bins, len(vac_edges) - 1), dtype=np.float64)
    for i, epka in enumerate(epkas):
        siasvacs = sizes_all_raw[epka]
        if rel:
            hist, _ = np.histogram(siasvacs["sias"], bins=sia_edges, density=True)
            ihist[i] = hist
            hist, _ = np.histogram(siasvacs["vacs"], bins=vac_edges, density=True)
            vhist[i] = hist
        else:
            hist, _ = np.histogram(siasvacs["sias"], bins=sia_edges, density=False)
            ihist[i] = hist / nfiles[epka]
            hist, _ = np.histogram(siasvacs["vacs"], bins=vac_edges, density=False)
            vhist[i] = hist / nfiles[epka]

    def plot(
        hist: npt.NDArray[np.float64],
        edges: npt.NDArray[np.float64],
        max_size: int,
        bin_width: int,
        title: str,
        path: None | Path = None,
    ) -> None:

        hist = np.ma.masked_where(hist == 0, hist, copy=False)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Cluster size")

        if rel:
            norm = mcolors.LogNorm(vmin=vmin, vmax=1.0)
        else:
            norm = mcolors.LogNorm(vmin=hist.min(), vmax=hist.max())
        im = ax.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap="viridis",
            extent=[0, nepka_bins, 0, hist.shape[1]],
        )

        xticks = np.arange(nepka_bins) + 0.5
        xlabels = [f"{epka/1000.0:<.1f}" for epka in epkas]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # Size labels: 1, 2, ..., 10, 11-20, ...
        if bin_width == 1:
            ylabels = [str(i) for i in range(1, max_size + 1)]
        else:
            ylabels = [str(i) for i in range(1, 11)] + [
                f"{i}-{i+bin_width-1}" for i in range(11, max_size, bin_width)
            ]
        yticks = np.arange(len(edges) - 1) + 0.5
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        cax = fig.add_subplot(gs[0, 1])
        fig.colorbar(im, cax=cax, label="% of counts per energy")

        fig.suptitle(title)
        fig.tight_layout()
        if path:
            plt.savefig(path, dpi=dpi)
        else:
            plt.show()
        plt.close(fig)

    plot(ihist, sia_edges, max_sia, sia_bin_width, "Interstitials", path=path_sias)
    plot(vhist, vac_edges, max_vac, vac_bin_width, "Vacancies", path=path_vacs)
