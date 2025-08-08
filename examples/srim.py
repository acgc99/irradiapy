"""This is an example of Fe self-ion irradiation simulation using SRIM + MD debris."""

# pylint: disable=invalid-name,line-too-long

from pathlib import Path

import numpy as np

import irradiapy as irp

# Use the style of the package
irp.config.use_style(latex=False)
# TRIM.exe directory (parent folder)
irp.config.DIR_SRIM = Path()
# Database of MD cascades
# For example: CascadesDefectsDB/Fe/cascadesdb_fe_granberg_sand
# Donwloaded from: https://github.com/acgc99/CascadesDefectsDB
dir_mddb = Path()
# Output files.
out_dir = Path(r"srim")
path_db = out_dir / "srimdb.db"
path_debris = out_dir / "debris.xyz"
path_aclusters = out_dir / "aclusters.xyz"
path_oclusters = out_dir / "oclusters.xyz"
out_dir.mkdir(parents=True, exist_ok=True)
# path_db.unlink(missing_ok=True)

# True if the database does not include electronic stopping
compute_tdam = False
# SRIM calculation mode
mode = "quick"
# Remove initial depth ion offsets.
remove_offsets = True
# True if you want to include defects placed outside the material.
outsiders = True
# Add the injected ion to the defects file.
add_injected = True
# Tdam calculation mode to estimate the electronic energy losses.
tdam_mode = irp.materials.Material.TdamMode.SRIM
# dpa calculation mode to estimate the number of Frenkel pairs for residual energies.
dpa_mode = irp.materials.Material.DpaMode.FERARC
# Target width, angstroms
width = 8000
# Fluence, ions/angstrom^2
fluence = 2.1e-1
# SIA and vacancy clustering cutoffs
Fe = irp.materials.Fe
cutoff_sia = Fe.cutoff_sia
cutoff_vac = Fe.cutoff_vac

# region SRIM

# Ion definition.
nions = int(1e1)
atomic_numbers = np.full(nions, 26)
energies = np.full(nions, 1e6)  # Ion energy, eV
depths = np.full(nions, 800.0)  # Initial depth, angstroms
ys = np.full(nions, 0.0)  # Initial y, angstroms
zs = np.full(nions, 0.0)  # Initial z, angstroms
cosxs = np.full(nions, 1.0)  # Initial x direction
cosys = np.full(nions, 0.0)  # Initial y direction
coszs = np.full(nions, 0.0)  # Initial z direction

# Target
element = irp.materials.Fe.srim_element
layer = irp.srim.target.Layer(
    width=width, phase=0, density=element.density, elements=[element], stoichs=[1.0]
)
target = irp.srim.target.Target(layers=[layer])


# Iterative calculation criterion.
def criterion(**kwargs: dict) -> bool:
    """Criterion that determines when SRIM must be run again."""
    recoil_energy = kwargs["recoil_energy"]
    if isinstance(recoil_energy, float) and recoil_energy > 250e3:
        return False
    return True


srimdb = irp.srim.SRIMDB(path_db=path_db, calculation=mode, target=target)
# Run
srimdb.run(
    criterion,
    atomic_numbers,
    energies,
    remove_offsets,
    depths,
    ys,
    zs,
    cosxs,
    cosys,
    coszs,
)

irp.srim.analysis.plot_injected(
    srimdb,
    bins=100,
    plot_path=out_dir / "injected.png",
    path_fit=out_dir / "injected_fit.txt",
)
irp.srim.analysis.plot_pka_distribution(
    srimdb, plot_path=out_dir / "pka.png", fit_path=out_dir / "pka_fit.txt"
)
irp.srim.analysis.plot_distances(
    srimdb, pka_e_lim=5000, plot_path=out_dir / "distances.png"
)
irp.srim.analysis.plot_energy_depth(
    srimdb,
    plot_low_path=out_dir / "pka_low.png",
    plot_high_path=out_dir / "pka_high.png",
)

# endregion

# region Debris and dpa

irp.srim.analysis.generate_debris(
    srimdb,
    dir_mddb=dir_mddb,
    compute_tdam=compute_tdam,
    path_debris=path_debris,
    tdam_mode=tdam_mode,
    dpa_mode=dpa_mode,
    add_injected=add_injected,
    outsiders=outsiders,
    seed=0,
)

irp.analysis.dpa.get_dpa_1d(
    srimdb=srimdb,
    path_debris=path_debris,
    path_db=path_db,
    fluence=fluence,
)
irp.analysis.dpa.plot_dpa_1d(
    path_db=path_db,
    path_plot=out_dir / "dpa_1d.png",
    path_fit=out_dir / "dpa_1d_fit.txt",
)

# endregion

# region Clusters

irp.analysis.clusters.clusterize_file(
    path_defects=path_debris,
    path_aclusters=path_aclusters,
    path_oclusters=path_oclusters,
    cutoff_sia=cutoff_sia,
    cutoff_vac=cutoff_vac,
)

irp.analysis.clusters.get_clusters_0d(
    path_oclusters=path_oclusters, path_db=path_db, scale=1.0 / nions
)
irp.analysis.clusters.get_clusters_1d(
    path_oclusters=path_oclusters, path_db=path_db, depth_bins=100, scale=1.0 / nions
)

irp.analysis.clusters.plot_size_1d(
    path_db,
    path_sias=out_dir / "depth_size_sias.png",
    path_vacs=out_dir / "depth_size_vacs.png",
)
irp.analysis.clusters.plot_clustering_fraction_1d(
    path_db,
    path_plot=out_dir / "clustering_fraction.png",
)

# endregion

# region MDDB analysis

irp.analysis.defects.plot_mddb_nd(
    target_dir=dir_mddb,
    mat_pka=Fe,
    mat_target=Fe,
    path_plot=out_dir / "mddb_nd.png",
)
irp.analysis.clusters.plot_mddb_cluster_size(
    dir_mddb,
    cutoff_sia,
    cutoff_vac,
    rel=True,
    sia_bin_width=10,
    vac_bin_width=10,
    path_sias=out_dir / "rel_mddb_sia_size_distribution.png",
    path_vacs=out_dir / "rel_mddb_vac_size_distribution.png",
)
irp.analysis.clusters.plot_mddb_cluster_size(
    dir_mddb,
    cutoff_sia,
    cutoff_vac,
    rel=False,
    sia_bin_width=10,
    vac_bin_width=10,
    path_sias=out_dir / "abs_mddb_sia_size_distribution.png",
    path_vacs=out_dir / "abs_mddb_vac_size_distribution.png",
)

# endregion
