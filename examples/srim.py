"""This is an example of Fe self-ion irradiation simulation using SRIM + MD debris."""

# pylint: disable=invalid-name

import os
from copy import deepcopy
from pathlib import Path

import numpy as np

import irradiapy as irpy

os.system("cls" if os.name == "nt" else "clear")

# region Configuration

# Use the style of the package
irpy.config.use_style(latex=False)
# TRIM.exe directory (parent folder)
irpy.config.SRIM_DIR = Path()
# Database of MD cascades
# For example: CascadesDefectsDB/Fe/cascadesdb_fe_granberg_sand
# Donwloaded from: https://github.com/acgc99/CascadesDefectsDB
mddb_dir = Path()
# True if the database does not include electronic stopping
compute_damage_energy = False
# SRIM calculation mode
# full is recommended for multielemental targets or non-self-ion irradiation
calculation = "quick"
# Tdam calculation mode to estimate the electronic energy losses.
damage_energy_mode = irpy.DamageEnergyMode.SRIM
# dpa calculation mode to estimate the number of Frenkel pairs for residual energies.
displacement_mode = irpy.DisplacementMode.FERARC
# Exclude these PKAs (atomic numbers) from creating vacancies at their original position
exclude_vacancies_ion = [1, 2]
# If a recoil energy exceeds this value, SRIM will be run again, electronvolts
max_recoil_energy = 250e3
# Maximum number of SRIM iterations
max_srim_iters = 10
# Execution will fail if an ion is transmitted
fail_on_transmit = False
# Execution will fail if an ion is backscattered
fail_on_backscatt = False
# Target width, angstroms
width = 8e3
# Fluence, ions/angstrom^2
fluence = 2.1e-1

# Ion definition.
nions = int(2e3)
atomic_numbers = np.full(nions, 26)
energies = np.full(nions, 1.0e6)  # Ion energy, eV
depths = np.full(nions, 800.0)  # Initial depth, angstroms
ys = np.full(nions, 0.0)  # Initial y, angstroms
zs = np.full(nions, 0.0)  # Initial z, angstroms
cosxs = np.full(nions, 1.0)  # Initial x direction
cosys = np.full(nions, 0.0)  # Initial y direction
coszs = np.full(nions, 0.0)  # Initial z direction

# Target
Fe = irpy.materials.Fe
Fe_bcc = irpy.materials.Fe_bcc
component = deepcopy(Fe_bcc)
component.width = width
target = [component]
# Important distances
fp_dist = 4.0 * Fe_bcc.ax
cutoff_sia = (np.sqrt(2.0) + np.sqrt(11.0) / 2.0) * Fe_bcc.ax / 2.0
cutoff_vac = (1.0 + np.sqrt(2.0)) * Fe_bcc.ax / 2.0

# Paths
root_dir = Path()
debris_path = root_dir / "debris.xyz"
analysisdb_path = root_dir / "analysis.db"
aclusters_path = root_dir / "aclusters.xyz"
oclusters_path = root_dir / "oclusters.xyz"

# endregion

# region Running

analysisdb = irpy.analysis.AnalysisDB(analysisdb_path)

py2srim = irpy.srim.Py2SRIM()
recoilsdb = py2srim.run(
    root_dir=root_dir,
    target=target,
    calculation=calculation,
    atomic_numbers=atomic_numbers,
    energies=energies,
    depths=depths,
    ys=ys,
    zs=zs,
    cosxs=cosxs,
    coszs=coszs,
    cosys=cosys,
    max_recoil_energy=max_recoil_energy,
    max_srim_iters=max_srim_iters,
    fail_on_transmit=fail_on_transmit,
    fail_on_backscatt=fail_on_backscatt,
)
# recoilsdb = irpy.RecoilsDB(root_dir / "recoils.db")

print("Generating debris...")
irpy.analysis.debris.generate_debris(
    recoilsdb=recoilsdb,
    mddb_dir=mddb_dir,
    compute_damage_energy=compute_damage_energy,
    debris_path=debris_path,
    damage_energy_mode=damage_energy_mode,
    displacement_mode=displacement_mode,
    exclude_from_vacs=exclude_vacancies_ion,
    fp_dist=fp_dist,
)

print("Clustering debris...")
irpy.analysis.clusters.clusterize_file(
    defects_path=debris_path,
    cutoff_sia=cutoff_sia,
    cutoff_vac=cutoff_vac,
    aclusters_path=aclusters_path,
    oclusters_path=oclusters_path,
)

print("Clustering fraction vs depth...")
irpy.analysis.clusters.depth_clustering_fraction_hist(
    analysisdb=analysisdb,
    oclusters_path=oclusters_path,
    axis="x",
    min_size=2,
    nbins=50,
)
irpy.analysis.clusters.depth_clustering_fraction_hist_plot(
    analysisdb=analysisdb,
    axis="x",
    min_size=2,
    plot_path=root_dir / "clustering_fraction_x.png",
)

print("Cluster sizes vs depth...")
irpy.analysis.clusters.depth_cluster_sizes_plot(
    oclusters_path=oclusters_path,
    axis="x",
    global_min_counts=1,
    plot_sias_path=root_dir / "depth_sia_cluster_sizes_x.png",
    plot_vacs_path=root_dir / "depth_vac_cluster_sizes_x.png",
)

print("Cluster size scaling law...")
irpy.analysis.clusters.cluster_size_scaling_law(
    analysisdb=analysisdb,
    oclusters_path=oclusters_path,
    min_size=5,
    nbins=6,
)
irpy.analysis.clusters.cluster_size_scaling_law_plot(
    analysisdb=analysisdb,
    vacs_plot_path=root_dir / "cluster_size_scaling_law_vacancies.png",
    sias_plot_path=root_dir / "cluster_size_scaling_law_interstitials.png",
)


print("Recoil energies distribution...")
irpy.analysis.recoils.recoil_energies_hist(
    recoilsdb=recoilsdb,
    analysisdb=analysisdb,
    nbins=80,
)
irpy.analysis.recoils.recoil_energies_hist_plot(
    analysisdb=analysisdb,
    plot_path=root_dir / "recoil_energies.png",
    fit=True,
)

print("Recoil energy vs depth distribution...")
irpy.analysis.recoils.depth_recoil_energy_hist_plot(
    recoilsdb=recoilsdb,
    axis="x",
    plot_path=root_dir / "recoil_energy_depth_high.png",
    kev=True,
    log=True,
)
irpy.analysis.recoils.depth_recoil_energy_hist_plot(
    recoilsdb=recoilsdb,
    axis="x",
    plot_path=root_dir / "recoil_energy_depth_low.png",
    max_recoil_energy=200.0,
)

print("Recoils distances distribution...")
irpy.analysis.recoils.recoils_distances_hist_plot(
    recoilsdb,
    plot_path=root_dir / "recoils_distances.png",
    log=True,
)

print("Injected ions distribution...")
irpy.analysis.recoils.depth_injected_ions_hist(
    analysisdb=analysisdb,
    recoilsdb=recoilsdb,
    axis="x",
)
irpy.analysis.recoils.depth_injected_ions_hist_plot(
    analysisdb=analysisdb,
    axis="x",
    plot_path=root_dir / "injected_ions.png",
    fit=True,
)

print("Depth-dpa distribution...")
irpy.analysis.dpa.depth_dpa_hist(
    recoilsdb=recoilsdb,
    analysisdb=analysisdb,
    damage_energy_mode=damage_energy_mode,
    debris_path=debris_path,
    fluence=fluence,
    axis="x",
)
irpy.analysis.dpa.depth_dpa_hist_plot(
    analysisdb=analysisdb,
    axis="x",
    plot_path=root_dir / "depth_dpa.png",
    fits=True,
)
irpy.analysis.dpa.dpa(recoilsdb=recoilsdb, analysisdb=analysisdb)

# endregion
