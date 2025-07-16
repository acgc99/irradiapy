"""This is an example of Fe self-ion irradiation simulation."""

# pylint: disable=invalid-name

from pathlib import Path

import numpy as np

import irradiapy as irp

# Use the style of the package
irp.use_style(latex=False)
# TRIM.exe directory (parent folder)
srim_dir = Path()
# Database of MD cascades
# For example: CascadesDefectsDB/Fe/cascadesdb_fe_granberg_sand
# Donwloaded from: https://github.com/acgc99/CascadesDefectsDB
dir_mddb = Path()
compute_tdam = False  # True if the database does not include electronic stopping
# Where to store the results
out_dir = Path(r"srimpy")
# SRIM calculation mode
mode = "quick"
# Remove initial depth ion offsets.
remove_offsets = True
# True if you want to include defects placed outside the material.
outsiders = True
# Add the injected ion to the defects file.
add_injected = True
# dpa calculation mode to estminate the number of Frenkel pairs for residual energies.
dpaMode = irp.dpa.DpaMode.FERARC
# Target width, angstroms
width = 8000
# Fluence, ions/angstrom^2
fluence = 2.1e-1
# SIA and vacancy clustering cutoffs
Fe = irp.materials.get_material_by_name("Fe")
cutoff_sia = Fe.cutoff_sia
cutoff_vac = Fe.cutoff_vac

# Ion definition.
nions = int(1)
atomic_numbers = np.full(nions, 26)
energies = np.full(nions, 1e6)  # Ion energy, eV
depths = np.full(nions, 800.0)  # Initial depth, angstroms
ys = np.full(nions, 0.0)  # Initial y, angstroms
zs = np.full(nions, 0.0)  # Initial z, angstroms
cosxs = np.full(nions, 1.0)  # Initial x direction
cosys = np.full(nions, 0.0)  # Initial y direction
coszs = np.full(nions, 0.0)  # Initial z direction


# Iterative calculation criterion.
def criterion(**kwargs: dict) -> bool:
    """Criterion that determines when SRIM must be run again."""
    if kwargs["recoil_energy"] > 250e3:
        return False
    return True


# Output files.
path_db = out_dir / "srimdb.db"
path_debris = out_dir / "collisions.xyz"
path_aclusters = out_dir / "aclusters.xyz"
path_oclusters = out_dir / "oclusters.xyz"
if not out_dir.exists():
    out_dir.mkdir(parents=True)
if path_db.exists():
    path_db.unlink()


# Target
element = irp.materials.Fe.srim_element
layer = irp.srimpy.target.Layer(
    width=width, phase=0, density=element.density, elements=[element], stoichs=[1.0]
)
target = irp.srimpy.target.Target(layers=[layer])

runner = irp.srimpy.SRIMDB(path_db=path_db, calculation=mode, target=target)

runner.run(
    srim_dir,
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

irp.srimpy.debris.generate_debris(
    runner,
    dir_mddb=dir_mddb,
    compute_tdam=compute_tdam,
    path_debris=path_debris,
    dpa_mode=dpaMode,
    add_injected=add_injected,
    outsiders=outsiders,
)
irp.srimpy.injected.plot_injected(
    runner,
    bins=100,
    plot_path=out_dir / "injected.png",
    path_fit=out_dir / "injected_fit.txt",
)
irp.srimpy.pka.plot_pka_distribution(
    runner, plot_path=out_dir / "pka.png", fit_path=out_dir / "pka_fit.txt"
)
irp.srimpy.pka.plot_distances(
    runner, pka_e_lim=5000, plot_path=out_dir / "distances.png"
)
irp.srimpy.pka.plot_energy_depth(
    runner,
    plot_low_path=out_dir / "pka_low.png",
    plot_high_path=out_dir / "pka_high.png",
)
irp.srimpy.dpa.get_dpas(
    runner, path_db=path_db, path_debris=path_debris, fluence=fluence
)
irp.srimpy.dpa.plot_dpa(
    path_db=path_db, path_plot=out_dir / "dpa.png", path_fit=out_dir / "dpa_fit.txt"
)

irp.analysis.clusters.clusterize_file(
    path_debris=path_debris,
    path_aclusters=path_aclusters,
    path_oclusters=path_oclusters,
    cutoff_sia=cutoff_sia,
    cutoff_vac=cutoff_vac,
)
irp.analysis.clusters.get_clusters(path_oclusters, path_db)
irp.analysis.clusters.plot_results(path_db, out_dir, nions=nions)
