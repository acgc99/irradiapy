"""This is an example of an alloy being neutron irradiated using SPECTRA-PKA + SRIM + MD debris."""

# pylint: disable=invalid-name


import os
from pathlib import Path

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
calculation = "full"
# Tdam calculation mode to estimate the electronic energy losses.
damage_energy_mode = irpy.DamageEnergyMode.LINDHARD
# dpa calculation mode to estimate the number of Frenkel pairs for residual energies.
displacement_mode = irpy.DisplacementMode.FERARC
# Exclude these PKAs (atomic numbers) from creating vacancies at their original position
exclude_vacancies_ion = [1, 2]
# If a recoil energy exceeds this value, SRIM will be run again, electronvolts
max_recoil_energy = 250e3
# Maximum number of SRIM iterations
max_srim_iters = 10
# Width of the SRIM target, see irpy.spectrapka.Spectra2SRIM.run, angstroms
srim_width = 4e8

# Important distances
fp_dist = 4.0 * irpy.materials.Fe_bcc.ax

# Paths
root_dir = Path()
spectrapka_in_path = root_dir / "spectrapka.in"
spectrapka_events_path = root_dir / "config_events.pka"
debris_path = root_dir / "debris.xyz"
debris_merged_path = root_dir / "debris_merged.xyz"
path_dpa = root_dir / "dpa.txt"
analysisdb_path = root_dir / "analysis.db"

# endregion

# region Running

analysisdb = irpy.analysis.AnalysisDB(analysisdb_path)

spectrapka2srim = irpy.spectrapka.Spectra2SRIM()
recoilsdb = spectrapka2srim.run(
    spectrapka_in_path=spectrapka_in_path,
    spectrapka_events_path=spectrapka_events_path,
    root_dir=root_dir,
    srim_width=srim_width,
    calculation=calculation,
    max_recoil_energy=max_recoil_energy,
    density=irpy.materials.Fe_bcc.density,
    max_srim_iters=max_srim_iters,
)
recoilsdb = irpy.RecoilsDB(root_dir / "recoils.db")

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
irpy.utils.io.merge_lammps_snapshots(
    in_path=debris_path,
    out_path=debris_merged_path,
    overwrite=True,
)

# Get dpa
target = recoilsdb.load_target()
component = target[0]
width = component.width
natoms = (width // component.ax) ** 3 * 2  # number of atoms in bcc cell
scale = 24.0 * 365.0  # hours in a year, if debris are for an hour without diffusion

# NRT, arc and ferarc
nrt, arc, ferarc = 0.0, 0.0, 0.0
recoils = recoilsdb.read("recoils", what="x, atom_numb, recoil_energy")
for depth, atom_numb, recoil_energy in recoils:
    damage_energy = component.recoil_energy_to_damage_energy(
        recoil_energy,
        irpy.materials.ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
        damage_energy_mode,
    )
    nrt += component.damage_energy_to_displacements(
        damage_energy,
        mode=irpy.DisplacementMode.NRT,
    )
    arc += component.damage_energy_to_displacements(
        damage_energy,
        mode=irpy.DisplacementMode.ARC,
    )
    ferarc += component.damage_energy_to_displacements(
        damage_energy,
        mode=irpy.DisplacementMode.FERARC,
    )
nrt *= scale / natoms
arc *= scale / natoms
ferarc *= scale / natoms
# debris dpa
data = irpy.utils.io.get_last_reader(irpy.io.LAMMPSReader(debris_merged_path))
defects = data["atoms"]
nvacs = len(defects[defects["type"] == 0])
debris = nvacs / natoms * scale

print(f"NRT-dpa/year: {nrt:.2g}")
print(f"arc-dpa/year: {arc:.2g}")
print(f"fer-arc-dpa/year: {ferarc:.2g}")
print(f"debris-dpa/year: {debris:.2g}")

# endregion
