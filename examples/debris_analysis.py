"""Example to analyse a database of debris."""

import os
import subprocess
from pathlib import Path

import numpy as np

import irradiapy as irpy

subprocess.run("cls" if os.name == "nt" else "clear", shell=True, check=False)
irpy.config.use_style(latex=False)

# Root database of MD cascades
# For example: CascadesDefectsDB
# Donwloaded from: https://github.com/acgc99/CascadesDefectsDB
mddb_dir = Path()
electronic_interactions = "SRIM"
irpy.config.set_debris_database(mddb_dir, electronic_interactions)

recoil = irpy.materials.Fe
component = irpy.materials.Fe_bcc
cutoff_sia = (np.sqrt(2.0) + np.sqrt(11.0) / 2.0) * component.ax / 2.0
cutoff_vac = (1.0 + np.sqrt(2.0)) * component.ax / 2.0

debrismanager = irpy.analysis.DebrisManager(
    recoil=recoil,
    component=component,
    damage_energy_mode=irpy.DamageEnergyMode.SRIM,
    displacement_mode=irpy.DisplacementMode.FERARC,
    fp_dist=4.0 * component.ax,
)
debrismanager.displacements_number_plot(show=True)

debrismanager.cluster_sizes_plot(
    vacs_cutoff=cutoff_vac,
    sias_cutoff=cutoff_sia,
    show=True,
)
