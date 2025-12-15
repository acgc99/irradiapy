"""Utility functions related to SPECTRA-PKa + SRIM data analysis and debris production."""

from pathlib import Path

import numpy as np

from irradiapy.enums import DamageEnergyMode, DpaMode
from irradiapy.io import LAMMPSReader
from irradiapy.materials import ELEMENT_BY_ATOMIC_NUMBER, Component
from irradiapy.recoilsdb import RecoilsDB
from irradiapy.utils.io import get_last_reader


def get_dpas(
    recoilsdb: RecoilsDB,
    path_debris: Path | None = None,
    component: Component | None = None,
    damage_energy_mode: DamageEnergyMode = DamageEnergyMode.LINDHARD,
) -> tuple[float, ...]:
    """Turns SPECTRA-PKA + SRIM's recoils into the different dpa metrics.

    Parameters
    ----------
    spectrapka2srim : Spectra2SRIM
        The SPECTRA-PKA to SRIM object containing the recoils database.
    path_debris : Path | None (default=None)
       Path to the debris LAMMPS file to compute the debris dpa, by default None.
    component : Component | None (default=None)
        Material object representing the target material,
        by default None (taken from spectrapka2srim).
    damage_energy_mode : DamageEnergyMode (default=DamageEnergyMode.LINDHARD)
        The damage energy calculation mode.

    Returns
    -------
    tuple[float, ...]
        A tuple containing the total NRT, arc, fer-arc (and debris) dpa values
        in the simulation box.
    """
    component = recoilsdb.load_target()[0]

    natoms_cell = 1  # number of atoms per unit cell
    if component.structure == "bcc":
        natoms_cell = 2
    elif component.structure == "fcc":
        natoms_cell = 4
    elif component.structure == "hcp":
        natoms_cell = 2
    else:
        raise ValueError(f"Unknown lattice type: {component.structure}")
    nsize = component.width // component.ax  # number of unit cells per side
    natoms = nsize**3 * natoms_cell

    nrt = 0
    arc = 0
    ferarc = 0
    recoils = recoilsdb.read("recoils", what="atom_numb, recoil_energy")
    for atom_numb, recoil_energy in recoils:
        damage_energy = component.recoil_energy_to_damage_energy(
            recoil_energy,
            ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
            damage_energy_mode,
        )
        nrt += component.damage_energy_to_dpa(damage_energy, mode=DpaMode.NRT)
        arc += component.damage_energy_to_dpa(damage_energy, mode=DpaMode.ARC)
        ferarc += component.damage_energy_to_dpa(damage_energy, mode=DpaMode.FERARC)
    nrt /= natoms
    arc /= natoms
    ferarc /= natoms

    debris = None
    if path_debris is not None:
        reader = LAMMPSReader(path_debris)
        defects = get_last_reader(reader)["atoms"]
        reader.close()
        debris = np.count_nonzero(defects["type"] == 0) / natoms

    return nrt, arc, ferarc, debris
