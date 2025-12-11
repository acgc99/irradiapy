"""Utility functions related to SPECTRA-PKa + SRIM data analysis and debris production."""

import warnings
from pathlib import Path

import numpy as np

from irradiapy import materials
from irradiapy.io import LAMMPSReader
from irradiapy.spectrapka.spectra2srim import Spectra2SRIM
from irradiapy.utils.io import get_last_reader


def get_dpas(
    spectrapka2srim: Spectra2SRIM,
    path_debris: Path | None = None,
    mat_target: materials.Material | None = None,
    tdam_mode: materials.Material.TdamMode = materials.Material.TdamMode.LINDHARD,
) -> tuple[float, ...]:
    """Turns SPECTRA-PKA + SRIM's recoils into the different dpa metrics.

    Parameters
    ----------
    spectrapka2srim : Spectra2SRIM
        The SPECTRA-PKA to SRIM object containing the recoils database.
    path_debris : Path | None (default=None)
       Path to the debris LAMMPS file to compute the debris dpa, by default None.
    mat_target : Material | None (default=None)
        Material object representing the target material,
        by default None (taken from spectrapka2srim).
    tdam_mode : Material.TdamMode (default=Material.TdamMode.LINDHARD)
        The damage energy calculation mode.

    Returns
    -------
    tuple[float, ...]
        A tuple containing the total NRT, arc, fer-arc (and debris) dpa values
        in the simulation box.
    """
    natoms_cell = 1  # number of atoms per unit cell
    if spectrapka2srim.matdict["lattice"] == "bcc":
        natoms_cell = 2
    elif spectrapka2srim.matdict["lattice"] == "fcc":
        natoms_cell = 4
    elif spectrapka2srim.matdict["lattice"] == "hcp":
        natoms_cell = 2
    else:
        raise ValueError(f"Unknown lattice type: {spectrapka2srim.matdict['lattice']}")

    natoms = spectrapka2srim.matdict["nsize"] ** 3 * natoms_cell

    symbols = spectrapka2srim.matdict["symbols"]
    stoichs = spectrapka2srim.matdict["stoichs"]
    if mat_target is None:
        max_stoich_idx = np.argmax(stoichs)
        mat_target = materials.MATERIALS_BY_SYMBOL[symbols[max_stoich_idx]]
        if len(symbols) > 1:
            warnings.warn(
                (
                    f"Material target for debris generation set to '{mat_target.symbol}' "
                    "based on the highest stoichiometry in the material. "
                    "This is a limitation of the current code. You can provide a different "
                    "target by using the 'mat_target' argument."
                ),
                RuntimeWarning,
            )

    recoils = spectrapka2srim.recoils_db.read(
        "recoils", what="atom_numb, recoil_energy"
    )

    nrt = 0
    arc = 0
    ferarc = 0
    for atom_numb, recoil_energy in recoils:
        mat_pka = materials.MATERIALS_BY_ATOMIC_NUMBER[atom_numb]
        tdam = mat_target.epka_to_tdam(mat_pka, recoil_energy, tdam_mode)
        nrt += mat_target.calc_nrt_dpa(tdam)
        arc += mat_target.calc_arc_dpa(tdam)
        ferarc += mat_target.calc_fer_arc_dpa(tdam)
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
