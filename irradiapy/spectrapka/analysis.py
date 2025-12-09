"""Utility functions related to SPECTRA-PKa + SRIM data analysis and debris production."""

import warnings
from pathlib import Path

import numpy as np

from irradiapy import dtypes, materials
from irradiapy.damagedb import DamageDB
from irradiapy.io import LAMMPSReader, LAMMPSWriter
from irradiapy.spectrapka.spectra2srim import Spectra2SRIM
from irradiapy.utils.io import get_last_reader
from irradiapy.utils.math import apply_boundary_conditions


def generate_debris(
    spectrapka2srim: Spectra2SRIM,
    dir_mddb: Path,
    compute_tdam: bool,
    path_debris: Path,
    tdam_mode: materials.Material.TdamMode,
    dpa_mode: materials.Material.DpaMode,
    boundaries: list[bool],
    mat_target: materials.Material | None = None,
    exclude_from_ions: list[int] | None = None,
    exclude_from_vacs: list[int] | None = None,
    seed: int = 0,
) -> None:
    """Generate MD debris from SPECTRA-PKA + SRIM results.

    Parameters
    ----------
    spectrapka2srim : Spectra2SRIM
        The SPECTRA-PKA to SRIM object containing the recoils database.
    dir_mddb : Path
        Path to the MD damage database directory.
    compute_tdam : bool
        Whether to compute the damage energy from the PKA energy. False if MD simulations already
        include it.
    path_debris : Path
        Path to the output debris LAMMPS file.
    tdam_mode : Material.TdamMode
        The damage energy calculation mode.
    dpa_mode : Material.DpaMode
        The DPA calculation mode for Frenkel-pairs MD debris generation.
    boundaries : list[bool]
        List of booleans indicating whether to apply periodic boundary conditions in each direction.
    mat_target : Material | None (default=None)
        Material object representing the target material,
        by default None (taken from spectrapka2srim).
    exclude_from_ions : list[int] | None (default=None)
        When a recoil is passed to SRIM, an interstitial-vacancy pair should be created. This list
        contains the atomic numbers of ions for which the interstitial will not be placed. The atom
        is placed where the SRIM ion stops.
    exclude_from_vacs : list[int] | None (default=None)
        When a recoil is passed to SRIM, an interstitial-vacancy pair should be created. This list
        contains the atomic numbers of ions for which the vacancy will not be placed. The vacancy
        is placed where the SRIM ion starts.
    seed : int (default=0)
        Random seed for reproducibility.
    """
    if exclude_from_ions is None:
        exclude_from_ions = []
    if exclude_from_vacs is None:
        exclude_from_vacs = []
    rng = np.random.default_rng(seed)
    events = spectrapka2srim.recoils_db.read("events", what="event, time, timestep")

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

    writer = LAMMPSWriter(path_debris, mode="w")
    for event, time, timestep in events:
        data = {
            "xlo": 0.0,
            "xhi": spectrapka2srim.matdict["sizex"],
            "ylo": 0.0,
            "yhi": spectrapka2srim.matdict["sizey"],
            "zlo": 0.0,
            "zhi": spectrapka2srim.matdict["sizez"],
            "boundary": ["p" if b else "f" for b in boundaries],
            "timestep": timestep,
            "time": time,
        }

        defects = np.empty(0, dtype=dtypes.defect)
        recoils = spectrapka2srim.recoils_db.read(
            "recoils",
            what="atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz",
            condition=f"WHERE event={event}",
        )
        ions_vacs = spectrapka2srim.recoils_db.read(
            "ions_vacs",
            what="atom_numb, depth, y, z",
            condition=f"WHERE event={event}",
        )

        for atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz in recoils:
            damagedb = DamageDB(
                dir_mddb=dir_mddb,
                compute_tdam=compute_tdam,
                mat_pka=materials.MATERIALS_BY_ATOMIC_NUMBER[atom_numb],
                mat_target=mat_target,
                dpa_mode=dpa_mode,
                tdam_mode=tdam_mode,
                seed=rng.integers(0, np.iinfo(np.int_).max, endpoint=True),
            )
            defects_ = damagedb.get_pka_debris(
                recoil_energy, np.array([depth, y, z]), np.array([cosx, cosy, cosz])
            )
            defects = np.concatenate((defects, defects_))

        # To place ion-vacs, we assume that ions_vacs table is ordered as:
        # ion1, vac1, ion2, vac2, ...
        # Logic to exclude some ions/vacs if needed based on the ion type
        count = 0
        past_atom_numb = 0
        ions_vacs = list(ions_vacs)
        for atom_numb, depth, y, z in ions_vacs:
            if count == 0 and atom_numb == 0:
                raise ValueError(
                    (
                        "ions_vacs table is disordered. "
                        "It must have the following structure: ion, vac, ion, vac, ..."
                        "This table should have been generated with the proper structure, "
                        "did you modify it?"
                    )
                )

            count += 1
            if count % 2 == 1:
                past_atom_numb = atom_numb
                if past_atom_numb in exclude_from_ions:
                    continue
            else:
                if past_atom_numb in exclude_from_vacs:
                    continue

            defect = np.array(
                [(atom_numb, depth, y, z)],
                dtype=dtypes.defect,
            )
            defects = np.concatenate((defects, defect))

        data["atoms"] = defects
        data = apply_boundary_conditions(data, *boundaries)
        if not boundaries[0]:
            cond = (data["atoms"]["x"] >= data["xlo"]) & (
                data["atoms"]["x"] <= data["xhi"]
            )
            data["atoms"] = data["atoms"][cond]
        if not boundaries[1]:
            cond = (data["atoms"]["y"] >= data["ylo"]) & (
                data["atoms"]["y"] <= data["yhi"]
            )
            data["atoms"] = data["atoms"][cond]
        if not boundaries[2]:
            cond = (data["atoms"]["z"] >= data["zlo"]) & (
                data["atoms"]["z"] <= data["zhi"]
            )
            data["atoms"] = data["atoms"][cond]
        data["natoms"] = len(data["atoms"])

        writer.write(data)
    writer.close()


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
