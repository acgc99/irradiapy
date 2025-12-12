"""This module contains functions to generate debris from `RecoilsDB`."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from irradiapy import dtypes, materials
from irradiapy.damagedb import DamageDB
from irradiapy.io.lammpswriter import LAMMPSWriter
from irradiapy.utils.math import apply_boundary_conditions

if TYPE_CHECKING:
    from irradiapy.recoilsdb import RecoilsDB


def generate_debris(
    recoilsdb: RecoilsDB,
    dir_mddb: Path,
    compute_tdam: bool,
    path_debris: Path,
    tdam_mode: materials.Material.TdamMode,
    dpa_mode: materials.Material.DpaMode,
    energy_tolerance: float = 0.1,
    exclude_from_ions: list[int] | None = None,
    exclude_from_vacs: list[int] | None = None,
    seed: int = 0,
    ylo: None | float = None,
    yhi: None | float = None,
    zlo: None | float = None,
    zhi: None | float = None,
):
    """Generate MD debris from RecoilsDB + SRIM results."""
    if exclude_from_ions is None:
        exclude_from_ions = []
    if exclude_from_vacs is None:
        exclude_from_vacs = []
    if recoilsdb.table_exists("spectrapkas"):
        __spectra2srim_generate_debris(
            recoilsdb=recoilsdb,
            dir_mddb=dir_mddb,
            compute_tdam=compute_tdam,
            path_debris=path_debris,
            tdam_mode=tdam_mode,
            dpa_mode=dpa_mode,
            energy_tolerance=energy_tolerance,
            exclude_from_ions=exclude_from_ions,
            exclude_from_vacs=exclude_from_vacs,
            seed=seed,
        )
    else:
        __py2srim_generate_debris(
            recoilsdb=recoilsdb,
            dir_mddb=dir_mddb,
            compute_tdam=compute_tdam,
            path_debris=path_debris,
            tdam_mode=tdam_mode,
            dpa_mode=dpa_mode,
            energy_tolerance=energy_tolerance,
            exclude_from_ions=exclude_from_ions,
            exclude_from_vacs=exclude_from_vacs,
            seed=seed,
            ylo=ylo,
            yhi=yhi,
            zlo=zlo,
            zhi=zhi,
        )


def __spectra2srim_generate_debris(
    recoilsdb: RecoilsDB,
    dir_mddb: Path,
    compute_tdam: bool,
    path_debris: Path,
    tdam_mode: materials.Material.TdamMode,
    dpa_mode: materials.Material.DpaMode,
    energy_tolerance: float,
    exclude_from_ions: list[int],
    exclude_from_vacs: list[int],
    seed: int = 0,
) -> None:
    """Generate MD debris from SPECTRA-PKA + SRIM results."""

    srim_target = recoilsdb.load_srim_target()
    width = srim_target.layers[0].width

    symbols = [el.symbol for el in srim_target.layers[0].elements]
    stoichs = srim_target.layers[0].stoichs
    max_stoich_idx = np.argmax(stoichs)
    material_target = materials.MATERIALS_BY_SYMBOL[symbols[max_stoich_idx]]
    warnings.warn(
        (
            f"Material target for debris generation set to '{material_target.symbol}' "
            "based on the highest stoichiometry in the material. "
            "This is a limitation of the current code. You can provide a different "
            "target by using the 'mat_target' argument."
        ),
        RuntimeWarning,
    )

    writer = LAMMPSWriter(path_debris, mode="w")
    events = recoilsdb.read("spectrapkas", what="event, time, timestep")
    for event, time, timestep in events:
        data = {
            "xlo": 0.0,
            "xhi": width,
            "ylo": 0.0,
            "yhi": width,
            "zlo": 0.0,
            "zhi": width,
            "boundary": ["pp", "pp", "pp"],
            "timestep": timestep,
            "time": time,
        }
        defects = np.empty(0, dtype=dtypes.defect)

        recoils = recoilsdb.read(
            "recoils",
            what="atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz",
            condition=f"WHERE event={event}",
        )
        for atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz in recoils:
            damagedb = DamageDB(
                dir_mddb=dir_mddb,
                compute_tdam=compute_tdam,
                mat_pka=materials.MATERIALS_BY_ATOMIC_NUMBER[atom_numb],
                mat_target=material_target,
                dpa_mode=dpa_mode,
                tdam_mode=tdam_mode,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = damagedb.get_pka_debris(
                recoil_energy, np.array([depth, y, z]), np.array([cosx, cosy, cosz])
            )
            defects = np.concatenate((defects, defects_))

        defects = __place_ions_vacs(
            recoilsdb,
            event,
            defects,
            exclude_from_ions,
            exclude_from_vacs,
        )

        data["atoms"] = defects
        __apply_boundary_conditions(data, True, True, True)

        writer.write(data)
    writer.close()


def __py2srim_generate_debris(
    recoilsdb: RecoilsDB,
    dir_mddb: Path,
    compute_tdam: bool,
    path_debris: Path,
    tdam_mode: materials.Material.TdamMode,
    dpa_mode: materials.Material.DpaMode,
    energy_tolerance: float,
    exclude_from_ions: list[int],
    exclude_from_vacs: list[int],
    seed: int = 0,
    ylo: None | float = None,
    yhi: None | float = None,
    zlo: None | float = None,
    zhi: None | float = None,
) -> None:
    """Generate MD debris from Python to SRIM results."""

    srim_target = recoilsdb.load_srim_target()
    width = sum(layer.width for layer in srim_target.layers)

    layers_edges = np.cumsum([0.0] + [layer.width for layer in srim_target.layers])
    material_targets = []
    for layer in srim_target.layers:
        symbols = [el.symbol for el in layer.elements]
        stoichs = layer.stoichs
        max_stoich_idx = np.argmax(stoichs)
        mat_target = materials.MATERIALS_BY_SYMBOL[symbols[max_stoich_idx]]
        material_targets.append(mat_target)
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
    nevents = recoilsdb.get_nevents()
    for event in range(1, nevents + 1):
        data = {
            "xlo": 0.0,
            "xhi": width,
            "ylo": -width if ylo is None else ylo,
            "yhi": width if yhi is None else yhi,
            "zlo": -width if zlo is None else zlo,
            "zhi": width if zhi is None else zhi,
            "boundary": ["ff", "ff", "ff"],
            "timestep": 0,
            "time": 0.0,
        }
        defects = np.empty(0, dtype=dtypes.defect)

        recoils = recoilsdb.read(
            "recoils",
            what="atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz",
            condition=f"WHERE event = {event}",
        )
        for atom_numb, recoil_energy, depth, y, z, cosx, cosy, cosz in recoils:
            # Determine layer and select target material
            layer_idx = np.searchsorted(layers_edges, depth, side="right") - 1
            material_target = material_targets[layer_idx]

            damagedb = DamageDB(
                dir_mddb=dir_mddb,
                compute_tdam=compute_tdam,
                mat_pka=materials.MATERIALS_BY_ATOMIC_NUMBER[atom_numb],
                mat_target=material_target,
                dpa_mode=dpa_mode,
                tdam_mode=tdam_mode,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = damagedb.get_pka_debris(
                recoil_energy, np.array([depth, y, z]), np.array([cosx, cosy, cosz])
            )
            defects = np.concatenate((defects, defects_))

        defects = __place_ions_vacs(
            recoilsdb,
            event,
            defects,
            exclude_from_ions,
            exclude_from_vacs,
        )

        data["atoms"] = defects
        __apply_boundary_conditions(data, False, False, False)

        writer.write(data)
    writer.close()


def __apply_boundary_conditions(
    data: dict,
    x: bool,
    y: bool,
    z: bool,
) -> dict:
    """Apply boundary conditions to debris data."""
    natoms0 = len(data["atoms"])
    data = apply_boundary_conditions(data, x, y, z)
    natoms1 = len(data["atoms"])
    if natoms0 != natoms1:
        warnings.warn(
            (
                f"Some defects were outside the boundaries and were removed: "
                f"{natoms0 - natoms1} defects removed."
            ),
            RuntimeWarning,
        )
    return data


def __place_ions_vacs(
    recoilsdb: RecoilsDB,
    event: int,
    defects: np.ndarray,
    exclude_from_ions: list,
    exclude_from_vacs: list,
):
    # To place ion-vacs, we assume that ions_vacs table is ordered as:
    # ion1, vac1, ion2, vac2, ...
    # Logic to exclude some ions/vacs if needed based on the ion type
    ions_vacs = recoilsdb.read(
        "ions_vacs",
        what="atom_numb, depth, y, z",
        condition=f"WHERE event={event}",
    )
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
    return defects
