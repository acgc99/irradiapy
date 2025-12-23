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
    compute_damage_energy: bool,
    path_debris: Path,
    damage_energy_mode: materials.DamageEnergyMode,
    displacement_mode: materials.DisplacementMode,
    dist_fp: float,
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
            compute_damage_energy=compute_damage_energy,
            path_debris=path_debris,
            damage_energy_mode=damage_energy_mode,
            displacement_mode=displacement_mode,
            dist_fp=dist_fp,
            energy_tolerance=energy_tolerance,
            exclude_from_ions=exclude_from_ions,
            exclude_from_vacs=exclude_from_vacs,
            seed=seed,
        )
    else:
        __py2srim_generate_debris(
            recoilsdb=recoilsdb,
            dir_mddb=dir_mddb,
            compute_damage_energy=compute_damage_energy,
            path_debris=path_debris,
            damage_energy_mode=damage_energy_mode,
            displacement_mode=displacement_mode,
            dist_fp=dist_fp,
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
    compute_damage_energy: bool,
    path_debris: Path,
    damage_energy_mode: materials.DamageEnergyMode,
    displacement_mode: materials.DisplacementMode,
    dist_fp: float,
    energy_tolerance: float,
    exclude_from_ions: list[int],
    exclude_from_vacs: list[int],
    seed: int = 0,
) -> None:
    """Generate MD debris from SPECTRA-PKA + SRIM results."""
    target = recoilsdb.load_target()
    width = target[0].width
    component = target[0]

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
            what="atom_numb, recoil_energy, x, y, z, cosx, cosy, cosz",
            conditions=f"WHERE event={event}",
        )
        for atom_numb, recoil_energy, x, y, z, cosx, cosy, cosz in recoils:
            damagedb = DamageDB(
                dir_mddb=dir_mddb,
                compute_damage_energy=compute_damage_energy,
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
                component=component,
                displacement_mode=displacement_mode,
                damage_energy_mode=damage_energy_mode,
                dist_fp=dist_fp,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = damagedb.get_pka_debris(
                recoil_energy, np.array([x, y, z]), np.array([cosx, cosy, cosz])
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
    compute_damage_energy: bool,
    path_debris: Path,
    damage_energy_mode: materials.DamageEnergyMode,
    displacement_mode: materials.DisplacementMode,
    dist_fp: float,
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
    target = recoilsdb.load_target()
    width = sum(component.width for component in target)
    component_edges = np.cumsum([0.0] + [component.width for component in target])

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
            what="atom_numb, recoil_energy, x, y, z, cosx, cosy, cosz",
            conditions=f"WHERE event = {event}",
        )
        for atom_numb, recoil_energy, x, y, z, cosx, cosy, cosz in recoils:
            # Determine layer and select target material
            component_idx = np.searchsorted(component_edges, x, side="right") - 1

            damagedb = DamageDB(
                dir_mddb=dir_mddb,
                compute_damage_energy=compute_damage_energy,
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
                component=target[component_idx],
                displacement_mode=displacement_mode,
                damage_energy_mode=damage_energy_mode,
                dist_fp=dist_fp,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = damagedb.get_pka_debris(
                recoil_energy, np.array([x, y, z]), np.array([cosx, cosy, cosz])
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
        what="atom_numb, x, y, z",
        conditions=f"WHERE event={event}",
    )
    count = 0
    past_atom_numb = 0
    ions_vacs = list(ions_vacs)
    for atom_numb, x, y, z in ions_vacs:
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
            [(atom_numb, x, y, z)],
            dtype=dtypes.defect,
        )
        defects = np.concatenate((defects, defect))
    return defects
