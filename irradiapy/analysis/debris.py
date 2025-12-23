"""This module contains functions to generate debris from `RecoilsDB`."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from irradiapy import dtypes, materials
from irradiapy.analysis.debrismanager import DebrisManager
from irradiapy.enums import DamageEnergyMode, DisplacementMode
from irradiapy.io.lammpswriter import LAMMPSWriter
from irradiapy.utils.math import apply_boundary_conditions

if TYPE_CHECKING:
    from irradiapy.recoilsdb import RecoilsDB


def generate_debris(
    recoilsdb: RecoilsDB,
    mddb_dir: Path,
    debris_path: Path,
    compute_damage_energy: bool,
    damage_energy_mode: DamageEnergyMode,
    displacement_mode: DisplacementMode,
    fp_dist: float,
    energy_tolerance: float = 0.1,
    surface_irradiation: bool = False,
    exclude_from_ions: list[int] | None = None,
    exclude_from_vacs: list[int] | None = None,
    seed: int = 0,
    ylo: None | float = None,
    yhi: None | float = None,
    zlo: None | float = None,
    zhi: None | float = None,
):
    """Generate MD debris from RecoilsDB.

    Parameters
    ----------
    recoilsdb
        RecoilsDB instance from Py2SRIM or SPECTRA2SRIM runs.
    mddb_dir
        Path to the MD debris database directory.
    debris_path : Path
        Output file path.
    compute_damage_energy : bool
        If MD simulations do not include electronic stopping, this should be set to `True`. The
        conversion from recoil energy to damage energy will be performed according to the selected
        `damage_energy_mode`.
    damage_energy_mode : materials.Material.DamageEnergyMode
        Mode for recoil to damage energy calculation.
    displacement_mode : materials.Material.DisplacementMode
        Mode for calculation of number of displacement atoms.
    fp_dist : float
        Distance between the vacancy and the interstitial of a Frenkel pair, in angstroms.
    energy_tolerance : float (default=0.1)
        Tolerance for energy decomposition. For example, if this value if ``0.1``, the recoil energy
        is 194 keV and the database contains an energy of 200 keV, then 194 will be in the range
        200 +/- 20 keV, therefore a cascade of 200 keV will be used, instead of decomposing 194 keV
        into, for example, 100x1 + 50x1 + 20x2 + 3x1 + 1xFP (Frenkel pairs). This fixes biases
        towards smaller clusters (lower energies) and helps reducing cascade overlapping. Set to
        ``0.0`` to disable this feature.
    surface_irradiation : bool, optional (default=False)
        If `True`, then the initial primary ion position will not be used to place a vacancy.
    exclude_from_ions : list[int], optional (default=None)
        Ions will not be placed for these ion types. Useful to exclude injected ion effects
        (surface irradiation).
    exclude_from_vacs : list[int], optional (default=None)
        Ions that have an atomic number in this list will not generate a vacancy at their initial
        position. For example, if the ion is Fe in a target of Fe, this list can be empty to
        simulate that the ion has left its lattice position; if the ion is H or He in a target of
        Fe, this list can include 1 and 2 to avoid generating a vacancy for these ions since they
        are likely due to irradiation and not to displacement.
    seed : int, optional (default=0)
        Random seed for random number generator.
    ylo : float | None, optional (default=None)
        Minimum y boundary (only for Py2SRIM recoilsdb).
    yhi : float | None, optional (default=None)
        Maximum y boundary (only for Py2SRIM recoilsdb).
    zlo : float | None, optional (default=None)
        Minimum z boundary (only for Py2SRIM recoilsdb).
    zhi : float | None, optional (default=None)
        Maximum z boundary (only for Py2SRIM recoilsdb).
    """
    if exclude_from_ions is None:
        exclude_from_ions = []
    if exclude_from_vacs is None:
        exclude_from_vacs = []
    if recoilsdb.table_exists("spectrapkas"):
        __spectra2srim_generate_debris(
            recoilsdb=recoilsdb,
            mddb_dir=mddb_dir,
            compute_damage_energy=compute_damage_energy,
            debris_path=debris_path,
            damage_energy_mode=damage_energy_mode,
            displacement_mode=displacement_mode,
            fp_dist=fp_dist,
            energy_tolerance=energy_tolerance,
            surface_irradiation=surface_irradiation,
            exclude_from_ions=exclude_from_ions,
            exclude_from_vacs=exclude_from_vacs,
            seed=seed,
        )
    else:
        __py2srim_generate_debris(
            recoilsdb=recoilsdb,
            mddb_dir=mddb_dir,
            compute_damage_energy=compute_damage_energy,
            debris_path=debris_path,
            damage_energy_mode=damage_energy_mode,
            displacement_mode=displacement_mode,
            fp_dist=fp_dist,
            energy_tolerance=energy_tolerance,
            surface_irradiation=surface_irradiation,
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
    mddb_dir: Path,
    compute_damage_energy: bool,
    debris_path: Path,
    damage_energy_mode: DamageEnergyMode,
    displacement_mode: DisplacementMode,
    fp_dist: float,
    energy_tolerance: float,
    surface_irradiation: bool,
    exclude_from_ions: list[int],
    exclude_from_vacs: list[int],
    seed: int = 0,
) -> None:
    """Generate MD debris from SPECTRA-PKA + SRIM results."""
    target = recoilsdb.load_target()
    width = target[0].width
    component = target[0]

    writer = LAMMPSWriter(debris_path, mode="w")
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
            debris_manager = DebrisManager(
                mddb_dir=mddb_dir,
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
                component=component,
                compute_damage_energy=compute_damage_energy,
                damage_energy_mode=damage_energy_mode,
                displacement_mode=displacement_mode,
                fp_dist=fp_dist,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = debris_manager.get_recoil_debris(
                recoil_energy,
                np.array([x, y, z]),
                np.array([cosx, cosy, cosz]),
            )
            defects = np.concatenate((defects, defects_))

        defects = __place_ions_vacs(
            recoilsdb,
            event,
            defects,
            surface_irradiation,
            exclude_from_ions,
            exclude_from_vacs,
        )

        data["atoms"] = defects
        __apply_boundary_conditions(data, True, True, True)

        writer.write(data)
    writer.close()


def __py2srim_generate_debris(
    recoilsdb: RecoilsDB,
    mddb_dir: Path,
    compute_damage_energy: bool,
    debris_path: Path,
    damage_energy_mode: DamageEnergyMode,
    displacement_mode: DisplacementMode,
    fp_dist: float,
    energy_tolerance: float,
    surface_irradiation: bool,
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

    writer = LAMMPSWriter(debris_path, mode="w")
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

            debris_manager = DebrisManager(
                mddb_dir=mddb_dir,
                recoil=materials.ELEMENT_BY_ATOMIC_NUMBER[atom_numb],
                component=target[component_idx],
                compute_damage_energy=compute_damage_energy,
                damage_energy_mode=damage_energy_mode,
                displacement_mode=displacement_mode,
                fp_dist=fp_dist,
                energy_tolerance=energy_tolerance,
                seed=seed + event,
            )
            defects_ = debris_manager.get_recoil_debris(
                recoil_energy,
                np.array([x, y, z]),
                np.array([cosx, cosy, cosz]),
            )
            defects = np.concatenate((defects, defects_))

        defects = __place_ions_vacs(
            recoilsdb,
            event,
            defects,
            surface_irradiation,
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
    surface_irradiation: bool,
    exclude_from_ions: list,
    exclude_from_vacs: list,
):
    ions_vacs = recoilsdb.read(
        "ions_vacs",
        what="atom_numb, x, y, z",
        conditions=f"WHERE event={event}",
    )
    # To simulate surface irradiation, skip the first row to avoid placing
    # a vacancy at the surface
    if surface_irradiation:
        next(ions_vacs)
    for atom_numb, x, y, z in ions_vacs:
        if atom_numb > 0:
            # Final ion position
            if atom_numb in exclude_from_ions:
                continue
        else:
            # Initial ion position: potential vacancy
            if -atom_numb in exclude_from_vacs:
                continue
            atom_numb = 0
        defect = np.array(
            [(atom_numb, x, y, z)],
            dtype=dtypes.defect,
        )
        defects = np.concatenate((defects, defect))

    return defects
