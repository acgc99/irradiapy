"""This subpackage provides a tool for irradiation damage analysis."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import typing as npt

from irradiapy.analysis import clusters, dpa
from irradiapy.analysis.defectsidentifier import DefectsIdentifier
from irradiapy.io import BZIP2LAMMPSReader, LAMMPSReader, LAMMPSWriter


def identify_defects(
    lattice: str,
    a0: float,
    data_atoms: defaultdict,
    a1: Optional[float] = None,
    pos_pka: Optional[npt.NDArray[np.float64]] = None,
    theta_pka: Optional[float] = None,
    phi_pka: Optional[float] = None,
    transform: Optional[bool] = False,
    debug: bool = False,
) -> np.ndarray:
    """Identify defects in a given atomic structure.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'natoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    a1 : float, optional
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : np.ndarray, optional
        Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
        recentered and aligned.
    theta_pka : float, optional
        Polar angle (in radians) for the PKA direction.
    phi_pka : float, optional
        Azimuthal angle (in radians) for the PKA direction.
    transform : bool, optional
        If True, defects are recentered and aligned with the PKA direction (if provided). If
        True but no PKA parameters are provided, defects are recentered based on their
        average position. Note that the box boundaries are not modified for visualization
        purposes, only the atomic positions are transformed.
    debug : bool, optional
        If `True`, enables debug mode for additional output. Default is `False`.

    Returns
    -------
    np.ndarray
        An array of identified defects in the structure.
    """
    defects_finder = DefectsIdentifier(lattice=lattice, a0=a0, debug=debug)
    defects = defects_finder.identify(
        data_atoms=data_atoms,
        a1=a1,
        pos_pka=pos_pka,
        theta_pka=theta_pka,
        phi_pka=phi_pka,
        transform=transform,
    )
    return defects


def identify_lammps_dump(
    lattice: str,
    a0: float,
    path_dump: Path,
    path_dump_defects: Path,
    a1: Optional[float] = None,
    pos_pka: Optional[npt.NDArray[np.float64]] = None,
    theta_pka: Optional[float] = None,
    phi_pka: Optional[float] = None,
    transform: Optional[bool] = False,
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    """Identify defects in a LAMMPS dump file.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
    data_atoms : defaultdict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'natoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    path_dump : Path
        Path to the LAMMPS dump file to read. Can be compressed with `.bz2` or not.
    path_dump_defects : Path
        Path to the output file where identified defects will be written (in text format).
    a1 : float, optional
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : np.ndarray, optional
        Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
        recentered and aligned.
    theta_pka : float, optional
        Polar angle (in radians) for the PKA direction.
    phi_pka : float, optional
        Azimuthal angle (in radians) for the PKA direction.
    transform : bool, optional
        If True, defects are recentered and aligned with the PKA direction (if provided). If
        True but no PKA parameters are provided, defects are recentered based on their
        average position. Note that the box boundaries are not modified for visualization
        purposes, only the atomic positions are transformed.
    overwrite : bool, optional
        If True, allows overwriting the output file if it already exists. Default is False.
    debug : bool, optional
        If `True`, enables debug mode for additional output. Default is `False`.
    """
    if path_dump_defects.exists():
        if overwrite:
            path_dump_defects.unlink()
        else:
            raise FileExistsError(f"Defects file {path_dump_defects} already exists.")
    if debug:
        print(f"Identifying defects in {path_dump}")
    if path_dump.suffix == ".bz2":
        reader = BZIP2LAMMPSReader(path_dump)
    else:
        reader = LAMMPSReader(path_dump)
    writer = LAMMPSWriter(path_dump_defects, mode="a")
    defects_finder = DefectsIdentifier(lattice=lattice, a0=a0, debug=debug)
    for data_atoms in reader:
        if debug:
            print(f"Timestep {data_atoms['timestep']}")
        defects = defects_finder.identify(
            data_atoms,
            a1=a1,
            pos_pka=pos_pka,
            theta_pka=theta_pka,
            phi_pka=phi_pka,
            transform=transform,
        )
        writer.write(defects)
