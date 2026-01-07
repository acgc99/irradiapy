"""Defects analysis module."""

from pathlib import Path

import numpy as np
from numpy import typing as npt

from irradiapy import dtypes
from irradiapy.analysis.defectsidentifier import DefectsIdentifier
from irradiapy.io import BZIP2LAMMPSReader, LAMMPSReader, LAMMPSWriter


def identify_defects(
    lattice: str,
    a0: float,
    data_atoms: dict,
    a1: None | float = None,
    pos_pka: None | npt.NDArray[np.float64] = None,
    theta_pka: None | float = None,
    phi_pka: None | float = None,
    transform: None | bool = False,
    debug: bool = False,
) -> dtypes.Defect:
    """Identify defects in a given atomic structure.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter.
    data_atoms : dict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    a1 : float, optional (default=None)
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : npt.NDArray[np.float64], optional (default=None)
        Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
        recentered and aligned.
    theta_pka : float, optional (default=None)
        Polar angle (in radians) for the PKA direction.
    phi_pka : float, optional (default=None)
        Azimuthal angle (in radians) for the PKA direction.
    transform : bool, optional (default=False)
        If True, defects are recentered and aligned with the PKA direction (if provided). If
        True but no PKA parameters are provided, defects are recentered based on their
        average position. Note that the box boundaries are not modified for visualization
        purposes, only the atomic positions are transformed.
    debug : bool, optional (default=False)
        If `True`, enables debug mode for additional output.

    Returns
    -------
    dtypes.Defect
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
    dump_path: Path,
    dump_defects_path: Path,
    a1: None | float = None,
    pos_pka: None | npt.NDArray[np.float64] = None,
    theta_pka: None | float = None,
    phi_pka: None | float = None,
    transform: None | bool = False,
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
    data_atoms : dict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
    data_atoms : dict
        Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
        Must include keys: 'atoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
        'zhi', 'timestep'.
    dump_path : Path
        Path to the LAMMPS dump file to read. Can be compressed with `.bz2` or not.
    dump_defects_path : Path
        Path to the output file where identified defects will be written (in text format).
    a1 : float, optional
        Final lattice parameter. If provided, defect positions are rescaled to this value
        (independently of the `transform` value).
    pos_pka : npt.NDArray[np.float64], optional
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
    overwrite : bool, optional (default=False)
        If True, allows overwriting the output file if it already exists.
    debug : bool, optional (default=False)
        If `True`, enables debug mode for additional output.
    """
    if dump_defects_path.exists():
        if overwrite:
            dump_defects_path.unlink()
        else:
            raise FileExistsError(f"Defects file {dump_defects_path} already exists.")
    if debug:
        print(f"Identifying defects in {dump_path}")
    if dump_path.suffix == ".bz2":
        reader = BZIP2LAMMPSReader(dump_path)
    else:
        reader = LAMMPSReader(dump_path)
    writer = LAMMPSWriter(dump_defects_path, mode="a")
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
