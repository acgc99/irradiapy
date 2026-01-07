"""Utility functions for I/O operations."""

import bz2
from collections import deque
from pathlib import Path

import numpy as np

from irradiapy.io import (
    BZIP2LAMMPSReader,
    BZIP2LAMMPSWriter,
    LAMMPSLogReader,
    LAMMPSReader,
    LAMMPSWriter,
)
from irradiapy.utils.math import apply_boundary_conditions


def compress_file_bz2(
    input_path: str, output_path: str, compresslevel: int = 9
) -> None:
    """Compress a file using bzip2.

    Parameters
    ----------
    input_path : str
        Path to the input file to be compressed.
    output_path : str
        Path where the compressed file will be saved.
    compresslevel : int, optional (default=9)
        Compression level for bzip2.
    """
    with (
        open(input_path, "rb") as f_in,
        bz2.open(output_path, "wb", compresslevel=compresslevel) as f_out,
    ):
        for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
            f_out.write(chunk)


def decompress_file_bz2(input_path: str, output_path: str) -> None:
    """Decompress a bzip2-compressed file.

    Parameters
    ----------
    input_path : str
        Path to the input .bz2 file.
    output_path : str
        Path to the output decompressed file.
    """
    with bz2.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
            f_out.write(chunk)


def get_last_reader(
    reader: list[LAMMPSReader, BZIP2LAMMPSReader, LAMMPSLogReader],
) -> any:
    """Get the last snapshot from a LAMMPS dump file using a reader.

    Parameters
    ----------
    reader : LAMMPSReader, BZIP2LAMMPSReader, LAMMPSLogReader
        An instance of a LAMMPS reader.

    Returns
    -------
    any
        The last snapshot from the LAMMPS file.
    """
    return deque(reader, maxlen=1).pop()


def merge_lammps_snapshots(
    in_path: Path, out_path: Path, overwrite: bool = False
) -> dict:
    """Merge multiple snapshots in a LAMMPS file into a single snapshot.

    Parameters
    ----------
    in_path : Path
        Path to the input LAMMPS file (bzip2 compressed or not).
    out_path : Path
        Path to the output LAMMPS file (bzip2 compressed or not).
    overwrite : bool, optional (default=False)
        Whether to overwrite the output file if it exists.

    Returns
    -------
    dict
        A dictionary containing the merged snapshot data.
    """

    if not overwrite and out_path.exists():
        raise FileExistsError(f"Output file {out_path} already exists.")
    elif out_path.exists():
        out_path.unlink()
    if in_path.suffix == ".bz2":
        reader = BZIP2LAMMPSReader(in_path)
    else:
        reader = LAMMPSReader(in_path)
    if out_path.suffix == ".bz2":
        writer = BZIP2LAMMPSWriter(out_path, mode="a")
    else:
        writer = LAMMPSWriter(out_path, mode="a")

    data_atoms_list = []
    for data_atoms in reader:
        data_atoms_list.append(data_atoms)
    reader.close()
    data_atoms_merged = {}
    data_atoms_merged["timestep"] = data_atoms_list[-1]["timestep"]
    data_atoms_merged["time"] = data_atoms_list[-1]["time"]
    data_atoms_merged["boundary"] = data_atoms_list[-1]["boundary"]
    data_atoms_merged["xlo"] = data_atoms_list[-1]["xlo"]
    data_atoms_merged["xhi"] = data_atoms_list[-1]["xhi"]
    data_atoms_merged["ylo"] = data_atoms_list[-1]["ylo"]
    data_atoms_merged["yhi"] = data_atoms_list[-1]["yhi"]
    data_atoms_merged["zlo"] = data_atoms_list[-1]["zlo"]
    data_atoms_merged["zhi"] = data_atoms_list[-1]["zhi"]
    data_atoms_merged["atoms"] = np.concatenate(
        [data_atoms["atoms"] for data_atoms in data_atoms_list]
    )
    writer.write(data_atoms_merged)
    writer.close()
    return data_atoms_merged


def apply_boundary_conditions_to_lammps(
    in_path: Path, out_path: Path, x: bool, y: bool, z: bool, overwrite: bool = False
) -> None:
    """Apply periodic boundary conditions to a LAMMPS dump file.

    Parameters
    ----------
    in_path : Path
        Path to the input LAMMPS file (bzip2 compressed or not).
    out_path : Path
        Path to the output LAMMPS file (bzip2 compressed or not).
    x : bool
        Whether to apply periodic boundary conditions in the x direction.
    y : bool
        Whether to apply periodic boundary conditions in the y direction.
    z : bool
        Whether to apply periodic boundary conditions in the z direction.
    overwrite : bool, optional (default=False)
        Whether to overwrite the output file if it exists.
    """

    if not overwrite and out_path.exists():
        raise FileExistsError(f"Output file {out_path} already exists.")
    elif out_path.exists():
        out_path.unlink()
    if in_path.suffix == ".bz2":
        reader = BZIP2LAMMPSReader(in_path)
    else:
        reader = LAMMPSReader(in_path)
    if out_path.suffix == ".bz2":
        writer = BZIP2LAMMPSWriter(out_path, mode="a")
    else:
        writer = LAMMPSWriter(out_path, mode="a")

    for data_atoms in reader:
        data_atoms = apply_boundary_conditions(
            data_atoms=data_atoms,
            x=x,
            y=y,
            z=z,
        )
        writer.write(data_atoms)
    reader.close()
    writer.close()
