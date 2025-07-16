"""Utility functions for I/O operations."""

import bz2
from collections import defaultdict, deque
from pathlib import Path

from irradiapy.io import LAMMPSReader


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


def get_last_lammps_dump(path: Path) -> defaultdict:
    """Get the last snaptshot from a LAMMPS dump file.

    Parameters
    ----------
    path : Path
        Path to the LAMMPS dump file.

    Returns
    -------
    defaultdict
        The last snapshot from the LAMMPS dump file.
    """
    return deque(LAMMPSReader(path), maxlen=1).pop()
