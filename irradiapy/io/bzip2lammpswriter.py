"""This module contains the `BZIP2LAMMPSWriter` class."""

import bz2
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BZIP2LAMMPSWriter:
    """A class to write data like a LAMMPS dump file, but compressed with bzip2.

    Note
    ----
    If you only need to compress a file, use `irradiapy.io.io_utils.compress_file_bz2` instead.

    Attributes
    ----------
    file_path : Path
        The path to the bzip2-compressed LAMMPS dump file.
    mode : str
        The file open mode (default: 'wt').
    encoding : str
        The file encoding (default: 'utf-8').
    compresslevel : int
        The bzip2 compression level (default: 9).
    excluded_items : list[str]
        Atom fields to exclude from output (default: ["xs", "ys", "zs"]).
    int_format : str
        The format for integers (default: "%d").
    float_format : str
        The format for floats (default: "%g").
    """

    file_path: Path
    mode: str = "wt"
    encoding: str = "utf-8"
    compresslevel: int = 9
    excluded_items: list[str] = field(default_factory=lambda: ["xs", "ys", "zs"])
    int_format: str = "%d"
    float_format: str = "%g"

    def __post_init__(self) -> None:
        self.file = bz2.open(
            self.file_path,
            self.mode,
            encoding=self.encoding,
            compresslevel=self.compresslevel,
        )

    def __enter__(self) -> "BZIP2LAMMPSWriter":
        return self

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None) -> bool:
        self.file.close()
        return False

    def close(self) -> None:
        """Closes the file associated with this writer."""
        self.file.close()

    def __del__(self) -> None:
        """Closes the file associated with this writer."""
        self.file.close()

    def write(self, data: dict) -> None:
        """Writes the data (from LAMMPSReader/BZIP2LAMMPSReader) to the file.

        Parameters
        ----------
        data : dict
            The dictionary containing the data.
        """
        if data.get("time") is not None:
            self.file.write(f"ITEM: TIME\n{data['time']}\n")
        self.file.write(f"ITEM: TIMESTEP\n{data['timestep']}\n")
        self.file.write(f"ITEM: NUMBER OF ATOMS\n{data['natoms']}\n")
        self.file.write(f"ITEM: BOX BOUNDS {' '.join(data['boundary'])}\n")
        self.file.write(f"{data['xlo']} {data['xhi']}\n")
        self.file.write(f"{data['ylo']} {data['yhi']}\n")
        self.file.write(f"{data['zlo']} {data['zhi']}\n")

        atoms = data["atoms"]
        field_names = [f for f in atoms.dtype.names if f not in self.excluded_items]
        self.file.write(f"ITEM: ATOMS {' '.join(field_names)}\n")

        formatters = []
        for field_name in field_names:
            dtype = atoms.dtype[field_name]
            if dtype.kind == "i":
                formatters.append(self.int_format)
            elif dtype.kind == "f":
                formatters.append(self.float_format)
            else:
                formatters.append("%s")

        for row in atoms:
            self.file.write(
                " ".join(
                    fmt % row[field_name]
                    for fmt, field_name in zip(formatters, field_names)
                )
                + "\n"
            )
