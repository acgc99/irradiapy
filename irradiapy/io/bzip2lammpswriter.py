"""This module contains the `BZIP2LAMMPSWriter` class."""

import bz2
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from irradiapy import config


@dataclass
class BZIP2LAMMPSWriter:
    """A class to write data like a LAMMPS dump file, but compressed with bzip2.

    Note
    ----
    Assumed orthogonal simulation box.

    Attributes
    ----------
    file_path : Path
        The path to the bzip2-compressed LAMMPS dump file.
    mode : str, optional (default="wt")
        The file open mode.
    encoding : str, optional (default=irradiapy.config.ENCODING)
        The file encoding.
    compresslevel : int, optional (default=9)
        The bzip2 compression level.
    int_format : str, optional (default=irradiapy.config.INT_FORMAT)
        The format for integers.
    float_format : str, optional (default=irradiapy.config.FLOAT_FORMAT)
        The format for floats.
    excluded_items : list[str], optional (default=irradiapy.config.EXCLUDED_ITEMS)
        Atom fields to exclude from output.
    """

    file_path: Path
    mode: str = "wt"
    encoding: str = field(default_factory=lambda: config.ENCODING)
    compresslevel: int = 9
    int_format: str = field(default_factory=lambda: config.INT_FORMAT)
    float_format: str = field(default_factory=lambda: config.FLOAT_FORMAT)
    excluded_items: list[str] = field(default_factory=lambda: config.EXCLUDED_ITEMS)

    __file: BinaryIO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.__file = bz2.open(
            self.file_path,
            self.mode,
            encoding=self.encoding,
            compresslevel=self.compresslevel,
        )

    def __enter__(self) -> "BZIP2LAMMPSWriter":
        return self

    def __del__(self) -> None:
        if self.__file is not None:
            self.__file.close()

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None) -> bool:
        if self.__file is not None:
            self.__file.close()
        return False

    def close(self) -> None:
        """Closes the file associated with this writer."""
        if self.__file is not None:
            self.__file.close()

    def write(self, data: dict) -> None:
        """Write the data to the file.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the data to be written. The keys should
            include "timestep", "boundary", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi",
            and "atoms". Optional keys: "time".
        """
        if data.get("time") is not None:
            self.__file.write(f"ITEM: TIME\n{data['time']}\n")
        self.__file.write(f"ITEM: TIMESTEP\n{data['timestep']}\n")
        self.__file.write(f"ITEM: NUMBER OF ATOMS\n{data['natoms']}\n")
        self.__file.write(f"ITEM: BOX BOUNDS {' '.join(data['boundary'])}\n")
        self.__file.write(
            f"{self.float_format % data['xlo']} {self.float_format % data['xhi']}\n"
        )
        self.__file.write(
            f"{self.float_format % data['ylo']} {self.float_format % data['yhi']}\n"
        )
        self.__file.write(
            f"{self.float_format % data['zlo']} {self.float_format % data['zhi']}\n"
        )

        atoms = data["atoms"]
        field_names = [f for f in atoms.dtype.names if f not in self.excluded_items]
        self.__file.write(f"ITEM: ATOMS {' '.join(field_names)}\n")

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
            self.__file.write(
                " ".join(
                    fmt % row[field_name]
                    for fmt, field_name in zip(formatters, field_names)
                )
                + "\n"
            )
