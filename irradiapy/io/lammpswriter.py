"""This module contains the `LAMMPSWriter` class."""

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TextIO

from irradiapy import config


@dataclass
class LAMMPSWriter:
    """A class to write data like a LAMMPS dump file.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    mode : str
        The file open mode. Default: `"w"`.
    excluded_items : list[str]
        Atom fields to exclude from output. Default: `irradiapy.config.EXCLUDED_ITEMS`.
    encoding : str
        The file encoding. Default: `irradiapy.config.ENCODING`.
    int_format : str
        The format for integers. Default: `irradiapy.config.INT_FORMAT`.
    float_format : str
        The format for floats. Default: `irradiapy.config.FLOAT_FORMAT`.
    """

    file_path: Path
    mode: str = "w"
    excluded_items: list[str] = field(default_factory=lambda: config.EXCLUDED_ITEMS)
    encoding: str = field(default_factory=lambda: config.ENCODING)
    int_format: str = field(default_factory=lambda: config.INT_FORMAT)
    float_format: str = field(default_factory=lambda: config.FLOAT_FORMAT)
    file: TextIO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.file = open(self.file_path, self.mode, encoding=self.encoding)

    def __enter__(self) -> "LAMMPSWriter":
        return self

    def __del__(self) -> None:
        self.file.close()

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        self.file.close()
        return False

    def close(self) -> None:
        """Closes the file associated with this writer."""
        self.file.close()

    def write(self, data: dict) -> None:
        """Writes the data (from LAMMPSReader/BZIP2LAMMPSReader) to the file.

        Parameters
        ----------
        data : dict
            The dictionary containing the timestep data.
        """
        if data.get("time") is not None:
            self.file.write(f"ITEM: TIME\n{data['time']}\n")
        self.file.write(f"ITEM: TIMESTEP\n{data['timestep']}\n")
        self.file.write(f"ITEM: NUMBER OF ATOMS\n{data['natoms']}\n")
        self.file.write(f"ITEM: BOX BOUNDS {' '.join(data['boundary'])}\n")
        self.file.write(
            f"{self.float_format % data['xlo']} {self.float_format % data['xhi']}\n"
        )
        self.file.write(
            f"{self.float_format % data['ylo']} {self.float_format % data['yhi']}\n"
        )
        self.file.write(
            f"{self.float_format % data['zlo']} {self.float_format % data['zhi']}\n"
        )

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
