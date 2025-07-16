"""Class to read LAMMPS log files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import numpy as np
from numpy import typing as npt


@dataclass
class LAMMPSLogReader:
    """Class to read LAMMPS log files.

    Parameters
    ----------
    path_log : Path
        The path to the LAMMPS log file.

    Yields
    -------
    Generator[dict[str, Any], None, None]
        A dictionary containing the thermo data.

    Note
    ----
    This generator yields a dictionary with a single key for the thermo data,
    this ensures compatibility if this reader is extended to read more data in the future.
    """

    path_log: Path
    data: dict = field(default_factory=lambda: {"thermo": None}, init=False)
    thermo: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.dtype([])), init=False
    )

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """Reads a LAMMPS log file.

        Parameters
        ----------
        path_log : Path
            The path to the log file.

        Returns
        -------
        npt.NDArray
            The data as a structured array.
        """
        self.__reset()
        expecting_header = False  # Next line will be a header
        expecting_data = False  # Next lines will be data
        for line in open(self.path_log, "r", encoding="utf-8"):
            if line.startswith("Per MPI "):
                expecting_header = True
                self.__reset()
                continue
            elif line.startswith("Loop time") or line.startswith("Fix halt"):
                if self.thermo.size > 0:
                    self.__fill()
                    yield self.data
                expecting_header = False
                expecting_data = False
                self.__reset()
                continue
            if expecting_header:
                items = line.split()
                types = [np.float64] * len(items)
                if "Step" in items:
                    types[items.index("Step")] = np.int64
                dtype = np.dtype(list(zip(items, types)))
                self.thermo = np.empty(0, dtype=dtype)
                expecting_header = False
                expecting_data = True
                continue
            if expecting_data:
                if line.startswith("WARNING"):
                    continue
                row = np.array(tuple(map(float, line.split())), dtype=dtype)
                self.thermo = np.append(self.thermo, row)

    def __reset(self) -> None:
        """Reset the internal state of the reader."""
        self.thermo = np.empty(0, dtype=np.dtype([]))
        self.data = {"thermo": self.thermo}

    def __fill(self) -> None:
        """Fill the data dictionary with data."""
        self.data["thermo"] = self.thermo
