"""This module contains the `LAMMPSReader` class."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, TextIO, Type

import numpy as np


@dataclass
class LAMMPSReader:
    """A class to read data from a LAMMPS dump file.

    Note
    ----
    Assumed orthogonal simulation box.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    encoding : str, optional (default="utf-8")
        The file encoding.
    """

    file_path: Path
    encoding: str = "utf-8"
    file: TextIO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.file = open(self.file_path, encoding=self.encoding)

    def __del__(self) -> None:
        if self.file is not None:
            self.file.close()

    def __iter__(
        self,
    ) -> Generator[
        dict,  # Changed from tuple[...] to dict
        None,
        None,
    ]:
        """Read the file as an iterator, timestep by timestep.

        Yields
        ------
        dict
            A dictionary containing the timestep data with keys:
            'time' (optional), 'timestep', 'natoms', 'boundary', 'xlo', 'xhi',
            'ylo', 'yhi', 'zlo', 'zhi', and 'atoms' (as a numpy structured array).
        """
        while True:
            data = defaultdict(None)
            line = self.file.readline()
            if not line:
                break
            if line == "ITEM: TIME\n":
                data["time"] = float(self.file.readline())
                self.file.readline()
            data["timestep"] = int(self.file.readline())
            self.file.readline()
            data["natoms"] = int(self.file.readline())
            data["boundary"] = self.file.readline().split()[-3:]
            data["xlo"], data["xhi"] = map(float, self.file.readline().split())
            data["ylo"], data["yhi"] = map(float, self.file.readline().split())
            data["zlo"], data["zhi"] = map(float, self.file.readline().split())

            line = self.file.readline()
            items, types, dtype = self.__get_dtype(line)

            data["atoms"] = np.empty(data["natoms"], dtype=dtype)
            for i in range(data["natoms"]):
                line = self.file.readline().split()
                for j, item in enumerate(items):
                    data["atoms"][i][item] = types[j](line[j])
            yield data

        self.file.close()

    def __get_dtype(
        self, line: str
    ) -> tuple[list[str], list[Type[int | float]], np.dtype]:
        """Get the data type of the simulation data.

        Parameters
        ----------
        line : str
            The line containing the data type.

        Returns
        -------
        tuple[list[str], list[Type[int | float]], np.dtype]
            The names of the data items, the types of the data items,
            and the data type.
        """
        items = line.split()[2:]
        types = [
            int if item in ("id", "type", "element", "size") else float
            for item in items
        ]
        dtype = np.dtype([(item, type) for item, type in zip(items, types)])
        return items, types, dtype

    def close(self) -> None:
        """Close the file associated with this reader."""
        self.file.close()
