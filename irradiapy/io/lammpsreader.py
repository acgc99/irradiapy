"""This module contains the `LAMMPSReader` class."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Type, Union

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
    """

    file_path: Path

    def __get_dtype(
        self, line: str
    ) -> tuple[list[str], list[Type[Union[int, float]]], np.dtype]:
        """Get the data type of the simulation data.

        Parameters
        ----------
        line : str
            The line containing the data type.

        Returns
        -------
        tuple[list[str], list[Type[Union[int, float]], np.dtype]
            The names of the data items, the types of the data items,
            and the data type.
        """
        items = line.split()[2:]
        types = [int if item in ["id", "type", "element"] else float for item in items]
        dtype = np.dtype([(item, type) for item, type in zip(items, types)])
        return items, types, dtype

    def __iter__(
        self,
    ) -> Generator[
        tuple[
            Union[float, None],
            int,
            int,
            tuple[str, str, str],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            np.ndarray,
        ],
        None,
        None,
    ]:
        """Read the file as an iterator, timestep by timestep.

        Yields
        ------
        tuple[int, np.ndarray]
            The timestep and the data at that timestep.
        """
        with open(self.file_path, encoding="utf-8") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                time = None
                if line == "ITEM: TIME\n":
                    time = float(file.readline())
                    file.readline()
                timestep = int(file.readline())
                file.readline()
                natoms = int(file.readline())
                boundary = list(file.readline().split()[-3:])
                boundary = (boundary[0][0], boundary[1][0], boundary[2][0])
                dimsx = tuple(map(float, file.readline().split()))
                dimsy = tuple(map(float, file.readline().split()))
                dimsz = tuple(map(float, file.readline().split()))

                line = file.readline()
                items, types, dtype = self.__get_dtype(line)
                data = np.empty(natoms, dtype=dtype)
                for i in range(natoms):
                    line = file.readline().split()
                    for j, item in enumerate(items):
                        data[i][item] = types[j](line[j])
                yield time, timestep, natoms, boundary, dimsx, dimsy, dimsz, data
