"""This module contains the `XYZReader` class."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, TextIO

import numpy as np
import numpy.typing as npt


@dataclass
class XYZReader:
    """A class to read data from an extended XYZ file.

    Attributes
    ----------
    file_path : Path
        The path to the XYZ file.
    dtype : npt.DTypeLike
        The data type of the properties in the file. By default, it is set to
        `None` and will be determined from the file.
    """

    file_path: Path
    encoding: str = "utf-8"
    dtype: npt.DTypeLike = field(default=None, init=False)
    file: TextIO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.file = self.file_path.open(encoding=self.encoding)

    def __del__(self) -> None:
        self.file.close()

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over subfiles in the XYZ file.

        Yields
        ------
        npt.NDArray
            Array of atom data for each subfile.
        """
        while True:
            line = self.file.readline()
            if not line:
                break
            natoms = int(line)
            line = self.file.readline()
            name_props, type_props, multiplicity_props, dtype = self.__get_properties(
                line
            )
            atoms = np.empty(natoms, dtype=dtype)
            for i in range(natoms):
                line = self.file.readline()
                atoms[i] = self.__line_to_data(
                    line, name_props, type_props, multiplicity_props, dtype
                )
            yield atoms
        self.file.close()

    def __get_properties(
        self, comment: str
    ) -> tuple[list[str], list[type], list[int], np.dtype]:
        """Sets properties using the comment line.

        Parameters
        ----------
        comment : str
            Comment line.

        Returns
        -------
        tuple[list[str], list[type], list[int], np.dtype]
            Properties names, types, multiplicities, and dtype.
        """
        match = re.search(r"Properties=([^ \n]+)", comment)
        if not match:
            raise ValueError("Missing or invalid comment line format.")
        properties = match.group(1).split(":")
        num_properties = len(properties) // 3
        name_props = [properties[i * 3] for i in range(num_properties)]
        type_props = [
            self.__map_type(properties[i * 3 + 1]) for i in range(num_properties)
        ]
        multiplicity_props = [int(properties[i * 3 + 2]) for i in range(num_properties)]
        dtype = np.dtype(
            [
                (
                    (name_props[i], type_props[i])
                    if multiplicity_props[i] == 1
                    else (
                        name_props[i],
                        type_props[i],
                        multiplicity_props[i],
                    )
                )
                for i in range(num_properties)
            ]
        )
        return name_props, type_props, multiplicity_props, dtype

    def __map_type(self, type_str: str) -> type:
        """Maps type string to Python type.

        Parameters
        ----------
        type_str : str
            Type string.
        """
        if type_str == "S":
            return str
        elif type_str == "I":
            return int
        elif type_str == "R":
            return float
        else:
            raise TypeError(f"Unexpected type string: {type_str}")

    def __line_to_data(
        self,
        line: str,
        name_props: list[str],
        type_props: list[type],
        multiplicity_props: list[int],
        dtype: np.dtype,
    ) -> npt.ArrayLike:
        """Turns one line of data into a numpy array.

        Parameters
        ----------
        line : str
            Line containing the data.
        name_props : list[str]
            Names of the properties.
        type_props : list[type]
            Types of the properties.
        multiplicity_props : list[int]
            Multiplicities of the properties.
        dtype : np.dtype
            Data type of the properties.

        Returns
        -------
        npt.ArrayLike
            The data in the line.
        """
        output = np.empty(1, dtype=dtype)
        data = line.split()
        col = 0
        for i, name_prop in enumerate(name_props):
            multiplicity_prop = multiplicity_props[i]
            type_prop = type_props[i]
            if multiplicity_prop == 1:
                output[name_prop] = type_prop(data[col])
            else:
                output[name_prop] = [
                    type_prop(data[col + j]) for j in range(multiplicity_prop)
                ]
            col += multiplicity_prop
        return output[0]
