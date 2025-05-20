"""This module contains the `XYZReader` class."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

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
        The data type of the properties in the file.
    """

    file_path: Path
    dtype: npt.DTypeLike = field(default=None, init=False)

    def _get_properties(
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
            self._map_type(properties[i * 3 + 1]) for i in range(num_properties)
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

    @staticmethod
    def _get_kind(kind: str) -> type:
        """Maps dtype kind to Python type.

        Parameters
        ----------
        kind : str
            Kind of the dtype.
        """
        if kind == "i":
            return int
        elif kind == "f":
            return float
        elif kind == "U":
            return str
        else:
            raise TypeError("Unexpected dtype")

    @staticmethod
    def _map_type(type_str: str) -> type:
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

    def _line_to_data(
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

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over subfiles in the XYZ file.

        Yields
        ------
        np.ndarray
            Array of atom data for each subfile.
        """
        with self.file_path.open(encoding="utf-8") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                natoms = int(line)
                line = file.readline()
                name_props, type_props, multiplicity_props, dtype = (
                    self._get_properties(line)
                )
                atoms = np.empty(natoms, dtype=dtype)
                for i in range(natoms):
                    line = file.readline()
                    atoms[i] = self._line_to_data(
                        line, name_props, type_props, multiplicity_props, dtype
                    )
                yield atoms
