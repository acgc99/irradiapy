"""This module contains the `XYZWriter` class."""

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TextIO

import numpy.typing as npt

from irradiapy import config


@dataclass
class XYZWriter:
    """Class for writing structured data to an XYZ file format.

    Parameters
    ----------
    file_path : Path
        Path to the file where data will be written.
    mode : str, optional (default="w")
        File open mode.
    encoding : str, optional (default=irradiapy.config.ENCODING)
        The file encoding.
    int_format : str, optional (default=irradiapy.config.INT_FORMAT)
        Format for integers.
    float_format : str, optional (default=irradiapy.config.FLOAT_FORMAT)
        Format for floats.
    """

    file_path: Path
    mode: str = "w"
    encoding: str = field(default_factory=lambda: config.ENCODING)
    int_format: str = field(default_factory=lambda: config.INT_FORMAT)
    float_format: str = field(default_factory=lambda: config.FLOAT_FORMAT)
    file: TextIO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.file = open(self.file_path, self.mode, encoding="utf-8")

    def __enter__(self) -> "XYZWriter":
        return self

    def __del__(self) -> None:
        self.file.close()

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        """Exit the runtime context related to this object."""
        if self.file and not self.file.closed:
            self.file.close()
        return False

    def __get_properties(
        self, dtype: npt.DTypeLike
    ) -> tuple[tuple[str, ...], int, list[str], list[int]]:
        """Get the properties of the given data.

        Parameters
        ----------
        dtype : npt.DTypeLike
            Datatype of the given data.

        Returns
        -------
        tuple
            A tuple containing the names, count, types, multiplicities and formatters
            of the properties.
        """
        name_props = dtype.names
        count_props = len(name_props)
        kind_map = {"i": "I", "f": "R", "U": "S"}
        type_props = []
        for descr in dtype.descr:
            kind = descr[1][1]
            if kind not in kind_map:
                raise TypeError(f"Unexpected dtype kind: {kind}")
            type_props.append(kind_map[kind])
        multiplicity_props = [
            dtype[name_prop].shape[0] if dtype[name_prop].shape else 1
            for name_prop in name_props
        ]

        formatters = []
        for kind in type_props:
            if kind == "I":
                formatters.append(self.int_format)
            elif kind == "R":
                formatters.append(self.float_format)
            else:
                formatters.append("%s")

        return name_props, count_props, type_props, multiplicity_props, formatters

    def __get_comment(
        self,
        name_props: tuple,
        count_props: int,
        type_props: list,
        multiplicity_props: list,
    ) -> str:
        """Generate file comment following xyz guidelines.

        Parameters
        ----------
        name_props : tuple
            Property names.
        count_props : int
            Number of properties.
        type_props : list
            Property types.
        multiplicity_props : list
            Property multiplicities.

        Returns
        -------
        str
            Comment.
        """
        comment = "Properties=" + ":".join(
            f"{name_props[i]}:{type_props[i]}:{multiplicity_props[i]}"
            for i in range(count_props)
        )
        return comment

    def __data_to_line(
        self,
        data: npt.NDArray,
        name_props: tuple,
        count_props: int,
        multiplicity_props: list,
        formatters: list,
    ) -> str:
        """Transform data into string to write.

        Parameters
        ----------
        data : npt.NDArray
            Data to write.
        name_props : tuple
            Property names.
        count_props : int
            Number of properties.
        multiplicity_props : list
            Property multiplicities.
        formatters : list
            Formatters for each property.

        Returns
        -------
        str
            Data as string.
        """
        line = " ".join(
            (
                formatters[i] % data[name_props[i]]
                if multiplicity_props[i] == 1
                else " ".join(formatters[i] % v for v in data[name_props[i]])
            )
            for i in range(count_props)
        )
        return line

    def write(self, datas: npt.NDArray, extra_comment: str = "") -> None:
        """
        Writes the given data into the file.

        Parameters
        ----------
        datas : npt.NDArray
            Data to write.
        extra_comment : str, optional
            Additional info to add at the end of the comment. Must follow xyz guidelines.
            Example: 'Info="Fe irradiatied in Fe, Ion 1"'
        """
        natoms = datas.size
        dtype = datas.dtype
        name_props, count_props, type_props, multiplicity_props, formatters = (
            self.__get_properties(dtype)
        )

        comment = self.__get_comment(
            name_props, count_props, type_props, multiplicity_props
        )
        full_comment = f"{comment} {extra_comment}" if extra_comment else comment
        self.file.write(f"{natoms}\n")
        self.file.write(f"{full_comment}\n")
        for data in datas:
            line = self.__data_to_line(
                data, name_props, count_props, multiplicity_props, formatters
            )
            self.file.write(f"{line}\n")

    def close(self) -> None:
        """Close the file associated with this writer."""
        if self.file and not self.file.closed:
            self.file.close()
