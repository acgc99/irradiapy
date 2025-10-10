"""This module contains the `LAMMPSWriterMPI` class."""

# pylint: disable=no-name-in-module, broad-except

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy.typing as npt
from mpi4py import MPI

from irradiapy import config
from irradiapy.utils.mpi import (
    MPIExceptionHandlerMixin,
    MPITagAllocator,
    mpi_safe_method,
)


@dataclass
class LAMMPSWriterMPI(MPIExceptionHandlerMixin):
    """A class to write data like a LAMMPS dump file in parallel using MPI.

    Note
    ----
    Assumes orthogonal simulation box.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    mode : str, optional (default="w")
        The file open mode.
    encoding : str, optional (default=irradiapy.config.ENCODING)
        The file encoding.
    comm : MPI.Comm, optional (default=mpi4py.MPI.COMM_WORLD)
        The MPI communicator.
    int_format : str, optional (default=irradiapy.config.INT_FORMAT)
        The format for integers.
    float_format : str, optional (default=irradiapy.config.FLOAT_FORMAT)
        The format for floats.
    excluded_items : list[str], optional (default=irradiapy.config.EXCLUDED_ITEMS)
        Atom fields to exclude from output.
    """

    file_path: Path
    mode: str = "w"
    encoding: str = field(default_factory=lambda: config.ENCODING)
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    int_format: str = field(default_factory=lambda: config.INT_FORMAT)
    float_format: str = field(default_factory=lambda: config.FLOAT_FORMAT)
    excluded_items: list[str] = field(default_factory=lambda: config.EXCLUDED_ITEMS)

    __file: TextIO = field(default=None, init=False)
    __rank: int = field(init=False)
    __commsize: int = field(init=False)
    __comm_tag: int = field(default_factory=MPITagAllocator.get_tag, init=False)

    def __post_init__(self) -> None:
        """Opens the file associated with this writer."""
        self.__rank = self.comm.Get_rank()
        self.__commsize = self.comm.Get_size()
        self.__file = None
        if self.__rank == 0:
            try:
                self.__file = open(self.file_path, self.mode, encoding=self.encoding)
            except Exception:
                self._handle_exception()

    def __enter__(self) -> "LAMMPSWriterMPI":
        return self

    def __del__(self) -> None:
        if self.__rank == 0 and self.__file is not None:
            self.__file.close()

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        if self.__rank == 0 and self.__file is not None:
            self.__file.close()
        return False

    def __atoms_rank_to_string(
        self, atoms_rank: npt.NDArray, field_names: list[str], formatters: list[str]
    ) -> str:
        """Converts the atoms_rank array to a formatted string.

        Parameters
        ----------
        atoms_rank : npt.NDArray
            The atoms_rank array to be converted.
        field_names : list[str]
            The names of the fields in the structured array.
        formatters : list[str]
            The format strings for each field.

        Returns
        -------
        str
            A formatted string representation of the atoms_rank array.
        """
        lines_chunk = "\n".join(
            " ".join(
                fmt % atom[field_name]
                for fmt, field_name in zip(formatters, field_names)
            )
            for atom in atoms_rank
        )
        return lines_chunk

    @mpi_safe_method
    def close(self) -> None:
        """Closes the file associated with this writer."""
        if self.__rank == 0 and not self.__file.closed:
            self.__file.close()

    @mpi_safe_method
    def write(self, data: dict[str, Any]) -> None:
        """Write the data to the file.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the data to be written. The keys should
            include "timestep", "boundary", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi",
            and "atoms". Optional keys: "time".
        """
        atoms = data["atoms"]
        field_names = [f for f in atoms.dtype.names if f not in self.excluded_items]

        formatters = []
        for field_name in field_names:
            dtype = atoms.dtype[field_name]
            if dtype.kind == "i":
                formatters.append(self.int_format)
            elif dtype.kind == "f":
                formatters.append(self.float_format)
            else:
                formatters.append("%s")

        if self.__rank == 0:
            if "time" in data:
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
            self.__file.write(f"ITEM: ATOMS {' '.join(field_names)}\n")

        self.comm.Barrier()
        lines_chunk = self.__atoms_rank_to_string(
            data["atoms"], field_names, formatters
        )
        if self.__rank == 0:
            self.__file.write(lines_chunk)
            if lines_chunk:
                self.__file.write("\n")
            for sender_rank in range(1, self.__commsize):
                self.comm.send(None, dest=sender_rank, tag=self.__comm_tag + 1)
                msg = self.comm.recv(source=sender_rank, tag=self.__comm_tag)
                self.__file.write(msg)
                if msg:
                    self.__file.write("\n")
        else:
            self.comm.recv(source=0, tag=self.__comm_tag + 1)
            self.comm.send(lines_chunk, dest=0, tag=self.__comm_tag)
        self.comm.Barrier()
