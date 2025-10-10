"""This module contains the `BZIP2LAMMPSWriterMPI` class."""

# pylint: disable=no-name-in-module, broad-except

import bz2
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
class BZIP2LAMMPSWriterMPI(MPIExceptionHandlerMixin):
    """A class to write data like a LAMMPS dump bzip2 compressed file in parallel using MPI.

    Note
    ----
    Assumed orthogonal simulation box.

    Note
    ----
    All ranks compute their local subdomain of atoms, then rank 0 collects and writes the data.

    Parameters
    ----------
    file_path : Path
        Output .bz2 path.
    mode : str
        File mode for the *container* file opened on rank 0 (default: 'wb').
    encoding : str, optional (default=irradiapy.config.ENCODING)
        The file encoding.
    comm : MPI.Comm, optional (default=mpi4py.MPI.COMM_WORLD)
        The MPI communicator.
    compresslevel : int, optional (default=9)
        Compression level for bzip2.
    int_format : str, optional (default=irradiapy.config.INT_FORMAT)
        The format for integers.
    float_format : str, optional (default=irradiapy.config.FLOAT_FORMAT)
        The format for floats.
    excluded_items : list[str], optional (default=irradiapy.config.EXCLUDED_ITEMS)
        Atom fields to exclude from output.
    """

    file_path: Path
    mode: str = "wb"
    encoding: str = field(default_factory=lambda: config.ENCODING)
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    compresslevel: int = 9
    int_format: str = field(default_factory=lambda: config.INT_FORMAT)
    float_format: str = field(default_factory=lambda: config.FLOAT_FORMAT)
    excluded_items: list[str] = field(default_factory=lambda: config.EXCLUDED_ITEMS)

    __file: TextIO | None = field(default=None, init=False)
    __rank: int = field(init=False, repr=False)
    __size: int = field(init=False, repr=False)
    __comm_tag: int = field(
        default_factory=MPITagAllocator.get_tag, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Opens the file associated with this writer."""
        self.__rank = self.comm.Get_rank()
        self.__size = self.comm.Get_size()
        self.__file = None
        if self.__rank == 0:
            try:
                self.__file = open(self.file_path, self.mode)
            except Exception:
                self._handle_exception()

    def __enter__(self) -> "BZIP2LAMMPSWriterMPI":
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
        if self.__rank == 0 and self.__file is not None and not self.__file.closed:
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
        atoms = data["subdomain_atoms"]
        field_names = [f for f in atoms.dtype.names if f not in self.excluded_items]

        formatters: list[str] = []
        for field_name in field_names:
            dtype = atoms.dtype[field_name]
            if dtype.kind == "i":
                formatters.append(self.int_format)
            elif dtype.kind == "f":
                formatters.append(self.float_format)
            else:
                formatters.append("%s")

        # Header
        if self.__rank == 0:
            header_lines = []
            if data.get("time") is not None:
                header_lines.append(f"ITEM: TIME\n{self.float_format % data['time']}\n")
            header_lines.append(
                f"ITEM: TIMESTEP\n{self.int_format % data['timestep']}\n"
            )
            header_lines.append(
                f"ITEM: NUMBER OF ATOMS\n{self.int_format % data['natoms']}\n"
            )
            header_lines.append(f"ITEM: BOX BOUNDS {' '.join(data['boundary'])}\n")
            header_lines.append(
                f"{self.float_format % data['xlo']} {self.float_format % data['xhi']}\n"
            )
            header_lines.append(
                f"{self.float_format % data['ylo']} {self.float_format % data['yhi']}\n"
            )
            header_lines.append(
                f"{self.float_format % data['zlo']} {self.float_format % data['zhi']}\n"
            )
            header_lines.append(f"ITEM: ATOMS {' '.join(field_names)}\n")
            header_bytes = "".join(header_lines).encode(self.encoding)
            self.__file.write(
                bz2.compress(header_bytes, compresslevel=self.compresslevel)
            )

        self.comm.Barrier()
        lines_chunk = self.__atoms_rank_to_string(atoms, field_names, formatters)
        if lines_chunk and not lines_chunk.endswith("\n"):
            lines_chunk += "\n"
        comp = bz2.compress(
            lines_chunk.encode(self.encoding), compresslevel=self.compresslevel
        )

        if self.__rank == 0:
            self.__file.write(comp)
            for sender__rank in range(1, self.__size):
                self.comm.send(None, dest=sender__rank, tag=self.__comm_tag + 1)
                payload = self.comm.recv(source=sender__rank, tag=self.__comm_tag)
                self.__file.write(payload)
        else:
            self.comm.recv(source=0, tag=self.__comm_tag + 1)
            self.comm.send(comp, dest=0, tag=self.__comm_tag)
        self.comm.Barrier()
