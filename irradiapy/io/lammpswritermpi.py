"""This module contains the `LAMMPSWriterMPI` class."""

# pylint: disable=no-name-in-module, broad-except

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import numpy.typing as npt
from mpi4py import MPI

from irradiapy.mpi_utils import (
    MPIExceptionHandlerMixin,
    MPITagAllocator,
    mpi_safe_method,
)


@dataclass
class LAMMPSWriterMPI(MPIExceptionHandlerMixin):
    """A class to write data like a LAMMPS dump file in parallel using MPI.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    mode : str
        The file open mode (default: 'w').
    excluded_items : list[str]
        Atom fields to exclude from output (default: ["xs", "ys", "zs"]).
    encoding : str
        The file encoding (default: 'utf-8').
    int_format : str
        The format for integers (default: '%d').
    float_format : str
        The format for floats (default: '%g').
    comm : MPI.Comm
        The MPI communicator (default: MPI.COMM_WORLD).
    """

    file_path: Path
    mode: str = "w"
    excluded_items: list[str] = field(default_factory=lambda: ["xs", "ys", "zs"])
    encoding: str = "utf-8"
    int_format: str = "%d"
    float_format: str = "%g"
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    __rank: int = field(init=False)
    __commsize: int = field(init=False)
    __closed: bool = field(default=False, init=False)
    __comm_tag: int = field(default_factory=MPITagAllocator.get_tag, init=False)

    def __post_init__(self) -> None:
        """Opens the file associated with this writer."""
        self.__rank = self.comm.Get_rank()
        self.__commsize = self.comm.Get_size()
        self.file = None
        if self.__rank == 0:
            try:
                self.file = open(self.file_path, self.mode, encoding=self.encoding)
            except Exception:
                self._handle_exception()

    def __enter__(self) -> "LAMMPSWriterMPI":
        """Enters the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> bool:
        """Exits the context manager."""
        self.close()
        return False

    def __atoms_rank_to_string(
        self, atoms_rank: npt.NDArray, field_names: list[str], formatters: list[str]
    ) -> str:
        """Converts the atoms_rank array to a formatted string.

        Parameters
        ----------
        atoms_rank : np.ndarray
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
        return "\n".join(
            " ".join(
                fmt % atom[field_name]
                for fmt, field_name in zip(formatters, field_names)
            )
            for atom in atoms_rank
        )

    @mpi_safe_method
    def close(self) -> None:
        """Closes the file associated with this writer."""
        if self.__rank == 0 and not self.__closed:
            self.__closed = True
            self.file.close()

    @mpi_safe_method
    def write(self, data: dict[str, Any]) -> None:
        """Writes the data to the file.

        Note
        ----
        Assumes orthogonal simulation box.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the data to be written. The keys should
            include 'timestep', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi',
            and any other fields to be written as atom properties.
        """
        items = data["items"].copy()
        for excluded_item in self.excluded_items:
            if excluded_item in items:
                items.remove(excluded_item)

        formatters = []
        for field_name in items:
            dtype = data["atoms_rank"].dtype[field_name]
            if dtype.kind == "i":
                formatters.append(self.int_format)
            elif dtype.kind == "f":
                formatters.append(self.float_format)
            else:
                formatters.append("%s")

        if self.__rank == 0:
            if "time" in data:
                self.file.write(f"ITEM: TIME\n{data['time']}\n")
            self.file.write(f"ITEM: TIMESTEP\n{data['timestep']}\n")
            self.file.write(f"ITEM: NUMBER OF ATOMS\n{data['natoms']}\n")
            self.file.write(f"ITEM: BOX BOUNDS {' '.join(data['boundary'])}\n")
            self.file.write(f"{data['xlo']} {data['xhi']}\n")
            self.file.write(f"{data['ylo']} {data['yhi']}\n")
            self.file.write(f"{data['zlo']} {data['zhi']}\n")
            self.file.write(f"ITEM: ATOMS {' '.join(items)}\n")

        self.comm.Barrier()
        atoms_rank_str = self.__atoms_rank_to_string(
            data["atoms_rank"], items, formatters
        )
        if self.__rank == 0:
            self.file.write(atoms_rank_str)
            if atoms_rank_str:
                self.file.write("\n")
            for sender_rank in range(1, self.__commsize):
                self.comm.send(None, dest=sender_rank, tag=self.__comm_tag + 1)
                msg = self.comm.recv(source=sender_rank, tag=self.__comm_tag)
                self.file.write(msg)
                if msg:
                    self.file.write("\n")
        else:
            self.comm.recv(source=0, tag=self.__comm_tag + 1)
            self.comm.send(atoms_rank_str, dest=0, tag=self.__comm_tag)
        self.comm.Barrier()
