"""This module contains the `LAMMPSReaderMPI` class."""

# pylint: disable=no-name-in-module, broad-except

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Generator, TextIO, Tuple, Type

import numpy as np
from mpi4py import MPI
from numpy import typing as npt

from irradiapy.utils.mpi import (
    MPIExceptionHandlerMixin,
    MPITagAllocator,
    mpi_safe_method,
    mpi_subdomains_decomposition,
)


@dataclass
class LAMMPSReaderMPI(MPIExceptionHandlerMixin):
    """A class to read data from a LAMMPS dump file in parallel using MPI.

    Note
    ----
    Assumed orthogonal simulation box.

    Note
    ----
    Rank 0 reads each timestep one by one, then scatters strings of atom data
    to all ranks, which build local numpy structured arrays.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    encoding : str, optional (default="utf-8")
        The file encoding.
    comm : MPI.Comm, optional (default=mpi4py.MPI.COMM_WORLD)
        The MPI communicator.

    Yields
    ------
    dict
        A dictionary containing the timestep data with keys:
        'time' (optional), 'timestep', 'natoms', 'boundary', 'xlo', 'xhi',
        'ylo', 'yhi', 'zlo', 'zhi', and 'atoms' (as a numpy structured array).
    """

    file_path: Path
    encoding: str = "utf-8"
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)

    __file: TextIO = field(default=None, init=False)
    __rank: int = field(init=False)
    __commsize: int = field(init=False)
    __comm_tag: int = field(default_factory=MPITagAllocator.get_tag, init=False)
    __nx: int = field(init=False)
    __ny: int = field(init=False)
    __nz: int = field(init=False)

    def __post_init__(self):
        self.__rank = self.comm.Get_rank()
        self.__commsize = self.comm.Get_size()
        self.__nx, self.__ny, self.__nz = mpi_subdomains_decomposition(self.__commsize)
        ix = self.__rank % self.__nx
        iy = (self.__rank // self.__nx) % self.__ny
        iz = self.__rank // (self.__nx * self.__ny)
        self._domain_index = (ix, iy, iz)
        self.__file = None
        if self.__rank == 0:
            try:
                self.__file = open(self.file_path, "r", encoding=self.encoding)
            except Exception:
                self._handle_exception()

    def __enter__(self) -> "LAMMPSReaderMPI":
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
        """Exits the context manager."""
        if self.__rank == 0 and self.__file is not None:
            self.__file.close()
        return False

    def __get_dtype(
        self, file: TextIO
    ) -> Tuple[list[str], list[Type[int | float]], np.dtype]:
        items = file.readline().split()[2:]
        types = [
            np.int64 if it in ("id", "type", "element", "size") else np.float64
            for it in items
        ]
        if all(c in items for c in ("x", "y", "z")):
            items += ["xs", "ys", "zs"]
            types += [np.float64] * 3
        return items, types, np.dtype(list(zip(items, types)))

    def __process_header(self, file: TextIO) -> Dict[str, Any]:
        data = defaultdict(None)
        line = file.readline()
        if not line:
            return {}
        if line.strip() == "ITEM: TIME":
            data["time"] = float(file.readline())
            file.readline()
        else:
            # rewind if no time item
            file.seek(file.tell() - len(line))
        data["timestep"] = int(file.readline())
        file.readline()
        data["natoms"] = int(file.readline())
        data["boundary"] = file.readline().split()[3:]
        bounds = []
        for _ in range(3):
            b = file.readline().split()
            bounds.append(b)
        data["xlo"], data["xhi"] = map(float, bounds[0][:2])
        data["ylo"], data["yhi"] = map(float, bounds[1][:2])
        data["zlo"], data["zhi"] = map(float, bounds[2][:2])
        return data

    @mpi_safe_method
    def __iter__(self) -> Generator[Tuple[Dict[str, Any], npt.NDArray], None, None]:
        while True:
            # header broadcast
            data = self.comm.bcast(
                self.__process_header(self.__file) if self.__rank == 0 else None, root=0
            )
            if data is None or not data:
                break

            # dtype broadcast
            items, types, dtype = self.comm.bcast(
                (
                    self.__get_dtype(self.__file)
                    if self.__rank == 0
                    else (None, None, None)
                ),
                root=0,
            )
            data.update({"items": items, "types": types, "dtype": dtype})

            # calculate raw line counts
            natoms = data["natoms"]
            counts = [
                (natoms // self.__commsize)
                + (1 if i < (natoms % self.__commsize) else 0)
                for i in range(self.__commsize)
            ]

            # distribute chunks
            if self.__rank == 0:
                chunks = []
                for cnt in counts:
                    chunk = [self.__file.readline().split() for _ in range(cnt)]
                    chunks.append(chunk)
                raw = chunks[0]
                for r in range(1, self.__commsize):
                    self.comm.send(chunks[r], dest=r, tag=self.__comm_tag)
            else:
                raw = self.comm.recv(source=0, tag=self.__comm_tag)

            # build structured array
            arr = np.empty(len(raw), dtype=dtype)
            for i, fields in enumerate(raw):
                for j, key in enumerate(items):
                    if key in ("xs", "ys", "zs"):
                        continue
                    arr[key][i] = types[j](fields[j])

            # subdomain info: indices and physical bounds
            xlo, xhi = data["xlo"], data["xhi"]
            ylo, yhi = data["ylo"], data["yhi"]
            zlo, zhi = data["zlo"], data["zhi"]
            dx = (xhi - xlo) / self.__nx
            dy = (yhi - ylo) / self.__ny
            dz = (zhi - zlo) / self.__nz
            ix, iy, iz = self._domain_index
            data["subdomain_index"] = (ix, iy, iz)
            data["subdomain_bounds"] = {
                "xlo": xlo + ix * dx,
                "xhi": xlo + (ix + 1) * dx,
                "ylo": ylo + iy * dy,
                "yhi": ylo + (iy + 1) * dy,
                "zlo": zlo + iz * dz,
                "zhi": zlo + (iz + 1) * dz,
            }
            # normalize scaled coordinates based on true positions
            if all(c in items for c in ("xs", "ys", "zs")):
                arr["xs"] = (arr["x"] - xlo) / (xhi - xlo)
                arr["ys"] = (arr["y"] - ylo) / (yhi - ylo)
                arr["zs"] = (arr["z"] - zlo) / (zhi - zlo)
            # attach atoms
            data["atoms"] = arr
            data["natoms"] = len(arr)
            yield data

        if self.__rank == 0 and self.__file is not None:
            self.__file.close()

    @mpi_safe_method
    def close(self) -> None:
        """Closes the file associated with this reader."""
        if self.__rank == 0 and not self.__file.closed:
            self.__file.close()
