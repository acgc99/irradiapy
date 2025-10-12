"""This module contains the `BZIP2LAMMPSReaderMPI` class."""

# pylint: disable=no-name-in-module, broad-except

import codecs
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Generator, Tuple, Type

import indexed_bzip2 as ibz2
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
class BZIP2LAMMPSReaderMPI(MPIExceptionHandlerMixin):
    """A class to read data from a LAMMPS dump file compressed with bzip2 in parallel using MPI.

    Note
    ----
    Assumed orthogonal simulation box.

    Note
    ----
    Rank 0 performs indexed, multi-threaded decompression using `indexed_bzip2`
    of each timestep one by one, then scatters strings of atom data to all ranks,
    which build local numpy structured arrays.

    Parameters
    ----------
    file_path : Path
        Path to the .bz2 LAMMPS dump file.
    encoding : str
        Text encoding used inside the dump (default: 'utf-8').
    comm : MPI.Comm
        The MPI communicator (default: MPI.COMM_WORLD).
    parallelization : int, optional (default=0)
        `indexed_bzip2` parallelization setting. 0 = use all cores (recommended);
        1 = serial; N > 1 = use N threads.

    Yields
    ------
    dict
        A dictionary containing the timestep data with keys:
        'time' (optional), 'timestep', 'natoms', 'boundary', 'xlo', 'xhi',
        'ylo', 'yhi', 'zlo', 'zhi', and 'atoms' (as a numpy structured array).
    """

    file_path: Path
    encoding: str = "utf-8"
    __file: "_LineReader" = field(default=None, init=False)
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    parallelization: int = 0

    __raw: Any = field(default=None, init=False, repr=False)
    __rank: int = field(init=False, repr=False)
    __size: int = field(init=False, repr=False)
    __comm_tag: int = field(
        default_factory=MPITagAllocator.get_tag, init=False, repr=False
    )
    __nx: int = field(init=False, repr=False)
    __ny: int = field(init=False, repr=False)
    __nz: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.__rank = self.comm.Get_rank()
        self.__size = self.comm.Get_size()
        self.__nx, self.__ny, self.__nz = mpi_subdomains_decomposition(self.__size)
        ix = self.__rank % self.__nx
        iy = (self.__rank // self.__nx) % self.__ny
        iz = self.__rank // (self.__nx * self.__ny)
        self._domain_index = (ix, iy, iz)

        self.__raw, self.__file = None, None
        if self.__rank == 0:
            try:
                # Open an indexed, multi-threaded bz2 reader. It yields *bytes*.
                self.__raw = ibz2.open(
                    str(self.file_path), parallelization=self.parallelization
                )
                # Wrap in a lightweight line reader that decodes to text on the fly.
                # We implement .readline() using a small buffer to avoid relying on
                # TextIOWrapper specifics.
                self.__file = _LineReader(self.__raw, encoding=self.encoding)
            except Exception:
                self._handle_exception()

    def __enter__(self) -> "BZIP2LAMMPSReaderMPI":
        return self

    def __del__(self) -> None:
        if self.__rank == 0:
            try:
                if self.__file is not None:
                    self.__file.close()
            except Exception:
                pass

    def __exit__(
        self,
        exc_type: None | type[BaseException] = None,
        exc_value: None | BaseException = None,
        exc_traceback: None | TracebackType = None,
    ) -> bool:
        if self.__rank == 0 and self.__file is not None:
            self.__file.close()
        return False

    def _get_dtype(self) -> Tuple[list[str], list[Type[int | float]], np.dtype]:
        items = self.__file.readline().split()[2:]
        types = [
            np.int64 if it in ("id", "type", "element", "size") else np.float64
            for it in items
        ]
        return items, types, np.dtype(list(zip(items, types)))

    def _process_header(self) -> Dict[str, Any]:
        data: Dict[str, Any] = defaultdict(None)
        line = self.__file.readline()
        if not line:
            return {}
        if line.strip() == "ITEM: TIME":
            data["time"] = float(self.__file.readline())
            self.__file.readline()  # "ITEM: TIMESTEP"
        # If no TIME item, the line we just read should have been "ITEM: TIMESTEP"
        if "timestep" not in data:
            data["timestep"] = int(self.__file.readline())
        else:
            data["timestep"] = int(self.__file.readline())
        self.__file.readline()  # "ITEM: NUMBER OF ATOMS"
        data["natoms"] = int(self.__file.readline())
        # BOX BOUNDS
        data["boundary"] = self.__file.readline().split()[3:]
        bounds = [self.__file.readline().split() for _ in range(3)]
        data["xlo"], data["xhi"] = map(float, bounds[0][:2])
        data["ylo"], data["yhi"] = map(float, bounds[1][:2])
        data["zlo"], data["zhi"] = map(float, bounds[2][:2])
        return data

    @mpi_safe_method
    def __iter__(self) -> Generator[Tuple[Dict[str, Any], npt.NDArray], None, None]:
        while True:
            # header broadcast
            data = self.comm.bcast(
                self._process_header() if self.__rank == 0 else None, root=0
            )
            if not data:
                break

            # dtype broadcast
            items, types, dtype = self.comm.bcast(
                self._get_dtype() if self.__rank == 0 else (None, None, None), root=0
            )
            data.update({"items": items, "types": types, "dtype": dtype})

            # calculate raw line counts
            natoms = int(data["natoms"])
            counts = [
                (natoms // self.__size) + (1 if i < (natoms % self.__size) else 0)
                for i in range(self.__size)
            ]

            # distribute chunks
            if self.__rank == 0:
                chunks = []
                for cnt in counts:
                    chunk = [self.__file.readline().split() for _ in range(cnt)]
                    chunks.append(chunk)
                raw = chunks[0]
                for r in range(1, self.__size):
                    self.comm.send(chunks[r], dest=r, tag=self.__comm_tag)
            else:
                raw = self.comm.recv(source=0, tag=self.__comm_tag)

            # build structured array
            arr = np.empty(len(raw), dtype=dtype)
            for i, fields in enumerate(raw):
                for j, key in enumerate(items):
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
            # attach atoms
            data["subdomain_atoms"] = arr
            data["subdomain_natoms"] = len(arr)

            yield data

        if self.__rank == 0 and self.__file:
            self.__file.close()

    @mpi_safe_method
    def close(self) -> None:
        """Closes the file associated with this reader."""
        if self.__rank == 0 and self.__file is not None:
            try:
                self.__file.close()
            except Exception:
                pass


class _LineReader:
    """Helper to read decoded text lines from a bytes-only stream.

    This avoids depending on TextIOWrapper behavior for custom backends.
    """

    __slots__ = ("__raw", "__decoder", "__buffer", "__closed")

    def __init__(self, raw, encoding: str = "utf-8"):

        self.__raw = raw
        self.__decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        self.__buffer = ""  # decoded text buffer
        self.__closed = False

    def readline(self) -> str:
        """Read one line, including the trailing newline character."""
        # Try to return a line from the decoded buffer first
        while True:
            nl = self.__buffer.find("\n")
            if nl != -1:
                line = self.__buffer[: nl + 1]
                self.__buffer = self.__buffer[nl + 1 :]
                return line
            # Need more bytes
            chunk = self.__raw.read(1 << 16)  # 64 KiB
            if not chunk:
                # EOF: flush remainder (if any)
                if self.__buffer:
                    line, self.__buffer = self.__buffer, ""
                    return line
                return ""
            self.__buffer += self.__decoder.decode(chunk, final=False)

    def read(self, n: int | None = None) -> str:
        """Read up to n characters, or all remaining if n is None."""
        if n is None:
            # Read all remaining
            chunks = [self.__buffer]
            self.__buffer = ""
            while True:
                b = self.__raw.read(1 << 16)
                if not b:
                    break
                chunks.append(self.__decoder.decode(b, final=False))
            return "".join(chunks)
        # Read up to n chars
        out = []
        while n > 0:
            if self.__buffer:
                take = min(len(self.__buffer), n)
                out.append(self.__buffer[:take])
                self.__buffer = self.__buffer[take:]
                n -= take
                if n == 0:
                    break
            b = self.__raw.read(min(1 << 16, n))
            if not b:
                break
            out.append(self.__decoder.decode(b, final=False))
        return "".join(out)

    def close(self) -> None:
        """Closes the underlying raw stream."""
        if not self.__closed:
            try:
                self.__raw.close()
            finally:
                self.__closed = True
