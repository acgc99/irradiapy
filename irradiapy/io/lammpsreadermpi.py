"""This module contains the `LAMMPSReaderMPI` class."""

# pylint: disable=no-name-in-module

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, TextIO, Tuple, Type, Union

import numpy as np
from mpi4py import MPI

from irradiapy.mpi_utils import MPITagAllocator, mpi_safe_method


def _factorize_to_3d(n: int) -> Tuple[int, int, int]:
    best = (1, 1, n)
    best_score = float("inf")
    for nx in range(1, int(n ** (1 / 3)) + 2):
        if n % nx:
            continue
        m = n // nx
        for ny in range(1, int(m**0.5) + 2):
            if m % ny:
                continue
            nz = m // ny
            dims = [nx, ny, nz]
            score = max(dims) / min(dims)
            if score < best_score:
                best_score = score
                best = (nx, ny, nz)
    return best


@dataclass
class LAMMPSReaderMPI:
    """A class to read data from a LAMMPS dump  file in parallel using MPI.

    Note
    ----
    Assumed orthogonal simulation box.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    debug : bool
        If True, prints debug information (default: False).
    comm : MPI.Comm
        The MPI communicator (default: MPI.COMM_WORLD).
    """

    file_path: Path
    debug: bool = False
    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    __rank: int = field(init=False)
    __commsize: int = field(init=False)
    __comm_tag: int = field(default_factory=MPITagAllocator.get_tag, init=False)
    __nx: int = field(init=False)
    __ny: int = field(init=False)
    __nz: int = field(init=False)

    def __post_init__(self):
        self.__rank = self.comm.Get_rank()
        self.__commsize = self.comm.Get_size()
        self.__nx, self.__ny, self.__nz = _factorize_to_3d(self.__commsize)
        ix = self.__rank % self.__nx
        iy = (self.__rank // self.__nx) % self.__ny
        iz = self.__rank // (self.__nx * self.__ny)
        self._domain_index = (ix, iy, iz)
        if self.debug and self.__rank == 0:
            print(
                f"[Debug] Domain grid: nx={self.__nx}, ny={self.__ny}, nz={self.__nz}"
            )

    def __get_dtype(
        self, file: TextIO
    ) -> Tuple[list[str], list[Type[Union[int, float]]], np.dtype]:
        items = file.readline().split()[2:]
        types = [
            np.int32 if it in ("id", "type", "element", "size") else np.float64
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
        if self.debug:
            print(f"[Rank {self.__rank}] Read header line: {line.strip()}")
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
            if self.debug:
                print(f"[Rank {self.rank}] Bound line: {b}")
        data["xlo"], data["xhi"] = map(float, bounds[0][:2])
        data["ylo"], data["yhi"] = map(float, bounds[1][:2])
        data["zlo"], data["zhi"] = map(float, bounds[2][:2])
        return data

    @mpi_safe_method
    def __iter__(self) -> Generator[Tuple[Dict[str, Any], np.ndarray], None, None]:
        file = open(self.file_path, encoding="utf-8") if self.__rank == 0 else None
        iteration = 0
        while True:
            if self.debug:
                print(f"[Rank {self.__rank}] Starting iteration {iteration}")
            # header broadcast
            data = self.comm.bcast(
                self.__process_header(file) if self.__rank == 0 else None, root=0
            )
            if data is None or not data:
                if self.debug:
                    print(f"[Rank {self.__rank}] No more data, breaking loop")
                break
            # dtype broadcast
            items, types, dtype = self.comm.bcast(
                self.__get_dtype(file) if self.__rank == 0 else (None, None, None),
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
            if self.debug:
                print(f"[Rank {self.__rank}] Raw counts per rank: {counts}")

            # distribute chunks
            if self.__rank == 0:
                chunks = []
                for cnt in counts:
                    chunk = [file.readline().split() for _ in range(cnt)]
                    chunks.append(chunk)
                raw = chunks[0]
                for r in range(1, self.__commsize):
                    if self.debug:
                        print(
                            f"[Rank 0] Sending chunk of size {len(chunks[r])} to rank {r}"
                        )
                    self.comm.send(chunks[r], dest=r, tag=self.__comm_tag)
            else:
                if self.debug:
                    print(f"[Rank {self.__rank}] Waiting to receive raw chunk")
                raw = self.comm.recv(source=0, tag=self.__comm_tag)
                if self.debug:
                    print(f"[Rank {self.__rank}] Received raw chunk of size {len(raw)}")

            # build structured array
            arr = np.empty(len(raw), dtype=dtype)
            for i, fields in enumerate(raw):
                for j, key in enumerate(items):
                    if key in ("xs", "ys", "zs"):
                        continue
                    arr[key][i] = types[j](fields[j])

            iteration += 1
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
            data["atoms_rank"] = arr
            data["natoms_rank"] = len(arr)
            yield data

        if file:
            file.close()
