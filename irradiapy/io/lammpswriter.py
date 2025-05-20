"""This module contains the `LAMMPSWriter` class."""

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional


@dataclass
class LAMMPSWriter:
    """A class to write data like a LAMMPS dump file.

    Attributes
    ----------
    file_path : Path
        The path to the LAMMPS dump file.
    """

    file_path: Path
    mode: str
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        """Opens the file associated with this writer."""
        self.file = open(self.file_path, self.mode, encoding=self.encoding)

    def __enter__(self) -> "LAMMPSWriter":
        """Enters the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> bool:
        """Exits the context manager."""
        self.file.close()
        return False

    def close(self) -> None:
        """Closes the file associated with this writer."""
        self.file.close()

    def __del__(self) -> None:
        """Closes the file associated with this writer."""
        self.file.close()

    def save(
        self,
        timestep: int,
        boundary: tuple[str, str, str],
        dimsx: tuple[float, float],
        dimsy: tuple[float, float],
        dimsz: tuple[float, float],
        time: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Writes the data to the file.

        Note
        ----
        Assumes orthogonal simulation box.

        Parameters
        ----------
        timestep : int
            The timestep of the simulation.
        boundary : tuple[str, str, str]
            The type of boundary conditions of the simulation.
        dimsx : tuple[float, float]
            The maximum extents of the simulation box in the x-dimension,
        dimsy : tuple[float, float]
            The maximum extents of the simulation box in the y-dimension.
        dimsz : tuple[float, float]
            The maximum extents of the simulation box in the z-dimension.
        time : float, optional
            The time of the simulation.
        **kwargs
            The data to be written to the file.
        """
        field_names = list(kwargs.keys())
        natoms = len(kwargs[field_names[0]])
        if time is not None:
            self.file.write(f"ITEM: TIME\n{time}\n")
        self.file.write(f"ITEM: TIMESTEP\n{timestep}\n")
        self.file.write(f"ITEM: NUMBER OF ATOMS\n{natoms}\n")
        self.file.write(
            f"ITEM: BOX BOUNDS {boundary[0]*2} {boundary[1]*2} {boundary[2]*2}\n"
        )
        self.file.write(f"{dimsx[0]} {dimsx[1]}\n")
        self.file.write(f"{dimsy[0]} {dimsy[1]}\n")
        self.file.write(f"{dimsz[0]} {dimsz[1]}\n")
        self.file.write(f"ITEM: ATOMS {' '.join(field_names)}\n")
        for i in range(natoms):
            row = [kwargs[field][i] for field in field_names]
            self.file.write(" ".join(map(str, row)) + "\n")
