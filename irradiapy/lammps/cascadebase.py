"""Base class for running LAMMPS simulations of collisional cascades."""

# pylint: disable=line-too-long, no-name-in-module

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
from lammps import lammps
from mpi4py import MPI

from irradiapy import materials, utils
from irradiapy.lammps.commands.command import Command


@dataclass(kw_only=True)
class CascadeBase(ABC):
    """Base class for running LAMMPS simulations of collisional cascades. Not to be used directly.

    Warning
    -------
    In order to use child classes:
    - Do not use "write_data" or "write_restart", these are created automatically.
    - Include in the "thermo" output: step, dt, time.
    - In the dump command, place only the file name, the path will be added automatically.

    Parameters
    ----------
    comm : MPI.Comm (default=MPI.COMM_WORLD)
        The MPI communicator.
    dir_parent : Path
        The parent directory for the simulation.
    nsims : int
        The number of simulations to run.
    cmds_preamble : list[Command]
        The preamble commands to run before the simulation. These commands are run before any
        thermalisation, run, and rerun.
    cmds_therma : list[Command] (default=None)
        The commands to run for the thermalization step.
    cmds_cascade : list[Command] (default=None)
        The commands to run for the cascade step.
    cmds_rerun : list[Command] (default=None)
        The commands to run for the rerun step.
    atomic_symbols : list[str]
        The atomic symbols of the atoms in the simulation in the same order as the atom types in LAMMPS.
    seed : int (default=1)
        The random seed for the simulation.
    finalize : bool (default=True)
        If True, after all simulations, the LAMMPS instance is finalized.
    skip : Union[str, list[int]] (default="")
        A string of comma-separated integers and ranges to skip simulations.
        For example, "1, 2, 3-5, 7-9" will skip simulations 1, 2, 3, 4, 5, 7, 8, and 9.
    eph_c : float (default=None)
        Only for EPH calculations. The EPH electron heat capacity in eV/K.
    eph_k : float (default=None)
        Only for EPH calculations. The EPH electron thermal conductivity in W/(m*K).
    eph_temperature : float (default=None)
        Only for EPH calculations. The EPH electron temperature in K.
    eph_nxyz : tuple[int, int, int] (default=None)
        Only for EPH calculations. The number of grid voxels in the x, y, and z directions.
    eph_extent : tuple[float, float, float, float, float, float] (default=None)
        Only for EPH calculations. The extent of the grid in the x, y, and z directions.
    """

    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)
    rank: int = field(init=False)
    path_log_cwd: Path = field(
        default_factory=lambda: Path.cwd() / "log.lammps", init=False
    )

    dir_parent: Path
    nsims: int
    cmds_preamble: list[Command]
    cmds_therma: Optional[list[Command]] = None
    atomic_symbols: list[str]
    cmds_cascade: Optional[list[Command]] = None
    cmds_rerun: Optional[list[Command]] = None
    seed: int = 1
    finalize: bool = True
    skip: Union[str, list[int]] = field(default_factory="")

    eph_c: Optional[float] = field(init=False, default=None)
    eph_k: Optional[float] = field(init=False, default=None)
    eph_temperature: Optional[float] = field(init=False, default=None)
    eph_nxyz: Optional[tuple[int, int, int]] = None
    eph_extent: Optional[tuple[float, float, float, float, float, float]] = None

    _rng: Optional[np.random.Generator] = field(init=False, default=None)
    _eph: bool = field(init=False, default=False)

    @abstractmethod
    def run(self) -> None:
        """Runs the cascade simulation."""

    def __post_init__(self) -> None:
        self.rank = self.comm.Get_rank()
        self.skip = self.__parse_ranges(self.skip)
        if all([self.eph_nxyz, self.eph_extent]):
            self._eph = True
        self._rng = None
        if self.rank == 0:
            self._rng = np.random.default_rng(self.seed)

    def __parse_ranges(self, s: str) -> list[int]:
        """Transform a string of comma-separated integers and ranges into a sorted list.

        - Single numbers must be >0.
        - Ranges “a-b” expand to all integers from a through b (inclusive),
        and both a and b must be >0.
        - Zero is forbidden.
        - Duplicates are removed.
        - The final list is sorted in ascending order.

        Parameters
        ----------
        s : str
            The string to parse, e.g. "1, 2, 3-5, 7-9".

        Raises
        ------
            ValueError: on any malformed token, zero, or inverted range.
        """
        if not s:
            return []
        nums = set()
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError as exc:
                    raise ValueError(f"Invalid range token: '{part}'") from exc
                if start <= 0 or end <= 0:
                    raise ValueError(f"Zero is forbidden in ranges: '{part}'")
                if start > end:
                    raise ValueError(f"Range start greater than end: '{part}'")
                nums.update(range(start, end + 1))
            else:
                try:
                    val = int(part)
                except ValueError as exc:
                    raise ValueError(f"Invalid number token: '{part}'") from exc
                if val == 0:
                    raise ValueError("Zero is forbidden")
                if val < 0:
                    raise ValueError("Negative numbers are not allowed")
                nums.add(val)
        return sorted(nums)

    # region Commands

    def exec_cmds(self, lmp: lammps, cmds: list[Command]) -> None:
        """Executes a list of commands.

        Parameters
        ----------
        cmds : list[Command]
            The commands to execute
        """
        for cmd in cmds:
            lmp.command(cmd.command())

    # endregion

    # region MPI

    def _broadcast_variables(self, *variables) -> list:
        """Broadcasts variables.

        Parameters
        ----------
        variables : tuple
            The variables to broadcast.

        Returns
        -------
        list
            The broadcasted variables.
        """
        return utils.mpi.broadcast_variables(0, self.comm, *variables)

    # endregion

    # region Physics
    def _calculate_atoms_speeds(
        self, kes: np.ndarray, atomic_numbers: np.ndarray
    ) -> np.ndarray:
        """Calculates the velocity of atoms in metal units (angstrom/ps).

        Parameters
        ----------
        kes : np.ndarray
            The kinetic energy of the atoms in eV.
        atomic_numbers : np.ndarray
            The atomic numbers of the atoms.

        Returns
        -------
        np.ndarray
            The speeds of the atoms in angstrom/ps.
        """
        e_charge = 1.60218e-19  # elementary charge, C
        amu = 1.66054e-27  # atomic mass unit, kg
        mass = np.array(
            [
                materials.MASS_NUMBER_BY_ATOMIC_NUMBER[int(atomic_number)]
                for atomic_number in atomic_numbers
            ]
        )
        speeds = (2 * kes * e_charge / (mass * amu)) ** 0.5 * 1e-2
        return speeds

    def types_to_atomic_numbers(self, types: np.ndarray) -> np.ndarray:
        """Converts atom types to atomic numbers.

        Parameters
        ----------
        types : np.ndarray
            The atom types.

        Returns
        -------
        np.ndarray
            The atomic numbers.
        """
        symbols = [self.atomic_symbols[t - 1] for t in types]
        atomic_numbers = np.array(
            [materials.ATOMIC_NUMBER_BY_SYMBOL[symbol] for symbol in symbols],
            dtype=int,
        )
        return atomic_numbers

    # endregion

    # region EPH

    def _generate_egrid(self) -> Path:
        """Generates the EPH electron grid."""
        path_egrid = self.dir_parent / "egrid0"
        with open(path_egrid, "w", encoding="utf-8") as file:
            file.write("# https://github.com/LLNL/USER-EPH\n")
            file.write("# at commit a836eb2f3dd8648160d59c64a6c4a69d82c4a6b9\n")
            file.write("# i j k T_e source rho_e C_e kappa_e updateTemp ReadFile\n")
            file.write(f"{self.eph_nxyz[0]} {self.eph_nxyz[1]} {self.eph_nxyz[2]} 10\n")
            file.write(f"{self.eph_extent[0]} {self.eph_extent[1]}\n")
            file.write(f"{self.eph_extent[2]} {self.eph_extent[3]}\n")
            file.write(f"{self.eph_extent[4]} {self.eph_extent[5]}\n")
            file.write("NULL\n")
            for i in range(self.eph_nxyz[0]):
                for j in range(self.eph_nxyz[1]):
                    for k in range(self.eph_nxyz[2]):
                        file.write(
                            f"{i} {j} {k} {self.eph_temperature} 0.0 1.0 {self.eph_c} {self.eph_k} 1 0\n"
                        )
        return path_egrid

    # region Directories
    @staticmethod
    def _is_dir_empty(directory: Path) -> bool:
        """Returns True if the given directory is empty.

        Parameters
        ----------
        directory : Path
            The directory.

        Returns
        -------
        bool
            True if the directory is empty.
        """
        return not any(directory.iterdir())

    def _rm_simulations(self, not_exists_ok: bool = True) -> None:
        """Removes the simulation directory.

        Parameters
        ----------
        not_exists_ok : bool, optional (default=True)
            If True, does not raise an exception if the directory does not
            exist.

        Raises
        ------
        ValueError
            If the directory is a sensitive directory: "/", "/home", "/root",
            "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/media",
            "/mnt", "/opt", "/proc", "/run", "/sbin", "/srv", "/sys", "/usr",
            "/var", "c:\\", "c:\\windows", "c:\\program files",
            "c:\\program files (x86)", "c:\\users".
        FileNotFoundError
            If the directory does not exist.
        """
        dir_path = Path(self.dir_parent).resolve()
        # Prevent deletion of sensitive directories
        sensitive_dirs = {
            "/",
            "/home",
            "/root",
            "/bin",
            "/boot",
            "/dev",
            "/etc",
            "/lib",
            "/lib64",
            "/media",
            "/mnt",
            "/opt",
            "/proc",
            "/run",
            "/sbin",
            "/srv",
            "/sys",
            "/usr",
            "/var",
            "c:\\",
            "c:\\windows",
            "c:\\program files",
            "c:\\program files (x86)",
            "c:\\users",
        }
        if dir_path == dir_path.anchor or dir_path.as_posix().lower() in sensitive_dirs:
            raise ValueError("Cannot delete sensitive directories.")
        # Check if exists
        if not dir_path.exists():
            if not_exists_ok:
                return
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        # Remove
        shutil.rmtree(dir_path, ignore_errors=True)

    def get_simulation_dirs(self) -> list[Path]:
        """Returns the simulation directories.

        Returns
        -------
        list[Path]
            The simulation directories.
        """
        simulation_dirs = [
            dir for dir in self.dir_parent.glob("simulation_*") if dir.is_dir()
        ]
        return simulation_dirs

    # region Reading results

    def get_pka_properties(self, path_log: Path) -> dict[str, int | float | np.ndarray]:
        """Returns the properties of the PKA from a LAMMPS log file.

        Parameters
        ----------
        path_log : Path
            The path to the log file.

        Returns
        -------
        dict[str, int | float | np.ndarray]
            The PKA properties.
        """
        with open(path_log, "r", encoding="utf-8") as file:
            data = {}
            for line in file:
                if line.startswith("Ion boost:"):
                    break
            file.readline()
            data["id"] = int(file.readline().split()[1])
            file.readline()
            data["element"] = int(file.readline().split()[1])
            file.readline()
            data["pos"] = np.array(
                list(map(float, file.readline().split("(")[1].split(")")[0].split(",")))
            )
            file.readline()
            data["energy"] = float(file.readline().split()[1])
            file.readline()
            data["speed"] = float(file.readline().split()[1])
            file.readline()
            data["velocity"] = np.array(
                list(map(float, file.readline().split("(")[1].split(")")[0].split(",")))
            )
            file.readline()
            data["theta"] = float(file.readline().split()[3])
            file.readline()
            data["phi"] = float(file.readline().split()[3])
        return data

    def get_pkas_properties(
        self,
    ) -> Generator[dict[str, int | float | np.ndarray], None, None]:
        """Returns the properties of the PKAs from a set of LAMMPS log files.

        Yields
        ------
        dict[str, int | float | np.ndarray]
            The PKA properties.
        """
        for dir_sim in self.get_simulation_dirs():
            idx = dir_sim.name.split("_")[-1]
            path_log = dir_sim / f"simulation_{idx}.log"
            structured = self.get_pka_properties(path_log)
            yield structured
