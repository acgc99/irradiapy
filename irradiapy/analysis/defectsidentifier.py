"""This module provides a class to find and analyze defects in crystalline structures."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import structured_to_unstructured as str2unstr
from scipy.spatial.transform import Rotation as R

import irradiapy as irpy


@dataclass
class DefectsIdentifier:
    """Class to identify  defects in crystalline structures.

    This class provides methods to identify point defects (vacancies and interstitials) in a
    body-centered cubic (bcc) lattice based on atomic positions from simulation data. It supports
    optional rescaling and recentering of defect positions, and can align the system with a
    specified primary knock-on atom (PKA) direction.

    Parameters
    ----------
    lattice : str
        Lattice type. Currently only "bcc" is supported.
    a0 : float
        Lattice parameter, in angstroms.
    debug : bool, optional
        If True, enables debug mode for additional output. Default is False.
    """

    lattice: str
    a0: float
    debug: bool = False
    __perx: bool = field(init=False)
    __pery: bool = field(init=False)
    __perz: bool = field(init=False)
    __nxhi: int = field(init=False)
    __nyhi: int = field(init=False)
    __nzhi: int = field(init=False)
    __nxlo: int = field(init=False)
    __nylo: int = field(init=False)
    __nzlo: int = field(init=False)
    __nx: int = field(init=False)
    __ny: int = field(init=False)
    __nz: int = field(init=False)
    __sub_count: int = 2  # Number of atoms per primitive unit cell (bcc).

    def __post_init__(self) -> None:
        if self.lattice == "bcc":
            self.__sub_count = 2
            self.__pos_ucell = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                ],
                dtype=np.float64,
            )
            self.__idx_ucell = np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.int32,
            )
            # Scale positions by a0
            self.__pos_ucell *= self.a0
        else:
            raise ValueError("Only bcc lattice is supported.")

    def __rescale_translate_rotate(
        self,
        atoms: npt.NDArray,
        a1: Union[float, np.number],
        pos_pka: npt.NDArray[np.float64],
        theta_pka: float,
        phi_pka: float,
    ) -> None:
        """Rescales, translates, and rotates the positions of atoms.

        Parameters
        ----------
        atoms : np.ndarray
            Structured array of atomic positions (fields: x, y, z).
        a1 : float
            Final lattice parameter to rescale positions.
        pos_pka : np.ndarray
            Position vector of the PKA for translation.
        theta_pka : float
            Polar angle (in radians) for the PKA direction (for rotation).
        phi_pka : float
            Azimuthal angle (in radians) for the PKA direction (for rotation).
        """
        # Translate
        atoms["x"] -= pos_pka[0]
        atoms["y"] -= pos_pka[1]
        atoms["z"] -= pos_pka[2]

        # Rotation matrix, align with PKA initial direction
        xaxis = np.array([1.0, 0.0, 0.0])
        pka_dir = np.array(
            [
                np.sin(theta_pka) * np.cos(phi_pka),
                np.sin(theta_pka) * np.sin(phi_pka),
                np.cos(theta_pka),
            ]
        )
        transform = R.align_vectors([xaxis], [pka_dir])[0].as_matrix()

        # Scaling matrix
        scaling = a1 / self.a0
        scaling_matrix = (
            np.diag([scaling] * 3) if isinstance(scaling, float) else np.diag(scaling)
        )

        # Combined scale + rotation
        transform = transform @ scaling_matrix

        # Apply transformations
        pos = str2unstr(atoms[["x", "y", "z"]])
        pos = transform.apply(pos)
        atoms["x"] = pos[:, 0]
        atoms["y"] = pos[:, 1]
        atoms["z"] = pos[:, 2]

    def __apply_boundary_conditions(self, idx_atoms: np.ndarray) -> None:
        """Applies periodic boundary conditions to atomic index coordinates.

        Only one unit cell around the simulation box is considered for periodic boundary conditions.

        Parameters
        ----------
        idx_atoms : np.ndarray
            Array of atomic positions with index coordinates (shape: [N, 4]).
        """
        if self.__perx:
            idx_atoms[:, 0][idx_atoms[:, 0] == self.__nxhi] = self.__nxlo
            idx_atoms[:, 0][idx_atoms[:, 0] == self.__nxlo - 1] = self.__nxhi - 1
        if self.__pery:
            idx_atoms[:, 1][idx_atoms[:, 1] == self.__nyhi] = self.__nylo
            idx_atoms[:, 1][idx_atoms[:, 1] == self.__nylo - 1] = self.__nyhi - 1
        if self.__perz:
            idx_atoms[:, 2][idx_atoms[:, 2] == self.__nzhi] = self.__nzlo
            idx_atoms[:, 2][idx_atoms[:, 2] == self.__nzlo - 1] = self.__nzhi - 1

    def __site_id_to_indices(self, i: int) -> tuple[int, int, int, int]:
        """Decodes a site ID into its corresponding lattice indices.

        Parameters
        ----------
        i : int
            Site ID to decode.

        Returns
        -------
        tuple of int
            (ix, iy, iz, ia) indices corresponding to the site ID.
        """
        ia = i % self.__sub_count
        tmp = i // self.__sub_count
        iz = tmp % self.__nz
        tmp //= self.__nz
        iy = tmp % self.__ny
        ix = tmp // self.__ny
        ix += self.__nxlo
        iy += self.__nylo
        iz += self.__nzlo
        return ix, iy, iz, ia

    def __site_id_to_cartesian(self, i: int) -> npt.NDArray[np.float64]:
        """Converts a site ID to its corresponding cartesian coordinates.

        Parameters
        ----------
        i : int
            Site ID to decode.

        Returns
        -------
        np.ndarray
            Cartesian coordinates (x, y, z) corresponding to the site ID.
        """
        ix, iy, iz, ia = self.__site_id_to_indices(i)
        x = (ix + 0.5 * ia) * self.a0
        y = (iy + 0.5 * ia) * self.a0
        z = (iz + 0.5 * ia) * self.a0
        return np.array([x, y, z], dtype=float)

    def __defect_identification(
        self,
        data_atoms: np.ndarray,
        idx_atoms: np.ndarray,
    ) -> np.ndarray:
        """Identifies defects based on the assignments.

        Parameters
        ----------
        data_atoms : np.ndarray
            Structured array of atomic coordinates and element types.
        idx_atoms : np.ndarray
            Array of atomic index coordinates (shape: [N, 4]).

        Returns
        -------
        np.ndarray
            Array of defects (structured array with fields: type, x, y, z).
        """
        # Build unique site IDs from idx_atoms
        ix = idx_atoms[:, 0] - self.__nxlo
        iy = idx_atoms[:, 1] - self.__nylo
        iz = idx_atoms[:, 2] - self.__nzlo
        s = idx_atoms[:, 3]
        nsites = self.__nx * self.__ny * self.__nz * self.__sub_count
        id_sites = ((ix * self.__ny + iy) * self.__nz + iz) * self.__sub_count + s

        # Sort and group
        sort_idx = np.argsort(id_sites)
        site_sorted = id_sites[sort_idx]
        split_points = np.cumsum(np.bincount(site_sorted, minlength=nsites))[:-1]
        grouped = np.split(sort_idx, split_points)

        defects = np.empty(0, dtype=irpy.dtypes.defect)
        for i, grp in enumerate(grouped):
            if len(grp) == 0:
                # Vacancy
                vac = np.array(
                    [(0, *self.__site_id_to_cartesian(i))], dtype=irpy.dtypes.defect
                )
                defects = np.concatenate((defects, vac))
            elif len(grp) > 1:
                # Interstitials
                xyz = self.__site_id_to_cartesian(i)
                coords = str2unstr(data_atoms["atoms"][["x", "y", "z"]][grp])
                dist2 = np.sum(np.square(coords - xyz), axis=1)
                keep_idx = np.argmin(dist2)
                inters_idx = np.ones(len(grp), dtype=bool)
                inters_idx[keep_idx] = False
                if np.any(inters_idx):
                    count = np.count_nonzero(inters_idx)
                    inters = np.zeros(count, dtype=irpy.dtypes.defect)
                    inters["type"] = data_atoms["atoms"]["element"][grp][inters_idx]
                    inters["x"] = coords[inters_idx, 0]
                    inters["y"] = coords[inters_idx, 1]
                    inters["z"] = coords[inters_idx, 2]
                    defects = np.concatenate((defects, inters))
        return defects

    def identify(
        self,
        data_atoms: defaultdict,
        a1: Optional[float] = None,
        pos_pka: Optional[npt.NDArray[np.float64]] = None,
        theta_pka: Optional[float] = None,
        phi_pka: Optional[float] = None,
        transform: Optional[bool] = False,
    ) -> np.ndarray:
        """Identify defects in the crystalline structure based on atomic positions.

        Parameters
        ----------
        data_atoms : defaultdict
            Dictionary containing simulation data as given by the LAMMPSReader and similar readers.
            Must include keys: 'atoms', 'natoms', 'boundary', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo',
            'zhi', 'timestep'.
        a1 : float, optional
            Final lattice parameter. If provided, defect positions are rescaled to this value
            (independently of the `transform` value).
        pos_pka : np.ndarray, optional
            Position vector of the PKA. If provided with theta_pka and phi_pka, defects are
            recentered and aligned.
        theta_pka : float, optional
            Polar angle (in radians) for the PKA direction.
        phi_pka : float, optional
            Azimuthal angle (in radians) for the PKA direction.
        transform : bool, optional
            If True, defects are recentered and aligned with the PKA direction (if provided). If
            True but no PKA parameters are provided, defects are recentered based on their
            average position. Note that the box boundaries are not modified for visualization
            purposes, only the atomic positions are transformed.

        Returns
        -------
        data_defects : dict
            Dictionary containing the defects found in the structure. Keys are the same as in
            `data_atoms`, but the 'atoms' key contains only defects and
            'natoms' reflects the number of defects found.
        """
        self.__nxlo = round(data_atoms["xlo"] / self.a0)
        self.__nxhi = round(data_atoms["xhi"] / self.a0)
        self.__nylo = round(data_atoms["ylo"] / self.a0)
        self.__nyhi = round(data_atoms["yhi"] / self.a0)
        self.__nzlo = round(data_atoms["zlo"] / self.a0)
        self.__nzhi = round(data_atoms["zhi"] / self.a0)
        self.__nx = self.__nxhi - self.__nxlo
        self.__ny = self.__nyhi - self.__nylo
        self.__nz = self.__nzhi - self.__nzlo
        self.__perx = data_atoms["boundary"][0] == "pp"
        self.__pery = data_atoms["boundary"][1] == "pp"
        self.__perz = data_atoms["boundary"][2] == "pp"

        idx_atoms = np.zeros((data_atoms["natoms"], 4), dtype=np.int32)
        mod_atoms = np.zeros((data_atoms["natoms"], 3), dtype=np.float64)
        idx_atoms[:, 0], mod_atoms[:, 0] = np.divmod(data_atoms["atoms"]["x"], self.a0)
        idx_atoms[:, 1], mod_atoms[:, 1] = np.divmod(data_atoms["atoms"]["y"], self.a0)
        idx_atoms[:, 2], mod_atoms[:, 2] = np.divmod(data_atoms["atoms"]["z"], self.a0)
        # idx_atoms[:, 3] is initially zero

        # Compute closest unit-cell positions
        cell_xyz = self.__pos_ucell
        dist = np.sum((mod_atoms[:, None, :] - cell_xyz[None, :, :]) ** 2, axis=2)
        closest_cell_indices = np.argmin(dist, axis=1)
        idx_atoms[:, 0] += self.__idx_ucell[closest_cell_indices, 0]
        idx_atoms[:, 1] += self.__idx_ucell[closest_cell_indices, 1]
        idx_atoms[:, 2] += self.__idx_ucell[closest_cell_indices, 2]
        idx_atoms[:, 3] += self.__idx_ucell[closest_cell_indices, 3]

        # Apply boundaries on idx_atoms
        self.__apply_boundary_conditions(idx_atoms)

        # Identify defects
        defects = self.__defect_identification(data_atoms, idx_atoms)

        # Recenter / scaling if requested
        if len(defects):
            if transform:
                if a1 is None:
                    a1 = self.a0
                if (
                    pos_pka is not None
                    and theta_pka is not None
                    and phi_pka is not None
                ):
                    self.__rescale_translate_rotate(
                        defects, a1, pos_pka, theta_pka, phi_pka
                    )
                else:
                    defects[["x", "y", "z"]] -= np.mean(
                        defects[["x", "y", "z"]], axis=1
                    )
                    defects[["x", "y", "z"]] *= a1
            elif a1 is not None:
                defects[["x", "y", "z"]] *= a1

        data_defects = defaultdict(None)
        data_defects["time"] = data_atoms["time"]
        data_defects["timestep"] = data_atoms["timestep"]
        data_defects["natoms"] = len(defects)
        data_defects["boundary"] = data_atoms["boundary"]
        data_defects["xlo"] = data_atoms["xlo"]
        data_defects["xhi"] = data_atoms["xhi"]
        data_defects["ylo"] = data_atoms["ylo"]
        data_defects["yhi"] = data_atoms["yhi"]
        data_defects["zlo"] = data_atoms["zlo"]
        data_defects["zhi"] = data_atoms["zhi"]
        data_defects["atoms"] = defects

        nvacs = np.count_nonzero(defects["type"] == 0)
        nsias = len(defects) - nvacs
        if self.debug:
            print(f"Number of interstitials: {nsias}")
            print(f"Number of vacancies: {nvacs}")
        return data_defects
