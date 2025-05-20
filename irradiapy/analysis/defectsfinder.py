"""This module provides a class to find and analyze defects in crystal structures."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from irradiapy import dtypes


@dataclass
class DefectsFinder:
    """Class to find and analyze defects in crystal structures.

    Note
    ----
    Not ready to be used. Targeted to LAMMPS dump results.

    Note
    ----
    Only for bcc lattice.

    Note
    ----
    Even if an atom is very far away from the simulation box, the closest lattice
    site will be asigned to it. This is a limitation of the current implementation.
    A cutoff distance should be implemented to avoid this and consider the atom as
    free.

    Note
    ----
    Only one unit cell around the simulation box is considered for periodic
    boundary conditions.
    """

    lattice: str
    a0: float
    perx: bool
    pery: bool
    perz: bool
    nxhi: int
    nyhi: int
    nzhi: int
    nxlo: int = 0
    nylo: int = 0
    nzlo: int = 0

    FinderAtom = np.dtype(
        [("type", int), ("pos", float, 3), ("index", int, 4), ("modular", float, 3)]
    )
    FinderAtom_Check = np.ndarray[tuple[int, ...], FinderAtom]
    UnitCell = np.dtype([("pos", float, 3), ("index", int, 4)])
    UnitCell_Check = np.ndarray[tuple[int, ...], UnitCell]

    def __post_init__(self) -> None:
        if self.lattice == "bcc":
            self.unit_cell = np.array(
                [
                    ([0.0, 0.0, 0.0], [0, 0, 0, 0]),
                    ([1.0, 0.0, 0.0], [1, 0, 0, 0]),
                    ([1.0, 1.0, 0.0], [1, 1, 0, 0]),
                    ([0.0, 1.0, 0.0], [0, 1, 0, 0]),
                    ([0.0, 0.0, 1.0], [0, 0, 1, 0]),
                    ([1.0, 0.0, 1.0], [1, 0, 1, 0]),
                    ([1.0, 1.0, 1.0], [1, 1, 1, 0]),
                    ([0.0, 1.0, 1.0], [0, 1, 1, 0]),
                    ([0.5, 0.5, 0.5], [0, 0, 0, 1]),
                ],
                dtype=self.UnitCell,
            )
            self.unit_cell["pos"] *= self.a0
        else:
            raise ValueError("Only bcc lattice is supported.")

        self.nx = self.nxhi - self.nxlo
        self.ny = self.nyhi - self.nylo
        self.nz = self.nzhi - self.nzlo

    def rescale_translate_rotate(
        self,
        pos: npt.NDArray[np.float64],
        a1: Union[float, np.number],
        pka_pos: npt.NDArray[np.float64],
        pka_theta: float,
        pka_phi: float,
    ) -> None:
        """Rescales, translates, and rotates the positions of atoms.

        Parameters
        ----------
        atoms : dict
            Dictionary containing atomic positions under the key "pos".
        pka_pos : array-like
            Position vector of the primary knock-on atom (PKA).
        pka_theta : float
            Polar angle (in radians) for the PKA direction.
        pka_phi : float
            Azimuthal angle (in radians) for the PKA direction.
        a0 : float or array-like
            Initial scaling factor or vector.
        a1 : float or array-like
            Final scaling factor or vector.
        """
        # Translate
        pos -= pka_pos
        # Rotation matrix, align with PKA initial direction
        xaxis = np.array([1.0, 0.0, 0.0])
        pka_dir = np.array(
            [
                np.sin(pka_theta) * np.cos(pka_phi),
                np.sin(pka_theta) * np.sin(pka_phi),
                np.cos(pka_theta),
            ]
        )
        transform = R.align_vectors([xaxis], [pka_dir])[0].as_matrix()
        # Scaling matrix
        scaling = a1 / self.a0
        scaling_matrix = (
            np.diag([scaling] * 3) if isinstance(scaling, float) else np.diag(scaling)
        )
        # Scale-rotate matrix
        transform = transform @ scaling_matrix
        # Apply scaling and rotation
        pos[:] = pos @ transform.T

    def __generate_assignments(self) -> dict:
        """Generates the corresponding lattice in index coordinates (bcc lattice).

        Returns
        -------
        dict
            Dictionary with lattice positions and assigned atoms.

        Examples
        --------
        assignments = {
            (1, 1, 1, 1): [123, 23],
            ...
        }
        lattice position with indices (1, 1, 1, 1) has two atoms assigned, whose
        numbers are 123 and 23
        """
        assignments = {
            (xi + self.nxlo, yi + self.nylo, zi + self.nzlo, sub): []
            for xi in range(self.nx)
            for yi in range(self.ny)
            for zi in range(self.nz)
            for sub in (0, 1)
        }
        return assignments

    def find(
        self,
        types: npt.NDArray[np.integer],
        xs: Optional[npt.NDArray[np.float64]] = None,
        ys: Optional[npt.NDArray[np.float64]] = None,
        zs: Optional[npt.NDArray[np.float64]] = None,
        poss: Optional[npt.NDArray[np.float64]] = None,
        a1: Optional[Union[float, np.number]] = None,
        pka_pos: Optional[npt.NDArray[np.float64]] = None,
        pka_theta: Optional[float] = None,
        pka_phi: Optional[float] = None,
        recenter: Optional[bool] = False,
    ) -> np.ndarray:
        """Finds defects in `ifile_path` and saves the info in `ofile_path`.

        Parameters
        ----------

        Returns
        -------
        dtypes.defect_check
            Array of defects.
        """
        natoms = len(types)
        atoms = np.zeros(natoms, dtype=self.FinderAtom)
        atoms["type"] = types
        if xs is None or ys is None or zs is None:
            atoms["pos"] = poss
        elif poss is None:
            poss = np.empty((len(types), 3))
            poss[:, 0] = xs
            poss[:, 1] = ys
            poss[:, 2] = zs
            atoms["pos"] = poss
        else:
            raise ValueError("Either xs, ys, zs or poss must be given.")

        # Atom (almost) index and modular coordinates
        atoms["index"][:, :-1], atoms["modular"] = np.divmod(atoms["pos"], self.a0)
        # Array of distances: each row (atom) contains the distance to each
        # unit cell position
        dist = np.sum(
            np.square(atoms["modular"][:, np.newaxis] - self.unit_cell["pos"]), axis=2
        )
        # Get index corrdinates for the atom
        atoms["index"] += self.unit_cell["index"][np.argmin(dist, axis=1)]
        # Apply periodic boundary conditions
        self.__apply_boundary_conditions(atoms)
        # Create all lattice positions and assign a lattice position to each atom
        assignments = self.__generate_assignments()
        print(f"Number of atoms: {natoms}")
        print(f"Lattice sites:   {len(assignments)}")
        for i in range(natoms):
            assignments[tuple(atoms["index"][i])].append(i)
        # Vacancies and interstitials identification
        defects = self.__defect_identification(assignments, atoms)
        # Set the origin to the PKA position and align its direction
        # with the x-axis
        if len(defects) and recenter:
            if (
                a1 is not None
                and pka_pos is not None
                and pka_theta is not None
                and pka_phi is not None
            ):
                self.rescale_translate_rotate(
                    defects["pos"], a1, pka_pos, pka_theta, pka_phi
                )
            else:
                defects["pos"] -= np.mean(defects["pos"], axis=0)
                if a1 is not None:
                    defects["pos"] *= a1
        nvacs = len(np.where(defects["type"] == 0)[0])
        nsias = len(defects) - nvacs
        print(f"Number of interstitials: {nsias}")
        print(f"Number of vacancies: {nvacs}")
        return defects

    def __apply_boundary_conditions(
        self, atoms: "DefectsFinder.FinderAtom_Check"
    ) -> None:
        """Applies boundary conditions to the atomic index coordinates.

        Note
        ----
        Only one unit cell around the simulation box is considered for periodic
        boundary conditions.

        Parameters
        ----------
        atoms : npt.NDArray[np.float64]
            Atomic positions.
        """
        if self.perx:
            atoms["index"][:, 0][atoms["index"][:, 0] == self.nxhi] = self.nxlo
            atoms["index"][:, 0][atoms["index"][:, 0] == self.nxlo - 1] = self.nxhi - 1
        if self.pery:
            atoms["index"][:, 1][atoms["index"][:, 1] == self.nyhi] = self.nylo
            atoms["index"][:, 1][atoms["index"][:, 1] == self.nylo - 1] = self.nyhi - 1
        if self.perz:
            atoms["index"][:, 2][atoms["index"][:, 2] == self.nzhi] = self.nzlo
            atoms["index"][:, 2][atoms["index"][:, 2] == self.nzlo - 1] = self.nzhi - 1

    def __index_to_normalised_cartesian(
        self, index: tuple[int, int, int, int]
    ) -> npt.NDArray[np.float64]:
        """Converts index coordinates to normalised Cartesian coordinates.

        Parameters
        ----------
        index : tuple
            Index coordinates.

        Returns
        -------
        npt.NDArray[np.float64]
            Normalised Cartesian coordinates.
        """
        return (np.array(index[:-1]) + index[-1] * 0.5) * self.a0

    def __defect_identification(
        self, assignments: dict, atoms: "DefectsFinder.FinderAtom_Check"
    ) -> np.ndarray:
        defects = np.empty(0, dtypes.defect)
        for key, value in assignments.items():
            # lattice site cartesian coordinates
            site_pos = self.__index_to_normalised_cartesian(key)
            # no atoms assigned > vacancy
            if len(value) == 0:
                defects = np.concatenate(
                    (defects, np.array([(0, site_pos)], dtype=dtypes.defect))
                )
            elif len(value) > 1:
                # do not consider as defect the closest atom to the lattice site
                dist = np.sum(np.square(site_pos - atoms[value]["pos"]))
                mask = np.full(len(value), True)
                mask[np.argmin(dist)] = False
                inters = np.empty(len(value) - 1, dtypes.defect)
                inters["type"] = atoms[value]["type"][mask]
                inters["pos"] = atoms[value]["pos"][mask]
                defects = np.concatenate((defects, inters))
        return defects
