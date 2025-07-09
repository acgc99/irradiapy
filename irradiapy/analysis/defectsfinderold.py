"""This module provides a class to find and analyze defects in crystal structures."""

# TODO: deprecate

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from irradiapy import dtypes
from irradiapy.io.xyzreader import XYZReader
from irradiapy.io.xyzwriter import XYZWriter


class DefectsFinderOld:
    """Class to find and analyze defects in crystal structures (deprecated).

    Warning
    -------
    This class is was used to analyse the defects on CascadesDB, but it is not
    the best option. The class `DefectsFinder` is the recommended one as it is
    optimized. This class is no longer maintained.

    Note
    ----
    Even if an atom is very far away from the simulation box, the closest lattice
    site will be asigned to it. This is a limitation of the current implementation.
    A cutoff distance should be implemented to avoid this and consider the atom as
    free.

    Attributes
    ----------
    nx, ny, nz : int
        Dimensions of the lattice.
    perx, pery, perz : bool
        Periodic boundary conditions along x, y, and z axes.
    dtype : npt.DTypeLike or None
        Data type for atomic positions.
    norm_cell : DefectsFinder._Cell
        Normalized cell for affine transformations.
    rng : np.random.Generator
        Random number generator.
    """

    # Full atom, used only by DefectsFinder
    fatom = np.dtype(
        [("type", int), ("pos", float, 3), ("index", int, 4), ("modular", float, 3)]
    )

    # Unit cell, used only by DefectsFinder
    ucell = np.dtype([("pos", float, 3), ("index", int, 4)])

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        perx: bool,
        pery: bool,
        perz: bool,
        dtype: Optional[npt.DTypeLike] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the `DefectsFinder` object.

        Parameters
        ----------
        nx, ny, nz : int
            Dimensions of the lattice.
        perx, pery, perz : bool
            Periodic boundary conditions along x, y, and z axes.
        dtype : npt.DTypeLike or None, optional
            Data type for atomic positions.
        seed : int or None, optional
            Seed for the random number generator.
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.perx, self.pery, self.perz = perx, pery, perz
        self.dtype = dtype
        self.norm_cell = self.__get_norm_cell()
        self.rng = np.random.default_rng(seed)

    def __get_norm_cell(self) -> "DefectsFinder._Cell":  # type: ignore
        """Calculates the normalized cell for affine transformations.

        Returns
        -------
        DefectsFinder._Cell
            Normalized cell with lattice parameters equal to 1.
        """
        vectors = np.diag([self.nx - 0.5, self.ny - 0.5, self.nz - 0.5])
        origin = np.zeros(3)
        return self._Cell(origin, vectors)

    def __get_curr_cell(
        self, atoms: "DefectsFinder.fatom"  # type: ignore
    ) -> "DefectsFinder._Cell":  # type: ignore
        """Calculates the current cell based on atomic positions.

        Parameters
        ----------
        atoms : dtypes.atom
            Atomic positions.

        Returns
        -------
        DefectsFinder._Cell
            Current cell based on atomic positions.
        """
        origin = np.min(atoms["pos"], axis=0)
        vectors = np.diag(np.max(atoms["pos"], axis=0) - origin)
        return self._Cell(origin, vectors)

    def __affine_transformation(
        self,
        curr_cell: "DefectsFinder._Cell",  # type: ignore
        tar_cell: "DefectsFinder._Cell",  # type: ignore
        atoms: Union["DefectsFinder.fatom", np.ndarray],  # type: ignore
    ) -> Union["DefectsFinder.fatom", np.ndarray]:  # type: ignore
        """Applies affine transformation to atomic positions.

        Parameters
        ----------
        curr_cell : DefectsFinder._Cell
            Current cell.
        tar_cell : DefectsFinder._Cell
            Target cell.
        atoms : dtypes.atom
            Atomic positions.

        Returns
        -------
        dtypes.atom
            Transformed atomic positions.
        """
        if len(atoms) == 0:
            return atoms
        atoms["pos"] -= curr_cell.origin
        transform = np.dot(tar_cell.vectors, np.linalg.inv(curr_cell.vectors))
        atoms["pos"] = np.dot(atoms["pos"], transform)
        atoms["pos"] += tar_cell.origin
        return atoms

    def rescale_translate_rotate(
        self,
        atoms: Union["DefectsFinder.fatom", np.ndarray],  # type: ignore
        pka_pos: np.ndarray,
        pka_theta: float,
        pka_phi: float,
        a0: Optional[Union[float, np.ndarray]] = None,
        a1: Optional[Union[float, np.ndarray]] = None,
    ) -> Union["DefectsFinder.fatom", np.ndarray]:  # type: ignore
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

        Returns
        -------
        dict
            Updated dictionary with rescaled, translated, and rotated atomic positions.
        """
        # Translate
        atoms["pos"] -= pka_pos
        # Rotation matrix, align with PKA initial direction
        xaxis = np.array([1.0, 0.0, 0.0])
        pka_dir = np.array(
            [
                np.sin(pka_theta) * np.cos(pka_phi),
                np.sin(pka_theta) * np.sin(pka_phi),
                np.cos(pka_theta),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transform = R.align_vectors([xaxis], [pka_dir])[0].as_matrix()
        # Scaling matrix
        scaling = a1 / a0
        scaling_matrix = (
            np.diag([scaling] * 3) if isinstance(scaling, float) else np.diag(scaling)
        )
        # Scale-rotate matrix
        transform = transform @ scaling_matrix
        # Apply scaling and rotation
        atoms["pos"] = atoms["pos"] @ transform.T
        return atoms

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
            (xi, yi, zi, sub): []
            for xi in range(self.nx)
            for yi in range(self.ny)
            for zi in range(self.nz)
            for sub in (0, 1)
        }
        return assignments

    def find(
        self,
        ifile_path: Path,
        ofile_path: Path,
        pka_pos: Optional[np.ndarray] = None,
        pka_theta: Optional[float] = None,
        pka_phi: Optional[float] = None,
        a0: Optional[float] = None,
        a1: Optional[float] = None,
    ) -> np.ndarray:  # type: ignore
        """Finds defects in `ifile_path` and saves the info in `ofile_path`.

        Parameters
        ----------
        ifile_path : Path
            Input file path.
        ofile_path : Path
            Output file path.
        pka_pos : None or npt.NDArray[np.float64], optional
            Position vector of the primary knock-on atom (PKA).
        pka_theta : None or float, optional
            Polar angle (in radians) for the PKA direction.
        pka_phi : None or float, optional
            Azimuthal angle (in radians) for the PKA direction.
        a0 : None or float, optional
            Initial scaling factor.
        a1 : None or float, optional
            Final scaling factor.

        Returns
        -------
        dtypes.defect
            Array of defects.
        """
        reader = XYZReader(ifile_path, self.dtype)
        atoms0 = list(reader)[0]
        natoms = len(atoms0)
        atoms = np.zeros(natoms, dtype=self.fatom)
        atoms["type"] = atoms0["type"]
        atoms["pos"] = atoms0["pos"]
        del atoms0
        print(f"Number of atoms: {natoms}")
        # Affine transformation
        curr_cell = self.__get_curr_cell(atoms)
        atoms = self.__affine_transformation(curr_cell, self.norm_cell, atoms)
        # Unit cell coordinates (bcc lattice)
        ucell = np.array(
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
            dtype=self.ucell,
        )
        # Atom index and modular coordinates
        index, modular = np.divmod(atoms["pos"], 1.0)
        atoms["index"][:, :-1], atoms["modular"] = index, modular
        del index, modular
        # Array of distances:
        # each row (atom) contains the distance to each unit cell position
        dist = np.sum(np.square(atoms["modular"][:, np.newaxis] - ucell["pos"]), axis=2)
        # Get assigned lattice index position for the atom
        atoms["index"] += ucell["index"][np.argmin(dist, axis=1)]
        # Periodic boundary conditions
        if self.perx:
            atoms["index"][:, 0][atoms["index"][:, 0] == self.nx] = 0
            atoms["index"][:, 0][atoms["index"][:, 0] == -1] = self.nx - 1
        if self.pery:
            atoms["index"][:, 1][atoms["index"][:, 1] == self.ny] = 0
            atoms["index"][:, 1][atoms["index"][:, 1] == -1] = self.ny - 1
        if self.perz:
            atoms["index"][:, 2][atoms["index"][:, 2] == self.nz] = 0
            atoms["index"][:, 2][atoms["index"][:, 2] == -1] = self.nz - 1
        # Create all lattice positions and assign a lattice position to each atom
        assignments = self.__generate_assignments()
        print(f"Lattice sites: {len(assignments)}")
        for i in range(natoms):
            assignments[tuple(atoms["index"][i])].append(i)
        # Vacancies and interstitials identification
        defects = np.empty(0, dtypes.defect)
        for key, value in assignments.items():
            # lattice site cartesian coordinates
            site_pos = np.array(key[:-1]) + key[-1] * 0.5
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
        # Undo affine transformation
        defects = self.__affine_transformation(self.norm_cell, curr_cell, defects)
        if pka_pos is None or pka_theta is None or pka_phi is None:
            print("PKA parameters or lattice parameter are ungiven.")
            defects["pos"] -= np.mean(defects["pos"], axis=0)
            if a0 is not None and a1 is not None:
                defects["pos"] *= a1 / a0
        else:
            defects = self.rescale_translate_rotate(
                defects, pka_pos, pka_theta, pka_phi, a0, a1
            )
        nvacs = len(np.where(defects["type"] == 0)[0])
        nsias = len(defects) - nvacs
        print(f"Number of interstitials: {nsias}")
        print(f"Number of vacancies: {nvacs}")
        writer = XYZWriter(ofile_path)
        writer.save(defects)
        writer.close()
        return defects

    @dataclass
    class _Cell:
        """Helper class that contains data of the simulation cell.

        Attributes
        ----------
        origin : npt.NDArray[np.float64]
            Origin of the cell.
        vectors : npt.NDArray[np.float64]
            Vectors defining the cell.
        """

        origin: np.ndarray
        vectors: np.ndarray
