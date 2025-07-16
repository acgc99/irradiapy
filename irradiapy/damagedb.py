"""This module contains the `DamageDB` class."""

import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as str2unstr
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from irradiapy import dtypes, materials
from irradiapy.io.lammpsreader import LAMMPSReader


@dataclass
class DamageDB:
    """Class used to reconstruct the damage produced by a PKA from a database of MD debris.

    Attributes
    ----------
    dir_mddb : Path
        Directory of the MD debris database.
    compute_tdam : bool
        Whether to apply Lindhard's formula to the recoil energy. It should be `True` for
        MD simulations without electronic stopping.
    mat_pka : materials.Material
        PKA material.
    mat_target : materials.Material
        Target material.
    dpa_mode : materials.Material.DpaMode
        Mode for dpa calculation.
    tdam_mode : materials.Material.TdamMode
        Mode for PKA to damage energy calculation.
    seed : int, optional (default=0)
        Random seed for random number generator.
    """

    dir_mddb: Path
    compute_tdam: bool
    mat_pka: "materials.Material"
    mat_target: "materials.Material"
    dpa_mode: "materials.Material.DpaMode"
    tdam_mode: "materials.Material.TdamMode"
    seed: int = 0
    __rng: np.random.Generator = field(init=False)
    __calc_nd: callable = field(init=False)
    __files: dict[float, list[Path]] = field(init=False)
    __energies: np.ndarray = field(init=False)
    __nenergies: int = field(init=False)

    def __post_init__(self) -> None:
        self.__rng = np.random.default_rng(self.seed)
        # Scan the database
        self.__files = {
            float(folder.name): list(folder.iterdir())
            for folder in self.dir_mddb.iterdir()
            if folder.is_dir()
        }
        self.__energies = np.array(sorted(self.__files.keys(), reverse=True))
        self.__nenergies = len(self.__energies)
        # PKA energy to damage energy conversion
        self.__compute_damage_energy = lambda x: self.mat_pka.epka_to_tdam(
            mat_pka=self.mat_target, epka=x, mode=self.tdam_mode
        )
        # Select the dpa model for residual energy
        self.__calc_nd = lambda x: self.mat_target.tdam_to_dpa(
            tdam=x, mode=self.dpa_mode
        )

    def __get_files(self, pka_e: float) -> tuple[dict[float, list[Path]], int]:
        """Get cascade files and number of residual FP for a given PKA energy.

        Parameters
        ----------
        pka_e : float
            PKA energy.

        Returns
        -------
        tuple[dict[float, list[Path]], int]
            Dictionary of selected paths and number of residual FP.
        """
        # Decompose the PKA energy into cascades and residual energy
        residual_energy = (
            self.__compute_damage_energy(pka_e) if self.compute_tdam else pka_e
        )
        cascade_counts = np.zeros(self.__nenergies, dtype=int)
        for i, energy in enumerate(self.__energies):
            cascade_counts[i], residual_energy = divmod(residual_energy, energy)
        # Select the files for each energy
        if residual_energy > 0:
            residual_energy = self.__compute_damage_energy(residual_energy)
        debris_files = {
            energy: self.__rng.choice(self.__files[energy], cascade_counts[i])
            for i, energy in enumerate(self.__energies)
        }
        # Get the number of residual FP
        nfp = np.round(self.__calc_nd(residual_energy)).astype(np.int32)
        return debris_files, nfp

    def get_pka_debris(
        self, pka_e: float, pka_pos: np.ndarray, pka_dir: np.ndarray
    ) -> np.ndarray:
        """Get PKA debris from its energy position, and direction.

        Parameters
        ----------
        pka_e : float
            PKA energy.
        pka_pos : np.ndarray
            PKA position.
        pka_dir : np.ndarray
            PKA direction.

        Returns
        -------

        np.ndarray
            Defects after the cascades.
        """
        files, nfp = self.__get_files(pka_e)
        # Get the maximum energy available in the database for the given PKA.
        # If no energy is available, return zero to place only FP.
        db_emax = next(
            (energy for energy in self.__energies if len(files[energy])), 0.0
        )
        # Possible to get cascades from the database
        if db_emax > 0.0:
            defects = self.__process_highest_energy_cascade(
                files, db_emax, pka_pos, pka_dir
            )
            parallelepiped = self.__get_parallelepiped(defects)
            defects = self.__place_other_debris(files, defects, parallelepiped)
            if nfp:
                defects = self.__place_fps_in_parallelepiped(
                    defects, nfp, parallelepiped
                )
            return defects
        # If no energy is available, generate FP only
        if nfp:
            defects = self.__place_fps_in_sphere(nfp, pka_pos, pka_dir)
            return defects
        defects = np.empty(0, dtype=dtypes.defect)
        return defects

    def __process_highest_energy_cascade(
        self,
        files: dict,
        db_emax: float,
        pka_pos: np.ndarray,
        pka_dir: np.ndarray,
    ) -> np.ndarray:
        """Process the highest energy cascade.

        Parameters
        ----------
        files : dict
            Dictionary of files for each energy.
        db_emax : float
            Energy of the highest energy cascade.
        pka_pos : np.ndarray
            PKA position.
        pka_dir : np.ndarray
            PKA direction.

        Returns
        -------
        np.ndarray
            Defects after the highest energy cascade.
        """
        file = files[db_emax][0]
        files[db_emax] = np.delete(files[db_emax], 0)
        defects = deque(LAMMPSReader(file), maxlen=1).pop()["atoms"]
        xaxis = np.array([1.0, 0.0, 0.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transform = Rotation.align_vectors([pka_dir], [xaxis])[0]

        pos = str2unstr(defects[["x", "y", "z"]], dtype=np.float64, copy=False)
        pos = transform.apply(pos) + pka_pos
        defects["x"] = pos[:, 0]
        defects["y"] = pos[:, 1]
        defects["z"] = pos[:, 2]

        return defects

    def __place_other_debris(
        self,
        files: dict,
        defects: np.ndarray,
        parallelepiped: tuple[PCA, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Place other debris in the parallelepiped.

        Parameters
        ----------
        files : dict
            Dictionary of files for each energy.
        defects : np.ndarray
            Defects after the highest energy cascade.
        parallelepiped : tuple[PCA, np.ndarray, np.ndarray]
            Parallelepiped definition.

        Returns
        -------
        np.ndarray
            Defects after placing the other debris.
        """
        for energy in self.__energies:
            for file0 in files[energy]:
                defects_ = deque(LAMMPSReader(file0), maxlen=1).pop()["atoms"]

                # TODO: why recenter?
                defects_["x"] -= np.mean(defects_["x"])
                defects_["y"] -= np.mean(defects_["y"])
                defects_["z"] -= np.mean(defects_["z"])
                transform = Rotation.random(rng=self.__rng)
                pos = str2unstr(defects_[["x", "y", "z"]], dtype=np.float64, copy=False)
                pos0 = self.__get_parallelepiped_points(*parallelepiped, 1)
                pos = transform.apply(pos) + pos0
                defects_["x"] = pos[:, 0]
                defects_["y"] = pos[:, 1]
                defects_["z"] = pos[:, 2]

                defects = np.concatenate((defects, defects_))
        return defects

    def __place_fps_in_parallelepiped(
        self,
        defects: np.ndarray,
        nfp: int,
        parallelepiped: tuple[PCA, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Place FPs anywhere in the parallelepiped.

        Parameters
        ----------
        defects : np.ndarray
            Defects after placing the other debris.
        nfp : int
            Number of FPs.
        parallelepiped : tuple[PCA, np.ndarray, np.ndarray]
            Parallelepiped definition.

        Returns
        -------
        np.ndarray
            Defects after placing the FPs.
        """
        defects_ = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects_["type"][:nfp] = self.mat_target.atomic_number
        defects_["x"][:nfp] = self.mat_target.dist_fp / 2.0
        pos = str2unstr(defects_[["x", "y", "z"]], dtype=np.float64, copy=False)
        pos[:nfp] = Rotation.random(nfp, rng=self.__rng).apply(pos[:nfp])
        pos[nfp:] = -pos[:nfp]
        pos0 = self.__get_parallelepiped_points(*parallelepiped, nfp)
        pos[:nfp] += pos0
        pos[nfp:] += pos0
        defects_["x"] = pos[:, 0]
        defects_["y"] = pos[:, 1]
        defects_["z"] = pos[:, 2]

        return np.concatenate((defects, defects_))

    def __place_fps_in_sphere(
        self, nfp: int, pka_pos: np.ndarray, pka_dir: np.ndarray
    ) -> np.ndarray:
        """Generate FPs in a sphere.

        Parameters
        ----------
        nfp : int
            Number of FPs.
        pka_pos : np.ndarray
            PKA position.
        pka_dir : np.ndarray
            PKA direction.

        Returns
        -------
        np.ndarray
            Defects after generating.
        """
        defects_ = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects_["type"][:nfp] = self.mat_target.atomic_number
        defects_["x"][:nfp] = self.mat_target.dist_fp / 2
        pos = str2unstr(defects_[["x", "y", "z"]], dtype=np.float64, copy=False)
        pos[:nfp] = Rotation.random(nfp, rng=self.__rng).apply(pos[:nfp])
        pos[nfp:] = -pos[:nfp]

        random = self.__rng.random((nfp, 3))
        theta = np.arccos(2.0 * random[:, 0] - 1.0)
        phi = 2.0 * np.pi * random[:, 1]
        radius = nfp * self.mat_target.dist_fp / 2.0
        r = radius * np.cbrt(random[:, 2])
        points = np.empty((nfp, 3))
        points[:, 0] = r * np.sin(theta) * np.cos(phi)
        points[:, 1] = r * np.sin(theta) * np.sin(phi)
        points[:, 2] = r * np.cos(theta)

        pos[:nfp] += points
        pos[nfp:] += points
        pos += pka_pos + pka_dir * radius
        defects_["x"] = pos[:, 0]
        defects_["y"] = pos[:, 1]
        defects_["z"] = pos[:, 2]

        return defects_

    def __get_parallelepiped(self, atoms: np.ndarray) -> tuple:
        """
        Define a parallelepiped from the atomic positions using PCA.

        Parameters
        ----------
        atoms : np.ndarray
            Atomic positions.

        Returns
        -------
        tuple
            PCA object, minimum PCA coordinates, maximum PCA coordinates.
        """
        pos = str2unstr(atoms[["x", "y", "z"]], dtype=np.float64, copy=False)
        pca = PCA(n_components=3)
        pca.fit(pos)
        atoms_pca = pca.transform(pos)
        min_pca = np.min(atoms_pca, axis=0)
        max_pca = np.max(atoms_pca, axis=0)
        return pca, min_pca, max_pca

    def __get_parallelepiped_points(
        self,
        pca: PCA,
        min_pca: np.ndarray,
        max_pca: np.ndarray,
        npoints: int,
    ) -> np.ndarray:
        """
        Generate random points within a parallelepiped.

        Parameters
        ----------
        pca : PCA
            PCA object.
        min_pca : np.ndarray
            Minimum PCA coordinates.
        max_pca : np.ndarray
            Maximum PCA coordinates.
        npoints : int
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Random points within the parallelepiped.
        """
        random_points_pca = self.__rng.uniform(min_pca, max_pca, size=(npoints, 3))
        return pca.inverse_transform(random_points_pca)
