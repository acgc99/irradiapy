"""This module contains the `DamageDB` class."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from irradiapy import dpa, dtypes, materials
from irradiapy.io.xyzreader import XYZReader


@dataclass
class DamageDB:
    """Class used to reconstruct the damage produced by a PKA from a database of MD debris.

    Attributes
    ----------
    dir_mddb : Path
        Directory of the MD debris database.
    mat_pka : materials.Material
        PKA material.
    mat_target : materials.Material
        Target material.
    compute_tdam : bool
        Whether to apply Lindhard's formula to the recoil energy. It should be `True` for
        MD simulations without electronic stopping.
    dpa_mode : dpa.DpaMode
        Mode for dpa calculation.
    force_lss : bool, optional
        If True, force the use of the Lindhard formula for damage energy calculation. Default is
        False.
    seed : int, optional
        Random seed for random number generator. Default is 0.
    """

    dir_mddb: Path
    compute_tdam: bool
    mat_pka: materials.Material
    mat_target: materials.Material
    dpa_mode: dpa.DpaMode
    force_lss: bool = False
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
        self.__compute_damage_energy = lambda x: dpa.compute_damage_energy(
            x, self.mat_pka, self.mat_target, force_lss=self.force_lss
        )
        # Select the dpa model for residual energy
        if self.dpa_mode == dpa.DpaMode.NRT:
            self.__calc_nd = lambda x: dpa.calc_nrt_dpa(x, self.mat_target)
        elif self.dpa_mode == dpa.DpaMode.ARC:
            self.__calc_nd = lambda x: dpa.calc_arc_dpa(x, self.mat_target)
        elif self.dpa_mode == dpa.DpaMode.FERARC:
            self.__calc_nd = lambda x: dpa.calc_fer_arc_dpa(x, self.mat_target)
        else:
            raise ValueError("Invalid dpa mode")

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
        nfp = self.__calc_nd(residual_energy)
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
            return self.__place_fps_in_sphere(nfp, pka_pos, pka_dir)
        return np.empty(0, dtype=dtypes.defect)

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
        defects = next(iter(XYZReader(file)))
        xaxis = np.array([1.0, 0.0, 0.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transform = Rotation.align_vectors([pka_dir], [xaxis])[0].as_matrix()
        defects["pos"] = defects["pos"] @ transform.T
        defects["pos"] += pka_pos
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
                defects0 = next(iter(XYZReader(file0)))
                defects0["pos"] -= np.mean(defects0["pos"], axis=0)
                defects0["pos"] = Rotation.random().apply(defects0["pos"])
                pos0 = self.__get_parallelepiped_points(*parallelepiped, 1)
                defects0["pos"] += pos0
                defects = np.concatenate((defects, defects0))
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
        defects0 = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects0["type"][:nfp] = self.mat_target.atomic_number
        defects0["type"][nfp:] = 0
        defects0["pos"][:nfp, 0] = self.mat_target.dist_fp / 2.0
        defects0["pos"][:nfp] = Rotation.random(nfp).apply(defects0["pos"][:nfp])
        defects0["pos"][nfp:] = -defects0["pos"][:nfp]
        pos0 = self.__get_parallelepiped_points(*parallelepiped, nfp)
        defects0["pos"][:nfp] += pos0
        defects0["pos"][nfp:] += pos0
        return np.concatenate((defects, defects0))

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
        defects = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects["type"][:nfp] = self.mat_target.atomic_number
        defects["type"][nfp:] = 0
        defects["pos"][:nfp, 0] = self.mat_target.dist_fp / 2.0
        defects["pos"][:nfp] = Rotation.random(nfp).apply(defects["pos"][:nfp])
        defects["pos"][nfp:] = -defects["pos"][:nfp]

        random = self.__rng.random((nfp, 3))
        theta = np.arccos(2.0 * random[:, 0] - 1.0)
        phi = 2.0 * np.pi * random[:, 1]
        radius = nfp * self.mat_target.dist_fp / 2.0
        r = radius * np.cbrt(random[:, 2])
        points = np.empty((nfp, 3))
        points[:, 0] = r * np.sin(theta) * np.cos(phi)
        points[:, 1] = r * np.sin(theta) * np.sin(phi)
        points[:, 2] = r * np.cos(theta)

        defects["pos"][:nfp] += points
        defects["pos"][nfp:] += points
        defects["pos"] += pka_pos + pka_dir * radius
        return defects

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
        pca = PCA(n_components=3)
        pca.fit(atoms["pos"])
        atoms_pca = pca.transform(atoms["pos"])
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
