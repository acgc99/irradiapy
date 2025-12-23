"""This module contains the `DebrisManager` class."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from numpy import typing as npt
from numpy.lib.recfunctions import structured_to_unstructured as str2unstr
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from irradiapy import dtypes, utils
from irradiapy.analysis.clusters import clusterize_atoms
from irradiapy.enums import DamageEnergyMode, DisplacementMode
from irradiapy.io.lammpsreader import LAMMPSReader
from irradiapy.materials import Component, Element


@dataclass
class DebrisManager:
    """Class used to reconstruct the damage produced by recoils from a database of MD debris.

    Attributes
    ----------
    mddb_dir : Path
        Directory of the MD debris database.
    recoil: Element
        Recoil element.
    component: Component
        Target material.
    compute_damage_energy : bool
        If MD simulations do not include electronic stopping, this should be set to `True`. The
        conversion from recoil energy to damage energy will be performed according to the selected
        `damage_energy_mode`.
    damage_energy_mode : materials.Material.DamageEnergyMode
        Mode for recoil to damage energy calculation.
    displacement_mode : materials.Material.DisplacementMode
        Mode for calculation of number of displacement atoms.
    fp_dist : float
        Distance between the vacancy and the interstitial of a Frenkel pair, in angstroms.
    energy_tolerance : float (default=0.1)
        Tolerance for energy decomposition. For example, if this value if ``0.1``, the recoil energy
        is 194 keV and the database contains an energy of 200 keV, then 194 will be in the range
        200 +/- 20 keV, therefore a cascade of 200 keV will be used, instead of decomposing 194 keV
        into, for example, 100x1 + 50x1 + 20x2 + 3x1 + 1xFP (Frenkel pairs). This fixes biases
        towards smaller clusters (lower energies) and helps reducing cascade overlapping. Set to
        ``0.0`` to disable this feature.
    seed : int, optional (default=0)
        Random seed for random number generator.
    """

    mddb_dir: Path
    recoil: Element
    component: Component
    compute_damage_energy: bool
    damage_energy_mode: DamageEnergyMode
    displacement_mode: DisplacementMode
    fp_dist: float
    energy_tolerance: float = 0.1
    seed: int = 0
    __rng: np.random.Generator = field(init=False)
    __calc_nd: Callable[[float], float] = field(init=False)
    __files: dict[float, list[Path]] = field(init=False)
    __energies: npt.NDArray[np.float64] = field(init=False)
    __nenergies: int = field(init=False)

    def __post_init__(self) -> None:
        self.__rng = np.random.default_rng(self.seed)
        # Scan the database
        self.__files = {
            float(folder.name): list(folder.iterdir())
            for folder in self.mddb_dir.iterdir()
            if folder.is_dir()
        }
        self.__energies = np.array(sorted(self.__files.keys(), reverse=True))
        self.__nenergies = len(self.__energies)
        # Recoil to damage energy conversion
        self.__compute_damage_energy = (
            lambda recoil_energy: self.component.recoil_energy_to_damage_energy(
                recoil_energy=recoil_energy,
                recoil=self.recoil,
                mode=self.damage_energy_mode,
            )
        )
        # Select the displacement model for residual energy
        self.__calc_nd = (
            lambda damage_energy: self.component.damage_energy_to_displacements(
                damage_energy=damage_energy,
                mode=self.displacement_mode,
            )
        )

    def __get_fp_types(self, nfp: int) -> npt.NDArray[np.int32]:
        """Get the types of Frenkel pairs.

        Parameters
        ----------
        nfp : int
            Number of Frenkel pairs.

        Returns
        -------
        npt.NDArray[np.int32]
            Array of Frenkel pair types.
        """
        stoichs = np.asarray(self.component.stoichs, dtype=np.float64)
        cdf = np.cumsum(stoichs)  # cumulative distribution function, e.g. [0.5, 1.0]
        atomic_numbers = np.asarray(
            [e.atomic_number for e in self.component.elements], dtype=np.int32
        )
        r = self.__rng.random(nfp)  # in [0, 1)
        idx = np.searchsorted(cdf, r, side="right")
        types = atomic_numbers[idx]
        return types

    def __get_files(self, recoil_energy: float) -> tuple[dict[float, list[Path]], int]:
        """Get cascade files and number of residual FP for a given recoil energy.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy.

        Returns
        -------
        tuple[dict[float, list[Path]], int]
            Dictionary of selected paths and number of residual FP.
        """
        # Decompose the recoil energy into cascades and residual energy

        # Rounding: if the recoil energy is closer than 10% to any energy of the database, use the
        # closest energy within that tolerance. This avoids:
        # Decomposition biasing towards smaller clusters (190 keV = 100 + 50 + 2x20 keV)
        # Residual FP causing artificial clustering (ignore them)
        diff = np.abs(self.__energies - recoil_energy)
        mask = diff <= self.energy_tolerance * self.__energies
        if np.any(mask):
            recoil_energy = self.__energies[mask][np.argmin(diff[mask])]

        residual_energy = (
            self.__compute_damage_energy(recoil_energy)
            if self.compute_damage_energy
            else recoil_energy
        )
        cascade_counts = np.zeros(self.__nenergies, dtype=np.int64)
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
        nfp = np.round(self.__calc_nd(residual_energy)).astype(np.int64)
        return debris_files, nfp

    def get_recoil_debris(
        self,
        recoil_energy: float,
        recoil_pos: npt.NDArray[np.float64],
        recoil_dir: npt.NDArray[np.float64],
    ) -> dtypes.Defect:
        """Get recoil debris from its energy position, and direction.

        Parameters
        ----------
        recoil_energy : float
            Recoil energy.
        recoil_pos : npt.NDArray[np.float64]
            Recoil position.
        recoil_dir : npt.NDArray[np.float64]
            Recoil direction.
        Returns
        -------

        dtypes.Defect
            Defects after the cascades.
        """
        files, nfp = self.__get_files(recoil_energy)
        # Get the maximum energy available in the database for the given recoil.
        # If no energy is available, return zero to place only FP.
        db_emax = next(
            (energy for energy in self.__energies if len(files[energy])), 0.0
        )
        # Possible to get cascades from the database
        if db_emax > 0.0:
            defects = self.__process_highest_energy_cascade(
                files,
                db_emax,
                recoil_pos,
                recoil_dir,
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
            defects = self.__place_fps_in_sphere(nfp, recoil_pos, recoil_dir)
            return defects
        defects = np.empty(0, dtype=dtypes.defect)
        return defects

    def __process_highest_energy_cascade(
        self,
        files: dict,
        db_emax: float,
        recoil_pos: npt.NDArray[np.float64],
        recoil_dir: npt.NDArray[np.float64],
    ) -> dtypes.Defect:
        """Process the highest energy cascade.

        Parameters
        ----------
        files : dict
            Dictionary of files for each energy.
        db_emax : float
            Energy of the highest energy cascade.
        recoil_pos : npt.NDArray[np.float64]
            Recoil position.
        recoil_dir : npt.NDArray[np.float64]
            Recoil direction.

        Returns
        -------
        dtypes.Defect
            Defects after the highest energy cascade.
        """
        file = files[db_emax][0]
        files[db_emax] = np.delete(files[db_emax], 0)
        defects = utils.io.get_last_reader(LAMMPSReader(file))["atoms"]
        xaxis = np.array([1.0, 0.0, 0.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transform = Rotation.align_vectors([recoil_dir], [xaxis])[0]

        pos = str2unstr(defects[["x", "y", "z"]], dtype=np.float64, copy=False)
        pos = transform.apply(pos) + recoil_pos
        defects["x"] = pos[:, 0]
        defects["y"] = pos[:, 1]
        defects["z"] = pos[:, 2]

        return defects

    def __place_other_debris(
        self,
        files: dict,
        defects: dtypes.Defect,
        parallelepiped: tuple[PCA, npt.NDArray, npt.NDArray],
    ) -> dtypes.Defect:
        """Place other debris in the parallelepiped.

        Parameters
        ----------
        files : dict
            Dictionary of files for each energy.
        defects : dtypes.Defect
            Defects after the highest energy cascade.
        parallelepiped : tuple[PCA, npt.NDArray, npt.NDArray]
            Parallelepiped definition.

        Returns
        -------
        dtypes.Defect
            Defects after placing the other debris.
        """
        for energy in self.__energies:
            for file0 in files[energy]:
                defects_ = utils.io.get_last_reader(LAMMPSReader(file0))["atoms"]

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
        defects: dtypes.Defect,
        nfp: int,
        parallelepiped: tuple[PCA, npt.NDArray, npt.NDArray],
    ) -> dtypes.Defect:
        """Place FPs anywhere in the parallelepiped.

        Parameters
        ----------
        defects : dtypes.Defect
            Defects after placing the other debris.
        nfp : int
            Number of FPs.
        parallelepiped : tuple[PCA, npt.NDArray, npt.NDArray]
            Parallelepiped definition.

        Returns
        -------
        dtypes.Defect
            Defects after placing the FPs.
        """
        defects_ = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects_["type"][:nfp] = self.__get_fp_types(nfp)
        defects_["x"][:nfp] = self.fp_dist / 2.0
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
        self,
        nfp: int,
        recoil_pos: npt.NDArray[np.float64],
        recoil_dir: npt.NDArray[np.float64],
    ) -> dtypes.Defect:
        """Generate FPs in a sphere.

        Parameters
        ----------
        nfp : int
            Number of FPs.
        recoil_pos : npt.NDArray[np.float64]
            Recoil position.
        recoil_dir : npt.NDArray[np.float64]
            Recoil direction.
        Returns
        -------
        dtypes.Defect
            Defects after generating.
        """
        defects_ = np.zeros(2 * nfp, dtype=dtypes.defect)
        defects_["type"][:nfp] = self.__get_fp_types(nfp)
        defects_["x"][:nfp] = self.fp_dist / 2.0
        pos = str2unstr(defects_[["x", "y", "z"]], dtype=np.float64, copy=False)
        pos[:nfp] = Rotation.random(nfp, rng=self.__rng).apply(pos[:nfp])
        pos[nfp:] = -pos[:nfp]

        random = self.__rng.random((nfp, 3))
        theta = np.arccos(2.0 * random[:, 0] - 1.0)
        phi = 2.0 * np.pi * random[:, 1]
        radius = nfp * self.fp_dist / 2.0
        r = radius * np.cbrt(random[:, 2])
        points = np.empty((nfp, 3))
        points[:, 0] = r * np.sin(theta) * np.cos(phi)
        points[:, 1] = r * np.sin(theta) * np.sin(phi)
        points[:, 2] = r * np.cos(theta)

        pos[:nfp] += points
        pos[nfp:] += points
        pos += recoil_pos + recoil_dir * radius
        defects_["x"] = pos[:, 0]
        defects_["y"] = pos[:, 1]
        defects_["z"] = pos[:, 2]

        return defects_

    def __get_parallelepiped(self, atoms: dtypes.Atom) -> tuple:
        """
        Define a parallelepiped from the atomic positions using PCA.

        Parameters
        ----------
        atoms : dtypes.Atom
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
        min_pca: npt.NDArray[np.float64],
        max_pca: npt.NDArray[np.float64],
        npoints: int,
    ) -> npt.NDArray[np.float64]:
        """
        Generate random points within a parallelepiped.

        Parameters
        ----------
        pca : PCA
            PCA object.
        min_pca : npt.NDArray[np.float64]
            Minimum PCA coordinates.
        max_pca : npt.NDArray[np.float64]
            Maximum PCA coordinates.
        npoints : int
            Number of points to generate.

        Returns
        -------
        npt.NDArray[np.float64]
            Random points within the parallelepiped.
        """
        random_points_pca = self.__rng.uniform(min_pca, max_pca, size=(npoints, 3))
        return pca.inverse_transform(random_points_pca)

    def displacements_number_plot(
        self,
        log: bool = True,
        show: bool = False,
        plot_path: Path | None = None,
        dpi: int = 300,
    ) -> None:
        """Plot the number of displacements vs recoil energy.

        Parameters
        ----------
        log : bool
            If True, use logarithmic scale for both axes.
        show : bool, optional (default=False)
            Whether to show the plot.
        plot_path : Path, optional (default=None)
            Output path for the plot.
        dpi : int, optional (default=300)
            Dots per inch for the plot.
        """
        # Debris from the database
        recoil_energies = list(self.__files.keys())
        nds = []
        stds = []
        stes = []
        for recoil_energy in recoil_energies:
            files = self.__files[recoil_energy]
            nfiles = len(files)
            nd = []
            for file in files:
                defects = utils.io.get_last_reader(LAMMPSReader(file))["atoms"]
                # print(file)
                # print(defects)
                nvacs = len(defects[defects["type"] == 0])
                nd.append(nvacs)
            mean = np.mean(nd)
            std = np.std(nd)
            ste = std / np.sqrt(nfiles)
            nds.append(mean)
            stds.append(std)
            stes.append(ste)
        recoil_energies = np.array(recoil_energies)
        nds = np.array(nds)
        stds = np.array(stds)
        stes = np.array(stes)

        # Theoretical models
        dx = (
            self.component.ed_min
            if self.component.ed_min is not None
            else self.component.ed_avr
        )
        recoil_energies_range = np.arange(
            0.0,
            recoil_energies.max(),
            dx,
        )
        damage_energies_range = np.array(
            [
                self.component.recoil_energy_to_damage_energy(
                    recoil_energy=recoil_energy,
                    recoil=self.recoil,
                    mode=self.damage_energy_mode,
                )
                for recoil_energy in recoil_energies_range
            ]
        )
        try:
            nds_nrt = np.array(
                [
                    self.component.damage_energy_to_displacements(
                        damage_energy=damage_energy,
                        mode=DisplacementMode.NRT,
                    )
                    for damage_energy in damage_energies_range
                ]
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            nds_nrt = None
            print(f"Could not compute NRT model: {e}")
        try:
            nds_ferarc = np.array(
                [
                    self.component.damage_energy_to_displacements(
                        damage_energy=damage_energy,
                        mode=DisplacementMode.FERARC,
                    )
                    for damage_energy in damage_energies_range
                ]
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            nds_ferarc = None
            print(f"Could not compute fer-arc model: {e}")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("Recoil energy (keV)")
        ax.set_ylabel("Number of vacancies")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        recoil_energies /= 1e3
        recoil_energies_range /= 1e3
        ax.errorbar(
            recoil_energies,
            nds,
            yerr=stes,
            fmt="o",
            label="debris",
        )
        if nds_nrt is not None:
            ax.plot(
                recoil_energies_range,
                nds_nrt,
                label="NRT",
            )
        if nds_ferarc is not None:
            ax.plot(
                recoil_energies_range,
                nds_ferarc,
                label="fer-arc",
            )

        ax.legend()
        fig.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    def cluster_sizes_plot(
        self,
        vacs_cutoff: float,
        sias_cutoff: float,
        vacs_binwidth: int = 10,
        sias_binwidth: int = 10,
        log: bool = True,
        show: bool = False,
        vacs_plot_path: Path | None = None,
        sias_plot_path: Path | None = None,
        dpi: int = 300,
    ) -> None:
        """Plot the cluster sizes distribution vs recoil energy.

        Parameters
        ----------
        vacs_cutoff : float
            Cutoff distance for vacancy clustering, in angstroms.
        sias_cutoff : float
            Cutoff distance for interstitial clustering, in angstroms.
        vacs_binwidth : int, optional (default=10)
            Bin width for vacancy cluster sizes histogram (bins >=11).
        sias_binwidth : int, optional (default=10)
            Bin width for interstitial cluster sizes histogram (bins >=11).
        log : bool
            If True, use logarithmic scale for the color map.
        show : bool, optional (default=False)
            Whether to show the plot.
        vacs_plot_path : Path, optional (default=None)
            Output path for the vacancy cluster sizes plot.
        sias_plot_path : Path, optional (default=None)
            Output path for the interstitial cluster sizes plot.
        """
        recoil_energies: list[float] = sorted(list(self.__files.keys()))

        vac_sizes: list[npt.NDArray[np.int64]] = []
        sia_sizes: list[npt.NDArray[np.int64]] = []
        nfiles: dict[int, int] = {}

        for recoil_energy in recoil_energies:
            files = self.__files[recoil_energy]
            nfiles[recoil_energy] = len(files)

            vac_sizes_files: list[npt.NDArray[np.int64]] = []
            sia_sizes_files: list[npt.NDArray[np.int64]] = []

            for file in files:
                defects = utils.io.get_last_reader(LAMMPSReader(file))["atoms"]
                vacs = defects[defects["type"] == 0]
                sias = defects[defects["type"] != 0]

                vac_oclusters = clusterize_atoms(vacs, vacs_cutoff)[-1]
                sia_oclusters = clusterize_atoms(sias, sias_cutoff)[-1]

                vac_sizes_files.append(vac_oclusters["size"])
                sia_sizes_files.append(sia_oclusters["size"])

            vac_sizes.append(
                np.concatenate(vac_sizes_files)
                if vac_sizes_files
                else np.empty(0, dtype=np.int64)
            )
            sia_sizes.append(
                np.concatenate(sia_sizes_files)
                if sia_sizes_files
                else np.empty(0, dtype=np.int64)
            )

        def rounded_max(m: int, binwidth: int) -> int:
            # keep 1..10 unbinned, bin >=11 by binwidth
            if m <= 10:
                return 10
            return 10 + int(np.ceil((m - 10) / binwidth) * binwidth)

        max_vac = max((int(v.max()) if v.size else 0) for v in vac_sizes)
        max_sia = max((int(s.max()) if s.size else 0) for s in sia_sizes)
        max_vac = rounded_max(max_vac, vacs_binwidth)
        max_sia = rounded_max(max_sia, sias_binwidth)

        vac_edges = np.concatenate(
            (
                np.arange(0.5, 10.5),
                np.arange(10.5, max_vac + vacs_binwidth + 0.5, vacs_binwidth),
            )
        )
        sia_edges = np.concatenate(
            (
                np.arange(0.5, 10.5),
                np.arange(10.5, max_sia + sias_binwidth + 0.5, sias_binwidth),
            )
        )

        recoils_nbins = len(recoil_energies)
        vhists = np.empty((recoils_nbins, len(vac_edges) - 1), dtype=np.float64)
        ihists = np.empty((recoils_nbins, len(sia_edges) - 1), dtype=np.float64)

        for i, e in enumerate(recoil_energies):
            vhist, _ = np.histogram(vac_sizes[i], bins=vac_edges, density=False)
            ihist, _ = np.histogram(sia_sizes[i], bins=sia_edges, density=False)
            vhist = vhist / nfiles[e]
            ihist = ihist / nfiles[e]
            vhists[i] = vhist
            ihists[i] = ihist

        def plot(
            hists: npt.NDArray[np.float64],
            edges: npt.NDArray[np.float64],
            max_size: int,
            bin_width: int,
            path: Path | None,
        ) -> None:
            # hists = np.ma.masked_where(hists == 0, hists)
            hists = np.ma.masked_less_equal(hists, 0.0)

            fig = plt.figure(figsize=(8, 8))
            gs = GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.05])
            ax = fig.add_subplot(gs[0, 0])
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Cluster size")

            im = ax.imshow(
                hists.T,
                origin="lower",
                aspect="auto",
                norm=mcolors.LogNorm() if log else None,
                cmap="viridis",
                extent=[0, recoils_nbins, 0, hists.shape[1]],
            )

            xticks = np.arange(recoils_nbins) + 0.5
            xlabels = [f"{e / 1000.0:<.1f}" for e in recoil_energies]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)

            if bin_width == 1:
                ylabels = [str(i) for i in range(1, max_size + 1)]
            else:
                ylabels = [str(i) for i in range(1, 11)] + [
                    f"{i}-{i + bin_width - 1}" for i in range(11, max_size, bin_width)
                ]
            yticks = np.arange(len(edges) - 1) + 0.5
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)

            cax = fig.add_subplot(gs[0, 1])
            fig.colorbar(im, cax=cax, label="Counts per cascade")

            fig.tight_layout()

            if path is not None:
                fig.savefig(path, dpi=dpi)
            if show or path is None:
                plt.show()
            plt.close(fig)

        plot(ihists, sia_edges, max_sia, sias_binwidth, sias_plot_path)
        plot(vhists, vac_edges, max_vac, vacs_binwidth, vacs_plot_path)
