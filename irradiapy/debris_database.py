"""This module contains the `DebrisDatabase` class."""

from dataclasses import dataclass, field
from pathlib import Path

from irradiapy.materials.component import Component
from irradiapy.materials.element import Element
from irradiapy.debris_dataset import DebrisDataset


@dataclass
class DebrisDatabase:
    """Database of MD debris datasets under a database root."""

    root: str | Path
    electronic_interactions: str
    target: dict[str, float]
    lattice: str
    interatomic_potentials: list[set[str]] | None = None
    doi: set[str] | None = None
    contributors: list[set[str]] | None = None
    datasets: tuple[DebrisDataset, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Build a database from a root directory."""
        root = Path(self.root)
        electronic_interactions = self.electronic_interactions
        if not isinstance(self.target, dict):
            raise TypeError("target must be a dict.")
        if not self.target:
            raise ValueError("target must contain at least one element.")
        self.target = {
            str(symbol): float(stoich) for symbol, stoich in self.target.items()
        }
        if self.interatomic_potentials is not None:
            self.interatomic_potentials = [
                set(interatomic_potentials)
                for interatomic_potentials in self.interatomic_potentials
            ]
        if self.doi is not None:
            self.doi = set(self.doi)
        if self.contributors is not None:
            self.contributors = [
                set(contributors) for contributors in self.contributors
            ]
        dataset_dirs = tuple(
            sorted(
                child
                for child in root.iterdir()
                if child.is_dir() and (child / "meta.json").is_file()
            )
        )

        if not dataset_dirs:
            raise ValueError(f"No debris datasets with meta.json found in {root}")

        datasets = tuple(
            dataset
            for dataset in (DebrisDataset(path) for path in dataset_dirs)
            if self.__matches_database_filters(dataset)
        )

        if not datasets:
            raise ValueError(
                "No debris datasets matching the database filters found in "
                f"{root}"
            )

        self.root = root
        self.electronic_interactions = electronic_interactions
        self.datasets = datasets

    def __matches_database_filters(self, dataset: DebrisDataset) -> bool:
        """Return whether a dataset matches the database-level filters."""
        if dataset.electronic_interactions != self.electronic_interactions:
            return False

        if not dataset.target_matches_metadata(dataset.target, self.target):
            return False

        if dataset.lattice != self.lattice:
            return False

        if self.interatomic_potentials is not None and (
            dataset.interatomic_potentials not in self.interatomic_potentials
        ):
            return False

        if self.doi is not None and dataset.doi not in self.doi:
            return False

        if self.contributors is not None and (
            dataset.contributors not in self.contributors
        ):
            return False

        return True

    def matching_datasets(
        self,
        recoil: Element,
        component: Component,
    ) -> tuple[DebrisDataset, ...]:
        """Return datasets matching the requested recoil and component."""
        return tuple(
            dataset
            for dataset in self.datasets
            if dataset.matches(recoil=recoil, component=component)
        )

    def has_matches(
        self,
        recoil: Element,
        component: Component,
    ) -> bool:
        """Return whether at least one dataset matches."""
        return bool(
            self.matching_datasets(
                recoil=recoil,
                component=component,
            )
        )

    def matching_files_by_energy(
        self,
        recoil: Element,
        component: Component,
    ) -> dict[float, list[Path]]:
        """Return merged cascade files by energy for all matching datasets."""
        files_by_energy: dict[float, list[Path]] = {}
        for dataset in self.matching_datasets(
            recoil=recoil,
            component=component,
        ):
            for energy, files in dataset.files_by_energy.items():
                files_by_energy.setdefault(energy, []).extend(files)
        return {
            energy: sorted(files)
            for energy, files in sorted(files_by_energy.items(), reverse=True)
        }
