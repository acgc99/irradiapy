"""This module contains the `DebrisDatabase` class."""

from dataclasses import dataclass
from pathlib import Path

from irradiapy.materials.component import Component
from irradiapy.materials.element import Element
from irradiapy.debris_dataset import DebrisDataset


@dataclass(frozen=True)
class DebrisDatabase:
    """Database of MD debris datasets under a database root."""

    root: Path
    datasets: tuple[DebrisDataset, ...]

    @classmethod
    def from_path(cls, root: Path) -> "DebrisDatabase":
        """Build a database from a root directory directory."""
        dataset_dirs = tuple(
            sorted(
                child
                for child in root.iterdir()
                if child.is_dir() and (child / "meta.json").is_file()
            )
        )

        if not dataset_dirs:
            raise ValueError(f"No debris datasets with meta.json found in {root}")

        datasets = tuple(DebrisDataset.from_path(path) for path in dataset_dirs)
        cls.__validate_electronic_interactions(datasets)
        return cls(root=root, datasets=datasets)

    @staticmethod
    def __validate_electronic_interactions(
        datasets: tuple[DebrisDataset, ...],
    ) -> None:
        """Require all datasets to use the same electronic interactions."""
        electronic_interactions = datasets[0].electronic_interactions
        for dataset in datasets[1:]:
            if dataset.electronic_interactions != electronic_interactions:
                raise ValueError(
                    "All datasets in a database must have the same electronic_interactions."
                )

    def matching_datasets(
        self,
        recoil: Element,
        component: Component,
        electronic_interactions: str | None,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> tuple[DebrisDataset, ...]:
        """Return datasets matching the requested metadata filters."""
        return tuple(
            dataset
            for dataset in self.datasets
            if dataset.matches(
                recoil=recoil,
                component=component,
                electronic_interactions=electronic_interactions,
                interatomic_potentials=interatomic_potentials,
                doi=doi,
                contributors=contributors,
            )
        )

    def has_matches(
        self,
        recoil: Element,
        component: Component,
        electronic_interactions: str | None,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> bool:
        """Return whether at least one dataset matches."""
        return bool(
            self.matching_datasets(
                recoil=recoil,
                component=component,
                electronic_interactions=electronic_interactions,
                interatomic_potentials=interatomic_potentials,
                doi=doi,
                contributors=contributors,
            )
        )

    def matching_files_by_energy(
        self,
        recoil: Element,
        component: Component,
        electronic_interactions: str | None,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> dict[float, list[Path]]:
        """Return merged cascade files by energy for all matching datasets."""
        files_by_energy: dict[float, list[Path]] = {}
        for dataset in self.matching_datasets(
            recoil=recoil,
            component=component,
            electronic_interactions=electronic_interactions,
            interatomic_potentials=interatomic_potentials,
            doi=doi,
            contributors=contributors,
        ):
            for energy, files in dataset.files_by_energy.items():
                files_by_energy.setdefault(energy, []).extend(files)
        return {
            energy: sorted(files)
            for energy, files in sorted(files_by_energy.items(), reverse=True)
        }
