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
    datasets: tuple[DebrisDataset, ...] = field(init=False)
    electronic_interactions: str = field(init=False)

    def __post_init__(self) -> None:
        """Build a database from a root directory."""
        root = Path(self.root)
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
        electronic_interactions = self.__validate_electronic_interactions(datasets)

        self.root = root
        self.datasets = datasets
        self.electronic_interactions = electronic_interactions

    @staticmethod
    def __validate_electronic_interactions(
        datasets: tuple[DebrisDataset, ...],
    ) -> str:
        """Require all datasets to use the same electronic interactions."""
        electronic_interactions = datasets[0].electronic_interactions
        for dataset in datasets[1:]:
            if dataset.electronic_interactions != electronic_interactions:
                raise ValueError(
                    "All datasets in a database must have the same electronic_interactions."
                )
        return electronic_interactions

    def matching_datasets(
        self,
        recoil: Element,
        component: Component,
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
                interatomic_potentials=interatomic_potentials,
                doi=doi,
                contributors=contributors,
            )
        )

    def has_matches(
        self,
        recoil: Element,
        component: Component,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> bool:
        """Return whether at least one dataset matches."""
        return bool(
            self.matching_datasets(
                recoil=recoil,
                component=component,
                interatomic_potentials=interatomic_potentials,
                doi=doi,
                contributors=contributors,
            )
        )

    def matching_files_by_energy(
        self,
        recoil: Element,
        component: Component,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> dict[float, list[Path]]:
        """Return merged cascade files by energy for all matching datasets."""
        files_by_energy: dict[float, list[Path]] = {}
        for dataset in self.matching_datasets(
            recoil=recoil,
            component=component,
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
