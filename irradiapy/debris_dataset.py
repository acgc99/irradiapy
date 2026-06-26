"""This module contains the `DebrisDataset` class."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from irradiapy.materials.component import Component
from irradiapy.materials.element import Element


@dataclass
class DebrisDataset:
    """Metadata and cascade files for one MD debris dataset."""

    path: str | Path
    recoil: str = field(init=False)
    target: dict[str, float] = field(init=False)
    interatomic_potentials: set[str] = field(init=False)
    electronic_interactions: str = field(init=False)
    doi: str = field(init=False)
    contributors: set[str] = field(init=False)
    files_by_energy: dict[float, tuple[Path, ...]] = field(init=False)
    max_energy: float = field(init=False)

    def __post_init__(self) -> None:
        """Load one dataset directory containing a ``meta.json`` file."""
        path = Path(self.path)
        meta_path = path / "meta.json"
        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)

        required_keys = {
            "recoil",
            "target",
            "interatomic_potentials",
            "electronic_interactions",
            "doi",
            "contributors",
        }
        missing_keys = required_keys - set(meta)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{meta_path} is missing required metadata: {missing}")

        self.recoil = meta["recoil"]
        self.target = meta["target"]
        self.electronic_interactions = meta["electronic_interactions"]
        self.doi = meta["doi"]
        self.interatomic_potentials = set(meta["interatomic_potentials"])
        self.contributors = set(meta["contributors"])

        files_by_energy: dict[float, tuple[Path, ...]] = {}
        for folder in path.iterdir():
            if not folder.is_dir():
                continue
            try:
                energy = float(folder.name)
            except ValueError:
                continue
            files = tuple(sorted(folder.glob("*.xyz")))
            if files:
                files_by_energy[energy] = files
        self.files_by_energy = files_by_energy
        self.max_energy = max(files_by_energy, default=0.0)

    def matches(
        self,
        recoil: Element,
        component: Component,
    ) -> bool:
        """Return whether this dataset matches the requested recoil/component."""
        if not self.files_by_energy:
            return False
        if self.recoil != recoil.symbol:
            return False
        if not self.target_matches_component(self.target, component):
            return False
        return True

    @staticmethod
    def target_matches_component(
        target: dict[str, float],
        component: Component,
    ) -> bool:
        """Return whether metadata target stoichiometry matches a component."""
        return DebrisDataset.target_matches_metadata(
            target,
            component.stoichiometry_dict,
        )

    @staticmethod
    def target_matches_metadata(
        target: dict[str, float],
        metadata_target: dict[str, float],
    ) -> bool:
        """Return whether two metadata target stoichiometries match."""
        if set(target) != set(metadata_target):
            return False
        return all(
            np.isclose(float(target[symbol]), float(metadata_target[symbol]))
            for symbol in target
        )
