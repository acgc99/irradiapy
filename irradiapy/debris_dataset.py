"""This module contains the `DebrisDataset` class."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from irradiapy.materials.component import Component
from irradiapy.materials.element import Element


@dataclass(frozen=True)
class DebrisDataset:
    """Metadata and cascade files for one MD debris dataset."""

    path: Path
    recoil: str
    target: dict[str, float]
    interatomic_potentials: tuple[str, ...]
    electronic_interactions: str
    doi: str
    contributors: tuple[str, ...]
    files_by_energy: dict[float, tuple[Path, ...]]

    @classmethod
    def from_path(cls, path: Path) -> "DebrisDataset":
        """Load one dataset directory containing a ``meta.json`` file."""
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

        target = meta["target"]
        if not isinstance(target, dict):
            raise TypeError(f"{meta_path}: target must be an object")

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

        return cls(
            path=path,
            recoil=str(meta["recoil"]),
            target={str(symbol): float(stoich) for symbol, stoich in target.items()},
            interatomic_potentials=tuple(str(v) for v in meta["interatomic_potentials"]),
            electronic_interactions=meta["electronic_interactions"],
            doi=str(meta["doi"]),
            contributors=tuple(str(v) for v in meta["contributors"]),
            files_by_energy=files_by_energy,
        )

    def matches(
        self,
        recoil: Element,
        component: Component,
        electronic_interactions: str | None,
        interatomic_potentials: list[str] | None = None,
        doi: str | None = None,
        contributors: list[str] | None = None,
    ) -> bool:
        """Return whether this dataset matches the requested recoil/component."""
        if not self.files_by_energy:
            return False
        if self.recoil != recoil.symbol:
            return False
        if not self.target_matches_component(self.target, component):
            return False
        if self.electronic_interactions != electronic_interactions:
            return False
        if doi is not None and self.doi != doi:
            return False
        if interatomic_potentials is not None and set(self.interatomic_potentials) != set(
            interatomic_potentials
        ):
            return False
        if contributors is not None and set(self.contributors) != set(contributors):
            return False
        return True

    @staticmethod
    def target_matches_component(
        target: dict[str, float],
        component: Component,
    ) -> bool:
        """Return whether metadata target stoichiometry matches a component."""
        component_target = component.stoichiometry_dict
        if set(target) != set(component_target):
            return False
        return all(
            np.isclose(float(target[symbol]), component_target[symbol])
            for symbol in target
        )
