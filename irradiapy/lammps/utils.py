"""Utilities for interacting with a LAMMPS instance."""

from importlib import import_module
from typing import Any

import numpy as np


class LAMMPSUnavailableError(ImportError):
    """Raised when the external LAMMPS Python module is unavailable."""


def load_lammps_class() -> type[Any]:
    """Return the external ``lammps.lammps`` class.

    The import is intentionally deferred until LAMMPS-backed functionality is
    called so that the rest of irradiapy remains usable without LAMMPS.
    """
    try:
        module = import_module("lammps")
        return module.lammps
    except (ImportError, AttributeError) as exc:
        raise LAMMPSUnavailableError(
            "The external LAMMPS Python module is required for this operation. "
            "Build LAMMPS with Python support and install the module from the "
            "LAMMPS source tree; do not install the unofficial 'lammps' package "
            "from PyPI."
        ) from exc


def get_properties_from_group(
    lmp: Any, group: str, properties: list[str]
) -> dict[str, np.ndarray]:
    """Return the requested properties of a group of atoms.

    Parameters
    ----------
    lmp
        A LAMMPS instance.
    group
        The name of the group.
    properties
        The properties to extract.

    Returns
    -------
    dict[str, np.ndarray]
        The properties of the atoms in the group.
    """
    masks = lmp.numpy.extract_atom("mask")
    group_list = lmp.available_ids("group")
    try:
        idx = group_list.index(group)
    except ValueError as exc:
        raise NameError(f"group '{group}' is not defined") from exc
    group_bit = 1 << idx
    membership = (masks & group_bit) != 0
    result = {}
    for prop in properties:
        try:
            result[prop] = lmp.numpy.extract_atom(prop)[membership]
        except TypeError as exc:  # LAMMPS returns None for undefined properties
            raise NameError(f"Property '{prop}' is not defined") from exc
    return result
