"""Utility functions for LAMMPS."""

import numpy as np
from lammps import lammps


def get_properties_from_group(
    lmp: lammps, group: str, properties: list[str]
) -> dict[str, np.ndarray]:
    """Returns the requested properties of a group of atoms.

    Parameters
    ----------
    lmp : lammps
        The LAMMPS object.
    group : str
        The name of the group.
    properties : list[str]
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
    except ValueError as e:
        raise NameError(f"group '{group}' is not defined") from e
    group_bit = 1 << idx
    membership = (masks & group_bit) != 0
    result = {}
    for prop in properties:
        try:
            result[prop] = lmp.numpy.extract_atom(prop)[membership]
        except TypeError as e:  # returning None if the property is not defined
            raise NameError(f"Property '{prop}' is not defined") from e
    return result
