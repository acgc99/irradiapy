"""Numpy structured array dtypes."""

from typing import Annotated, Any

import numpy as np
from numpy import typing as npt

# region General

atom = np.dtype(
    [
        ("type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ]
)
Atom = Annotated[npt.NDArray[Any], atom]
defect = np.dtype(
    [
        ("type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ]
)
Defect = Annotated[npt.NDArray[Any], defect]
acluster = np.dtype(
    [
        ("type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("cluster", np.int64),
    ]
)
Acluster = Annotated[npt.NDArray[Any], acluster]
ocluster = np.dtype(
    [
        ("type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("size", np.int64),
    ]
)
Ocluster = Annotated[npt.NDArray[Any], ocluster]

# endregion

# region SRIM

trimdat = np.dtype(
    [
        ("name", str),
        ("atomic_number", int),
        ("energy", float),
        ("pos", float, 3),
        ("dir", float, 3),
    ]
)
Trimdat = Annotated[npt.NDArray[Any], trimdat]

# endregion
