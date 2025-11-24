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

# region SPECTRA-PKA

spectra_event = np.dtype(
    [
        ("atom", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("vx", np.float64),
        ("vy", np.float64),
        ("vz", np.float64),
        ("element", "U16"),
        ("mass", np.float64),
        ("timestep", np.int32),
        ("recoil_energy", np.float64),
        ("time", np.float64),
        ("event", np.int32),
    ]
)
Spectra_Event = Annotated[npt.NDArray[Any], spectra_event]

# endregion
