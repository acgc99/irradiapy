"""Numpy structured array dtypes."""

import numpy as np

# region General
atom = np.dtype(
    [
        ("type", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ]
)
defect = np.dtype(
    [
        ("type", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ]
)
acluster = np.dtype(
    [
        ("type", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("cluster", np.int32),
    ]
)
ocluster = np.dtype(
    [
        ("type", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("size", np.int32),
    ]
)

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
