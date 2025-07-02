"""."""

import numpy as np

field_type = ("type", int)
field_pos = ("pos", float, 3)
field_cluster = ("cluster", int)
field_size = ("size", int)
field_name = ("name", str)
field_atomic_number = ("atomic_number", int)
field_energy = ("energy", float)
field_dir = ("dir", float, 3)

atom = np.dtype([field_type, field_pos])
defect = np.dtype([field_type, field_pos])
acluster = np.dtype([field_pos, field_type, field_cluster])
ocluster = np.dtype([field_pos, field_type, field_size])
trimdat = np.dtype(
    [field_name, field_atomic_number, field_energy, field_pos, field_dir]
)
defect2 = np.dtype(
    [
        ("type", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ]
)
