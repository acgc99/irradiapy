"""This subpackage provides a Python interface for LAMMPS for simulating collisional cascades."""

from irradiapy.lammps.cascade import Cascade
from irradiapy.lammps.utils import (
    LAMMPSUnavailableError,
    load_lammps_class,
    get_properties_from_group,
)
