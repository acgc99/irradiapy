"""This subpackage provides classes for reading and writing various file formats."""

from importlib import import_module
from typing import Any

from .bzip2lammpsreader import BZIP2LAMMPSReader
from .bzip2lammpswriter import BZIP2LAMMPSWriter
from .lammpslogreader import LAMMPSLogReader
from .lammpsreader import LAMMPSReader
from .lammpswriter import LAMMPSWriter
from .xyzreader import XYZReader
from .xyzwriter import XYZWriter

_MPI_EXPORTS = {
    "BZIP2LAMMPSReaderMPI": ".bzip2lammpsreadermpi",
    "BZIP2LAMMPSWriterMPI": ".bzip2lammpswritermpi",
    "LAMMPSReaderMPI": ".lammpsreadermpi",
    "LAMMPSWriterMPI": ".lammpswritermpi",
}

__all__ = [
    "BZIP2LAMMPSReader",
    "BZIP2LAMMPSReaderMPI",  # pylint: disable=undefined-all-variable
    "BZIP2LAMMPSWriter",
    "BZIP2LAMMPSWriterMPI",  # pylint: disable=undefined-all-variable
    "LAMMPSLogReader",
    "LAMMPSReader",
    "LAMMPSReaderMPI",  # pylint: disable=undefined-all-variable
    "LAMMPSWriter",
    "LAMMPSWriterMPI",  # pylint: disable=undefined-all-variable
    "XYZReader",
    "XYZWriter",
]


def __getattr__(name: str) -> Any:
    """Load MPI I/O classes only when they are requested."""
    module_name = _MPI_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        value = getattr(import_module(module_name, __name__), name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        # pylint: disable=no-member
        if missing_name == "mpi4py" or missing_name.startswith("mpi4py."):
            raise ModuleNotFoundError(
                "MPI I/O requires the optional MPI dependencies. "
                "Install them with `python -m pip install 'irradiapy[mpi]'`."
            ) from exc
        raise

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MPI_EXPORTS))
