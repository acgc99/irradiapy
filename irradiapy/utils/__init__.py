"""Utilities subpackage."""

from importlib import import_module
from types import ModuleType

from . import io, math, sqlite

__all__ = ["io", "math", "mpi", "sqlite"]


def __getattr__(name: str) -> ModuleType:
    """Load MPI utilities only when they are requested."""
    if name != "mpi":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(".mpi", __name__)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        # pylint: disable=no-member
        if missing_name == "mpi4py" or missing_name.startswith("mpi4py."):
            raise ModuleNotFoundError(
                "MPI utilities require the optional MPI dependencies. "
                "Install them with `python -m pip install 'irradiapy[mpi]'`."
            ) from exc
        raise

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"mpi"})
